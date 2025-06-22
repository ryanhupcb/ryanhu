"""
Model Adapters for Universal Agent System
=========================================
Integration layer for Claude 4 and Qwen model families
"""

import asyncio
import aiohttp
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from enum import Enum
import backoff
import tiktoken
from collections import deque
import hashlib
import numpy as np

# ========== Base Model Adapter ==========

@dataclass
class ModelResponse:
    """Standardized model response"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any] = None
    cached: bool = False
    latency: float = 0.0

@dataclass
class ModelRequest:
    """Standardized model request"""
    messages: List[Dict[str, str]]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    tools: Optional[List[Dict]] = None
    system_prompt: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class ModelAdapter(ABC):
    """Base class for model adapters"""
    
    def __init__(self, api_key: str, base_url: str, cache_ttl: int = 3600):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_cache = {}
        self.cache_ttl = cache_ttl
        self.rate_limiter = RateLimiter()
        self.token_counter = TokenCounter()
        self.metrics = ModelMetrics()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete a prompt - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def stream_complete(self, request: ModelRequest) -> AsyncIterator[str]:
        """Stream completion - must be implemented by subclasses"""
        pass
    
    def _get_cache_key(self, request: ModelRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'messages': request.messages,
            'model': request.model,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'system_prompt': request.system_prompt
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[ModelResponse]:
        """Check if response is cached"""
        if cache_key in self.request_cache:
            cached_data = self.request_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                response = cached_data['response']
                response.cached = True
                return response
            else:
                del self.request_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: ModelResponse):
        """Cache a response"""
        self.request_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }

# ========== Claude Adapter ==========

class ClaudeAdapter(ModelAdapter):
    """Adapter for Claude 4 models"""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.anthropic.com/v1"
        )
        self.model_mapping = {
            "claude-opus-4-20250514": "claude-opus-4-20250514",
            "claude-sonnet-4-20250514": "claude-sonnet-4-20250514"
        }
        
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete using Claude API"""
        # Check cache
        cache_key = self._get_cache_key(request)
        cached = self._check_cache(cache_key)
        if cached:
            self.metrics.record_hit(cache=True)
            return cached
        
        # Rate limiting
        await self.rate_limiter.acquire("claude", 1)
        
        # Prepare request
        claude_request = self._prepare_claude_request(request)
        
        # Make API call
        start_time = time.time()
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2024-01-01",
            "content-type": "application/json"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/messages",
                json=claude_request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
            latency = time.time() - start_time
            
            # Parse response
            model_response = ModelResponse(
                content=data['content'][0]['text'],
                model=data['model'],
                usage={
                    'prompt_tokens': data['usage']['input_tokens'],
                    'completion_tokens': data['usage']['output_tokens'],
                    'total_tokens': data['usage']['input_tokens'] + data['usage']['output_tokens']
                },
                finish_reason=data.get('stop_reason', 'stop'),
                metadata={
                    'id': data['id'],
                    'type': data['type']
                },
                latency=latency
            )
            
            # Cache response
            self._cache_response(cache_key, model_response)
            
            # Record metrics
            self.metrics.record_request(
                model=request.model,
                tokens=model_response.usage['total_tokens'],
                latency=latency,
                success=True
            )
            
            return model_response
            
        except Exception as e:
            self.metrics.record_request(
                model=request.model,
                tokens=0,
                latency=time.time() - start_time,
                success=False
            )
            raise
    
    async def stream_complete(self, request: ModelRequest) -> AsyncIterator[str]:
        """Stream completion from Claude"""
        # Rate limiting
        await self.rate_limiter.acquire("claude", 1)
        
        # Prepare request
        claude_request = self._prepare_claude_request(request)
        claude_request['stream'] = True
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2024-01-01",
            "content-type": "application/json"
        }
        
        async with self.session.post(
            f"{self.base_url}/messages",
            json=claude_request,
            headers=headers
        ) as response:
            response.raise_for_status()
            
            async for line in response.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if data['type'] == 'content_block_delta':
                                yield data['delta']['text']
                        except json.JSONDecodeError:
                            continue
    
    def _prepare_claude_request(self, request: ModelRequest) -> Dict[str, Any]:
        """Convert standard request to Claude format"""
        claude_request = {
            "model": self.model_mapping.get(request.model, request.model),
            "messages": [],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p
        }
        
        # Handle system prompt
        if request.system_prompt:
            claude_request["system"] = request.system_prompt
        
        # Convert messages
        for msg in request.messages:
            role = msg['role']
            if role == 'system' and not request.system_prompt:
                claude_request["system"] = msg['content']
            else:
                # Claude uses 'user' and 'assistant' roles
                if role == 'system':
                    role = 'user'
                claude_request["messages"].append({
                    "role": role,
                    "content": msg['content']
                })
        
        # Add tools if specified
        if request.tools:
            claude_request["tools"] = self._convert_tools_to_claude_format(request.tools)
        
        # Add stop sequences
        if request.stop_sequences:
            claude_request["stop_sequences"] = request.stop_sequences
        
        return claude_request
    
    def _convert_tools_to_claude_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style tools to Claude format"""
        claude_tools = []
        
        for tool in tools:
            claude_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"]
            }
            claude_tools.append(claude_tool)
        
        return claude_tools

# ========== Qwen Adapter ==========

class QwenAdapter(ModelAdapter):
    """Adapter for Qwen model family"""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation"
        )
        self.model_mapping = {
            "qwen-max": "qwen-max",
            "qwen-plus": "qwen-plus", 
            "qwen-turbo": "qwen-turbo"
        }
        
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete using Qwen API"""
        # Check cache
        cache_key = self._get_cache_key(request)
        cached = self._check_cache(cache_key)
        if cached:
            self.metrics.record_hit(cache=True)
            return cached
        
        # Rate limiting
        await self.rate_limiter.acquire("qwen", 1)
        
        # Prepare request
        qwen_request = self._prepare_qwen_request(request)
        
        # Make API call
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/generation",
                json=qwen_request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
            latency = time.time() - start_time
            
            # Parse response
            output = data['output']
            usage = data['usage']
            
            model_response = ModelResponse(
                content=output['text'],
                model=data['model'],
                usage={
                    'prompt_tokens': usage['input_tokens'],
                    'completion_tokens': usage['output_tokens'],
                    'total_tokens': usage['total_tokens']
                },
                finish_reason=output.get('finish_reason', 'stop'),
                metadata={
                    'request_id': data.get('request_id')
                },
                latency=latency
            )
            
            # Cache response
            self._cache_response(cache_key, model_response)
            
            # Record metrics
            self.metrics.record_request(
                model=request.model,
                tokens=model_response.usage['total_tokens'],
                latency=latency,
                success=True
            )
            
            return model_response
            
        except Exception as e:
            self.metrics.record_request(
                model=request.model,
                tokens=0,
                latency=time.time() - start_time,
                success=False
            )
            raise
    
    async def stream_complete(self, request: ModelRequest) -> AsyncIterator[str]:
        """Stream completion from Qwen"""
        # Rate limiting
        await self.rate_limiter.acquire("qwen", 1)
        
        # Prepare request
        qwen_request = self._prepare_qwen_request(request)
        qwen_request['parameters']['incremental_output'] = True
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "enable"
        }
        
        async with self.session.post(
            f"{self.base_url}/generation",
            json=qwen_request,
            headers=headers
        ) as response:
            response.raise_for_status()
            
            async for line in response.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data:'):
                        data_str = line_str[5:].strip()
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if 'output' in data and 'text' in data['output']:
                                yield data['output']['text']
                        except json.JSONDecodeError:
                            continue
    
    def _prepare_qwen_request(self, request: ModelRequest) -> Dict[str, Any]:
        """Convert standard request to Qwen format"""
        # Build input with messages
        messages = []
        
        # Add system message if present
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        
        # Add conversation messages
        for msg in request.messages:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        qwen_request = {
            "model": self.model_mapping.get(request.model, request.model),
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "repetition_penalty": 1.0 + request.frequency_penalty,
                "presence_penalty": request.presence_penalty
            }
        }
        
        # Add stop sequences
        if request.stop_sequences:
            qwen_request["parameters"]["stop"] = request.stop_sequences
        
        # Add tools if specified
        if request.tools:
            qwen_request["parameters"]["tools"] = self._convert_tools_to_qwen_format(request.tools)
        
        return qwen_request
    
    def _convert_tools_to_qwen_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style tools to Qwen format"""
        qwen_tools = []
        
        for tool in tools:
            qwen_tool = {
                "type": "function",
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"]
                }
            }
            qwen_tools.append(qwen_tool)
        
        return qwen_tools

# ========== Model Router ==========

class ModelRouter:
    """Intelligent routing between different models"""
    
    def __init__(self, cost_manager):
        self.adapters = {}
        self.cost_manager = cost_manager
        self.routing_history = deque(maxlen=1000)
        self.performance_tracker = ModelPerformanceTracker()
        
    def register_adapter(self, provider: str, adapter: ModelAdapter):
        """Register a model adapter"""
        self.adapters[provider] = adapter
        
    async def route_request(
        self,
        request: ModelRequest,
        routing_strategy: str = "cost_optimized"
    ) -> ModelResponse:
        """Route request to appropriate model"""
        # Select provider based on strategy
        if routing_strategy == "cost_optimized":
            provider = self._select_cost_optimized_provider(request)
        elif routing_strategy == "quality_first":
            provider = self._select_quality_first_provider(request)
        elif routing_strategy == "speed_optimized":
            provider = self._select_speed_optimized_provider(request)
        else:
            provider = request.model.split('-')[0]  # Extract provider from model name
        
        # Record routing decision
        self.routing_history.append({
            'timestamp': time.time(),
            'request_model': request.model,
            'selected_provider': provider,
            'strategy': routing_strategy
        })
        
        # Get adapter and make request
        adapter = self.adapters.get(provider)
        if not adapter:
            raise ValueError(f"No adapter registered for provider: {provider}")
        
        # Execute request
        start_time = time.time()
        try:
            response = await adapter.complete(request)
            
            # Track performance
            self.performance_tracker.record(
                provider=provider,
                latency=response.latency,
                tokens=response.usage['total_tokens'],
                success=True
            )
            
            return response
            
        except Exception as e:
            self.performance_tracker.record(
                provider=provider,
                latency=time.time() - start_time,
                tokens=0,
                success=False
            )
            raise
    
    def _select_cost_optimized_provider(self, request: ModelRequest) -> str:
        """Select provider optimizing for cost"""
        # Estimate tokens
        estimated_tokens = self._estimate_tokens(request)
        
        # Calculate cost for each provider
        costs = {}
        for provider in ['claude', 'qwen']:
            if provider == 'claude':
                # Claude pricing (example)
                cost_per_1k = 0.015 if 'opus' in request.model else 0.003
            else:
                # Qwen pricing (example)
                if 'max' in request.model:
                    cost_per_1k = 0.002
                elif 'plus' in request.model:
                    cost_per_1k = 0.001
                else:
                    cost_per_1k = 0.0003
            
            costs[provider] = (estimated_tokens / 1000) * cost_per_1k
        
        # Select cheapest
        return min(costs, key=costs.get)
    
    def _select_quality_first_provider(self, request: ModelRequest) -> str:
        """Select provider optimizing for quality"""
        # For complex tasks, prefer Claude Opus
        if self._estimate_complexity(request) > 0.7:
            return 'claude'
        # For simpler tasks, Qwen Max is sufficient
        return 'qwen'
    
    def _select_speed_optimized_provider(self, request: ModelRequest) -> str:
        """Select provider optimizing for speed"""
        # Check recent performance metrics
        claude_avg_latency = self.performance_tracker.get_average_latency('claude')
        qwen_avg_latency = self.performance_tracker.get_average_latency('qwen')
        
        # Select faster provider
        if claude_avg_latency < qwen_avg_latency:
            return 'claude'
        return 'qwen'
    
    def _estimate_tokens(self, request: ModelRequest) -> int:
        """Estimate token count for request"""
        # Simplified estimation
        text = request.system_prompt or ""
        for msg in request.messages:
            text += msg['content']
        
        # Rough estimate: 1 token per 4 characters
        return len(text) // 4 + request.max_tokens
    
    def _estimate_complexity(self, request: ModelRequest) -> float:
        """Estimate task complexity from request"""
        complexity_indicators = [
            'analyze', 'complex', 'detailed', 'comprehensive',
            'algorithm', 'architecture', 'design', 'optimize'
        ]
        
        text = request.system_prompt or ""
        for msg in request.messages:
            text += msg['content'].lower()
        
        # Count complexity indicators
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in text)
        
        # Normalize to 0-1 scale
        return min(indicator_count / 5.0, 1.0)

# ========== Rate Limiter ==========

class RateLimiter:
    """Token bucket rate limiter for API calls"""
    
    def __init__(self):
        self.buckets = {
            'claude': TokenBucket(rate=100, capacity=100),  # 100 requests/minute
            'qwen': TokenBucket(rate=200, capacity=200)     # 200 requests/minute
        }
        
    async def acquire(self, provider: str, tokens: int = 1):
        """Acquire tokens from rate limiter"""
        if provider in self.buckets:
            await self.buckets[provider].acquire(tokens)

class TokenBucket:
    """Token bucket implementation"""
    
    def __init__(self, rate: float, capacity: float):
        self.rate = rate  # tokens per minute
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self, tokens: int = 1):
        """Acquire tokens, waiting if necessary"""
        async with self.lock:
            # Refill bucket
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate / 60)
            self.last_update = now
            
            # Wait if not enough tokens
            while self.tokens < tokens:
                wait_time = (tokens - self.tokens) * 60 / self.rate
                await asyncio.sleep(wait_time)
                
                # Refill again
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate / 60)
                self.last_update = now
            
            # Consume tokens
            self.tokens -= tokens

# ========== Token Counter ==========

class TokenCounter:
    """Count tokens for different models"""
    
    def __init__(self):
        self.encoders = {
            'claude': tiktoken.get_encoding("cl100k_base"),
            'qwen': tiktoken.get_encoding("cl100k_base")  # Qwen uses similar tokenization
        }
        
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens for text"""
        provider = model.split('-')[0]
        encoder = self.encoders.get(provider, self.encoders['claude'])
        return len(encoder.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """Count tokens for messages"""
        total = 0
        for message in messages:
            # Role tokens
            total += 4  # Approximate tokens for role formatting
            # Content tokens
            total += self.count_tokens(message['content'], model)
        return total

# ========== Model Metrics ==========

class ModelMetrics:
    """Track model performance metrics"""
    
    def __init__(self):
        self.requests = deque(maxlen=10000)
        self.cache_hits = 0
        self.cache_misses = 0
        
    def record_request(self, model: str, tokens: int, latency: float, success: bool):
        """Record a model request"""
        self.requests.append({
            'timestamp': time.time(),
            'model': model,
            'tokens': tokens,
            'latency': latency,
            'success': success
        })
        
    def record_hit(self, cache: bool = False):
        """Record cache hit"""
        if cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_stats(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get metrics for time window"""
        cutoff = time.time() - time_window
        recent_requests = [r for r in self.requests if r['timestamp'] > cutoff]
        
        if not recent_requests:
            return {
                'total_requests': 0,
                'success_rate': 0,
                'avg_latency': 0,
                'avg_tokens': 0,
                'cache_hit_rate': 0
            }
        
        successful = [r for r in recent_requests if r['success']]
        
        return {
            'total_requests': len(recent_requests),
            'success_rate': len(successful) / len(recent_requests),
            'avg_latency': sum(r['latency'] for r in successful) / len(successful) if successful else 0,
            'avg_tokens': sum(r['tokens'] for r in successful) / len(successful) if successful else 0,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'requests_by_model': self._count_by_model(recent_requests)
        }
    
    def _count_by_model(self, requests: List[Dict]) -> Dict[str, int]:
        """Count requests by model"""
        counts = {}
        for req in requests:
            model = req['model']
            counts[model] = counts.get(model, 0) + 1
        return counts

# ========== Model Performance Tracker ==========

class ModelPerformanceTracker:
    """Track and analyze model performance"""
    
    def __init__(self):
        self.performance_data = defaultdict(lambda: {
            'latencies': deque(maxlen=1000),
            'success_count': 0,
            'failure_count': 0,
            'total_tokens': 0
        })
        
    def record(self, provider: str, latency: float, tokens: int, success: bool):
        """Record performance data"""
        data = self.performance_data[provider]
        data['latencies'].append(latency)
        
        if success:
            data['success_count'] += 1
            data['total_tokens'] += tokens
        else:
            data['failure_count'] += 1
    
    def get_average_latency(self, provider: str) -> float:
        """Get average latency for provider"""
        data = self.performance_data[provider]
        if not data['latencies']:
            return float('inf')
        return sum(data['latencies']) / len(data['latencies'])
    
    def get_success_rate(self, provider: str) -> float:
        """Get success rate for provider"""
        data = self.performance_data[provider]
        total = data['success_count'] + data['failure_count']
        if total == 0:
            return 1.0
        return data['success_count'] / total
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {}
        
        for provider, data in self.performance_data.items():
            report[provider] = {
                'avg_latency': self.get_average_latency(provider),
                'success_rate': self.get_success_rate(provider),
                'total_requests': data['success_count'] + data['failure_count'],
                'total_tokens': data['total_tokens'],
                'latency_percentiles': self._calculate_percentiles(data['latencies'])
            }
        
        return report
    
    def _calculate_percentiles(self, latencies: deque) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not latencies:
            return {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            'p50': sorted_latencies[int(n * 0.5)],
            'p90': sorted_latencies[int(n * 0.9)],
            'p95': sorted_latencies[int(n * 0.95)],
            'p99': sorted_latencies[int(n * 0.99)]
        }

# ========== Model Manager ==========

class ModelManager:
    """High-level manager for all model operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapters = {}
        self.router = None
        self.cost_manager = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize model manager with adapters"""
        # Initialize Claude adapter
        if 'claude_api_key' in self.config:
            claude_adapter = ClaudeAdapter(self.config['claude_api_key'])
            self.adapters['claude'] = claude_adapter
        
        # Initialize Qwen adapter
        if 'qwen_api_key' in self.config:
            qwen_adapter = QwenAdapter(self.config['qwen_api_key'])
            self.adapters['qwen'] = qwen_adapter
        
        # Initialize router
        self.router = ModelRouter(self.cost_manager)
        for provider, adapter in self.adapters.items():
            self.router.register_adapter(provider, adapter)
        
        # Open sessions
        for adapter in self.adapters.values():
            await adapter.__aenter__()
        
        self.initialized = True
    
    async def cleanup(self):
        """Cleanup resources"""
        for adapter in self.adapters.values():
            await adapter.__aexit__(None, None, None)
    
    async def complete(
        self,
        prompt: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        routing_strategy: str = "cost_optimized"
    ) -> ModelResponse:
        """High-level completion interface"""
        if not self.initialized:
            await self.initialize()
        
        request = ModelRequest(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )
        
        return await self.router.route_request(request, routing_strategy)
    
    async def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        routing_strategy: str = "cost_optimized"
    ) -> ModelResponse:
        """Chat completion interface"""
        if not self.initialized:
            await self.initialize()
        
        request = ModelRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tools=tools
        )
        
        return await self.router.route_request(request, routing_strategy)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report across all models"""
        return self.router.performance_tracker.get_performance_report()
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get cost report"""
        if self.cost_manager:
            return self.cost_manager.get_cost_report()
        return {}

# ========== Embedding Models ==========

class EmbeddingAdapter(ABC):
    """Base class for embedding adapters"""
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for texts"""
        pass

class ClaudeEmbeddingAdapter(EmbeddingAdapter):
    """Claude embedding adapter"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        
    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using Claude"""
        # Note: This is a placeholder - Claude doesn't currently offer embeddings
        # In practice, you might use a different service or local model
        embeddings = []
        for text in texts:
            # Simulate embedding generation
            embedding = np.random.randn(768)  # 768-dimensional embedding
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
        return embeddings

class LocalEmbeddingAdapter(EmbeddingAdapter):
    """Local embedding model adapter"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        
    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using local model"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self.model.encode,
            texts
        )
        return embeddings

# ========== Example Usage ==========

async def example_usage():
    """Example of using the model adapters"""
    
    # Initialize model manager
    config = {
        'claude_api_key': 'your-claude-api-key',
        'qwen_api_key': 'your-qwen-api-key'
    }
    
    manager = ModelManager(config)
    await manager.initialize()
    
    try:
        # Simple completion
        response = await manager.complete(
            prompt="Explain quantum computing in simple terms",
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=500
        )
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.usage}")
        
        # Chat completion with conversation history
        messages = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."},
            {"role": "user", "content": "Can you give me an example?"}
        ]
        
        chat_response = await manager.chat_complete(
            messages=messages,
            model="qwen-max",
            routing_strategy="cost_optimized"
        )
        print(f"Chat response: {chat_response.content}")
        
        # Get performance report
        perf_report = manager.get_performance_report()
        print(f"Performance report: {json.dumps(perf_report, indent=2)}")
        
    finally:
        await manager.cleanup()

if __name__ == "__main__":
    asyncio.run(example_usage())
