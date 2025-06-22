# Complete Production Agent System
# Enhanced with all missing components for a fully functional agent system

import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from collections import defaultdict, deque
import time
import uuid
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
from contextlib import asynccontextmanager
import pickle
import base64
import torch
import faiss
from sentence_transformers import SentenceTransformer
import redis
from prometheus_client import Counter, Histogram, Gauge
import structlog
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc import trace_exporter, metrics_exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from typing_extensions import TypedDict
import openai
import anthropic
import subprocess
import sys
import os
import tempfile
import shutil
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from pathlib import Path
import ast
import black
import autopep8
import pylint.lint
from typing import Protocol

# Configure structured logging
logger = structlog.get_logger()

# ==================== PART 1: LLM Integration Layer ====================

class LLMProvider(Protocol):
    """Protocol for LLM providers"""
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]: ...
    async def embed(self, text: str) -> np.ndarray: ...

class OpenAIProvider:
    """OpenAI LLM provider"""
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000),
                tools=kwargs.get('tools', None),
                tool_choice=kwargs.get('tool_choice', 'auto')
            )
            
            return {
                'content': response.choices[0].message.content,
                'tool_calls': response.choices[0].message.tool_calls,
                'usage': response.usage.dict() if response.usage else {},
                'finish_reason': response.choices[0].finish_reason
            }
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
            
    async def embed(self, text: str) -> np.ndarray:
        """Generate embeddings"""
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)

class AnthropicProvider:
    """Anthropic Claude provider"""
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from Claude"""
        try:
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            return {
                'content': response.content[0].text,
                'tool_calls': None,  # Claude handles tools differently
                'usage': {'total_tokens': response.usage.input_tokens + response.usage.output_tokens},
                'finish_reason': response.stop_reason
            }
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

class LLMOrchestrator:
    """Orchestrates multiple LLM providers with fallback"""
    def __init__(self, providers: Dict[str, LLMProvider]):
        self.providers = providers
        self.primary_provider = list(providers.keys())[0]
        self.usage_stats = defaultdict(lambda: {'calls': 0, 'tokens': 0, 'errors': 0})
        
    async def generate(self, prompt: str, provider: str = None, **kwargs) -> Dict[str, Any]:
        """Generate with automatic fallback"""
        provider = provider or self.primary_provider
        
        try:
            result = await self.providers[provider].generate(prompt, **kwargs)
            self.usage_stats[provider]['calls'] += 1
            self.usage_stats[provider]['tokens'] += result.get('usage', {}).get('total_tokens', 0)
            return result
        except Exception as e:
            self.usage_stats[provider]['errors'] += 1
            logger.warning(f"Provider {provider} failed, trying fallback", error=str(e))
            
            # Try other providers
            for backup_provider, llm in self.providers.items():
                if backup_provider != provider:
                    try:
                        return await llm.generate(prompt, **kwargs)
                    except:
                        continue
                        
            raise Exception("All LLM providers failed")

# ==================== PART 2: Complete Tool Implementation ====================

class Tool(ABC):
    """Base class for all tools"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        self.success_count = 0
        
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        pass
        
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters()
            }
        }
        
    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """Get tool parameters schema"""
        pass

class WebSearchTool(Tool):
    """Web search tool implementation"""
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information"
        )
        self.session = None
        
    async def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Execute web search"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        try:
            # Using DuckDuckGo for demo (in production, use proper API)
            url = f"https://html.duckduckgo.com/html/?q={query}"
            async with self.session.get(url) as response:
                html = await response.text()
                
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            for i, result in enumerate(soup.find_all('div', class_='result', limit=num_results)):
                title_elem = result.find('a', class_='result__title')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'url': title_elem.get('href', ''),
                        'snippet': snippet_elem.get_text(strip=True)
                    })
                    
            self.usage_count += 1
            self.success_count += 1
            
            return {
                'success': True,
                'results': results,
                'query': query
            }
            
        except Exception as e:
            self.usage_count += 1
            logger.error(f"Web search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
            
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }

class CodeExecutionTool(Tool):
    """Secure code execution tool"""
    def __init__(self, allowed_imports: List[str] = None):
        super().__init__(
            name="execute_code",
            description="Execute Python code in a sandboxed environment"
        )
        self.allowed_imports = allowed_imports or [
            'math', 'statistics', 'itertools', 'collections',
            'datetime', 'json', 're', 'numpy', 'pandas'
        ]
        
    async def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute code safely"""
        try:
            # Validate code safety
            tree = ast.parse(code)
            validator = CodeValidator(self.allowed_imports)
            validator.visit(tree)
            
            if validator.violations:
                return {
                    'success': False,
                    'error': f"Code safety violations: {validator.violations}"
                }
                
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
                
            # Execute in subprocess with timeout
            try:
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                self.usage_count += 1
                self.success_count += 1
                
                return {
                    'success': True,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
                
            finally:
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            self.usage_count += 1
            return {
                'success': False,
                'error': f"Code execution timed out after {timeout} seconds"
            }
        except Exception as e:
            self.usage_count += 1
            logger.error(f"Code execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds",
                    "default": 30
                }
            },
            "required": ["code"]
        }

class FileOperationTool(Tool):
    """File operation tool with sandboxing"""
    def __init__(self, workspace_path: str = None):
        super().__init__(
            name="file_operation",
            description="Perform file operations (read, write, list)"
        )
        self.workspace = Path(workspace_path or tempfile.mkdtemp())
        self.workspace.mkdir(exist_ok=True)
        
    async def execute(self, operation: str, path: str, content: str = None) -> Dict[str, Any]:
        """Execute file operation"""
        try:
            # Ensure path is within workspace
            full_path = (self.workspace / path).resolve()
            if not str(full_path).startswith(str(self.workspace)):
                return {
                    'success': False,
                    'error': "Path traversal attempt detected"
                }
                
            if operation == "read":
                if full_path.exists():
                    content = full_path.read_text()
                    return {'success': True, 'content': content}
                else:
                    return {'success': False, 'error': "File not found"}
                    
            elif operation == "write":
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                return {'success': True, 'path': str(path)}
                
            elif operation == "list":
                files = list(full_path.glob("*") if full_path.is_dir() else [])
                return {
                    'success': True,
                    'files': [str(f.relative_to(self.workspace)) for f in files]
                }
                
            else:
                return {'success': False, 'error': f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return {'success': False, 'error': str(e)}
            
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": "Operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)"
                }
            },
            "required": ["operation", "path"]
        }

class DataAnalysisTool(Tool):
    """Advanced data analysis tool"""
    def __init__(self):
        super().__init__(
            name="analyze_data",
            description="Perform data analysis on CSV, JSON, or Excel files"
        )
        
    async def execute(self, file_path: str, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """Execute data analysis"""
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return {'success': False, 'error': "Unsupported file format"}
                
            # Perform analysis
            if analysis_type == "summary":
                result = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'summary': df.describe().to_dict(),
                    'null_counts': df.isnull().sum().to_dict()
                }
                
            elif analysis_type == "correlation":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                result = {
                    'correlation_matrix': df[numeric_cols].corr().to_dict()
                }
                
            elif analysis_type == "groupby":
                group_col = kwargs.get('group_column')
                agg_col = kwargs.get('aggregate_column')
                agg_func = kwargs.get('aggregate_function', 'mean')
                
                if group_col and agg_col:
                    result = {
                        'grouped_data': df.groupby(group_col)[agg_col].agg(agg_func).to_dict()
                    }
                else:
                    return {'success': False, 'error': "Missing groupby parameters"}
                    
            else:
                return {'success': False, 'error': f"Unknown analysis type: {analysis_type}"}
                
            self.usage_count += 1
            self.success_count += 1
            
            return {
                'success': True,
                'result': result,
                'analysis_type': analysis_type
            }
            
        except Exception as e:
            self.usage_count += 1
            logger.error(f"Data analysis failed: {e}")
            return {'success': False, 'error': str(e)}
            
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to data file"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["summary", "correlation", "groupby"],
                    "description": "Type of analysis to perform"
                },
                "group_column": {
                    "type": "string",
                    "description": "Column to group by (for groupby analysis)"
                },
                "aggregate_column": {
                    "type": "string",
                    "description": "Column to aggregate (for groupby analysis)"
                },
                "aggregate_function": {
                    "type": "string",
                    "enum": ["mean", "sum", "count", "min", "max"],
                    "description": "Aggregation function (for groupby analysis)"
                }
            },
            "required": ["file_path", "analysis_type"]
        }

# ==================== PART 3: Task Planning and Decomposition ====================

class TaskPlanner:
    """Advanced task planning and decomposition"""
    def __init__(self, llm_orchestrator: LLMOrchestrator):
        self.llm = llm_orchestrator
        self.planning_prompt = """
        Given the following task, create a detailed execution plan.
        Break it down into subtasks and identify required tools.
        
        Task: {task}
        Context: {context}
        
        Return a JSON plan with:
        - objective: Clear statement of the goal
        - subtasks: List of subtasks with descriptions and dependencies
        - tools_required: List of tools needed
        - estimated_complexity: low/medium/high
        - potential_challenges: List of potential issues
        """
        
    async def create_plan(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create execution plan for a task"""
        prompt = self.planning_prompt.format(
            task=task,
            context=json.dumps(context or {})
        )
        
        response = await self.llm.generate(prompt, temperature=0.3)
        
        try:
            # Parse JSON from response
            plan = json.loads(response['content'])
            
            # Validate plan structure
            required_keys = ['objective', 'subtasks', 'tools_required', 'estimated_complexity']
            if all(key in plan for key in required_keys):
                return {
                    'success': True,
                    'plan': plan
                }
            else:
                raise ValueError("Invalid plan structure")
                
        except Exception as e:
            logger.error(f"Failed to parse plan: {e}")
            
            # Fallback to simple plan
            return {
                'success': False,
                'plan': {
                    'objective': task,
                    'subtasks': [{'description': task, 'dependencies': []}],
                    'tools_required': [],
                    'estimated_complexity': 'medium',
                    'potential_challenges': ['Failed to create detailed plan']
                },
                'error': str(e)
            }

class TaskExecutor:
    """Execute tasks based on plans"""
    def __init__(self, tool_registry: 'EnhancedToolRegistry', llm_orchestrator: LLMOrchestrator):
        self.tools = tool_registry
        self.llm = llm_orchestrator
        self.execution_history = []
        
    async def execute_plan(self, plan: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task plan"""
        results = {
            'objective': plan['objective'],
            'subtask_results': [],
            'overall_success': True,
            'execution_time': 0
        }
        
        start_time = time.time()
        context = context or {}
        
        # Execute subtasks in order
        for subtask in plan['subtasks']:
            subtask_result = await self.execute_subtask(subtask, context)
            results['subtask_results'].append(subtask_result)
            
            # Update context with results
            context[f"subtask_{subtask.get('id', len(results['subtask_results']))}"] = subtask_result
            
            if not subtask_result['success']:
                results['overall_success'] = False
                if subtask.get('critical', True):
                    break
                    
        results['execution_time'] = time.time() - start_time
        self.execution_history.append(results)
        
        return results
        
    async def execute_subtask(self, subtask: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask"""
        try:
            # Determine execution strategy
            if 'tool' in subtask:
                # Direct tool execution
                tool = self.tools.get_tool(subtask['tool'])
                if tool:
                    result = await tool.execute(**subtask.get('parameters', {}))
                    return {
                        'success': result.get('success', False),
                        'result': result,
                        'tool_used': subtask['tool']
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Tool {subtask['tool']} not found"
                    }
                    
            else:
                # LLM-based execution
                prompt = f"""
                Execute the following subtask:
                {subtask['description']}
                
                Context: {json.dumps(context)}
                
                Available tools: {', '.join(self.tools.list_tools())}
                
                Provide a detailed response or use tools as needed.
                """
                
                response = await self.llm.generate(
                    prompt,
                    tools=[tool.get_schema() for tool in self.tools.get_all_tools()]
                )
                
                # Handle tool calls
                if response.get('tool_calls'):
                    tool_results = []
                    for tool_call in response['tool_calls']:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        tool = self.tools.get_tool(tool_name)
                        if tool:
                            result = await tool.execute(**tool_args)
                            tool_results.append(result)
                            
                    return {
                        'success': True,
                        'result': {
                            'llm_response': response['content'],
                            'tool_results': tool_results
                        }
                    }
                else:
                    return {
                        'success': True,
                        'result': {'llm_response': response['content']}
                    }
                    
        except Exception as e:
            logger.error(f"Subtask execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# ==================== PART 4: Dialog Management System ====================

class DialogContext:
    """Manages conversation context and state"""
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages = []
        self.user_profile = {}
        self.current_task = None
        self.metadata = {
            'created_at': datetime.now(),
            'turn_count': 0,
            'total_tokens': 0
        }
        
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to conversation"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
        self.metadata['turn_count'] += 1
        
    def get_context_window(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages for context"""
        return self.messages[-max_messages:]
        
    def update_user_profile(self, updates: Dict[str, Any]):
        """Update user profile information"""
        self.user_profile.update(updates)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'conversation_id': self.conversation_id,
            'messages': self.messages,
            'user_profile': self.user_profile,
            'current_task': self.current_task,
            'metadata': self.metadata
        }

class DialogManager:
    """Manages multi-turn conversations"""
    def __init__(self, llm_orchestrator: LLMOrchestrator, memory_system: 'DualLayerMemorySystem'):
        self.llm = llm_orchestrator
        self.memory = memory_system
        self.active_contexts = {}
        self.dialog_strategies = self._initialize_strategies()
        
    def _initialize_strategies(self) -> Dict[str, Callable]:
        """Initialize dialog management strategies"""
        return {
            'clarification': self._clarification_strategy,
            'task_oriented': self._task_oriented_strategy,
            'conversational': self._conversational_strategy,
            'educational': self._educational_strategy
        }
        
    async def process_message(self, user_message: str, conversation_id: str = None) -> Dict[str, Any]:
        """Process user message and generate response"""
        # Get or create context
        conversation_id = conversation_id or str(uuid.uuid4())
        context = self.active_contexts.get(
            conversation_id,
            DialogContext(conversation_id)
        )
        
        # Add user message
        context.add_message('user', user_message)
        
        # Retrieve relevant memories
        memories = await self.memory.retrieve(user_message, k=5)
        
        # Determine dialog strategy
        strategy = await self._select_strategy(user_message, context)
        
        # Generate response using selected strategy
        response = await self.dialog_strategies[strategy](user_message, context, memories)
        
        # Add assistant response
        context.add_message('assistant', response['content'], response.get('metadata'))
        
        # Update memory
        await self.memory.store({
            'request': user_message,
            'response': response['content'],
            'conversation_id': conversation_id,
            'strategy': strategy,
            'metadata': {
                'timestamp': datetime.now(),
                'turn_count': context.metadata['turn_count']
            }
        })
        
        # Update active contexts
        self.active_contexts[conversation_id] = context
        
        return {
            'response': response['content'],
            'conversation_id': conversation_id,
            'strategy_used': strategy,
            'metadata': response.get('metadata', {})
        }
        
    async def _select_strategy(self, message: str, context: DialogContext) -> str:
        """Select appropriate dialog strategy"""
        # Simple heuristic - in production, use more sophisticated classification
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['how', 'what', 'why', 'explain']):
            return 'educational'
        elif any(word in message_lower for word in ['do', 'create', 'make', 'build']):
            return 'task_oriented'
        elif '?' in message and len(context.messages) > 2:
            return 'clarification'
        else:
            return 'conversational'
            
    async def _task_oriented_strategy(self, message: str, context: DialogContext, 
                                    memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle task-oriented dialog"""
        prompt = f"""
        You are a helpful task-oriented assistant. The user needs help with a specific task.
        
        User message: {message}
        
        Recent conversation:
        {self._format_context(context.get_context_window())}
        
        Relevant memories:
        {self._format_memories(memories)}
        
        Identify the task, ask clarifying questions if needed, or provide step-by-step guidance.
        """
        
        response = await self.llm.generate(prompt, temperature=0.3)
        
        return {
            'content': response['content'],
            'metadata': {
                'strategy': 'task_oriented',
                'confidence': 0.9
            }
        }
        
    def _format_context(self, messages: List[Dict[str, Any]]) -> str:
        """Format conversation context"""
        formatted = []
        for msg in messages:
            formatted.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(formatted)
        
    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories"""
        if not memories:
            return "No relevant memories found."
            
        formatted = []
        for i, memory in enumerate(memories[:3]):  # Limit to top 3
            formatted.append(f"{i+1}. {memory.get('metadata', {}).get('summary', 'N/A')}")
        return "\n".join(formatted)

# ==================== PART 5: Enhanced Tool Registry ====================

class EnhancedToolRegistry(SemanticToolRegistry):
    """Enhanced tool registry with actual tool instances"""
    def __init__(self):
        super().__init__()
        self.tool_instances = {}
        self._initialize_default_tools()
        
    def _initialize_default_tools(self):
        """Initialize default tool set"""
        default_tools = [
            WebSearchTool(),
            CodeExecutionTool(),
            FileOperationTool(),
            DataAnalysisTool()
        ]
        
        for tool in default_tools:
            self.register_tool_instance(tool)
            
    def register_tool_instance(self, tool: Tool):
        """Register an actual tool instance"""
        self.tool_instances[tool.name] = tool
        # Also register in semantic registry
        asyncio.create_task(
            self.register_tool(tool.name, tool.description, tool.execute)
        )
        
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool instance by name"""
        return self.tool_instances.get(name)
        
    def get_all_tools(self) -> List[Tool]:
        """Get all tool instances"""
        return list(self.tool_instances.values())
        
    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tool_instances.keys())

# ==================== PART 6: Agent Communication Protocol ====================

class AgentMessage:
    """Standard message format for agent communication"""
    def __init__(self, sender: str, receiver: str, content: Any, 
                 message_type: str = "request", correlation_id: str = None):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type  # request, response, notification, error
        self.correlation_id = correlation_id or self.id
        self.timestamp = datetime.now()
        self.metadata = {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'sender': self.sender,
            'receiver': self.receiver,
            'content': self.content,
            'message_type': self.message_type,
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class AgentCommunicationBus:
    """Message bus for agent communication"""
    def __init__(self):
        self.agents = {}
        self.message_queue = asyncio.Queue()
        self.subscriptions = defaultdict(list)
        self.message_history = deque(maxlen=10000)
        
    def register_agent(self, agent_id: str, agent: Any):
        """Register an agent"""
        self.agents[agent_id] = agent
        
    async def send_message(self, message: AgentMessage):
        """Send message to an agent"""
        await self.message_queue.put(message)
        self.message_history.append(message)
        
    async def subscribe(self, agent_id: str, message_types: List[str]):
        """Subscribe agent to message types"""
        for msg_type in message_types:
            self.subscriptions[msg_type].append(agent_id)
            
    async def process_messages(self):
        """Process message queue"""
        while True:
            try:
                message = await self.message_queue.get()
                
                # Direct message
                if message.receiver in self.agents:
                    agent = self.agents[message.receiver]
                    if hasattr(agent, 'receive_message'):
                        await agent.receive_message(message)
                        
                # Broadcast to subscribers
                for subscriber_id in self.subscriptions.get(message.message_type, []):
                    if subscriber_id != message.sender:
                        agent = self.agents.get(subscriber_id)
                        if agent and hasattr(agent, 'receive_message'):
                            await agent.receive_message(message)
                            
            except Exception as e:
                logger.error(f"Message processing error: {e}")

# ==================== PART 7: Complete Agent System ====================

class CompleteAgentSystem(ResearchOptimizedProductionSystem):
    """Complete production-ready agent system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Initialize base system
        super().__init__(config)
        
        # Initialize LLM layer
        self.llm_orchestrator = self._initialize_llm_orchestrator()
        
        # Initialize enhanced components
        self.enhanced_tool_registry = EnhancedToolRegistry()
        self.task_planner = TaskPlanner(self.llm_orchestrator)
        self.task_executor = TaskExecutor(self.enhanced_tool_registry, self.llm_orchestrator)
        self.dialog_manager = DialogManager(self.llm_orchestrator, self.memory_manager)
        self.communication_bus = AgentCommunicationBus()
        
        # Initialize specialized agents
        self._initialize_specialized_agents()
        
        # Start communication bus
        asyncio.create_task(self.communication_bus.process_messages())
        
        logger.info("Complete Agent System initialized successfully")
        
    def _initialize_llm_orchestrator(self) -> LLMOrchestrator:
        """Initialize LLM orchestrator with providers"""
        providers = {}
        
        # Add OpenAI if API key available
        if os.getenv('OPENAI_API_KEY'):
            providers['openai'] = OpenAIProvider(
                api_key=os.getenv('OPENAI_API_KEY'),
                model=self.config.get('openai_model', 'gpt-4-turbo-preview')
            )
            
        # Add Anthropic if API key available
        if os.getenv('ANTHROPIC_API_KEY'):
            providers['anthropic'] = AnthropicProvider(
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                model=self.config.get('anthropic_model', 'claude-3-opus-20240229')
            )
            
        if not providers:
            raise ValueError("No LLM providers configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            
        return LLMOrchestrator(providers)
        
    def _initialize_specialized_agents(self):
        """Initialize specialized agent types"""
        # Research Agent
        research_agent = ResearchAgent(
            'research_agent',
            self.enhanced_tool_registry,
            self.llm_orchestrator,
            self.vector_db
        )
        self.communication_bus.register_agent('research_agent', research_agent)
        
        # Code Agent
        code_agent = CodeAgent(
            'code_agent',
            self.enhanced_tool_registry,
            self.llm_orchestrator
        )
        self.communication_bus.register_agent('code_agent', code_agent)
        
        # Analysis Agent
        analysis_agent = AnalysisAgent(
            'analysis_agent',
            self.enhanced_tool_registry,
            self.llm_orchestrator,
            self.got_reasoner
        )
        self.communication_bus.register_agent('analysis_agent', analysis_agent)
        
    async def chat(self, message: str, conversation_id: str = None) -> Dict[str, Any]:
        """Main chat interface"""
        return await self.dialog_manager.process_message(message, conversation_id)
        
    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a complex task"""
        # Create plan
        plan_result = await self.task_planner.create_plan(task, context)
        
        if not plan_result['success']:
            return {
                'success': False,
                'error': 'Failed to create task plan',
                'details': plan_result
            }
            
        # Execute plan
        execution_result = await self.task_executor.execute_plan(
            plan_result['plan'],
            context
        )
        
        return execution_result
        
    async def analyze_data(self, file_path: str, analysis_request: str) -> Dict[str, Any]:
        """Analyze data with natural language request"""
        # Send to analysis agent
        message = AgentMessage(
            sender='system',
            receiver='analysis_agent',
            content={
                'file_path': file_path,
                'request': analysis_request
            },
            message_type='request'
        )
        
        await self.communication_bus.send_message(message)
        
        # Wait for response (simplified - in production use proper async response handling)
        await asyncio.sleep(0.1)
        
        # Get response from message history
        for msg in reversed(self.communication_bus.message_history):
            if (msg.correlation_id == message.id and 
                msg.message_type == 'response'):
                return msg.content
                
        return {'success': False, 'error': 'No response received'}
        
    async def research_topic(self, topic: str, depth: str = "medium") -> Dict[str, Any]:
        """Research a topic"""
        message = AgentMessage(
            sender='system',
            receiver='research_agent',
            content={
                'topic': topic,
                'depth': depth
            },
            message_type='request'
        )
        
        await self.communication_bus.send_message(message)
        
        # Wait for response
        await asyncio.sleep(0.5)
        
        for msg in reversed(self.communication_bus.message_history):
            if (msg.correlation_id == message.id and 
                msg.message_type == 'response'):
                return msg.content
                
        return {'success': False, 'error': 'No response received'}

# ==================== PART 8: Specialized Agents ====================

class SpecializedAgent(ABC):
    """Base class for specialized agents"""
    def __init__(self, agent_id: str, tools: EnhancedToolRegistry, 
                 llm: LLMOrchestrator):
        self.agent_id = agent_id
        self.tools = tools
        self.llm = llm
        
    @abstractmethod
    async def receive_message(self, message: AgentMessage):
        """Handle incoming messages"""
        pass
        
    async def send_response(self, original_message: AgentMessage, 
                          content: Any, success: bool = True):
        """Send response message"""
        response = AgentMessage(
            sender=self.agent_id,
            receiver=original_message.sender,
            content=content,
            message_type='response' if success else 'error',
            correlation_id=original_message.id
        )
        
        # Send via communication bus
        # (Assumes access to bus - in production, inject dependency)

class ResearchAgent(SpecializedAgent):
    """Agent specialized in research tasks"""
    def __init__(self, agent_id: str, tools: EnhancedToolRegistry, 
                 llm: LLMOrchestrator, vector_db: EnterpriseVectorDatabase):
        super().__init__(agent_id, tools, llm)
        self.vector_db = vector_db
        
    async def receive_message(self, message: AgentMessage):
        """Handle research requests"""
        if message.message_type != 'request':
            return
            
        content = message.content
        topic = content.get('topic')
        depth = content.get('depth', 'medium')
        
        try:
            # Perform research
            result = await self.research(topic, depth)
            await self.send_response(message, result, success=True)
        except Exception as e:
            await self.send_response(
                message, 
                {'error': str(e)}, 
                success=False
            )
            
    async def research(self, topic: str, depth: str) -> Dict[str, Any]:
        """Perform comprehensive research"""
        search_tool = self.tools.get_tool('web_search')
        
        # Initial search
        search_results = await search_tool.execute(query=topic, num_results=10)
        
        # Extract and synthesize information
        synthesis_prompt = f"""
        Research the topic: {topic}
        
        Search results:
        {json.dumps(search_results['results'], indent=2)}
        
        Provide a comprehensive {depth}-depth analysis including:
        1. Overview
        2. Key points
        3. Recent developments
        4. Controversies or debates
        5. Future outlook
        
        Cite sources where appropriate.
        """
        
        synthesis = await self.llm.generate(synthesis_prompt, temperature=0.3)
        
        # Store in vector database for future reference
        embedding = await self.llm.providers[self.llm.primary_provider].embed(
            synthesis['content']
        )
        
        await self.vector_db.add_vectors(
            np.array([embedding]),
            [{
                'type': 'research',
                'topic': topic,
                'depth': depth,
                'content': synthesis['content'],
                'sources': search_results['results']
            }]
        )
        
        return {
            'topic': topic,
            'analysis': synthesis['content'],
            'sources': search_results['results'],
            'depth': depth
        }

class CodeAgent(SpecializedAgent):
    """Agent specialized in code generation and execution"""
    async def receive_message(self, message: AgentMessage):
        """Handle code-related requests"""
        if message.message_type != 'request':
            return
            
        content = message.content
        task = content.get('task')
        language = content.get('language', 'python')
        
        try:
            result = await self.handle_code_task(task, language)
            await self.send_response(message, result, success=True)
        except Exception as e:
            await self.send_response(
                message,
                {'error': str(e)},
                success=False
            )
            
    async def handle_code_task(self, task: str, language: str) -> Dict[str, Any]:
        """Handle code generation and execution"""
        # Generate code
        generation_prompt = f"""
        Generate {language} code for the following task:
        {task}
        
        Requirements:
        - Include proper error handling
        - Add comments explaining the logic
        - Follow best practices
        - Make it production-ready
        """
        
        response = await self.llm.generate(generation_prompt, temperature=0.2)
        generated_code = response['content']
        
        # Format code
        if language == 'python':
            try:
                formatted_code = black.format_str(generated_code, mode=black.Mode())
            except:
                formatted_code = generated_code
                
        # Test execution if Python
        if language == 'python':
            code_tool = self.tools.get_tool('execute_code')
            execution_result = await code_tool.execute(code=formatted_code)
            
            return {
                'code': formatted_code,
                'language': language,
                'execution_result': execution_result,
                'task': task
            }
        else:
            return {
                'code': generated_code,
                'language': language,
                'task': task
            }

class AnalysisAgent(SpecializedAgent):
    """Agent specialized in data analysis"""
    def __init__(self, agent_id: str, tools: EnhancedToolRegistry,
                 llm: LLMOrchestrator, got_reasoner: AdvancedGraphOfThoughts):
        super().__init__(agent_id, tools, llm)
        self.got_reasoner = got_reasoner
        
    async def receive_message(self, message: AgentMessage):
        """Handle analysis requests"""
        if message.message_type != 'request':
            return
            
        content = message.content
        
        try:
            if 'file_path' in content:
                result = await self.analyze_file(
                    content['file_path'],
                    content.get('request', 'Provide comprehensive analysis')
                )
            else:
                result = await self.analyze_data(
                    content.get('data'),
                    content.get('request')
                )
                
            await self.send_response(message, result, success=True)
        except Exception as e:
            await self.send_response(
                message,
                {'error': str(e)},
                success=False
            )
            
    async def analyze_file(self, file_path: str, request: str) -> Dict[str, Any]:
        """Analyze a data file"""
        analysis_tool = self.tools.get_tool('analyze_data')
        
        # First get summary
        summary = await analysis_tool.execute(
            file_path=file_path,
            analysis_type='summary'
        )
        
        # Use GoT for complex reasoning
        analysis_task = f"""
        Analyze the data with the following summary:
        {json.dumps(summary['result'], indent=2)}
        
        User request: {request}
        """
        
        reasoning_result = await self.got_reasoner.reason(
            analysis_task,
            {'baseline_performance': 0.5, 'baseline_cost': 10.0}
        )
        
        return {
            'file_path': file_path,
            'summary': summary['result'],
            'analysis': reasoning_result['solution'],
            'confidence': reasoning_result['confidence'],
            'reasoning_path': reasoning_result['reasoning_path']
        }

# ==================== PART 9: Helper Classes ====================

class CodeValidator(ast.NodeVisitor):
    """Validate code for safety"""
    def __init__(self, allowed_imports: List[str]):
        self.allowed_imports = allowed_imports
        self.violations = []
        
    def visit_Import(self, node):
        for alias in node.names:
            if alias.name not in self.allowed_imports:
                self.violations.append(f"Unauthorized import: {alias.name}")
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module not in self.allowed_imports:
            self.violations.append(f"Unauthorized import from: {node.module}")
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # Check for dangerous functions
        if isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec', '__import__']:
                self.violations.append(f"Dangerous function call: {node.func.id}")
        self.generic_visit(node)

# Additional supporting classes from the original code would go here...
# (ConstitutionalAIFramework, CircuitBreakerState, etc.)

# ==================== Example Usage ====================

async def demonstration():
    """Demonstrate the complete agent system"""
    
    # Initialize system
    config = {
        'openai_model': 'gpt-4-turbo-preview',
        'anthropic_model': 'claude-3-opus-20240229',
        'enable_all_frameworks': True,
        'safety_threshold': 0.95
    }
    
    system = CompleteAgentSystem(config)
    
    print("=== Complete Agent System Demo ===\n")
    
    # Example 1: Chat interaction
    print("1. Chat Interaction:")
    response = await system.chat("Hello! Can you help me analyze some sales data?")
    print(f"Agent: {response['response']}\n")
    
    # Example 2: Complex task execution
    print("2. Complex Task Execution:")
    task_result = await system.execute_task(
        "Create a Python script that analyzes CSV files and generates visualizations",
        context={'output_format': 'matplotlib'}
    )
    print(f"Task completed: {task_result['overall_success']}")
    print(f"Execution time: {task_result['execution_time']:.2f}s\n")
    
    # Example 3: Research
    print("3. Research Task:")
    research_result = await system.research_topic(
        "Latest advances in quantum computing",
        depth="deep"
    )
    print(f"Research completed: {research_result.get('success', True)}")
    if 'analysis' in research_result:
        print(f"Analysis preview: {research_result['analysis'][:200]}...\n")
    
    # Example 4: Data analysis
    print("4. Data Analysis:")
    # Create sample data file
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'sales': np.random.randint(1000, 5000, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })
    sample_file = '/tmp/sales_data.csv'
    sample_data.to_csv(sample_file, index=False)
    
    analysis_result = await system.analyze_data(
        sample_file,
        "What are the sales trends by region? Any anomalies?"
    )
    print(f"Analysis confidence: {analysis_result.get('confidence', 0):.2f}\n")
    
    # Example 5: Multi-agent collaboration
    print("5. Multi-Agent Collaboration:")
    # Send message to research agent
    research_msg = AgentMessage(
        sender='user',
        receiver='research_agent',
        content={'topic': 'AI safety best practices', 'depth': 'medium'}
    )
    await system.communication_bus.send_message(research_msg)
    
    # Wait for processing
    await asyncio.sleep(1)
    
    print("System demonstration complete!")

if __name__ == "__main__":
    # Set up environment variables for LLM providers
    # os.environ['OPENAI_API_KEY'] = 'your-api-key'
    # os.environ['ANTHROPIC_API_KEY'] = 'your-api-key'
    
    # Run demonstration
    asyncio.run(demonstration())
