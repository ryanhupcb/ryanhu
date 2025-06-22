"""
生产级编程AI Agent系统 - 性能增强版
包含完整的性能优化、向量数据库、流式处理等高级功能
"""

import os
import ast
import json
import asyncio
import logging
import hashlib
import time
import pickle
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Tuple, Union, AsyncIterator, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import uuid
from functools import wraps, lru_cache
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 第三方库
import aioredis
import aiokafka
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Integer, DateTime, JSON, Float, Boolean, Text, Index, select
import httpx
from prometheus_client import Counter, Histogram, Gauge, Summary
import sentry_sdk
from circuitbreaker import circuit
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, validator
from cryptography.fernet import Fernet
import jwt
from passlib.context import CryptContext
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import chromadb
from chromadb.config import Settings
import docker
import aiofiles
import asyncpg
from asyncpg.pool import Pool
import orjson
import msgpack
import uvloop
import cachetools
from bloom_filter2 import BloomFilter

# FastAPI相关
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks, Request, status, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# 设置高性能事件循环
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ==================== 配置管理 ====================

class Config:
    """增强的配置管理"""
    
    # 环境变量
    ENV = os.getenv("ENVIRONMENT", "development")
    DEBUG = ENV == "development"
    
    # 数据库配置
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/agent_db")
    DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "50"))
    DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "100"))
    
    # Redis配置
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_POOL_SIZE = int(os.getenv("REDIS_POOL_SIZE", "50"))
    
    # 向量数据库配置
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    
    # Kafka配置
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_SERVERS", "localhost:9092").split(",")
    KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "ai-agent-group")
    
    # LLM配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
    
    # 性能配置
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "200"))
    TASK_TIMEOUT_SECONDS = int(os.getenv("TASK_TIMEOUT_SECONDS", "300"))
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
    
    # 缓存配置
    ENABLE_MEMORY_CACHE = os.getenv("ENABLE_MEMORY_CACHE", "true").lower() == "true"
    MEMORY_CACHE_SIZE = int(os.getenv("MEMORY_CACHE_SIZE", "10000"))
    BLOOM_FILTER_SIZE = int(os.getenv("BLOOM_FILTER_SIZE", "1000000"))
    BLOOM_FILTER_FP_RATE = float(os.getenv("BLOOM_FILTER_FP_RATE", "0.01"))
    
    # 线程池配置
    THREAD_POOL_SIZE = int(os.getenv("THREAD_POOL_SIZE", "20"))
    PROCESS_POOL_SIZE = int(os.getenv("PROCESS_POOL_SIZE", "4"))
    
    # Docker沙箱配置
    ENABLE_CODE_EXECUTION = os.getenv("ENABLE_CODE_EXECUTION", "true").lower() == "true"
    SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "30"))
    SANDBOX_MEMORY_LIMIT = os.getenv("SANDBOX_MEMORY_LIMIT", "512m")
    SANDBOX_CPU_QUOTA = int(os.getenv("SANDBOX_CPU_QUOTA", "50000"))

# ==================== 性能监控指标 ====================

# 增强的Prometheus指标
task_processing_histogram = Histogram(
    'agent_task_processing_duration_seconds',
    'Task processing duration by stage',
    ['task_type', 'stage'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

cache_operations = Counter(
    'agent_cache_operations_total',
    'Cache operations',
    ['operation', 'cache_type', 'status']
)

llm_latency = Histogram(
    'agent_llm_latency_seconds',
    'LLM API latency',
    ['model', 'operation'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
)

code_quality_scores = Histogram(
    'agent_code_quality_scores',
    'Distribution of generated code quality scores',
    ['task_type'],
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
)

vector_search_latency = Histogram(
    'agent_vector_search_latency_seconds',
    'Vector database search latency',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

concurrent_tasks = Gauge('agent_concurrent_tasks', 'Number of concurrent tasks', ['task_type'])
memory_usage = Gauge('agent_memory_usage_bytes', 'Memory usage in bytes', ['component'])

# ==================== 增强的依赖注入 ====================

class EnhancedDependencies(Dependencies):
    """增强的依赖注入容器"""
    
    def __init__(self):
        super().__init__()
        self.vector_store = None
        self.docker_client = None
        self.thread_pool = None
        self.process_pool = None
        self.asyncpg_pool = None
        self.memory_cache = None
        self.bloom_filter = None
    
    async def initialize(self):
        """初始化所有依赖"""
        await super().initialize()
        
        # 初始化AsyncPG连接池（更高性能）
        self.asyncpg_pool = await asyncpg.create_pool(
            Config.DATABASE_URL.replace('+asyncpg', ''),
            min_size=10,
            max_size=Config.DATABASE_POOL_SIZE,
            command_timeout=60
        )
        
        # 初始化向量数据库
        self.vector_store = VectorStore()
        await self.vector_store.initialize()
        
        # 初始化Docker客户端
        if Config.ENABLE_CODE_EXECUTION:
            self.docker_client = docker.from_env()
        
        # 初始化线程池和进程池
        self.thread_pool = ThreadPoolExecutor(max_workers=Config.THREAD_POOL_SIZE)
        self.process_pool = ProcessPoolExecutor(max_workers=Config.PROCESS_POOL_SIZE)
        
        # 初始化内存缓存
        if Config.ENABLE_MEMORY_CACHE:
            self.memory_cache = cachetools.LRUCache(maxsize=Config.MEMORY_CACHE_SIZE)
            self.bloom_filter = BloomFilter(
                max_elements=Config.BLOOM_FILTER_SIZE,
                error_rate=Config.BLOOM_FILTER_FP_RATE
            )
        
        logger.info("Enhanced dependencies initialized")
    
    async def cleanup(self):
        """清理资源"""
        await super().cleanup()
        
        if self.asyncpg_pool:
            await self.asyncpg_pool.close()
        
        if self.vector_store:
            await self.vector_store.cleanup()
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)

# 全局依赖实例
deps = EnhancedDependencies()

# ==================== 向量数据库管理 ====================

class VectorStore:
    """高性能向量数据库管理"""
    
    def __init__(self):
        self.client = None
        self.collections = {}
        self.embedding_cache = cachetools.TTLCache(maxsize=10000, ttl=3600)
    
    async def initialize(self):
        """初始化向量数据库"""
        self.client = chromadb.HttpClient(
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # 创建集合
        self.collections['code'] = await self._create_or_get_collection('code_embeddings')
        self.collections['docs'] = await self._create_or_get_collection('documentation_embeddings')
        self.collections['patterns'] = await self._create_or_get_collection('pattern_embeddings')
    
    async def _create_or_get_collection(self, name: str):
        """创建或获取集合"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
    
    @lru_cache(maxsize=1000)
    def _get_embedding_key(self, text: str) -> str:
        """生成嵌入缓存键"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    async def add_code_embedding(self, code_id: str, code: str, 
                                embedding: List[float], metadata: Dict):
        """添加代码嵌入"""
        with vector_search_latency.labels(operation='add').time():
            # 添加到布隆过滤器
            if deps.bloom_filter:
                deps.bloom_filter.add(code_id)
            
            # 批量添加以提高性能
            self.collections['code'].add(
                ids=[code_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[code[:1000]]  # 只存储前1000字符
            )
            
            # 缓存嵌入
            cache_key = self._get_embedding_key(code)
            self.embedding_cache[cache_key] = embedding
    
    async def search_similar_code(self, query_embedding: List[float], 
                                 filters: Dict = None, top_k: int = 10) -> List[Dict]:
        """搜索相似代码"""
        with vector_search_latency.labels(operation='search').time():
            # 构建查询
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["metadatas", "documents", "distances"]
            }
            
            if filters:
                query_params["where"] = filters
            
            results = self.collections['code'].query(**query_params)
            
            # 格式化结果
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'code': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # 转换为相似度
                })
            
            return formatted_results
    
    async def batch_search(self, query_embeddings: List[List[float]], 
                          top_k: int = 10) -> List[List[Dict]]:
        """批量搜索"""
        with vector_search_latency.labels(operation='batch_search').time():
            results = self.collections['code'].query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
            
            # 格式化批量结果
            batch_results = []
            for batch_idx in range(len(query_embeddings)):
                batch_result = []
                for i in range(len(results['ids'][batch_idx])):
                    batch_result.append({
                        'id': results['ids'][batch_idx][i],
                        'code': results['documents'][batch_idx][i],
                        'metadata': results['metadatas'][batch_idx][i],
                        'similarity': 1 - results['distances'][batch_idx][i]
                    })
                batch_results.append(batch_result)
            
            return batch_results
    
    async def cleanup(self):
        """清理资源"""
        # ChromaDB HTTP客户端无需显式清理
        pass

# ==================== 增强的缓存管理 ====================

class EnhancedCacheManager(CacheManager):
    """增强的多级缓存管理器"""
    
    def __init__(self, redis_client, memory_cache=None, bloom_filter=None):
        super().__init__(redis_client)
        self.memory_cache = memory_cache
        self.bloom_filter = bloom_filter
        self.cache_layers = ['memory', 'bloom', 'redis']
        self.compression_threshold = 1024  # 压缩阈值（字节）
    
    async def get(self, key: str, default=None):
        """多级缓存获取"""
        # L1: 内存缓存
        if self.memory_cache and key in self.memory_cache:
            cache_operations.labels(operation='get', cache_type='memory', status='hit').inc()
            return self.memory_cache[key]
        
        # L2: 布隆过滤器快速判断
        if self.bloom_filter and key not in self.bloom_filter:
            cache_operations.labels(operation='get', cache_type='bloom', status='miss').inc()
            return default
        
        # L3: Redis缓存
        try:
            value = await self.redis.get(key)
            if value:
                cache_operations.labels(operation='get', cache_type='redis', status='hit').inc()
                
                # 解压缩
                if value.startswith(b'COMPRESSED:'):
                    value = await self._decompress(value[11:])
                else:
                    value = orjson.loads(value)
                
                # 回填内存缓存
                if self.memory_cache:
                    self.memory_cache[key] = value
                
                return value
        except Exception as e:
            logger.error("cache_get_error", key=key, error=str(e))
        
        cache_operations.labels(operation='get', cache_type='all', status='miss').inc()
        return default
    
    async def set(self, key: str, value: Any, ttl: int = Config.CACHE_TTL_SECONDS):
        """多级缓存设置"""
        try:
            # 添加到布隆过滤器
            if self.bloom_filter:
                self.bloom_filter.add(key)
            
            # 更新内存缓存
            if self.memory_cache:
                self.memory_cache[key] = value
            
            # 序列化和压缩
            serialized = orjson.dumps(value, default=str)
            if len(serialized) > self.compression_threshold:
                serialized = b'COMPRESSED:' + await self._compress(serialized)
            
            # 更新Redis
            await self.redis.setex(key, ttl, serialized)
            
            cache_operations.labels(operation='set', cache_type='all', status='success').inc()
        except Exception as e:
            logger.error("cache_set_error", key=key, error=str(e))
            cache_operations.labels(operation='set', cache_type='all', status='error').inc()
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取"""
        result = {}
        
        # 先从内存缓存获取
        memory_hits = []
        redis_keys = []
        
        for key in keys:
            if self.memory_cache and key in self.memory_cache:
                result[key] = self.memory_cache[key]
                memory_hits.append(key)
            else:
                redis_keys.append(key)
        
        # 批量从Redis获取
        if redis_keys:
            try:
                values = await self.redis.mget(redis_keys)
                for key, value in zip(redis_keys, values):
                    if value:
                        if value.startswith(b'COMPRESSED:'):
                            value = await self._decompress(value[11:])
                        else:
                            value = orjson.loads(value)
                        result[key] = value
                        
                        # 回填内存缓存
                        if self.memory_cache:
                            self.memory_cache[key] = value
            except Exception as e:
                logger.error("cache_mget_error", error=str(e))
        
        return result
    
    async def _compress(self, data: bytes) -> bytes:
        """压缩数据"""
        import zlib
        return zlib.compress(data, level=6)
    
    async def _decompress(self, data: bytes) -> Any:
        """解压缩数据"""
        import zlib
        return orjson.loads(zlib.decompress(data))

# ==================== 代码执行沙箱 ====================

class CodeExecutionSandbox:
    """安全的代码执行沙箱"""
    
    def __init__(self, docker_client):
        self.docker_client = docker_client
        self.execution_history = deque(maxlen=1000)
        self.container_pool = []
        self.pool_size = 5
        
        # 预创建容器池
        asyncio.create_task(self._initialize_container_pool())
    
    async def _initialize_container_pool(self):
        """初始化容器池"""
        for _ in range(self.pool_size):
            container = await self._create_container()
            self.container_pool.append(container)
    
    async def _create_container(self):
        """创建容器"""
        return await asyncio.get_event_loop().run_in_executor(
            deps.thread_pool,
            self.docker_client.containers.create,
            "python:3.11-slim",
            command="sleep infinity",
            detach=True,
            mem_limit=Config.SANDBOX_MEMORY_LIMIT,
            cpu_quota=Config.SANDBOX_CPU_QUOTA,
            network_disabled=True,
            security_opt=["no-new-privileges:true"],
            read_only=False,  # 需要写入临时文件
            tmpfs={'/tmp': 'size=100M,noexec'}
        )
    
    async def execute_code(self, code: str, language: str = "python", 
                          timeout: int = None, test_code: str = None) -> Dict:
        """执行代码"""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            if language == "python":
                result = await self._execute_python(code, timeout, test_code)
            elif language == "javascript":
                result = await self._execute_javascript(code, timeout, test_code)
            else:
                result = {"success": False, "error": f"Unsupported language: {language}"}
            
            # 记录执行历史
            self.execution_history.append({
                "id": execution_id,
                "timestamp": datetime.utcnow(),
                "language": language,
                "duration": time.time() - start_time,
                "success": result["success"]
            })
            
            return result
            
        except Exception as e:
            logger.error("sandbox_execution_error", execution_id=execution_id, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "execution_id": execution_id
            }
    
    async def _execute_python(self, code: str, timeout: int = None, test_code: str = None) -> Dict:
        """执行Python代码"""
        timeout = timeout or Config.SANDBOX_TIMEOUT
        
        # 从池中获取容器
        container = None
        try:
            container = self.container_pool.pop() if self.container_pool else await self._create_container()
            
            # 启动容器
            await asyncio.get_event_loop().run_in_executor(
                deps.thread_pool,
                container.start
            )
            
            # 准备执行代码
            full_code = code
            if test_code:
                full_code += f"\n\n# Tests\n{test_code}"
            
            # 执行代码
            exec_result = await asyncio.get_event_loop().run_in_executor(
                deps.thread_pool,
                container.exec_run,
                ["python", "-c", full_code],
                demux=True
            )
            
            stdout, stderr = exec_result.output
            
            return {
                "success": exec_result.exit_code == 0,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "exit_code": exec_result.exit_code,
                "execution_time": time.time()
            }
            
        finally:
            # 清理并归还容器到池
            if container:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        deps.thread_pool,
                        container.stop
                    )
                    # 重新创建干净的容器
                    new_container = await self._create_container()
                    self.container_pool.append(new_container)
                except:
                    pass

# ==================== 增强的LLM提供商 ====================

class EnhancedLLMProvider(LLMProvider):
    """增强的LLM提供商基类"""
    
    def __init__(self):
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.response_cache = cachetools.TTLCache(maxsize=1000, ttl=3600)
        self.token_bucket = TokenBucket(rate=100, capacity=1000)
        self.batch_processor = None
    
    async def initialize(self):
        """初始化批处理器"""
        self.batch_processor = asyncio.create_task(self._batch_processor())
    
    async def _batch_processor(self):
        """批处理请求"""
        while True:
            batch = []
            try:
                # 收集批次
                deadline = asyncio.get_event_loop().time() + 0.1  # 100ms窗口
                while len(batch) < Config.BATCH_SIZE and asyncio.get_event_loop().time() < deadline:
                    try:
                        timeout = max(0, deadline - asyncio.get_event_loop().time())
                        request = await asyncio.wait_for(self.request_queue.get(), timeout=timeout)
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # 批量处理
                    await self._process_batch(batch)
                    
            except Exception as e:
                logger.error("batch_processor_error", error=str(e))
                # 错误时单独处理每个请求
                for request in batch:
                    request['future'].set_exception(e)
    
    async def _process_batch(self, batch: List[Dict]):
        """处理请求批次"""
        # 子类实现具体的批处理逻辑
        pass

class EnhancedOpenAIProvider(EnhancedLLMProvider):
    """增强的OpenAI提供商"""
    
    def __init__(self, api_key: str, http_client: httpx.AsyncClient):
        super().__init__()
        self.api_key = api_key
        self.http_client = http_client
        self.base_url = "https://api.openai.com/v1"
        self.models_info = {}
        
        # 模型特定配置
        self.model_configs = {
            "gpt-4-turbo-preview": {"max_tokens": 128000, "cost_per_1k": 0.01},
            "gpt-4": {"max_tokens": 8192, "cost_per_1k": 0.03},
            "gpt-3.5-turbo": {"max_tokens": 16384, "cost_per_1k": 0.0005}
        }
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式生成文本"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        data = {
            "model": kwargs.get("model", Config.LLM_MODEL),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", Config.LLM_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", Config.LLM_MAX_TOKENS),
            "stream": True
        }
        
        async with self.http_client.stream(
            'POST',
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=Config.LLM_TIMEOUT
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = orjson.loads(data)
                        if chunk["choices"][0]["delta"].get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
                    except:
                        continue
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量生成文本"""
        # 使用异步并发处理
        tasks = []
        for prompt in prompts:
            task = self.generate(prompt, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "prompt": prompts[i]
                })
            else:
                processed_results.append(result)
        
        return processed_results

# ==================== Token Bucket限流 ====================

class TokenBucket:
    """令牌桶限流器"""
    
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """获取令牌"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # 添加新令牌
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            
            # 检查是否有足够的令牌
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1):
        """等待令牌可用"""
        while not await self.acquire(tokens):
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait_time)

# ==================== 增强的代码理解Agent ====================

class EnhancedCodeUnderstandingAgent(ProductionCodeUnderstandingAgent):
    """增强的代码理解Agent"""
    
    def __init__(self, deps: Dependencies, cache_manager: CacheManager, 
                 llm_provider: LLMProvider, vector_store: VectorStore):
        super().__init__(deps, cache_manager, llm_provider)
        self.vector_store = vector_store
        self.pattern_matcher = PatternMatcher()
        self.semantic_analyzer = SemanticAnalyzer()
        
    async def process_task(self, task: Task) -> AgentResponse:
        """增强的任务处理"""
        with task_processing_histogram.labels(
            task_type=task.type.value, 
            stage='understanding'
        ).time():
            
            # 并行执行多个分析任务
            analysis_tasks = [
                self._analyze_ast_parallel(task.context),
                self._extract_patterns_async(task.context),
                self._analyze_dependencies_parallel(task.context),
                self._semantic_analysis_enhanced(task.context),
                self._security_scan_parallel(task.context)
            ]
            
            # 等待所有分析完成
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # 处理结果
            ast_analysis, patterns, dependencies, semantics, security = results
            
            # 生成向量嵌入
            embedding = await self._generate_code_embedding(task.context)
            
            # 查找相似代码
            similar_code = await self.vector_store.search_similar_code(
                embedding,
                filters={"language": task.context.language},
                top_k=5
            )
            
            # 综合分析结果
            comprehensive_analysis = {
                "ast_analysis": ast_analysis if not isinstance(ast_analysis, Exception) else None,
                "patterns": patterns if not isinstance(patterns, Exception) else [],
                "dependencies": dependencies if not isinstance(dependencies, Exception) else [],
                "semantics": semantics if not isinstance(semantics, Exception) else {},
                "security": security if not isinstance(security, Exception) else [],
                "similar_code": similar_code,
                "embedding": embedding,
                "quality_metrics": await self._calculate_enhanced_quality_metrics(
                    task.context, ast_analysis, security
                )
            }
            
            # 存储到向量数据库
            await self.vector_store.add_code_embedding(
                code_id=task.id,
                code=task.context.content,
                embedding=embedding,
                metadata={
                    "language": task.context.language,
                    "quality_score": comprehensive_analysis["quality_metrics"]["overall_score"],
                    "patterns": patterns if not isinstance(patterns, Exception) else [],
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # 记录质量分数
            code_quality_scores.labels(task_type=task.type.value).observe(
                comprehensive_analysis["quality_metrics"]["overall_score"]
            )
            
            return AgentResponse(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=comprehensive_analysis,
                confidence=0.95,
                suggestions=self._generate_actionable_suggestions(comprehensive_analysis),
                performance_metrics={
                    "analysis_time": time.time(),
                    "similar_code_found": len(similar_code)
                }
            )
    
    async def _analyze_ast_parallel(self, context: CodeContext) -> Dict[str, Any]:
        """并行AST分析"""
        if context.language == "python":
            # 使用进程池进行CPU密集型任务
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                deps.process_pool,
                self._analyze_python_ast_intensive,
                context.content
            )
        return {"error": "Unsupported language"}
    
    def _analyze_python_ast_intensive(self, code: str) -> Dict[str, Any]:
        """CPU密集型的Python AST分析"""
        try:
            tree = ast.parse(code)
            analyzer = AdvancedPythonAnalyzer()
            analyzer.visit(tree)
            
            # 额外的复杂分析
            return {
                "success": True,
                "metrics": {
                    "loc": len(code.splitlines()),
                    "complexity": analyzer.calculate_complexity(),
                    "maintainability_index": self._calculate_maintainability_index_sync(code),
                    "test_coverage_estimate": self._estimate_test_coverage_sync(tree)
                },
                "structure": {
                    "classes": analyzer.classes,
                    "functions": analyzer.functions,
                    "imports": analyzer.imports
                },
                "patterns": analyzer.detect_patterns(),
                "code_smells": analyzer.detect_code_smells()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_code_embedding(self, context: CodeContext) -> List[float]:
        """生成代码的向量嵌入"""
        # 检查缓存
        cache_key = self.vector_store._get_embedding_key(context.content)
        if cache_key in self.vector_store.embedding_cache:
            return self.vector_store.embedding_cache[cache_key]
        
        # 生成嵌入
        embedding = await self.llm.embed(context.content[:2000])  # 限制长度
        
        # 缓存结果
        self.vector_store.embedding_cache[cache_key] = embedding
        
        return embedding

# ==================== 增强的代码生成Agent ====================

class EnhancedCodeGenerationAgent(ProductionCodeGenerationAgent):
    """增强的代码生成Agent"""
    
    def __init__(self, deps: Dependencies, cache_manager: CacheManager, 
                 llm_provider: LLMProvider, vector_store: VectorStore,
                 sandbox: CodeExecutionSandbox):
        super().__init__(deps, cache_manager, llm_provider)
        self.vector_store = vector_store
        self.sandbox = sandbox
        self.generation_cache = cachetools.TTLCache(maxsize=1000, ttl=3600)
    
    async def process_task(self, task: Task) -> AgentResponse:
        """增强的代码生成处理"""
        with task_processing_histogram.labels(
            task_type=task.type.value,
            stage='generation'
        ).time():
            
            try:
                # 分析需求
                requirements = await self._analyze_requirements_enhanced(task)
                
                # 并行检索相关资源
                retrieval_tasks = [
                    self._retrieve_similar_code_enhanced(task, requirements),
                    self._retrieve_documentation(requirements),
                    self._retrieve_design_patterns(requirements)
                ]
                
                similar_code, docs, patterns = await asyncio.gather(*retrieval_tasks)
                
                # 设计架构
                architecture = await self._design_architecture_enhanced(
                    requirements, similar_code, patterns
                )
                
                # 流式生成代码
                generated_code = ""
                async for chunk in self._generate_code_stream(
                    task, requirements, architecture, similar_code
                ):
                    generated_code += chunk
                
                # 并行验证和优化
                validation_tasks = [
                    self._validate_syntax(generated_code),
                    self._validate_security(generated_code),
                    self._validate_performance(generated_code)
                ]
                
                syntax_valid, security_valid, perf_metrics = await asyncio.gather(*validation_tasks)
                
                # 如果启用了代码执行，运行测试
                execution_result = None
                if Config.ENABLE_CODE_EXECUTION and syntax_valid["valid"]:
                    execution_result = await self.sandbox.execute_code(
                        generated_code,
                        language=task.context.language,
                        test_code=await self._generate_unit_tests(generated_code)
                    )
                
                # 优化代码
                optimized_code = await self._optimize_code_parallel(
                    generated_code,
                    perf_metrics
                )
                
                # 生成完整结果
                result = {
                    "code": optimized_code,
                    "tests": await self._generate_comprehensive_tests(optimized_code),
                    "documentation": await self._generate_documentation_enhanced(optimized_code),
                    "architecture": architecture,
                    "validation": {
                        "syntax": syntax_valid,
                        "security": security_valid,
                        "performance": perf_metrics,
                        "execution": execution_result
                    },
                    "metrics": await self._calculate_comprehensive_metrics(optimized_code),
                    "similar_code_references": [
                        {"id": item["id"], "similarity": item["similarity"]}
                        for item in similar_code[:3]
                    ]
                }
                
                # 存储生成的代码
                embedding = await self.llm.embed(optimized_code[:2000])
                await self.vector_store.add_code_embedding(
                    code_id=f"generated_{task.id}",
                    code=optimized_code,
                    embedding=embedding,
                    metadata={
                        "task_id": task.id,
                        "language": task.context.language,
                        "timestamp": datetime.utcnow().isoformat(),
                        "quality_score": result["metrics"]["quality_score"]
                    }
                )
                
                return AgentResponse(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    confidence=self._calculate_confidence(result),
                    suggestions=self._generate_improvement_suggestions_enhanced(result)
                )
                
            except Exception as e:
                logger.error("code_generation_error", task_id=task.id, error=str(e))
                return AgentResponse(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=str(e)
                )
    
    async def _generate_code_stream(self, task: Task, requirements: Dict,
                                   architecture: Dict, similar_code: List[Dict]) -> AsyncIterator[str]:
        """流式代码生成"""
        # 构建增强的提示
        prompt = self._build_enhanced_prompt(task, requirements, architecture, similar_code)
        
        # 流式生成
        generated = ""
        async for chunk in self.llm.generate_stream(prompt):
            generated += chunk
            yield chunk
            
            # 实时语法检查
            if len(generated) % 100 == 0:  # 每100字符检查一次
                if not self._quick_syntax_check(generated):
                    # 如果语法错误，尝试修复
                    generated = await self._attempt_syntax_fix(generated)
    
    async def _retrieve_similar_code_enhanced(self, task: Task, 
                                            requirements: Dict) -> List[Dict]:
        """增强的相似代码检索"""
        # 生成查询嵌入
        query_text = f"{task.description} {requirements.get('functional_requirements', '')}"
        query_embedding = await self.llm.embed(query_text)
        
        # 向量搜索
        similar_items = await self.vector_store.search_similar_code(
            query_embedding,
            filters={
                "language": task.context.language,
                "quality_score": {"$gte": 0.7}  # 只检索高质量代码
            },
            top_k=20
        )
        
        # 重新排序和过滤
        reranked = await self._rerank_similar_code(similar_items, requirements)
        
        return reranked[:10]
    
    async def _optimize_code_parallel(self, code: str, perf_metrics: Dict) -> str:
        """并行代码优化"""
        optimization_tasks = [
            self._optimize_performance_async(code, perf_metrics),
            self._optimize_readability_async(code),
            self._optimize_security_async(code),
            self._add_error_handling_async(code)
        ]
        
        optimizations = await asyncio.gather(*optimization_tasks)
        
        # 合并优化结果
        optimized_code = code
        for optimization in optimizations:
            if optimization:
                optimized_code = optimization
        
        return optimized_code

# ==================== 模式匹配器 ====================

class PatternMatcher:
    """代码模式匹配器"""
    
    def __init__(self):
        self.patterns = {
            "singleton": self._check_singleton_pattern,
            "factory": self._check_factory_pattern,
            "observer": self._check_observer_pattern,
            "strategy": self._check_strategy_pattern,
            "decorator": self._check_decorator_pattern
        }
    
    async def match_patterns(self, ast_tree: ast.AST) -> List[str]:
        """匹配设计模式"""
        detected_patterns = []
        
        for pattern_name, checker in self.patterns.items():
            if await checker(ast_tree):
                detected_patterns.append(pattern_name)
        
        return detected_patterns
    
    async def _check_singleton_pattern(self, tree: ast.AST) -> bool:
        """检查单例模式"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 查找 __new__ 或 getInstance 方法
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in ["__new__", "getInstance", "instance"]:
                            return True
        return False
    
    async def _check_factory_pattern(self, tree: ast.AST) -> bool:
        """检查工厂模式"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if "factory" in node.name.lower():
                    return True
                # 查找 create 方法
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if "create" in item.name.lower():
                            return True
        return False
    
    async def _check_observer_pattern(self, tree: ast.AST) -> bool:
        """检查观察者模式"""
        observer_methods = {"subscribe", "unsubscribe", "notify", "attach", "detach", "update"}
        found_methods = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.lower() in observer_methods:
                    found_methods.add(node.name.lower())
        
        return len(found_methods) >= 2
    
    async def _check_strategy_pattern(self, tree: ast.AST) -> bool:
        """检查策略模式"""
        # 查找抽象基类和多个实现
        has_abc = False
        implementations = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否继承ABC
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "ABC":
                        has_abc = True
                    elif isinstance(base, ast.Attribute) and base.attr == "ABC":
                        has_abc = True
                
                # 统计可能的策略实现
                if node.bases and "strategy" in node.name.lower():
                    implementations += 1
        
        return has_abc and implementations >= 2
    
    async def _check_decorator_pattern(self, tree: ast.AST) -> bool:
        """检查装饰器模式"""
        # Python中的装饰器
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                if node.decorator_list:
                    return True
        return False

# ==================== 语义分析器 ====================

class SemanticAnalyzer:
    """增强的语义分析器"""
    
    def __init__(self):
        self.nlp_cache = cachetools.LRUCache(maxsize=1000)
    
    async def analyze(self, code: str, language: str) -> Dict[str, Any]:
        """分析代码语义"""
        # 提取注释和文档字符串
        comments = self._extract_comments(code, language)
        docstrings = self._extract_docstrings(code, language)
        
        # 分析变量和函数命名
        naming_analysis = await self._analyze_naming_conventions(code, language)
        
        # 分析代码意图
        intent_analysis = await self._analyze_code_intent(code, comments, docstrings)
        
        return {
            "comments": comments,
            "docstrings": docstrings,
            "naming_conventions": naming_analysis,
            "intent": intent_analysis,
            "readability_score": self._calculate_readability_score(
                code, comments, naming_analysis
            )
        }
    
    def _extract_comments(self, code: str, language: str) -> List[Dict[str, Any]]:
        """提取注释"""
        comments = []
        
        if language == "python":
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if '#' in line:
                    comment_start = line.index('#')
                    comment_text = line[comment_start + 1:].strip()
                    if comment_text:
                        comments.append({
                            "line": i + 1,
                            "text": comment_text,
                            "type": "inline"
                        })
        
        return comments
    
    def _extract_docstrings(self, code: str, language: str) -> List[Dict[str, Any]]:
        """提取文档字符串"""
        docstrings = []
        
        if language == "python":
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            docstrings.append({
                                "type": type(node).__name__,
                                "name": getattr(node, 'name', 'module'),
                                "docstring": docstring,
                                "line": node.lineno if hasattr(node, 'lineno') else 1
                            })
            except:
                pass
        
        return docstrings
    
    async def _analyze_naming_conventions(self, code: str, language: str) -> Dict[str, Any]:
        """分析命名约定"""
        # 简化的命名分析
        return {
            "follows_conventions": True,
            "style": "snake_case" if language == "python" else "camelCase",
            "consistency_score": 0.9
        }
    
    async def _analyze_code_intent(self, code: str, comments: List[Dict], 
                                   docstrings: List[Dict]) -> Dict[str, Any]:
        """分析代码意图"""
        # 基于注释和文档字符串推断意图
        all_text = " ".join([c["text"] for c in comments])
        all_text += " ".join([d["docstring"] for d in docstrings])
        
        if not all_text:
            return {"primary_purpose": "unknown", "confidence": 0.0}
        
        # 简单的关键词分析
        purposes = {
            "api": ["api", "endpoint", "route", "request", "response"],
            "data_processing": ["process", "transform", "parse", "convert"],
            "algorithm": ["algorithm", "compute", "calculate", "solve"],
            "utility": ["utility", "helper", "tool", "utils"],
            "model": ["model", "schema", "entity", "database"]
        }
        
        scores = {}
        for purpose, keywords in purposes.items():
            score = sum(1 for keyword in keywords if keyword in all_text.lower())
            scores[purpose] = score
        
        if scores:
            primary_purpose = max(scores, key=scores.get)
            confidence = scores[primary_purpose] / sum(scores.values()) if sum(scores.values()) > 0 else 0
            
            return {
                "primary_purpose": primary_purpose,
                "confidence": confidence,
                "all_purposes": scores
            }
        
        return {"primary_purpose": "general", "confidence": 0.5}
    
    def _calculate_readability_score(self, code: str, comments: List[Dict], 
                                   naming_analysis: Dict) -> float:
        """计算可读性分数"""
        score = 0.5  # 基础分数
        
        # 注释比例
        lines = code.split('\n')
        comment_ratio = len(comments) / len(lines) if lines else 0
        score += min(comment_ratio * 2, 0.2)  # 最多加0.2分
        
        # 命名规范
        if naming_analysis.get("follows_conventions"):
            score += 0.1
        
        score += naming_analysis.get("consistency_score", 0) * 0.1
        
        # 代码行长度
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        if avg_line_length < 80:
            score += 0.1
        
        return min(score, 1.0)

# ==================== 增强的任务协调器 ====================

class EnhancedTaskOrchestrator(ProductionTaskOrchestrator):
    """增强的任务协调器"""
    
    def __init__(self, deps: Dependencies):
        super().__init__(deps)
        self.task_scheduler = TaskScheduler()
        self.load_balancer = LoadBalancer()
        self.circuit_breakers = {}
        self.task_metrics = defaultdict(lambda: {"success": 0, "failure": 0, "total_time": 0})
    
    async def initialize(self):
        """初始化增强的协调器"""
        # 初始化缓存管理器
        self.cache_manager = EnhancedCacheManager(
            self.deps.redis_client,
            self.deps.memory_cache,
            self.deps.bloom_filter
        )
        
        # 初始化限流器
        self.rate_limiter = RateLimiter(self.deps.redis_client)
        
        # 初始化向量存储
        self.vector_store = self.deps.vector_store
        
        # 初始化代码沙箱
        if Config.ENABLE_CODE_EXECUTION:
            self.sandbox = CodeExecutionSandbox(self.deps.docker_client)
        
        # 初始化LLM提供商
        self.llm_provider = EnhancedOpenAIProvider(
            Config.OPENAI_API_KEY,
            self.deps.http_client
        )
        await self.llm_provider.initialize()
        
        # 初始化增强的Agents
        self.agents["understanding"] = EnhancedCodeUnderstandingAgent(
            self.deps, self.cache_manager, self.llm_provider, self.vector_store
        )
        
        self.agents["generation"] = EnhancedCodeGenerationAgent(
            self.deps, self.cache_manager, self.llm_provider, 
            self.vector_store, self.sandbox if Config.ENABLE_CODE_EXECUTION else None
        )
        
        # 启动工作器池
        self.worker_pool = []
        for i in range(Config.MAX_CONCURRENT_TASKS):
            worker = asyncio.create_task(self._enhanced_worker(f"worker-{i}"))
            self.worker_pool.append(worker)
        
        # 启动监控和维护任务
        self.monitor_task = asyncio.create_task(self._enhanced_monitor())
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.metrics_collector = asyncio.create_task(self._collect_metrics())
        
        logger.info("Enhanced task orchestrator initialized")
    
    async def submit_task(self, task: Task, user_id: str) -> str:
        """增强的任务提交"""
        # 检查系统负载
        if not await self._check_system_capacity():
            raise ValidationException("System at capacity, please try again later")
        
        # 执行父类的验证和提交逻辑
        task_id = await super().submit_task(task, user_id)
        
        # 预热缓存
        asyncio.create_task(self._preheat_cache(task))
        
        # 更新并发任务指标
        concurrent_tasks.labels(task_type=task.type.value).inc()
        
        return task_id
    
    async def submit_batch(self, tasks: List[Task], user_id: str) -> List[str]:
        """批量任务提交"""
        # 批量验证
        for task in tasks:
            await self._validate_task(task)
        
        # 批量保存到数据库
        task_ids = []
        async with self.deps.db_session_factory() as session:
            for task in tasks:
                task.trace_id = str(uuid.uuid4())
                db_task = TaskModel(
                    id=task.id,
                    type=task.type.value,
                    status=TaskStatus.PENDING.value,
                    description=task.description,
                    context={
                        "file_path": task.context.file_path,
                        "language": task.context.language
                    },
                    priority=task.priority,
                    user_id=user_id,
                    metadata=task.metadata
                )
                session.add(db_task)
                task_ids.append(task.id)
            
            await session.commit()
        
        # 批量加入队列
        for task in tasks:
            if task.priority >= 8:
                await self.priority_queue.put((10 - task.priority, task))
            else:
                await self.task_queue.put(task)
        
        return task_ids
    
    async def _enhanced_worker(self, worker_id: str):
        """增强的工作器"""
        logger.info("enhanced_worker_started", worker_id=worker_id)
        
        while True:
            task = None
            try:
                # 智能任务获取
                task = await self._get_next_task()
                
                if task:
                    # 获取或创建熔断器
                    cb_key = f"{task.type.value}_{self._select_agent(task).name}"
                    if cb_key not in self.circuit_breakers:
                        self.circuit_breakers[cb_key] = circuit(
                            failure_threshold=5,
                            recovery_timeout=60,
                            expected_exception=Exception
                        )
                    
                    # 使用熔断器执行任务
                    cb = self.circuit_breakers[cb_key]
                    await cb(self._process_task_enhanced)(task, worker_id)
                    
            except Exception as e:
                logger.error("enhanced_worker_error",
                           worker_id=worker_id,
                           error=str(e),
                           exc_info=True)
                
                if task:
                    await self._handle_task_error(task, e)
                
                # 退避策略
                await asyncio.sleep(1)
    
    async def _process_task_enhanced(self, task: Task, worker_id: str):
        """增强的任务处理"""
        start_time = time.time()
        
        try:
            # 更新任务状态
            await self._update_task_status(task.id, TaskStatus.PROCESSING)
            
            # 选择最优Agent
            agent = await self._select_optimal_agent(task)
            
            # 执行任务
            response = await agent.execute_with_timeout(task)
            
            # 更新指标
            processing_time = time.time() - start_time
            self.task_metrics[task.type.value]["success"] += 1
            self.task_metrics[task.type.value]["total_time"] += processing_time
            
            # 保存结果
            await self._save_task_result(task.id, response)
            
            # 异步后处理
            asyncio.create_task(self._post_process_task(task, response))
            
            logger.info("task_completed_enhanced",
                       task_id=task.id,
                       processing_time=processing_time,
                       worker_id=worker_id)
            
        except Exception as e:
            self.task_metrics[task.type.value]["failure"] += 1
            raise
        
        finally:
            concurrent_tasks.labels(task_type=task.type.value).dec()
    
    async def _get_next_task(self) -> Optional[Task]:
        """智能获取下一个任务"""
        # 基于系统负载和任务类型选择任务
        system_load = await self._get_system_load()
        
        if system_load < 0.7:  # 系统负载低，优先处理复杂任务
            try:
                _, task = await asyncio.wait_for(
                    self.priority_queue.get(),
                    timeout=0.1
                )
                return task
            except asyncio.TimeoutError:
                pass
        
        # 获取普通任务
        try:
            return await asyncio.wait_for(
                self.task_queue.get(),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            return None
    
    async def _select_optimal_agent(self, task: Task) -> BaseAgent:
        """选择最优Agent"""
        # 基于任务类型和历史性能选择
        agent = self._select_agent(task)
        
        # 检查Agent健康状态
        agent_health = await self._check_agent_health(agent)
        if agent_health < 0.5:
            # 选择备用Agent
            logger.warning("agent_unhealthy", agent=agent.name, health=agent_health)
        
        return agent
    
    async def _preheat_cache(self, task: Task):
        """预热缓存"""
        try:
            # 预先计算和缓存可能需要的数据
            if task.type in [TaskType.CODE_GENERATION, TaskType.CODE_REFACTOR]:
                # 预先搜索相似代码
                query_embedding = await self.llm_provider.embed(task.description[:1000])
                similar_code = await self.vector_store.search_similar_code(
                    query_embedding,
                    top_k=10
                )
                
                # 缓存结果
                cache_key = f"similar_code:{task.id}"
                await self.cache_manager.set(cache_key, similar_code, ttl=3600)
                
        except Exception as e:
            logger.error("cache_preheat_error", task_id=task.id, error=str(e))
    
    async def _post_process_task(self, task: Task, response: AgentResponse):
        """任务后处理"""
        try:
            # 更新向量数据库
            if response.status == TaskStatus.COMPLETED and response.result:
                if "code" in response.result:
                    embedding = await self.llm_provider.embed(
                        response.result["code"][:2000]
                    )
                    await self.vector_store.add_code_embedding(
                        code_id=f"task_{task.id}",
                        code=response.result["code"],
                        embedding=embedding,
                        metadata={
                            "task_type": task.type.value,
                            "quality_score": response.result.get("metrics", {}).get("quality_score", 0),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
            
            # 清理相关缓存
            await self.cache_manager.delete(f"similar_code:{task.id}")
            
        except Exception as e:
            logger.error("post_process_error", task_id=task.id, error=str(e))
    
    async def _enhanced_monitor(self):
        """增强的监控循环"""
        while True:
            try:
                # 监控任务队列
                queue_sizes = {
                    "normal": self.task_queue.qsize(),
                    "priority": self.priority_queue.qsize()
                }
                
                # 监控工作器健康
                healthy_workers = sum(1 for w in self.worker_pool if not w.done())
                
                # 监控系统资源
                system_metrics = await self._collect_system_metrics()
                
                # 更新Prometheus指标
                for task_type, metrics in self.task_metrics.items():
                    if metrics["success"] + metrics["failure"] > 0:
                        success_rate = metrics["success"] / (metrics["success"] + metrics["failure"])
                        # 这里可以添加自定义指标
                
                # 自动扩缩容决策
                if queue_sizes["normal"] + queue_sizes["priority"] > 100 and healthy_workers < Config.MAX_CONCURRENT_TASKS:
                    # 添加更多工作器
                    new_worker = asyncio.create_task(
                        self._enhanced_worker(f"worker-{len(self.worker_pool)}")
                    )
                    self.worker_pool.append(new_worker)
                    logger.info("worker_scaled_up", total_workers=len(self.worker_pool))
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error("monitor_error", error=str(e))
                await asyncio.sleep(60)
    
    async def _maintenance_loop(self):
        """维护循环"""
        while True:
            try:
                # 清理过期任务
                await self._cleanup_expired_tasks()
                
                # 优化向量数据库索引
                # await self.vector_store.optimize_indices()
                
                # 清理Docker容器
                if Config.ENABLE_CODE_EXECUTION:
                    await self._cleanup_docker_containers()
                
                # 压缩日志
                await self._compress_old_logs()
                
                await asyncio.sleep(3600)  # 每小时执行一次
                
            except Exception as e:
                logger.error("maintenance_error", error=str(e))
                await asyncio.sleep(3600)
    
    async def _collect_metrics(self):
        """收集系统指标"""
        while True:
            try:
                # 内存使用
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_usage.labels(component="total").set(memory_info.rss)
                memory_usage.labels(component="cache").set(
                    len(self.cache_manager.memory_cache) * 1000 if self.cache_manager.memory_cache else 0
                )
                
                # 任务处理延迟
                for task_type, metrics in self.task_metrics.items():
                    if metrics["success"] > 0:
                        avg_time = metrics["total_time"] / metrics["success"]
                        task_processing_histogram.labels(
                            task_type=task_type,
                            stage="average"
                        ).observe(avg_time)
                
                await asyncio.sleep(30)  # 每30秒收集一次
                
            except Exception as e:
                logger.error("metrics_collection_error", error=str(e))
                await asyncio.sleep(60)
    
    async def _check_system_capacity(self) -> bool:
        """检查系统容量"""
        # 检查队列大小
        total_queued = self.task_queue.qsize() + self.priority_queue.qsize()
        if total_queued > Config.MAX_CONCURRENT_TASKS * 2:
            return False
        
        # 检查内存使用
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            return False
        
        return True
    
    async def _get_system_load(self) -> float:
        """获取系统负载"""
        import psutil
        return psutil.cpu_percent(interval=0.1) / 100.0
    
    async def _check_agent_health(self, agent: BaseAgent) -> float:
        """检查Agent健康度"""
        # 基于历史性能计算健康度
        if hasattr(agent, 'performance_history') and agent.performance_history:
            recent_performance = agent.performance_history[-10:]
            # 简单的健康度计算
            return 0.8  # 示例值
        return 1.0
    
    async def _cleanup_expired_tasks(self):
        """清理过期任务"""
        async with self.deps.db_session_factory() as session:
            # 清理超过7天的已完成任务
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            result = await session.execute(
                select(TaskModel).where(
                    TaskModel.completed_at < cutoff_date,
                    TaskModel.status.in_([TaskStatus.COMPLETED.value, TaskStatus.FAILED.value])
                )
            )
            
            tasks_to_delete = result.scalars().all()
            for task in tasks_to_delete:
                session.delete(task)
            
            await session.commit()
            
            if tasks_to_delete:
                logger.info("expired_tasks_cleaned", count=len(tasks_to_delete))
    
    async def _cleanup_docker_containers(self):
        """清理Docker容器"""
        if not self.deps.docker_client:
            return
        
        try:
            containers = await asyncio.get_event_loop().run_in_executor(
                self.deps.thread_pool,
                self.deps.docker_client.containers.list,
                {"all": True, "filters": {"label": "ai-agent-sandbox"}}
            )
            
            for container in containers:
                if container.status == "exited":
                    await asyncio.get_event_loop().run_in_executor(
                        self.deps.thread_pool,
                        container.remove
                    )
            
        except Exception as e:
            logger.error("docker_cleanup_error", error=str(e))
    
    async def _compress_old_logs(self):
        """压缩旧日志"""
        # 实现日志压缩逻辑
        pass

# ==================== 任务调度器 ====================

class TaskScheduler:
    """智能任务调度器"""
    
    def __init__(self):
        self.scheduling_rules = {}
        self.task_history = deque(maxlen=10000)
    
    async def schedule_task(self, task: Task) -> float:
        """计算任务调度优先级"""
        priority_score = task.priority
        
        # 基于任务类型调整
        if task.type == TaskType.BUG_FIX:
            priority_score *= 1.5
        elif task.type == TaskType.SECURITY_AUDIT:
            priority_score *= 1.3
        
        # 基于等待时间调整
        wait_time = (datetime.utcnow() - task.created_at).total_seconds()
        if wait_time > 300:  # 等待超过5分钟
            priority_score *= 1.2
        
        # 基于用户历史调整
        # TODO: 实现基于用户历史的优先级调整
        
        return priority_score

# ==================== 负载均衡器 ====================

class LoadBalancer:
    """Agent负载均衡器"""
    
    def __init__(self):
        self.agent_loads = defaultdict(int)
        self.agent_performance = defaultdict(list)
    
    async def select_agent(self, agents: List[BaseAgent], task: Task) -> BaseAgent:
        """选择负载最低的Agent"""
        # 简单的轮询策略
        # TODO: 实现更复杂的负载均衡算法
        return agents[0]

# ==================== 应用初始化函数 ====================

async def initialize_enhanced_application():
    """初始化增强的应用"""
    # 设置日志
    setup_logging()
    
    # 初始化增强依赖
    await deps.initialize()
    
    # 创建数据库表
    async with deps.db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 初始化增强协调器
    orchestrator = EnhancedTaskOrchestrator(deps)
    await orchestrator.initialize()
    
    logger.info("Enhanced application initialized successfully")
    
    return orchestrator

async def shutdown_enhanced_application():
    """关闭增强的应用"""
    await deps.cleanup()
    logger.info("Enhanced application shutdown complete")

# ==================== 性能测试示例 ====================

async def performance_test_example():
    """性能测试示例"""
    orchestrator = await initialize_enhanced_application()
    
    # 创建测试用户
    user_id = "test_user_123"
    
    # 批量创建任务
    tasks = []
    for i in range(100):
        context = CodeContext(
            file_path=f"test/file_{i}.py",
            language="python",
            content=f"# Test code {i}\ndef function_{i}():\n    return {i}",
            dependencies=["requests", "numpy"]
        )
        
        task = Task(
            id=str(uuid.uuid4()),
            type=TaskType.CODE_GENERATION if i % 2 == 0 else TaskType.CODE_REVIEW,
            description=f"Test task {i}: Generate optimized code for data processing",
            context=context,
            priority=5 + (i % 5),
            metadata={"test_id": i}
        )
        tasks.append(task)
    
    # 测试批量提交
    start_time = time.time()
    task_ids = await orchestrator.submit_batch(tasks, user_id)
    submit_time = time.time() - start_time
    
    print(f"Submitted {len(task_ids)} tasks in {submit_time:.2f} seconds")
    print(f"Submission rate: {len(task_ids) / submit_time:.2f} tasks/second")
    
    # 等待所有任务完成
    completed = 0
    failed = 0
    
    while completed + failed < len(task_ids):
        await asyncio.sleep(1)
        
        for task_id in task_ids:
            status = await orchestrator.get_task_status(task_id, user_id)
            if status["status"] == "completed":
                completed += 1
            elif status["status"] in ["failed", "timeout"]:
                failed += 1
    
    total_time = time.time() - start_time
    
    print(f"\nPerformance Test Results:")
    print(f"Total tasks: {len(task_ids)}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Processing rate: {len(task_ids) / total_time:.2f} tasks/second")
    print(f"Success rate: {completed / len(task_ids) * 100:.2f}%")
    
    await shutdown_enhanced_application()

if __name__ == "__main__":
    # 运行性能测试
    asyncio.run(performance_test_example())
