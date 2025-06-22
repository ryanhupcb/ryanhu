# production_agent_system.py
"""
Production-Ready Multi-Agent System
A comprehensive, scalable, and secure agent orchestration platform
"""

import asyncio
import os
import sys
import json
import uuid
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from contextlib import asynccontextmanager
from functools import wraps
import pickle
import base64

# Third-party imports
import aiohttp
import redis.asyncio as redis
import numpy as np
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Text, ForeignKey, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool
from celery import Celery
from kombu import Queue
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc import trace_exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from jose import JWTError, jwt
from passlib.context import CryptContext
import bleach
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# ==================== Configuration Management ====================

class Settings(BaseModel):
    """Application settings with validation"""
    # API Configuration
    api_port: int = Field(default=8000, env="API_PORT")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    database_max_overflow: int = Field(default=40, env="DB_MAX_OVERFLOW")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    default_llm_provider: str = Field(default="openai", env="DEFAULT_LLM_PROVIDER")
    llm_timeout: int = Field(default=60, env="LLM_TIMEOUT")
    
    # Security Configuration
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=30, env="JWT_EXPIRATION_MINUTES")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    otlp_endpoint: str = Field(default="http://localhost:4317", env="OTLP_ENDPOINT")
    
    # Rate Limiting Configuration
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    # Performance Configuration
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    batch_size: int = Field(default=50, env="BATCH_SIZE")
    batch_timeout: float = Field(default=1.0, env="BATCH_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Load settings
settings = Settings()

# ==================== Database Models ====================

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)
    
    user = relationship("User", back_populates="api_keys")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    tool_calls = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(20), default="pending")
    priority = Column(Integer, default=5)
    result = Column(JSON)
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    execution_time = Column(Float)
    
    user = relationship("User", back_populates="tasks")
    subtasks = relationship("SubTask", back_populates="task", cascade="all, delete-orphan")

class SubTask(Base):
    __tablename__ = "subtasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(20), default="pending")
    result = Column(JSON)
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    task = relationship("Task", back_populates="subtasks")

class ToolUsage(Base):
    __tablename__ = "tool_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    tool_name = Column(String(100), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    task_id = Column(String(36), ForeignKey("tasks.id"))
    parameters = Column(JSON)
    result = Column(JSON)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    execution_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# ==================== Database Manager ====================

class DatabaseManager:
    """Manages database connections and operations"""
    def __init__(self):
        self.engine = create_engine(
            settings.database_url,
            poolclass=QueuePool,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        
    @asynccontextmanager
    async def get_db(self):
        """Async context manager for database sessions"""
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
            
    def get_sync_db(self) -> Session:
        """Get synchronous database session"""
        return self.SessionLocal()

# ==================== Cache Manager ====================

class CacheManager:
    """Redis-based caching with TTL and invalidation"""
    def __init__(self):
        self.redis_pool = None
        self.default_ttl = settings.cache_ttl
        
    async def initialize(self):
        """Initialize Redis connection pool"""
        self.redis_pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=False
        )
        
    async def get_client(self) -> redis.Redis:
        """Get Redis client"""
        if not self.redis_pool:
            await self.initialize()
        return redis.Redis(connection_pool=self.redis_pool)
        
    def cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key"""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        if kwargs:
            key_data += f":{':'.join(f'{k}={v}' for k, v in sorted(kwargs.items()))}"
        return hashlib.sha256(key_data.encode()).hexdigest()
        
    def cache_result(self, prefix: str, ttl: int = None):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                cache_key = self.cache_key(prefix, *args, **kwargs)
                client = await self.get_client()
                
                # Try to get from cache
                cached = await client.get(cache_key)
                if cached:
                    return pickle.loads(cached)
                    
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                ttl_seconds = ttl or self.default_ttl
                await client.setex(cache_key, ttl_seconds, pickle.dumps(result))
                
                return result
            return wrapper
        return decorator
        
    async def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        client = await self.get_client()
        async for key in client.scan_iter(match=pattern):
            await client.delete(key)

# ==================== Metrics Collector ====================

class MetricsCollector:
    """Prometheus metrics collection"""
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'agent_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        self.request_duration = Histogram(
            'agent_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
        )
        
        # Agent metrics
        self.active_agents = Gauge(
            'agent_active_count',
            'Number of active agents',
            ['agent_type']
        )
        self.agent_tasks = Counter(
            'agent_tasks_total',
            'Total agent tasks',
            ['agent_type', 'status']
        )
        
        # LLM metrics
        self.llm_requests = Counter(
            'llm_requests_total',
            'Total LLM requests',
            ['provider', 'model', 'status']
        )
        self.llm_tokens = Counter(
            'llm_tokens_total',
            'Total LLM tokens used',
            ['provider', 'model', 'token_type']
        )
        self.llm_latency = Histogram(
            'llm_request_duration_seconds',
            'LLM request duration',
            ['provider', 'model']
        )
        
        # Tool metrics
        self.tool_usage = Counter(
            'tool_usage_total',
            'Total tool usage',
            ['tool_name', 'status']
        )
        self.tool_duration = Histogram(
            'tool_execution_duration_seconds',
            'Tool execution duration',
            ['tool_name']
        )
        
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
    def record_llm_request(self, provider: str, model: str, tokens: int, duration: float, success: bool):
        """Record LLM request metrics"""
        status = "success" if success else "failure"
        self.llm_requests.labels(provider=provider, model=model, status=status).inc()
        self.llm_tokens.labels(provider=provider, model=model, token_type="total").inc(tokens)
        self.llm_latency.labels(provider=provider, model=model).observe(duration)
        
    def record_tool_usage(self, tool_name: str, duration: float, success: bool):
        """Record tool usage metrics"""
        status = "success" if success else "failure"
        self.tool_usage.labels(tool_name=tool_name, status=status).inc()
        self.tool_duration.labels(tool_name=tool_name).observe(duration)

# ==================== Authentication Manager ====================

class AuthenticationManager:
    """JWT-based authentication"""
    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_expiration_minutes
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
        
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
        
    def create_access_token(self, data: dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
        
    def decode_token(self, token: str) -> dict:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise ValueError("Invalid authentication token")
            
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return hashlib.sha256(os.urandom(32)).hexdigest()

# ==================== Input Validation ====================

class TaskRequest(BaseModel):
    """Validated task request"""
    task: str = Field(..., min_length=1, max_length=10000)
    context: Optional[dict] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)
    timeout: int = Field(default=300, ge=10, le=3600)
    
    @validator('task')
    def sanitize_task(cls, v):
        """Sanitize task input"""
        return bleach.clean(v, tags=[], strip=True)
        
    @validator('context')
    def validate_context(cls, v):
        """Validate context size"""
        if sys.getsizeof(v) > 1024 * 1024:  # 1MB limit
            raise ValueError("Context too large")
        return v

class ChatRequest(BaseModel):
    """Validated chat request"""
    message: str = Field(..., min_length=1, max_length=5000)
    conversation_id: Optional[str] = None
    
    @validator('message')
    def sanitize_message(cls, v):
        """Sanitize message input"""
        return bleach.clean(v, tags=[], strip=True)

# ==================== Circuit Breaker ====================

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")
                    
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.expected_exception as e:
            await self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
        
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
            
    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(
                    "Circuit breaker opened",
                    failure_count=self.failure_count,
                    threshold=self.failure_threshold
                )

# ==================== Batch Processor ====================

class BatchProcessor:
    """Batch processing for efficiency"""
    def __init__(self, batch_size: int = None, batch_timeout: float = None):
        self.batch_size = batch_size or settings.batch_size
        self.batch_timeout = batch_timeout or settings.batch_timeout
        self.pending_items = []
        self.pending_futures = []
        self.lock = asyncio.Lock()
        self.timer_task = None
        
    async def add_item(self, item: Any) -> Any:
        """Add item to batch"""
        future = asyncio.Future()
        
        async with self.lock:
            self.pending_items.append(item)
            self.pending_futures.append(future)
            
            if len(self.pending_items) >= self.batch_size:
                await self._process_batch()
            elif not self.timer_task:
                self.timer_task = asyncio.create_task(self._batch_timer())
                
        return await future
        
    async def _batch_timer(self):
        """Timer for batch timeout"""
        await asyncio.sleep(self.batch_timeout)
        async with self.lock:
            if self.pending_items:
                await self._process_batch()
            self.timer_task = None
            
    async def _process_batch(self):
        """Process accumulated batch"""
        if not self.pending_items:
            return
            
        items = self.pending_items[:]
        futures = self.pending_futures[:]
        self.pending_items.clear()
        self.pending_futures.clear()
        
        if self.timer_task:
            self.timer_task.cancel()
            self.timer_task = None
            
        # Process batch
        try:
            results = await self._batch_execute(items)
            for future, result in zip(futures, results):
                future.set_result(result)
        except Exception as e:
            for future in futures:
                future.set_exception(e)
                
    async def _batch_execute(self, items: List[Any]) -> List[Any]:
        """Execute batch - to be overridden by subclasses"""
        raise NotImplementedError

# ==================== LLM Service with Retry ====================

class LLMService:
    """LLM service with retry and circuit breaker"""
    def __init__(self, provider: str, api_key: str, metrics: MetricsCollector):
        self.provider = provider
        self.api_key = api_key
        self.metrics = metrics
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate LLM response with retry logic"""
        start_time = time.time()
        
        async def _generate():
            # Implementation depends on provider
            # This is a placeholder
            return {
                'content': 'Generated response',
                'tokens': 100,
                'model': 'gpt-4'
            }
            
        try:
            result = await self.circuit_breaker.call(_generate)
            duration = time.time() - start_time
            
            self.metrics.record_llm_request(
                self.provider,
                result.get('model', 'unknown'),
                result.get('tokens', 0),
                duration,
                True
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_llm_request(
                self.provider,
                'unknown',
                0,
                duration,
                False
            )
            raise

# ==================== Health Check Manager ====================

class HealthCheckManager:
    """Manages health checks for the system"""
    def __init__(self):
        self.checks = {}
        
    def register_check(self, name: str, check_func: Callable):
        """Register a health check"""
        self.checks[name] = check_func
        
    async def check_health(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for name, check in self.checks.items():
            try:
                result = await check()
                results[name] = result
                if not result.get('healthy', False):
                    overall_healthy = False
            except Exception as e:
                results[name] = {
                    'healthy': False,
                    'error': str(e)
                }
                overall_healthy = False
                
        return {
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'checks': results,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def database_check(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            db_manager = DatabaseManager()
            async with db_manager.get_db() as db:
                db.execute("SELECT 1")
            return {'healthy': True}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
            
    async def redis_check(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            cache = CacheManager()
            client = await cache.get_client()
            await client.ping()
            return {'healthy': True}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}

# ==================== Task Queue Configuration ====================

# Celery configuration
celery_app = Celery('agent_system')
celery_app.conf.update(
    broker_url=settings.redis_url,
    result_backend=settings.redis_url,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'agent.tasks.process_task': {'queue': 'high_priority'},
        'agent.tasks.analyze_data': {'queue': 'analysis'},
        'agent.tasks.generate_report': {'queue': 'low_priority'},
    },
    task_queues=(
        Queue('high_priority', priority=10),
        Queue('analysis', priority=5),
        Queue('low_priority', priority=1),
    ),
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# ==================== Core Agent Service ====================

class ProductionAgentService:
    """Core production agent service"""
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        self.metrics = MetricsCollector()
        self.auth_manager = AuthenticationManager()
        self.health_manager = HealthCheckManager()
        
        # Register health checks
        self.health_manager.register_check('database', self.health_manager.database_check)
        self.health_manager.register_check('redis', self.health_manager.redis_check)
        
        # Initialize components
        self._initialized = False
        
    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
            
        # Initialize cache
        await self.cache_manager.initialize()
        
        # Initialize tracing if enabled
        if settings.enable_tracing:
            self._setup_tracing()
            
        self._initialized = True
        logger.info("Production Agent Service initialized")
        
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        # Configure OTLP exporter
        otlp_exporter = trace_exporter.OTLPSpanExporter(
            endpoint=settings.otlp_endpoint,
            insecure=True
        )
        
        # Add batch processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        
    @CacheManager.cache_result("user", ttl=3600)
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID with caching"""
        async with self.db_manager.get_db() as db:
            return db.query(User).filter(User.id == user_id).first()
            
    async def create_user(self, username: str, email: str, password: str) -> User:
        """Create new user"""
        async with self.db_manager.get_db() as db:
            hashed_password = self.auth_manager.get_password_hash(password)
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Invalidate user cache
            await self.cache_manager.invalidate(f"user:{user.id}:*")
            
            return user
            
    async def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token"""
        async with self.db_manager.get_db() as db:
            user = db.query(User).filter(User.username == username).first()
            
            if not user or not self.auth_manager.verify_password(password, user.hashed_password):
                return None
                
            token_data = {"sub": str(user.id), "username": user.username}
            return self.auth_manager.create_access_token(token_data)
            
    async def process_chat_request(self, request: ChatRequest, user_id: int) -> Dict[str, Any]:
        """Process chat request"""
        start_time = time.time()
        
        try:
            # Create or get conversation
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            async with self.db_manager.get_db() as db:
                conversation = db.query(Conversation).filter(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id
                ).first()
                
                if not conversation:
                    conversation = Conversation(
                        id=conversation_id,
                        user_id=user_id,
                        title=request.message[:50]
                    )
                    db.add(conversation)
                    
                # Add user message
                message = Message(
                    conversation_id=conversation_id,
                    role="user",
                    content=request.message
                )
                db.add(message)
                db.commit()
                
            # Process with LLM (placeholder)
            response_content = f"Response to: {request.message}"
            
            # Add assistant message
            async with self.db_manager.get_db() as db:
                assistant_message = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response_content
                )
                db.add(assistant_message)
                db.commit()
                
            duration = time.time() - start_time
            self.metrics.record_request("POST", "/chat", 200, duration)
            
            return {
                'conversation_id': conversation_id,
                'response': response_content,
                'duration': duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_request("POST", "/chat", 500, duration)
            logger.error("Chat request failed", error=str(e), user_id=user_id)
            raise
            
    async def process_task_request(self, request: TaskRequest, user_id: int) -> Dict[str, Any]:
        """Process task request"""
        # Create task record
        task = Task(
            id=str(uuid.uuid4()),
            user_id=user_id,
            description=request.task,
            priority=request.priority
        )
        
        async with self.db_manager.get_db() as db:
            db.add(task)
            db.commit()
            
        # Queue task for processing
        from .tasks import process_task
        result = process_task.apply_async(
            args=[task.id, request.dict()],
            priority=request.priority,
            expires=request.timeout
        )
        
        self.metrics.agent_tasks.labels(agent_type='task_processor', status='queued').inc()
        
        return {
            'task_id': task.id,
            'status': 'queued',
            'celery_task_id': result.id
        }
        
    async def get_task_status(self, task_id: str, user_id: int) -> Dict[str, Any]:
        """Get task status"""
        async with self.db_manager.get_db() as db:
            task = db.query(Task).filter(
                Task.id == task_id,
                Task.user_id == user_id
            ).first()
            
            if not task:
                raise ValueError("Task not found")
                
            subtasks = db.query(SubTask).filter(SubTask.task_id == task_id).all()
            
            return {
                'task_id': task.id,
                'description': task.description,
                'status': task.status,
                'priority': task.priority,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'execution_time': task.execution_time,
                'result': task.result,
                'error': task.error,
                'subtasks': [
                    {
                        'description': st.description,
                        'status': st.status,
                        'completed_at': st.completed_at.isoformat() if st.completed_at else None
                    }
                    for st in subtasks
                ]
            }

# ==================== Example Usage ====================

async def example_usage():
    """Example usage of the production agent system"""
    
    # Initialize service
    service = ProductionAgentService()
    await service.initialize()
    
    # Create user
    user = await service.create_user(
        username="testuser",
        email="test@example.com",
        password="securepassword123"
    )
    print(f"Created user: {user.username}")
    
    # Authenticate
    token = await service.authenticate_user("testuser", "securepassword123")
    print(f"Authentication token: {token[:20]}...")
    
    # Process chat request
    chat_request = ChatRequest(message="Hello, how can you help me?")
    chat_response = await service.process_chat_request(chat_request, user.id)
    print(f"Chat response: {chat_response}")
    
    # Process task request
    task_request = TaskRequest(
        task="Analyze the latest sales data and create a summary report",
        priority=8
    )
    task_response = await service.process_task_request(task_request, user.id)
    print(f"Task queued: {task_response}")
    
    # Check health
    health = await service.health_manager.check_health()
    print(f"System health: {health['status']}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
