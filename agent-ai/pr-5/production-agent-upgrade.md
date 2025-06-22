# 生产级Agent协作系统升级方案

## 一、架构改进

### 1.1 微服务架构重构
```yaml
# 新的服务架构
services:
  - api-gateway      # API网关服务
  - agent-core       # 核心Agent服务
  - tool-executor    # 工具执行服务
  - task-scheduler   # 任务调度服务
  - memory-service   # 内存管理服务
  - monitoring       # 监控服务
```

### 1.2 添加消息队列解耦
```python
# 使用Celery进行异步任务处理
from celery import Celery
from kombu import Queue

celery_app = Celery('agent_system')
celery_app.conf.update(
    broker_url='redis://redis:6379/0',
    result_backend='redis://redis:6379/1',
    task_routes={
        'agent.tasks.research': {'queue': 'research'},
        'agent.tasks.code': {'queue': 'code'},
        'agent.tasks.analysis': {'queue': 'analysis'},
    },
    task_queues=(
        Queue('research', priority=10),
        Queue('code', priority=5),
        Queue('analysis', priority=1),
    ),
)
```

## 二、安全性增强

### 2.1 API认证和授权
```python
# JWT认证实现
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

class AuthenticationManager:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
        
    async def get_current_user(self, token: str = Depends(oauth2_scheme)):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception
        return username
```

### 2.2 输入验证和清理
```python
from pydantic import BaseModel, validator, Field
from typing import Optional, List
import bleach

class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=10000)
    context: Optional[dict] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)
    timeout: int = Field(default=300, ge=10, le=3600)
    
    @validator('task')
    def sanitize_task(cls, v):
        # 清理潜在的恶意输入
        return bleach.clean(v, tags=[], strip=True)
        
    @validator('context')
    def validate_context(cls, v):
        # 限制context大小
        if sys.getsizeof(v) > 1024 * 1024:  # 1MB限制
            raise ValueError("Context too large")
        return v
```

### 2.3 沙箱执行环境
```python
import docker
from typing import Dict, Any

class SecureCodeExecutor:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.execution_image = "python:3.11-slim-sandbox"
        
    async def execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """在Docker容器中安全执行代码"""
        try:
            # 创建临时容器
            container = self.docker_client.containers.run(
                self.execution_image,
                command=f"python -c '{code}'",
                detach=True,
                mem_limit="512m",
                cpu_quota=50000,  # 50% CPU
                network_disabled=True,
                read_only=True,
                user="nobody",
                security_opt=["no-new-privileges"],
                remove=True
            )
            
            # 等待执行完成
            result = container.wait(timeout=timeout)
            logs = container.logs(stdout=True, stderr=True).decode()
            
            return {
                'success': result['StatusCode'] == 0,
                'output': logs,
                'exit_code': result['StatusCode']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
```

## 三、性能优化

### 3.1 数据库连接池
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(
            os.getenv("DATABASE_URL"),
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    @contextmanager
    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
```

### 3.2 缓存策略
```python
import redis
from functools import wraps
import pickle
import hashlib

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='redis',
            port=6379,
            decode_responses=False,
            connection_pool_kwargs={
                'max_connections': 50,
                'socket_keepalive': True,
                'socket_keepalive_options': {}
            }
        )
        
    def cache_result(self, ttl: int = 3600):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # 尝试从缓存获取
                cached = self.redis_client.get(cache_key)
                if cached:
                    return pickle.loads(cached)
                    
                # 执行函数
                result = await func(*args, **kwargs)
                
                # 存入缓存
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    pickle.dumps(result)
                )
                
                return result
            return wrapper
        return decorator
        
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.sha256(key_data.encode()).hexdigest()
```

### 3.3 批处理优化
```python
from typing import List, Dict, Any
import asyncio

class BatchProcessor:
    def __init__(self, batch_size: int = 100, batch_timeout: float = 1.0):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_items = []
        self.pending_futures = []
        self.lock = asyncio.Lock()
        self.timer_task = None
        
    async def add_item(self, item: Any) -> Any:
        future = asyncio.Future()
        
        async with self.lock:
            self.pending_items.append(item)
            self.pending_futures.append(future)
            
            if len(self.pending_items) >= self.batch_size:
                await self._process_batch()
            elif not self.timer_task:
                self.timer_task = asyncio.create_task(self._batch_timer())
                
        return await future
        
    async def _process_batch(self):
        if not self.pending_items:
            return
            
        items = self.pending_items[:]
        futures = self.pending_futures[:]
        self.pending_items.clear()
        self.pending_futures.clear()
        
        # 批量处理
        try:
            results = await self._batch_execute(items)
            for future, result in zip(futures, results):
                future.set_result(result)
        except Exception as e:
            for future in futures:
                future.set_exception(e)
                
    async def _batch_execute(self, items: List[Any]) -> List[Any]:
        # 实现批量执行逻辑
        pass
```

## 四、监控和可观测性

### 4.1 分布式追踪
```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc import trace_exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.aiohttp import AioHttpClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

def setup_tracing():
    # 设置追踪提供者
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    
    # 配置OTLP导出器
    otlp_exporter = trace_exporter.OTLPSpanExporter(
        endpoint="http://jaeger:4317",
        insecure=True
    )
    
    # 添加批处理器
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # 自动仪表化
    AioHttpClientInstrumentor().instrument()
    RedisInstrumentor().instrument()
    
    return trace.get_tracer("agent_system")
```

### 4.2 指标收集
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

class MetricsCollector:
    def __init__(self):
        # 计数器
        self.request_count = Counter(
            'agent_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        # 直方图
        self.request_duration = Histogram(
            'agent_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
        )
        
        # 仪表
        self.active_agents = Gauge(
            'agent_active_count',
            'Number of active agents',
            ['agent_type']
        )
        
        self.llm_token_usage = Counter(
            'llm_tokens_total',
            'Total LLM tokens used',
            ['provider', 'model']
        )
        
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
    def update_active_agents(self, agent_type: str, count: int):
        self.active_agents.labels(agent_type=agent_type).set(count)
```

### 4.3 结构化日志
```python
import structlog
from pythonjsonlogger import jsonlogger

def setup_logging():
    # 配置结构化日志
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
    
    # 配置Python标准日志
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    logHandler.setFormatter(formatter)
    logging.root.addHandler(logHandler)
    logging.root.setLevel(logging.INFO)
    
    return structlog.get_logger()
```

## 五、错误处理和容错

### 5.1 断路器模式
```python
from typing import Callable, Any
import asyncio
from datetime import datetime, timedelta

class CircuitBreaker:
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
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
        
    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
        
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
```

### 5.2 重试机制
```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_retry,
    after_retry
)

class RetryManager:
    @staticmethod
    def create_retry_decorator(
        max_attempts: int = 3,
        min_wait: int = 1,
        max_wait: int = 10,
        exceptions: tuple = (Exception,)
    ):
        def log_before_retry(retry_state):
            logger.warning(
                "Retrying function",
                function=retry_state.fn.__name__,
                attempt=retry_state.attempt_number
            )
            
        def log_after_retry(retry_state):
            logger.info(
                "Retry successful",
                function=retry_state.fn.__name__,
                attempts=retry_state.attempt_number
            )
            
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before=log_before_retry,
            after=log_after_retry
        )
```

## 六、部署和运维

### 6.1 Kubernetes部署
```yaml
# agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-system
  labels:
    app: agent-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-system
  template:
    metadata:
      labels:
        app: agent-system
    spec:
      containers:
      - name: agent-system
        image: agent-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: agent-system-service
spec:
  selector:
    app: agent-system
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-system-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-system
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 6.2 健康检查
```python
from fastapi import FastAPI, status
from typing import Dict, Any
import asyncio

class HealthChecker:
    def __init__(self, app: FastAPI):
        self.app = app
        self.checks = {}
        
        # 注册健康检查端点
        app.get("/health", status_code=status.HTTP_200_OK)(self.health_check)
        app.get("/ready", status_code=status.HTTP_200_OK)(self.readiness_check)
        app.get("/health/detailed")(self.detailed_health_check)
        
    def register_check(self, name: str, check_func: Callable):
        """注册健康检查函数"""
        self.checks[name] = check_func
        
    async def health_check(self) -> Dict[str, str]:
        """基本健康检查"""
        return {"status": "healthy"}
        
    async def readiness_check(self) -> Dict[str, Any]:
        """就绪检查"""
        results = await asyncio.gather(
            *[check() for check in self.checks.values()],
            return_exceptions=True
        )
        
        all_healthy = all(
            result.get("healthy", False) if isinstance(result, dict) else False
            for result in results
        )
        
        if not all_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
            
        return {"status": "ready"}
        
    async def detailed_health_check(self) -> Dict[str, Any]:
        """详细健康检查"""
        results = {}
        
        for name, check in self.checks.items():
            try:
                results[name] = await check()
            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "error": str(e)
                }
                
        overall_health = all(
            result.get("healthy", False)
            for result in results.values()
        )
        
        return {
            "status": "healthy" if overall_health else "unhealthy",
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }
```

## 七、测试策略

### 7.1 单元测试
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

class TestAgentSystem:
    @pytest.fixture
    async def agent_system(self):
        """创建测试用的Agent系统实例"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'ANTHROPIC_API_KEY': 'test-key'
        }):
            system = CompleteAgentSystem({"enable_all_frameworks": False})
            yield system
            # 清理
            await system.cleanup()
            
    @pytest.mark.asyncio
    async def test_chat_functionality(self, agent_system):
        """测试聊天功能"""
        # Mock LLM响应
        agent_system.llm_orchestrator.generate = AsyncMock(
            return_value={
                'content': 'Test response',
                'tool_calls': None,
                'usage': {'total_tokens': 100}
            }
        )
        
        response = await agent_system.chat("Hello")
        
        assert response['response'] == 'Test response'
        assert 'conversation_id' in response
        
    @pytest.mark.asyncio
    async def test_task_execution_with_timeout(self, agent_system):
        """测试任务执行超时"""
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                agent_system.execute_task("Infinite task"),
                timeout=1.0
            )
```

### 7.2 集成测试
```python
import pytest
from testcontainers.redis import RedisContainer
from testcontainers.rabbitmq import RabbitMQContainer

@pytest.fixture(scope="session")
def redis_container():
    """启动Redis测试容器"""
    with RedisContainer() as redis:
        yield redis
        
@pytest.fixture(scope="session")
def rabbitmq_container():
    """启动RabbitMQ测试容器"""
    with RabbitMQContainer() as rabbitmq:
        yield rabbitmq
        
class TestIntegration:
    @pytest.mark.integration
    async def test_multi_agent_collaboration(
        self,
        redis_container,
        rabbitmq_container
    ):
        """测试多Agent协作"""
        # 配置测试环境
        config = {
            'redis': {
                'host': redis_container.get_container_host_ip(),
                'port': redis_container.get_exposed_port(6379)
            },
            'rabbitmq': {
                'url': rabbitmq_container.get_connection_url()
            }
        }
        
        system = ExtendedCompleteAgentSystem(config)
        await system.enable_redis_communication(config['redis'])
        await system.enable_rabbitmq_communication(config['rabbitmq']['url'])
        
        # 创建测试场景
        scenario = CollaborationScenario(
            "test_scenario",
            "Test multi-agent collaboration"
        )
        
        # 执行并验证
        result = await system.coordinator.execute_scenario(scenario)
        assert result['success']
```

### 7.3 负载测试
```python
# locustfile.py
from locust import HttpUser, task, between
import json

class AgentSystemUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """获取认证token"""
        response = self.client.post(
            "/auth/token",
            json={"username": "test", "password": "test"}
        )
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
        
    @task(3)
    def chat(self):
        """测试聊天端点"""
        self.client.post(
            "/chat",
            json={"message": "What is the weather today?"},
            headers=self.headers
        )
        
    @task(1)
    def execute_task(self):
        """测试任务执行"""
        self.client.post(
            "/task",
            json={
                "task": "Generate a simple Python function",
                "context": {"language": "python"}
            },
            headers=self.headers
        )
        
    @task(2)
    def get_status(self):
        """测试状态查询"""
        self.client.get("/status", headers=self.headers)
```

## 八、配置管理

### 8.1 环境配置
```python
from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any

class Settings(BaseSettings):
    # API配置
    api_port: int = Field(default=8000, env="API_PORT")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # 数据库配置
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    
    # Redis配置
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    
    # LLM配置
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    default_llm_provider: str = Field(default="openai", env="DEFAULT_LLM_PROVIDER")
    
    # 安全配置
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=30, env="JWT_EXPIRATION_MINUTES")
    
    # 监控配置
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    otlp_endpoint: str = Field(default="http://localhost:4317", env="OTLP_ENDPOINT")
    
    # 限流配置
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
settings = Settings()
```

## 九、数据持久化

### 9.1 数据模型
```python
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    conversations = relationship("Conversation", back_populates="user")
    tasks = relationship("Task", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    priority = Column(Integer, default=5)
    result = Column(JSON)
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    user = relationship("User", back_populates="tasks")
    subtasks = relationship("SubTask", back_populates="task")

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
```

## 十、备份和恢复

### 10.1 自动备份
```python
import schedule
import boto3
from datetime import datetime
import subprocess
import os

class BackupManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.backup_bucket = os.getenv('BACKUP_BUCKET', 'agent-system-backups')
        
    async def backup_database(self):
        """备份数据库"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"db_backup_{timestamp}.sql"
        
        try:
            # 执行数据库备份
            subprocess.run([
                'pg_dump',
                os.getenv('DATABASE_URL'),
                '-f', backup_file
            ], check=True)
            
            # 上传到S3
            self.s3_client.upload_file(
                backup_file,
                self.backup_bucket,
                f"database/{backup_file}"
            )
            
            # 清理本地文件
            os.remove(backup_file)
            
            logger.info(f"Database backup completed: {backup_file}")
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            
    async def backup_vector_store(self):
        """备份向量存储"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 导出向量数据
        vectors_data = await self.vector_db.export_all()
        
        # 保存并上传
        backup_file = f"vectors_{timestamp}.pkl"
        with open(backup_file, 'wb') as f:
            pickle.dump(vectors_data, f)
            
        self.s3_client.upload_file(
            backup_file,
            self.backup_bucket,
            f"vectors/{backup_file}"
        )
        
        os.remove(backup_file)
        
    def schedule_backups(self):
        """调度备份任务"""
        schedule.every(6).hours.do(lambda: asyncio.create_task(self.backup_database()))
        schedule.every(12).hours.do(lambda: asyncio.create_task(self.backup_vector_store()))
```

## 十一、性能基准测试

### 11.1 基准测试套件
```python
import asyncio
import time
from typing import List, Dict, Any
import statistics

class BenchmarkSuite:
    def __init__(self, system: CompleteAgentSystem):
        self.system = system
        self.results = {}
        
    async def run_all_benchmarks(self):
        """运行所有基准测试"""
        benchmarks = [
            self.benchmark_chat_latency,
            self.benchmark_task_throughput,
            self.benchmark_tool_execution,
            self.benchmark_memory_operations,
            self.benchmark_concurrent_users
        ]
        
        for benchmark in benchmarks:
            await benchmark()
            
        return self.results
        
    async def benchmark_chat_latency(self, iterations: int = 100):
        """测试聊天响应延迟"""
        latencies = []
        
        for _ in range(iterations):
            start_time = time.time()
            await self.system.chat("Hello, how are you?")
            latency = time.time() - start_time
            latencies.append(latency)
            
        self.results['chat_latency'] = {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'p95': statistics.quantiles(latencies, n=20)[18],
            'p99': statistics.quantiles(latencies, n=100)[98]
        }
        
    async def benchmark_concurrent_users(self, user_count: int = 50):
        """测试并发用户处理能力"""
        async def simulate_user(user_id: int):
            tasks = []
            for _ in range(10):
                task = asyncio.create_task(
                    self.system.chat(f"User {user_id} message")
                )
                tasks.append(task)
            return await asyncio.gather(*tasks)
            
        start_time = time.time()
        await asyncio.gather(*[
            simulate_user(i) for i in range(user_count)
        ])
        total_time = time.time() - start_time
        
        self.results['concurrent_users'] = {
            'user_count': user_count,
            'total_time': total_time,
            'avg_time_per_user': total_time / user_count
        }
```

## 十二、生产环境配置文件

### 12.1 production.json
```json
{
  "system": {
    "enable_all_frameworks": true,
    "safety_threshold": 0.98,
    "openai_model": "gpt-4-turbo-preview",
    "anthropic_model": "claude-3-opus-20240229",
    "max_concurrent_tasks": 100,
    "task_timeout": 600,
    "memory_limit_mb": 8192,
    "enable_profiling": false
  },
  
  "api_server": {
    "enabled": true,
    "port": 8000,
    "host": "0.0.0.0",
    "workers": 4,
    "cors_origins": ["*"],
    "rate_limit": {
      "enabled": true,
      "requests_per_minute": 100,
      "requests_per_hour": 5000,
      "burst_size": 20
    },
    "authentication": {
      "enabled": true,
      "type": "jwt",
      "jwt_secret": "${JWT_SECRET_KEY}",
      "token_expiration": 3600
    },
    "ssl": {
      "enabled": true,
      "cert_file": "/etc/ssl/certs/agent.crt",
      "key_file": "/etc/ssl/private/agent.key"
    }
  },
  
  "database": {
    "url": "${DATABASE_URL}",
    "pool_size": 50,
    "max_overflow": 100,
    "pool_timeout": 30,
    "pool_recycle": 1800,
    "echo": false
  },
  
  "redis": {
    "url": "${REDIS_URL}",
    "max_connections": 100,
    "decode_responses": false,
    "health_check_interval": 30,
    "socket_keepalive": true,
    "socket_keepalive_options": {
      "TCP_KEEPINTVL": 30,
      "TCP_KEEPCNT": 3,
      "TCP_KEEPIDLE": 120
    }
  },
  
  "monitoring": {
    "metrics": {
      "enabled": true,
      "export_interval": 15,
      "prometheus_port": 9091,
      "include_process_metrics": true
    },
    "logging": {
      "level": "INFO",
      "format": "json",
      "output": "stdout",
      "error_output": "stderr",
      "include_trace_id": true
    },
    "tracing": {
      "enabled": true,
      "sample_rate": 0.1,
      "otlp_endpoint": "${OTLP_ENDPOINT}",
      "service_name": "agent-system",
      "trace_id_header": "X-Trace-ID"
    },
    "alerting": {
      "enabled": true,
      "webhook_url": "${ALERT_WEBHOOK_URL}",
      "critical_error_threshold": 10,
      "alert_interval": 300
    }
  },
  
  "security": {
    "input_validation": {
      "max_input_length": 50000,
      "allowed_file_types": ["txt", "csv", "json", "pdf"],
      "max_file_size_mb": 50,
      "sanitize_html": true
    },
    "rate_limiting": {
      "by_ip": true,
      "by_user": true,
      "by_api_key": true
    },
    "encryption": {
      "encrypt_at_rest": true,
      "key_rotation_days": 90
    }
  },
  
  "performance": {
    "caching": {
      "enabled": true,
      "ttl_seconds": 3600,
      "max_cache_size_mb": 1024,
      "eviction_policy": "lru"
    },
    "batching": {
      "enabled": true,
      "batch_size": 50,
      "batch_timeout_ms": 500
    },
    "connection_pooling": {
      "http_pool_size": 100,
      "http_pool_maxsize": 200
    }
  },
  
  "backup": {
    "enabled": true,
    "schedule": "0 2 * * *",
    "retention_days": 30,
    "s3_bucket": "${BACKUP_BUCKET}",
    "encryption": true
  }
}
```

## 十三、部署检查清单

### 生产环境部署前检查项：

- [ ] 所有环境变量已正确设置
- [ ] SSL证书已配置
- [ ] 数据库已创建并迁移
- [ ] Redis集群已就绪
- [ ] 监控和告警已配置
- [ ] 日志收集已设置
- [ ] 备份策略已实施
- [ ] 安全扫描已通过
- [ ] 负载测试已完成
- [ ] 容灾计划已制定
- [ ] 文档已更新
- [ ] 运维手册已准备

### 性能指标目标：

- API响应时间 P95 < 200ms
- 系统可用性 > 99.9%
- 并发用户支持 > 1000
- 每秒请求处理 > 500 RPS
- 错误率 < 0.1%

## 十四、总结

这个生产级升级方案涵盖了：

1. **架构优化**：微服务化、异步处理、消息队列
2. **安全加固**：认证授权、输入验证、沙箱执行
3. **性能提升**：缓存、连接池、批处理
4. **可靠性**：错误处理、重试、断路器
5. **可观测性**：监控、日志、追踪
6. **运维友好**：健康检查、自动伸缩、备份恢复
7. **测试完善**：单元测试、集成测试、性能测试
8. **部署自动化**：容器化、K8s编排、CI/CD

通过实施这些改进，系统将具备生产环境所需的稳定性、安全性和可扩展性。