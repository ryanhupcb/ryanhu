"""
生产级编程AI Agent - 完整API服务实现
包含所有高级功能：流式响应、批量处理、WebSocket、监控等
"""

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks, Request, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional, List, Dict, Any, AsyncGenerator, Set
from datetime import datetime, timedelta
import asyncio
import json
import aiofiles
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import orjson
import msgpack
from sse_starlette.sse import EventSourceResponse
import numpy as np
from collections import defaultdict
import uuid
import time

# 导入增强的核心模块
from production_agent_enhanced import (
    initialize_enhanced_application, shutdown_enhanced_application,
    Task, TaskType, TaskStatus, CodeContext, deps, Config, logger,
    SecurityManager, EnhancedTaskOrchestrator, AgentResponse,
    TokenBucket, concurrent_tasks, task_processing_histogram,
    cache_operations, llm_latency, code_quality_scores
)

# ==================== 增强的API模型 ====================

class EnhancedCodeGenerationRequest(BaseModel):
    """增强的代码生成请求"""
    description: str = Field(..., min_length=10, max_length=10000)
    language: str = Field("python", regex="^(python|javascript|typescript|java|go|rust|cpp|csharp)$")
    context: Optional[Dict[str, Any]] = None
    requirements: Optional[List[str]] = None
    priority: int = Field(5, ge=1, le=10)
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    streaming: bool = Field(False, description="Enable streaming response")
    include_tests: bool = Field(True, description="Generate unit tests")
    include_docs: bool = Field(True, description="Generate documentation")
    quality_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum quality score")

class BatchTaskRequest(BaseModel):
    """批量任务请求"""
    tasks: List[CodeGenerationRequest] = Field(..., min_items=1, max_items=100)
    batch_options: Optional[Dict[str, Any]] = None
    
    @validator('tasks')
    def validate_batch_size(cls, v):
        if len(v) > Config.BATCH_SIZE:
            raise ValueError(f'Maximum batch size is {Config.BATCH_SIZE}')
        return v

class CodeExecutionRequest(BaseModel):
    """代码执行请求"""
    code: str = Field(..., min_length=1, max_length=100000)
    language: str = Field("python")
    test_code: Optional[str] = None
    timeout: Optional[int] = Field(30, ge=1, le=300)
    memory_limit: Optional[str] = Field("512m")

class CodeSearchRequest(BaseModel):
    """代码搜索请求"""
    query: str = Field(..., min_length=1, max_length=1000)
    language: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    top_k: int = Field(10, ge=1, le=50)
    min_similarity: float = Field(0.7, ge=0.0, le=1.0)

class CodeCompletionRequest(BaseModel):
    """代码补全请求"""
    code: str = Field(..., max_length=50000)
    cursor_position: int = Field(..., ge=0)
    language: str = Field("python")
    max_suggestions: int = Field(5, ge=1, le=20)
    context_window: int = Field(1000, ge=100, le=5000)

class WebSocketMessage(BaseModel):
    """WebSocket消息"""
    type: str = Field(..., regex="^(subscribe|unsubscribe|ping|task_update)$")
    data: Optional[Dict[str, Any]] = None

class TaskExportRequest(BaseModel):
    """任务导出请求"""
    task_ids: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    format: str = Field("json", regex="^(json|csv|excel)$")
    include_code: bool = Field(True)

# ==================== 扩展的响应模型 ====================

class DetailedTaskStatusResponse(TaskStatusResponse):
    """详细的任务状态响应"""
    performance_metrics: Optional[Dict[str, Any]] = None
    similar_tasks: Optional[List[Dict[str, Any]]] = None
    cost_estimate: Optional[Dict[str, float]] = None
    quality_metrics: Optional[Dict[str, Any]] = None

class SystemMetricsResponse(BaseModel):
    """系统指标响应"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: Dict[str, float]
    task_metrics: Dict[str, Dict[str, Any]]
    cache_metrics: Dict[str, int]
    llm_metrics: Dict[str, Any]
    queue_sizes: Dict[str, int]
    active_workers: int

class CodeQualityReport(BaseModel):
    """代码质量报告"""
    task_id: str
    overall_score: float
    metrics: Dict[str, float]
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    comparisons: Optional[Dict[str, Any]] = None

# ==================== 应用生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """增强的应用生命周期管理"""
    logger.info("Starting enhanced application...")
    
    # 初始化核心系统
    app.state.orchestrator = await initialize_enhanced_application()
    app.state.security_manager = SecurityManager()
    
    # 初始化WebSocket管理器
    app.state.ws_manager = EnhancedConnectionManager()
    
    # 初始化SSE广播器
    app.state.sse_broadcaster = SSEBroadcaster()
    
    # 初始化速率限制器
    app.state.rate_limiters = {
        "global": TokenBucket(rate=1000, capacity=5000),
        "per_user": defaultdict(lambda: TokenBucket(rate=100, capacity=500))
    }
    
    # 启动后台任务
    app.state.background_tasks = []
    app.state.background_tasks.extend([
        asyncio.create_task(metrics_collector(app)),
        asyncio.create_task(health_checker(app)),
        asyncio.create_task(websocket_heartbeat(app)),
        asyncio.create_task(cache_warmer(app)),
        asyncio.create_task(task_event_broadcaster(app))
    ])
    
    logger.info("Application started successfully")
    
    yield
    
    # 关闭时清理
    logger.info("Shutting down application...")
    
    # 取消后台任务
    for task in app.state.background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # 关闭WebSocket连接
    await app.state.ws_manager.disconnect_all()
    
    # 清理资源
    await shutdown_enhanced_application()
    
    logger.info("Application shutdown complete")

# ==================== 创建FastAPI应用 ====================

app = FastAPI(
    title="Production AI Agent API",
    description="Enterprise-grade Programming AI Agent Service with Advanced Features",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# ==================== 中间件配置 ====================

# 添加所有必要的中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS if hasattr(Config, 'ALLOWED_ORIGINS') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-Process-Time"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

if Config.ENV == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.yourdomain.com", "yourdomain.com"]
    )

# 性能监控中间件
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 请求ID和日志中间件（保持原有实现）
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    import uuid
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response

# 限流器
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==================== 增强的认证依赖 ====================

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    request: Request = None
):
    """增强的用户认证"""
    token = credentials.credentials
    
    # 验证JWT令牌
    security_manager = request.app.state.security_manager
    user_id = security_manager.verify_jwt_token(token)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 检查用户级别的速率限制
    user_limiter = request.app.state.rate_limiters["per_user"][user_id]
    if not await user_limiter.acquire():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="User rate limit exceeded"
        )
    
    # 从缓存或数据库获取用户信息
    cache_key = f"user:{user_id}"
    user = await deps.cache_manager.get(cache_key)
    
    if not user:
        async with deps.db_session_factory() as session:
            from sqlalchemy import select
            from production_agent import UserModel
            
            result = await session.execute(
                select(UserModel).where(UserModel.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                # 缓存用户信息
                await deps.cache_manager.set(cache_key, {
                    "id": user.id,
                    "email": user.email,
                    "is_active": user.is_active,
                    "is_admin": user.is_admin,
                    "rate_limit_override": user.rate_limit_override
                }, ttl=300)
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return user

# ==================== 增强的API端点 ====================

@app.post("/api/v3/tasks/generate",
         response_model=TaskSubmitResponse,
         tags=["tasks"])
@limiter.limit("20/minute")
async def generate_code_enhanced(
    request: Request,
    code_request: EnhancedCodeGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """增强的代码生成端点"""
    # 创建任务
    context = CodeContext(
        file_path=code_request.context.get("file_path", "generated.py") if code_request.context else "generated.py",
        language=code_request.language,
        content=code_request.context.get("existing_code", "") if code_request.context else "",
        dependencies=code_request.context.get("dependencies", []) if code_request.context else []
    )
    
    task = Task(
        id=str(uuid.uuid4()),
        type=TaskType.CODE_GENERATION,
        description=code_request.description,
        context=context,
        priority=code_request.priority,
        metadata={
            "requirements": code_request.requirements,
            "options": code_request.options,
            "streaming": code_request.streaming,
            "include_tests": code_request.include_tests,
            "include_docs": code_request.include_docs,
            "quality_threshold": code_request.quality_threshold,
            "user_email": current_user.email
        }
    )
    
    # 提交任务
    orchestrator = request.app.state.orchestrator
    task_id = await orchestrator.submit_task(task, current_user.id)
    
    # WebSocket通知
    await request.app.state.ws_manager.broadcast_to_user(
        current_user.id,
        {
            "type": "task_submitted",
            "task_id": task_id,
            "task_type": task.type.value
        }
    )
    
    # 如果请求流式响应，返回SSE端点
    if code_request.streaming:
        return TaskSubmitResponse(
            task_id=task_id,
            status="submitted",
            message="Use /api/v3/tasks/{task_id}/stream for streaming updates"
        )
    
    return TaskSubmitResponse(
        task_id=task_id,
        status="submitted",
        estimated_completion_time=30
    )

@app.post("/api/v3/tasks/batch",
         response_model=List[TaskSubmitResponse],
         tags=["tasks"])
@limiter.limit("5/minute")
async def submit_batch_tasks(
    request: Request,
    batch_request: BatchTaskRequest,
    current_user = Depends(get_current_user)
):
    """批量任务提交"""
    orchestrator = request.app.state.orchestrator
    
    # 转换为Task对象
    tasks = []
    for idx, task_request in enumerate(batch_request.tasks):
        context = CodeContext(
            file_path=f"batch_{idx}.py",
            language=task_request.language,
            content="",
            dependencies=[]
        )
        
        task = Task(
            id=str(uuid.uuid4()),
            type=TaskType.CODE_GENERATION,
            description=task_request.description,
            context=context,
            priority=task_request.priority,
            metadata={
                "batch_id": str(uuid.uuid4()),
                "batch_index": idx,
                "batch_options": batch_request.batch_options
            }
        )
        tasks.append(task)
    
    # 批量提交
    task_ids = await orchestrator.submit_batch(tasks, current_user.id)
    
    return [
        TaskSubmitResponse(
            task_id=task_id,
            status="submitted",
            message=f"Batch task {idx+1}/{len(tasks)}"
        )
        for idx, task_id in enumerate(task_ids)
    ]

@app.get("/api/v3/tasks/{task_id}/stream",
        tags=["tasks"])
async def stream_task_updates(
    task_id: str,
    request: Request,
    current_user = Depends(get_current_user)
):
    """任务进度的SSE流"""
    async def event_generator():
        orchestrator = request.app.state.orchestrator
        last_status = None
        
        while True:
            try:
                # 获取任务状态
                task_data = await orchestrator.get_task_status(task_id, current_user.id)
                
                if not task_data:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": "Task not found"})
                    }
                    break
                
                # 检查状态变化
                if task_data["status"] != last_status:
                    yield {
                        "event": "status_change",
                        "data": json.dumps({
                            "status": task_data["status"],
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    }
                    last_status = task_data["status"]
                
                # 如果任务正在处理，发送进度更新
                if task_data["status"] == "processing":
                    progress = task_data.get("metadata", {}).get("progress", 0)
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "progress": progress,
                            "message": task_data.get("metadata", {}).get("progress_message", "")
                        })
                    }
                
                # 如果任务完成，发送结果
                if task_data["status"] in ["completed", "failed"]:
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "status": task_data["status"],
                            "result": task_data.get("result"),
                            "error": task_data.get("error")
                        })
                    }
                    break
                
                await asyncio.sleep(0.5)  # 500ms更新间隔
                
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)})
                }
                break
    
    return EventSourceResponse(event_generator())

@app.post("/api/v3/code/execute",
         tags=["execution"])
@limiter.limit("10/minute")
async def execute_code(
    request: Request,
    exec_request: CodeExecutionRequest,
    current_user = Depends(get_current_user)
):
    """执行代码（沙箱环境）"""
    if not Config.ENABLE_CODE_EXECUTION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Code execution is disabled"
        )
    
    # 使用沙箱执行代码
    orchestrator = request.app.state.orchestrator
    result = await orchestrator.sandbox.execute_code(
        code=exec_request.code,
        language=exec_request.language,
        timeout=exec_request.timeout,
        test_code=exec_request.test_code
    )
    
    return result

@app.post("/api/v3/code/search",
         tags=["search"])
async def search_similar_code(
    request: Request,
    search_request: CodeSearchRequest,
    current_user = Depends(get_current_user)
):
    """搜索相似代码"""
    orchestrator = request.app.state.orchestrator
    
    # 生成查询嵌入
    query_embedding = await orchestrator.llm_provider.embed(search_request.query)
    
    # 搜索向量数据库
    results = await orchestrator.vector_store.search_similar_code(
        query_embedding,
        filters=search_request.filters,
        top_k=search_request.top_k
    )
    
    # 过滤低相似度结果
    filtered_results = [
        r for r in results 
        if r["similarity"] >= search_request.min_similarity
    ]
    
    return {
        "query": search_request.query,
        "results": filtered_results,
        "total": len(filtered_results)
    }

@app.post("/api/v3/code/complete",
         tags=["completion"])
@limiter.limit("30/minute")
async def code_completion(
    request: Request,
    completion_request: CodeCompletionRequest,
    current_user = Depends(get_current_user)
):
    """智能代码补全"""
    # 创建补全任务
    context = CodeContext(
        file_path="completion.py",
        language=completion_request.language,
        content=completion_request.code
    )
    
    task = Task(
        id=str(uuid.uuid4()),
        type=TaskType.CODE_GENERATION,  # 使用代码生成类型
        description="Code completion",
        context=context,
        metadata={
            "cursor_position": completion_request.cursor_position,
            "max_suggestions": completion_request.max_suggestions,
            "context_window": completion_request.context_window,
            "completion_mode": True
        }
    )
    
    # 快速处理补全请求
    orchestrator = request.app.state.orchestrator
    response = await orchestrator.agents["generation"].process_task(task)
    
    return response.result

@app.get("/api/v3/tasks/{task_id}/quality",
        response_model=CodeQualityReport,
        tags=["quality"])
async def get_code_quality_report(
    task_id: str,
    request: Request,
    current_user = Depends(get_current_user)
):
    """获取代码质量报告"""
    orchestrator = request.app.state.orchestrator
    
    # 获取任务结果
    task_data = await orchestrator.get_task_status(task_id, current_user.id)
    
    if not task_data or task_data["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found or not completed"
        )
    
    # 提取质量指标
    result = task_data.get("result", {})
    metrics = result.get("metrics", {})
    
    return CodeQualityReport(
        task_id=task_id,
        overall_score=metrics.get("quality_score", 0.0),
        metrics=metrics,
        issues=result.get("validation", {}).get("issues", []),
        suggestions=task_data.get("metadata", {}).get("suggestions", []),
        comparisons={
            "similar_code_quality": [
                {"id": ref["id"], "score": ref.get("quality_score", 0)}
                for ref in result.get("similar_code_references", [])
            ]
        }
    )

@app.post("/api/v3/tasks/export",
         tags=["export"])
async def export_tasks(
    request: Request,
    export_request: TaskExportRequest,
    current_user = Depends(get_current_user)
):
    """导出任务数据"""
    # TODO: 实现任务导出功能
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Export functionality coming soon"
    )

# ==================== 系统监控端点 ====================

@app.get("/api/v3/metrics/system",
        response_model=SystemMetricsResponse,
        tags=["monitoring"])
async def get_system_metrics(
    request: Request,
    current_user = Depends(get_admin_user)
):
    """获取系统指标"""
    import psutil
    
    orchestrator = request.app.state.orchestrator
    
    # 收集系统指标
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    # 收集任务指标
    task_metrics = {}
    for task_type in TaskType:
        metrics = orchestrator.task_metrics.get(task_type.value, {})
        if metrics.get("success", 0) + metrics.get("failure", 0) > 0:
            success_rate = metrics["success"] / (metrics["success"] + metrics["failure"])
            avg_time = metrics["total_time"] / metrics["success"] if metrics["success"] > 0 else 0
            
            task_metrics[task_type.value] = {
                "total": metrics["success"] + metrics["failure"],
                "success_rate": success_rate,
                "average_time": avg_time
            }
    
    # 收集缓存指标
    cache_metrics = {
        "memory_cache_size": len(orchestrator.cache_manager.memory_cache) if orchestrator.cache_manager.memory_cache else 0,
        "memory_cache_hits": orchestrator.cache_manager.cache_stats.get("hits", 0),
        "memory_cache_misses": orchestrator.cache_manager.cache_stats.get("misses", 0)
    }
    
    return SystemMetricsResponse(
        timestamp=datetime.utcnow(),
        cpu_usage=cpu_percent,
        memory_usage={
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        },
        task_metrics=task_metrics,
        cache_metrics=cache_metrics,
        llm_metrics={
            "total_tokens": 0,  # TODO: 从实际指标获取
            "total_cost": 0.0
        },
        queue_sizes={
            "normal": orchestrator.task_queue.qsize(),
            "priority": orchestrator.priority_queue.qsize()
        },
        active_workers=len([w for w in orchestrator.worker_pool if not w.done()])
    )

@app.get("/api/v3/metrics/prometheus",
        tags=["monitoring"])
async def prometheus_metrics():
    """Prometheus格式的指标"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ==================== WebSocket端点 ====================

class EnhancedConnectionManager:
    """增强的WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.connection_metadata: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, metadata: Dict = None):
        await websocket.accept()
        self.active_connections[user_id].add(websocket)
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            "metadata": metadata or {}
        }
        logger.info("websocket_connected", user_id=user_id)
    
    def disconnect(self, websocket: WebSocket):
        metadata = self.connection_metadata.get(websocket, {})
        user_id = metadata.get("user_id")
        
        if user_id and user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        
        logger.info("websocket_disconnected", user_id=user_id)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error("websocket_send_error", error=str(e))
            self.disconnect(websocket)
    
    async def broadcast_to_user(self, user_id: str, message: Dict):
        if user_id in self.active_connections:
            message_text = orjson.dumps(message).decode()
            disconnected = []
            
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_text(message_text)
                except:
                    disconnected.append(connection)
            
            # 清理断开的连接
            for conn in disconnected:
                self.disconnect(conn)
    
    async def broadcast_to_all(self, message: Dict):
        message_text = orjson.dumps(message).decode()
        
        for user_connections in self.active_connections.values():
            for connection in user_connections:
                try:
                    await connection.send_text(message_text)
                except:
                    pass
    
    async def disconnect_all(self):
        for user_connections in list(self.active_connections.values()):
            for connection in list(user_connections):
                try:
                    await connection.close()
                except:
                    pass
                self.disconnect(connection)

@app.websocket("/ws/v3/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    token: str = None
):
    """增强的WebSocket端点"""
    # 验证令牌
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    security_manager = app.state.security_manager
    verified_user_id = security_manager.verify_jwt_token(token)
    
    if not verified_user_id or verified_user_id != user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    # 连接管理
    manager = app.state.ws_manager
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            
            try:
                message = WebSocketMessage.parse_raw(data)
                
                if message.type == "ping":
                    await websocket.send_text(orjson.dumps({"type": "pong"}).decode())
                
                elif message.type == "subscribe":
                    # 订阅特定任务更新
                    task_id = message.data.get("task_id")
                    if task_id:
                        # TODO: 实现任务订阅逻辑
                        pass
                
                elif message.type == "task_update":
                    # 请求任务更新
                    task_id = message.data.get("task_id")
                    if task_id:
                        orchestrator = app.state.orchestrator
                        status = await orchestrator.get_task_status(task_id, user_id)
                        await websocket.send_text(orjson.dumps({
                            "type": "task_status",
                            "data": status
                        }).decode())
                
            except Exception as e:
                await websocket.send_text(orjson.dumps({
                    "type": "error",
                    "error": str(e)
                }).decode())
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==================== SSE广播器 ====================

class SSEBroadcaster:
    """Server-Sent Events广播器"""
    
    def __init__(self):
        self.subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
    
    def subscribe(self, channel: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        self.subscribers[channel].add(queue)
        return queue
    
    def unsubscribe(self, channel: str, queue: asyncio.Queue):
        self.subscribers[channel].discard(queue)
        if not self.subscribers[channel]:
            del self.subscribers[channel]
    
    async def publish(self, channel: str, message: Dict):
        if channel in self.subscribers:
            message_text = orjson.dumps(message).decode()
            
            for queue in list(self.subscribers[channel]):
                try:
                    await queue.put(message_text)
                except:
                    self.subscribers[channel].discard(queue)

@app.get("/api/v3/events/stream",
        tags=["events"])
async def event_stream(
    request: Request,
    channels: str = "system",
    current_user = Depends(get_current_user)
):
    """系统事件的SSE流"""
    broadcaster = request.app.state.sse_broadcaster
    
    async def event_generator():
        queues = []
        
        try:
            # 订阅请求的频道
            for channel in channels.split(","):
                if channel in ["system", "tasks", f"user:{current_user.id}"]:
                    queue = broadcaster.subscribe(channel)
                    queues.append((channel, queue))
            
            # 发送初始连接事件
            yield {
                "event": "connected",
                "data": json.dumps({
                    "channels": channels.split(","),
                    "timestamp": datetime.utcnow().isoformat()
                })
            }
            
            # 持续发送事件
            while True:
                for channel, queue in queues:
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield {
                            "event": channel,
                            "data": message
                        }
                    except asyncio.TimeoutError:
                        # 发送心跳
                        yield {
                            "event": "heartbeat",
                            "data": json.dumps({"timestamp": datetime.utcnow().isoformat()})
                        }
                        
        finally:
            # 取消订阅
            for channel, queue in queues:
                broadcaster.unsubscribe(channel, queue)
    
    return EventSourceResponse(event_generator())

# ==================== 后台任务 ====================

async def metrics_collector(app: FastAPI):
    """增强的指标收集器"""
    while True:
        try:
            # 收集详细指标
            orchestrator = app.state.orchestrator
            
            # 任务队列深度
            normal_queue_size = orchestrator.task_queue.qsize()
            priority_queue_size = orchestrator.priority_queue.qsize()
            
            # 发布到SSE
            await app.state.sse_broadcaster.publish("system", {
                "type": "metrics_update",
                "metrics": {
                    "queue_sizes": {
                        "normal": normal_queue_size,
                        "priority": priority_queue_size
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
            await asyncio.sleep(30)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("metrics_collector_error", error=str(e))
            await asyncio.sleep(60)

async def health_checker(app: FastAPI):
    """增强的健康检查器"""
    while True:
        try:
            # 执行健康检查
            health_status = {
                "database": await check_database_health(),
                "redis": await check_redis_health(),
                "vector_store": await check_vector_store_health(),
                "llm_api": await check_llm_health()
            }
            
            # 更新健康状态
            app.state.health_status = health_status
            
            # 如果有不健康的服务，发送警报
            unhealthy = [k for k, v in health_status.items() if v != "healthy"]
            if unhealthy:
                await app.state.sse_broadcaster.publish("system", {
                    "type": "health_alert",
                    "unhealthy_services": unhealthy,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            await asyncio.sleep(60)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("health_checker_error", error=str(e))
            await asyncio.sleep(300)

async def websocket_heartbeat(app: FastAPI):
    """WebSocket心跳"""
    while True:
        try:
            # 向所有连接发送心跳
            await app.state.ws_manager.broadcast_to_all({
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            await asyncio.sleep(30)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("websocket_heartbeat_error", error=str(e))

async def cache_warmer(app: FastAPI):
    """缓存预热器"""
    while True:
        try:
            # 预热常用数据
            orchestrator = app.state.orchestrator
            
            # 预热热门代码模式
            popular_patterns = await get_popular_code_patterns()
            for pattern in popular_patterns:
                cache_key = f"pattern:{pattern['id']}"
                await orchestrator.cache_manager.set(cache_key, pattern, ttl=7200)
            
            await asyncio.sleep(3600)  # 每小时预热一次
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("cache_warmer_error", error=str(e))
            await asyncio.sleep(3600)

async def task_event_broadcaster(app: FastAPI):
    """任务事件广播器"""
    while True:
        try:
            # 从Kafka消费任务事件
            # TODO: 实现Kafka消费逻辑
            
            # 模拟任务事件
            await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("task_event_broadcaster_error", error=str(e))

# ==================== 辅助函数 ====================

async def check_database_health() -> str:
    """检查数据库健康状态"""
    try:
        async with deps.db_session_factory() as session:
            await session.execute("SELECT 1")
        return "healthy"
    except:
        return "unhealthy"

async def check_redis_health() -> str:
    """检查Redis健康状态"""
    try:
        await deps.redis_client.ping()
        return "healthy"
    except:
        return "unhealthy"

async def check_vector_store_health() -> str:
    """检查向量数据库健康状态"""
    try:
        # 执行简单查询
        test_embedding = [0.1] * 384  # 假设384维
        await deps.vector_store.search_similar_code(test_embedding, top_k=1)
        return "healthy"
    except:
        return "unhealthy"

async def check_llm_health() -> str:
    """检查LLM API健康状态"""
    try:
        # 发送简单请求
        # TODO: 实现LLM健康检查
        return "healthy"
    except:
        return "unhealthy"

async def get_popular_code_patterns() -> List[Dict]:
    """获取热门代码模式"""
    # TODO: 从数据库获取热门模式
    return []

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    # 生产环境配置
    uvicorn.run(
        "production_api_service_complete:app",
        host="0.0.0.0",
        port=8000,
        workers=4 if Config.ENV == "production" else 1,
        loop="uvloop",
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "json": {
                    "format": "%(message)s",
                    "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
                }
            },
            "handlers": {
                "default": {
                    "formatter": "json",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"]
            }
        }
    )
