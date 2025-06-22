# api_server.py
"""
Production-Ready API Server
FastAPI application with security, monitoring, and rate limiting
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import json
import uuid

from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import uvicorn

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from production_agent_system import (
    ProductionAgentService, Settings, ChatRequest, TaskRequest,
    MetricsCollector, AuthenticationManager, DatabaseManager,
    HealthCheckManager, logger
)

# Load settings
settings = Settings()

# Initialize services
agent_service = ProductionAgentService()
metrics_collector = MetricsCollector()
auth_manager = AuthenticationManager()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ==================== Lifespan Management ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting API server...")
    await agent_service.initialize()
    
    # Start background tasks
    asyncio.create_task(cleanup_expired_sessions())
    asyncio.create_task(metrics_aggregator())
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    # Cleanup tasks would go here

# ==================== FastAPI Application ====================

app = FastAPI(
    title="Production Agent System API",
    description="Scalable multi-agent orchestration platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.api_host == "localhost" else None,  # Disable in production
    redoc_url="/redoc" if settings.api_host == "localhost" else None
)

# ==================== Middleware ====================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Host
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==================== Request/Response Models ====================

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    duration: float

class TaskResponse(BaseModel):
    task_id: str
    status: str
    celery_task_id: Optional[str] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    description: str
    status: str
    priority: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    execution_time: Optional[float]
    result: Optional[dict]
    error: Optional[str]
    subtasks: List[dict]

class HealthResponse(BaseModel):
    status: str
    checks: Dict[str, Any]
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[dict] = None
    request_id: Optional[str] = None

# ==================== Middleware Functions ====================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None,
        request_id=getattr(request.state, 'request_id', None)
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response
    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration,
        request_id=getattr(request.state, 'request_id', None)
    )
    
    # Record metrics
    metrics_collector.record_request(
        request.method,
        request.url.path,
        response.status_code,
        duration
    )
    
    # Add timing header
    response.headers["X-Response-Time"] = f"{duration:.3f}"
    
    return response

# ==================== Authentication Dependencies ====================

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from JWT token"""
    try:
        payload = auth_manager.decode_token(token)
        user_id = int(payload.get("sub"))
        
        user = await agent_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
            
        return user
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

async def get_api_key_user(api_key: str = Depends(api_key_header)):
    """Get user from API key"""
    if not api_key:
        return None
        
    # Validate API key and get user
    # This is a simplified version - implement proper API key validation
    return None

async def get_authenticated_user(
    token_user=Depends(get_current_user),
    api_key_user=Depends(get_api_key_user)
):
    """Get authenticated user from either JWT or API key"""
    if token_user:
        return token_user
    elif api_key_user:
        return api_key_user
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

# ==================== Exception Handlers ====================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "request_id": getattr(request.state, 'request_id', None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        request_id=getattr(request.state, 'request_id', None),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )

# ==================== Health Check Endpoints ====================

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "checks": {},
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health/detailed", response_model=HealthResponse, tags=["health"])
async def detailed_health_check():
    """Detailed health check"""
    result = await agent_service.health_manager.check_health()
    
    if result['status'] != 'healthy':
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result
        )
        
    return result

@app.get("/ready", tags=["health"])
async def readiness_check():
    """Readiness check for Kubernetes"""
    result = await agent_service.health_manager.check_health()
    
    if result['status'] != 'healthy':
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
        
    return {"status": "ready"}

# ==================== Metrics Endpoint ====================

@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# ==================== Authentication Endpoints ====================

@app.post("/auth/register", response_model=UserResponse, tags=["auth"])
@limiter.limit("5/hour")
async def register(request: Request, user_data: UserCreate):
    """Register new user"""
    try:
        user = await agent_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password
        )
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at
        )
        
    except Exception as e:
        logger.error("User registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed"
        )

@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
@limiter.limit("10/hour")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token"""
    token = await agent_service.authenticate_user(
        username=form_data.username,
        password=form_data.password
    )
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expiration_minutes * 60
    )

# ==================== Chat Endpoints ====================

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    current_user=Depends(get_authenticated_user)
):
    """Process chat message"""
    try:
        result = await agent_service.process_chat_request(
            chat_request,
            current_user.id
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(
            "Chat request failed",
            error=str(e),
            user_id=current_user.id,
            request_id=request.state.request_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat processing failed"
        )

@app.get("/chat/{conversation_id}/history", tags=["chat"])
async def get_conversation_history(
    conversation_id: str,
    limit: int = 50,
    current_user=Depends(get_authenticated_user)
):
    """Get conversation history"""
    # Implementation would fetch from database
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "total": 0
    }

# ==================== Task Endpoints ====================

@app.post("/tasks", response_model=TaskResponse, tags=["tasks"])
@limiter.limit(f"{settings.rate_limit_requests//2}/minute")
async def create_task(
    request: Request,
    task_request: TaskRequest,
    current_user=Depends(get_authenticated_user)
):
    """Create new task"""
    try:
        result = await agent_service.process_task_request(
            task_request,
            current_user.id
        )
        
        return TaskResponse(**result)
        
    except Exception as e:
        logger.error(
            "Task creation failed",
            error=str(e),
            user_id=current_user.id,
            request_id=request.state.request_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task creation failed"
        )

@app.get("/tasks/{task_id}", response_model=TaskStatusResponse, tags=["tasks"])
async def get_task_status(
    task_id: str,
    current_user=Depends(get_authenticated_user)
):
    """Get task status"""
    try:
        result = await agent_service.get_task_status(task_id, current_user.id)
        return TaskStatusResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Failed to get task status",
            error=str(e),
            task_id=task_id,
            user_id=current_user.id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve task status"
        )

@app.get("/tasks", tags=["tasks"])
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user=Depends(get_authenticated_user)
):
    """List user's tasks"""
    # Implementation would fetch from database
    return {
        "tasks": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    }

# ==================== WebSocket Endpoint ====================

from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    """Manage WebSocket connections"""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
            
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Process WebSocket messages
            await manager.send_personal_message(f"Echo: {data}", client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# ==================== Admin Endpoints ====================

@app.get("/admin/stats", tags=["admin"])
async def get_system_stats(current_user=Depends(get_authenticated_user)):
    """Get system statistics (admin only)"""
    if not getattr(current_user, 'is_admin', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
        
    # Return system statistics
    return {
        "total_users": 0,
        "active_tasks": 0,
        "total_conversations": 0,
        "system_load": 0.0
    }

# ==================== Background Tasks ====================

async def cleanup_expired_sessions():
    """Clean up expired sessions periodically"""
    while True:
        try:
            # Cleanup logic would go here
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error("Session cleanup failed", error=str(e))

async def metrics_aggregator():
    """Aggregate metrics periodically"""
    while True:
        try:
            # Metrics aggregation logic would go here
            await asyncio.sleep(60)  # Run every minute
        except Exception as e:
            logger.error("Metrics aggregation failed", error=str(e))

# ==================== Main Entry Point ====================

def create_app() -> FastAPI:
    """Create FastAPI application"""
    return app

if __name__ == "__main__":
    # Configure uvicorn
    uvicorn.run(
        "api_server:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
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
        },
        access_log=True,
        use_colors=False,
        reload=False  # Never use reload in production
    )
