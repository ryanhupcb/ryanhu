# agent_client_sdk.py
# Agent系统客户端SDK - 简化API调用和集成

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import time
from functools import wraps

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class TaskResult:
    """任务结果数据类"""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentClientError(Exception):
    """Agent客户端异常基类"""
    pass

class ConnectionError(AgentClientError):
    """连接错误"""
    pass

class AuthenticationError(AgentClientError):
    """认证错误"""
    pass

class TaskExecutionError(AgentClientError):
    """任务执行错误"""
    pass

def retry_on_failure(max_retries: int = 3, backoff_factor: float = 2.0):
    """失败重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionError, aiohttp.ClientError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator

class AgentClient:
    """Agent系统客户端"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 auth_token: Optional[str] = None,
                 timeout: int = 300,
                 max_concurrent_requests: int = 10):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self._session:
            await self._session.close()
            
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers
        
    @retry_on_failure(max_retries=3)
    async def execute(self, 
                     request: str, 
                     context: Optional[Dict[str, Any]] = None,
                     callback: Optional[Callable[[TaskResult], None]] = None) -> TaskResult:
        """执行任务"""
        async with self.semaphore:
            if not self._session:
                self._session = aiohttp.ClientSession(timeout=self.timeout)
                
            task_id = f"task_{int(time.time() * 1000)}"
            
            try:
                payload = {
                    "request": request,
                    "context": context or {}
                }
                
                start_time = time.time()
                
                async with self._session.post(
                    f"{self.base_url}/execute",
                    json=payload,
                    headers=self._get_headers()
                ) as response:
                    if response.status == 401:
                        raise AuthenticationError("Invalid authentication token")
                    elif response.status != 200:
                        raise TaskExecutionError(f"Server returned {response.status}")
                        
                    data = await response.json()
                    execution_time = time.time() - start_time
                    
                    result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.SUCCESS if data.get('success') else TaskStatus.FAILED,
                        result=data.get('result'),
                        error=data.get('error'),
                        execution_time=execution_time,
                        metadata=data
                    )
                    
                    if callback:
                        callback(result)
                        
                    return result
                    
            except aiohttp.ClientError as e:
                raise ConnectionError(f"Failed to connect to agent system: {e}")
                
    async def execute_batch(self, 
                          requests: List[Union[str, Dict[str, Any]]],
                          parallel: bool = True) -> List[TaskResult]:
        """批量执行任务"""
        tasks = []
        
        for req in requests:
            if isinstance(req, str):
                task = self.execute(req)
            else:
                task = self.execute(req.get('request', ''), req.get('context'))
            tasks.append(task)
            
        if parallel:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    results.append(TaskResult(
                        task_id="error",
                        status=TaskStatus.FAILED,
                        error=str(e)
                    ))
                    
        return results
        
    async def chat(self, 
                   message: str, 
                   conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """聊天接口"""
        if not self._session:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
            
        payload = {
            "message": message,
            "conversation_id": conversation_id
        }
        
        async with self._session.post(
            f"{self.base_url}/chat",
            json=payload,
            headers=self._get_headers()
        ) as response:
            if response.status != 200:
                raise TaskExecutionError(f"Chat request failed: {response.status}")
            return await response.json()
            
    async def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if not self._session:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
            
        async with self._session.get(
            f"{self.base_url}/status",
            headers=self._get_headers()
        ) as response:
            if response.status != 200:
                raise ConnectionError(f"Failed to get status: {response.status}")
            return await response.json()
            
    async def get_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        if not self._session:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
            
        async with self._session.get(
            f"{self.base_url}/metrics",
            headers=self._get_headers()
        ) as response:
            if response.status != 200:
                raise ConnectionError(f"Failed to get metrics: {response.status}")
            return await response.json()
            
    async def call_tool(self, 
                       tool_name: str, 
                       method_name: str, 
                       **kwargs) -> Dict[str, Any]:
        """调用特定工具"""
        if not self._session:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
            
        async with self._session.post(
            f"{self.base_url}/tools/{tool_name}/{method_name}",
            json=kwargs,
            headers=self._get_headers()
        ) as response:
            if response.status == 404:
                raise TaskExecutionError(f"Tool or method not found: {tool_name}.{method_name}")
            elif response.status != 200:
                raise TaskExecutionError(f"Tool execution failed: {response.status}")
            return await response.json()

class AgentTaskBuilder:
    """任务构建器 - 提供流畅的API"""
    
    def __init__(self, client: AgentClient):
        self.client = client
        self.request = ""
        self.context = {}
        
    def with_request(self, request: str) -> 'AgentTaskBuilder':
        """设置请求"""
        self.request = request
        return self
        
    def with_context(self, **kwargs) -> 'AgentTaskBuilder':
        """设置上下文"""
        self.context.update(kwargs)
        return self
        
    def with_language(self, language: str) -> 'AgentTaskBuilder':
        """设置编程语言"""
        self.context['language'] = language
        return self
        
    def with_framework(self, framework: str) -> 'AgentTaskBuilder':
        """设置框架"""
        self.context['framework'] = framework
        return self
        
    def with_output_format(self, format: str) -> 'AgentTaskBuilder':
        """设置输出格式"""
        self.context['output_format'] = format
        return self
        
    async def execute(self) -> TaskResult:
        """执行任务"""
        return await self.client.execute(self.request, self.context)

class AgentTools:
    """工具调用简化接口"""
    
    def __init__(self, client: AgentClient):
        self.client = client
        
    async def format_python(self, code: str, line_length: int = 88) -> Dict[str, Any]:
        """格式化Python代码"""
        return await self.client.call_tool(
            'code_formatter', 
            'format_python',
            code=code,
            line_length=line_length
        )
        
    async def analyze_python(self, code: str) -> Dict[str, Any]:
        """分析Python代码"""
        return await self.client.call_tool(
            'code_analyzer',
            'analyze_python',
            code=code
        )
        
    async def create_project_structure(self, 
                                     project_name: str,
                                     structure: Dict[str, Any]) -> Dict[str, Any]:
        """创建项目结构"""
        return await self.client.call_tool(
            'file_operations',
            'create_project_structure',
            project_name=project_name,
            structure=structure
        )
        
    async def generate_readme(self, project_info: Dict[str, Any]) -> str:
        """生成README文件"""
        result = await self.client.call_tool(
            'documentation',
            'generate_readme',
            project_info=project_info
        )
        return result

# 高级功能：流式处理
class StreamingAgentClient(AgentClient):
    """支持流式响应的客户端"""
    
    async def execute_stream(self, 
                            request: str,
                            context: Optional[Dict[str, Any]] = None,
                            chunk_callback: Optional[Callable[[str], None]] = None):
        """流式执行任务"""
        # 实现WebSocket或SSE连接
        pass

# 使用示例
async def example_usage():
    """SDK使用示例"""
    
    # 1. 基础使用
    async with AgentClient(auth_token="your-token") as client:
        # 执行单个任务
        result = await client.execute("创建一个Python REST API")
        print(f"Status: {result.status}")
        print(f"Result: {result.result}")
        
        # 批量执行
        tasks = [
            "写一个排序算法",
            "创建数据库模型",
            "生成API文档"
        ]
        results = await client.execute_batch(tasks)
        
        # 使用任务构建器
        builder = AgentTaskBuilder(client)
        result = await (builder
                       .with_request("创建用户认证系统")
                       .with_language("python")
                       .with_framework("FastAPI")
                       .execute())
        
        # 使用工具
        tools = AgentTools(client)
        
        # 格式化代码
        formatted = await tools.format_python("def hello():print('world')")
        
        # 分析代码
        analysis = await tools.analyze_python(formatted['formatted_code'])
        
        # 获取系统状态
        status = await client.get_status()
        print(f"System status: {status}")

# 集成示例：Flask应用
def create_flask_integration():
    """Flask集成示例"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    agent_client = AgentClient()
    
    @app.route('/api/execute', methods=['POST'])
    async def execute_task():
        data = request.json
        result = await agent_client.execute(
            data.get('request'),
            data.get('context')
        )
        return jsonify({
            'task_id': result.task_id,
            'status': result.status.value,
            'result': result.result
        })
        
    return app

# 集成示例：Django
def create_django_integration():
    """Django集成示例"""
    from django.http import JsonResponse
    from django.views import View
    import json
    
    class AgentExecuteView(View):
        def __init__(self):
            super().__init__()
            self.client = AgentClient()
            
        async def post(self, request):
            data = json.loads(request.body)
            result = await self.client.execute(
                data.get('request'),
                data.get('context')
            )
            return JsonResponse({
                'task_id': result.task_id,
                'status': result.status.value,
                'result': result.result
            })

# 错误处理示例
async def error_handling_example():
    """错误处理示例"""
    client = AgentClient()
    
    try:
        result = await client.execute("complex task")
    except AuthenticationError:
        logger.error("Authentication failed - check your token")
    except ConnectionError:
        logger.error("Cannot connect to agent system")
    except TaskExecutionError as e:
        logger.error(f"Task execution failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())