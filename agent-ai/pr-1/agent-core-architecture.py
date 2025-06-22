# LocalAgentSystem - Core Architecture
# 本地Agent系统核心架构

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import time
from datetime import datetime
import logging
from enum import Enum
import aiohttp
from collections import deque
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 基础数据结构 ====================

class AgentRole(Enum):
    """Agent角色枚举"""
    ORCHESTRATOR = "orchestrator"  # 协调者
    CODER = "coder"  # 代码开发
    RESEARCHER = "researcher"  # 研究者
    EXECUTOR = "executor"  # 执行者
    REVIEWER = "reviewer"  # 审核者
    UI_CONTROLLER = "ui_controller"  # UI控制


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Thought:
    """思考节点 - 用于Tree of Thoughts"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    score: float = 0.0
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Message:
    """消息结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = ""  # system, user, assistant, tool
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Task:
    """任务结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 记忆系统 ====================

class Memory:
    """记忆管理系统"""
    
    def __init__(self, max_short_term: int = 100, max_long_term: int = 10000):
        self.short_term: deque = deque(maxlen=max_short_term)
        self.long_term: List[Dict[str, Any]] = []
        self.max_long_term = max_long_term
        self.working_memory: Dict[str, Any] = {}
        
    def add_to_short_term(self, item: Dict[str, Any]):
        """添加到短期记忆"""
        self.short_term.append({
            **item,
            "timestamp": datetime.now().isoformat()
        })
        
    def add_to_long_term(self, item: Dict[str, Any]):
        """添加到长期记忆"""
        if len(self.long_term) >= self.max_long_term:
            # 移除最旧的记忆
            self.long_term.pop(0)
        self.long_term.append({
            **item,
            "timestamp": datetime.now().isoformat()
        })
        
    def search_memory(self, query: str, memory_type: str = "all", limit: int = 10) -> List[Dict[str, Any]]:
        """搜索记忆"""
        results = []
        
        if memory_type in ["all", "short"]:
            for item in self.short_term:
                if query.lower() in str(item).lower():
                    results.append(item)
                    
        if memory_type in ["all", "long"]:
            for item in self.long_term:
                if query.lower() in str(item).lower():
                    results.append(item)
                    
        return results[:limit]
        
    def get_context(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取上下文（最近的记忆）"""
        recent_short = list(self.short_term)[-limit:]
        return recent_short


# ==================== 工具系统 ====================

class Tool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """执行工具"""
        pass


class PythonExecutor(Tool):
    """Python代码执行工具"""
    
    def __init__(self):
        super().__init__(
            name="python_executor",
            description="Execute Python code safely in a sandboxed environment"
        )
        
    async def execute(self, code: str) -> Dict[str, Any]:
        """执行Python代码"""
        try:
            # 这里应该使用安全的沙箱环境
            # 简化示例，实际应使用 subprocess 或 Docker
            exec_globals = {}
            exec(code, exec_globals)
            
            return {
                "success": True,
                "output": exec_globals.get("result", "Code executed successfully"),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }


class FileOperator(Tool):
    """文件操作工具"""
    
    def __init__(self, workspace_dir: str = "./workspace"):
        super().__init__(
            name="file_operator",
            description="Read, write, and manipulate files"
        )
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
        
    async def execute(self, action: str, path: str, content: str = None) -> Dict[str, Any]:
        """执行文件操作"""
        try:
            full_path = os.path.join(self.workspace_dir, path)
            
            if action == "read":
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {"success": True, "content": content, "error": None}
                
            elif action == "write":
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {"success": True, "message": f"File written: {path}", "error": None}
                
            elif action == "delete":
                os.remove(full_path)
                return {"success": True, "message": f"File deleted: {path}", "error": None}
                
            elif action == "list":
                files = os.listdir(os.path.dirname(full_path) or self.workspace_dir)
                return {"success": True, "files": files, "error": None}
                
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}


class WebSearcher(Tool):
    """网络搜索工具"""
    
    def __init__(self):
        super().__init__(
            name="web_searcher",
            description="Search the web for information"
        )
        
    async def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """执行网络搜索"""
        # 这里应该接入实际的搜索API
        # 简化示例
        return {
            "success": True,
            "results": [
                {
                    "title": f"Result {i+1} for: {query}",
                    "url": f"https://example.com/{i+1}",
                    "snippet": f"This is a sample result for the query: {query}"
                }
                for i in range(num_results)
            ],
            "error": None
        }


# ==================== LLM接口 ====================

class LLMInterface(ABC):
    """LLM接口基类"""
    
    @abstractmethod
    async def generate(self, messages: List[Message], **kwargs) -> str:
        """生成响应"""
        pass
        
    @abstractmethod
    async def generate_with_tools(self, messages: List[Message], tools: List[Tool], **kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
        """生成响应并可能调用工具"""
        pass


class ClaudeLLM(LLMInterface):
    """Claude LLM接口"""
    
    def __init__(self, api_key: str, model: str = "claude-opus-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        
    async def generate(self, messages: List[Message], **kwargs) -> str:
        """调用Claude API生成响应"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ],
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                return result.get("content", [{}])[0].get("text", "")
                
    async def generate_with_tools(self, messages: List[Message], tools: List[Tool], **kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
        """调用Claude API并支持工具调用"""
        # 将工具转换为Claude的函数调用格式
        tool_definitions = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            for tool in tools
        ]
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ],
                "tools": tool_definitions,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                
                # 检查是否有工具调用
                content = result.get("content", [])
                text_response = ""
                tool_call = None
                
                for item in content:
                    if item.get("type") == "text":
                        text_response += item.get("text", "")
                    elif item.get("type") == "tool_use":
                        tool_call = {
                            "name": item.get("name"),
                            "arguments": item.get("input", {})
                        }
                        
                return text_response, tool_call


class DeepSeekLLM(LLMInterface):
    """DeepSeek-Coder LLM接口"""
    
    def __init__(self, api_key: str = None, model: str = "deepseek-coder"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com"
        
    async def generate(self, messages: List[Message], **kwargs) -> str:
        """调用DeepSeek API生成响应"""
        # 如果没有API key，尝试本地运行
        if not self.api_key:
            # 这里应该调用本地运行的DeepSeek-Coder
            # 使用 Ollama 或其他本地推理框架
            return "# DeepSeek-Coder response (local mode)\n# Code implementation here..."
            
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ],
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.3)
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
    async def generate_with_tools(self, messages: List[Message], tools: List[Tool], **kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
        """DeepSeek暂不支持工具调用，仅返回文本"""
        response = await self.generate(messages, **kwargs)
        return response, None


class QwenLLM(LLMInterface):
    """Qwen LLM接口 - 用于用户交互"""
    
    def __init__(self, api_key: str, model: str = "qwen-plus"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        
    async def generate(self, messages: List[Message], **kwargs) -> str:
        """调用Qwen API生成响应"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ],
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
    async def generate_with_tools(self, messages: List[Message], tools: List[Tool], **kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Qwen支持函数调用"""
        tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            for tool in tools
        ]
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ],
                "tools": tool_definitions,
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                
                choice = result.get("choices", [{}])[0]
                message = choice.get("message", {})
                
                text_response = message.get("content", "")
                tool_call = None
                
                if message.get("tool_calls"):
                    tool_call_data = message["tool_calls"][0]
                    tool_call = {
                        "name": tool_call_data["function"]["name"],
                        "arguments": json.loads(tool_call_data["function"]["arguments"])
                    }
                    
                return text_response, tool_call
