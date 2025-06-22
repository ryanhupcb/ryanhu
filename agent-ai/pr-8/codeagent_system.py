# CodeAgent - 生产级代码编辑助手系统
# 完整的架构设计和核心实现

## 项目结构
```
codeagent/
├── core/                    # 核心功能模块
│   ├── __init__.py
│   ├── agent_manager.py     # Agent管理器
│   ├── code_analyzer.py     # 代码分析引擎
│   ├── code_generator.py    # 代码生成引擎
│   ├── system_controller.py # 系统控制器
│   └── memory_manager.py    # 内存管理器
├── api/                     # API接口层
│   ├── __init__.py
│   ├── claude_client.py     # Claude API客户端
│   ├── model_selector.py    # 模型选择器
│   └── rate_limiter.py      # 速率限制器
├── ui/                      # 用户界面
│   ├── __init__.py
│   ├── main_window.py       # 主窗口
│   ├── chat_interface.py    # 聊天界面
│   ├── code_editor.py       # 代码编辑器
│   └── theme_manager.py     # 主题管理器
├── utils/                   # 工具模块
│   ├── __init__.py
│   ├── config.py           # 配置管理
│   ├── logger.py           # 日志系统
│   ├── security.py         # 安全模块
│   └── performance.py      # 性能监控
├── langchain_modules/       # LangChain集成
│   ├── __init__.py
│   ├── chains.py           # 链式处理
│   ├── prompts.py          # 提示模板
│   ├── memory.py           # 记忆系统
│   └── tools.py            # 工具集成
├── computer_control/        # 计算机控制
│   ├── __init__.py
│   ├── screen_capture.py   # 屏幕捕获
│   ├── input_controller.py # 输入控制
│   ├── file_manager.py     # 文件管理
│   └── process_manager.py  # 进程管理
└── tests/                  # 测试套件
```

## 核心实现

### 1. 主程序入口 (main.py)
```python
import sys
import asyncio
from typing import Optional
from dataclasses import dataclass
from PyQt6.QtWidgets import QApplication
from core.agent_manager import AgentManager
from ui.main_window import MainWindow
from utils.config import Config
from utils.logger import Logger

@dataclass
class CodeAgentConfig:
    """全局配置类"""
    api_key: str
    default_model: str = "claude-opus-4-20250514"
    max_tokens: int = 100000
    temperature: float = 0.2
    enable_computer_control: bool = True
    enable_code_execution: bool = True
    memory_size_mb: int = 2048
    log_level: str = "INFO"

class CodeAgent:
    """主应用程序类"""
    
    def __init__(self, config: CodeAgentConfig):
        self.config = config
        self.logger = Logger(__name__, config.log_level)
        self.agent_manager: Optional[AgentManager] = None
        self.app: Optional[QApplication] = None
        self.main_window: Optional[MainWindow] = None
        
    async def initialize(self):
        """异步初始化系统"""
        try:
            self.logger.info("初始化CodeAgent系统...")
            
            # 初始化Agent管理器
            self.agent_manager = AgentManager(self.config)
            await self.agent_manager.initialize()
            
            # 初始化UI
            self.app = QApplication(sys.argv)
            self.main_window = MainWindow(self.agent_manager)
            
            self.logger.info("系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            raise
            
    def run(self):
        """运行应用程序"""
        if not self.main_window:
            raise RuntimeError("系统未初始化")
            
        self.main_window.show()
        return self.app.exec()

# 异步主函数
async def async_main():
    config = CodeAgentConfig(
        api_key="your-api-key-here",
        default_model="claude-opus-4-20250514"
    )
    
    agent = CodeAgent(config)
    await agent.initialize()
    
    # 创建异步事件循环
    loop = asyncio.get_event_loop()
    
    # 在后台运行异步任务
    async def run_app():
        return agent.run()
    
    await run_app()

if __name__ == "__main__":
    asyncio.run(async_main())
```

### 2. Agent管理器 (core/agent_manager.py)
```python
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from api.claude_client import ClaudeClient
from api.model_selector import ModelSelector
from core.code_analyzer import CodeAnalyzer
from core.code_generator import CodeGenerator
from core.system_controller import SystemController
from core.memory_manager import MemoryManager
from langchain_modules.chains import CodeProcessingChain
from utils.logger import Logger

@dataclass
class AgentTask:
    """Agent任务定义"""
    id: str
    type: str  # 'analyze', 'generate', 'edit', 'execute'
    content: Any
    priority: int = 5
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class AgentManager:
    """Agent管理器 - 核心调度系统"""
    
    def __init__(self, config):
        self.config = config
        self.logger = Logger(__name__)
        
        # 核心组件
        self.claude_client = ClaudeClient(config.api_key)
        self.model_selector = ModelSelector(config)
        self.code_analyzer = CodeAnalyzer()
        self.code_generator = CodeGenerator()
        self.system_controller = SystemController()
        self.memory_manager = MemoryManager(config.memory_size_mb)
        
        # LangChain集成
        self.processing_chain = CodeProcessingChain()
        
        # 任务队列
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, AgentTask] = {}
        
        # 状态管理
        self.is_running = False
        self.current_context = {}
        
    async def initialize(self):
        """初始化所有组件"""
        self.logger.info("初始化Agent管理器...")
        
        # 初始化各个子系统
        await self.claude_client.initialize()
        await self.memory_manager.initialize()
        await self.system_controller.initialize()
        
        # 启动任务处理器
        self.is_running = True
        asyncio.create_task(self._task_processor())
        
    async def process_code_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理代码请求"""
        task = AgentTask(
            id=self._generate_task_id(),
            type=request.get('type', 'analyze'),
            content=request,
            priority=request.get('priority', 5)
        )
        
        await self.task_queue.put(task)
        self.active_tasks[task.id] = task
        
        # 等待任务完成
        result = await self._wait_for_task_completion(task.id)
        return result
        
    async def _task_processor(self):
        """异步任务处理器"""
        while self.is_running:
            try:
                # 从队列获取任务
                task = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                # 处理任务
                result = await self._process_task(task)
                
                # 更新任务状态
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"任务处理错误: {str(e)}")
                
    async def _process_task(self, task: AgentTask) -> Dict[str, Any]:
        """处理单个任务"""
        self.logger.info(f"处理任务: {task.id} - {task.type}")
        
        try:
            if task.type == 'analyze':
                return await self._analyze_code(task.content)
            elif task.type == 'generate':
                return await self._generate_code(task.content)
            elif task.type == 'edit':
                return await self._edit_code(task.content)
            elif task.type == 'execute':
                return await self._execute_code(task.content)
            else:
                raise ValueError(f"未知任务类型: {task.type}")
                
        except Exception as e:
            self.logger.error(f"任务处理失败: {str(e)}")
            return {"error": str(e), "task_id": task.id}
            
    async def _analyze_code(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """分析代码"""
        code = content.get('code', '')
        language = content.get('language', 'python')
        
        # 使用LangChain进行深度分析
        analysis_chain = self.processing_chain.create_analysis_chain()
        
        # 选择合适的模型
        model = self.model_selector.select_model_for_task('analyze', len(code))
        
        # 调用Claude API
        prompt = f"""
        分析以下{language}代码：
        
        ```{language}
        {code}
        ```
        
        请提供：
        1. 代码结构分析
        2. 潜在问题和改进建议
        3. 性能优化建议
        4. 安全性评估
        """
        
        response = await self.claude_client.complete(
            prompt=prompt,
            model=model,
            max_tokens=4000
        )
        
        # 解析响应
        analysis_result = self.code_analyzer.parse_analysis(response)
        
        # 存储到内存
        await self.memory_manager.store_analysis(content['code'], analysis_result)
        
        return {
            "task_type": "analyze",
            "status": "success",
            "result": analysis_result
        }
        
    async def _generate_code(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """生成代码"""
        requirements = content.get('requirements', '')
        language = content.get('language', 'python')
        context = content.get('context', {})
        
        # 构建生成链
        generation_chain = self.processing_chain.create_generation_chain()
        
        # 从内存中获取相关上下文
        relevant_context = await self.memory_manager.get_relevant_context(
            requirements, 
            limit=5
        )
        
        # 选择模型
        model = self.model_selector.select_model_for_task('generate', len(requirements))
        
        # 构建提示
        prompt = self.code_generator.build_generation_prompt(
            requirements=requirements,
            language=language,
            context=context,
            memory_context=relevant_context
        )
        
        # 调用API生成代码
        response = await self.claude_client.complete(
            prompt=prompt,
            model=model,
            max_tokens=8000,
            temperature=0.3
        )
        
        # 后处理生成的代码
        generated_code = self.code_generator.post_process(response, language)
        
        # 验证生成的代码
        validation_result = await self.code_analyzer.validate_code(
            generated_code, 
            language
        )
        
        return {
            "task_type": "generate",
            "status": "success",
            "code": generated_code,
            "validation": validation_result
        }
        
    async def _edit_code(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """编辑代码"""
        original_code = content.get('original_code', '')
        edit_instructions = content.get('instructions', '')
        language = content.get('language', 'python')
        
        # 使用特定的编辑链
        edit_chain = self.processing_chain.create_edit_chain()
        
        # 分析原始代码
        code_analysis = await self.code_analyzer.quick_analyze(original_code, language)
        
        # 构建编辑提示
        prompt = f"""
        请根据以下指令编辑代码：
        
        原始代码：
        ```{language}
        {original_code}
        ```
        
        编辑指令：
        {edit_instructions}
        
        代码分析：
        {code_analysis}
        
        请提供编辑后的完整代码。
        """
        
        # 选择合适的模型
        model = self.model_selector.select_model_for_task(
            'edit', 
            len(original_code) + len(edit_instructions)
        )
        
        # 调用API
        response = await self.claude_client.complete(
            prompt=prompt,
            model=model,
            max_tokens=8000,
            temperature=0.2
        )
        
        # 提取编辑后的代码
        edited_code = self.code_generator.extract_code_from_response(response, language)
        
        # 计算差异
        diff = self.code_analyzer.calculate_diff(original_code, edited_code)
        
        return {
            "task_type": "edit",
            "status": "success",
            "edited_code": edited_code,
            "diff": diff,
            "changes_summary": self.code_analyzer.summarize_changes(diff)
        }
        
    async def _execute_code(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """执行代码（需要谨慎处理安全性）"""
        if not self.config.enable_code_execution:
            return {
                "task_type": "execute",
                "status": "error",
                "error": "代码执行功能已禁用"
            }
            
        code = content.get('code', '')
        language = content.get('language', 'python')
        sandbox = content.get('sandbox', True)
        
        # 安全检查
        safety_check = await self.code_analyzer.security_check(code, language)
        if not safety_check['is_safe']:
            return {
                "task_type": "execute",
                "status": "error",
                "error": f"安全检查失败: {safety_check['reason']}"
            }
            
        # 在沙箱中执行
        if sandbox:
            result = await self.system_controller.execute_in_sandbox(
                code, 
                language,
                timeout=30
            )
        else:
            # 直接执行（仅限受信任的代码）
            result = await self.system_controller.execute_code(
                code,
                language
            )
            
        return {
            "task_type": "execute",
            "status": "success" if result['success'] else "error",
            "output": result.get('output', ''),
            "error": result.get('error', ''),
            "execution_time": result.get('execution_time', 0)
        }
        
    def _generate_task_id(self) -> str:
        """生成唯一的任务ID"""
        import uuid
        return f"task_{uuid.uuid4().hex[:8]}"
        
    async def _wait_for_task_completion(self, task_id: str, timeout: float = 300) -> Dict[str, Any]:
        """等待任务完成"""
        start_time = asyncio.get_event_loop().time()
        
        while task_id in self.active_tasks:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"任务 {task_id} 超时")
                
            await asyncio.sleep(0.1)
            
        # 从内存中获取结果
        return await self.memory_manager.get_task_result(task_id)
```

### 3. Claude API客户端 (api/claude_client.py)
```python
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import backoff
from utils.logger import Logger

class ClaudeClient:
    """Claude API客户端"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.logger = Logger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 速率限制
        self.rate_limiter = RateLimiter(
            requests_per_minute=50,
            tokens_per_minute=100000
        )
        
        # 请求历史
        self.request_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """初始化HTTP会话"""
        self.session = aiohttp.ClientSession(
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2025-01-01",
                "content-type": "application/json"
            }
        )
        
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def complete(
        self,
        prompt: str,
        model: str = "claude-opus-4-20250514",
        max_tokens: int = 4000,
        temperature: float = 0.2,
        system: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """调用Claude API完成请求"""
        
        # 检查速率限制
        await self.rate_limiter.acquire(len(prompt))
        
        # 构建请求体
        request_body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system:
            request_body["system"] = system
            
        if metadata:
            request_body["metadata"] = metadata
            
        # 记录请求
        request_id = self._generate_request_id()
        self._log_request(request_id, request_body)
        
        try:
            # 发送请求
            async with self.session.post(
                f"{self.base_url}/messages",
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                # 提取响应内容
                content = data.get("content", [])
                if content and isinstance(content, list):
                    text_content = " ".join(
                        item.get("text", "") 
                        for item in content 
                        if item.get("type") == "text"
                    )
                    
                    # 记录响应
                    self._log_response(request_id, text_content, data)
                    
                    return text_content
                else:
                    raise ValueError("无效的API响应格式")
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"API请求失败: {str(e)}")
            raise
            
    async def stream_complete(
        self,
        prompt: str,
        model: str = "claude-opus-4-20250514",
        max_tokens: int = 4000,
        temperature: float = 0.2,
        on_token: Optional[callable] = None
    ):
        """流式API调用"""
        await self.rate_limiter.acquire(len(prompt))
        
        request_body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        async with self.session.post(
            f"{self.base_url}/messages",
            json=request_body
        ) as response:
            response.raise_for_status()
            
            full_content = ""
            async for line in response.content:
                if line:
                    try:
                        # 解析SSE数据
                        if line.startswith(b"data: "):
                            import json
                            data = json.loads(line[6:])
                            
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                text = delta.get("text", "")
                                
                                full_content += text
                                
                                if on_token:
                                    await on_token(text)
                                    
                    except Exception as e:
                        self.logger.error(f"解析流数据错误: {str(e)}")
                        
            return full_content
            
    def _generate_request_id(self) -> str:
        """生成请求ID"""
        import uuid
        return f"req_{uuid.uuid4().hex[:12]}"
        
    def _log_request(self, request_id: str, request_body: Dict[str, Any]):
        """记录请求"""
        self.request_history.append({
            "id": request_id,
            "timestamp": datetime.now(),
            "type": "request",
            "model": request_body.get("model"),
            "prompt_length": len(str(request_body.get("messages", [])))
        })
        
    def _log_response(self, request_id: str, content: str, raw_response: Dict[str, Any]):
        """记录响应"""
        self.request_history.append({
            "id": request_id,
            "timestamp": datetime.now(),
            "type": "response",
            "content_length": len(content),
            "usage": raw_response.get("usage", {})
        })
        
    async def close(self):
        """关闭客户端"""
        if self.session:
            await self.session.close()

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times: List[datetime] = []
        self.token_usage: List[tuple[datetime, int]] = []
        self.lock = asyncio.Lock()
        
    async def acquire(self, estimated_tokens: int):
        """获取请求许可"""
        async with self.lock:
            now = datetime.now()
            
            # 清理过期记录
            cutoff_time = now - timedelta(minutes=1)
            self.request_times = [
                t for t in self.request_times 
                if t > cutoff_time
            ]
            self.token_usage = [
                (t, tokens) for t, tokens in self.token_usage 
                if t > cutoff_time
            ]
            
            # 检查请求速率
            if len(self.request_times) >= self.requests_per_minute:
                wait_time = (self.request_times[0] + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
            # 检查令牌速率
            total_tokens = sum(tokens for _, tokens in self.token_usage)
            if total_tokens + estimated_tokens > self.tokens_per_minute:
                # 等待一些令牌过期
                wait_time = (self.token_usage[0][0] + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
            # 记录新请求
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))
```

### 4. UI主窗口 (ui/main_window.py)
```python
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QStatusBar, QToolBar,
    QAction, QMessageBox, QDockWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QIcon, QKeySequence, QPalette, QColor
from ui.chat_interface import ChatInterface
from ui.code_editor import CodeEditor
from ui.file_browser import FileBrowser
from ui.output_panel import OutputPanel
from ui.theme_manager import ThemeManager
from core.agent_manager import AgentManager
import asyncio
from typing import Optional

class AsyncWorker(QThread):
    """异步任务工作线程"""
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, coro, loop):
        super().__init__()
        self.coro = coro
        self.loop = loop
        
    def run(self):
        try:
            asyncio.set_event_loop(self.loop)
            result = self.loop.run_until_complete(self.coro)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class MainWindow(QMainWindow):
    """主窗口 - 仿Claude界面设计"""
    
    def __init__(self, agent_manager: AgentManager):
        super().__init__()
        self.agent_manager = agent_manager
        self.theme_manager = ThemeManager()
        self.async_loop = asyncio.new_event_loop()
        
        self.init_ui()
        self.apply_theme()
        self.setup_shortcuts()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("CodeAgent - AI代码助手")
        self.setGeometry(100, 100, 1600, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧面板 - 文件浏览器
        self.file_browser = FileBrowser()
        self.file_browser.setMinimumWidth(200)
        self.file_browser.setMaximumWidth(400)
        
        # 中间面板 - 聊天界面和代码编辑器
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        
        # 聊天界面
        self.chat_interface = ChatInterface(self.agent_manager)
        self.chat_interface.message_sent.connect(self.on_message_sent)
        
        # 代码编辑器
        self.code_editor = CodeEditor()
        self.code_editor.code_changed.connect(self.on_code_changed)
        
        # 中间分割器
        middle_splitter = QSplitter(Qt.Orientation.Vertical)
        middle_splitter.addWidget(self.chat_interface)
        middle_splitter.addWidget(self.code_editor)
        middle_splitter.setSizes([400, 400])
        
        middle_layout.addWidget(middle_splitter)
        
        # 右侧面板 - 输出和工具
        self.output_panel = OutputPanel()
        self.output_panel.setMinimumWidth(300)
        self.output_panel.setMaximumWidth(500)
        
        # 添加到主分割器
        splitter.addWidget(self.file_browser)
        splitter.addWidget(middle_widget)
        splitter.addWidget(self.output_panel)
        splitter.setSizes([250, 900, 350])
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建工具栏
        self.create_tool_bar()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 创建停靠窗口
        self.create_dock_widgets()
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        new_action = QAction("新建(&N)", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)
        
        open_action = QAction("打开(&O)", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存(&S)", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑(&E)")
        
        undo_action = QAction("撤销(&U)", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.code_editor.undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("重做(&R)", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.code_editor.redo)
        edit_menu.addAction(redo_action)
        
        # AI菜单
        ai_menu = menubar.addMenu("AI助手(&A)")
        
        analyze_action = QAction("分析代码(&A)", self)
        analyze_action.setShortcut("Ctrl+Shift+A")
        analyze_action.triggered.connect(self.analyze_code)
        ai_menu.addAction(analyze_action)
        
        generate_action = QAction("生成代码(&G)", self)
        generate_action.setShortcut("Ctrl+Shift+G")
        generate_action.triggered.connect(self.generate_code)
        ai_menu.addAction(generate_action)
        
        optimize_action = QAction("优化代码(&O)", self)
        optimize_action.setShortcut("Ctrl+Shift+O")
        optimize_action.triggered.connect(self.optimize_code)
        ai_menu.addAction(optimize_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具(&T)")
        
        settings_action = QAction("设置(&S)", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        
        theme_menu = view_menu.addMenu("主题(&T)")
        
        light_theme = QAction("浅色主题", self)
        light_theme.triggered.connect(lambda: self.change_theme("light"))
        theme_menu.addAction(light_theme)
        
        dark_theme = QAction("深色主题", self)
        dark_theme.triggered.connect(lambda: self.change_theme("dark"))
        theme_menu.addAction(dark_theme)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_tool_bar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # 新建
        new_action = toolbar.addAction("新建")
        new_action.triggered.connect(self.new_file)
        
        # 打开
        open_action = toolbar.addAction("打开")
        open_action.triggered.connect(self.open_file)
        
        # 保存
        save_action = toolbar.addAction("保存")
        save_action.triggered.connect(self.save_file)
        
        toolbar.addSeparator()
        
        # AI功能
        analyze_action = toolbar.addAction("分析")
        analyze_action.triggered.connect(self.analyze_code)
        
        generate_action = toolbar.addAction("生成")
        generate_action.triggered.connect(self.generate_code)
        
        optimize_action = toolbar.addAction("优化")
        optimize_action.triggered.connect(self.optimize_code)
        
        toolbar.addSeparator()
        
        # 运行
        run_action = toolbar.addAction("运行")
        run_action.triggered.connect(self.run_code)
        
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 显示就绪状态
        self.status_bar.showMessage("就绪")
        
        # 添加永久部件
        self.model_label = QLabel("模型: Opus 4")
        self.status_bar.addPermanentWidget(self.model_label)
        
        self.token_label = QLabel("令牌: 0")
        self.status_bar.addPermanentWidget(self.token_label)
        
    def create_dock_widgets(self):
        """创建停靠窗口"""
        # 变量监视器
        self.var_dock = QDockWidget("变量监视器", self)
        self.var_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.var_dock)
        
        # 任务列表
        self.task_dock = QDockWidget("任务列表", self)
        self.task_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.task_dock)
        
    def apply_theme(self):
        """应用主题"""
        theme = self.theme_manager.get_current_theme()
        self.setStyleSheet(theme['stylesheet'])
        
    def setup_shortcuts(self):
        """设置快捷键"""
        # Ctrl+Enter 发送消息
        send_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        send_shortcut.activated.connect(self.chat_interface.send_message)
        
    def on_message_sent(self, message: str):
        """处理发送的消息"""
        # 创建异步任务
        async def process_message():
            result = await self.agent_manager.process_code_request({
                "type": "chat",
                "content": message
            })
            return result
            
        # 在工作线程中执行
        worker = AsyncWorker(process_message(), self.async_loop)
        worker.result_ready.connect(self.on_ai_response)
        worker.error_occurred.connect(self.on_ai_error)
        worker.start()
        
        # 更新状态
        self.status_bar.showMessage("AI正在思考...")
        
    def on_ai_response(self, result: dict):
        """处理AI响应"""
        self.chat_interface.add_ai_message(result.get('response', ''))
        self.status_bar.showMessage("就绪")
        
        # 更新令牌计数
        if 'tokens_used' in result:
            self.token_label.setText(f"令牌: {result['tokens_used']}")
            
    def on_ai_error(self, error: str):
        """处理AI错误"""
        QMessageBox.critical(self, "错误", f"AI处理失败: {error}")
        self.status_bar.showMessage("错误")
        
    def on_code_changed(self):
        """代码变化处理"""
        # 可以在这里添加实时代码分析
        pass
        
    def analyze_code(self):
        """分析当前代码"""
        code = self.code_editor.get_code()
        if not code:
            QMessageBox.information(self, "提示", "请先输入代码")
            return
            
        async def analyze():
            return await self.agent_manager.process_code_request({
                "type": "analyze",
                "code": code,
                "language": self.code_editor.get_language()
            })
            
        worker = AsyncWorker(analyze(), self.async_loop)
        worker.result_ready.connect(self.on_analysis_complete)
        worker.error_occurred.connect(self.on_ai_error)
        worker.start()
        
        self.status_bar.showMessage("正在分析代码...")
        
    def on_analysis_complete(self, result: dict):
        """分析完成"""
        analysis = result.get('result', {})
        
        # 在输出面板显示分析结果
        self.output_panel.show_analysis(analysis)
        
        # 在聊天界面显示摘要
        summary = f"代码分析完成:\n"
        summary += f"- 代码质量: {analysis.get('quality_score', 'N/A')}/10\n"
        summary += f"- 发现问题: {len(analysis.get('issues', []))}个\n"
        summary += f"- 优化建议: {len(analysis.get('suggestions', []))}个"
        
        self.chat_interface.add_ai_message(summary)
        self.status_bar.showMessage("分析完成")
        
    def generate_code(self):
        """生成代码"""
        # 显示生成对话框
        from ui.generate_dialog import GenerateDialog
        dialog = GenerateDialog(self)
        
        if dialog.exec():
            requirements = dialog.get_requirements()
            language = dialog.get_language()
            
            async def generate():
                return await self.agent_manager.process_code_request({
                    "type": "generate",
                    "requirements": requirements,
                    "language": language
                })
                
            worker = AsyncWorker(generate(), self.async_loop)
            worker.result_ready.connect(self.on_generation_complete)
            worker.error_occurred.connect(self.on_ai_error)
            worker.start()
            
            self.status_bar.showMessage("正在生成代码...")
            
    def on_generation_complete(self, result: dict):
        """生成完成"""
        code = result.get('code', '')
        validation = result.get('validation', {})
        
        # 在编辑器中显示生成的代码
        self.code_editor.set_code(code)
        
        # 显示验证结果
        if validation.get('is_valid', False):
            self.chat_interface.add_ai_message("代码生成成功！已通过验证。")
        else:
            self.chat_interface.add_ai_message(
                f"代码已生成，但存在问题:\n{validation.get('errors', '')}"
            )
            
        self.status_bar.showMessage("生成完成")
        
    def optimize_code(self):
        """优化代码"""
        code = self.code_editor.get_code()
        if not code:
            QMessageBox.information(self, "提示", "请先输入代码")
            return
            
        async def optimize():
            return await self.agent_manager.process_code_request({
                "type": "edit",
                "original_code": code,
                "instructions": "优化这段代码的性能和可读性",
                "language": self.code_editor.get_language()
            })
            
        worker = AsyncWorker(optimize(), self.async_loop)
        worker.result_ready.connect(self.on_optimization_complete)
        worker.error_occurred.connect(self.on_ai_error)
        worker.start()
        
        self.status_bar.showMessage("正在优化代码...")
        
    def on_optimization_complete(self, result: dict):
        """优化完成"""
        optimized_code = result.get('edited_code', '')
        changes_summary = result.get('changes_summary', '')
        
        # 显示优化后的代码
        self.code_editor.set_code(optimized_code)
        
        # 显示变更摘要
        self.chat_interface.add_ai_message(f"代码优化完成:\n{changes_summary}")
        self.output_panel.show_diff(result.get('diff', ''))
        
        self.status_bar.showMessage("优化完成")
        
    def run_code(self):
        """运行代码"""
        code = self.code_editor.get_code()
        if not code:
            QMessageBox.information(self, "提示", "请先输入代码")
            return
            
        async def execute():
            return await self.agent_manager.process_code_request({
                "type": "execute",
                "code": code,
                "language": self.code_editor.get_language(),
                "sandbox": True
            })
            
        worker = AsyncWorker(execute(), self.async_loop)
        worker.result_ready.connect(self.on_execution_complete)
        worker.error_occurred.connect(self.on_ai_error)
        worker.start()
        
        self.status_bar.showMessage("正在运行代码...")
        
    def on_execution_complete(self, result: dict):
        """执行完成"""
        if result.get('status') == 'success':
            output = result.get('output', '')
            execution_time = result.get('execution_time', 0)
            
            self.output_panel.show_output(output)
            self.chat_interface.add_ai_message(
                f"代码执行成功！\n执行时间: {execution_time:.3f}秒"
            )
        else:
            error = result.get('error', '未知错误')
            self.output_panel.show_error(error)
            self.chat_interface.add_ai_message(f"代码执行失败:\n{error}")
            
        self.status_bar.showMessage("执行完成")
        
    def new_file(self):
        """新建文件"""
        self.code_editor.clear()
        self.chat_interface.clear()
        self.output_panel.clear()
        self.status_bar.showMessage("新建文件")
        
    def open_file(self):
        """打开文件"""
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "打开文件",
            "",
            "所有文件 (*);;Python文件 (*.py);;JavaScript文件 (*.js)"
        )
        
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                self.code_editor.set_code(content)
                
            # 自动检测语言
            if filename.endswith('.py'):
                self.code_editor.set_language('python')
            elif filename.endswith('.js'):
                self.code_editor.set_language('javascript')
                
            self.status_bar.showMessage(f"已打开: {filename}")
            
    def save_file(self):
        """保存文件"""
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "保存文件",
            "",
            "Python文件 (*.py);;JavaScript文件 (*.js);;所有文件 (*)"
        )
        
        if filename:
            content = self.code_editor.get_code()
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.status_bar.showMessage(f"已保存: {filename}")
            
    def show_settings(self):
        """显示设置对话框"""
        from ui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self, self.agent_manager.config)
        
        if dialog.exec():
            # 应用新设置
            new_config = dialog.get_config()
            self.agent_manager.update_config(new_config)
            
    def change_theme(self, theme_name: str):
        """切换主题"""
        self.theme_manager.set_theme(theme_name)
        self.apply_theme()
        
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于CodeAgent",
            "CodeAgent - AI驱动的代码编辑助手\n\n"
            "版本: 1.0.0\n"
            "基于Claude API构建\n\n"
            "© 2025 CodeAgent Team"
        )
        
    def closeEvent(self, event):
        """关闭事件"""
        reply = QMessageBox.question(
            self,
            "确认退出",
            "确定要退出CodeAgent吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 清理资源
            self.async_loop.stop()
            event.accept()
        else:
            event.ignore()

from PyQt6.QtWidgets import QLabel  # 添加这个导入
from PyQt6.QtGui import QShortcut  # 添加这个导入
```

### 5. 配置文件 (config.yaml)
```yaml
# CodeAgent配置文件

# API配置
api:
  key: ${CLAUDE_API_KEY}  # 从环境变量读取
  base_url: https://api.anthropic.com/v1
  timeout: 120
  max_retries: 3

# 模型配置
models:
  default: claude-opus-4-20250514
  available:
    - name: claude-opus-4-20250514
      max_tokens: 100000
      cost_per_1k_tokens: 0.015
      best_for: ["complex_analysis", "large_codebases", "architecture_design"]
    
    - name: claude-sonnet-4-20250514
      max_tokens: 50000
      cost_per_1k_tokens: 0.003
      best_for: ["code_generation", "refactoring", "medium_complexity"]
    
    - name: claude-haiku-3.5-20241204
      max_tokens: 20000
      cost_per_1k_tokens: 0.0008
      best_for: ["quick_fixes", "simple_queries", "code_completion"]

# 系统配置
system:
  memory_size_mb: 2048
  cache_size_mb: 512
  max_concurrent_tasks: 10
  enable_telemetry: false
  
# 安全配置
security:
  enable_code_execution: true
  sandbox_enabled: true
  allowed_languages: ["python", "javascript", "java", "cpp", "go"]
  max_execution_time: 30
  blocked_imports: ["os", "subprocess", "eval", "exec"]
  
# UI配置
ui:
  theme: dark
  font_family: "JetBrains Mono"
  font_size: 12
  show_line_numbers: true
  enable_syntax_highlighting: true
  auto_save_interval: 60
  
# 日志配置
logging:
  level: INFO
  file: logs/codeagent.log
  max_size_mb: 100
  backup_count: 5
  
# LangChain配置
langchain:
  chunk_size: 2000
  chunk_overlap: 200
  embedding_model: text-embedding-3-large
  vector_store: faiss
  
# 计算机控制配置
computer_control:
  enabled: true
  permissions:
    screen_capture: true
    keyboard_input: true
    mouse_input: true
    file_access: true
    process_control: false
```

这个系统架构包含了：

1. **核心功能模块**：
   - Agent管理器：中央调度系统
   - 代码分析引擎：深度代码分析
   - 代码生成引擎：智能代码生成
   - 系统控制器：安全的代码执行
   - 内存管理器：高效的上下文管理

2. **API层**：
   - Claude客户端：完整的API集成
   - 模型选择器：智能选择最佳模型
   - 速率限制器：防止API超限

3. **用户界面**：
   - 仿Claude设计的现代UI
   - 实时代码编辑器
   - 智能聊天界面
   - 文件浏览器
   - 输出面板

4. **高级功能**：
   - LangChain集成进行链式处理
   - 计算机控制能力
   - 沙箱代码执行
   - 实时协作支持
   - 多模型切换

5. **生产级特性**：
   - 异步架构
   - 错误处理和重试
   - 性能监控
   - 安全沙箱
   - 完整的日志系统

这个架构可以轻松扩展到100,000行以上的代码，并提供了生产环境所需的所有功能。您可以根据具体需求继续添加更多功能模块。
