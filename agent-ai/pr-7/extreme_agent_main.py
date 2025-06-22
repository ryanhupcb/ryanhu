#!/usr/bin/env python3
"""
极限性能Agent系统
集成Claude API，支持AutoGen+LangChain架构
版本: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# UI框架
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import tkinter.font as tkfont

# AutoGen和LangChain
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from langchain.llms.base import LLM
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

# Claude API
import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# 其他依赖
import aiohttp
import requests
import numpy as np
import pandas as pd
from queue import Queue, PriorityQueue
import pickle
import sqlite3
import redis
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==================== 配置和常量 ====================

@dataclass
class Config:
    """系统配置"""
    app_name: str = "Extreme Performance Agent"
    version: str = "1.0.0"
    db_path: str = "agent_data.db"
    log_path: str = "agent_logs"
    cache_path: str = "agent_cache"
    models: Dict[str, str] = field(default_factory=lambda: {
        "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
        "Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
        "Claude 4 Sonnet": "claude-4-sonnet-20250514",
        "Claude 4 Opus": "claude-4-opus-20250514"
    })
    max_workers: int = 10
    max_retries: int = 3
    timeout: int = 30
    cache_size: int = 1000
    batch_size: int = 10

# ==================== 安全和加密 ====================

class SecurityManager:
    """安全管理器"""
    
    def __init__(self):
        self.key_file = Path("key.enc")
        self.salt_file = Path("salt.bin")
        self.cipher_suite = None
        
    def create_key_from_password(self, password: str) -> bytes:
        """从密码创建加密密钥"""
        if self.salt_file.exists():
            salt = self.salt_file.read_bytes()
        else:
            salt = os.urandom(16)
            self.salt_file.write_bytes(salt)
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def initialize(self, password: str):
        """初始化加密系统"""
        key = self.create_key_from_password(password)
        self.cipher_suite = Fernet(key)
        
    def encrypt_data(self, data: str) -> bytes:
        """加密数据"""
        if not self.cipher_suite:
            raise ValueError("Security not initialized")
        return self.cipher_suite.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """解密数据"""
        if not self.cipher_suite:
            raise ValueError("Security not initialized")
        return self.cipher_suite.decrypt(encrypted_data).decode()
    
    def save_api_key(self, api_key: str):
        """保存加密的API密钥"""
        encrypted = self.encrypt_data(api_key)
        self.key_file.write_bytes(encrypted)
        
    def load_api_key(self) -> Optional[str]:
        """加载并解密API密钥"""
        if not self.key_file.exists():
            return None
        encrypted = self.key_file.read_bytes()
        return self.decrypt_data(encrypted)

# ==================== 日志系统 ====================

class LogManager:
    """日志管理器"""
    
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.mkdir(exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志系统"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # 主日志
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.log_path / 'agent.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # 性能日志
        self.perf_logger = logging.getLogger('performance')
        perf_handler = logging.FileHandler(self.log_path / 'performance.log')
        perf_handler.setFormatter(logging.Formatter(log_format))
        self.perf_logger.addHandler(perf_handler)
        
        # 错误日志
        self.error_logger = logging.getLogger('errors')
        error_handler = logging.FileHandler(self.log_path / 'errors.log')
        error_handler.setFormatter(logging.Formatter(log_format))
        self.error_logger.addHandler(error_handler)

# ==================== 思维链管理 ====================

@dataclass
class ThoughtNode:
    """思维节点"""
    id: str
    content: str
    node_type: str  # 'thought', 'action', 'observation', 'result'
    timestamp: datetime
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = 'pending'  # 'pending', 'processing', 'completed', 'failed'
    editable: bool = True

class ThoughtChain:
    """思维链管理器"""
    
    def __init__(self):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_nodes: List[str] = []
        self.graph = nx.DiGraph()
        self.lock = threading.Lock()
        
    def add_node(self, content: str, node_type: str, parent_id: Optional[str] = None) -> str:
        """添加思维节点"""
        with self.lock:
            node_id = f"{node_type}_{datetime.now().timestamp()}_{secrets.token_hex(4)}"
            node = ThoughtNode(
                id=node_id,
                content=content,
                node_type=node_type,
                timestamp=datetime.now(),
                parent_id=parent_id
            )
            
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.__dict__)
            
            if parent_id:
                self.nodes[parent_id].children_ids.append(node_id)
                self.graph.add_edge(parent_id, node_id)
            else:
                self.root_nodes.append(node_id)
                
            return node_id
    
    def update_node(self, node_id: str, content: Optional[str] = None, 
                   status: Optional[str] = None, metadata: Optional[Dict] = None):
        """更新节点"""
        with self.lock:
            if node_id not in self.nodes:
                return
                
            node = self.nodes[node_id]
            if content and node.editable and node.status == 'pending':
                node.content = content
            if status:
                node.status = status
                if status in ['processing', 'completed', 'failed']:
                    node.editable = False
            if metadata:
                node.metadata.update(metadata)
                
    def get_chain_visualization(self) -> Dict[str, Any]:
        """获取思维链可视化数据"""
        with self.lock:
            return {
                'nodes': list(self.nodes.values()),
                'edges': list(self.graph.edges()),
                'layout': nx.spring_layout(self.graph) if self.graph.nodes() else {}
            }

# ==================== 任务管理 ====================

@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    description: str
    priority: int = 0
    status: str = 'pending'
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_queue = PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=Config.max_workers)
        self.lock = threading.Lock()
        
    def create_task(self, name: str, description: str, priority: int = 0) -> str:
        """创建任务"""
        with self.lock:
            task_id = f"task_{datetime.now().timestamp()}_{secrets.token_hex(4)}"
            task = Task(
                id=task_id,
                name=name,
                description=description,
                priority=priority
            )
            self.tasks[task_id] = task
            self.task_queue.put((-priority, task_id))  # 负数实现高优先级
            return task_id
    
    def update_task(self, task_id: str, **kwargs):
        """更新任务"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
    
    def execute_task(self, task_id: str, executor_func: Callable):
        """执行任务"""
        task = self.tasks.get(task_id)
        if not task or task.status != 'pending':
            return
            
        task.status = 'running'
        task.started_at = datetime.now()
        
        try:
            result = executor_func(task)
            task.result = result
            task.status = 'completed'
        except Exception as e:
            task.error = str(e)
            task.status = 'failed'
            logging.error(f"Task {task_id} failed: {e}")
        finally:
            task.completed_at = datetime.now()

# ==================== Claude API集成 ====================

class ClaudeLLM(LLM):
    """Claude LLM集成"""
    
    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.client = Anthropic(api_key=api_key)
        self.model = model
        
    @property
    def _llm_type(self) -> str:
        return "claude"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """调用Claude API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Claude API error: {e}")
            raise

class ClaudeAgent:
    """Claude Agent封装"""
    
    def __init__(self, api_key: str, model: str, thought_chain: ThoughtChain):
        self.llm = ClaudeLLM(api_key, model)
        self.thought_chain = thought_chain
        self.memory = ConversationBufferMemory()
        self.setup_chains()
        
    def setup_chains(self):
        """设置LangChain链"""
        # 思考链
        think_prompt = PromptTemplate(
            template="""Based on the following context and question, provide a detailed thought process.
            
Context: {context}
Question: {question}

Thought process:""",
            input_variables=["context", "question"]
        )
        self.think_chain = LLMChain(llm=self.llm, prompt=think_prompt)
        
        # 行动链
        action_prompt = PromptTemplate(
            template="""Based on the thought process, determine the next action.
            
Thought: {thought}
Available actions: {actions}

Next action:""",
            input_variables=["thought", "actions"]
        )
        self.action_chain = LLMChain(llm=self.llm, prompt=action_prompt)
        
        # 观察链
        observe_prompt = PromptTemplate(
            template="""Analyze the result of the action.
            
Action: {action}
Result: {result}

Observation:""",
            input_variables=["action", "result"]
        )
        self.observe_chain = LLMChain(llm=self.llm, prompt=observe_prompt)
        
    async def think(self, question: str, context: str = "") -> str:
        """思考过程"""
        thought_id = self.thought_chain.add_node(
            content=f"Thinking about: {question}",
            node_type="thought"
        )
        
        try:
            thought = await asyncio.get_event_loop().run_in_executor(
                None, self.think_chain.run, {"context": context, "question": question}
            )
            self.thought_chain.update_node(thought_id, content=thought, status="completed")
            return thought
        except Exception as e:
            self.thought_chain.update_node(thought_id, status="failed", 
                                         metadata={"error": str(e)})
            raise

# ==================== AutoGen集成 ====================

class AutoGenManager:
    """AutoGen管理器"""
    
    def __init__(self, claude_agent: ClaudeAgent):
        self.claude_agent = claude_agent
        self.agents = {}
        self.setup_agents()
        
    def setup_agents(self):
        """设置AutoGen代理"""
        # 助手代理
        self.assistant = AssistantAgent(
            name="Claude_Assistant",
            llm_config={
                "model": self.claude_agent.llm.model,
                "api_key": self.claude_agent.llm.client.api_key,
                "api_type": "claude"
            }
        )
        
        # 用户代理
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": "agent_workspace"}
        )
        
        # 专家代理
        self.code_expert = AssistantAgent(
            name="Code_Expert",
            system_message="You are a coding expert.",
            llm_config=self.assistant.llm_config
        )
        
        self.analyst = AssistantAgent(
            name="Data_Analyst",
            system_message="You are a data analysis expert.",
            llm_config=self.assistant.llm_config
        )
        
        # 群聊
        self.groupchat = GroupChat(
            agents=[self.assistant, self.code_expert, self.analyst, self.user_proxy],
            messages=[],
            max_round=50
        )
        
        self.manager = GroupChatManager(groupchat=self.groupchat, llm_config=self.assistant.llm_config)

# ==================== 性能优化 ====================

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.cache = {}
        self.cache_queue = deque(maxlen=Config.cache_size)
        
    def measure_time(self, func_name: str):
        """装饰器：测量函数执行时间"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                self.metrics[func_name].append(duration)
                logging.info(f"{func_name} took {duration:.4f} seconds")
                return result
            return wrapper
        return decorator
    
    def cache_result(self, key_prefix: str):
        """装饰器：缓存函数结果"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                cache_key = f"{key_prefix}:{str(args)}:{str(kwargs)}"
                
                if cache_key in self.cache:
                    return self.cache[cache_key]
                
                result = func(*args, **kwargs)
                self.cache[cache_key] = result
                self.cache_queue.append(cache_key)
                
                # 清理旧缓存
                if len(self.cache_queue) >= Config.cache_size:
                    old_key = self.cache_queue.popleft()
                    if old_key in self.cache:
                        del self.cache[old_key]
                
                return result
            return wrapper
        return decorator
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {}
        for func_name, times in self.metrics.items():
            if times:
                stats[func_name] = {
                    'count': len(times),
                    'avg': np.mean(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'std': np.std(times)
                }
        return stats

# ==================== 数据持久化 ====================

class DataManager:
    """数据管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.setup_database()
        
    def setup_database(self):
        """设置数据库"""
        with self.lock:
            cursor = self.conn.cursor()
            
            # 创建表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    model TEXT,
                    messages TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS thoughts (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    node_type TEXT,
                    content TEXT,
                    timestamp DATETIME,
                    parent_id TEXT,
                    status TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    priority INTEGER,
                    status TEXT,
                    created_at DATETIME,
                    completed_at DATETIME,
                    result TEXT,
                    error TEXT
                )
            ''')
            
            self.conn.commit()
    
    def save_conversation(self, conversation_id: str, model: str, messages: List[Dict], metadata: Dict):
        """保存对话"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO conversations (id, timestamp, model, messages, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                conversation_id,
                datetime.now().isoformat(),
                model,
                json.dumps(messages),
                json.dumps(metadata)
            ))
            self.conn.commit()
    
    def save_thought_chain(self, conversation_id: str, thought_chain: ThoughtChain):
        """保存思维链"""
        with self.lock:
            cursor = self.conn.cursor()
            for node in thought_chain.nodes.values():
                cursor.execute('''
                    INSERT OR REPLACE INTO thoughts 
                    (id, conversation_id, node_type, content, timestamp, parent_id, status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    node.id,
                    conversation_id,
                    node.node_type,
                    node.content,
                    node.timestamp.isoformat(),
                    node.parent_id,
                    node.status,
                    json.dumps(node.metadata)
                ))
            self.conn.commit()

# ==================== UI界面 ====================

class AgentUI:
    """主界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{Config.app_name} v{Config.version}")
        self.root.geometry("1400x900")
        
        # 系统组件
        self.security_manager = SecurityManager()
        self.log_manager = LogManager(Config.log_path)
        self.thought_chain = ThoughtChain()
        self.task_manager = TaskManager()
        self.performance_optimizer = PerformanceOptimizer()
        self.data_manager = DataManager(Config.db_path)
        
        self.claude_agent = None
        self.autogen_manager = None
        self.current_model = tk.StringVar(value=list(Config.models.keys())[0])
        
        # UI组件
        self.setup_ui()
        self.check_authentication()
        
    def setup_ui(self):
        """设置UI界面"""
        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶部工具栏
        self.setup_toolbar()
        
        # 主要区域
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板
        self.left_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_panel, weight=1)
        self.setup_left_panel()
        
        # 中间面板
        self.center_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.center_panel, weight=2)
        self.setup_center_panel()
        
        # 右侧面板
        self.right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_panel, weight=1)
        self.setup_right_panel()
        
        # 状态栏
        self.setup_statusbar()
        
    def setup_toolbar(self):
        """设置工具栏"""
        toolbar = ttk.Frame(self.main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # 模型选择
        ttk.Label(toolbar, text="Model:").pack(side=tk.LEFT, padx=5)
        model_combo = ttk.Combobox(toolbar, textvariable=self.current_model, 
                                  values=list(Config.models.keys()), state="readonly", width=20)
        model_combo.pack(side=tk.LEFT, padx=5)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # 控制按钮
        ttk.Button(toolbar, text="New Session", command=self.new_session).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Save Session", command=self.save_session).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Load Session", command=self.load_session).pack(side=tk.LEFT, padx=5)
        
        # 性能监控
        ttk.Button(toolbar, text="Performance", command=self.show_performance).pack(side=tk.RIGHT, padx=5)
        ttk.Button(toolbar, text="Settings", command=self.show_settings).pack(side=tk.RIGHT, padx=5)
        
    def setup_left_panel(self):
        """设置左侧面板 - 思维链可视化"""
        ttk.Label(self.left_panel, text="Thought Chain", font=('Arial', 12, 'bold')).pack(pady=5)
        
        # 思维链画布
        self.thought_canvas_frame = ttk.Frame(self.left_panel)
        self.thought_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(5, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.thought_canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮
        control_frame = ttk.Frame(self.left_panel)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Refresh", command=self.refresh_thought_chain).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Export", command=self.export_thought_chain).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Clear", command=self.clear_thought_chain).pack(side=tk.LEFT, padx=2)
        
    def setup_center_panel(self):
        """设置中间面板 - 主交互区"""
        # 标签页
        self.notebook = ttk.Notebook(self.center_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 对话标签页
        self.chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_frame, text="Chat")
        self.setup_chat_tab()
        
        # 任务标签页
        self.task_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.task_frame, text="Tasks")
        self.setup_task_tab()
        
        # 代码标签页
        self.code_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.code_frame, text="Code")
        self.setup_code_tab()
        
        # 分析标签页
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        self.setup_analysis_tab()
        
    def setup_chat_tab(self):
        """设置对话标签页"""
        # 对话显示区
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, height=25)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 输入区
        input_frame = ttk.Frame(self.chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD)
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 发送按钮
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message, width=10)
        send_button.pack(side=tk.RIGHT)
        
        # 绑定快捷键
        self.input_text.bind('<Control-Return>', lambda e: self.send_message())
        
    def setup_task_tab(self):
        """设置任务标签页"""
        # 任务列表
        columns = ('ID', 'Name', 'Status', 'Priority', 'Created')
        self.task_tree = ttk.Treeview(self.task_frame, columns=columns, show='tree headings', height=15)
        
        for col in columns:
            self.task_tree.heading(col, text=col)
            self.task_tree.column(col, width=100)
        
        self.task_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 任务详情
        detail_frame = ttk.LabelFrame(self.task_frame, text="Task Details")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.task_detail = tk.Text(detail_frame, height=8, wrap=tk.WORD)
        self.task_detail.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 任务控制
        control_frame = ttk.Frame(self.task_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="New Task", command=self.new_task).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Edit Task", command=self.edit_task).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Run Task", command=self.run_task).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Delete Task", command=self.delete_task).pack(side=tk.LEFT, padx=2)
        
    def setup_code_tab(self):
        """设置代码标签页"""
        # 代码编辑器
        self.code_editor = scrolledtext.ScrolledText(self.code_frame, wrap=tk.NONE, 
                                                    font=('Consolas', 10))
        self.code_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 代码控制
        control_frame = ttk.Frame(self.code_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Run Code", command=self.run_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Debug", command=self.debug_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Format", command=self.format_code).pack(side=tk.LEFT, padx=2)
        
        # 语言选择
        self.code_lang = tk.StringVar(value="Python")
        ttk.Label(control_frame, text="Language:").pack(side=tk.LEFT, padx=(20, 5))
        lang_combo = ttk.Combobox(control_frame, textvariable=self.code_lang,
                                 values=["Python", "JavaScript", "Java", "C++", "Go"],
                                 state="readonly", width=15)
        lang_combo.pack(side=tk.LEFT)
        
    def setup_analysis_tab(self):
        """设置分析标签页"""
        # 分析结果显示
        self.analysis_display = scrolledtext.ScrolledText(self.analysis_frame, wrap=tk.WORD)
        self.analysis_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 分析控制
        control_frame = ttk.Frame(self.analysis_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Analyze Data", command=self.analyze_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Generate Report", command=self.generate_report).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=2)
        
    def setup_right_panel(self):
        """设置右侧面板 - 系统信息和日志"""
        # 系统信息
        info_frame = ttk.LabelFrame(self.right_panel, text="System Info")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_display = tk.Text(info_frame, height=10, wrap=tk.WORD)
        self.info_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 日志显示
        log_frame = ttk.LabelFrame(self.right_panel, text="Logs")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_display = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 日志控制
        log_control = ttk.Frame(log_frame)
        log_control.pack(fill=tk.X, padx=5, pady=5)
        
        self.log_level = tk.StringVar(value="INFO")
        ttk.Label(log_control, text="Level:").pack(side=tk.LEFT)
        level_combo = ttk.Combobox(log_control, textvariable=self.log_level,
                                  values=["DEBUG", "INFO", "WARNING", "ERROR"],
                                  state="readonly", width=10)
        level_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(log_control, text="Clear", command=self.clear_logs).pack(side=tk.RIGHT)
        
    def setup_statusbar(self):
        """设置状态栏"""
        self.statusbar = ttk.Frame(self.root)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(self.statusbar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.progress_bar = ttk.Progressbar(self.statusbar, mode='indeterminate', length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        
    def check_authentication(self):
        """检查认证"""
        auth_window = tk.Toplevel(self.root)
        auth_window.title("Authentication")
        auth_window.geometry("400x200")
        auth_window.transient(self.root)
        auth_window.grab_set()
        
        # 居中显示
        auth_window.update_idletasks()
        x = (auth_window.winfo_screenwidth() // 2) - (auth_window.winfo_width() // 2)
        y = (auth_window.winfo_screenheight() // 2) - (auth_window.winfo_height() // 2)
        auth_window.geometry(f"+{x}+{y}")
        
        # 密码输入
        ttk.Label(auth_window, text="Please enter password to protect your API key:",
                 font=('Arial', 10)).pack(pady=20)
        
        password_var = tk.StringVar()
        password_entry = ttk.Entry(auth_window, textvariable=password_var, show='*', width=30)
        password_entry.pack(pady=10)
        password_entry.focus()
        
        # API Key输入
        api_key_var = tk.StringVar()
        api_frame = ttk.Frame(auth_window)
        
        def show_api_input():
            api_frame.pack(pady=10)
            ttk.Label(api_frame, text="API Key:").pack(side=tk.LEFT)
            ttk.Entry(api_frame, textvariable=api_key_var, width=40).pack(side=tk.LEFT, padx=5)
        
        # 按钮
        button_frame = ttk.Frame(auth_window)
        button_frame.pack(pady=20)
        
        def authenticate():
            password = password_var.get()
            if not password:
                messagebox.showerror("Error", "Password is required")
                return
                
            try:
                self.security_manager.initialize(password)
                
                # 尝试加载已保存的API Key
                saved_key = self.security_manager.load_api_key()
                if saved_key:
                    self.initialize_agents(saved_key)
                    auth_window.destroy()
                else:
                    show_api_input()
                    
            except Exception as e:
                messagebox.showerror("Error", "Invalid password")
        
        def save_and_continue():
            api_key = api_key_var.get()
            if not api_key:
                messagebox.showerror("Error", "API key is required")
                return
                
            self.security_manager.save_api_key(api_key)
            self.initialize_agents(api_key)
            auth_window.destroy()
        
        ttk.Button(button_frame, text="Login", command=authenticate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save & Continue", command=save_and_continue).pack(side=tk.LEFT, padx=5)
        
        # 绑定回车键
        password_entry.bind('<Return>', lambda e: authenticate())
        
    def initialize_agents(self, api_key: str):
        """初始化代理"""
        try:
            model = Config.models[self.current_model.get()]
            self.claude_agent = ClaudeAgent(api_key, model, self.thought_chain)
            self.autogen_manager = AutoGenManager(self.claude_agent)
            self.update_status("Agents initialized successfully")
            self.update_system_info()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize agents: {e}")
            
    def on_model_change(self, event=None):
        """模型切换处理"""
        if self.claude_agent:
            model = Config.models[self.current_model.get()]
            self.claude_agent.llm.model = model
            self.update_status(f"Switched to model: {self.current_model.get()}")
            
    def send_message(self):
        """发送消息"""
        message = self.input_text.get("1.0", tk.END).strip()
        if not message:
            return
            
        if not self.claude_agent:
            messagebox.showerror("Error", "Please authenticate first")
            return
            
        # 清空输入框
        self.input_text.delete("1.0", tk.END)
        
        # 显示用户消息
        self.chat_display.insert(tk.END, f"\nUser: {message}\n", "user")
        self.chat_display.see(tk.END)
        
        # 异步处理消息
        self.process_message_async(message)
        
    def process_message_async(self, message: str):
        """异步处理消息"""
        def process():
            try:
                self.update_status("Processing...")
                self.progress_bar.start()
                
                # 思考过程
                thought = asyncio.run(self.claude_agent.think(message))
                
                # 生成响应
                response = self.claude_agent.llm._call(message)
                
                # 显示响应
                self.root.after(0, self.display_response, response)
                
                # 更新思维链
                self.root.after(0, self.refresh_thought_chain)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, self.progress_bar.stop)
                self.root.after(0, lambda: self.update_status("Ready"))
        
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()
        
    def display_response(self, response: str):
        """显示响应"""
        self.chat_display.insert(tk.END, f"\nAssistant: {response}\n", "assistant")
        self.chat_display.see(tk.END)
        
    def refresh_thought_chain(self):
        """刷新思维链可视化"""
        try:
            viz_data = self.thought_chain.get_chain_visualization()
            
            self.ax.clear()
            
            if viz_data['nodes']:
                # 创建节点颜色映射
                color_map = {
                    'thought': 'lightblue',
                    'action': 'lightgreen',
                    'observation': 'lightyellow',
                    'result': 'lightcoral'
                }
                
                # 绘制节点
                for node in viz_data['nodes']:
                    if node.id in viz_data['layout']:
                        pos = viz_data['layout'][node.id]
                        color = color_map.get(node.node_type, 'lightgray')
                        
                        # 绘制节点
                        circle = plt.Circle(pos, 0.1, color=color, ec='black', linewidth=2)
                        self.ax.add_patch(circle)
                        
                        # 添加标签
                        label = node.content[:20] + '...' if len(node.content) > 20 else node.content
                        self.ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=8)
                
                # 绘制边
                for edge in viz_data['edges']:
                    if edge[0] in viz_data['layout'] and edge[1] in viz_data['layout']:
                        pos1 = viz_data['layout'][edge[0]]
                        pos2 = viz_data['layout'][edge[1]]
                        self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', alpha=0.5)
                
                self.ax.set_xlim(-1.5, 1.5)
                self.ax.set_ylim(-1.5, 1.5)
                self.ax.axis('off')
            
            self.canvas.draw()
            
        except Exception as e:
            logging.error(f"Error refreshing thought chain: {e}")
            
    def new_task(self):
        """创建新任务"""
        dialog = tk.Toplevel(self.root)
        dialog.title("New Task")
        dialog.geometry("400x300")
        
        # 任务名称
        ttk.Label(dialog, text="Task Name:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var, width=40).grid(row=0, column=1, padx=10, pady=5)
        
        # 任务描述
        ttk.Label(dialog, text="Description:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.NW)
        desc_text = tk.Text(dialog, width=40, height=10)
        desc_text.grid(row=1, column=1, padx=10, pady=5)
        
        # 优先级
        ttk.Label(dialog, text="Priority:").grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        priority_var = tk.IntVar(value=0)
        ttk.Spinbox(dialog, from_=0, to=10, textvariable=priority_var, width=10).grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
        
        # 按钮
        def create_task():
            name = name_var.get()
            desc = desc_text.get("1.0", tk.END).strip()
            priority = priority_var.get()
            
            if name and desc:
                task_id = self.task_manager.create_task(name, desc, priority)
                self.update_task_list()
                dialog.destroy()
                self.update_status(f"Task created: {task_id}")
            else:
                messagebox.showerror("Error", "Please fill all fields")
        
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Create", command=create_task).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
    def update_task_list(self):
        """更新任务列表"""
        # 清空列表
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
        
        # 添加任务
        for task in self.task_manager.tasks.values():
            self.task_tree.insert('', 'end', values=(
                task.id,
                task.name,
                task.status,
                task.priority,
                task.created_at.strftime("%Y-%m-%d %H:%M")
            ))
            
    def edit_task(self):
        """编辑任务"""
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a task to edit")
            return
            
        task_id = self.task_tree.item(selection[0])['values'][0]
        task = self.task_manager.tasks.get(task_id)
        
        if task and task.status == 'pending':
            # 在任务详情中显示可编辑内容
            self.task_detail.delete("1.0", tk.END)
            self.task_detail.insert("1.0", task.description)
            self.task_detail.config(state=tk.NORMAL)
            messagebox.showinfo("Info", "You can now edit the task description")
        else:
            messagebox.showwarning("Warning", "Only pending tasks can be edited")
            
    def run_task(self):
        """运行任务"""
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a task to run")
            return
            
        task_id = self.task_tree.item(selection[0])['values'][0]
        
        def task_executor(task):
            # 这里实现实际的任务执行逻辑
            # 可以调用Claude Agent来处理任务
            return f"Task {task.name} completed"
        
        self.task_manager.execute_task(task_id, task_executor)
        self.update_task_list()
        
    def delete_task(self):
        """删除任务"""
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a task to delete")
            return
            
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this task?"):
            task_id = self.task_tree.item(selection[0])['values'][0]
            if task_id in self.task_manager.tasks:
                del self.task_manager.tasks[task_id]
                self.update_task_list()
                
    def run_code(self):
        """运行代码"""
        code = self.code_editor.get("1.0", tk.END)
        language = self.code_lang.get()
        
        if not code.strip():
            messagebox.showwarning("Warning", "No code to run")
            return
            
        # 这里可以集成代码执行功能
        # 对于Python代码，可以使用exec()
        # 对于其他语言，可以调用相应的编译器/解释器
        
        self.update_status(f"Running {language} code...")
        
    def debug_code(self):
        """调试代码"""
        # 实现代码调试功能
        messagebox.showinfo("Info", "Debug feature coming soon")
        
    def format_code(self):
        """格式化代码"""
        # 实现代码格式化功能
        code = self.code_editor.get("1.0", tk.END)
        language = self.code_lang.get()
        
        # 根据语言调用相应的格式化工具
        # 例如：Python使用black，JavaScript使用prettier等
        
        self.update_status("Code formatted")
        
    def analyze_data(self):
        """分析数据"""
        # 实现数据分析功能
        self.analysis_display.delete("1.0", tk.END)
        self.analysis_display.insert("1.0", "Starting data analysis...\n")
        
        # 可以集成pandas、numpy等进行数据分析
        
    def generate_report(self):
        """生成报告"""
        # 实现报告生成功能
        report = "Analysis Report\n" + "="*50 + "\n"
        report += f"Generated at: {datetime.now()}\n\n"
        
        # 添加分析结果
        
        self.analysis_display.insert(tk.END, report)
        
    def export_results(self):
        """导出结果"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            content = self.analysis_display.get("1.0", tk.END)
            with open(filename, 'w') as f:
                f.write(content)
            self.update_status(f"Results exported to {filename}")
            
    def export_thought_chain(self):
        """导出思维链"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            data = {
                'nodes': [node.__dict__ for node in self.thought_chain.nodes.values()],
                'edges': list(self.thought_chain.graph.edges())
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.update_status(f"Thought chain exported to {filename}")
            
    def clear_thought_chain(self):
        """清空思维链"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the thought chain?"):
            self.thought_chain = ThoughtChain()
            if self.claude_agent:
                self.claude_agent.thought_chain = self.thought_chain
            self.refresh_thought_chain()
            
    def show_performance(self):
        """显示性能统计"""
        stats = self.performance_optimizer.get_performance_stats()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Performance Statistics")
        dialog.geometry("600x400")
        
        # 创建表格显示性能数据
        columns = ('Function', 'Count', 'Avg Time', 'Min Time', 'Max Time', 'Std Dev')
        tree = ttk.Treeview(dialog, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        for func_name, data in stats.items():
            tree.insert('', 'end', values=(
                func_name,
                data['count'],
                f"{data['avg']:.4f}s",
                f"{data['min']:.4f}s",
                f"{data['max']:.4f}s",
                f"{data['std']:.4f}s"
            ))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
        
    def show_settings(self):
        """显示设置"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("500x400")
        
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # API设置
        api_frame = ttk.Frame(notebook)
        notebook.add(api_frame, text="API")
        
        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        api_entry = ttk.Entry(api_frame, width=40, show='*')
        api_entry.grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Button(api_frame, text="Update API Key", 
                  command=lambda: self.update_api_key(api_entry.get())).grid(row=1, column=1, padx=10, pady=5)
        
        # 性能设置
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Performance")
        
        ttk.Label(perf_frame, text="Max Workers:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        workers_var = tk.IntVar(value=Config.max_workers)
        ttk.Spinbox(perf_frame, from_=1, to=20, textvariable=workers_var, width=10).grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        
        ttk.Label(perf_frame, text="Cache Size:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        cache_var = tk.IntVar(value=Config.cache_size)
        ttk.Spinbox(perf_frame, from_=100, to=10000, increment=100, textvariable=cache_var, width=10).grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
        
        # 保存按钮
        ttk.Button(dialog, text="Save", command=dialog.destroy).pack(pady=10)
        
    def update_api_key(self, new_key: str):
        """更新API密钥"""
        if new_key:
            self.security_manager.save_api_key(new_key)
            self.initialize_agents(new_key)
            messagebox.showinfo("Success", "API key updated successfully")
            
    def new_session(self):
        """新建会话"""
        if messagebox.askyesno("Confirm", "Start a new session? Current session will be saved."):
            self.save_session()
            self.chat_display.delete("1.0", tk.END)
            self.clear_thought_chain()
            self.update_status("New session started")
            
    def save_session(self):
        """保存会话"""
        try:
            conversation_id = f"conv_{datetime.now().timestamp()}"
            messages = []  # 从chat_display提取消息
            
            self.data_manager.save_conversation(
                conversation_id,
                self.current_model.get(),
                messages,
                {"timestamp": datetime.now().isoformat()}
            )
            
            self.data_manager.save_thought_chain(conversation_id, self.thought_chain)
            
            self.update_status("Session saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save session: {e}")
            
    def load_session(self):
        """加载会话"""
        # 实现会话加载功能
        messagebox.showinfo("Info", "Load session feature coming soon")
        
    def clear_logs(self):
        """清空日志"""
        self.log_display.delete("1.0", tk.END)
        
    def update_status(self, message: str):
        """更新状态栏"""
        self.status_label.config(text=message)
        
        # 同时更新日志
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_display.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_display.see(tk.END)
        
    def update_system_info(self):
        """更新系统信息"""
        info = f"Model: {self.current_model.get()}\n"
        info += f"Agent Status: {'Active' if self.claude_agent else 'Inactive'}\n"
        info += f"Tasks: {len(self.task_manager.tasks)}\n"
        info += f"Thought Nodes: {len(self.thought_chain.nodes)}\n"
        info += f"Cache Size: {len(self.performance_optimizer.cache)}\n"
        
        self.info_display.delete("1.0", tk.END)
        self.info_display.insert("1.0", info)
        
        # 定时更新
        self.root.after(5000, self.update_system_info)
        
    def run(self):
        """运行应用"""
        self.root.mainloop()

# ==================== 主程序入口 ====================

def main():
    """主函数"""
    # 创建必要的目录
    for path in [Config.log_path, Config.cache_path]:
        Path(path).mkdir(exist_ok=True)
    
    # 启动应用
    app = AgentUI()
    app.run()

if __name__ == "__main__":
    main()
