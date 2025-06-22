# LocalAgentSystem - Web Interface & API
# Web界面和API服务

import streamlit as st
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import asyncio
import json
from datetime import datetime
import threading
import queue
import gradio as gr

# ==================== API模型 ====================

class TaskRequest(BaseModel):
    """任务请求模型"""
    description: str
    task_type: Optional[str] = "auto"
    priority: Optional[int] = 5
    metadata: Optional[Dict[str, Any]] = {}


class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SystemStatus(BaseModel):
    """系统状态模型"""
    status: str
    active_agents: int
    active_tasks: int
    completed_tasks: int
    total_cost: float
    uptime: float


# ==================== FastAPI应用 ====================

import logging
from datetime import datetime
from typing import Dict, Any

class APIError(Exception):
    """自定义API异常"""
    def __init__(self, status_code: int, detail: str, error_type: str = None):
        self.status_code = status_code
        self.detail = detail
        self.error_type = error_type or "api_error"
        super().__init__(detail)

class AgentSystemAPI:
    """Agent系统API"""
    
    def __init__(self, agent_system: LocalAgentSystem):
        self.agent_system = agent_system
        
        # 配置日志
        self.logger = logging.getLogger("agent_system_api")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler("api_audit.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        self.app = FastAPI(title="Local Agent System API", version="1.0.0")
        self.setup_routes()
        self.setup_middleware()
        self.setup_exception_handlers()
        self.websocket_clients = set()
        self.task_results = {}
        
    def log_operation(self, action: str, metadata: Dict[str, Any]):
        """记录关键操作"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            **metadata
        }
        self.logger.info(log_entry)
        
    def setup_exception_handlers(self):
        """设置异常处理器"""
        @self.app.exception_handler(APIError)
        async def api_error_handler(request, exc: APIError):
            self.logger.error(f"API Error: {exc.detail}")
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.error_type,
                    "message": exc.detail,
                    "status_code": exc.status_code
                }
            )
            
        @self.app.exception_handler(Exception)
        async def generic_error_handler(request, exc: Exception):
            self.logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "status_code": 500
                }
            )
        
    def setup_middleware(self):
        """设置中间件"""
        from fastapi.security import APIKeyHeader
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        
        # API密钥认证
        API_KEYS = {"your-secret-api-key"}  # 应从环境变量获取
        api_key_scheme = APIKeyHeader(name="X-API-Key")
        
        def get_api_key(api_key: str = Depends(api_key_scheme)):
            if api_key not in API_KEYS:
                raise HTTPException(status_code=403, detail="Invalid API Key")
            return api_key
            
        # 速率限制
        self.limiter = Limiter(key_func=get_remote_address)
        
        # 安全的CORS设置
        ALLOWED_ORIGINS = [
            "http://localhost",
            "http://localhost:3000",
            "https://your-production-domain.com"
        ]
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
            expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"]
        )
        
    def setup_routes(self):
        """设置路由"""
        
        @self.app.get("/")
        @self.limiter.limit("10/minute")
        async def root(api_key: str = Depends(get_api_key)):
            try:
                self.log_operation("root_access", {"api_key": api_key})
                return {"message": "Local Agent System API", "status": "online"}
            except Exception as e:
                self.logger.error(f"Root access error: {str(e)}")
                raise APIError(500, "Internal server error")
            
        @self.app.get("/status", response_model=SystemStatus)
        @self.limiter.limit("30/minute")
        async def get_status(api_key: str = Depends(get_api_key)):
            """获取系统状态"""
            try:
                if not self.agent_system.is_initialized:
                    await self.agent_system.initialize()
                    
                orchestrator = self.agent_system.orchestrator
                cost_report = orchestrator.cost_optimizer.get_usage_report() if hasattr(orchestrator, 'cost_optimizer') else {"total_cost": 0}
                
                status = SystemStatus(
                    status="online" if self.agent_system.is_initialized else "offline",
                    active_agents=len(orchestrator.agents),
                    active_tasks=len(orchestrator.active_tasks),
                    completed_tasks=len(orchestrator.completed_tasks),
                    total_cost=cost_report["total_cost"],
                    uptime=0  # 简化示例
                )
                
                self.log_operation("get_status", {
                    "api_key": api_key,
                    "status": status.dict()
                })
                
                return status
            except Exception as e:
                self.logger.error(f"Get status error: {str(e)}")
                raise APIError(500, "Failed to get system status")
            
        @self.app.post("/tasks", response_model=TaskResponse)
        @self.limiter.limit("5/minute")
        async def create_task(
            request: TaskRequest,
            background_tasks: BackgroundTasks,
            api_key: str = Depends(get_api_key)
        ):
            """创建新任务"""
            try:
                self.log_operation("create_task_request", {
                    "api_key": api_key,
                    "task_type": request.task_type,
                    "description": request.description[:100]  # 只记录前100个字符
                })

                if not self.agent_system.is_initialized:
                    await self.agent_system.initialize()
                    
                # 验证任务类型
                ALLOWED_TASK_TYPES = ["auto", "code_generation", "research", "browser_automation", "file_operation"]
                if request.task_type not in ALLOWED_TASK_TYPES:
                    error_msg = f"Invalid task type: {request.task_type}"
                    self.logger.warning(error_msg)
                    raise APIError(400, error_msg, "invalid_task_type")
                    
                # 创建任务
                task = Task(
                    type=request.task_type,
                    description=request.description,
                    metadata=request.metadata
                )
                
                # 记录任务创建
                self.log_operation("task_created", {
                    "task_id": task.id,
                    "task_type": task.type,
                    "created_at": task.created_at.isoformat()
                })
                
                # 异步处理任务
                background_tasks.add_task(self.process_task_async, task)
                
                response = TaskResponse(
                    task_id=task.id,
                    status=task.status.value,
                    created_at=task.created_at
                )
                
                self.log_operation("task_response", {
                    "task_id": task.id,
                    "status": response.status
                })
                
                return response
            except APIError:
                raise
            except Exception as e:
                self.logger.error(f"Create task error: {str(e)}")
                raise APIError(500, "Failed to create task", "task_creation_failed")
            
        @self.app.get("/tasks/{task_id}", response_model=TaskResponse)
        async def get_task(task_id: str):
            """获取任务状态"""
            task_result = self.task_results.get(task_id)
            
            if not task_result:
                raise HTTPException(status_code=404, detail="Task not found")
                
            return TaskResponse(**task_result)
            
        @self.app.get("/agents")
        async def list_agents():
            """列出所有Agent"""
            if not self.agent_system.is_initialized:
                await self.agent_system.initialize()
                
            agents = []
            for agent_id, agent in self.agent_system.orchestrator.agents.items():
                agents.append({
                    "id": agent_id,
                    "role": agent.role.value,
                    "tools": [tool.name for tool in agent.tools],
                    "memory_size": len(agent.memory.short_term) + len(agent.memory.long_term)
                })
                
            return {"agents": agents}
            
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket连接用于实时更新"""
            client_ip = websocket.client.host if websocket.client else "unknown"
            
            # 记录连接建立
            self.log_operation("websocket_connected", {
                "client_ip": client_ip,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            await websocket.accept()
            self.websocket_clients.add(websocket)
            
            try:
                while True:
                    # 接收消息
                    message = await websocket.receive_text()
                    
                    # 记录接收到的消息
                    self.log_operation("websocket_message_received", {
                        "client_ip": client_ip,
                        "message_length": len(message),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # 保持连接
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    
            except WebSocketDisconnect as e:
                # 记录连接断开
                self.log_operation("websocket_disconnected", {
                    "client_ip": client_ip,
                    "reason": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                self.websocket_clients.remove(websocket)
                
            except Exception as e:
                # 记录错误
                self.logger.error(f"WebSocket error: {str(e)}", exc_info=True)
                self.websocket_clients.remove(websocket)
                raise
                
    async def process_task_async(self, task: Task):
        """异步处理任务"""
        try:
            result = await self.agent_system.orchestrator.process_user_request(task.description)
            
            self.task_results[task.id] = {
                "task_id": task.id,
                "status": "completed",
                "created_at": task.created_at,
                "completed_at": datetime.now(),
                "result": result,
                "error": None
            }
            
            # 通知WebSocket客户端
            await self.broadcast_update({
                "type": "task_completed",
                "task_id": task.id,
                "result": result
            })
            
        except Exception as e:
            self.task_results[task.id] = {
                "task_id": task.id,
                "status": "failed",
                "created_at": task.created_at,
                "completed_at": datetime.now(),
                "result": None,
                "error": str(e)
            }
            
            await self.broadcast_update({
                "type": "task_failed",
                "task_id": task.id,
                "error": str(e)
            })
            
    async def broadcast_update(self, message: dict):
        """广播更新到所有WebSocket客户端"""
        disconnected = set()
        
        for websocket in self.websocket_clients:
            try:
                await websocket.send_json(message)
            except:
                disconnected.add(websocket)
                
        # 移除断开的连接
        self.websocket_clients -= disconnected
        
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行API服务器"""
        uvicorn.run(self.app, host=host, port=port)


# ==================== Streamlit界面 ====================

class StreamlitUI:
    """Streamlit用户界面"""
    
    def __init__(self, agent_system: LocalAgentSystem):
        self.agent_system = agent_system
        
    def run(self):
        """运行Streamlit界面"""
        st.set_page_config(
            page_title="Local Agent System",
            page_icon="🤖",
            layout="wide"
        )
        
        # 侧边栏
        with st.sidebar:
            st.title("🤖 Local Agent System")
            st.markdown("---")
            
            # 系统状态
            if st.button("🔄 Refresh Status"):
                st.rerun()
                
            self.show_system_status()
            
            st.markdown("---")
            
            # 设置
            st.subheader("⚙️ Settings")
            max_cost = st.number_input("Max Cost per Request ($)", min_value=0.1, max_value=10.0, value=2.0)
            
        # 主界面
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Tasks", "🤖 Agents", "📈 Analytics"])
        
        with tab1:
            self.chat_interface()
            
        with tab2:
            self.task_management()
            
        with tab3:
            self.agent_overview()
            
        with tab4:
            self.analytics_dashboard()
            
    def show_system_status(self):
        """显示系统状态"""
        if not self.agent_system.is_initialized:
            st.warning("System not initialized")
            if st.button("Initialize System"):
                with st.spinner("Initializing..."):
                    asyncio.run(self.agent_system.initialize())
                st.success("System initialized!")
                st.rerun()
            return
            
        orchestrator = self.agent_system.orchestrator
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Agents", len(orchestrator.agents))
            st.metric("Active Tasks", len(orchestrator.active_tasks))
            
        with col2:
            st.metric("Completed Tasks", len(orchestrator.completed_tasks))
            cost_report = orchestrator.cost_optimizer.get_usage_report() if hasattr(orchestrator, 'cost_optimizer') else {"total_cost": 0}
            st.metric("Total Cost", f"${cost_report['total_cost']:.2f}")
            
    def chat_interface(self):
        """聊天界面"""
        st.header("💬 Chat with Agent System")
        
        # 聊天历史
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # 输入框
        if prompt := st.chat_input("What would you like me to help with?"):
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # 处理请求
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = asyncio.run(self.agent_system.process_request(prompt))
                    
                if response["success"]:
                    message = response["response"]
                    st.markdown(message)
                    
                    # 显示详细信息
                    with st.expander("Details"):
                        st.json(response.get("detailed_results", {}))
                else:
                    st.error(f"Error: {response['error']}")
                    message = f"I encountered an error: {response['error']}"
                    
            # 添加助手消息
            st.session_state.messages.append({"role": "assistant", "content": message})
            
    def task_management(self):
        """任务管理界面"""
        st.header("📊 Task Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 创建新任务
            st.subheader("Create New Task")
            
            task_type = st.selectbox(
                "Task Type",
                ["auto", "code_generation", "research", "browser_automation", "file_operation", "review"]
            )
            
            task_description = st.text_area("Task Description", height=100)
            
            if st.button("Create Task", type="primary"):
                if task_description:
                    with st.spinner("Creating task..."):
                        task = Task(type=task_type, description=task_description)
                        # 这里应该调用API创建任务
                        st.success(f"Task created with ID: {task.id}")
                else:
                    st.error("Please enter a task description")
                    
        with col2:
            # 任务统计
            st.subheader("Task Statistics")
            
            orchestrator = self.agent_system.orchestrator
            
            task_stats = {
                "Pending": sum(1 for t in orchestrator.active_tasks.values() if t.status == TaskStatus.PENDING),
                "In Progress": sum(1 for t in orchestrator.active_tasks.values() if t.status == TaskStatus.IN_PROGRESS),
                "Completed": len(orchestrator.completed_tasks),
                "Failed": sum(1 for t in orchestrator.completed_tasks.values() if t.status == TaskStatus.FAILED)
            }
            
            for status, count in task_stats.items():
                st.metric(status, count)
                
        # 任务列表
        st.subheader("Recent Tasks")
        
        # 合并活动和完成的任务
        all_tasks = list(orchestrator.active_tasks.values()) + list(orchestrator.completed_tasks.values())
        all_tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        if all_tasks:
            task_data = []
            for task in all_tasks[:10]:  # 显示最近10个
                task_data.append({
                    "ID": task.id[:8] + "...",
                    "Type": task.type,
                    "Status": task.status.value,
                    "Created": task.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "Agent": task.assigned_agent or "N/A",
                    "Description": task.description[:50] + "..."
                })
                
            st.dataframe(task_data)
        else:
            st.info("No tasks yet")
            
    def agent_overview(self):
        """Agent概览"""
        st.header("🤖 Agent Overview")
        
        if not self.agent_system.is_initialized:
            st.warning("System not initialized")
            return
            
        orchestrator = self.agent_system.orchestrator
        
        # Agent卡片
        cols = st.columns(3)
        
        for i, (agent_id, agent) in enumerate(orchestrator.agents.items()):
            col = cols[i % 3]
            
            with col:
                with st.container():
                    st.subheader(f"� {agent_id}")
                    st.text(f"Role: {agent.role.value}")
                    st.text(f"Tools: {len(agent.tools)}")
                    
                    # 记忆统计
                    short_term = len(agent.memory.short_term)
                    long_term = len(agent.memory.long_term)
                    st.text(f"Memory: {short_term} / {long_term}")
                    
                    # 工具列表
                    with st.expander("Tools"):
                        for tool in agent.tools:
                            st.text(f"• {tool.name}")
                            
    def analytics_dashboard(self):
        """分析仪表板"""
        st.header("📈 Analytics Dashboard")
        
        if not self.agent_system.is_initialized:
            st.warning("System not initialized")
            return
            
        orchestrator = self.agent_system.orchestrator
        
        # 成本分析
        if hasattr(orchestrator, 'cost_optimizer'):
            cost_report = orchestrator.cost_optimizer.get_usage_report()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Cost", f"${cost_report['total_cost']:.2f}")
                
            with col2:
                st.metric("Requests", cost_report['usage_count'])
                
            with col3:
                st.metric("Avg Cost/Request", f"${cost_report['average_cost_per_request']:.2f}")
                
            # 模型成本分布
            if cost_report['model_costs']:
                st.subheader("Cost by Model")
                
                import pandas as pd
                
                df = pd.DataFrame(
                    list(cost_report['model_costs'].items()),
                    columns=['Model', 'Cost']
                )
                
                st.bar_chart(df.set_index('Model'))
                
        # 任务完成时间分析
        st.subheader("Task Completion Times")
        
        completed_tasks = list(orchestrator.completed_tasks.values())
        if completed_tasks:
            completion_times = []
            
            for task in completed_tasks:
                if task.completed_at:
                    duration = (task.completed_at - task.created_at).total_seconds()
                    completion_times.append({
                        "Task Type": task.type,
                        "Duration (s)": duration
                    })
                    
            if completion_times:
                import pandas as pd
                
                df = pd.DataFrame(completion_times)
                avg_times = df.groupby("Task Type")["Duration (s)"].mean()
                
                st.bar_chart(avg_times)


# ==================== Gradio界面（替代方案）====================

class GradioUI:
    """Gradio用户界面 - 更简单的替代方案"""
    
    def __init__(self, agent_system: LocalAgentSystem):
        self.agent_system = agent_system
        
    def create_interface(self):
        """创建Gradio界面"""
        
        async def process_input(user_input, task_type, max_cost):
            """处理用户输入"""
            if not self.agent_system.is_initialized:
                await self.agent_system.initialize()
                
            # 设置成本限制
            if hasattr(self.agent_system.orchestrator, 'cost_optimizer'):
                self.agent_system.orchestrator.cost_optimizer.max_cost = max_cost
                
            result = await self.agent_system.process_request(user_input)
            
            if result["success"]:
                return result["response"], json.dumps(result.get("detailed_results", {}), indent=2)
            else:
                return f"Error: {result['error']}", "{}"
                
        # 创建界面
        with gr.Blocks(title="Local Agent System") as interface:
            gr.Markdown("# 🤖 Local Agent System")
            gr.Markdown("Intelligent multi-agent system for code development and automation")
            
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=2):
                        user_input = gr.Textbox(
                            label="Your Request",
                            placeholder="Describe what you want me to help with...",
                            lines=3
                        )
                        
                        task_type = gr.Dropdown(
                            choices=["auto", "code_generation", "research", "browser_automation", "file_operation"],
                            value="auto",
                            label="Task Type"
                        )
                        
                        max_cost = gr.Slider(
                            minimum=0.1,
                            maximum=10.0,
                            value=2.0,
                            step=0.1,
                            label="Max Cost ($)"
                        )
                        
                        submit_btn = gr.Button("Submit", variant="primary")
                        
                    with gr.Column(scale=3):
                        output = gr.Textbox(
                            label="Response",
                            lines=10,
                            max_lines=20
                        )
                        
                        details = gr.JSON(label="Task Details")
                        
            with gr.Tab("Examples"):
                gr.Examples(
                    examples=[
                        ["Create a Python web scraper for news headlines", "code_generation", 2.0],
                        ["Research the latest trends in AI agents", "research", 1.0],
                        ["Build a simple TODO list API with FastAPI", "code_generation", 3.0],
                        ["Automate filling a web form", "browser_automation", 1.5],
                    ],
                    inputs=[user_input, task_type, max_cost],
                )
                
            with gr.Tab("System Info"):
                system_info = gr.JSON(label="System Status")
                refresh_btn = gr.Button("Refresh")
                
                def get_system_info():
                    if not self.agent_system.is_initialized:
                        return {"status": "not initialized"}
                        
                    orchestrator = self.agent_system.orchestrator
                    
                    return {
                        "status": "online",
                        "agents": len(orchestrator.agents),
                        "active_tasks": len(orchestrator.active_tasks),
                        "completed_tasks": len(orchestrator.completed_tasks),
                        "agent_list": [
                            {
                                "id": agent_id,
                                "role": agent.role.value,
                                "tools": [tool.name for tool in agent.tools]
                            }
                            for agent_id, agent in orchestrator.agents.items()
                        ]
                    }
                    
                refresh_btn.click(get_system_info, outputs=system_info)
                
            # 事件处理
            submit_btn.click(
                process_input,
                inputs=[user_input, task_type, max_cost],
                outputs=[output, details]
            )
            
        return interface
        
    def run(self, share: bool = False):
        """运行Gradio界面"""
        interface = self.create_interface()
        interface.launch(share=share, server_name="0.0.0.0", server_port=8501)


# ==================== 主程序 ====================

async def run_api_server(agent_system: LocalAgentSystem):
    """运行API服务器"""
    api = AgentSystemAPI(agent_system)
    api.run()


def run_streamlit_ui(agent_system: LocalAgentSystem):
    """运行Streamlit界面"""
    ui = StreamlitUI(agent_system)
    ui.run()


def run_gradio_ui(agent_system: LocalAgentSystem):
    """运行Gradio界面"""
    ui = GradioUI(agent_system)
    ui.run()


# main.py
async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local Agent System")
    parser.add_argument("--mode", choices=["api", "streamlit", "gradio", "all"], default="all", help="Run mode")
    parser.add_argument("--config", default=".env", help="Config file path")
    
    args = parser.parse_args()
    
    # 加载配置
    from dotenv import load_dotenv
    load_dotenv(args.config)
    
    config = AgentSystemConfig.from_env()
    
    # 创建系统
    agent_system = LocalAgentSystem(config)
    
    # 初始化
    await agent_system.initialize()
    
    # 根据模式运行
    if args.mode == "api":
        await run_api_server(agent_system)
    elif args.mode == "streamlit":
        run_streamlit_ui(agent_system)
    elif args.mode == "gradio":
        run_gradio_ui(agent_system)
    else:  # all
        # 在不同线程中运行
        api_thread = threading.Thread(
            target=lambda: asyncio.run(run_api_server(agent_system))
        )
        api_thread.start()
        
        # 运行UI（主线程）
        run_gradio_ui(agent_system)


if __name__ == "__main__":
    asyncio.run(main())