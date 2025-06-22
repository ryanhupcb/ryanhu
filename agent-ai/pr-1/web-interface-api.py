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

class AgentSystemAPI:
    """Agent系统API"""
    
    def __init__(self, agent_system: LocalAgentSystem):
        self.agent_system = agent_system
        self.app = FastAPI(title="Local Agent System API", version="1.0.0")
        self.setup_routes()
        self.setup_middleware()
        self.websocket_clients = set()
        self.task_results = {}
        
    def setup_middleware(self):
        """设置中间件"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def setup_routes(self):
        """设置路由"""
        
        @self.app.get("/")
        async def root():
            return {"message": "Local Agent System API", "status": "online"}
            
        @self.app.get("/status", response_model=SystemStatus)
        async def get_status():
            """获取系统状态"""
            if not self.agent_system.is_initialized:
                await self.agent_system.initialize()
                
            orchestrator = self.agent_system.orchestrator
            cost_report = orchestrator.cost_optimizer.get_usage_report() if hasattr(orchestrator, 'cost_optimizer') else {"total_cost": 0}
            
            return SystemStatus(
                status="online" if self.agent_system.is_initialized else "offline",
                active_agents=len(orchestrator.agents),
                active_tasks=len(orchestrator.active_tasks),
                completed_tasks=len(orchestrator.completed_tasks),
                total_cost=cost_report["total_cost"],
                uptime=0  # 简化示例
            )
            
        @self.app.post("/tasks", response_model=TaskResponse)
        async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
            """创建新任务"""
            if not self.agent_system.is_initialized:
                await self.agent_system.initialize()
                
            # 创建任务
            task = Task(
                type=request.task_type,
                description=request.description,
                metadata=request.metadata
            )
            
            # 异步处理任务
            background_tasks.add_task(self.process_task_async, task)
            
            return TaskResponse(
                task_id=task.id,
                status=task.status.value,
                created_at=task.created_at
            )
            
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
            await websocket.accept()
            self.websocket_clients.add(websocket)
            
            try:
                while True:
                    # 保持连接
                    await websocket.receive_text()
            except:
                self.websocket_clients.remove(websocket)
                
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
                    st.subheader(f"📍 {agent_id}")
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
