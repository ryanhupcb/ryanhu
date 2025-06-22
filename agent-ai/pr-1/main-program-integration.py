# LocalAgentSystem - Main Program & Integration
# 主程序和系统集成

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import signal
import logging
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入所有组件
from agent_core_architecture import (
    AgentRole, TaskStatus, Task, Message, Memory,
    Tool, PythonExecutor, FileOperator, WebSearcher,
    ClaudeLLM, DeepSeekLLM, QwenLLM, AgentSystemConfig
)

from agent_system_implementation import (
    BaseAgent, ReactToTAgent, CoderAgent, ResearcherAgent,
    ReviewerAgent, MultiAgentOrchestrator, LocalAgentSystem
)

from browser_system_control import (
    BrowserController, SystemController, UIControllerAgent,
    ExecutorAgent, EnhancedMultiAgentOrchestrator,
    CostOptimizer, SmartCache, HallucinationMitigator,
    PromptOptimizer, TaskParallelizer, ErrorRecoverySystem
)

from github_integration_tools import (
    GitHubTool, CodeAnalyzer, FactChecker
)

from monitoring_observability import (
    StructuredLogger, MetricsCollector, DistributedTracer,
    HealthChecker, AuditLogger, PerformanceAnalyzer,
    MonitoringSystem
)

from advanced_tools_integration import (
    DatabaseTool, APIIntegrationTool, DocumentGeneratorTool,
    ContainerOrchestrationTool, DataProcessingTool,
    SecurityTool, NotificationTool
)

from web_interface_api import (
    AgentSystemAPI, StreamlitUI, GradioUI
)


# ==================== 增强的主系统类 ====================

class EnhancedLocalAgentSystem(LocalAgentSystem):
    """增强的本地Agent系统"""
    
    def __init__(self, config: AgentSystemConfig):
        super().__init__(config)
        self.monitoring = None
        self.cache = None
        self.error_recovery = None
        self.hallucination_mitigator = None
        self.prompt_optimizer = None
        self.task_parallelizer = None
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """初始化增强系统"""
        logger.info("Initializing Enhanced Local Agent System...")
        
        # 初始化基础系统
        await super().initialize()
        
        # 初始化监控系统
        self.monitoring = MonitoringSystem({
            "metrics_port": int(os.getenv("METRICS_PORT", "8001"))
        })
        await self.monitoring.start()
        
        # 初始化缓存
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.cache = SmartCache(redis_url)
        
        # 初始化错误恢复
        self.error_recovery = ErrorRecoverySystem()
        
        # 初始化幻觉缓解
        self.hallucination_mitigator = HallucinationMitigator()
        
        # 初始化提示优化器
        self.prompt_optimizer = PromptOptimizer()
        
        # 初始化任务并行化器
        self.task_parallelizer = TaskParallelizer(
            max_concurrent=self.config.max_concurrent_tasks
        )
        
        # 使用增强的协调器
        self._upgrade_orchestrator()
        
        # 注册额外的工具
        await self._register_additional_tools()
        
        # 注册额外的Agent
        await self._register_additional_agents()
        
        logger.info("Enhanced system initialized successfully")
        
    def _upgrade_orchestrator(self):
        """升级协调器到增强版本"""
        # 保存现有的agents
        existing_agents = self.orchestrator.agents
        
        # 创建增强协调器
        enhanced_orchestrator = EnhancedMultiAgentOrchestrator(
            self.orchestrator.orchestrator_llm
        )
        
        # 转移agents
        enhanced_orchestrator.agents = existing_agents
        
        # 添加成本优化器
        enhanced_orchestrator.cost_optimizer = CostOptimizer({
            "claude": 0.01,  # 每1K token的成本
            "qwen": 0.001,
            "deepseek": 0.0001
        })
        
        # 替换协调器
        self.orchestrator = enhanced_orchestrator
        
    async def _register_additional_tools(self):
        """注册额外的工具"""
        additional_tools = [
            GitHubTool(os.getenv("GITHUB_TOKEN")),
            CodeAnalyzer(),
            DatabaseTool(),
            APIIntegrationTool(),
            DocumentGeneratorTool(),
            ContainerOrchestrationTool(),
            DataProcessingTool(),
            SecurityTool(),
            NotificationTool(),
            BrowserController(),
            SystemController()
        ]
        
        # 将工具添加到相应的agents
        for agent in self.orchestrator.agents.values():
            if agent.role == AgentRole.CODER:
                agent.tools.extend([
                    tool for tool in additional_tools 
                    if tool.name in ["github_tool", "code_analyzer", "container_orchestration"]
                ])
            elif agent.role == AgentRole.RESEARCHER:
                agent.tools.extend([
                    tool for tool in additional_tools 
                    if tool.name in ["api_integration", "database_tool", "data_processing"]
                ])
            elif agent.role == AgentRole.EXECUTOR:
                agent.tools.extend([
                    tool for tool in additional_tools 
                    if tool.name in ["system_controller", "security_tool", "notification_tool"]
                ])
                
    async def _register_additional_agents(self):
        """注册额外的Agent"""
        # UI控制Agent
        ui_agent = UIControllerAgent(
            "ui_controller_1",
            self.orchestrator.orchestrator_llm
        )
        self.orchestrator.register_agent(ui_agent)
        
        # 执行Agent
        executor_agent = ExecutorAgent(
            "executor_1",
            self.orchestrator.orchestrator_llm
        )
        self.orchestrator.register_agent(executor_agent)
        
    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """处理用户请求 - 增强版本"""
        if not self.is_initialized:
            await self.initialize()
            
        # 记录请求
        request_id = self.monitoring.tracer.start_trace("user_request").trace_id
        
        try:
            # 优化提示
            optimized_prompt = self.prompt_optimizer.optimize_prompt(
                Task(description=user_input, type="general"),
                user_input
            )
            
            # 检查缓存
            cache_key = f"request:{user_input}"
            cached_result = await self.cache.get_or_compute(
                "request",
                {"prompt": user_input},
                lambda prompt: self._process_with_monitoring(optimized_prompt, request_id),
                ttl=3600  # 缓存1小时
            )
            
            if cached_result:
                self.monitoring.logger.log_event(
                    "cache_hit",
                    request_id=request_id,
                    prompt=user_input
                )
                return cached_result
                
            # 处理请求
            result = await self._process_with_monitoring(optimized_prompt, request_id)
            
            # 检查幻觉
            if result.get("success"):
                hallucination_check = await self.hallucination_mitigator.check_response(
                    result.get("response", ""),
                    user_input
                )
                
                if not hallucination_check["is_reliable"]:
                    self.monitoring.logger.log_event(
                        "hallucination_detected",
                        request_id=request_id,
                        confidence=hallucination_check["overall_confidence"],
                        issues=hallucination_check["problematic_claims"]
                    )
                    
            return result
            
        except Exception as e:
            self.monitoring.logger.log_error(e, {"request_id": request_id})
            
            # 尝试错误恢复
            recovery_result = await self.error_recovery.execute_with_recovery(
                self.orchestrator.process_user_request,
                user_input
            )
            
            if recovery_result["success"]:
                return recovery_result["result"]
            else:
                return {
                    "success": False,
                    "error": str(e),
                    "request_id": request_id
                }
        finally:
            self.monitoring.tracer.finish_span(
                self.monitoring.tracer.current_span_id
            )
            
    async def _process_with_monitoring(self, prompt: str, request_id: str) -> Dict[str, Any]:
        """带监控的处理流程"""
        start_time = time.time()
        
        # 性能测量
        async with self.monitoring.performance_analyzer.measure_performance("request_processing"):
            result = await self.orchestrator.process_user_request(prompt)
            
        # 记录指标
        duration = time.time() - start_time
        self.monitoring.metrics.record_task(
            task_type="user_request",
            status="success" if result.get("success") else "failed",
            duration=duration
        )
        
        # 记录成本
        if hasattr(self.orchestrator, 'cost_optimizer'):
            cost_report = self.orchestrator.cost_optimizer.get_usage_report()
            self.monitoring.metrics.update_cost("total", cost_report["total_cost"])
            
        return result
        
    async def run_background_tasks(self):
        """运行后台任务"""
        tasks = [
            self._periodic_health_check(),
            self._periodic_cleanup(),
            self._periodic_optimization()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _periodic_health_check(self):
        """定期健康检查"""
        while not self.shutdown_event.is_set():
            try:
                health_status = await self.monitoring.health_checker.get_system_health()
                
                if health_status["status"] != "healthy":
                    # 发送警报
                    notification_tool = NotificationTool()
                    await notification_tool.send_email(
                        to=os.getenv("ADMIN_EMAIL"),
                        subject="Agent System Health Alert",
                        body=f"System status: {health_status['status']}\n\nDetails: {health_status}"
                    )
                    
            except Exception as e:
                logger.error(f"Health check error: {e}")
                
            await asyncio.sleep(300)  # 5分钟检查一次
            
    async def _periodic_cleanup(self):
        """定期清理"""
        while not self.shutdown_event.is_set():
            try:
                # 清理旧任务
                cutoff_time = datetime.now().timestamp() - (7 * 24 * 3600)  # 7天前
                
                for task_id, task in list(self.orchestrator.completed_tasks.items()):
                    if task.created_at.timestamp() < cutoff_time:
                        del self.orchestrator.completed_tasks[task_id]
                        
                # 清理缓存
                self.cache.invalidate_pattern("request:*")
                
                # 垃圾回收
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                
            await asyncio.sleep(3600)  # 每小时清理一次
            
    async def _periodic_optimization(self):
        """定期优化"""
        while not self.shutdown_event.is_set():
            try:
                # 分析性能数据
                perf_analysis = self.monitoring.performance_analyzer.analyze_recent_performance(60)
                
                # 调整并发限制
                if perf_analysis.get("slow_operations"):
                    avg_duration = sum(op["duration"] for op in perf_analysis["slow_operations"]) / len(perf_analysis["slow_operations"])
                    
                    if avg_duration > 30:  # 平均超过30秒
                        # 降低并发
                        self.task_parallelizer.max_concurrent = max(1, self.task_parallelizer.max_concurrent - 1)
                    elif avg_duration < 5:  # 平均低于5秒
                        # 增加并发
                        self.task_parallelizer.max_concurrent = min(10, self.task_parallelizer.max_concurrent + 1)
                        
                # 更新提示模板
                self.prompt_optimizer._analyze_and_update_templates()
                
            except Exception as e:
                logger.error(f"Optimization error: {e}")
                
            await asyncio.sleep(1800)  # 每30分钟优化一次
            
    async def shutdown(self):
        """优雅关闭系统"""
        logger.info("Shutting down Enhanced Local Agent System...")
        
        # 设置关闭事件
        self.shutdown_event.set()
        
        # 等待活动任务完成
        active_tasks = list(self.orchestrator.active_tasks.values())
        if active_tasks:
            logger.info(f"Waiting for {len(active_tasks)} active tasks to complete...")
            await asyncio.sleep(5)  # 给任务一些时间完成
            
        # 清理资源
        if hasattr(self.orchestrator, 'browser_controller'):
            await self.orchestrator.browser_controller.cleanup()
            
        # 关闭监控
        if self.monitoring:
            # 导出最终指标
            final_metrics = self.monitoring.get_dashboard_data()
            logger.info(f"Final metrics: {final_metrics}")
            
        # 调用父类关闭
        await super().shutdown()
        
        logger.info("System shutdown complete")


# ==================== 应用程序类 ====================

class LocalAgentApplication:
    """本地Agent应用程序"""
    
    def __init__(self):
        self.config = None
        self.system = None
        self.api_server = None
        self.ui_server = None
        self.running = False
        
    async def initialize(self):
        """初始化应用程序"""
        # 加载配置
        self.config = AgentSystemConfig.from_env()
        
        # 验证配置
        if not self._validate_config():
            raise ValueError("Invalid configuration. Please check your .env file.")
            
        # 创建系统
        self.system = EnhancedLocalAgentSystem(self.config)
        await self.system.initialize()
        
        # 设置信号处理
        self._setup_signal_handlers()
        
        logger.info("Application initialized successfully")
        
    def _validate_config(self) -> bool:
        """验证配置"""
        required_keys = ["claude_api_key", "qwen_api_key"]
        
        for key in required_keys:
            if not getattr(self.config, key):
                logger.error(f"Missing required configuration: {key}")
                return False
                
        return True
        
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def run(self, mode: str = "all"):
        """运行应用程序"""
        self.running = True
        
        try:
            if mode == "cli":
                await self.run_cli()
            elif mode == "api":
                await self.run_api()
            elif mode == "ui":
                await self.run_ui()
            elif mode == "all":
                await self.run_all()
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except Exception as e:
            logger.error(f"Application error: {e}")
            raise
            
    async def run_cli(self):
        """运行CLI模式"""
        logger.info("Running in CLI mode")
        
        # 启动后台任务
        background_task = asyncio.create_task(self.system.run_background_tasks())
        
        try:
            while self.running:
                # CLI交互循环
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, input, "\n> Enter your request (or 'quit' to exit): "
                    )
                    
                    if user_input.lower() in ['quit', 'exit']:
                        break
                        
                    # 处理请求
                    print("\n🤔 Processing...")
                    result = await self.system.process_request(user_input)
                    
                    if result["success"]:
                        print(f"\n✅ Response:\n{result['response']}")
                        
                        if result.get("detailed_results"):
                            print("\n📊 Details:")
                            print(json.dumps(result["detailed_results"], indent=2))
                    else:
                        print(f"\n❌ Error: {result['error']}")
                        
                except KeyboardInterrupt:
                    break
                    
        finally:
            background_task.cancel()
            
    async def run_api(self):
        """运行API服务器"""
        logger.info("Running API server")
        
        # 创建API服务器
        self.api_server = AgentSystemAPI(self.system)
        
        # 在后台运行
        api_task = asyncio.create_task(
            asyncio.to_thread(self.api_server.run, "0.0.0.0", 8000)
        )
        
        # 运行后台任务
        background_task = asyncio.create_task(self.system.run_background_tasks())
        
        try:
            await asyncio.gather(api_task, background_task)
        except asyncio.CancelledError:
            pass
            
    async def run_ui(self):
        """运行UI服务器"""
        logger.info("Running UI server")
        
        # 使用Gradio作为默认UI
        self.ui_server = GradioUI(self.system)
        
        # 在后台运行
        ui_task = asyncio.create_task(
            asyncio.to_thread(self.ui_server.run, share=False)
        )
        
        # 运行后台任务
        background_task = asyncio.create_task(self.system.run_background_tasks())
        
        try:
            await asyncio.gather(ui_task, background_task)
        except asyncio.CancelledError:
            pass
            
    async def run_all(self):
        """运行所有服务"""
        logger.info("Running all services")
        
        # 启动所有服务
        tasks = [
            asyncio.create_task(self.run_api()),
            asyncio.create_task(self.run_ui()),
            asyncio.create_task(self.system.run_background_tasks())
        ]
        
        # 等待直到关闭
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
            
    async def shutdown(self):
        """关闭应用程序"""
        logger.info("Shutting down application...")
        
        self.running = False
        
        # 关闭系统
        if self.system:
            await self.system.shutdown()
            
        logger.info("Application shutdown complete")


# ==================== 主函数 ====================

async def async_main():
    """异步主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local Agent System")
    parser.add_argument("--mode", choices=["cli", "api", "ui", "all"], default="all", help="Run mode")
    parser.add_argument("--config", default=".env", help="Config file path")
    parser.add_argument("--init", action="store_true", help="Initialize system")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    global logger
    logger = logging.getLogger(__name__)
    
    # 初始化模式
    if args.init:
        from example_application import AgentSystemInitializer
        initializer = AgentSystemInitializer()
        await initializer.initialize_system()
        return
        
    # 加载配置
    load_dotenv(args.config)
    
    # 创建并运行应用程序
    app = LocalAgentApplication()
    
    try:
        await app.initialize()
        await app.run(args.mode)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        await app.shutdown()


def main():
    """主函数入口"""
    # Windows兼容性
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    # 运行异步主函数
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
