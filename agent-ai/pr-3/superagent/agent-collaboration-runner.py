# Agent Collaboration Runner
# Agent协作运行系统 - 整合所有组件实现多Agent协作

import asyncio
import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import signal

# 导入主系统和扩展模块
from complete_agent_system import (
    CompleteAgentSystem,
    AgentMessage,
    AgentCommunicationBus
)
from extended_tools import ExtendedToolManager, integrate_with_system
from system_integration import ExtendedCompleteAgentSystem, SystemAPIServer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 协作场景定义 ====================

class CollaborationScenario:
    """定义Agent协作场景"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tasks = []
        
    def add_task(self, agent: str, action: str, params: Dict[str, Any], 
                 depends_on: List[str] = None):
        """添加协作任务"""
        task = {
            'id': f"{self.name}_task_{len(self.tasks)}",
            'agent': agent,
            'action': action,
            'params': params,
            'depends_on': depends_on or [],
            'status': 'pending',
            'result': None
        }
        self.tasks.append(task)
        return task['id']

# ==================== 协作协调器 ====================

class CollaborationCoordinator:
    """协调多个Agent的协作"""
    
    def __init__(self, system: ExtendedCompleteAgentSystem):
        self.system = system
        self.running_scenarios = {}
        self.task_results = {}
        
    async def execute_scenario(self, scenario: CollaborationScenario) -> Dict[str, Any]:
        """执行协作场景"""
        logger.info(f"Starting collaboration scenario: {scenario.name}")
        
        self.running_scenarios[scenario.name] = {
            'status': 'running',
            'start_time': datetime.now(),
            'tasks': scenario.tasks.copy()
        }
        
        results = {
            'scenario': scenario.name,
            'description': scenario.description,
            'tasks': {},
            'success': True,
            'execution_time': 0
        }
        
        start_time = asyncio.get_event_loop().time()
        
        # 执行任务
        while True:
            # 找到可以执行的任务
            ready_tasks = self._find_ready_tasks(scenario)
            
            if not ready_tasks:
                # 检查是否所有任务都完成
                if all(task['status'] == 'completed' for task in scenario.tasks):
                    break
                else:
                    # 有任务失败或死锁
                    results['success'] = False
                    break
                    
            # 并行执行就绪的任务
            task_futures = []
            for task in ready_tasks:
                task['status'] = 'running'
                task_futures.append(self._execute_task(task))
                
            # 等待任务完成
            task_results = await asyncio.gather(*task_futures, return_exceptions=True)
            
            # 更新结果
            for task, result in zip(ready_tasks, task_results):
                if isinstance(result, Exception):
                    task['status'] = 'failed'
                    task['result'] = {'error': str(result)}
                    results['success'] = False
                    logger.error(f"Task {task['id']} failed: {result}")
                else:
                    task['status'] = 'completed'
                    task['result'] = result
                    self.task_results[task['id']] = result
                    
                results['tasks'][task['id']] = {
                    'agent': task['agent'],
                    'action': task['action'],
                    'status': task['status'],
                    'result': task['result']
                }
                
        results['execution_time'] = asyncio.get_event_loop().time() - start_time
        
        # 更新场景状态
        self.running_scenarios[scenario.name]['status'] = 'completed'
        self.running_scenarios[scenario.name]['end_time'] = datetime.now()
        
        logger.info(f"Scenario {scenario.name} completed. Success: {results['success']}")
        
        return results
        
    def _find_ready_tasks(self, scenario: CollaborationScenario) -> List[Dict[str, Any]]:
        """找到可以执行的任务"""
        ready_tasks = []
        
        for task in scenario.tasks:
            if task['status'] == 'pending':
                # 检查依赖是否满足
                dependencies_met = all(
                    self.task_results.get(dep_id) is not None
                    for dep_id in task['depends_on']
                )
                
                if dependencies_met:
                    ready_tasks.append(task)
                    
        return ready_tasks
        
    async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个任务"""
        agent = task['agent']
        action = task['action']
        params = task['params']
        
        # 注入依赖任务的结果
        for dep_id in task['depends_on']:
            if dep_id in self.task_results:
                params[f'dep_{dep_id}'] = self.task_results[dep_id]
                
        # 发送消息给指定Agent
        message = AgentMessage(
            sender='coordinator',
            receiver=agent,
            content={
                'action': action,
                'params': params
            }
        )
        
        await self.system.communication_bus.send_message(message)
        
        # 等待响应（简化实现）
        await asyncio.sleep(0.5)
        
        # 查找响应
        for msg in reversed(self.system.communication_bus.message_history):
            if (msg.correlation_id == message.id and 
                msg.message_type == 'response'):
                return msg.content
                
        # 如果没有响应，尝试直接调用
        if action == 'research':
            return await self.system.research_topic(
                params.get('topic'),
                params.get('depth', 'medium')
            )
        elif action == 'analyze':
            return await self.system.analyze_data(
                params.get('file_path'),
                params.get('request')
            )
        elif action == 'execute':
            return await self.system.execute_task(
                params.get('task'),
                params.get('context', {})
            )
        else:
            raise ValueError(f"Unknown action: {action}")

# ==================== 预定义协作场景 ====================

def create_research_and_report_scenario() -> CollaborationScenario:
    """创建研究并生成报告的协作场景"""
    scenario = CollaborationScenario(
        name="research_and_report",
        description="多个Agent协作研究主题并生成综合报告"
    )
    
    # 任务1: 研究Agent收集信息
    task1_id = scenario.add_task(
        agent='research_agent',
        action='research',
        params={
            'topic': 'Latest advances in quantum computing applications',
            'depth': 'deep'
        }
    )
    
    # 任务2: 代码Agent生成示例代码
    task2_id = scenario.add_task(
        agent='code_agent',
        action='generate',
        params={
            'task': 'Create a simple quantum circuit simulation using Python',
            'language': 'python'
        }
    )
    
    # 任务3: 分析Agent综合分析
    task3_id = scenario.add_task(
        agent='analysis_agent',
        action='analyze',
        params={
            'request': 'Synthesize research findings and code example into insights'
        },
        depends_on=[task1_id, task2_id]
    )
    
    return scenario

def create_data_pipeline_scenario() -> CollaborationScenario:
    """创建数据处理管道协作场景"""
    scenario = CollaborationScenario(
        name="data_pipeline",
        description="多Agent协作处理数据管道"
    )
    
    # 任务1: 搜索数据源
    task1_id = scenario.add_task(
        agent='research_agent',
        action='research',
        params={
            'topic': 'public datasets for machine learning',
            'depth': 'medium'
        }
    )
    
    # 任务2: 生成数据处理代码
    task2_id = scenario.add_task(
        agent='code_agent',
        action='generate',
        params={
            'task': 'Create data preprocessing pipeline for CSV files',
            'language': 'python'
        },
        depends_on=[task1_id]
    )
    
    # 任务3: 分析处理结果
    task3_id = scenario.add_task(
        agent='analysis_agent',
        action='analyze',
        params={
            'request': 'Analyze the data processing pipeline efficiency'
        },
        depends_on=[task2_id]
    )
    
    return scenario

# ==================== 系统启动器 ====================

class AgentSystemLauncher:
    """Agent系统启动器"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.system = None
        self.coordinator = None
        self.api_server = None
        self.extended_tools = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            'system': {
                'enable_all_frameworks': True,
                'safety_threshold': 0.95,
                'openai_model': 'gpt-4-turbo-preview',
                'anthropic_model': 'claude-3-opus-20240229'
            },
            'api_server': {
                'enabled': True,
                'port': 8000
            },
            'communication': {
                'redis': {
                    'enabled': False,
                    'host': 'localhost',
                    'port': 6379
                },
                'rabbitmq': {
                    'enabled': False,
                    'url': 'amqp://guest:guest@localhost/'
                }
            },
            'extended_tools': {
                'enabled': True,
                'standalone_port': 8001
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # 合并配置
                self._merge_config(default_config, user_config)
                
        return default_config
        
    def _merge_config(self, default: Dict, user: Dict):
        """递归合并配置"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
                
    async def launch(self):
        """启动完整的Agent系统"""
        logger.info("=== Launching Agent Collaboration System ===")
        
        # 1. 创建扩展系统
        logger.info("Creating extended agent system...")
        self.system = ExtendedCompleteAgentSystem(self.config['system'])
        
        # 2. 集成扩展工具
        if self.config['extended_tools']['enabled']:
            logger.info("Integrating extended tools...")
            self.extended_tools = integrate_with_system(self.system)
            logger.info(f"Integrated {len(self.extended_tools)} extended tools")
            
        # 3. 启动API服务器
        if self.config['api_server']['enabled']:
            logger.info("Starting API server...")
            await self.system.enable_api_server(self.config['api_server']['port'])
            
        # 4. 启用通信机制
        if self.config['communication']['redis']['enabled']:
            logger.info("Enabling Redis communication...")
            await self.system.enable_redis_communication(
                self.config['communication']['redis']
            )
            
        if self.config['communication']['rabbitmq']['enabled']:
            logger.info("Enabling RabbitMQ communication...")
            await self.system.enable_rabbitmq_communication(
                self.config['communication']['rabbitmq']['url']
            )
            
        # 5. 创建协作协调器
        logger.info("Creating collaboration coordinator...")
        self.coordinator = CollaborationCoordinator(self.system)
        
        logger.info("System launched successfully!")
        
        # 显示系统信息
        self._display_system_info()
        
    def _display_system_info(self):
        """显示系统信息"""
        print("\n" + "="*60)
        print("Agent Collaboration System - Ready")
        print("="*60)
        print(f"\nAvailable Agents:")
        for agent_id in self.system.communication_bus.agents:
            print(f"  - {agent_id}")
            
        print(f"\nAvailable Tools:")
        tools = self.system.enhanced_tool_registry.list_tools()
        for tool in tools[:10]:  # 显示前10个
            print(f"  - {tool}")
        if len(tools) > 10:
            print(f"  ... and {len(tools) - 10} more")
            
        if self.config['api_server']['enabled']:
            print(f"\nAPI Server: http://localhost:{self.config['api_server']['port']}")
            print("  - POST /chat - Chat with the system")
            print("  - POST /task - Execute tasks")
            print("  - GET /status - System status")
            print("  - GET /tools - List tools")
            
        print("\n" + "="*60 + "\n")

# ==================== 主程序 ====================

async def main():
    """主程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Agent Collaboration System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--scenario', type=str, choices=['research', 'pipeline', 'demo', 'interactive'],
                       default='interactive', help='Scenario to run')
    parser.add_argument('--api-only', action='store_true', help='Only run API server')
    
    args = parser.parse_args()
    
    # 设置环境变量（如果需要）
    # os.environ['OPENAI_API_KEY'] = 'your-key'
    # os.environ['ANTHROPIC_API_KEY'] = 'your-key'
    
    # 创建并启动系统
    launcher = AgentSystemLauncher(args.config)
    await launcher.launch()
    
    if args.api_only:
        # 只运行API服务器
        print("Running in API-only mode. Press Ctrl+C to stop.")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
            
    elif args.scenario == 'demo':
        # 运行演示场景
        print("Running demo scenarios...\n")
        
        # 场景1: 研究和报告
        scenario1 = create_research_and_report_scenario()
        result1 = await launcher.coordinator.execute_scenario(scenario1)
        print(f"\nScenario 1 Results: {json.dumps(result1, indent=2)}")
        
        # 场景2: 数据管道
        scenario2 = create_data_pipeline_scenario()
        result2 = await launcher.coordinator.execute_scenario(scenario2)
        print(f"\nScenario 2 Results: {json.dumps(result2, indent=2)}")
        
    elif args.scenario == 'research':
        # 运行研究场景
        scenario = create_research_and_report_scenario()
        result = await launcher.coordinator.execute_scenario(scenario)
        print(f"\nResearch Scenario Results: {json.dumps(result, indent=2)}")
        
    elif args.scenario == 'pipeline':
        # 运行数据管道场景
        scenario = create_data_pipeline_scenario()
        result = await launcher.coordinator.execute_scenario(scenario)
        print(f"\nPipeline Scenario Results: {json.dumps(result, indent=2)}")
        
    else:
        # 交互模式
        await interactive_mode(launcher)

async def interactive_mode(launcher: AgentSystemLauncher):
    """交互式模式"""
    print("\n=== Interactive Mode ===")
    print("Commands:")
    print("  chat <message> - Chat with the system")
    print("  task <description> - Execute a task")
    print("  scenario <name> - Run a predefined scenario")
    print("  create - Create custom scenario")
    print("  status - Show system status")
    print("  quit - Exit\n")
    
    while True:
        try:
            command = input("> ").strip()
            
            if not command:
                continue
                
            parts = command.split(' ', 1)
            cmd = parts[0].lower()
            
            if cmd == 'quit':
                break
                
            elif cmd == 'chat' and len(parts) > 1:
                response = await launcher.system.chat(parts[1])
                print(f"\nAgent: {response['response']}\n")
                
            elif cmd == 'task' and len(parts) > 1:
                result = await launcher.system.execute_task(parts[1])
                print(f"\nTask Result: Success={result['overall_success']}")
                print(f"Execution Time: {result['execution_time']:.2f}s\n")
                
            elif cmd == 'scenario' and len(parts) > 1:
                scenario_name = parts[1]
                if scenario_name == 'research':
                    scenario = create_research_and_report_scenario()
                elif scenario_name == 'pipeline':
                    scenario = create_data_pipeline_scenario()
                else:
                    print("Unknown scenario. Available: research, pipeline")
                    continue
                    
                print(f"\nExecuting scenario: {scenario.name}...")
                result = await launcher.coordinator.execute_scenario(scenario)
                print(f"Result: {json.dumps(result, indent=2)}\n")
                
            elif cmd == 'create':
                scenario = await create_custom_scenario(launcher)
                if scenario:
                    print(f"\nExecuting custom scenario...")
                    result = await launcher.coordinator.execute_scenario(scenario)
                    print(f"Result: {json.dumps(result, indent=2)}\n")
                    
            elif cmd == 'status':
                status = {
                    'agents': len(launcher.system.communication_bus.agents),
                    'tools': len(launcher.system.enhanced_tool_registry.list_tools()),
                    'running_scenarios': len(launcher.coordinator.running_scenarios)
                }
                print(f"\nSystem Status: {json.dumps(status, indent=2)}\n")
                
            else:
                print("Invalid command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit properly.")
        except Exception as e:
            print(f"Error: {e}")

async def create_custom_scenario(launcher: AgentSystemLauncher) -> Optional[CollaborationScenario]:
    """创建自定义协作场景"""
    print("\n=== Create Custom Scenario ===")
    
    name = input("Scenario name: ").strip()
    if not name:
        return None
        
    description = input("Description: ").strip()
    
    scenario = CollaborationScenario(name, description)
    
    print("\nAdd tasks (empty agent to finish):")
    task_ids = []
    
    while True:
        print(f"\nTask {len(task_ids) + 1}:")
        agent = input("  Agent (research_agent/code_agent/analysis_agent): ").strip()
        
        if not agent:
            break
            
        action = input("  Action (research/generate/analyze/execute): ").strip()
        
        # 收集参数
        params = {}
        if action == 'research':
            params['topic'] = input("  Topic: ").strip()
            params['depth'] = input("  Depth (shallow/medium/deep): ").strip() or 'medium'
        elif action == 'generate':
            params['task'] = input("  Task description: ").strip()
            params['language'] = input("  Language (python/javascript/etc): ").strip() or 'python'
        elif action == 'analyze':
            params['request'] = input("  Analysis request: ").strip()
        elif action == 'execute':
            params['task'] = input("  Task to execute: ").strip()
            
        # 依赖关系
        depends_on = []
        if task_ids:
            deps_input = input(f"  Depends on tasks (comma-separated indices, available: {list(range(len(task_ids)))}): ").strip()
            if deps_input:
                try:
                    dep_indices = [int(i.strip()) for i in deps_input.split(',')]
                    depends_on = [task_ids[i] for i in dep_indices if 0 <= i < len(task_ids)]
                except:
                    print("Invalid dependencies, ignoring...")
                    
        task_id = scenario.add_task(agent, action, params, depends_on)
        task_ids.append(task_id)
        
    if not task_ids:
        print("No tasks added.")
        return None
        
    return scenario

# ==================== 配置文件模板 ====================

def create_config_template():
    """创建配置文件模板"""
    template = {
        "system": {
            "enable_all_frameworks": True,
            "safety_threshold": 0.95,
            "openai_model": "gpt-4-turbo-preview",
            "anthropic_model": "claude-3-opus-20240229"
        },
        "api_server": {
            "enabled": True,
            "port": 8000
        },
        "communication": {
            "redis": {
                "enabled": False,
                "host": "localhost",
                "port": 6379,
                "decode_responses": True
            },
            "rabbitmq": {
                "enabled": False,
                "url": "amqp://guest:guest@localhost/"
            }
        },
        "extended_tools": {
            "enabled": True,
            "tools": {
                "stock_analysis": True,
                "email": False,
                "translation": True,
                "qrcode": True
            }
        },
        "scenarios": {
            "auto_run": [],
            "available": ["research_and_report", "data_pipeline"]
        }
    }
    
    with open("agent_config_template.json", "w") as f:
        json.dump(template, f, indent=2)
        
    print("Created configuration template: agent_config_template.json")

if __name__ == "__main__":
    # 检查是否需要创建配置模板
    if len(sys.argv) > 1 and sys.argv[1] == '--create-config':
        create_config_template()
    else:
        # 运行主程序
        asyncio.run(main())
