# System Integration Extension
# 主系统集成扩展 - 处理外部工具的注册和通信

import asyncio
from typing import Dict, List, Any, Optional, Callable
from aiohttp import web
import json
import redis
import aio_pika
from dataclasses import dataclass
import logging
from datetime import datetime

# 假设主系统模块可以导入
from complete_agent_system import (
    CompleteAgentSystem, 
    Tool, 
    AgentMessage,
    AgentCommunicationBus
)

logger = logging.getLogger(__name__)

# ==================== 远程工具代理 ====================

class RemoteToolProxy(Tool):
    """远程工具的本地代理"""
    
    def __init__(self, name: str, description: str, schema: Dict[str, Any], 
                 endpoint: str, timeout: int = 30):
        super().__init__(name, description)
        self.schema = schema
        self.endpoint = endpoint
        self.timeout = timeout
        self.session = None
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """通过网络调用远程工具"""
        import aiohttp
        
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        try:
            async with self.session.post(
                self.endpoint,
                json=kwargs,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                result = await response.json()
                
                self.usage_count += 1
                if result.get('success', False):
                    self.success_count += 1
                    
                return result
                
        except asyncio.TimeoutError:
            self.usage_count += 1
            return {
                'success': False,
                'error': f'Remote tool execution timed out after {self.timeout}s'
            }
        except Exception as e:
            self.usage_count += 1
            logger.error(f"Remote tool execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def get_schema(self) -> Dict[str, Any]:
        """返回工具架构"""
        return self.schema
        
    def _get_parameters(self) -> Dict[str, Any]:
        """从架构中提取参数"""
        return self.schema.get('function', {}).get('parameters', {})

# ==================== 通信处理器 ====================

class CommunicationHandler:
    """处理不同通信方式的处理器"""
    
    def __init__(self, system: CompleteAgentSystem):
        self.system = system
        self.handlers = {
            'tool_registration': self.handle_tool_registration,
            'tool_execution': self.handle_tool_execution,
            'agent_message': self.handle_agent_message,
            'system_query': self.handle_system_query
        }
        
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理传入的消息"""
        msg_type = message.get('type')
        handler = self.handlers.get(msg_type)
        
        if handler:
            return await handler(message)
        else:
            return {
                'success': False,
                'error': f'Unknown message type: {msg_type}'
            }
            
    async def handle_tool_registration(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具注册"""
        try:
            # 创建远程工具代理
            proxy = RemoteToolProxy(
                name=message['tool_name'],
                description=message['tool_description'],
                schema=message['tool_schema'],
                endpoint=message['callback_url']
            )
            
            # 注册到系统
            self.system.enhanced_tool_registry.register_tool_instance(proxy)
            
            logger.info(f"Registered remote tool: {message['tool_name']}")
            
            return {
                'success': True,
                'message': f"Tool {message['tool_name']} registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Tool registration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def handle_tool_execution(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具执行请求"""
        tool_name = message.get('tool_name')
        parameters = message.get('parameters', {})
        
        tool = self.system.enhanced_tool_registry.get_tool(tool_name)
        if not tool:
            return {
                'success': False,
                'error': f'Tool {tool_name} not found'
            }
            
        try:
            result = await tool.execute(**parameters)
            return result
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    async def handle_agent_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理代理消息"""
        agent_msg = AgentMessage(
            sender=message.get('sender', 'external'),
            receiver=message.get('receiver'),
            content=message.get('content'),
            message_type=message.get('message_type', 'request')
        )
        
        await self.system.communication_bus.send_message(agent_msg)
        
        return {
            'success': True,
            'message_id': agent_msg.id
        }
        
    async def handle_system_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理系统查询"""
        query_type = message.get('query_type')
        
        if query_type == 'status':
            return {
                'success': True,
                'status': {
                    'active_agents': len(self.system.communication_bus.agents),
                    'available_tools': self.system.enhanced_tool_registry.list_tools(),
                    'message_queue_size': self.system.communication_bus.message_queue.qsize()
                }
            }
        elif query_type == 'tool_stats':
            tool_stats = {}
            for tool_name in self.system.enhanced_tool_registry.list_tools():
                tool = self.system.enhanced_tool_registry.get_tool(tool_name)
                if tool:
                    tool_stats[tool_name] = {
                        'usage_count': tool.usage_count,
                        'success_count': tool.success_count,
                        'success_rate': tool.success_count / tool.usage_count if tool.usage_count > 0 else 0
                    }
            return {
                'success': True,
                'tool_stats': tool_stats
            }
        else:
            return {
                'success': False,
                'error': f'Unknown query type: {query_type}'
            }

# ==================== HTTP API 服务器 ====================

class SystemAPIServer:
    """系统 API 服务器"""
    
    def __init__(self, system: CompleteAgentSystem, port: int = 8000):
        self.system = system
        self.port = port
        self.app = web.Application()
        self.communication_handler = CommunicationHandler(system)
        self._setup_routes()
        
    def _setup_routes(self):
        """设置 API 路由"""
        self.app.router.add_post('/message', self.handle_message)
        self.app.router.add_post('/register_tool', self.handle_tool_registration)
        self.app.router.add_post('/execute_tool/{tool_name}', self.handle_tool_execution)
        self.app.router.add_post('/chat', self.handle_chat)
        self.app.router.add_post('/task', self.handle_task)
        self.app.router.add_get('/status', self.handle_status)
        self.app.router.add_get('/tools', self.handle_list_tools)
        
    async def start(self):
        """启动 API 服务器"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        logger.info(f"System API server started on port {self.port}")
        
    async def handle_message(self, request):
        """处理通用消息"""
        try:
            message = await request.json()
            result = await self.communication_handler.process_message(message)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    async def handle_tool_registration(self, request):
        """处理工具注册"""
        try:
            data = await request.json()
            message = {
                'type': 'tool_registration',
                **data
            }
            result = await self.communication_handler.process_message(message)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    async def handle_tool_execution(self, request):
        """处理工具执行"""
        tool_name = request.match_info['tool_name']
        try:
            parameters = await request.json()
            message = {
                'type': 'tool_execution',
                'tool_name': tool_name,
                'parameters': parameters
            }
            result = await self.communication_handler.process_message(message)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    async def handle_chat(self, request):
        """处理聊天请求"""
        try:
            data = await request.json()
            result = await self.system.chat(
                data['message'],
                data.get('conversation_id')
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    async def handle_task(self, request):
        """处理任务执行请求"""
        try:
            data = await request.json()
            result = await self.system.execute_task(
                data['task'],
                data.get('context', {})
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    async def handle_status(self, request):
        """获取系统状态"""
        message = {
            'type': 'system_query',
            'query_type': 'status'
        }
        result = await self.communication_handler.process_message(message)
        return web.json_response(result)
        
    async def handle_list_tools(self, request):
        """列出所有工具"""
        tools = {}
        for tool_name in self.system.enhanced_tool_registry.list_tools():
            tool = self.system.enhanced_tool_registry.get_tool(tool_name)
            if tool:
                tools[tool_name] = {
                    'description': tool.description,
                    'schema': tool.get_schema(),
                    'is_remote': isinstance(tool, RemoteToolProxy)
                }
        return web.json_response(tools)

# ==================== Redis 通信适配器 ====================

class RedisAdapter:
    """Redis 发布/订阅适配器"""
    
    def __init__(self, system: CompleteAgentSystem, redis_config: Dict[str, Any]):
        self.system = system
        self.redis_client = redis.Redis(**redis_config)
        self.pubsub = self.redis_client.pubsub()
        self.communication_handler = CommunicationHandler(system)
        
    async def start(self):
        """启动 Redis 监听"""
        self.pubsub.subscribe('agent_system_messages')
        
        # 在后台线程中监听消息
        asyncio.create_task(self._listen_messages())
        
        logger.info("Redis adapter started")
        
    async def _listen_messages(self):
        """监听 Redis 消息"""
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    result = await self.communication_handler.process_message(data)
                    
                    # 发送响应
                    response_channel = data.get('response_channel')
                    if response_channel:
                        self.redis_client.publish(
                            response_channel,
                            json.dumps(result)
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to process Redis message: {e}")

# ==================== RabbitMQ 通信适配器 ====================

class RabbitMQAdapter:
    """RabbitMQ 消息队列适配器"""
    
    def __init__(self, system: CompleteAgentSystem, amqp_url: str):
        self.system = system
        self.amqp_url = amqp_url
        self.connection = None
        self.channel = None
        self.communication_handler = CommunicationHandler(system)
        
    async def start(self):
        """启动 RabbitMQ 连接"""
        self.connection = await aio_pika.connect_robust(self.amqp_url)
        self.channel = await self.connection.channel()
        
        # 声明队列
        queue = await self.channel.declare_queue('agent_system_queue', durable=True)
        
        # 开始消费消息
        await queue.consume(self._process_message)
        
        logger.info("RabbitMQ adapter started")
        
    async def _process_message(self, message: aio_pika.IncomingMessage):
        """处理 RabbitMQ 消息"""
        async with message.process():
            try:
                data = json.loads(message.body.decode())
                result = await self.communication_handler.process_message(data)
                
                # 如果有回复队列，发送响应
                if message.reply_to:
                    await self.channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(result).encode(),
                            correlation_id=message.correlation_id
                        ),
                        routing_key=message.reply_to
                    )
                    
            except Exception as e:
                logger.error(f"Failed to process RabbitMQ message: {e}")

# ==================== 扩展的完整系统 ====================

class ExtendedCompleteAgentSystem(CompleteAgentSystem):
    """支持外部通信的扩展系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 通信适配器
        self.api_server = None
        self.redis_adapter = None
        self.rabbitmq_adapter = None
        
    async def enable_api_server(self, port: int = 8000):
        """启用 HTTP API 服务器"""
        self.api_server = SystemAPIServer(self, port)
        await self.api_server.start()
        
    async def enable_redis_communication(self, redis_config: Dict[str, Any]):
        """启用 Redis 通信"""
        self.redis_adapter = RedisAdapter(self, redis_config)
        await self.redis_adapter.start()
        
    async def enable_rabbitmq_communication(self, amqp_url: str):
        """启用 RabbitMQ 通信"""
        self.rabbitmq_adapter = RabbitMQAdapter(self, amqp_url)
        await self.rabbitmq_adapter.start()
        
    async def import_tools_from_module(self, module_path: str):
        """从外部模块导入工具"""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("extended_tools", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 集成工具
        if hasattr(module, 'integrate_with_system'):
            tools = module.integrate_with_system(self)
            logger.info(f"Imported {len(tools)} tools from {module_path}")
            
# ==================== 使用示例 ====================

async def demo_extended_system():
    """演示扩展系统功能"""
    
    # 创建扩展系统
    config = {
        'enable_all_frameworks': True,
        'safety_threshold': 0.95
    }
    
    system = ExtendedCompleteAgentSystem(config)
    
    print("=== Extended System Demo ===\n")
    
    # 1. 启动 API 服务器
    print("1. Starting API server...")
    await system.enable_api_server(port=8000)
    print("API server started on http://localhost:8000\n")
    
    # 2. 启用 Redis 通信（如果 Redis 可用）
    try:
        print("2. Enabling Redis communication...")
        await system.enable_redis_communication({
            'host': 'localhost',
            'port': 6379,
            'decode_responses': True
        })
        print("Redis communication enabled\n")
    except:
        print("Redis not available, skipping\n")
        
    # 3. 导入外部工具
    print("3. Importing external tools...")
    # await system.import_tools_from_module("./extended_tools.py")
    
    # 4. 测试远程工具注册
    print("4. Testing remote tool registration...")
    handler = CommunicationHandler(system)
    result = await handler.handle_tool_registration({
        'tool_name': 'remote_calculator',
        'tool_description': 'A remote calculator tool',
        'tool_schema': {
            'type': 'function',
            'function': {
                'name': 'remote_calculator',
                'description': 'Perform calculations remotely',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'expression': {
                            'type': 'string',
                            'description': 'Mathematical expression to evaluate'
                        }
                    },
                    'required': ['expression']
                }
            }
        },
        'callback_url': 'http://localhost:8001/execute/remote_calculator'
    })
    print(f"Registration result: {result}\n")
    
    # 5. 列出所有工具
    print("5. Available tools:")
    for tool_name in system.enhanced_tool_registry.list_tools():
        tool = system.enhanced_tool_registry.get_tool(tool_name)
        print(f"  - {tool_name}: {tool.description}")
        if isinstance(tool, RemoteToolProxy):
            print(f"    (Remote tool at {tool.endpoint})")
    print()
    
    # 6. 测试系统状态查询
    print("6. System status:")
    status_result = await handler.handle_system_query({
        'query_type': 'status'
    })
    print(f"Status: {json.dumps(status_result, indent=2)}\n")
    
    print("Demo completed!")
    
    # 保持服务运行
    print("\nSystem is running. Press Ctrl+C to stop.")
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down...")

# ==================== 客户端示例 ====================

class SystemClient:
    """与扩展系统通信的客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        
    async def register_tool(self, name: str, description: str, 
                          schema: Dict[str, Any], endpoint: str):
        """注册远程工具"""
        async with self.session.post(
            f"{self.base_url}/register_tool",
            json={
                'tool_name': name,
                'tool_description': description,
                'tool_schema': schema,
                'callback_url': endpoint
            }
        ) as response:
            return await response.json()
            
    async def execute_tool(self, tool_name: str, **parameters):
        """执行工具"""
        async with self.session.post(
            f"{self.base_url}/execute_tool/{tool_name}",
            json=parameters
        ) as response:
            return await response.json()
            
    async def chat(self, message: str, conversation_id: str = None):
        """发送聊天消息"""
        async with self.session.post(
            f"{self.base_url}/chat",
            json={
                'message': message,
                'conversation_id': conversation_id
            }
        ) as response:
            return await response.json()
            
    async def execute_task(self, task: str, context: Dict[str, Any] = None):
        """执行任务"""
        async with self.session.post(
            f"{self.base_url}/task",
            json={
                'task': task,
                'context': context or {}
            }
        ) as response:
            return await response.json()
            
    async def get_status(self):
        """获取系统状态"""
        async with self.session.get(f"{self.base_url}/status") as response:
            return await response.json()
            
    async def list_tools(self):
        """列出所有工具"""
        async with self.session.get(f"{self.base_url}/tools") as response:
            return await response.json()

async def demo_client():
    """演示客户端使用"""
    print("=== System Client Demo ===\n")
    
    async with SystemClient() as client:
        # 1. 获取系统状态
        print("1. Getting system status...")
        status = await client.get_status()
        print(f"Status: {json.dumps(status, indent=2)}\n")
        
        # 2. 列出工具
        print("2. Listing available tools...")
        tools = await client.list_tools()
        print(f"Available tools: {list(tools.keys())}\n")
        
        # 3. 执行聊天
        print("3. Chatting with the system...")
        chat_response = await client.chat("Hello! What tools do you have available?")
        print(f"Response: {chat_response['response']}\n")
        
        # 4. 执行任务
        print("4. Executing a task...")
        task_response = await client.execute_task(
            "Search for information about Python async programming",
            {"max_results": 3}
        )
        print(f"Task result: {task_response['overall_success']}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='System Integration Extension')
    parser.add_argument('--mode', choices=['server', 'client', 'demo'], 
                       default='demo', help='Run mode')
    parser.add_argument('--port', type=int, default=8000,
                       help='API server port')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        # 运行扩展系统服务器
        asyncio.run(demo_extended_system())
    elif args.mode == 'client':
        # 运行客户端演示
        asyncio.run(demo_client())
    else:
        # 运行完整演示
        asyncio.run(demo_extended_system())
