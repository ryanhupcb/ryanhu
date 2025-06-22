# enhanced_agent_launcher.py
# 增强版Agent系统启动器 - 集成所有组件并提供统一接口

import asyncio
import os
import sys
import json
import argparse
import signal
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
import yaml
from dataclasses import dataclass, asdict
import uvloop  # For better async performance

# Import core components
from enhanced_production_agent import (
    EnhancedProductionAgent,
    TinyLLMProvider,
    ComputerUseController,
    GitHubIntegration,
    BrowserAutomation,
    ReactTotHybridReasoner,
    IntelligentTaskScheduler
)
from enhanced_tools import EnhancedToolRegistry

# Import original system components if available
try:
    from complete_agent_system import (
        CompleteAgentSystem,
        AgentCommunicationBus,
        AgentMessage
    )
    ORIGINAL_SYSTEM_AVAILABLE = True
except ImportError:
    ORIGINAL_SYSTEM_AVAILABLE = False

# Configure high-performance async
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Configuration Management ====================

@dataclass
class SystemConfig:
    """系统配置数据类"""
    # LLM Configuration
    use_tiny_llm: bool = True
    tiny_llm_threshold: float = 0.8  # Confidence threshold for TinyLLM
    tiny_llm_model: str = "deepseek-coder-1.3b"
    openai_model: str = "gpt-4-turbo-preview"
    anthropic_model: str = "claude-3-opus-20240229"
    
    # System Features
    enable_computer_control: bool = True
    enable_browser_automation: bool = True
    enable_github_integration: bool = True
    enable_multi_agent_collab: bool = True
    
    # Performance Settings
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # seconds
    memory_limit_mb: int = 4096
    
    # Security Settings
    sandbox_code_execution: bool = True
    allowed_system_commands: List[str] = None
    restricted_paths: List[str] = None
    
    # API Configuration
    api_enabled: bool = True
    api_port: int = 8000
    api_host: str = "0.0.0.0"
    api_auth_enabled: bool = False
    api_auth_tokens: List[str] = None
    
    # Communication Settings
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    rabbitmq_enabled: bool = False
    rabbitmq_url: str = "amqp://guest:guest@localhost/"
    
    # Monitoring
    metrics_enabled: bool = True
    metrics_export_interval: int = 60
    log_level: str = "INFO"
    log_file: str = "agent_system.log"
    
    @classmethod
    def from_file(cls, file_path: str) -> 'SystemConfig':
        """从文件加载配置"""
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {file_path}, using defaults")
            return cls()
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls(**data)
    
    def save(self, file_path: str):
        """保存配置到文件"""
        path = Path(file_path)
        data = asdict(self)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

# ==================== System Launcher ====================

class EnhancedAgentLauncher:
    """增强版Agent系统启动器"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.agent_system = None
        self.tool_registry = None
        self.api_server = None
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        """处理关闭信号"""
        logger.info("Shutdown signal received")
        self.shutdown_event.set()
        
    async def initialize(self):
        """初始化系统组件"""
        logger.info("Initializing Enhanced Agent System...")
        
        # Create agent system
        agent_config = {
            'use_tiny_llm': self.config.use_tiny_llm,
            'openai_model': self.config.openai_model,
            'anthropic_model': self.config.anthropic_model
        }
        
        self.agent_system = EnhancedProductionAgent(agent_config)
        
        # Initialize tool registry
        self.tool_registry = EnhancedToolRegistry()
        
        # Start API server if enabled
        if self.config.api_enabled:
            await self._start_api_server()
        
        # Initialize communication channels
        if self.config.redis_enabled:
            await self._init_redis()
            
        if self.config.rabbitmq_enabled:
            await self._init_rabbitmq()
        
        logger.info("System initialization complete")
        
    async def _start_api_server(self):
        """启动API服务器"""
        from aiohttp import web
        
        app = web.Application()
        
        # Setup routes
        app.router.add_post('/execute', self.handle_execute)
        app.router.add_post('/chat', self.handle_chat)
        app.router.add_get('/status', self.handle_status)
        app.router.add_get('/metrics', self.handle_metrics)
        app.router.add_post('/tools/{tool_name}/{method_name}', self.handle_tool_execution)
        
        # Add middleware for authentication if enabled
        if self.config.api_auth_enabled:
            app.middlewares.append(self._auth_middleware)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.api_host, self.config.api_port)
        await site.start()
        
        self.api_server = runner
        logger.info(f"API server started on {self.config.api_host}:{self.config.api_port}")
        
    @web.middleware
    async def _auth_middleware(self, request, handler):
        """API认证中间件"""
        from aiohttp import web
        
        # Skip auth for health check
        if request.path == '/status':
            return await handler(request)
        
        # Check authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return web.json_response({'error': 'Unauthorized'}, status=401)
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        if token not in self.config.api_auth_tokens:
            return web.json_response({'error': 'Invalid token'}, status=401)
        
        return await handler(request)
        
    async def handle_execute(self, request):
        """处理执行请求"""
        from aiohttp import web
        
        try:
            data = await request.json()
            request_text = data.get('request', '')
            context = data.get('context', {})
            
            result = await self.agent_system.execute(request_text, context)
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Execute request failed: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    async def handle_chat(self, request):
        """处理聊天请求"""
        from aiohttp import web
        
        try:
            data = await request.json()
            message = data.get('message', '')
            conversation_id = data.get('conversation_id')
            
            # Simple chat implementation
            result = await self.agent_system.execute(message)
            
            return web.json_response({
                'response': result.get('result', {}).get('summary', 'I apologize, but I encountered an error processing your request.'),
                'conversation_id': conversation_id or 'default',
                'success': result['success']
            })
            
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            return web.json_response({
                'error': str(e),
                'success': False
            }, status=500)
            
    async def handle_status(self, request):
        """处理状态请求"""
        from aiohttp import web
        
        status = {
            'status': 'healthy',
            'version': '2.0.0',
            'uptime': self.agent_system.get_metrics()['uptime_seconds'],
            'features': {
                'tiny_llm': self.config.use_tiny_llm,
                'computer_control': self.config.enable_computer_control,
                'browser_automation': self.config.enable_browser_automation,
                'github_integration': self.config.enable_github_integration
            }
        }
        
        return web.json_response(status)
        
    async def handle_metrics(self, request):
        """处理指标请求"""
        from aiohttp import web
        
        metrics = self.agent_system.get_metrics()
        return web.json_response(metrics)
        
    async def handle_tool_execution(self, request):
        """处理工具执行请求"""
        from aiohttp import web
        
        tool_name = request.match_info['tool_name']
        method_name = request.match_info['method_name']
        
        try:
            data = await request.json()
            
            # Get tool
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return web.json_response({
                    'error': f'Tool not found: {tool_name}'
                }, status=404)
            
            # Get method
            method = getattr(tool, method_name, None)
            if not method or not callable(method):
                return web.json_response({
                    'error': f'Method not found: {method_name}'
                }, status=404)
            
            # Execute method
            result = await method(**data)
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    async def _init_redis(self):
        """初始化Redis连接"""
        # Implementation here
        pass
        
    async def _init_rabbitmq(self):
        """初始化RabbitMQ连接"""
        # Implementation here
        pass
        
    async def run(self):
        """运行系统"""
        await self.initialize()
        
        logger.info("Enhanced Agent System is running")
        logger.info(f"API: http://{self.config.api_host}:{self.config.api_port}")
        
        # Wait for shutdown
        await self.shutdown_event.wait()
        
        # Cleanup
        await self.cleanup()
        
    async def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up resources...")
        
        if self.api_server:
            await self.api_server.cleanup()
            
        logger.info("Cleanup complete")
        
    async def interactive_mode(self):
        """交互模式"""
        await self.initialize()
        
        print("\n=== Enhanced Agent System - Interactive Mode ===")
        print("Commands:")
        print("  help - Show this help")
        print("  execute <request> - Execute a request")
        print("  status - Show system status")
        print("  metrics - Show system metrics")
        print("  tools - List available tools")
        print("  quit - Exit\n")
        
        while not self.shutdown_event.is_set():
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None, input, "> "
                )
                
                if not command:
                    continue
                    
                parts = command.split(' ', 1)
                cmd = parts[0].lower()
                
                if cmd == 'quit':
                    break
                elif cmd == 'help':
                    print("Available commands: help, execute, status, metrics, tools, quit")
                elif cmd == 'status':
                    metrics = self.agent_system.get_metrics()
                    print(f"System Status: Healthy")
                    print(f"Uptime: {metrics['uptime_seconds']:.2f}s")
                    print(f"Total executions: {metrics['total_executions']}")
                elif cmd == 'metrics':
                    metrics = self.agent_system.get_metrics()
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
                elif cmd == 'tools':
                    tools = self.tool_registry.list_tools()
                    print("Available tools:")
                    for tool in tools:
                        print(f"  - {tool}")
                elif cmd == 'execute' and len(parts) > 1:
                    request = parts[1]
                    print("Executing...")
                    result = await self.agent_system.execute(request)
                    if result['success']:
                        print(f"Success! Execution time: {result['execution_time']:.2f}s")
                        if 'summary' in result.get('result', {}):
                            print(f"Summary: {result['result']['summary']}")
                    else:
                        print(f"Failed: {result.get('error', 'Unknown error')}")
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit properly")
            except Exception as e:
                print(f"Error: {e}")
                
        self.shutdown_event.set()

# ==================== CLI Interface ====================

def create_default_config():
    """创建默认配置文件"""
    config = SystemConfig()
    config.save('agent_config.yaml')
    print("Created default configuration file: agent_config.yaml")
    
def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description='Enhanced Agent System - Production Grade AI Agent'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='agent_config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['server', 'interactive', 'execute'],
        default='server',
        help='Run mode'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file'
    )
    
    parser.add_argument(
        '--request', '-r',
        type=str,
        help='Request to execute (for execute mode)'
    )
    
    parser.add_argument(
        '--context',
        type=str,
        help='Context JSON for request (for execute mode)'
    )
    
    args = parser.parse_args()
    
    # Create config if requested
    if args.create_config:
        create_default_config()
        return
    
    # Load configuration
    config = SystemConfig.from_file(args.config)
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, config.log_level))
    
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Create launcher
    launcher = EnhancedAgentLauncher(config)
    
    # Run based on mode
    if args.mode == 'server':
        asyncio.run(launcher.run())
    elif args.mode == 'interactive':
        asyncio.run(launcher.interactive_mode())
    elif args.mode == 'execute':
        if not args.request:
            print("Error: --request is required for execute mode")
            return
            
        async def execute_single():
            await launcher.initialize()
            
            context = {}
            if args.context:
                context = json.loads(args.context)
                
            result = await launcher.agent_system.execute(args.request, context)
            
            print(json.dumps(result, indent=2))
            
        asyncio.run(execute_single())

# ==================== Docker Configuration ====================

DOCKERFILE_CONTENT = '''# Enhanced Agent System Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    xvfb \
    x11vnc \
    fluxbox \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for JavaScript tools
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Install browser dependencies for automation
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
RUN npm install -g prettier eslint typescript

# Copy application code
COPY enhanced_production_agent.py .
COPY enhanced_tools.py .
COPY enhanced_agent_launcher.py .

# Create necessary directories
RUN mkdir -p /app/workspace /app/logs /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/status || exit 1

# Start script for virtual display
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
sleep 3\n\
x11vnc -display :99 -nopw -listen localhost -xkb -forever -bg\n\
fluxbox > /dev/null 2>&1 &\n\
python enhanced_agent_launcher.py "$@"' > /app/start.sh && \
    chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh", "--mode", "server"]
'''

DOCKER_COMPOSE_CONTENT = '''version: '3.8'

services:
  agent-system:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: enhanced-agent-system
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GITHUB_ACCESS_TOKEN=${GITHUB_ACCESS_TOKEN}
      - TINY_LLM_PATH=/app/models/tinyllm
    ports:
      - "8000:8000"
      - "5900:5900"  # VNC for debugging
    volumes:
      - ./workspace:/app/workspace
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models:/app/models
      - ./agent_config.yaml:/app/agent_config.yaml
    depends_on:
      - redis
    networks:
      - agent-network
    restart: unless-stopped
    command: ["--config", "/app/agent_config.yaml", "--mode", "server"]

  redis:
    image: redis:7-alpine
    container_name: agent-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - agent-network
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: agent-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - agent-system
    networks:
      - agent-network
    restart: unless-stopped

volumes:
  redis-data:

networks:
  agent-network:
    driver: bridge
'''

REQUIREMENTS_CONTENT = '''# Core dependencies
asyncio>=3.4.3
aiohttp>=3.9.0
uvloop>=0.19.0

# LLM providers
openai>=1.6.0
anthropic>=0.8.0
transformers>=4.36.0
torch>=2.0.0

# Code development tools
black>=23.0.0
autopep8>=2.0.0
pylint>=3.0.0
radon>=6.0.0
esprima>=4.0.1

# System control
pyautogui>=0.9.54
pygetwindow>=0.0.9
psutil>=5.9.0
Pillow>=10.0.0

# Web and API
beautifulsoup4>=4.12.0
requests>=2.31.0
aiofiles>=23.0.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0.1
toml>=0.10.2

# Database
sqlite3
redis>=5.0.0
aio-pika>=9.3.1

# GitHub integration
PyGithub>=2.1.0

# Monitoring and logging
structlog>=23.2.0
prometheus-client>=0.19.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Documentation
sphinx>=7.2.0

# Browser automation (optional)
# browser-use  # Uncomment if available

# Development
mypy>=1.7.0
typing-extensions>=4.8.0
'''

# ==================== Quick Setup Script ====================

SETUP_SCRIPT = '''#!/bin/bash
# Enhanced Agent System - Quick Setup Script

set -e

echo "=== Enhanced Agent System Setup ==="

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ required, found $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv agent_env
source agent_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download TinyLLM model (optional)
echo "Downloading TinyLLM model..."
mkdir -p models
# Add model download command here

# Create directories
echo "Creating directories..."
mkdir -p workspace logs data

# Create default config
echo "Creating default configuration..."
python enhanced_agent_launcher.py --create-config

echo ""
echo "Setup complete! To start the system:"
echo "  source agent_env/bin/activate"
echo "  python enhanced_agent_launcher.py --mode interactive"
echo ""
echo "For production deployment:"
echo "  docker-compose up -d"
'''

# Save configuration files
def save_deployment_files():
    """保存部署文件"""
    files = {
        'Dockerfile': DOCKERFILE_CONTENT,
        'docker-compose.yml': DOCKER_COMPOSE_CONTENT,
        'requirements.txt': REQUIREMENTS_CONTENT,
        'setup.sh': SETUP_SCRIPT
    }
    
    for filename, content in files.items():
        with open(filename, 'w') as f:
            f.write(content.strip() + '\n')
            
        if filename == 'setup.sh':
            os.chmod(filename, 0o755)
            
    print("Deployment files created:")
    for filename in files.keys():
        print(f"  - {filename}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--save-deployment-files':
        save_deployment_files()
    else:
        main()
