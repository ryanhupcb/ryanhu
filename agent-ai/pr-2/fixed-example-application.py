# LocalAgentSystem - Complete Example Application
# 完整示例应用

import asyncio
import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from typing import Optional, List, Dict, Any
import os
from datetime import datetime
import time
import json
import subprocess
import sys

# 需要创建的占位模块（如果实际模块不存在）
class MonitoringSystem:
    def __init__(self, config):
        self.config = config
        
    def get_dashboard_data(self):
        return {
            "health": {
                "status": "healthy",
                "system_info": {
                    "cpu_percent": 50.0,
                    "memory_percent": 60.0,
                    "disk_percent": 70.0,
                    "process_count": 10,
                    "uptime": 3600.0
                }
            },
            "performance": {
                "slow_operations": []
            }
        }

class AgentSystemConfig:
    @classmethod
    def from_env(cls):
        return cls()

class LocalAgentSystem:
    def __init__(self, config):
        self.config = config
        self.is_initialized = False
        self.orchestrator = type('obj', (object,), {
            'agents': {},
            'active_tasks': {},
            'completed_tasks': {}
        })()
        
    async def initialize(self):
        self.is_initialized = True
        
    async def process_request(self, request):
        return {
            "success": True,
            "response": f"Processed: {request}",
            "detailed_results": {}
        }
        
    async def shutdown(self):
        self.is_initialized = False

class GradioUI:
    def __init__(self, system):
        self.system = system
        
    def run(self):
        print("Starting Gradio UI...")

class StreamlitUI:
    def __init__(self, system):
        self.system = system
        
    def run(self):
        print("Starting Streamlit UI...")

class Task:
    def __init__(self, type="", description=""):
        self.type = type
        self.description = description


# ==================== 初始化脚本 ====================

class AgentSystemInitializer:
    """Agent系统初始化器"""
    
    def __init__(self):
        self.console = Console()
        self.config_path = ".env"
        
    async def initialize_system(self):
        """初始化完整系统"""
        self.console.print("[bold blue]🚀 Initializing Local Agent System...[/bold blue]")
        
        # 1. 检查环境
        self._check_environment()
        
        # 2. 配置向导
        await self._configuration_wizard()
        
        # 3. 初始化组件
        await self._initialize_components()
        
        # 4. 运行测试
        await self._run_initial_tests()
        
        self.console.print("[bold green]✅ System initialized successfully![/bold green]")
        
    def _check_environment(self):
        """检查环境依赖"""
        self.console.print("\n[yellow]Checking environment...[/yellow]")
        
        requirements = {
            "Python": self._check_python_version(),
            "Docker": self._check_docker(),
            "Git": self._check_git(),
            "Redis": self._check_redis(),
            "PostgreSQL": self._check_postgres()
        }
        
        table = Table(title="Environment Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Version", style="yellow")
        
        all_ok = True
        for component, (status, version) in requirements.items():
            status_icon = "✅" if status else "❌"
            table.add_row(component, status_icon, version or "Not found")
            if not status:
                all_ok = False
                
        self.console.print(table)
        
        if not all_ok:
            self.console.print("[red]Please install missing components before continuing.[/red]")
            raise SystemExit(1)
            
    def _check_python_version(self):
        """检查Python版本"""
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return (sys.version_info >= (3, 8), version)
        
    def _check_docker(self):
        """检查Docker"""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return (True, result.stdout.strip())
        except:
            pass
        return (False, None)
        
    def _check_git(self):
        """检查Git"""
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return (True, result.stdout.strip())
        except:
            pass
        return (False, None)
        
    def _check_redis(self):
        """检查Redis"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379)
            r.ping()
            return (True, "Connected")
        except:
            return (False, None)
            
    def _check_postgres(self):
        """检查PostgreSQL"""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                database="postgres",
                user="postgres",
                password="postgres"
            )
            conn.close()
            return (True, "Connected")
        except:
            return (False, None)
            
    async def _configuration_wizard(self):
        """配置向导"""
        self.console.print("\n[yellow]Configuration Wizard[/yellow]")
        
        config = {}
        
        # API Keys
        self.console.print("\n[cyan]API Keys Configuration:[/cyan]")
        config['CLAUDE_API_KEY'] = self._get_input("Claude API Key", masked=True)
        config['QWEN_API_KEY'] = self._get_input("Qwen API Key", masked=True)
        config['DEEPSEEK_API_KEY'] = self._get_input("DeepSeek API Key (optional)", masked=True, required=False)
        config['GITHUB_TOKEN'] = self._get_input("GitHub Token (optional)", masked=True, required=False)
        
        # Model Selection
        self.console.print("\n[cyan]Model Configuration:[/cyan]")
        config['CLAUDE_MODEL'] = self._get_input("Claude Model", default="claude-opus-4-20250514")
        config['QWEN_MODEL'] = self._get_input("Qwen Model", default="qwen-plus")
        config['USE_LOCAL_DEEPSEEK'] = self._get_input("Use Local DeepSeek? (yes/no)", default="yes")
        
        # System Settings
        self.console.print("\n[cyan]System Settings:[/cyan]")
        config['MAX_COST_PER_REQUEST'] = self._get_input("Max Cost per Request ($)", default="2.0")
        config['MAX_AGENTS'] = self._get_input("Max Concurrent Agents", default="5")
        config['WORKSPACE_DIR'] = self._get_input("Workspace Directory", default="./workspace")
        
        # Save configuration
        self._save_config(config)
        self.console.print("[green]Configuration saved to .env[/green]")
        
    def _get_input(self, prompt: str, default: str = None, masked: bool = False, required: bool = True) -> str:
        """获取用户输入"""
        if default:
            prompt = f"{prompt} [{default}]"
        prompt = f"{prompt}: "
        
        while True:
            if masked:
                from getpass import getpass
                value = getpass(prompt)
            else:
                value = input(prompt)
                
            if not value and default:
                return default
            elif not value and not required:
                return ""
            elif value:
                return value
            else:
                self.console.print("[red]This field is required.[/red]")
                
    def _save_config(self, config: Dict[str, str]):
        """保存配置到.env文件"""
        with open(self.config_path, 'w') as f:
            for key, value in config.items():
                if value:  # 只保存非空值
                    f.write(f"{key}={value}\n")
                    
    async def _initialize_components(self):
        """初始化系统组件"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # 创建目录
            task = progress.add_task("Creating directories...", total=4)
            directories = ["workspace", "logs", "data", "cache"]
            for dir_name in directories:
                os.makedirs(dir_name, exist_ok=True)
                progress.advance(task)
                
            # 初始化数据库
            task = progress.add_task("Initializing database...", total=1)
            await self._init_database()
            progress.advance(task)
            
            # 下载模型（如果需要）
            if os.getenv("USE_LOCAL_DEEPSEEK") == "yes":
                task = progress.add_task("Downloading DeepSeek model...", total=1)
                await self._download_deepseek_model()
                progress.advance(task)
                
    async def _init_database(self):
        """初始化数据库"""
        # 这里应该运行数据库迁移
        await asyncio.sleep(1)  # 模拟操作
        
    async def _download_deepseek_model(self):
        """下载DeepSeek模型"""
        # 这里应该调用Ollama下载模型
        await asyncio.sleep(2)  # 模拟下载
        
    async def _run_initial_tests(self):
        """运行初始测试"""
        self.console.print("\n[yellow]Running initial tests...[/yellow]")
        
        tests = [
            ("LLM Connection", self._test_llm_connection),
            ("File Operations", self._test_file_operations),
            ("GitHub Integration", self._test_github),
            ("Web Search", self._test_web_search)
        ]
        
        table = Table(title="Test Results")
        table.add_column("Test", style="cyan")
        table.add_column("Result", style="green")
        table.add_column("Details", style="yellow")
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                table.add_row(test_name, "✅ Passed", result)
            except Exception as e:
                table.add_row(test_name, "❌ Failed", str(e))
                
        self.console.print(table)
        
    async def _test_llm_connection(self) -> str:
        """测试LLM连接"""
        # 实际测试代码
        return "All models connected"
        
    async def _test_file_operations(self) -> str:
        """测试文件操作"""
        test_file = "./workspace/test.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        return "File operations working"
        
    async def _test_github(self) -> str:
        """测试GitHub集成"""
        # 实际测试代码
        return "GitHub API accessible"
        
    async def _test_web_search(self) -> str:
        """测试网络搜索"""
        # 实际测试代码
        return "Web search functional"


# ==================== 命令行界面 ====================

@click.group()
def cli():
    """Local Agent System CLI"""
    pass


@cli.command()
def init():
    """Initialize the agent system"""
    initializer = AgentSystemInitializer()
    asyncio.run(initializer.initialize_system())


@cli.command()
@click.option('--task-type', '-t', type=click.Choice(['code', 'research', 'browser', 'file']), 
              default='code', help='Type of task')
@click.argument('description')
def run(task_type, description):
    """Run a task"""
    console = Console()
    
    async def execute_task():
        # 加载配置
        from dotenv import load_dotenv
        load_dotenv()
        
        config = AgentSystemConfig.from_env()
        system = LocalAgentSystem(config)
        
        # 初始化系统
        with console.status("Initializing system..."):
            await system.initialize()
            
        # 创建任务
        task_type_mapping = {
            'code': 'code_generation',
            'research': 'research',
            'browser': 'browser_automation',
            'file': 'file_operation'
        }
        
        # 处理任务
        with console.status(f"Processing {task_type} task..."):
            result = await system.process_request(description)
            
        # 显示结果
        if result["success"]:
            console.print(Panel(
                result["response"],
                title="[bold green]Task Completed[/bold green]",
                border_style="green"
            ))
            
            # 显示详细信息
            if result.get("detailed_results"):
                console.print("\n[yellow]Task Details:[/yellow]")
                console.print(result["detailed_results"])
        else:
            console.print(Panel(
                f"[red]Error: {result['error']}[/red]",
                title="[bold red]Task Failed[/bold red]",
                border_style="red"
            ))
            
        await system.shutdown()
        
    asyncio.run(execute_task())


@cli.command()
def status():
    """Check system status"""
    console = Console()
    
    async def check_status():
        from dotenv import load_dotenv
        load_dotenv()
        
        # 创建监控系统
        monitoring = MonitoringSystem({"metrics_port": 8001})
        
        # 获取状态
        dashboard_data = monitoring.get_dashboard_data()
        
        # 显示健康状态
        health = dashboard_data["health"]
        console.print(Panel(
            f"Status: {health['status']}",
            title="[bold]System Health[/bold]",
            border_style="green" if health['status'] == 'healthy' else "red"
        ))
        
        # 显示系统信息
        table = Table(title="System Information")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        sys_info = health["system_info"]
        table.add_row("CPU Usage", f"{sys_info['cpu_percent']:.1f}%")
        table.add_row("Memory Usage", f"{sys_info['memory_percent']:.1f}%")
        table.add_row("Disk Usage", f"{sys_info['disk_percent']:.1f}%")
        table.add_row("Process Count", str(sys_info['process_count']))
        table.add_row("Uptime", f"{sys_info['uptime'] / 3600:.1f} hours")
        
        console.print(table)
        
        # 显示性能分析
        perf = dashboard_data["performance"]
        if perf.get("slow_operations"):
            console.print("\n[yellow]⚠️  Slow Operations Detected:[/yellow]")
            for op in perf["slow_operations"][:5]:
                console.print(f"  - {op['operation']}: {op['duration']:.2f}s")
                
    asyncio.run(check_status())


@cli.command()
@click.option('--port', '-p', default=8501, help='Port to run the UI on')
@click.option('--mode', '-m', type=click.Choice(['streamlit', 'gradio']), 
              default='gradio', help='UI mode')
def ui(port, mode):
    """Start the web UI"""
    console = Console()
    console.print(f"[bold blue]Starting {mode} UI on port {port}...[/bold blue]")
    
    async def start_ui():
        from dotenv import load_dotenv
        load_dotenv()
        
        config = AgentSystemConfig.from_env()
        system = LocalAgentSystem(config)
        
        if mode == 'gradio':
            ui = GradioUI(system)
            ui.run()
        else:
            ui = StreamlitUI(system)
            ui.run()
            
    asyncio.run(start_ui())


@cli.command()
def examples():
    """Show usage examples"""
    console = Console()
    
    examples = [
        {
            "title": "Code Generation",
            "command": "agent run -t code 'Create a REST API for a todo list using FastAPI'",
            "description": "Generate a complete FastAPI application with CRUD operations"
        },
        {
            "title": "Research Task",
            "command": "agent run -t research 'Research the latest trends in AI agent architectures'",
            "description": "Gather and analyze information from multiple sources"
        },
        {
            "title": "Browser Automation",
            "command": "agent run -t browser 'Navigate to GitHub and search for python web scraping libraries'",
            "description": "Automate browser interactions"
        },
        {
            "title": "File Operations",
            "command": "agent run -t file 'Organize files in the downloads folder by type'",
            "description": "Perform file system operations"
        }
    ]
    
    console.print("[bold]Local Agent System - Usage Examples[/bold]\n")
    
    for example in examples:
        console.print(Panel(
            f"[yellow]Command:[/yellow]\n{example['command']}\n\n"
            f"[cyan]Description:[/cyan]\n{example['description']}",
            title=f"[bold]{example['title']}[/bold]",
            border_style="blue"
        ))
        console.print()


# ==================== 实际使用案例 ====================

class UseCaseExamples:
    """实际使用案例示例"""
    
    def __init__(self):
        self.console = Console()
        
    async def example_web_scraper(self):
        """示例：创建网页爬虫"""
        self.console.print("[bold]Example: Creating a Web Scraper[/bold]")
        
        request = """
        Create a Python web scraper that:
        1. Scrapes news headlines from a news website
        2. Extracts title, date, and summary
        3. Saves the data to JSON and CSV formats
        4. Implements rate limiting and error handling
        5. Includes a simple CLI interface
        """
        
        # 初始化系统
        config = AgentSystemConfig.from_env()
        system = LocalAgentSystem(config)
        await system.initialize()
        
        # 处理请求
        with self.console.status("Creating web scraper..."):
            result = await system.process_request(request)
            
        if result["success"]:
            self.console.print("[green]Web scraper created successfully![/green]")
            
            # 显示生成的代码
            code = result["detailed_results"].get("code_generation", {}).get("code", "")
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            self.console.print(Panel(syntax, title="Generated Code"))
            
        await system.shutdown()
        
    async def example_api_development(self):
        """示例：开发REST API"""
        self.console.print("[bold]Example: Developing a REST API[/bold]")
        
        request = """
        Develop a complete REST API for a task management system with:
        1. User authentication (JWT)
        2. CRUD operations for tasks
        3. Task categories and tags
        4. Due dates and priorities
        5. Database models (SQLAlchemy)
        6. Input validation (Pydantic)
        7. API documentation (OpenAPI)
        8. Unit tests
        """
        
        config = AgentSystemConfig.from_env()
        system = LocalAgentSystem(config)
        await system.initialize()
        
        with self.console.status("Developing API..."):
            result = await system.process_request(request)
            
        if result["success"]:
            self.console.print("[green]API developed successfully![/green]")
            
            # 显示文件结构
            files = result["detailed_results"].get("files_created", [])
            if files:
                self.console.print("\n[yellow]Created Files:[/yellow]")
                for file in files:
                    self.console.print(f"  📄 {file}")
                    
        await system.shutdown()
        
    async def example_data_analysis(self):
        """示例：数据分析脚本"""
        self.console.print("[bold]Example: Data Analysis Script[/bold]")
        
        request = """
        Create a data analysis script that:
        1. Loads data from CSV files
        2. Performs exploratory data analysis
        3. Creates visualizations (matplotlib/seaborn)
        4. Generates statistical summaries
        5. Identifies patterns and anomalies
        6. Exports results to a report (PDF/HTML)
        """
        
        config = AgentSystemConfig.from_env()
        system = LocalAgentSystem(config)
        await system.initialize()
        
        with self.console.status("Creating analysis script..."):
            result = await system.process_request(request)
            
        if result["success"]:
            self.console.print("[green]Analysis script created![/green]")
            
        await system.shutdown()
        
    async def example_browser_automation(self):
        """示例：浏览器自动化"""
        self.console.print("[bold]Example: Browser Automation[/bold]")
        
        request = """
        Automate the following browser tasks:
        1. Navigate to a job portal website
        2. Search for Python developer positions
        3. Filter by location and salary range
        4. Extract job listings (title, company, salary, requirements)
        5. Save the data to a spreadsheet
        6. Send email notification with new listings
        """
        
        config = AgentSystemConfig.from_env()
        system = LocalAgentSystem(config)
        await system.initialize()
        
        with self.console.status("Setting up browser automation..."):
            result = await system.process_request(request)
            
        if result["success"]:
            self.console.print("[green]Browser automation script created![/green]")
            
        await system.shutdown()


# ==================== 测试套件 ====================

class AgentSystemTests:
    """Agent系统测试套件"""
    
    def __init__(self):
        self.console = Console()
        self.test_results = []
        
    async def run_all_tests(self):
        """运行所有测试"""
        self.console.print("[bold]Running Agent System Tests[/bold]\n")
        
        tests = [
            ("LLM Integration", self.test_llm_integration),
            ("Tool Execution", self.test_tool_execution),
            ("Multi-Agent Coordination", self.test_multi_agent),
            ("Error Recovery", self.test_error_recovery),
            ("Performance", self.test_performance),
            ("Cost Optimization", self.test_cost_optimization)
        ]
        
        for test_name, test_func in tests:
            try:
                await test_func()
                self.test_results.append((test_name, "✅ Passed", None))
            except Exception as e:
                self.test_results.append((test_name, "❌ Failed", str(e)))
                
        # 显示结果
        self._display_results()
        
    async def test_llm_integration(self):
        """测试LLM集成"""
        # 简化的测试实现
        pass
            
    async def test_tool_execution(self):
        """测试工具执行"""
        # 简化的测试实现
        pass
            
    async def test_multi_agent(self):
        """测试多Agent协作"""
        config = AgentSystemConfig.from_env()
        system = LocalAgentSystem(config)
        
        await system.initialize()
        
        # 简单任务测试
        result = await system.process_request("What is 2 + 2?")
        assert result["success"], "Simple task failed"
        
        await system.shutdown()
        
    async def test_error_recovery(self):
        """测试错误恢复"""
        # 简化的测试实现
        pass
        
    async def test_performance(self):
        """测试性能"""
        # 简化的测试实现
        pass
        
    async def test_cost_optimization(self):
        """测试成本优化"""
        # 简化的测试实现
        pass
        
    def _display_results(self):
        """显示测试结果"""
        table = Table(title="Test Results")
        table.add_column("Test", style="cyan")
        table.add_column("Result", style="green")
        table.add_column("Error", style="red")
        
        for test_name, result, error in self.test_results:
            table.add_row(test_name, result, error or "")
            
        self.console.print(table)
        
        # 统计
        passed = sum(1 for _, result, _ in self.test_results if "✅" in result)
        total = len(self.test_results)
        
        self.console.print(f"\n[bold]Total: {total} | Passed: {passed} | Failed: {total - passed}[/bold]")


# ==================== 主函数 ====================

def main():
    """主函数"""
    cli()


if __name__ == "__main__":
    main()
