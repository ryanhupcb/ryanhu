# test_agent_system.py
# 增强版Agent系统完整测试套件

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from datetime import datetime
import os

# 导入要测试的模块
from enhanced_production_agent import (
    EnhancedProductionAgent,
    TinyLLMProvider,
    ComputerUseController,
    ReactTotHybridReasoner,
    IntelligentTaskScheduler,
    CodeDevelopmentAgent
)
from enhanced_tools import (
    CodeFormatter,
    CodeAnalyzer,
    CodeGenerator,
    EnhancedFileOperations
)
from agent_client_sdk import AgentClient, TaskStatus, TaskResult

# ==================== Fixtures ====================

@pytest.fixture
def mock_config():
    """模拟配置"""
    return {
        'use_tiny_llm': True,
        'openai_model': 'gpt-4-turbo-preview',
        'anthropic_model': 'claude-3-opus-20240229'
    }

@pytest.fixture
def mock_llm_provider():
    """模拟LLM提供者"""
    provider = AsyncMock()
    provider.generate.return_value = {
        'content': 'Generated response',
        'model': 'mock-model',
        'tokens_used': 100
    }
    return provider

@pytest.fixture
async def agent_system(mock_config, mock_llm_provider):
    """创建测试用Agent系统"""
    with patch('enhanced_production_agent.TinyLLMProvider'):
        with patch('enhanced_production_agent.AnthropicProvider', return_value=mock_llm_provider):
            system = EnhancedProductionAgent(mock_config)
            yield system

@pytest.fixture
def temp_workspace():
    """创建临时工作目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

# ==================== Unit Tests ====================

class TestTinyLLMProvider:
    """TinyLLM提供者测试"""
    
    def test_can_handle_simple_tasks(self):
        """测试简单任务识别"""
        provider = TinyLLMProvider()
        
        # 应该处理的任务
        assert provider.can_handle("write a function to add two numbers")
        assert provider.can_handle("fix this code")
        assert provider.can_handle("explain this concept")
        
        # 不应该处理的任务
        assert not provider.can_handle("design a complex distributed system")
        assert not provider.can_handle("analyze market trends")
        
    @pytest.mark.asyncio
    async def test_generate_response(self):
        """测试响应生成"""
        provider = TinyLLMProvider()
        provider.model = Mock()
        provider.tokenizer = Mock()
        
        # 模拟tokenizer和model
        provider.tokenizer.return_value = {'input_ids': [1, 2, 3]}
        provider.model.generate.return_value = [[1, 2, 3, 4, 5]]
        provider.tokenizer.decode.return_value = "def add(a, b): return a + b"
        
        response = await provider.generate("write a function")
        
        assert 'content' in response
        assert 'model' in response
        assert response['model'] == 'tiny_llm'

class TestCodeFormatter:
    """代码格式化工具测试"""
    
    @pytest.mark.asyncio
    async def test_format_python_code(self):
        """测试Python代码格式化"""
        formatter = CodeFormatter()
        
        messy_code = "def hello(   ):print('world')"
        result = await formatter.format_python(messy_code)
        
        assert result['success']
        assert 'formatted_code' in result
        assert "def hello():" in result['formatted_code']
        
    @pytest.mark.asyncio
    async def test_format_invalid_python(self):
        """测试无效Python代码格式化"""
        formatter = CodeFormatter()
        
        invalid_code = "def hello( print('world'"
        result = await formatter.format_python(invalid_code)
        
        # autopep8可能会尝试修复或返回错误
        assert 'success' in result or 'error' in result

class TestCodeAnalyzer:
    """代码分析工具测试"""
    
    @pytest.mark.asyncio
    async def test_analyze_valid_python(self):
        """测试有效Python代码分析"""
        analyzer = CodeAnalyzer()
        
        code = """
def calculate_factorial(n):
    '''Calculate factorial of n'''
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)
"""
        
        result = await analyzer.analyze_python(code)
        
        assert result['syntax_valid']
        assert 'complexity' in result
        assert 'metrics' in result
        assert len(result['issues']) == 0
        
    @pytest.mark.asyncio
    async def test_analyze_complex_code(self):
        """测试复杂代码分析"""
        analyzer = CodeAnalyzer()
        
        # 创建一个高复杂度的函数
        complex_code = """
def complex_function(a, b, c, d, e):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        for i in range(10):
                            for j in range(10):
                                for k in range(10):
                                    print(i, j, k)
    return None
"""
        
        result = await analyzer.analyze_python(complex_code)
        
        assert result['syntax_valid']
        assert len(result['suggestions']) > 0  # 应该有复杂度建议

class TestEnhancedFileOperations:
    """文件操作工具测试"""
    
    @pytest.mark.asyncio
    async def test_create_project_structure(self, temp_workspace):
        """测试项目结构创建"""
        file_ops = EnhancedFileOperations(temp_workspace)
        
        structure = {
            'src': {
                '__init__.py': '# Init file',
                'main.py': 'print("Hello")'
            },
            'tests': {
                'test_main.py': '# Tests'
            },
            'README.md': '# Project'
        }
        
        result = await file_ops.create_project_structure('test_project', structure)
        
        assert result['success']
        assert (temp_workspace / 'test_project' / 'src' / 'main.py').exists()
        assert len(result['created_files']) == 4
        
    @pytest.mark.asyncio
    async def test_find_files(self, temp_workspace):
        """测试文件查找"""
        file_ops = EnhancedFileOperations(temp_workspace)
        
        # 创建测试文件
        (temp_workspace / 'test1.py').write_text('# Test 1')
        (temp_workspace / 'test2.py').write_text('# Test 2')
        (temp_workspace / 'data.json').write_text('{}')
        
        # 查找Python文件
        py_files = await file_ops.find_files('*.py')
        
        assert len(py_files) == 2
        assert all(f.endswith('.py') for f in py_files)

class TestReactTotHybridReasoner:
    """混合推理器测试"""
    
    @pytest.mark.asyncio
    async def test_reasoning_process(self, mock_llm_provider):
        """测试推理过程"""
        reasoner = ReactTotHybridReasoner(mock_llm_provider)
        
        task = "Create a web scraping tool"
        context = {'language': 'python'}
        
        # 模拟LLM响应
        mock_llm_provider.generate.return_value = {
            'content': json.dumps({
                'objective': 'Create web scraper',
                'steps': [
                    {'description': 'Setup requests library', 'complexity': 'low'},
                    {'description': 'Parse HTML', 'complexity': 'medium'}
                ],
                'tools_required': ['requests', 'beautifulsoup4']
            })
        }
        
        result = await reasoner.reason(task, context)
        
        assert 'objective' in result
        assert 'steps' in result
        assert len(result['steps']) > 0

class TestIntelligentTaskScheduler:
    """任务调度器测试"""
    
    @pytest.mark.asyncio
    async def test_task_decomposition(self, mock_llm_provider):
        """测试任务分解"""
        agents = {'test_agent': Mock()}
        scheduler = IntelligentTaskScheduler(agents)
        
        task = "Build a REST API with authentication"
        context = {}
        
        # 模拟TinyLLM分解
        with patch.object(scheduler, '_tiny_llm_decompose') as mock_decompose:
            mock_decompose.return_value = [
                {'id': 'step_1', 'description': 'Create API structure', 'type': 'code'},
                {'id': 'step_2', 'description': 'Add authentication', 'type': 'code'}
            ]
            
            result = await scheduler.decompose_task(task, context)
            
            assert len(result) == 2
            assert all('id' in step for step in result)
            
    def test_dependency_graph_building(self):
        """测试依赖图构建"""
        agents = {}
        scheduler = IntelligentTaskScheduler(agents)
        
        tasks = [
            {'id': 'task1', 'dependencies': []},
            {'id': 'task2', 'dependencies': ['task1']},
            {'id': 'task3', 'dependencies': ['task1', 'task2']}
        ]
        
        scheduler._build_dependency_graph(tasks)
        
        assert scheduler.task_graph.number_of_nodes() == 3
        assert scheduler.task_graph.has_edge('task1', 'task2')
        assert scheduler.task_graph.has_edge('task2', 'task3')

# ==================== Integration Tests ====================

class TestEnhancedProductionAgent:
    """生产级Agent系统集成测试"""
    
    @pytest.mark.asyncio
    async def test_simple_task_execution(self, agent_system):
        """测试简单任务执行"""
        result = await agent_system.execute(
            "Write a hello world function",
            {'language': 'python'}
        )
        
        assert result['success']
        assert 'execution_time' in result
        assert result['execution_time'] > 0
        
    @pytest.mark.asyncio
    async def test_complex_task_execution(self, agent_system):
        """测试复杂任务执行"""
        result = await agent_system.execute(
            "Design and implement a caching system with LRU eviction",
            {'requirements': ['thread-safe', 'TTL support']}
        )
        
        assert 'result' in result
        assert 'tasks_executed' in result
        
    @pytest.mark.asyncio
    async def test_error_handling(self, agent_system):
        """测试错误处理"""
        # 模拟错误
        with patch.object(agent_system, '_handle_with_tiny_llm', side_effect=Exception("Test error")):
            result = await agent_system.execute("Test task")
            
            assert not result['success']
            assert 'error' in result
            assert result['error'] == "Test error"
            
    def test_metrics_collection(self, agent_system):
        """测试指标收集"""
        metrics = agent_system.get_metrics()
        
        assert 'uptime_seconds' in metrics
        assert 'total_executions' in metrics
        assert 'success_rate' in metrics
        assert metrics['success_rate'] >= 0 and metrics['success_rate'] <= 1

# ==================== Client SDK Tests ====================

class TestAgentClient:
    """Agent客户端SDK测试"""
    
    @pytest.mark.asyncio
    async def test_client_execute(self):
        """测试客户端执行"""
        with aiohttp.ClientSession() as session:
            # 模拟服务器响应
            with patch.object(session, 'post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = {
                    'success': True,
                    'result': 'Test result'
                }
                mock_post.return_value.__aenter__.return_value = mock_response
                
                client = AgentClient()
                client._session = session
                
                result = await client.execute("Test task")
                
                assert result.status == TaskStatus.SUCCESS
                assert result.result == 'Test result'
                
    @pytest.mark.asyncio
    async def test_batch_execution(self):
        """测试批量执行"""
        client = AgentClient()
        
        with patch.object(client, 'execute') as mock_execute:
            mock_execute.side_effect = [
                TaskResult('task1', TaskStatus.SUCCESS, result="Result 1"),
                TaskResult('task2', TaskStatus.SUCCESS, result="Result 2"),
                TaskResult('task3', TaskStatus.FAILED, error="Error")
            ]
            
            tasks = ["Task 1", "Task 2", "Task 3"]
            results = await client.execute_batch(tasks)
            
            assert len(results) == 3
            assert results[0].status == TaskStatus.SUCCESS
            assert results[2].status == TaskStatus.FAILED

# ==================== Performance Tests ====================

class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, agent_system):
        """测试并发执行性能"""
        import time
        
        tasks = [
            agent_system.execute(f"Task {i}", {'index': i})
            for i in range(5)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        assert all(r['success'] for r in results)
        assert end_time - start_time < 60  # 应该在60秒内完成
        
    @pytest.mark.asyncio
    async def test_memory_usage(self, agent_system):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行多个任务
        for i in range(10):
            await agent_system.execute(f"Simple task {i}")
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内
        assert memory_increase < 100  # 小于100MB

# ==================== End-to-End Tests ====================

class TestEndToEnd:
    """端到端测试"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_workflow(self, temp_workspace):
        """测试完整工作流"""
        # 这个测试需要实际的服务运行
        # 可以通过环境变量跳过
        if not os.getenv('RUN_INTEGRATION_TESTS'):
            pytest.skip("Integration tests not enabled")
            
        async with AgentClient() as client:
            # 1. 创建项目
            result = await client.execute(
                "Create a Python web API project with user authentication",
                {'framework': 'FastAPI', 'database': 'PostgreSQL'}
            )
            
            assert result.status == TaskStatus.SUCCESS
            
            # 2. 格式化代码
            tools = AgentTools(client)
            code = "def hello():print('world')"
            formatted = await tools.format_python(code)
            
            assert formatted['success']
            
            # 3. 检查系统状态
            status = await client.get_status()
            assert status['status'] == 'healthy'

# ==================== Test Configuration ====================

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

def pytest_configure(config):
    """配置pytest"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )

# 运行测试的脚本
if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
    
    # 只运行单元测试
    # pytest.main([__file__, "-v", "-m", "not integration"])
    
    # 运行性能测试
    # pytest.main([__file__, "-v", "-m", "performance"])
    
    # 生成覆盖率报告
    # pytest.main([__file__, "--cov=enhanced_production_agent", "--cov-report=html"])