#!/usr/bin/env python3
# Fix Imports Script
# 修复导入问题的脚本

import os
import shutil
from pathlib import Path

def fix_complete_agent_system():
    """修复 complete_agent_system.py 的导入问题"""
    
    print("修复 complete_agent_system.py 的导入...")
    
    # 读取原始文件
    original_file = "complete_agent_system.py"
    if not Path(original_file).exists():
        print(f"错误: {original_file} 不存在")
        return False
        
    # 备份原始文件
    backup_file = f"{original_file}.backup"
    shutil.copy(original_file, backup_file)
    print(f"已创建备份: {backup_file}")
    
    # 读取文件内容
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 添加基础组件导入
    import_statement = """# Fixed imports - 导入基础组件
from base_components import (
    EnterpriseVectorDatabase,
    DualLayerMemorySystem,
    AdvancedGraphOfThoughts,
    SemanticToolRegistry,
    ConstitutionalAIFramework,
    CircuitBreakerState,
    CircuitBreaker,
    ResearchOptimizedProductionSystem,
    TaskContext,
    MetricsCollector
)

# 添加缺失的collections导入
from collections import defaultdict

"""
    
    # 在文件开头添加导入（在第一个导入语句之前）
    if "from base_components import" not in content:
        # 找到第一个import语句的位置
        import_pos = content.find("import")
        if import_pos != -1:
            # 在第一个import之前插入
            content = content[:import_pos] + import_statement + content[import_pos:]
        else:
            # 如果没有找到import，就在文件开头添加
            content = import_statement + content
            
    # 写回文件
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"✓ 已修复 {original_file}")
    return True

def create_minimal_requirements():
    """创建最小依赖文件"""
    
    minimal_requirements = """# Minimal requirements for Agent Collaboration System
# 最小依赖列表 - 仅包含必需的包

# Core dependencies
aiohttp>=3.8.0
numpy>=1.21.0
pandas>=1.3.0

# Basic tools
beautifulsoup4>=4.10.0
requests>=2.26.0

# Optional but recommended
redis>=4.0.0
python-dotenv>=0.19.0

# For development
pytest>=6.2.0
pytest-asyncio>=0.16.0
"""
    
    with open("requirements_minimal.txt", "w") as f:
        f.write(minimal_requirements)
        
    print("✓ 已创建 requirements_minimal.txt")

def create_env_template():
    """创建环境变量模板"""
    
    env_template = """# Environment variables template
# 环境变量模板 - 复制为 .env 并填入实际值

# LLM API Keys (至少需要一个)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional - Email configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_ADDRESS=your-email@gmail.com
EMAIL_PASSWORD=your-app-specific-password

# Optional - External services
REDIS_HOST=localhost
REDIS_PORT=6379
RABBITMQ_URL=amqp://guest:guest@localhost/
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
        
    print("✓ 已创建 .env.template")

def create_quick_test():
    """创建快速测试脚本"""
    
    test_script = """#!/usr/bin/env python3
# Quick test script
# 快速测试脚本 - 验证基本功能

import asyncio
import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.getcwd())

async def test_basic_functionality():
    '''测试基本功能'''
    
    print("测试基本导入...")
    try:
        from base_components import ResearchOptimizedProductionSystem
        from complete_agent_system import CompleteAgentSystem
        print("✓ 导入成功")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
        
    print("\\n测试系统初始化...")
    try:
        # 设置测试配置
        config = {
            'enable_all_frameworks': False,
            'safety_threshold': 0.9
        }
        
        # 注意：实际运行需要设置API密钥
        if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
            print("⚠ 警告: 未设置LLM API密钥，某些功能将无法使用")
            print("  请设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY 环境变量")
            return True  # 仍然返回True，因为导入成功了
            
        system = CompleteAgentSystem(config)
        print("✓ 系统初始化成功")
        
        # 测试基本功能
        print("\\n测试健康检查...")
        health = await system.health_check()
        print(f"✓ 健康状态: {health['status']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 系统初始化失败: {e}")
        return False

if __name__ == "__main__":
    print("=== Agent系统快速测试 ===\\n")
    
    # 运行测试
    success = asyncio.run(test_basic_functionality())
    
    if success:
        print("\\n✓ 基本测试通过！")
        print("\\n下一步:")
        print("1. 设置API密钥: export OPENAI_API_KEY='your-key'")
        print("2. 运行系统: python agent_collaboration_runner.py")
    else:
        print("\\n✗ 测试失败，请检查错误信息")
"""
    
    with open("quick_test.py", "w") as f:
        f.write(test_script)
        
    os.chmod("quick_test.py", 0o755)
    print("✓ 已创建 quick_test.py")

def main():
    """主函数"""
    print("=== 修复Agent协作系统 ===\n")
    
    # 1. 检查base_components.py是否存在
    if not Path("base_components.py").exists():
        print("错误: base_components.py 不存在")
        print("请先创建该文件（使用提供的base_components.py内容）")
        return
        
    # 2. 修复导入
    fix_complete_agent_system()
    
    # 3. 创建辅助文件
    create_minimal_requirements()
    create_env_template()
    create_quick_test()
    
    print("\n✓ 修复完成！")
    print("\n建议的下一步操作：")
    print("1. 安装最小依赖: pip install -r requirements_minimal.txt")
    print("2. 配置环境变量: cp .env.template .env && nano .env")
    print("3. 运行快速测试: python quick_test.py")
    print("4. 启动系统: python agent_collaboration_runner.py")

if __name__ == "__main__":
    main()