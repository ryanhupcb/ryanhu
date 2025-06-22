#!/usr/bin/env python3
"""
快速启动脚本 - 极限性能AI Agent系统
自动设置环境并启动系统
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {sys.version}")
        print("需要Python 3.8或更高版本")
        return False
    print(f"✅ Python版本: {sys.version}")
    return True

def create_virtual_env():
    """创建虚拟环境"""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ 虚拟环境已存在")
        return True
    
    print("创建虚拟环境...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ 虚拟环境创建成功")
        return True
    except subprocess.CalledProcessError:
        print("❌ 创建虚拟环境失败")
        return False

def get_pip_command():
    """获取pip命令"""
    system = platform.system()
    if system == "Windows":
        return str(Path("venv/Scripts/pip"))
    else:
        return str(Path("venv/bin/pip"))

def get_python_command():
    """获取python命令"""
    system = platform.system()
    if system == "Windows":
        return str(Path("venv/Scripts/python"))
    else:
        return str(Path("venv/bin/python"))

def install_dependencies():
    """安装依赖"""
    pip_cmd = get_pip_command()
    
    # 首先升级pip
    print("\n升级pip...")
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], capture_output=True)
    
    # 安装核心依赖
    print("\n安装核心依赖...")
    core_deps = [
        "numpy",
        "pandas",
        "matplotlib",
        "networkx",
        "aiohttp",
        "pyyaml",
        "psutil",
        "tqdm"
    ]
    
    for dep in core_deps:
        print(f"安装 {dep}...")
        result = subprocess.run(
            [pip_cmd, "install", dep],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ {dep} 安装成功")
        else:
            print(f"⚠️  {dep} 安装失败，但继续...")
    
    # 尝试安装可选依赖
    print("\n安装可选依赖...")
    optional_deps = ["anthropic", "tk", "redis"]
    
    for dep in optional_deps:
        print(f"尝试安装 {dep}...")
        subprocess.run(
            [pip_cmd, "install", dep],
            capture_output=True,
            text=True
        )

def create_directories():
    """创建必要的目录"""
    print("\n创建目录结构...")
    dirs = ["data", "logs", "cache", "output", "config"]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ 创建目录: {dir_name}/")

def create_default_config():
    """创建默认配置文件"""
    config_path = Path("config.json")
    if config_path.exists():
        print("✅ 配置文件已存在")
        return
    
    print("\n创建默认配置文件...")
    default_config = {
        "app_name": "Extreme Performance AI Agent System",
        "version": "1.0.0",
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "model": "claude-3-5-sonnet-20241022",
        "modules": {
            "code_generator": True,
            "learning_system": True,
            "multi_agent": True,
            "decision_engine": True,
            "knowledge_graph": True,
            "analyzer": True,
            "visualizer": True,
            "debugger": True
        },
        "paths": {
            "data": "data",
            "logs": "logs",
            "cache": "cache",
            "output": "output"
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4, ensure_ascii=False)
    
    print("✅ 配置文件创建成功")

def check_api_key():
    """检查API密钥"""
    print("\n检查API密钥...")
    
    # 检查环境变量
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print("✅ 在环境变量中找到API密钥")
        return True
    
    # 检查配置文件
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if config.get("api_key"):
                print("✅ 在配置文件中找到API密钥")
                return True
    
    print("⚠️  未找到API密钥")
    print("请设置环境变量 ANTHROPIC_API_KEY 或在 config.json 中配置")
    return False

def create_demo_files():
    """创建演示文件"""
    print("\n创建演示文件...")
    
    # 创建简单的演示模块
    demo_module = '''"""
演示模块 - 用于测试系统功能
"""

class DemoCodeGenerator:
    """简化的代码生成器"""
    
    async def generate_code(self, requirements):
        """生成演示代码"""
        return {
            'code': f'# Generated code for: {requirements.get("name", "Demo")}\\n'
                   f'def main():\\n'
                   f'    print("Hello from generated code!")\\n'
                   f'    return True',
            'language': 'python',
            'confidence': 0.95,
            'success': True
        }

class DemoAnalyzer:
    """简化的分析器"""
    
    async def analyze(self, data, analysis_types=None):
        """分析演示数据"""
        import numpy as np
        
        values = data.get('values', [])
        if values:
            return {
                'summary': {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values)
                },
                'insights': [
                    f'数据集包含 {len(values)} 个值',
                    f'平均值为 {np.mean(values):.2f}',
                    f'标准差为 {np.std(values):.2f}'
                ],
                'success': True
            }
        return {'error': 'No data provided'}
'''
    
    with open("demo_modules.py", 'w', encoding='utf-8') as f:
        f.write(demo_module)
    
    print("✅ 演示模块创建成功")

def start_system():
    """启动系统"""
    python_cmd = get_python_command()
    
    print("\n" + "="*60)
    print("系统准备就绪！")
    print("="*60)
    
    print("\n选择启动模式:")
    print("1. 命令行界面 (CLI) - 推荐")
    print("2. 图形界面 (GUI) - 需要tkinter")
    print("3. 演示模式 - 简化功能")
    print("4. 退出")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == "1":
        print("\n启动命令行界面...")
        subprocess.run([python_cmd, "integrated_agent_runner.py"])
    elif choice == "2":
        print("\n启动图形界面...")
        subprocess.run([python_cmd, "integrated_agent_runner.py", "--gui"])
    elif choice == "3":
        print("\n启动演示模式...")
        # 创建简单的演示脚本
        demo_script = '''
import asyncio
from demo_modules import DemoCodeGenerator, DemoAnalyzer

async def demo():
    print("\\n=== 极限性能AI Agent系统 - 演示模式 ===\\n")
    
    # 演示代码生成
    print("1. 代码生成演示:")
    generator = DemoCodeGenerator()
    result = await generator.generate_code({
        'name': 'DemoFunction',
        'description': 'A demo function'
    })
    print("生成的代码:")
    print(result['code'])
    
    # 演示数据分析
    print("\\n2. 数据分析演示:")
    analyzer = DemoAnalyzer()
    result = await analyzer.analyze({
        'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    print("分析结果:")
    for key, value in result['summary'].items():
        print(f"  {key}: {value}")
    
    print("\\n演示完成！")

if __name__ == "__main__":
    asyncio.run(demo())
'''
        with open("demo_runner.py", 'w', encoding='utf-8') as f:
            f.write(demo_script)
        
        subprocess.run([python_cmd, "demo_runner.py"])
    else:
        print("\n退出设置程序")

def main():
    """主函数"""
    print("="*60)
    print("极限性能AI Agent系统 - 快速启动脚本")
    print("="*60)
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 创建虚拟环境
    if not create_virtual_env():
        return
    
    # 安装依赖
    install_dependencies()
    
    # 创建目录
    create_directories()
    
    # 创建配置文件
    create_default_config()
    
    # 检查API密钥
    check_api_key()
    
    # 创建演示文件
    create_demo_files()
    
    # 启动系统
    start_system()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
