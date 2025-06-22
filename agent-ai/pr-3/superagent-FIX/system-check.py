#!/usr/bin/env python3
# System Check Script
# 系统检查脚本 - 验证Agent协作系统是否可以运行

import sys
import os
import importlib
import subprocess
from pathlib import Path

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(message, status):
    """打印状态信息"""
    if status == "OK":
        print(f"{Colors.GREEN}✓{Colors.END} {message}")
    elif status == "WARNING":
        print(f"{Colors.YELLOW}⚠{Colors.END} {message}")
    elif status == "ERROR":
        print(f"{Colors.RED}✗{Colors.END} {message}")
    else:
        print(f"{Colors.BLUE}→{Colors.END} {message}")

def check_python_version():
    """检查Python版本"""
    print_status("Checking Python version...", "INFO")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro}", "OK")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor} (需要 3.8+)", "ERROR")
        return False

def check_required_files():
    """检查必需的文件"""
    print_status("Checking required files...", "INFO")
    required_files = [
        "complete_agent_system.py",
        "extended_tools.py", 
        "system_integration.py",
        "agent_collaboration_runner.py",
        "base_components.py"  # 新增的基础组件文件
    ]
    
    all_present = True
    for file in required_files:
        if Path(file).exists():
            print_status(f"  {file}", "OK")
        else:
            print_status(f"  {file} - Missing", "ERROR")
            all_present = False
            
    return all_present

def check_imports():
    """检查导入"""
    print_status("Checking module imports...", "INFO")
    
    # 添加当前目录到Python路径
    sys.path.insert(0, os.getcwd())
    
    modules_to_check = [
        ("base_components", "基础组件"),
        ("complete_agent_system", "主系统模块"),
        ("extended_tools", "扩展工具模块"),
        ("system_integration", "系统集成模块")
    ]
    
    all_imported = True
    for module_name, description in modules_to_check:
        try:
            importlib.import_module(module_name)
            print_status(f"  {description} ({module_name})", "OK")
        except ImportError as e:
            print_status(f"  {description} ({module_name}) - {str(e)}", "ERROR")
            all_imported = False
            
    return all_imported

def check_dependencies():
    """检查Python依赖"""
    print_status("Checking Python dependencies...", "INFO")
    
    critical_deps = [
        "aiohttp",
        "numpy", 
        "pandas",
        "asyncio"
    ]
    
    optional_deps = [
        "openai",
        "anthropic",
        "redis",
        "aio_pika",
        "faiss-cpu",
        "sentence-transformers",
        "yfinance",
        "googletrans",
        "qrcode"
    ]
    
    all_critical = True
    
    # 检查关键依赖
    for dep in critical_deps:
        try:
            __import__(dep.replace("-", "_"))
            print_status(f"  {dep}", "OK")
        except ImportError:
            print_status(f"  {dep} - Not installed", "ERROR")
            all_critical = False
            
    # 检查可选依赖
    for dep in optional_deps:
        try:
            __import__(dep.replace("-", "_"))
            print_status(f"  {dep}", "OK")
        except ImportError:
            print_status(f"  {dep} - Not installed (optional)", "WARNING")
            
    return all_critical

def check_environment_vars():
    """检查环境变量"""
    print_status("Checking environment variables...", "INFO")
    
    env_vars = {
        "OPENAI_API_KEY": "OpenAI API密钥",
        "ANTHROPIC_API_KEY": "Anthropic API密钥",
        "SMTP_SERVER": "SMTP服务器（可选）",
        "EMAIL_ADDRESS": "邮件地址（可选）"
    }
    
    has_llm_key = False
    for var, description in env_vars.items():
        if os.getenv(var):
            print_status(f"  {var} - {description}", "OK")
            if var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                has_llm_key = True
        else:
            if var in ["SMTP_SERVER", "EMAIL_ADDRESS"]:
                print_status(f"  {var} - {description}", "WARNING")
            else:
                print_status(f"  {var} - {description}", "WARNING")
                
    if not has_llm_key:
        print_status("  至少需要一个LLM API密钥", "ERROR")
        return False
        
    return True

def create_fix_script():
    """创建修复脚本"""
    fix_script = '''#!/bin/bash
# 修复脚本 - 安装缺失的依赖

echo "开始修复Agent协作系统..."

# 创建虚拟环境
if [ ! -d "agent_env" ]; then
    echo "创建虚拟环境..."
    python3 -m venv agent_env
fi

# 激活虚拟环境
source agent_env/bin/activate

# 升级pip
pip install --upgrade pip

# 安装基础依赖
echo "安装基础依赖..."
pip install aiohttp numpy pandas asyncio

# 修复complete_agent_system.py的导入
echo "修复导入问题..."
if [ -f "complete_agent_system.py" ]; then
    # 在文件开头添加base_components导入
    sed -i '1s/^/from base_components import *\\n/' complete_agent_system.py
fi

echo "修复完成！"
echo "请运行以下命令设置API密钥："
echo "export OPENAI_API_KEY='your-key'"
echo "export ANTHROPIC_API_KEY='your-key'"
'''
    
    with open("fix_system.sh", "w") as f:
        f.write(fix_script)
        
    os.chmod("fix_system.sh", 0o755)
    print_status("Created fix_system.sh", "OK")

def main():
    """主检查函数"""
    print(f"\n{Colors.BLUE}=== Agent协作系统检查 ==={Colors.END}\n")
    
    checks = [
        ("Python版本", check_python_version),
        ("必需文件", check_required_files),
        ("环境变量", check_environment_vars),
        ("Python依赖", check_dependencies),
        ("模块导入", check_imports)
    ]
    
    all_passed = True
    results = {}
    
    for name, check_func in checks:
        try:
            result = check_func()
            results[name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print_status(f"{name} - 检查失败: {str(e)}", "ERROR")
            results[name] = False
            all_passed = False
        print()
        
    # 总结
    print(f"{Colors.BLUE}=== 检查结果 ==={Colors.END}\n")
    
    if all_passed:
        print_status("所有检查通过！系统可以运行。", "OK")
        print("\n运行以下命令启动系统：")
        print("  python agent_collaboration_runner.py")
    else:
        print_status("系统存在问题，需要修复。", "ERROR")
        create_fix_script()
        print("\n建议操作：")
        print("1. 运行修复脚本: ./fix_system.sh")
        print("2. 安装缺失的依赖: pip install -r requirements.txt")
        print("3. 设置必要的环境变量")
        
    # 详细问题
    print(f"\n{Colors.BLUE}详细结果：{Colors.END}")
    for check_name, passed in results.items():
        status = "OK" if passed else "FAILED"
        print(f"  {check_name}: {status}")

if __name__ == "__main__":
    main()