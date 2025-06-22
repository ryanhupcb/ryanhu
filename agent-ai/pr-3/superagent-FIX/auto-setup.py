#!/usr/bin/env python3
# Auto Setup Script for Agent Collaboration System
# 自动安装和修复脚本 - 一键配置整个系统

import os
import sys
import subprocess
import shutil
from pathlib import Path

class AutoSetup:
    """自动设置和修复系统"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.fixed_items = []
        
    def print_header(self, text):
        """打印标题"""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}\n")
        
    def print_status(self, message, status="info"):
        """打印状态信息"""
        symbols = {
            "info": "ℹ️ ",
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌",
            "fix": "🔧"
        }
        print(f"{symbols.get(status, '')} {message}")
        
    def check_python_version(self):
        """检查Python版本"""
        self.print_status("检查Python版本...", "info")
        
        version = sys.version_info
        if version.major >= 3 and version.minor >= 7:
            self.print_status(f"Python {version.major}.{version.minor}.{version.micro} - 符合要求", "success")
            return True
        else:
            self.print_status(f"Python版本过低 ({version.major}.{version.minor})，需要3.7+", "error")
            self.errors.append("Python版本不符合要求")
            return False
            
    def install_package(self, package):
        """安装单个包"""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            return True
        except:
            return False
            
    def check_and_install_dependencies(self):
        """检查并安装依赖"""
        self.print_status("检查和安装必要的依赖...", "info")
        
        # 最小依赖列表
        required_packages = [
            ("numpy", "数值计算"),
            ("aiohttp", "异步HTTP"),
        ]
        
        optional_packages = [
            ("pandas", "数据处理"),
            ("beautifulsoup4", "网页解析"),
            ("requests", "HTTP请求"),
        ]
        
        # 检查并安装必需包
        for package, description in required_packages:
            try:
                __import__(package)
                self.print_status(f"{package} ({description}) - 已安装", "success")
            except ImportError:
                self.print_status(f"安装 {package} ({description})...", "fix")
                if self.install_package(package):
                    self.print_status(f"{package} 安装成功", "success")
                    self.fixed_items.append(f"安装了 {package}")
                else:
                    self.print_status(f"{package} 安装失败", "error")
                    self.errors.append(f"无法安装 {package}")
                    
        # 检查可选包
        print("\n可选依赖：")
        for package, description in optional_packages:
            try:
                __import__(package)
                self.print_status(f"{package} ({description}) - 已安装", "success")
            except ImportError:
                self.print_status(f"{package} ({description}) - 未安装（可选）", "warning")
                self.warnings.append(f"{package} 未安装（可选）")
                
    def create_fixed_files(self):
        """创建修复后的文件"""
        self.print_status("创建修复后的文件...", "info")
        
        # 检查是否已存在修复后的文件
        if Path("complete_agent_system_fixed.py").exists():
            self.print_status("修复后的文件已存在", "success")
            return
            
        # 如果原始文件存在，复制并修复
        if Path("complete_agent_system.py").exists():
            self.print_status("备份原始文件...", "fix")
            shutil.copy("complete_agent_system.py", "complete_agent_system.backup.py")
            
            # 这里应该添加实际的修复逻辑
            # 但由于我们已经提供了fixed版本，这里跳过
            self.print_status("请使用提供的 complete_agent_system_fixed.py", "warning")
            self.warnings.append("需要手动使用修复后的文件")
        else:
            self.print_status("未找到 complete_agent_system.py", "warning")
            self.warnings.append("缺少主系统文件")
            
    def create_env_file(self):
        """创建环境变量文件"""
        self.print_status("创建环境变量模板...", "info")
        
        env_template = """# Agent协作系统环境变量
# 请填入您的API密钥

# LLM API密钥（至少需要一个）
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# 可选配置
# SMTP_SERVER=smtp.gmail.com
# EMAIL_ADDRESS=your-email@gmail.com
# EMAIL_PASSWORD=your-app-password
"""
        
        if not Path(".env").exists():
            with open(".env", "w") as f:
                f.write(env_template)
            self.print_status("创建了 .env 文件模板", "success")
            self.fixed_items.append("创建了环境变量模板")
            
            print("\n" + "="*60)
            print("重要：请编辑 .env 文件并添加您的API密钥")
            print("使用文本编辑器打开 .env 文件")
            print("="*60)
        else:
            self.print_status(".env 文件已存在", "success")
            
    def create_quick_start_script(self):
        """创建快速启动脚本"""
        self.print_status("创建快速启动脚本...", "info")
        
        # Windows批处理文件
        if sys.platform == "win32":
            with open("start.bat", "w") as f:
                f.write("""@echo off
echo Starting Agent Collaboration System...
python run_agent_system.py
pause
""")
            self.print_status("创建了 start.bat (Windows)", "success")
            
        # Unix/Linux shell脚本
        else:
            with open("start.sh", "w") as f:
                f.write("""#!/bin/bash
echo "Starting Agent Collaboration System..."
python3 run_agent_system.py
""")
            os.chmod("start.sh", 0o755)
            self.print_status("创建了 start.sh (Unix/Linux)", "success")
            
        self.fixed_items.append("创建了快速启动脚本")
        
    def test_system(self):
        """测试系统是否可以运行"""
        self.print_status("测试系统...", "info")
        
        try:
            # 尝试导入主模块
            if Path("complete_agent_system_fixed.py").exists():
                sys.path.insert(0, os.getcwd())
                from complete_agent_system_fixed import CompleteAgentSystem
                self.print_status("成功导入修复后的系统", "success")
                return True
            elif Path("complete_agent_system.py").exists():
                sys.path.insert(0, os.getcwd())
                from complete_agent_system import CompleteAgentSystem
                self.print_status("成功导入原始系统", "success")
                return True
            else:
                self.print_status("未找到系统文件", "error")
                self.errors.append("缺少主系统文件")
                return False
                
        except ImportError as e:
            self.print_status(f"导入失败: {e}", "error")
            self.errors.append(f"系统导入错误: {e}")
            return False
            
    def show_summary(self):
        """显示总结"""
        self.print_header("设置完成总结")
        
        if self.fixed_items:
            print("✅ 已完成的修复：")
            for item in self.fixed_items:
                print(f"   - {item}")
                
        if self.warnings:
            print("\n⚠️  警告：")
            for warning in self.warnings:
                print(f"   - {warning}")
                
        if self.errors:
            print("\n❌ 错误：")
            for error in self.errors:
                print(f"   - {error}")
                
        print("\n" + "="*60)
        
        if not self.errors:
            print("✅ 系统设置完成！")
            print("\n下一步：")
            print("1. 编辑 .env 文件，添加您的API密钥")
            print("2. 运行系统：")
            if sys.platform == "win32":
                print("   - 双击 start.bat")
                print("   - 或运行: python run_agent_system.py")
            else:
                print("   - 运行: ./start.sh")
                print("   - 或运行: python3 run_agent_system.py")
        else:
            print("❌ 设置过程中遇到错误，请手动修复")
            
    def run(self):
        """运行自动设置"""
        self.print_header("Agent协作系统自动设置")
        
        # 1. 检查Python版本
        if not self.check_python_version():
            return
            
        # 2. 检查并安装依赖
        self.check_and_install_dependencies()
        
        # 3. 创建修复文件
        self.create_fixed_files()
        
        # 4. 创建环境变量文件
        self.create_env_file()
        
        # 5. 创建快速启动脚本
        self.create_quick_start_script()
        
        # 6. 测试系统
        self.test_system()
        
        # 7. 显示总结
        self.show_summary()

def main():
    """主函数"""
    setup = AutoSetup()
    
    try:
        setup.run()
    except KeyboardInterrupt:
        print("\n\n设置被用户中断")
    except Exception as e:
        print(f"\n\n设置过程中发生错误: {e}")
        
    # 等待用户确认
    if sys.platform == "win32":
        input("\n按Enter键退出...")

if __name__ == "__main__":
    main()