#!/usr/bin/env python3
# Simple Runner for Agent Collaboration System
# 简单启动脚本 - 一键运行Agent系统

import asyncio
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.getcwd())

# ==================== 检查和修复导入 ====================

def check_and_fix_imports():
    """检查并修复必要的导入"""
    
    # 检查是否有修复后的文件
    if Path("complete_agent_system_fixed.py").exists():
        # 使用修复后的版本
        try:
            from complete_agent_system_fixed import CompleteAgentSystem
            return CompleteAgentSystem, "fixed"
        except ImportError as e:
            print(f"导入修复版本失败: {e}")
    
    # 尝试导入原始版本
    try:
        from complete_agent_system import CompleteAgentSystem
        return CompleteAgentSystem, "original"
    except ImportError as e:
        print(f"导入原始版本失败: {e}")
        return None, None

# ==================== 简单的Agent系统包装器 ====================

class SimpleAgentRunner:
    """简单的Agent系统运行器"""
    
    def __init__(self):
        self.system = None
        self.conversation_id = None
        
    async def initialize(self):
        """初始化系统"""
        print("正在初始化Agent系统...")
        
        # 检查导入
        AgentSystemClass, version = check_and_fix_imports()
        
        if AgentSystemClass is None:
            print("\n❌ 无法导入Agent系统")
            print("请确保以下文件之一存在：")
            print("  - complete_agent_system_fixed.py (推荐)")
            print("  - complete_agent_system.py")
            return False
            
        print(f"✓ 使用{version}版本的Agent系统")
        
        # 检查环境变量
        has_openai = bool(os.getenv('OPENAI_API_KEY'))
        has_anthropic = bool(os.getenv('ANTHROPIC_API_KEY'))
        
        if not has_openai and not has_anthropic:
            print("\n⚠️  未检测到LLM API密钥")
            print("系统将在模拟模式下运行（功能受限）")
            print("\n要使用完整功能，请设置环境变量：")
            print("  export OPENAI_API_KEY='your-key'")
            print("  export ANTHROPIC_API_KEY='your-key'")
        else:
            print("✓ 检测到LLM API密钥")
            
        # 初始化系统
        try:
            config = {
                'enable_all_frameworks': True,
                'safety_threshold': 0.95
            }
            
            self.system = AgentSystemClass(config)
            print("✓ Agent系统初始化成功")
            return True
            
        except Exception as e:
            print(f"\n❌ 系统初始化失败: {e}")
            return False
            
    async def run_interactive_mode(self):
        """运行交互模式"""
        print("""
╔════════════════════════════════════════════════════════╗
║            Agent协作系统 - 交互模式                      ║
╚════════════════════════════════════════════════════════╝

命令：
  chat <消息>     - 与Agent对话
  task <描述>     - 执行任务
  research <主题> - 研究主题
  analyze <文件>  - 分析数据
  status         - 系统状态
  help           - 显示帮助
  quit           - 退出系统

示例：
  chat 你好，介绍一下你的功能
  task 创建一个Python脚本来处理CSV文件
  research 最新的AI技术趋势
""")
        
        while True:
            try:
                # 获取用户输入
                command = input("\n> ").strip()
                
                if not command:
                    continue
                    
                # 解析命令
                parts = command.split(' ', 1)
                cmd = parts[0].lower()
                
                # 退出
                if cmd in ['quit', 'exit', 'q']:
                    print("\n再见！感谢使用Agent协作系统。")
                    break
                    
                # 帮助
                elif cmd == 'help':
                    print("""
可用命令：
  chat <消息>     - 与Agent对话
  task <描述>     - 执行任务  
  research <主题> - 研究主题
  analyze <文件>  - 分析数据
  status         - 查看系统状态
  clear          - 清空屏幕
  quit           - 退出系统
""")
                    
                # 聊天
                elif cmd == 'chat' and len(parts) > 1:
                    message = parts[1]
                    print("\n🤖 Agent正在思考...")
                    
                    try:
                        response = await self.system.chat(message, self.conversation_id)
                        self.conversation_id = response.get('conversation_id')
                        print(f"\n💬 Agent: {response['response']}")
                    except Exception as e:
                        print(f"\n❌ 聊天失败: {e}")
                        
                # 执行任务
                elif cmd == 'task' and len(parts) > 1:
                    task = parts[1]
                    print(f"\n🔧 正在执行任务: {task}")
                    
                    try:
                        result = await self.system.execute_task(task)
                        if result['overall_success']:
                            print(f"\n✅ 任务完成！")
                            print(f"执行时间: {result.get('execution_time', 0):.2f}秒")
                            
                            # 显示子任务结果
                            for i, subtask in enumerate(result.get('subtask_results', [])):
                                if subtask.get('success'):
                                    print(f"  ✓ 子任务{i+1}: 成功")
                                else:
                                    print(f"  ✗ 子任务{i+1}: 失败")
                        else:
                            print(f"\n❌ 任务执行失败")
                            
                    except Exception as e:
                        print(f"\n❌ 任务执行出错: {e}")
                        
                # 研究主题
                elif cmd == 'research' and len(parts) > 1:
                    topic = parts[1]
                    print(f"\n🔍 正在研究: {topic}")
                    
                    try:
                        result = await self.system.research_topic(topic)
                        if result.get('success'):
                            print(f"\n📚 研究完成！")
                            print(f"主题: {result.get('topic')}")
                            print(f"分析: {result.get('analysis', '研究结果已生成')}")
                        else:
                            print(f"\n❌ 研究失败")
                            
                    except Exception as e:
                        print(f"\n❌ 研究出错: {e}")
                        
                # 分析数据
                elif cmd == 'analyze' and len(parts) > 1:
                    file_path = parts[1]
                    print(f"\n📊 正在分析: {file_path}")
                    
                    try:
                        result = await self.system.analyze_data(file_path, "请分析这个文件")
                        if result.get('success'):
                            print(f"\n✅ 分析完成！")
                            print(f"结果: {result.get('result', {})}")
                        else:
                            print(f"\n❌ 分析失败: {result.get('error')}")
                            
                    except Exception as e:
                        print(f"\n❌ 分析出错: {e}")
                        
                # 系统状态
                elif cmd == 'status':
                    print("\n📊 系统状态：")
                    try:
                        health = await self.system.health_check()
                        print(f"  状态: {health.get('status', 'unknown')}")
                        
                        components = health.get('components', {})
                        print("  组件状态:")
                        for comp, status in components.items():
                            print(f"    - {comp}: {status}")
                            
                        # 显示可用工具
                        tools = self.system.enhanced_tool_registry.list_tools()
                        print(f"\n  可用工具 ({len(tools)}个):")
                        for tool in tools[:5]:
                            print(f"    - {tool}")
                        if len(tools) > 5:
                            print(f"    ... 还有{len(tools)-5}个工具")
                            
                        # 显示可用Agent
                        agents = list(self.system.communication_bus.agents.keys())
                        print(f"\n  可用Agent ({len(agents)}个):")
                        for agent in agents:
                            print(f"    - {agent}")
                            
                    except Exception as e:
                        print(f"❌ 获取状态失败: {e}")
                        
                # 清屏
                elif cmd == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                    
                # 无效命令
                else:
                    print(f"❓ 未知命令: {cmd}")
                    print("输入 'help' 查看可用命令")
                    
            except KeyboardInterrupt:
                print("\n\n使用 'quit' 命令退出系统")
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                print("您可以继续使用其他命令")

# ==================== 主程序 ====================

async def main():
    """主程序入口"""
    
    print("""
    🤖 Agent协作系统启动器
    ========================
    """)
    
    # 创建运行器
    runner = SimpleAgentRunner()
    
    # 初始化系统
    if await runner.initialize():
        # 运行交互模式
        await runner.run_interactive_mode()
    else:
        print("\n系统初始化失败，请检查错误信息")
        
        # 提供修复建议
        print("\n建议的修复步骤：")
        print("1. 使用修复后的文件：")
        print("   - 将 complete_agent_system_fixed.py 重命名为 complete_agent_system.py")
        print("   - 或直接使用 complete_agent_system_fixed.py")
        print("\n2. 安装基本依赖：")
        print("   pip install numpy")
        print("\n3. 设置环境变量（可选）：")
        print("   export OPENAI_API_KEY='your-key'")

if __name__ == "__main__":
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 错误：需要Python 3.7或更高版本")
        print(f"当前版本：{sys.version}")
        sys.exit(1)
        
    # 运行主程序
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        sys.exit(1)