#!/usr/bin/env python3
# Minimal Runner for Agent Collaboration System
# 最小化运行脚本 - 用最少的依赖运行系统

import asyncio
import os
import sys
import json
from typing import Dict, Any

# 添加当前目录到Python路径
sys.path.insert(0, os.getcwd())

# ==================== 模拟的LLM提供者 ====================

class MockLLMProvider:
    """模拟的LLM提供者（用于测试）"""
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成模拟响应"""
        return {
            'content': f"Mock response for: {prompt[:50]}...",
            'tool_calls': None,
            'usage': {'total_tokens': 100},
            'finish_reason': 'stop'
        }
        
    async def embed(self, text: str):
        """生成模拟嵌入"""
        import numpy as np
        return np.random.rand(1536)

# ==================== 最小化系统 ====================

class MinimalAgentSystem:
    """最小化的Agent系统"""
    
    def __init__(self):
        print("初始化最小化Agent系统...")
        self.llm = MockLLMProvider()
        self.tools = {}
        self.agents = {}
        
    async def chat(self, message: str) -> Dict[str, Any]:
        """简单的聊天接口"""
        print(f"\n用户: {message}")
        
        # 使用模拟LLM生成响应
        response = await self.llm.generate(message)
        
        # 简单的响应逻辑
        if "help" in message.lower():
            response['content'] = """
我是一个最小化的Agent系统。当前功能：
- 基本对话
- 简单的任务处理
- 系统状态检查

注意：这是一个测试模式，没有连接真实的LLM。
要使用完整功能，请设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY。
"""
        elif "status" in message.lower():
            response['content'] = """
系统状态：
- 运行模式：最小化测试模式
- 可用Agent：0
- 可用工具：0
- LLM连接：模拟模式
"""
        
        print(f"助手: {response['content']}")
        
        return {
            'response': response['content'],
            'conversation_id': 'test-session'
        }
        
    async def execute_task(self, task: str) -> Dict[str, Any]:
        """执行任务（简化版）"""
        print(f"\n执行任务: {task}")
        
        # 模拟任务执行
        await asyncio.sleep(1)
        
        return {
            'overall_success': True,
            'execution_time': 1.0,
            'result': f"任务 '{task}' 已完成（模拟模式）"
        }

# ==================== 主程序 ====================

async def run_minimal_system():
    """运行最小化系统"""
    
    print("""
╔════════════════════════════════════════════════════════╗
║          Agent协作系统 - 最小化测试模式                   ║
╚════════════════════════════════════════════════════════╝

注意：当前运行在最小化模式，不需要LLM API密钥。
这个模式用于测试基本功能和系统结构。

命令：
- chat <消息> - 发送聊天消息
- task <描述> - 执行任务
- status - 查看系统状态
- quit - 退出
""")
    
    system = MinimalAgentSystem()
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if not command:
                continue
                
            parts = command.split(' ', 1)
            cmd = parts[0].lower()
            
            if cmd == 'quit':
                print("再见！")
                break
                
            elif cmd == 'chat' and len(parts) > 1:
                await system.chat(parts[1])
                
            elif cmd == 'task' and len(parts) > 1:
                result = await system.execute_task(parts[1])
                print(f"结果: {result['result']}")
                
            elif cmd == 'status':
                await system.chat("status")
                
            elif cmd == 'help':
                await system.chat("help")
                
            else:
                print("无效命令。输入 'help' 查看帮助。")
                
        except KeyboardInterrupt:
            print("\n使用 'quit' 退出。")
        except Exception as e:
            print(f"错误: {e}")

async def test_full_system():
    """尝试运行完整系统"""
    try:
        # 尝试导入完整系统
        from base_components import ResearchOptimizedProductionSystem
        from complete_agent_system import CompleteAgentSystem
        from extended_tools import integrate_with_system
        from system_integration import ExtendedCompleteAgentSystem
        
        print("✓ 成功导入所有模块")
        
        # 检查API密钥
        if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
            print("\n⚠ 未设置LLM API密钥")
            print("将运行最小化测试模式...")
            await run_minimal_system()
        else:
            print("\n✓ 检测到API密钥")
            print("启动完整的Agent协作系统...")
            
            # 导入并运行主程序
            from agent_collaboration_runner import main
            await main()
            
    except ImportError as e:
        print(f"\n导入错误: {e}")
        print("运行最小化测试模式...")
        await run_minimal_system()
    except Exception as e:
        print(f"\n系统错误: {e}")
        print("运行最小化测试模式...")
        await run_minimal_system()

def main():
    """主入口"""
    print("=== Agent协作系统启动器 ===\n")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("错误：需要Python 3.8或更高版本")
        sys.exit(1)
        
    # 尝试运行系统
    asyncio.run(test_full_system())

if __name__ == "__main__":
    main()