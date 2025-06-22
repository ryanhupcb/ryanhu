#!/usr/bin/env python3
"""
测试EnhancedProductionAgent的评分功能
"""

import asyncio
from enhanced_agent_core import EnhancedProductionAgent

async def test_scoring():
    # 初始化Agent系统
    agent = EnhancedProductionAgent()
    
    # 模拟一些任务执行
    await agent.execute("Write a Python function to calculate factorial")
    await agent.execute("Research the latest Python web frameworks")
    
    # 获取评分
    score = agent.calculate_score()
    print(f"Agent系统评分: {score}/100")
    
    # 显示详细指标
    metrics = agent.get_metrics()
    print("\n详细指标:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    asyncio.run(test_scoring())