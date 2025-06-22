# Local Agent System - 快速启动指南

## 🚀 系统概述

Local Agent System 是一个类似 Manus 的本地部署 AI Agent 系统，专门设计用于让任何人都能轻松开发复杂软件和执行通用任务。

### 主要特性

- **多智能体协作**: ReAct + Tree of Thoughts 混合架构
- **智能成本优化**: 80% 简单任务用本地 DeepSeek-Coder，复杂任务用 Claude
- **全面的工具集成**: 代码开发、文件操作、浏览器控制、系统操作等
- **幻觉缓解**: 自动检测和纠正 AI 幻觉
- **完整的可观测性**: 监控、日志、性能分析

## 📋 系统要求

- **硬件**: 4核 CPU, 8GB RAM, 512MB 显存
- **软件**: Python 3.8+, Docker, Redis, PostgreSQL
- **API Keys**: Claude, Qwen (必需), DeepSeek, GitHub (可选)

## 🛠️ 快速安装

### 1. 克隆仓库

```bash
git clone https://github.com/your-repo/local-agent-system.git
cd local-agent-system
```

### 2. 初始化系统

```bash
# 运行初始化向导
python main.py --init

# 或者手动创建配置
cp .env.example .env
# 编辑 .env 文件，填入你的 API Keys
```

### 3. 使用 Docker Compose 启动

```bash
# 构建并启动所有服务
docker-compose up -d

# 检查服务状态
docker-compose ps
```

### 4. 本地开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动系统
python main.py
```

## 💻 使用方式

### 1. Web UI (推荐)

打开浏览器访问: http://localhost:8501

```python
# 或指定端口启动
python main.py --mode ui
```

### 2. CLI 模式

```bash
# 交互式 CLI
python main.py --mode cli

# 直接执行任务
python agent.py run -t code "创建一个 REST API"
```

### 3. API 模式

```bash
# 启动 API 服务器
python main.py --mode api

# API 文档: http://localhost:8000/docs
```

### 4. 完整模式 (默认)

```bash
# 同时启动 API + UI + 后台任务
python main.py
```

## 📚 使用示例

### 代码生成

```python
# CLI 示例
> 创建一个 Python Web 爬虫，爬取新闻网站的头条并保存到 JSON 文件

# API 示例
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "创建一个任务管理 REST API",
    "task_type": "code_generation"
  }'
```

### 研究任务

```python
> 研究最新的 AI Agent 架构趋势，总结关键发现
```

### 浏览器自动化

```python
> 自动化以下任务：打开 GitHub，搜索 Python 爬虫项目，提取前 10 个项目信息
```

### 文件操作

```python
> 整理 downloads 文件夹，按文件类型分类到不同子文件夹
```

## 🔧 配置说明

### 环境变量 (.env)

```bash
# 必需的 API Keys
CLAUDE_API_KEY=your_claude_api_key
QWEN_API_KEY=your_qwen_api_key

# 可选配置
DEEPSEEK_API_KEY=  # 留空使用本地模型
GITHUB_TOKEN=your_github_token
USE_LOCAL_DEEPSEEK=true

# 成本控制
MAX_COST_PER_REQUEST=2.0  # 单个请求最大成本 ($)
MAX_DAILY_COST=100.0      # 每日最大成本 ($)

# 性能设置
MAX_AGENTS=10
MAX_CONCURRENT_TASKS=5
TASK_TIMEOUT=300  # 秒
```

### 高级配置

```yaml
# config/performance.yaml
model_selection:
  simple_tasks:
    primary: deepseek-local
    fallback: qwen
  complex_tasks:
    primary: claude
    fallback: qwen

cost_control:
  max_cost_per_task: 0.5
  alert_threshold: 0.8
```

## 📊 监控和调试

### 查看系统状态

```bash
# CLI
python agent.py status

# Web UI
访问 http://localhost:8501 -> System Info 标签

# Prometheus 指标
http://localhost:8001/metrics
```

### 查看日志

```bash
# 实时日志
tail -f logs/agent_system.log

# 错误日志
tail -f logs/agent_system_error.log

# 结构化日志查询
cat logs/agent_system.log | jq '.event_type == "task_completed"'
```

### 性能分析

```python
# 获取性能报告
curl http://localhost:8000/performance/report

# 查看慢操作
curl http://localhost:8000/performance/slow-operations
```

## 🚨 常见问题

### 1. API Key 错误

```
错误: Claude API authentication failed
解决: 检查 .env 文件中的 CLAUDE_API_KEY 是否正确
```

### 2. 内存不足

```
错误: Out of memory
解决: 
- 增加 Docker 内存限制
- 减少 MAX_CONCURRENT_TASKS
- 启用任务队列
```

### 3. 连接错误

```
错误: Cannot connect to Redis/PostgreSQL
解决:
- 确保服务已启动: docker-compose ps
- 检查端口是否被占用
- 查看服务日志: docker-compose logs redis
```

## 🎯 最佳实践

### 1. 任务描述

```python
# ❌ 不好的描述
"写代码"

# ✅ 好的描述
"创建一个 Python FastAPI 应用，包含用户认证、CRUD 操作和 API 文档"
```

### 2. 成本优化

- 使用任务类型提示: `-t code` 会优先使用 DeepSeek
- 批量处理相似任务
- 启用缓存减少重复请求

### 3. 安全建议

- 不要在代码中硬编码 API Keys
- 使用沙箱环境执行未知代码
- 定期更新依赖包

## 📖 进阶使用

### 自定义 Agent

```python
# custom_agent.py
from agent_system_implementation import ReactToTAgent, AgentRole

class DataAnalystAgent(ReactToTAgent):
    def __init__(self, agent_id: str, llm):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.RESEARCHER,
            llm=llm,
            tools=[DataProcessingTool(), DatabaseTool()]
        )
```

### 自定义工具

```python
# custom_tool.py
from agent_core_architecture import Tool

class CustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="My custom tool"
        )
        
    async def execute(self, **kwargs):
        # 实现你的工具逻辑
        return {"success": True, "result": "Done"}
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- Claude (Anthropic) - 强大的 AI 模型
- DeepSeek - 优秀的代码生成模型
- Qwen (阿里巴巴) - 用户交互模型
- 所有开源项目贡献者

---

需要帮助？查看 [完整文档](docs/) 或提交 [Issue](issues/)















🚀 更新版快速运行指南
最简单的3步运行：
步骤 1：安装依赖
bash# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Mac/Linux
# 或
venv\Scripts\activate     # Windows

# 安装核心依赖
pip install aiohttp pydantic python-dotenv anthropic openai langchain gradio fastapi uvicorn streamlit
步骤 2：创建最小配置文件
创建 .env 文件（只需要这些）：
env# 至少需要一个API密钥
QWEN_API_KEY=你的通义千问API密钥

# 可选（如果有的话）
CLAUDE_API_KEY=你的Claude密钥
步骤 3：运行
bash# 方法1：直接运行主程序（最简单）
python fixed-main-program.py

# 方法2：使用初始化向导（推荐首次）
python fixed-main-program.py --init

# 方法3：快速测试
python -c "
import asyncio
from fixed_agent_core import AgentSystemConfig
from fixed_agent_implementation import LocalAgentSystem

async def quick_test():
    config = AgentSystemConfig.from_env()
    system = LocalAgentSystem(config)
    await system.initialize()
    result = await system.process_request('说你好')
    print(result.get('response', '无响应'))
    await system.shutdown()

asyncio.run(quick_test())
"
⚡ 一键启动脚本
创建 quick_start.py：
python#!/usr/bin/env python
import os
import sys
import asyncio

# 设置最小环境变量（如果.env不存在）
if not os.path.exists('.env'):
    print("创建默认配置...")
    with open('.env', 'w') as f:
        f.write("""
QWEN_API_KEY=替换为你的API密钥
LOG_LEVEL=INFO
WORKSPACE_DIR=./workspace
""")
    print("请编辑 .env 文件，添加你的API密钥")
    sys.exit(1)

# 导入必要模块
try:
    from fixed_main_program import main
    asyncio.run(main())
except ImportError as e:
    print(f"错误：{e}")
    print("请确保所有 fixed_*.py 文件都在当前目录")
运行：
bashpython quick_start.py
🎮 交互式运行（最友好）
bash# 启动Web界面 - 打开浏览器访问 http://localhost:8501
python -m fixed_web_interface_api
🔥 超简化Docker运行
创建 docker-compose.simple.yml：
yamlversion: '3'
services:
  agent:
    image: python:3.11-slim
    volumes:
      - .:/app
    working_dir: /app
    environment:
      - QWEN_API_KEY=${QWEN_API_KEY}
    ports:
      - "8501:8501"
    command: |
      bash -c "
      pip install -r requirements.txt &&
      python fixed-main-program.py --mode ui
      "
运行：
bash# 设置API密钥
export QWEN_API_KEY="你的密钥"

# 启动
docker-compose -f docker-compose.simple.yml up
💡 快速测试命令
bash# 测试代码生成
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"description": "写一个Python函数计算斐波那契数列"}'

# 或使用Python
python -c "
import requests
r = requests.post('http://localhost:8000/tasks', 
    json={'description': '写个Hello World'})
print(r.json())
"
⚠️ 如果遇到问题

最小化测试：

python# test_minimal.py
from fixed_agent_core import QwenLLM, Message

async def test():
    llm = QwenLLM("你的API密钥")
    msg = Message(role="user", content="你好")
    response = await llm.generate([msg])
    print(response)

import asyncio
asyncio.run(test())

检查依赖：

bashpip list | grep -E "aiohttp|anthropic|openai"

查看日志：

bashtail -f logs/*.log
选择最适合你的方式运行即可！🚀RetryClaude can make mistakes. Please double-check responses.Researchbeta Opus 4