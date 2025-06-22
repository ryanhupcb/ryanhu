# Agent协作系统使用指南

## 系统概述

这是一个完整的生产级多Agent协作系统，支持：
- 🤖 多个专业Agent（研究、代码、分析等）
- 🔧 丰富的工具集（搜索、执行代码、数据分析、股票分析等）
- 🔗 多种通信方式（HTTP API、Redis、RabbitMQ）
- 📝 任务规划和协作执行
- 💬 自然语言对话接口

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv agent_env
source agent_env/bin/activate  # Linux/Mac
# 或
agent_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置LLM密钥

```bash
# 设置环境变量
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# 可选：邮件配置
export SMTP_SERVER="smtp.gmail.com"
export EMAIL_ADDRESS="your-email@gmail.com"
export EMAIL_PASSWORD="your-app-password"
```

### 3. 启动系统

#### 交互模式（推荐初次使用）
```bash
python agent_collaboration_runner.py
```

#### API服务器模式
```bash
python agent_collaboration_runner.py --api-only
```

#### 运行预定义场景
```bash
# 运行研究场景
python agent_collaboration_runner.py --scenario research

# 运行数据管道场景
python agent_collaboration_runner.py --scenario pipeline

# 运行所有演示
python agent_collaboration_runner.py --scenario demo
```

#### 使用自定义配置
```bash
# 先创建配置模板
python agent_collaboration_runner.py --create-config

# 编辑配置文件
# 编辑 agent_config.json

# 使用配置启动
python agent_collaboration_runner.py --config agent_config.json
```

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   用户接口层                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │   CLI   │  │HTTP API │  │  Redis  │  │RabbitMQ │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                  协作协调层                              │
│  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │ CollaborationCoordinator │  │ DialogManager     │  │
│  └──────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Agent层                                │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐   │
│  │Research    │  │Code Agent  │  │Analysis Agent  │   │
│  │Agent       │  │            │  │                │   │
│  └────────────┘  └────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    工具层                                │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    │
│  │Web  │ │Code │ │File │ │Data │ │Stock│ │Email│    │
│  │Search│ │Exec │ │Ops  │ │Analy│ │Analy│ │Tool │    │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘    │
└─────────────────────────────────────────────────────────┘
```

## 主要功能

### 1. 对话交互
```python
# 在交互模式下
> chat 帮我分析一下最近的AI发展趋势
```

### 2. 任务执行
```python
# 执行复杂任务
> task 创建一个数据分析脚本，处理CSV文件并生成可视化报告
```

### 3. 协作场景

#### 预定义场景
- **研究和报告场景**：多个Agent协作研究主题并生成综合报告
- **数据管道场景**：构建完整的数据处理管道

#### 自定义场景
```python
> create
# 按提示创建自定义的多Agent协作场景
```

### 4. HTTP API使用

#### 聊天接口
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，介绍一下你的功能"}'
```

#### 执行任务
```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"task": "分析这段代码的性能问题", "context": {"code": "..."}}'
```

#### 系统状态
```bash
curl http://localhost:8000/status
```

## 配置说明

### 基本配置
```json
{
  "system": {
    "enable_all_frameworks": true,
    "safety_threshold": 0.95,
    "openai_model": "gpt-4-turbo-preview"
  },
  "api_server": {
    "enabled": true,
    "port": 8000
  }
}
```

### 通信配置
```json
{
  "communication": {
    "redis": {
      "enabled": true,
      "host": "localhost",
      "port": 6379
    },
    "rabbitmq": {
      "enabled": false,
      "url": "amqp://guest:guest@localhost/"
    }
  }
}
```

## 扩展开发

### 添加新的Agent
```python
class CustomAgent(SpecializedAgent):
    async def receive_message(self, message: AgentMessage):
        # 实现消息处理逻辑
        pass
```

### 添加新的工具
```python
class CustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="自定义工具描述"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        # 实现工具逻辑
        pass
```

### 创建新的协作场景
```python
def create_custom_scenario() -> CollaborationScenario:
    scenario = CollaborationScenario(
        name="custom",
        description="自定义协作场景"
    )
    
    # 添加任务
    task1 = scenario.add_task(
        agent='research_agent',
        action='research',
        params={'topic': '...'}
    )
    
    return scenario
```

## 故障排除

### 常见问题

1. **LLM连接失败**
   - 检查API密钥是否正确设置
   - 确认网络连接正常

2. **工具执行失败**
   - 查看日志文件了解详细错误
   - 确认所需依赖已安装

3. **Agent通信问题**
   - 检查消息总线是否正常运行
   - 确认Agent已正确注册

### 日志查看
```bash
# 查看系统日志
tail -f agent_system.log

# 调试模式
python agent_collaboration_runner.py --debug
```

## 性能优化

1. **并发控制**
   - 调整任务并发数
   - 优化Agent响应时间

2. **缓存策略**
   - 启用向量数据库缓存
   - 配置Redis缓存

3. **资源限制**
   - 设置代码执行超时
   - 限制内存使用

## 安全注意事项

1. **API密钥管理**
   - 使用环境变量或密钥管理服务
   - 不要在代码中硬编码密钥

2. **代码执行安全**
   - 代码在沙箱环境中执行
   - 限制可导入的模块

3. **访问控制**
   - API服务器建议配置认证
   - 限制网络访问范围

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请通过以下方式联系：
- Issue: [GitHub Issues]
- Email: [huzhanrui0521@gmail.com]
