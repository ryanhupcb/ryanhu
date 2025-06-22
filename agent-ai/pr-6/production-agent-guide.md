# 增强版生产级Agent系统使用指南

## 系统架构概述

### 核心特性
1. **多层次LLM架构**
   - TinyLLM处理简单任务，降低成本
   - 主LLM（Claude/GPT-4）处理复杂任务
   - 智能任务路由

2. **高级推理能力**
   - ReAct + Tree of Thoughts混合推理
   - 智能任务分解与调度
   - 依赖图自动构建

3. **系统控制能力**
   - 基于Computer Use的系统控制
   - 浏览器自动化
   - 文件系统操作
   - Git集成

4. **专业化Agent**
   - 代码开发Agent
   - 系统控制Agent
   - Web研究Agent
   - 分析Agent

## 快速开始

### 1. 环境准备

```bash
# 克隆代码
git clone <your-repo>
cd enhanced-agent-system

# 运行设置脚本
chmod +x setup.sh
./setup.sh
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# GitHub
GITHUB_ACCESS_TOKEN=your_github_token

# TinyLLM
TINY_LLM_PATH=./models/deepseek-coder-1.3b
```

### 3. 创建配置文件

```bash
python enhanced_agent_launcher.py --create-config
```

编辑 `agent_config.yaml`：

```yaml
# LLM配置
use_tiny_llm: true
tiny_llm_threshold: 0.8
tiny_llm_model: "deepseek-coder-1.3b"
openai_model: "gpt-4-turbo-preview"
anthropic_model: "claude-3-opus-20240229"

# 系统功能
enable_computer_control: true
enable_browser_automation: true
enable_github_integration: true

# 性能设置
max_concurrent_tasks: 10
task_timeout: 300

# API配置
api_enabled: true
api_port: 8000
api_host: "0.0.0.0"
```

## 使用方式

### 1. 交互模式

```bash
python enhanced_agent_launcher.py --mode interactive
```

交互模式命令：
- `execute <request>` - 执行请求
- `status` - 查看系统状态
- `metrics` - 查看性能指标
- `tools` - 列出可用工具
- `help` - 显示帮助
- `quit` - 退出

### 2. API服务器模式

```bash
python enhanced_agent_launcher.py --mode server
```

API端点：
- `POST /execute` - 执行任务
- `POST /chat` - 聊天接口
- `GET /status` - 系统状态
- `GET /metrics` - 性能指标
- `POST /tools/{tool_name}/{method_name}` - 调用特定工具

### 3. 单次执行模式

```bash
python enhanced_agent_launcher.py --mode execute \
  --request "创建一个Python REST API项目" \
  --context '{"framework": "FastAPI"}'
```

## 常见使用场景

### 1. 代码开发

```python
# 交互模式示例
> execute 创建一个用户认证系统，包含注册、登录和JWT令牌管理

# API调用示例
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "request": "实现一个高性能的缓存系统",
    "context": {
      "language": "python",
      "features": ["LRU", "TTL", "分布式"]
    }
  }'
```

### 2. 系统自动化

```python
# 自动化部署脚本
> execute 创建一个自动化部署脚本，包含代码拉取、测试运行和Docker构建

# 系统监控
> execute 监控系统资源使用情况并生成报告
```

### 3. 研究与分析

```python
# 技术研究
> execute 研究最新的Python Web框架并创建对比分析报告

# 代码分析
> execute 分析项目代码质量并提供改进建议
```

### 4. 工具使用

```python
# 使用特定工具
curl -X POST http://localhost:8000/tools/code_formatter/format_python \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello(name):print(f\"Hello {name}\")",
    "line_length": 88
  }'
```

## 高级功能

### 1. 自定义Agent

创建新的Agent类：

```python
from enhanced_production_agent import BaseAgent

class DataAnalysisAgent(BaseAgent):
    def __init__(self, agent_id: str, llm, tools):
        super().__init__(agent_id)
        self.llm = llm
        self.tools = tools
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # 实现数据分析逻辑
        pass
```

### 2. 扩展工具集

添加新工具到 `enhanced_tools.py`：

```python
class CustomTool:
    @staticmethod
    async def custom_method(param1: str, param2: int) -> Dict[str, Any]:
        # 实现自定义功能
        return {"success": True, "result": "..."}
```

### 3. 任务编排

创建复杂的任务流：

```python
task_flow = {
    "name": "完整项目开发",
    "steps": [
        {
            "id": "design",
            "type": "analysis",
            "description": "设计系统架构"
        },
        {
            "id": "implement",
            "type": "code",
            "description": "实现核心功能",
            "dependencies": ["design"]
        },
        {
            "id": "test",
            "type": "code",
            "description": "编写测试用例",
            "dependencies": ["implement"]
        }
    ]
}
```

## 部署指南

### Docker部署

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f agent-system
```

### Kubernetes部署

创建 `k8s-deployment.yaml`：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-system
  template:
    metadata:
      labels:
        app: agent-system
    spec:
      containers:
      - name: agent-system
        image: enhanced-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-key
```

## 监控与运维

### 1. 健康检查

```bash
# 检查系统状态
curl http://localhost:8000/status

# Prometheus指标
curl http://localhost:8000/metrics
```

### 2. 日志管理

```yaml
# 配置日志级别
log_level: "INFO"
log_file: "agent_system.log"

# 日志轮转配置
logging:
  handlers:
    file:
      class: logging.handlers.RotatingFileHandler
      maxBytes: 10485760  # 10MB
      backupCount: 5
```

### 3. 性能优化

- 启用TinyLLM处理简单任务
- 调整并发任务数量
- 使用Redis缓存结果
- 启用任务优先级队列

## 安全最佳实践

### 1. API认证

启用API认证：

```yaml
api_auth_enabled: true
api_auth_tokens:
  - "your-secure-token-1"
  - "your-secure-token-2"
```

使用认证：

```bash
curl -X POST http://localhost:8000/execute \
  -H "Authorization: Bearer your-secure-token-1" \
  -H "Content-Type: application/json" \
  -d '{"request": "..."}'
```

### 2. 沙箱执行

```yaml
sandbox_code_execution: true
allowed_system_commands:
  - "ls"
  - "pwd"
  - "echo"
restricted_paths:
  - "/etc"
  - "/root"
```

### 3. 资源限制

```yaml
memory_limit_mb: 4096
task_timeout: 300
max_concurrent_tasks: 10
```

## 故障排除

### 常见问题

1. **TinyLLM加载失败**
   ```bash
   # 检查模型路径
   ls -la ./models/
   # 重新下载模型
   python download_tinyllm.py
   ```

2. **API连接错误**
   ```bash
   # 检查端口占用
   lsof -i :8000
   # 更改端口
   vim agent_config.yaml
   ```

3. **内存不足**
   ```yaml
   # 减少并发数
   max_concurrent_tasks: 5
   # 禁用不需要的功能
   enable_browser_automation: false
   ```

## 性能基准

典型任务性能：

| 任务类型 | TinyLLM | 主LLM | 执行时间 |
|---------|---------|-------|----------|
| 简单代码生成 | ✓ | - | <2s |
| 复杂系统设计 | - | ✓ | 10-30s |
| 文件操作 | ✓ | - | <1s |
| Web研究 | - | ✓ | 5-15s |

## 贡献指南

1. Fork仓库
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License

## 支持

- GitHub Issues: [创建Issue]
- 文档: [查看完整文档]
- 社区: [加入Discord]