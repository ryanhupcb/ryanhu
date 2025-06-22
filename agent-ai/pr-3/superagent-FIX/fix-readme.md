# 🔧 Agent协作系统修复指南

## 快速修复（推荐）

### 方法1：使用自动修复脚本（最简单）

```bash
# 1. 运行自动设置脚本
python auto_setup.py

# 2. 编辑.env文件，添加API密钥（可选）
# 3. 启动系统
python run_agent_system.py
```

### 方法2：使用修复后的文件

1. **使用修复后的主系统文件**
   ```bash
   # 备份原文件（如果存在）
   mv complete_agent_system.py complete_agent_system_original.py
   
   # 使用修复后的版本
   mv complete_agent_system_fixed.py complete_agent_system.py
   ```

2. **安装最小依赖**
   ```bash
   pip install numpy aiohttp
   ```

3. **运行系统**
   ```bash
   python run_agent_system.py
   ```

## 📁 文件说明

### 新增的修复文件

1. **`complete_agent_system_fixed.py`**
   - 完整的、自包含的Agent系统
   - 包含所有必要的基类定义
   - 可以独立运行，不依赖其他模块

2. **`run_agent_system.py`**
   - 简单的启动脚本
   - 自动检测可用的系统文件
   - 提供友好的交互界面

3. **`auto_setup.py`**
   - 自动安装和配置脚本
   - 检查Python版本和依赖
   - 创建必要的配置文件

## 🚀 启动选项

### 选项1：最小模式（无需API密钥）
```bash
python run_agent_system.py
# 系统将在模拟模式下运行
```

### 选项2：完整模式（需要API密钥）
```bash
# 设置环境变量
export OPENAI_API_KEY="your-openai-key"
# 或
export ANTHROPIC_API_KEY="your-anthropic-key"

# 运行系统
python run_agent_system.py
```

### 选项3：使用.env文件
```bash
# 创建.env文件
echo "OPENAI_API_KEY=your-key" > .env

# 运行系统（需要python-dotenv）
pip install python-dotenv
python run_agent_system.py
```

## 📋 系统功能

修复后的系统支持以下功能：

### 基础功能（无需API密钥）
- ✅ 系统初始化和健康检查
- ✅ 工具注册和管理
- ✅ Agent通信总线
- ✅ 基本的任务规划
- ✅ 模拟的对话响应

### 高级功能（需要API密钥）
- ✅ 真实的LLM对话
- ✅ 智能任务执行
- ✅ 研究和分析功能
- ✅ 多Agent协作

## 🎯 使用示例

启动系统后，可以使用以下命令：

```
> chat 你好，介绍一下你的功能
> task 创建一个Python脚本
> research AI技术趋势
> status
> help
```

## ❓ 常见问题

### Q: 提示"无法导入Agent系统"
A: 确保 `complete_agent_system_fixed.py` 或 `complete_agent_system.py` 在当前目录

### Q: 提示"未检测到LLM API密钥"
A: 这是正常的，系统会在模拟模式下运行。如需完整功能，请设置API密钥

### Q: 安装依赖失败
A: 尝试升级pip：`pip install --upgrade pip`

### Q: Python版本错误
A: 需要Python 3.7或更高版本

## 📞 获取帮助

如果仍有问题：

1. 检查所有文件是否完整
2. 确保Python版本 >= 3.7
3. 尝试在虚拟环境中运行：
   ```bash
   python -m venv agent_env
   source agent_env/bin/activate  # Linux/Mac
   # 或
   agent_env\Scripts\activate  # Windows
   ```

## ✅ 验证安装

运行以下命令验证系统是否正常：

```bash
python -c "from complete_agent_system_fixed import CompleteAgentSystem; print('✅ 系统导入成功')"
```

如果看到"✅ 系统导入成功"，说明修复完成！