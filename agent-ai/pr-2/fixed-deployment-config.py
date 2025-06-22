# Docker Compose Configuration
# docker-compose.yml

version: '3.8'

services:
  # 主Agent系统服务
  agent-system:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: local-agent-system
    environment:
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - QWEN_API_KEY=${QWEN_API_KEY}
      - WORKSPACE_DIR=/app/workspace
      - LOG_LEVEL=INFO
      - PYTHONUNBUFFERED=1
    volumes:
      - ./workspace:/app/workspace
      - ./logs:/app/logs
      - /tmp/.X11-unix:/tmp/.X11-unix:rw  # For GUI apps
    ports:
      - "8000:8000"  # API端口
      - "8501:8501"  # Streamlit UI端口
    networks:
      - agent-network
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped
    # 对于浏览器自动化，需要特殊权限
    privileged: true
    shm_size: '2gb'
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    
  # Redis用于任务队列和缓存
  redis:
    image: redis:7-alpine
    container_name: agent-redis
    volumes:
      - redis-data:/data
    networks:
      - agent-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    
  # PostgreSQL用于持久化存储
  postgres:
    image: postgres:15-alpine
    container_name: agent-postgres
    environment:
      - POSTGRES_USER=agent
      - POSTGRES_PASSWORD=agent_pass
      - POSTGRES_DB=agent_system
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - agent-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    
  # 本地DeepSeek-Coder服务（使用Ollama）
  deepseek-local:
    image: ollama/ollama:latest
    container_name: deepseek-local
    volumes:
      - ollama-data:/root/.ollama
    ports:
      - "11434:11434"
    networks:
      - agent-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
              options:
                fail-mode: "ignore"
              
networks:
  agent-network:
    driver: bridge
    
volumes:
  redis-data:
  postgres-data:
  ollama-data:

---
# Dockerfile

# 构建阶段
FROM python:3.11-slim as builder

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 运行时阶段
FROM python:3.11-slim

# 仅安装运行时必需的系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制已安装的Python包
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p /app/workspace /app/logs /app/data

# 设置环境变量
ENV DISPLAY=:99
ENV PYTHONPATH=/app

# 启动脚本
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "main.py"]

---
# requirements.txt

# 核心依赖
aiohttp>=3.9.0,<4.0.0
pydantic>=2.5.0,<3.0.0
python-dotenv>=1.0.0,<2.0.0

# LLM相关
anthropic>=0.25.0,<1.0.0
openai>=1.40.0,<2.0.0
langchain>=0.1.0,<0.2.0
langchain-community>=0.0.10,<0.1.0
langgraph>=0.0.26,<0.1.0

# 浏览器自动化
playwright>=1.40.0,<2.0.0
beautifulsoup4>=4.12.0,<5.0.0

# 系统自动化
pyautogui>=0.9.50,<1.0.0
keyboard>=0.13.0,<1.0.0
mouse>=0.7.0,<1.0.0
mss>=9.0.0,<10.0.0
Pillow>=10.1.0,<11.0.0

# 图像处理和OCR
opencv-python>=4.8.0,<5.0.0
pytesseract>=0.3.0,<1.0.0
numpy>=1.24.0,<2.0.0

# 数据库
redis>=5.0.0,<6.0.0
psycopg2-binary>=2.9.0,<3.0.0
sqlalchemy>=2.0.0,<3.0.0
alembic>=1.12.0,<2.0.0

# Web框架
fastapi>=0.104.0,<1.0.0
uvicorn>=0.24.0,<1.0.0
streamlit>=1.28.0,<2.0.0
gradio>=4.7.0,<5.0.0

# 工具
pyyaml>=6.0.0,<7.0.0
click>=8.1.0,<9.0.0
rich>=13.7.0,<14.0.0
loguru>=0.7.0,<1.0.0

# 监控
prometheus-client>=0.19.0,<1.0.0
psutil>=5.9.0,<6.0.0

# 测试
pytest>=7.4.0,<8.0.0
pytest-asyncio>=0.21.0,<1.0.0
pytest-cov>=4.1.0,<5.0.0

# Windows特定依赖（可选）
# pywin32>=306  # Windows only

---
# docker-entrypoint.sh

#!/bin/bash
set -euo pipefail

# 清理函数，确保退出时清理后台进程
cleanup() {
    jobs -p | xargs -r kill
    rm -f /tmp/.X99-lock
}
trap cleanup EXIT

# 启动虚拟显示器（用于无头环境的GUI操作）
if [ "$DISPLAY" = ":99" ]; then
    Xvfb :99 -screen 0 1920x1080x24 &
    sleep 2
    fluxbox &
    x11vnc -display :99 -nopw -listen localhost -xkb -forever &
fi

# 带超时和重试的服务等待函数
wait_for_service() {
    local host=$1 port=$2 timeout=${3:-30} attempt=0
    
    echo "Waiting for $host:$port..."
    while ! nc -z $host $port; do
        attempt=$((attempt+1))
        if [ $attempt -ge $timeout ]; then
            echo "Error: $host:$port not available after $timeout seconds" >&2
            exit 1
        fi
        sleep 1
    done
    echo "$host:$port is ready!"
}

# 等待依赖服务
wait_for_service postgres 5432 60
wait_for_service redis 6379 30

# 运行数据库迁移
alembic upgrade head

# 带重试的模型下载函数
download_model() {
    local model=$1 retries=3 delay=5
    
    for i in $(seq 1 $retries); do
        echo "Attempt $i to download model $model..."
        if curl -X POST http://deepseek-local:11434/api/pull -d '{"name": "'$model'"}' --fail --silent --show-error; then
            echo "Successfully downloaded model $model"
            return 0
        fi
        sleep $delay
    done
    echo "Failed to download model $model after $retries attempts" >&2
    return 1
}

# 下载DeepSeek-Coder模型到本地Ollama（如果配置了）
if [ "${USE_LOCAL_DEEPSEEK:-false}" = "true" ]; then
    download_model "deepseek-coder:6.7b" || true
fi

# 安装Playwright浏览器
playwright install chromium

# 启动健康检查服务器（后台运行）
python3 - << 'EOF' &
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()

def run_health_check():
    server = HTTPServer(('0.0.0.0', 8000), HealthCheckHandler)
    server.serve_forever()

threading.Thread(target=run_health_check, daemon=True).start()

# 启动应用
exec "$@"
EOF