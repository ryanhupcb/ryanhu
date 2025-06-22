#!/bin/bash
# Agent Collaboration System - Quick Start Script
# 快速启动脚本 - 自动配置和启动系统

set -e  # 出错时停止执行

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示欢迎信息
show_welcome() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║         Agent Collaboration System - Quick Start        ║"
    echo "║              多Agent协作系统 - 快速启动                  ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 检查Python版本
check_python() {
    print_info "检查Python版本..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "未找到Python3，请先安装Python 3.8或更高版本"
        exit 1
    fi
    
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_error "Python版本过低，需要3.8或更高版本，当前版本: $python_version"
        exit 1
    fi
    
    print_success "Python版本检查通过: $python_version"
}

# 检查并创建虚拟环境
setup_venv() {
    print_info "设置虚拟环境..."
    
    if [ ! -d "agent_env" ]; then
        print_info "创建虚拟环境..."
        python3 -m venv agent_env
        print_success "虚拟环境创建成功"
    else
        print_info "虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    source agent_env/bin/activate
    
    # 升级pip
    print_info "升级pip..."
    pip install --upgrade pip > /dev/null 2>&1
}

# 安装依赖
install_dependencies() {
    print_info "安装依赖包..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "依赖安装完成"
    else
        print_warning "未找到requirements.txt，安装基础依赖..."
        pip install aiohttp openai anthropic numpy pandas redis aio-pika
    fi
}

# 检查环境变量
check_env_vars() {
    print_info "检查环境变量..."
    
    missing_vars=()
    
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
        print_warning "未设置LLM API密钥"
        missing_vars+=("LLM_API_KEY")
    fi
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_warning "缺少以下环境变量："
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        
        read -p "是否要现在配置？(y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            setup_env_vars
        fi
    else
        print_success "环境变量检查通过"
    fi
}

# 设置环境变量
setup_env_vars() {
    print_info "配置环境变量..."
    
    # 创建.env文件
    touch .env
    
    read -p "请输入OpenAI API密钥 (回车跳过): " openai_key
    if [ ! -z "$openai_key" ]; then
        echo "OPENAI_API_KEY=$openai_key" >> .env
        export OPENAI_API_KEY=$openai_key
    fi
    
    read -p "请输入Anthropic API密钥 (回车跳过): " anthropic_key
    if [ ! -z "$anthropic_key" ]; then
        echo "ANTHROPIC_API_KEY=$anthropic_key" >> .env
        export ANTHROPIC_API_KEY=$anthropic_key
    fi
    
    print_success "环境变量配置完成"
}

# 检查服务
check_services() {
    print_info "检查外部服务..."
    
    # 检查Redis
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            print_success "Redis服务正常"
        else
            print_warning "Redis未运行，某些功能可能受限"
        fi
    else
        print_warning "Redis未安装，某些功能可能受限"
    fi
    
    # 检查Docker
    if command -v docker &> /dev/null; then
        print_info "Docker已安装"
        read -p "是否使用Docker Compose启动完整系统？(y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            start_with_docker
            exit 0
        fi
    fi
}

# 使用Docker启动
start_with_docker() {
    print_info "使用Docker Compose启动系统..."
    
    # 检查docker-compose文件
    if [ ! -f "docker-compose.yml" ]; then
        print_error "未找到docker-compose.yml文件"
        exit 1
    fi
    
    # 创建必要的目录
    mkdir -p data logs workspace config
    
    # 复制配置文件
    if [ ! -f "config/production.json" ]; then
        if [ -f "production.json" ]; then
            cp production.json config/
        else
            print_warning "未找到生产配置文件，使用默认配置"
        fi
    fi
    
    # 启动服务
    docker-compose up -d
    
    print_success "Docker服务启动成功"
    print_info "服务地址："
    echo "  - API服务器: http://localhost:8000"
    echo "  - RabbitMQ管理界面: http://localhost:15672"
    echo "  - Grafana监控: http://localhost:3000"
}

# 创建必要的目录
create_directories() {
    print_info "创建必要的目录..."
    
    directories=("data" "logs" "workspace" "config" "config/scenarios")
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_success "目录创建完成"
}

# 生成配置文件
generate_config() {
    print_info "生成配置文件..."
    
    if [ ! -f "agent_config.json" ]; then
        python3 agent_collaboration_runner.py --create-config
        print_success "配置文件生成成功"
    else
        print_info "配置文件已存在"
    fi
}

# 选择启动模式
select_mode() {
    echo
    print_info "请选择启动模式："
    echo "  1) 交互模式 - 命令行交互"
    echo "  2) API服务器 - HTTP API服务"
    echo "  3) 演示模式 - 运行所有演示"
    echo "  4) 研究场景 - 运行研究协作场景"
    echo "  5) 自定义配置 - 使用配置文件启动"
    echo
    read -p "请输入选项 (1-5): " choice
    
    case $choice in
        1)
            print_info "启动交互模式..."
            python3 agent_collaboration_runner.py
            ;;
        2)
            print_info "启动API服务器..."
            python3 agent_collaboration_runner.py --api-only
            ;;
        3)
            print_info "运行演示..."
            python3 agent_collaboration_runner.py --scenario demo
            ;;
        4)
            print_info "运行研究场景..."
            python3 agent_collaboration_runner.py --scenario research
            ;;
        5)
            print_info "使用自定义配置启动..."
            python3 agent_collaboration_runner.py --config agent_config.json
            ;;
        *)
            print_error "无效的选项"
            select_mode
            ;;
    esac
}

# 主函数
main() {
    show_welcome
    
    # 执行检查和设置
    check_python
    setup_venv
    install_dependencies
    check_env_vars
    check_services
    create_directories
    generate_config
    
    print_success "系统准备就绪！"
    
    # 选择启动模式
    select_mode
}

# 错误处理
trap 'print_error "脚本执行失败"; exit 1' ERR

# 运行主函数
main