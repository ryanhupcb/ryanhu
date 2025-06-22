# Extended Tools Module
# 扩展工具模块 - 与主系统分离但可通信

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import aiohttp
from datetime import datetime
import numpy as np
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tweepy
import praw
from googletrans import Translator
import qrcode
from PIL import Image
import io
import base64
import os
import sys
from pathlib import Path

# 导入主系统的基类
# 假设主系统文件在同一目录或已添加到 Python 路径
sys.path.append(str(Path(__file__).parent))
from complete_agent_system import Tool, AgentMessage

# ==================== 新工具定义 ====================

class StockAnalysisTool(Tool):
    """股票分析工具"""
    def __init__(self):
        super().__init__(
            name="stock_analysis",
            description="Analyze stock data and provide insights"
        )
        
    async def execute(self, symbol: str, period: str = "1mo", analysis_type: str = "basic") -> Dict[str, Any]:
        """执行股票分析"""
        try:
            # 获取股票数据
            stock = yf.Ticker(symbol)
            
            if analysis_type == "basic":
                info = stock.info
                history = stock.history(period=period)
                
                result = {
                    'symbol': symbol,
                    'name': info.get('longName', 'N/A'),
                    'current_price': info.get('currentPrice', 0),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'price_history': {
                        'high': float(history['High'].max()),
                        'low': float(history['Low'].min()),
                        'average': float(history['Close'].mean()),
                        'volatility': float(history['Close'].std())
                    }
                }
                
            elif analysis_type == "technical":
                history = stock.history(period=period)
                
                # 计算技术指标
                sma_20 = history['Close'].rolling(window=20).mean().iloc[-1]
                sma_50 = history['Close'].rolling(window=50).mean().iloc[-1]
                rsi = self._calculate_rsi(history['Close'])
                
                result = {
                    'symbol': symbol,
                    'technical_indicators': {
                        'sma_20': float(sma_20) if not np.isnan(sma_20) else None,
                        'sma_50': float(sma_50) if not np.isnan(sma_50) else None,
                        'rsi': float(rsi),
                        'volume_trend': 'increasing' if history['Volume'].iloc[-5:].mean() > history['Volume'].mean() else 'decreasing'
                    }
                }
                
            self.usage_count += 1
            self.success_count += 1
            
            return {
                'success': True,
                'data': result,
                'analysis_type': analysis_type
            }
            
        except Exception as e:
            self.usage_count += 1
            return {
                'success': False,
                'error': str(e)
            }
            
    def _calculate_rsi(self, prices, period=14):
        """计算 RSI 指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
        
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g., AAPL, GOOGL)"
                },
                "period": {
                    "type": "string",
                    "description": "Time period for analysis",
                    "default": "1mo",
                    "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
                },
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to perform",
                    "default": "basic",
                    "enum": ["basic", "technical"]
                }
            },
            "required": ["symbol"]
        }

class EmailTool(Tool):
    """邮件发送工具"""
    def __init__(self, smtp_server: str = None, smtp_port: int = 587,
                 email: str = None, password: str = None):
        super().__init__(
            name="send_email",
            description="Send emails with attachments support"
        )
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port
        self.email = email or os.getenv('EMAIL_ADDRESS')
        self.password = password or os.getenv('EMAIL_PASSWORD')
        
    async def execute(self, to: str, subject: str, body: str, 
                     html: bool = False, attachments: List[Dict] = None) -> Dict[str, Any]:
        """发送邮件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = to
            msg['Subject'] = subject
            
            # 添加正文
            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
                
            # 添加附件
            if attachments:
                for attachment in attachments:
                    # attachment: {'filename': str, 'content': bytes}
                    part = MIMEText(attachment['content'])
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename={attachment["filename"]}'
                    )
                    msg.attach(part)
                    
            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)
                
            self.usage_count += 1
            self.success_count += 1
            
            return {
                'success': True,
                'message': f'Email sent to {to}',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.usage_count += 1
            return {
                'success': False,
                'error': str(e)
            }
            
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content"
                },
                "html": {
                    "type": "boolean",
                    "description": "Whether body is HTML",
                    "default": False
                },
                "attachments": {
                    "type": "array",
                    "description": "List of attachments",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["to", "subject", "body"]
        }

class TranslationTool(Tool):
    """多语言翻译工具"""
    def __init__(self):
        super().__init__(
            name="translate",
            description="Translate text between multiple languages"
        )
        self.translator = Translator()
        
    async def execute(self, text: str, target_lang: str, source_lang: str = 'auto') -> Dict[str, Any]:
        """执行翻译"""
        try:
            # 在异步环境中运行同步代码
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.translator.translate,
                text,
                target_lang,
                source_lang
            )
            
            self.usage_count += 1
            self.success_count += 1
            
            return {
                'success': True,
                'original_text': text,
                'translated_text': result.text,
                'source_language': result.src,
                'target_language': target_lang,
                'confidence': result.extra_data.get('confidence', 1.0) if hasattr(result, 'extra_data') else 1.0
            }
            
        except Exception as e:
            self.usage_count += 1
            return {
                'success': False,
                'error': str(e)
            }
            
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to translate"
                },
                "target_lang": {
                    "type": "string",
                    "description": "Target language code (e.g., 'en', 'zh-cn', 'es')"
                },
                "source_lang": {
                    "type": "string",
                    "description": "Source language code (auto-detect if not specified)",
                    "default": "auto"
                }
            },
            "required": ["text", "target_lang"]
        }

class QRCodeTool(Tool):
    """二维码生成工具"""
    def __init__(self):
        super().__init__(
            name="generate_qrcode",
            description="Generate QR codes for various data types"
        )
        
    async def execute(self, data: str, size: int = 10, border: int = 4,
                     error_correction: str = "M", return_base64: bool = True) -> Dict[str, Any]:
        """生成二维码"""
        try:
            # 错误纠正级别映射
            error_levels = {
                "L": qrcode.constants.ERROR_CORRECT_L,
                "M": qrcode.constants.ERROR_CORRECT_M,
                "Q": qrcode.constants.ERROR_CORRECT_Q,
                "H": qrcode.constants.ERROR_CORRECT_H
            }
            
            # 创建 QR 码
            qr = qrcode.QRCode(
                version=1,
                error_correction=error_levels.get(error_correction, qrcode.constants.ERROR_CORRECT_M),
                box_size=size,
                border=border,
            )
            
            qr.add_data(data)
            qr.make(fit=True)
            
            # 创建图像
            img = qr.make_image(fill_color="black", back_color="white")
            
            # 转换为 base64
            if return_base64:
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                result = {
                    'image_base64': img_base64,
                    'image_format': 'PNG'
                }
            else:
                # 保存到临时文件
                temp_path = f"/tmp/qrcode_{datetime.now().timestamp()}.png"
                img.save(temp_path)
                result = {
                    'file_path': temp_path,
                    'image_format': 'PNG'
                }
                
            self.usage_count += 1
            self.success_count += 1
            
            return {
                'success': True,
                'data': data,
                'result': result,
                'size': f"{img.size[0]}x{img.size[1]}"
            }
            
        except Exception as e:
            self.usage_count += 1
            return {
                'success': False,
                'error': str(e)
            }
            
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data to encode in QR code"
                },
                "size": {
                    "type": "integer",
                    "description": "Size of each box in pixels",
                    "default": 10
                },
                "border": {
                    "type": "integer",
                    "description": "Border size in boxes",
                    "default": 4
                },
                "error_correction": {
                    "type": "string",
                    "description": "Error correction level",
                    "default": "M",
                    "enum": ["L", "M", "Q", "H"]
                },
                "return_base64": {
                    "type": "boolean",
                    "description": "Return as base64 string or file path",
                    "default": True
                }
            },
            "required": ["data"]
        }

# ==================== 工具管理器 ====================

class ExtendedToolManager:
    """扩展工具管理器 - 负责与主系统通信"""
    
    def __init__(self, system_url: str = "http://localhost:8000"):
        self.system_url = system_url
        self.tools = {}
        self.session = None
        self._initialize_tools()
        
    def _initialize_tools(self):
        """初始化所有扩展工具"""
        extended_tools = [
            StockAnalysisTool(),
            EmailTool(),
            TranslationTool(),
            QRCodeTool()
        ]
        
        for tool in extended_tools:
            self.tools[tool.name] = tool
            
    async def register_with_system(self):
        """向主系统注册所有工具"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        for tool_name, tool in self.tools.items():
            try:
                # 通过 HTTP API 注册工具
                async with self.session.post(
                    f"{self.system_url}/register_tool",
                    json={
                        'name': tool.name,
                        'description': tool.description,
                        'schema': tool.get_schema(),
                        'endpoint': f"{self.system_url}/execute_tool/{tool.name}"
                    }
                ) as response:
                    result = await response.json()
                    print(f"Registered tool {tool_name}: {result}")
                    
            except Exception as e:
                print(f"Failed to register tool {tool_name}: {e}")
                
    async def start_tool_server(self, port: int = 8001):
        """启动工具服务器"""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_post('/execute/{tool_name}', self.handle_tool_execution)
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/list_tools', self.list_tools)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', port)
        await site.start()
        
        print(f"Extended tools server started on port {port}")
        
    async def handle_tool_execution(self, request):
        """处理工具执行请求"""
        from aiohttp import web
        
        tool_name = request.match_info['tool_name']
        
        if tool_name not in self.tools:
            return web.json_response({
                'success': False,
                'error': f'Tool {tool_name} not found'
            }, status=404)
            
        try:
            params = await request.json()
            tool = self.tools[tool_name]
            result = await tool.execute(**params)
            
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    async def health_check(self, request):
        """健康检查端点"""
        from aiohttp import web
        
        return web.json_response({
            'status': 'healthy',
            'tools_count': len(self.tools),
            'tools': list(self.tools.keys())
        })
        
    async def list_tools(self, request):
        """列出所有可用工具"""
        from aiohttp import web
        
        tools_info = {}
        for name, tool in self.tools.items():
            tools_info[name] = {
                'description': tool.description,
                'usage_count': tool.usage_count,
                'success_count': tool.success_count,
                'schema': tool.get_schema()
            }
            
        return web.json_response(tools_info)

# ==================== 与主系统的通信接口 ====================

class SystemConnector:
    """与主系统通信的连接器"""
    
    def __init__(self, communication_method: str = "http"):
        self.communication_method = communication_method
        self.connection = None
        
    async def connect_to_system(self, connection_params: Dict[str, Any]):
        """连接到主系统"""
        if self.communication_method == "http":
            self.connection = aiohttp.ClientSession()
            self.base_url = connection_params.get('base_url', 'http://localhost:8000')
            
        elif self.communication_method == "redis":
            import redis
            self.connection = redis.Redis(
                host=connection_params.get('host', 'localhost'),
                port=connection_params.get('port', 6379),
                decode_responses=True
            )
            
        elif self.communication_method == "rabbitmq":
            import aio_pika
            self.connection = await aio_pika.connect_robust(
                connection_params.get('url', 'amqp://guest:guest@localhost/')
            )
            
    async def send_message(self, message: Dict[str, Any]):
        """发送消息到主系统"""
        if self.communication_method == "http":
            async with self.connection.post(
                f"{self.base_url}/message",
                json=message
            ) as response:
                return await response.json()
                
        elif self.communication_method == "redis":
            # 使用 Redis 发布/订阅
            self.connection.publish('agent_messages', json.dumps(message))
            
        elif self.communication_method == "rabbitmq":
            channel = await self.connection.channel()
            await channel.default_exchange.publish(
                aio_pika.Message(body=json.dumps(message).encode()),
                routing_key='agent_messages'
            )
            
    async def register_tool_remote(self, tool: Tool):
        """远程注册工具到主系统"""
        message = {
            'type': 'tool_registration',
            'tool_name': tool.name,
            'tool_description': tool.description,
            'tool_schema': tool.get_schema(),
            'callback_url': f'http://localhost:8001/execute/{tool.name}'
        }
        
        return await self.send_message(message)

# ==================== 独立运行模式 ====================

async def run_standalone():
    """独立运行扩展工具服务"""
    # 创建工具管理器
    manager = ExtendedToolManager()
    
    # 创建系统连接器
    connector = SystemConnector(communication_method="http")
    await connector.connect_to_system({'base_url': 'http://localhost:8000'})
    
    # 注册所有工具到远程系统
    for tool_name, tool in manager.tools.items():
        result = await connector.register_tool_remote(tool)
        print(f"Remote registration of {tool_name}: {result}")
        
    # 启动工具服务器
    await manager.start_tool_server(port=8001)
    
    # 保持服务运行
    print("Extended tools service is running...")
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Shutting down extended tools service...")

# ==================== 集成模式 ====================

def integrate_with_system(system):
    """直接集成到主系统"""
    # 创建所有扩展工具
    extended_tools = [
        StockAnalysisTool(),
        EmailTool(),
        TranslationTool(),
        QRCodeTool()
    ]
    
    # 注册到系统的工具注册表
    for tool in extended_tools:
        system.enhanced_tool_registry.register_tool_instance(tool)
        
    print(f"Integrated {len(extended_tools)} extended tools into the system")
    
    return extended_tools

# ==================== 使用示例 ====================

async def demo_extended_tools():
    """演示扩展工具的使用"""
    manager = ExtendedToolManager()
    
    print("=== Extended Tools Demo ===\n")
    
    # 1. 股票分析
    print("1. Stock Analysis:")
    stock_tool = manager.tools['stock_analysis']
    result = await stock_tool.execute(symbol="AAPL", period="1mo", analysis_type="basic")
    if result['success']:
        print(f"Apple stock analysis: {result['data']}\n")
        
    # 2. 翻译
    print("2. Translation:")
    translate_tool = manager.tools['translate']
    result = await translate_tool.execute(
        text="Hello, how are you?",
        target_lang="zh-cn"
    )
    if result['success']:
        print(f"Translation: {result['translated_text']}\n")
        
    # 3. 二维码生成
    print("3. QR Code Generation:")
    qr_tool = manager.tools['generate_qrcode']
    result = await qr_tool.execute(
        data="https://www.example.com",
        size=10,
        error_correction="H"
    )
    if result['success']:
        print(f"QR code generated, size: {result['size']}\n")
        
    # 4. 邮件发送（需要配置）
    print("4. Email (configuration required):")
    email_tool = manager.tools['send_email']
    # result = await email_tool.execute(
    #     to="recipient@example.com",
    #     subject="Test Email",
    #     body="This is a test email from the extended tools module."
    # )
    print("Email tool available but requires configuration\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extended Tools Module')
    parser.add_argument('--mode', choices=['standalone', 'demo'], default='demo',
                       help='Run mode: standalone server or demo')
    parser.add_argument('--port', type=int, default=8001,
                       help='Port for standalone server')
    
    args = parser.parse_args()
    
    if args.mode == 'standalone':
        asyncio.run(run_standalone())
    else:
        asyncio.run(demo_extended_tools())
