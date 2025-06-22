# LocalAgentSystem - Browser & System Control (Fixed)
# 浏览器和系统控制模块

import asyncio
import os
import subprocess
import platform
import psutil
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from datetime import datetime
import logging
from dataclasses import dataclass
import aioredis
import hashlib
from collections import deque
import threading
import numpy as np

# 导入必需的核心模块
from agent_core_architecture import Tool, BaseAgent, AgentRole, LLMInterface, Task

logger = logging.getLogger(__name__)


# ==================== 浏览器控制 ====================

class BrowserController(Tool):
    """浏览器控制工具"""
    
    def __init__(self):
        super().__init__(
            name="browser_controller",
            description="Control web browser for automation tasks"
        )
        self.browser = None
        self.page = None
        
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """执行浏览器操作"""
        try:
            # 延迟导入以避免依赖问题
            from playwright.async_api import async_playwright
            
            if action == "launch":
                return await self._launch_browser(**kwargs)
            elif action == "navigate":
                return await self._navigate(**kwargs)
            elif action == "click":
                return await self._click(**kwargs)
            elif action == "type":
                return await self._type_text(**kwargs)
            elif action == "screenshot":
                return await self._take_screenshot(**kwargs)
            elif action == "extract":
                return await self._extract_data(**kwargs)
            elif action == "close":
                return await self._close_browser()
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except ImportError:
            return {"success": False, "error": "Playwright not installed. Please install with: pip install playwright"}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _launch_browser(self, headless: bool = True, **kwargs) -> Dict[str, Any]:
        """启动浏览器"""
        from playwright.async_api import async_playwright
        
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=headless)
        self.page = await self.browser.new_page()
        
        return {"success": True, "message": "Browser launched"}
        
    async def _navigate(self, url: str, **kwargs) -> Dict[str, Any]:
        """导航到URL"""
        if not self.page:
            return {"success": False, "error": "Browser not launched"}
            
        await self.page.goto(url)
        return {"success": True, "url": url, "title": await self.page.title()}
        
    async def _click(self, selector: str, **kwargs) -> Dict[str, Any]:
        """点击元素"""
        if not self.page:
            return {"success": False, "error": "Browser not launched"}
            
        await self.page.click(selector)
        return {"success": True, "clicked": selector}
        
    async def _type_text(self, selector: str, text: str, **kwargs) -> Dict[str, Any]:
        """输入文本"""
        if not self.page:
            return {"success": False, "error": "Browser not launched"}
            
        await self.page.fill(selector, text)
        return {"success": True, "typed": text}
        
    async def _take_screenshot(self, path: str = None, **kwargs) -> Dict[str, Any]:
        """截图"""
        if not self.page:
            return {"success": False, "error": "Browser not launched"}
            
        if not path:
            path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
        await self.page.screenshot(path=path)
        return {"success": True, "screenshot_path": path}
        
    async def _extract_data(self, selector: str, attribute: str = None, **kwargs) -> Dict[str, Any]:
        """提取数据"""
        if not self.page:
            return {"success": False, "error": "Browser not launched"}
            
        elements = await self.page.query_selector_all(selector)
        data = []
        
        for element in elements:
            if attribute:
                value = await element.get_attribute(attribute)
            else:
                value = await element.text_content()
            data.append(value)
            
        return {"success": True, "data": data}
        
    async def _close_browser(self) -> Dict[str, Any]:
        """关闭浏览器"""
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None
            
        return {"success": True, "message": "Browser closed"}
        
    async def cleanup(self):
        """清理资源"""
        await self._close_browser()


# ==================== 系统控制 ====================

class SystemController(Tool):
    """系统控制工具"""
    
    def __init__(self):
        super().__init__(
            name="system_controller",
            description="Control system operations and execute commands"
        )
        self.allowed_commands = {"ls", "pwd", "echo", "cat", "grep", "find", "ps", "df", "du"}
        
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """执行系统操作"""
        try:
            if action == "run_command":
                return await self._run_command(**kwargs)
            elif action == "get_system_info":
                return await self._get_system_info()
            elif action == "manage_process":
                return await self._manage_process(**kwargs)
            elif action == "file_operations":
                return await self._file_operations(**kwargs)
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _run_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """运行系统命令"""
        # 安全检查
        cmd_parts = command.split()
        if not cmd_parts:
            return {"success": False, "error": "Empty command"}
            
        base_cmd = cmd_parts[0]
        if base_cmd not in self.allowed_commands:
            return {"success": False, "error": f"Command '{base_cmd}' not allowed"}
            
        try:
            # 使用subprocess运行命令
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "return_code": process.returncode
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            info = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "used": psutil.disk_usage('/').used,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                }
            }
            
            return {"success": True, "system_info": info}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _manage_process(self, action: str, pid: int = None, name: str = None, **kwargs) -> Dict[str, Any]:
        """管理进程"""
        try:
            if action == "list":
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    processes.append(proc.info)
                return {"success": True, "processes": processes[:50]}  # 限制返回数量
                
            elif action == "kill" and pid:
                process = psutil.Process(pid)
                process.terminate()
                return {"success": True, "message": f"Process {pid} terminated"}
                
            else:
                return {"success": False, "error": "Invalid process action"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _file_operations(self, operation: str, path: str, **kwargs) -> Dict[str, Any]:
        """文件操作"""
        try:
            # 安全检查路径
            if ".." in path or path.startswith("/etc") or path.startswith("/sys"):
                return {"success": False, "error": "Path not allowed"}
                
            if operation == "size":
                size = os.path.getsize(path)
                return {"success": True, "size": size}
                
            elif operation == "exists":
                exists = os.path.exists(path)
                return {"success": True, "exists": exists}
                
            elif operation == "list_dir":
                if os.path.isdir(path):
                    files = os.listdir(path)
                    return {"success": True, "files": files}
                else:
                    return {"success": False, "error": "Not a directory"}
                    
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}


# ==================== UI控制Agent ====================

class UIControllerAgent(BaseAgent):
    """UI控制Agent"""
    
    def __init__(self, agent_id: str, llm: LLMInterface):
        browser_tool = BrowserController()
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.UI_CONTROLLER,
            llm=llm,
            tools=[browser_tool]
        )
        self.browser_tool = browser_tool
        
    async def process(self, task: Task) -> Dict[str, Any]:
        """处理UI自动化任务"""
        try:
            # 启动浏览器
            await self.browser_tool.execute("launch", headless=False)
            
            # 执行任务
            result = await super().process(task)
            
            # 清理
            await self.browser_tool.cleanup()
            
            return result
        except Exception as e:
            logger.error(f"UI Controller error: {e}")
            return {"success": False, "error": str(e)}


# ==================== 执行Agent ====================

class ExecutorAgent(BaseAgent):
    """执行Agent"""
    
    def __init__(self, agent_id: str, llm: LLMInterface):
        system_tool = SystemController()
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EXECUTOR,
            llm=llm,
            tools=[system_tool]
        )
        
    async def process(self, task: Task) -> Dict[str, Any]:
        """处理系统执行任务"""
        # 增强任务描述
        if task.type == "system_command":
            task.description = f"""Execute system task: {task.description}
            
Safety notes:
1. Only use allowed commands
2. Check file paths carefully
3. Monitor resource usage
4. Handle errors gracefully"""
            
        return await super().process(task)


# ==================== 成本优化器 ====================

class CostOptimizer:
    """成本优化器"""
    
    def __init__(self, model_costs: Dict[str, float]):
        self.model_costs = model_costs
        self.usage_history = []
        self.total_cost = 0.0
        self.max_cost = float(os.getenv("MAX_COST_PER_REQUEST", "2.0"))
        
    def estimate_task_complexity(self, task: Task) -> str:
        """估计任务复杂度"""
        description_length = len(task.description)
        
        # 简单的复杂度估计
        if description_length < 100:
            return "simple"
        elif description_length < 500:
            return "medium"
        else:
            return "complex"
            
    def select_model(self, task: Task) -> str:
        """根据任务选择模型"""
        complexity = self.estimate_task_complexity(task)
        
        if complexity == "simple":
            return "deepseek"  # 最便宜
        elif complexity == "medium":
            return "qwen"
        else:
            return "claude"  # 最强大
            
    def record_usage(self, model: str, tokens: int):
        """记录使用情况"""
        cost = self.model_costs.get(model, 0) * tokens / 1000
        self.usage_history.append({
            "model": model,
            "tokens": tokens,
            "cost": cost,
            "timestamp": datetime.now()
        })
        self.total_cost += cost
        
    def get_usage_report(self) -> Dict[str, Any]:
        """获取使用报告"""
        model_costs = {}
        for record in self.usage_history:
            model = record["model"]
            if model not in model_costs:
                model_costs[model] = 0
            model_costs[model] += record["cost"]
            
        return {
            "total_cost": self.total_cost,
            "model_costs": model_costs,
            "usage_count": len(self.usage_history),
            "average_cost_per_request": self.total_cost / max(len(self.usage_history), 1)
        }


# ==================== 智能缓存 ====================

class SmartCache:
    """智能缓存系统"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or "redis://localhost:6379"
        self.redis_client = None
        self.local_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
    async def connect(self):
        """连接Redis"""
        try:
            # 使用新版本的aioredis API
            self.redis_client = await aioredis.from_url(self.redis_url)
        except Exception as e:
            logger.warning(f"Redis connection failed, using local cache only: {e}")
            
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        # 先检查本地缓存
        if key in self.local_cache:
            self.cache_stats["hits"] += 1
            return self.local_cache[key]
            
        # 检查Redis
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    self.cache_stats["hits"] += 1
                    decoded = json.loads(value)
                    self.local_cache[key] = decoded
                    return decoded
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                
        self.cache_stats["misses"] += 1
        return None
        
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存"""
        # 本地缓存
        self.local_cache[key] = value
        
        # Redis缓存
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                
    async def get_or_compute(self, cache_type: str, params: Dict[str, Any], 
                           compute_func, ttl: int = 3600) -> Any:
        """获取或计算并缓存"""
        # 生成缓存键
        key = self._generate_key(cache_type, params)
        
        # 尝试从缓存获取
        cached = await self.get(key)
        if cached is not None:
            return cached
            
        # 计算结果
        result = await compute_func(params.get("prompt", ""))
        
        # 缓存结果
        await self.set(key, result, ttl)
        
        return result
        
    def _generate_key(self, cache_type: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        param_str = json.dumps(params, sort_keys=True)
        hash_str = hashlib.md5(param_str.encode()).hexdigest()
        return f"{cache_type}:{hash_str}"
        
    def invalidate_pattern(self, pattern: str):
        """失效匹配模式的缓存"""
        # 清理本地缓存
        keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.local_cache[key]
            
    async def close(self):
        """关闭连接"""
        if self.redis_client:
            await self.redis_client.close()


# ==================== 幻觉缓解器 ====================

class HallucinationMitigator:
    """幻觉缓解系统"""
    
    def __init__(self):
        self.fact_checkers = []
        self.confidence_threshold = 0.7
        
    async def check_response(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """检查响应中的潜在幻觉"""
        checks = {
            "has_specific_claims": self._has_specific_claims(response),
            "confidence_words": self._check_confidence_words(response),
            "consistency": await self._check_consistency(response, original_prompt),
            "fact_verification": await self._verify_facts(response)
        }
        
        # 计算整体可信度
        confidence_scores = []
        for check, result in checks.items():
            if isinstance(result, dict) and "confidence" in result:
                confidence_scores.append(result["confidence"])
                
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        return {
            "is_reliable": overall_confidence >= self.confidence_threshold,
            "overall_confidence": overall_confidence,
            "checks": checks,
            "problematic_claims": self._extract_problematic_claims(checks)
        }
        
    def _has_specific_claims(self, text: str) -> Dict[str, Any]:
        """检查是否包含具体的声明"""
        # 查找数字、日期、名称等具体信息
        import re
        
        patterns = {
            "numbers": r'\b\d+\.?\d*\b',
            "dates": r'\b\d{4}[-/]\d{2}[-/]\d{2}\b',
            "percentages": r'\b\d+\.?\d*%\b',
            "urls": r'https?://\S+',
        }
        
        specific_claims = {}
        for claim_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            specific_claims[claim_type] = matches
            
        # 具体声明越多，需要更仔细的验证
        claim_count = sum(len(matches) for matches in specific_claims.values())
        confidence = 0.9 if claim_count == 0 else max(0.3, 0.9 - claim_count * 0.1)
        
        return {
            "has_claims": claim_count > 0,
            "claims": specific_claims,
            "confidence": confidence
        }
        
    def _check_confidence_words(self, text: str) -> Dict[str, Any]:
        """检查置信度词汇"""
        high_confidence_words = ["definitely", "certainly", "absolutely", "guaranteed", "always", "never"]
        hedging_words = ["might", "maybe", "possibly", "could", "perhaps", "approximately"]
        
        text_lower = text.lower()
        
        high_confidence_count = sum(1 for word in high_confidence_words if word in text_lower)
        hedging_count = sum(1 for word in hedging_words if word in text_lower)
        
        # 过度自信的回答可能包含幻觉
        if high_confidence_count > hedging_count * 2:
            confidence = 0.6
        else:
            confidence = 0.8
            
        return {
            "high_confidence_words": high_confidence_count,
            "hedging_words": hedging_count,
            "confidence": confidence
        }
        
    async def _check_consistency(self, response: str, prompt: str) -> Dict[str, Any]:
        """检查响应的内部一致性"""
        # 简化版本 - 检查响应是否与提示相关
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        overlap = len(prompt_words & response_words)
        relevance = overlap / max(len(prompt_words), 1)
        
        return {
            "relevance": relevance,
            "confidence": min(0.9, relevance * 2)  # 相关性越高，可信度越高
        }
        
    async def _verify_facts(self, text: str) -> Dict[str, Any]:
        """验证事实声明"""
        # 简化版本 - 实际应该调用事实核查API或数据库
        unverifiable_claims = []
        
        # 检查常见的可疑模式
        suspicious_patterns = [
            r"invented in \d{4}",  # 发明日期容易出错
            r"population of .* is \d+",  # 人口数据容易过时
            r"\d+% of people",  # 统计数据需要来源
        ]
        
        import re
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                unverifiable_claims.append(pattern)
                
        confidence = 0.9 if not unverifiable_claims else 0.5
        
        return {
            "unverifiable_claims": unverifiable_claims,
            "confidence": confidence
        }
        
    def _extract_problematic_claims(self, checks: Dict[str, Any]) -> List[str]:
        """提取有问题的声明"""
        problematic = []
        
        if checks["has_specific_claims"]["has_claims"]:
            for claim_type, claims in checks["has_specific_claims"]["claims"].items():
                if claims:
                    problematic.extend([f"{claim_type}: {claim}" for claim in claims[:3]])
                    
        return problematic


# ==================== 提示优化器 ====================

class PromptOptimizer:
    """提示优化器"""
    
    def __init__(self):
        self.templates = {
            "code_generation": """You are an expert programmer. {description}

Please provide:
1. Clean, well-documented code
2. Error handling
3. Example usage
4. Performance considerations""",
            
            "research": """You are a research analyst. {description}

Please provide:
1. Comprehensive analysis
2. Multiple perspectives
3. Credible sources
4. Key insights and conclusions""",
            
            "general": """{description}"""
        }
        
        self.optimization_history = []
        
    def optimize_prompt(self, task: Task, original_prompt: str) -> str:
        """优化提示"""
        # 选择模板
        template = self.templates.get(task.type, self.templates["general"])
        
        # 填充模板
        optimized = template.format(description=original_prompt)
        
        # 添加上下文增强
        optimized = self._add_context_enhancement(optimized, task)
        
        # 记录优化
        self.optimization_history.append({
            "original": original_prompt,
            "optimized": optimized,
            "task_type": task.type,
            "timestamp": datetime.now()
        })
        
        return optimized
        
    def _add_context_enhancement(self, prompt: str, task: Task) -> str:
        """添加上下文增强"""
        enhancements = []
        
        # 根据任务类型添加特定增强
        if task.type == "code_generation":
            enhancements.append("\nUse modern Python 3.8+ features and type hints.")
        elif task.type == "research":
            enhancements.append("\nFocus on recent developments (2023-2024).")
            
        # 添加通用增强
        enhancements.append("\nBe specific and provide concrete examples.")
        
        return prompt + "\n".join(enhancements)
        
    def _analyze_and_update_templates(self):
        """分析历史并更新模板"""
        # 这里可以实现基于反馈的模板优化
        pass


# ==================== 任务并行化器 ====================

class TaskParallelizer:
    """任务并行化器"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.running_tasks = {}
        self.task_queue = asyncio.Queue()
        
    async def add_task(self, task_func, *args, **kwargs) -> str:
        """添加任务到队列"""
        task_id = str(time.time())
        await self.task_queue.put((task_id, task_func, args, kwargs))
        return task_id
        
    async def execute_parallel(self, tasks: List[Tuple]) -> Dict[str, Any]:
        """并行执行多个任务"""
        results = {}
        
        async def run_task(task_id: str, task_func, args, kwargs):
            async with self.semaphore:
                try:
                    self.running_tasks[task_id] = asyncio.current_task()
                    result = await task_func(*args, **kwargs)
                    results[task_id] = {"success": True, "result": result}
                except Exception as e:
                    results[task_id] = {"success": False, "error": str(e)}
                finally:
                    self.running_tasks.pop(task_id, None)
                    
        # 创建所有任务
        task_coroutines = []
        for task_id, task_func, args, kwargs in tasks:
            coro = run_task(task_id, task_func, args, kwargs)
            task_coroutines.append(coro)
            
        # 并行执行
        await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        return results
        
    def get_running_tasks(self) -> List[str]:
        """获取正在运行的任务"""
        return list(self.running_tasks.keys())


# ==================== 错误恢复系统 ====================

class ErrorRecoverySystem:
    """错误恢复系统"""
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delays = [1, 2, 5]  # 指数退避
        self.error_patterns = {
            "rate_limit": {"pattern": "rate limit", "wait": 60},
            "timeout": {"pattern": "timeout", "wait": 5},
            "connection": {"pattern": "connection", "wait": 2}
        }
        
    async def execute_with_recovery(self, func, *args, **kwargs) -> Dict[str, Any]:
        """带恢复机制执行函数"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1
                }
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                
                # 检查错误模式
                wait_time = self._get_wait_time(last_error, attempt)
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    
        return {
            "success": False,
            "error": last_error,
            "attempts": self.max_retries
        }
        
    def _get_wait_time(self, error: str, attempt: int) -> int:
        """根据错误类型获取等待时间"""
        error_lower = error.lower()
        
        # 检查已知错误模式
        for error_type, config in self.error_patterns.items():
            if config["pattern"] in error_lower:
                return config["wait"]
                
        # 默认指数退避
        return self.retry_delays[min(attempt, len(self.retry_delays) - 1)]


# ==================== 增强的多Agent协调器 ====================

class EnhancedMultiAgentOrchestrator:
    """增强的多Agent协调器"""
    
    def __init__(self, orchestrator_llm: LLMInterface):
        # 导入基础的MultiAgentOrchestrator
        from agent_system_implementation import MultiAgentOrchestrator
        
        # 继承基础功能
        self.base = MultiAgentOrchestrator(orchestrator_llm)
        self.orchestrator_llm = orchestrator_llm
        self.agents = {}
        self.active_tasks = {}
        self.completed_tasks = {}
        
        # 增强组件
        self.cost_optimizer = None
        self.cache = None
        self.error_recovery = ErrorRecoverySystem()
        self.task_parallelizer = TaskParallelizer()
        self.prompt_optimizer = PromptOptimizer()
        
    def register_agent(self, agent: BaseAgent):
        """注册Agent"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
        
    async def process_user_request(self, user_input: str) -> Dict[str, Any]:
        """处理用户请求 - 增强版本"""
        try:
            # 优化提示
            optimized_prompt = self.prompt_optimizer.optimize_prompt(
                Task(type="general", description=user_input),
                user_input
            )
            
            # 使用基础协调器处理
            return await self.base.process_user_request(optimized_prompt)
            
        except Exception as e:
            logger.error(f"Enhanced orchestrator error: {e}")
            return {"success": False, "error": str(e)}