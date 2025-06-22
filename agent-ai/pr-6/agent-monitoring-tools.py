# agent_monitoring.py
# Agent系统监控和管理工具

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import aiohttp
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import logging

logger = logging.getLogger(__name__)

class AgentSystemMonitor:
    """Agent系统监控器"""
    
    def __init__(self, api_url: str = "http://localhost:8000", 
                 auth_token: Optional[str] = None):
        self.api_url = api_url
        self.auth_token = auth_token
        self.metrics_history = deque(maxlen=1000)
        self.system_metrics = deque(maxlen=1000)
        
    async def collect_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': await self._get_system_metrics(),
            'agent': await self._get_agent_metrics()
        }
        
        self.metrics_history.append(metrics)
        return metrics
        
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """获取系统资源指标"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'total': psutil.virtual_memory().total,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'percent': psutil.disk_usage('/').percent
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            }
        }
        
    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """获取Agent系统指标"""
        headers = {}
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/metrics", 
                                     headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            logger.error(f"Failed to fetch agent metrics: {e}")
            
        return {}
        
    async def generate_report(self, output_file: str = "monitoring_report.html"):
        """生成监控报告"""
        if not self.metrics_history:
            return
            
        # 准备数据
        df = pd.DataFrame(list(self.metrics_history))
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU使用率
        cpu_data = [m['system']['cpu_percent'] for m in self.metrics_history]
        axes[0, 0].plot(cpu_data)
        axes[0, 0].set_title('CPU Usage %')
        axes[0, 0].set_ylim(0, 100)
        
        # 内存使用率
        mem_data = [m['system']['memory']['percent'] for m in self.metrics_history]
        axes[0, 1].plot(mem_data)
        axes[0, 1].set_title('Memory Usage %')
        axes[0, 1].set_ylim(0, 100)
        
        # Agent执行统计
        if self.metrics_history[-1].get('agent'):
            agent_metrics = self.metrics_history[-1]['agent']
            labels = ['Total', 'Success', 'Failed', 'TinyLLM']
            values = [
                agent_metrics.get('total_executions', 0),
                agent_metrics.get('total_executions', 0) - agent_metrics.get('failed_executions', 0),
                agent_metrics.get('failed_executions', 0),
                agent_metrics.get('tiny_llm_handled', 0)
            ]
            axes[1, 0].bar(labels, values)
            axes[1, 0].set_title('Execution Statistics')
        
        # 成功率趋势
        success_rates = []
        for m in self.metrics_history:
            if m.get('agent') and m['agent'].get('success_rate') is not None:
                success_rates.append(m['agent']['success_rate'] * 100)
        if success_rates:
            axes[1, 1].plot(success_rates)
            axes[1, 1].set_title('Success Rate %')
            axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('monitoring_chart.png')
        plt.close()
        
        # 生成HTML报告
        html_content = self._generate_html_report()
        with open(output_file, 'w') as f:
            f.write(html_content)
            
    def _generate_html_report(self) -> str:
        """生成HTML报告内容"""
        latest = self.metrics_history[-1] if self.metrics_history else {}
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent System Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .chart {{ margin: 20px 0; }}
                h1, h2 {{ color: #333; }}
                .status-ok {{ color: green; }}
                .status-warning {{ color: orange; }}
                .status-error {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Agent System Monitoring Report</h1>
            <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>System Metrics</h2>
            <div class="metric">
                <strong>CPU Usage:</strong> {latest.get('system', {}).get('cpu_percent', 'N/A')}%
            </div>
            <div class="metric">
                <strong>Memory Usage:</strong> {latest.get('system', {}).get('memory', {}).get('percent', 'N/A')}%
            </div>
            
            <h2>Agent Metrics</h2>
            <div class="metric">
                <strong>Total Executions:</strong> {latest.get('agent', {}).get('total_executions', 'N/A')}
            </div>
            <div class="metric">
                <strong>Success Rate:</strong> {latest.get('agent', {}).get('success_rate', 0) * 100:.2f}%
            </div>
            
            <h2>Performance Charts</h2>
            <div class="chart">
                <img src="monitoring_chart.png" alt="Performance Charts">
            </div>
        </body>
        </html>
        """

class AgentHealthChecker:
    """Agent健康检查器"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.checks = {
            'api_available': self._check_api,
            'memory_usage': self._check_memory,
            'disk_space': self._check_disk,
            'response_time': self._check_response_time
        }
        
    async def run_health_checks(self) -> Dict[str, Any]:
        """运行所有健康检查"""
        results = {}
        
        for check_name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[check_name] = result
            except Exception as e:
                results[check_name] = {
                    'status': 'error',
                    'message': str(e)
                }
                
        overall_status = self._determine_overall_status(results)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'checks': results
        }
        
    async def _check_api(self) -> Dict[str, Any]:
        """检查API可用性"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/status", 
                                     timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        return {'status': 'ok', 'message': 'API is available'}
                    else:
                        return {'status': 'error', 'message': f'API returned {response.status}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    async def _check_memory(self) -> Dict[str, Any]:
        """检查内存使用"""
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            return {'status': 'error', 'message': f'Memory usage critical: {mem.percent}%'}
        elif mem.percent > 80:
            return {'status': 'warning', 'message': f'Memory usage high: {mem.percent}%'}
        else:
            return {'status': 'ok', 'message': f'Memory usage normal: {mem.percent}%'}
            
    async def _check_disk(self) -> Dict[str, Any]:
        """检查磁盘空间"""
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            return {'status': 'error', 'message': f'Disk usage critical: {disk.percent}%'}
        elif disk.percent > 80:
            return {'status': 'warning', 'message': f'Disk usage high: {disk.percent}%'}
        else:
            return {'status': 'ok', 'message': f'Disk usage normal: {disk.percent}%'}
            
    async def _check_response_time(self) -> Dict[str, Any]:
        """检查响应时间"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/status") as response:
                    response_time = time.time() - start_time
                    
                    if response_time > 5:
                        return {'status': 'error', 'message': f'Response time too high: {response_time:.2f}s'}
                    elif response_time > 2:
                        return {'status': 'warning', 'message': f'Response time elevated: {response_time:.2f}s'}
                    else:
                        return {'status': 'ok', 'message': f'Response time normal: {response_time:.2f}s'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def _determine_overall_status(self, results: Dict[str, Any]) -> str:
        """确定整体状态"""
        statuses = [r.get('status', 'unknown') for r in results.values()]
        
        if 'error' in statuses:
            return 'error'
        elif 'warning' in statuses:
            return 'warning'
        else:
            return 'ok'

class AgentPerformanceOptimizer:
    """Agent性能优化器"""
    
    def __init__(self, config_path: str = "agent_config.yaml"):
        self.config_path = config_path
        
    async def analyze_performance(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析性能并提供优化建议"""
        recommendations = []
        
        # 分析CPU使用
        cpu_usage = [m['system']['cpu_percent'] for m in metrics_history]
        avg_cpu = sum(cpu_usage) / len(cpu_usage)
        
        if avg_cpu > 80:
            recommendations.append({
                'type': 'cpu',
                'severity': 'high',
                'recommendation': 'Consider reducing max_concurrent_tasks or enabling task queuing'
            })
            
        # 分析内存使用
        mem_usage = [m['system']['memory']['percent'] for m in metrics_history]
        avg_mem = sum(mem_usage) / len(mem_usage)
        
        if avg_mem > 80:
            recommendations.append({
                'type': 'memory',
                'severity': 'high',
                'recommendation': 'Consider reducing memory_limit_mb or optimizing model loading'
            })
            
        # 分析成功率
        if metrics_history[-1].get('agent'):
            success_rate = metrics_history[-1]['agent'].get('success_rate', 1)
            if success_rate < 0.9:
                recommendations.append({
                    'type': 'reliability',
                    'severity': 'medium',
                    'recommendation': 'Review failed tasks and consider adjusting task_timeout'
                })
                
        # 分析TinyLLM使用率
        if metrics_history[-1].get('agent'):
            total = metrics_history[-1]['agent'].get('total_executions', 1)
            tiny_llm = metrics_history[-1]['agent'].get('tiny_llm_handled', 0)
            tiny_llm_ratio = tiny_llm / total if total > 0 else 0
            
            if tiny_llm_ratio < 0.3:
                recommendations.append({
                    'type': 'cost',
                    'severity': 'low',
                    'recommendation': 'Consider lowering tiny_llm_threshold to handle more tasks locally'
                })
                
        return {
            'avg_cpu': avg_cpu,
            'avg_memory': avg_mem,
            'recommendations': recommendations
        }
        
    async def auto_optimize(self, metrics_history: List[Dict[str, Any]]):
        """自动优化配置"""
        analysis = await self.analyze_performance(metrics_history)
        
        # 根据分析结果调整配置
        import yaml
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        original_config = config.copy()
        
        # 应用优化
        if analysis['avg_cpu'] > 80:
            config['max_concurrent_tasks'] = max(5, config.get('max_concurrent_tasks', 10) - 2)
            
        if analysis['avg_memory'] > 80:
            config['memory_limit_mb'] = int(config.get('memory_limit_mb', 4096) * 0.8)
            
        # 保存优化后的配置
        if config != original_config:
            backup_path = f"{self.config_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            Path(self.config_path).rename(backup_path)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            logger.info(f"Configuration optimized. Backup saved to {backup_path}")

# 监控脚本主函数
async def run_monitoring(interval: int = 60):
    """运行持续监控"""
    monitor = AgentSystemMonitor()
    health_checker = AgentHealthChecker()
    optimizer = AgentPerformanceOptimizer()
    
    while True:
        try:
            # 收集指标
            metrics = await monitor.collect_metrics()
            
            # 健康检查
            health = await health_checker.run_health_checks()
            
            # 打印状态
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
            print(f"Status: {health['overall_status'].upper()}")
            print(f"CPU: {metrics['system']['cpu_percent']}%")
            print(f"Memory: {metrics['system']['memory']['percent']}%")
            
            if metrics.get('agent'):
                print(f"Executions: {metrics['agent'].get('total_executions', 'N/A')}")
                print(f"Success Rate: {metrics['agent'].get('success_rate', 0) * 100:.2f}%")
            
            # 每小时生成报告
            if len(monitor.metrics_history) % 60 == 0:
                await monitor.generate_report()
                print("Generated monitoring report")
                
                # 分析并优化
                if len(monitor.metrics_history) >= 10:
                    await optimizer.auto_optimize(list(monitor.metrics_history)[-60:])
            
            await asyncio.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(interval)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Agent System Monitor')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--report', action='store_true', help='Generate report and exit')
    
    args = parser.parse_args()
    
    if args.report:
        async def generate_report_only():
            monitor = AgentSystemMonitor()
            # 收集一些样本数据
            for _ in range(10):
                await monitor.collect_metrics()
                await asyncio.sleep(1)
            await monitor.generate_report()
            print("Report generated: monitoring_report.html")
            
        asyncio.run(generate_report_only())
    else:
        asyncio.run(run_monitoring(args.interval))