"""
智能调试和诊断系统
提供代码调试、性能诊断、错误追踪和自动修复功能
"""

import asyncio
import ast
import dis
import inspect
import linecache
import sys
import traceback
import cProfile
import pstats
import io
import gc
import psutil
import objgraph
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
from contextlib import contextmanager
import threading
import time
import re
import json
import logging
from pathlib import Path
import numpy as np

# 调试和分析工具
import py_compile
import pylint.lint
from pylint.reporters.text import TextReporter
import autopep8
import black
import isort
from memory_profiler import profile as memory_profile
import line_profiler

# ==================== 调试器核心 ====================

@dataclass
class DebugContext:
    """调试上下文"""
    session_id: str
    target_code: str
    breakpoints: List[int] = field(default_factory=list)
    watch_variables: List[str] = field(default_factory=list)
    call_stack: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiagnosticResult:
    """诊断结果"""
    issue_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    location: Dict[str, Any]
    description: str
    suggested_fix: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class IntelligentDebugger:
    """智能调试器"""
    
    def __init__(self):
        self.sessions: Dict[str, DebugContext] = {}
        self.code_analyzer = CodeAnalyzer()
        self.error_predictor = ErrorPredictor()
        self.fix_suggester = FixSuggester()
        self.performance_profiler = PerformanceProfiler()
        self.memory_analyzer = MemoryAnalyzer()
        
    async def create_debug_session(self, code: str, config: Dict[str, Any] = None) -> str:
        """创建调试会话"""
        session_id = f"debug_{datetime.now().timestamp()}"
        
        context = DebugContext(
            session_id=session_id,
            target_code=code
        )
        
        self.sessions[session_id] = context
        
        # 初始分析
        initial_analysis = await self.analyze_code(code)
        context.variables['__analysis__'] = initial_analysis
        
        return session_id
    
    async def analyze_code(self, code: str) -> Dict[str, Any]:
        """分析代码"""
        analysis = {
            'syntax_check': await self.check_syntax(code),
            'complexity': await self.analyze_complexity(code),
            'potential_issues': await self.detect_potential_issues(code),
            'dependencies': await self.analyze_dependencies(code),
            'metrics': await self.calculate_code_metrics(code)
        }
        
        return analysis
    
    async def check_syntax(self, code: str) -> Dict[str, Any]:
        """检查语法"""
        try:
            ast.parse(code)
            return {'valid': True, 'errors': []}
        except SyntaxError as e:
            return {
                'valid': False,
                'errors': [{
                    'line': e.lineno,
                    'column': e.offset,
                    'message': str(e),
                    'text': e.text
                }]
            }
    
    async def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """分析代码复杂度"""
        try:
            tree = ast.parse(code)
            analyzer = ComplexityAnalyzer()
            analyzer.visit(tree)
            
            return {
                'cyclomatic_complexity': analyzer.complexity,
                'cognitive_complexity': analyzer.cognitive_complexity,
                'nesting_depth': analyzer.max_nesting,
                'function_complexities': analyzer.function_complexities
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def detect_potential_issues(self, code: str) -> List[DiagnosticResult]:
        """检测潜在问题"""
        issues = []
        
        # 使用AST检测常见问题
        try:
            tree = ast.parse(code)
            issue_detector = IssueDetector()
            issue_detector.visit(tree)
            issues.extend(issue_detector.issues)
        except:
            pass
        
        # 使用模式匹配检测问题
        pattern_issues = await self.detect_pattern_issues(code)
        issues.extend(pattern_issues)
        
        # 使用AI预测潜在错误
        predicted_errors = await self.error_predictor.predict(code)
        issues.extend(predicted_errors)
        
        return issues
    
    async def detect_pattern_issues(self, code: str) -> List[DiagnosticResult]:
        """使用模式匹配检测问题"""
        issues = []
        patterns = [
            {
                'pattern': r'except\s*:',
                'issue': 'Bare except clause',
                'severity': 'medium',
                'fix': 'Specify exception type'
            },
            {
                'pattern': r'import\s+\*',
                'issue': 'Wildcard import',
                'severity': 'low',
                'fix': 'Import specific names'
            },
            {
                'pattern': r'eval\s*\(',
                'issue': 'Use of eval()',
                'severity': 'high',
                'fix': 'Use ast.literal_eval() or alternative'
            },
            {
                'pattern': r'global\s+',
                'issue': 'Use of global variable',
                'severity': 'medium',
                'fix': 'Consider refactoring to avoid global state'
            }
        ]
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            for match in re.finditer(pattern, code):
                line_no = code[:match.start()].count('\n') + 1
                
                issue = DiagnosticResult(
                    issue_type='pattern_match',
                    severity=pattern_info['severity'],
                    location={'line': line_no, 'column': match.start()},
                    description=pattern_info['issue'],
                    suggested_fix=pattern_info['fix'],
                    confidence=0.9
                )
                issues.append(issue)
                
        return issues
    
    async def debug_execution(self, session_id: str, 
                            input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """调试执行"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        context = self.sessions[session_id]
        
        # 创建调试环境
        debug_env = DebugEnvironment(context)
        
        # 设置断点
        for bp in context.breakpoints:
            debug_env.set_breakpoint(bp)
            
        # 执行代码
        try:
            result = await debug_env.execute(context.target_code, input_data)
            
            return {
                'success': True,
                'result': result,
                'execution_history': context.execution_history,
                'variables': context.variables,
                'call_stack': context.call_stack,
                'performance': context.performance_data
            }
            
        except Exception as e:
            # 捕获并分析错误
            error_analysis = await self.analyze_error(e, context)
            
            return {
                'success': False,
                'error': str(e),
                'error_analysis': error_analysis,
                'execution_history': context.execution_history,
                'variables': context.variables,
                'call_stack': context.call_stack
            }
    
    async def analyze_error(self, error: Exception, 
                          context: DebugContext) -> Dict[str, Any]:
        """分析错误"""
        tb = traceback.extract_tb(error.__traceback__)
        
        analysis = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': [
                {
                    'filename': frame.filename,
                    'line': frame.lineno,
                    'function': frame.name,
                    'code': frame.line
                }
                for frame in tb
            ],
            'root_cause': await self.find_root_cause(error, context),
            'suggested_fixes': await self.fix_suggester.suggest_fixes(error, context),
            'similar_errors': await self.find_similar_errors(error)
        }
        
        return analysis
    
    async def find_root_cause(self, error: Exception, 
                            context: DebugContext) -> Dict[str, Any]:
        """查找根本原因"""
        # 分析错误类型
        if isinstance(error, NameError):
            return await self.analyze_name_error(error, context)
        elif isinstance(error, TypeError):
            return await self.analyze_type_error(error, context)
        elif isinstance(error, ValueError):
            return await self.analyze_value_error(error, context)
        elif isinstance(error, AttributeError):
            return await self.analyze_attribute_error(error, context)
        else:
            return {'type': 'unknown', 'description': 'Unable to determine root cause'}
    
    async def analyze_name_error(self, error: NameError, 
                               context: DebugContext) -> Dict[str, Any]:
        """分析名称错误"""
        error_msg = str(error)
        match = re.search(r"name '(\w+)' is not defined", error_msg)
        
        if match:
            undefined_name = match.group(1)
            
            # 查找相似变量名
            similar_names = []
            for var_name in context.variables:
                similarity = self.calculate_similarity(undefined_name, var_name)
                if similarity > 0.7:
                    similar_names.append((var_name, similarity))
                    
            similar_names.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'type': 'undefined_variable',
                'variable': undefined_name,
                'similar_variables': [name for name, _ in similar_names[:3]],
                'suggestion': f"Did you mean '{similar_names[0][0]}'?" if similar_names else None
            }
            
        return {'type': 'name_error', 'description': error_msg}
    
    def calculate_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        # 使用Levenshtein距离
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
            
        max_len = max(len(s1), len(s2))
        distance = self.levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算Levenshtein距离"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]

# ==================== 性能分析器 ====================

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profile_data = {}
        self.bottlenecks = []
        self.optimization_suggestions = []
        
    @contextmanager
    def profile(self, name: str):
        """性能分析上下文管理器"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        yield
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        self.profile_data[name] = {
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'timestamp': datetime.now()
        }
    
    async def profile_code(self, code: str, test_inputs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """分析代码性能"""
        results = {
            'execution_profile': await self.profile_execution(code, test_inputs),
            'memory_profile': await self.profile_memory(code, test_inputs),
            'complexity_analysis': await self.analyze_algorithmic_complexity(code),
            'bottlenecks': await self.identify_bottlenecks(code),
            'optimization_suggestions': await self.generate_optimizations(code)
        }
        
        return results
    
    async def profile_execution(self, code: str, test_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行性能分析"""
        profiler = cProfile.Profile()
        
        # 编译代码
        compiled_code = compile(code, '<string>', 'exec')
        
        # 对每个测试输入进行分析
        results = []
        for test_input in test_inputs or [{}]:
            profiler.enable()
            
            exec_globals = test_input.copy()
            try:
                exec(compiled_code, exec_globals)
            except Exception as e:
                logging.error(f"Profiling error: {e}")
                
            profiler.disable()
            
            # 获取统计信息
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # 打印前20个函数
            
            results.append({
                'input': test_input,
                'profile': s.getvalue(),
                'top_functions': self.extract_top_functions(ps)
            })
            
        return {
            'test_results': results,
            'summary': self.summarize_profile_results(results)
        }
    
    def extract_top_functions(self, stats: pstats.Stats) -> List[Dict[str, Any]]:
        """提取顶部函数"""
        top_functions = []
        
        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:10]:
            top_functions.append({
                'function': f"{func[0]}:{func[1]}:{func[2]}",
                'call_count': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / nc if nc > 0 else 0
            })
            
        return top_functions
    
    async def profile_memory(self, code: str, test_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """内存性能分析"""
        memory_usage = []
        
        # 使用memory_profiler
        for test_input in test_inputs or [{}]:
            # 监控内存使用
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            exec_globals = test_input.copy()
            try:
                exec(compile(code, '<string>', 'exec'), exec_globals)
            except Exception as e:
                logging.error(f"Memory profiling error: {e}")
                
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            memory_usage.append({
                'input': test_input,
                'memory_before': mem_before,
                'memory_after': mem_after,
                'memory_delta': mem_after - mem_before
            })
            
        return {
            'memory_usage': memory_usage,
            'peak_memory': max(u['memory_after'] for u in memory_usage) if memory_usage else 0,
            'average_delta': np.mean([u['memory_delta'] for u in memory_usage]) if memory_usage else 0
        }
    
    async def analyze_algorithmic_complexity(self, code: str) -> Dict[str, Any]:
        """分析算法复杂度"""
        try:
            tree = ast.parse(code)
            complexity_analyzer = AlgorithmicComplexityAnalyzer()
            complexity_analyzer.visit(tree)
            
            return {
                'time_complexity': complexity_analyzer.time_complexity,
                'space_complexity': complexity_analyzer.space_complexity,
                'loop_analysis': complexity_analyzer.loop_analysis,
                'recursive_analysis': complexity_analyzer.recursive_analysis
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def identify_bottlenecks(self, code: str) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        try:
            tree = ast.parse(code)
            bottleneck_detector = BottleneckDetector()
            bottleneck_detector.visit(tree)
            
            bottlenecks = [
                {
                    'type': b['type'],
                    'location': b['location'],
                    'description': b['description'],
                    'impact': b['impact'],
                    'suggestion': b['suggestion']
                }
                for b in bottleneck_detector.bottlenecks
            ]
            
        except Exception as e:
            logging.error(f"Bottleneck detection error: {e}")
            
        return bottlenecks
    
    async def generate_optimizations(self, code: str) -> List[Dict[str, Any]]:
        """生成优化建议"""
        optimizations = []
        
        # 分析代码模式
        patterns = [
            {
                'pattern': r'for .+ in range\(len\((.+)\)\):',
                'suggestion': 'Use enumerate() instead of range(len())',
                'example': 'for i, item in enumerate(items):'
            },
            {
                'pattern': r'\.append\(.+\) for .+ in',
                'suggestion': 'Use list comprehension instead of append in loop',
                'example': '[item for item in items]'
            },
            {
                'pattern': r'if .+ == True:',
                'suggestion': 'Simplify boolean comparison',
                'example': 'if condition:'
            }
        ]
        
        for pattern_info in patterns:
            matches = re.finditer(pattern_info['pattern'], code)
            for match in matches:
                line_no = code[:match.start()].count('\n') + 1
                
                optimizations.append({
                    'type': 'pattern_optimization',
                    'location': {'line': line_no},
                    'current_code': match.group(0),
                    'suggestion': pattern_info['suggestion'],
                    'example': pattern_info['example'],
                    'estimated_improvement': 'minor'
                })
                
        return optimizations

# ==================== 内存分析器 ====================

class MemoryAnalyzer:
    """内存分析器"""
    
    def __init__(self):
        self.memory_snapshots = []
        self.leak_suspects = []
        
    async def analyze_memory(self, code: str, duration: int = 60) -> Dict[str, Any]:
        """分析内存使用"""
        results = {
            'memory_usage': await self.track_memory_usage(code, duration),
            'object_growth': await self.analyze_object_growth(code),
            'leak_detection': await self.detect_memory_leaks(code),
            'gc_stats': await self.analyze_garbage_collection(code)
        }
        
        return results
    
    async def track_memory_usage(self, code: str, duration: int) -> Dict[str, Any]:
        """跟踪内存使用"""
        memory_timeline = []
        start_time = time.time()
        
        # 启动代码执行
        exec_globals = {}
        exec_thread = threading.Thread(
            target=lambda: exec(compile(code, '<string>', 'exec'), exec_globals)
        )
        exec_thread.start()
        
        # 监控内存
        while time.time() - start_time < duration and exec_thread.is_alive():
            mem_info = psutil.Process().memory_info()
            memory_timeline.append({
                'timestamp': time.time() - start_time,
                'rss': mem_info.rss / 1024 / 1024,  # MB
                'vms': mem_info.vms / 1024 / 1024,  # MB
                'percent': psutil.Process().memory_percent()
            })
            await asyncio.sleep(0.1)
            
        exec_thread.join(timeout=1)
        
        return {
            'timeline': memory_timeline,
            'peak_rss': max(m['rss'] for m in memory_timeline) if memory_timeline else 0,
            'average_rss': np.mean([m['rss'] for m in memory_timeline]) if memory_timeline else 0
        }
    
    async def analyze_object_growth(self, code: str) -> Dict[str, Any]:
        """分析对象增长"""
        # 获取初始对象计数
        gc.collect()
        before_objects = objgraph.by_type('dict')
        before_count = len(before_objects)
        
        # 执行代码
        exec_globals = {}
        try:
            exec(compile(code, '<string>', 'exec'), exec_globals)
        except Exception as e:
            logging.error(f"Object growth analysis error: {e}")
            
        # 获取执行后对象计数
        gc.collect()
        after_objects = objgraph.by_type('dict')
        after_count = len(after_objects)
        
        # 分析增长
        growth = after_count - before_count
        
        # 获取最常见的对象类型
        type_stats = objgraph.typestats()
        top_types = sorted(type_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'object_growth': growth,
            'before_count': before_count,
            'after_count': after_count,
            'top_object_types': dict(top_types),
            'growth_rate': growth / before_count if before_count > 0 else 0
        }
    
    async def detect_memory_leaks(self, code: str) -> Dict[str, Any]:
        """检测内存泄漏"""
        leaks = []
        
        # 使用tracemalloc
        import tracemalloc
        tracemalloc.start()
        
        # 执行代码多次
        for i in range(5):
            exec_globals = {}
            try:
                exec(compile(code, '<string>', 'exec'), exec_globals)
            except Exception as e:
                logging.error(f"Memory leak detection error: {e}")
                
            if i == 0:
                snapshot1 = tracemalloc.take_snapshot()
            elif i == 4:
                snapshot2 = tracemalloc.take_snapshot()
                
        # 比较快照
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        for stat in top_stats[:10]:
            if stat.size_diff > 0:
                leaks.append({
                    'location': str(stat),
                    'size_diff': stat.size_diff,
                    'count_diff': stat.count_diff,
                    'severity': 'high' if stat.size_diff > 1024 * 1024 else 'medium'
                })
                
        tracemalloc.stop()
        
        return {
            'leak_suspects': leaks,
            'total_leaked': sum(l['size_diff'] for l in leaks),
            'leak_detected': len(leaks) > 0
        }
    
    async def analyze_garbage_collection(self, code: str) -> Dict[str, Any]:
        """分析垃圾回收"""
        # 获取GC统计前的状态
        gc.collect()
        before_stats = gc.get_stats()
        before_count = gc.get_count()
        
        # 执行代码
        exec_globals = {}
        try:
            exec(compile(code, '<string>', 'exec'), exec_globals)
        except Exception as e:
            logging.error(f"GC analysis error: {e}")
            
        # 获取GC统计后的状态
        gc.collect()
        after_stats = gc.get_stats()
        after_count = gc.get_count()
        
        return {
            'collections': {
                'gen0': after_count[0] - before_count[0],
                'gen1': after_count[1] - before_count[1],
                'gen2': after_count[2] - before_count[2]
            },
            'gc_enabled': gc.isenabled(),
            'threshold': gc.get_threshold(),
            'stats': after_stats
        }

# ==================== 错误预测器 ====================

class ErrorPredictor:
    """错误预测器"""
    
    def __init__(self):
        self.error_patterns = self.load_error_patterns()
        self.ml_model = None  # 可以集成ML模型
        
    def load_error_patterns(self) -> List[Dict[str, Any]]:
        """加载错误模式"""
        return [
            {
                'pattern': r'(\w+)\s*=\s*\1\s*\+\s*1',
                'error_type': 'potential_infinite_loop',
                'description': 'Variable increment might cause infinite loop',
                'confidence': 0.7
            },
            {
                'pattern': r'open\(.+\)(?!.*\.close\(\))',
                'error_type': 'unclosed_file',
                'description': 'File opened but not closed',
                'confidence': 0.8
            },
            {
                'pattern': r'def\s+\w+\(.*\):\s*$',
                'error_type': 'empty_function',
                'description': 'Function has no implementation',
                'confidence': 0.9
            }
        ]
    
    async def predict(self, code: str) -> List[DiagnosticResult]:
        """预测潜在错误"""
        predictions = []
        
        # 基于模式的预测
        pattern_predictions = await self.pattern_based_prediction(code)
        predictions.extend(pattern_predictions)
        
        # 基于AST的预测
        ast_predictions = await self.ast_based_prediction(code)
        predictions.extend(ast_predictions)
        
        # 基于数据流的预测
        dataflow_predictions = await self.dataflow_based_prediction(code)
        predictions.extend(dataflow_predictions)
        
        return predictions
    
    async def pattern_based_prediction(self, code: str) -> List[DiagnosticResult]:
        """基于模式的预测"""
        predictions = []
        
        for pattern_info in self.error_patterns:
            pattern = pattern_info['pattern']
            for match in re.finditer(pattern, code, re.MULTILINE):
                line_no = code[:match.start()].count('\n') + 1
                
                prediction = DiagnosticResult(
                    issue_type=pattern_info['error_type'],
                    severity='medium',
                    location={'line': line_no, 'column': match.start()},
                    description=pattern_info['description'],
                    confidence=pattern_info['confidence']
                )
                predictions.append(prediction)
                
        return predictions
    
    async def ast_based_prediction(self, code: str) -> List[DiagnosticResult]:
        """基于AST的预测"""
        predictions = []
        
        try:
            tree = ast.parse(code)
            predictor = ASTErrorPredictor()
            predictor.visit(tree)
            predictions.extend(predictor.predictions)
        except:
            pass
            
        return predictions
    
    async def dataflow_based_prediction(self, code: str) -> List[DiagnosticResult]:
        """基于数据流的预测"""
        predictions = []
        
        try:
            analyzer = DataFlowAnalyzer()
            issues = await analyzer.analyze(code)
            
            for issue in issues:
                prediction = DiagnosticResult(
                    issue_type='dataflow_issue',
                    severity=issue['severity'],
                    location=issue['location'],
                    description=issue['description'],
                    confidence=issue.get('confidence', 0.7)
                )
                predictions.append(prediction)
                
        except Exception as e:
            logging.error(f"Dataflow prediction error: {e}")
            
        return predictions

# ==================== 修复建议器 ====================

class FixSuggester:
    """修复建议器"""
    
    def __init__(self):
        self.fix_patterns = self.load_fix_patterns()
        self.code_transformer = CodeTransformer()
        
    def load_fix_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载修复模式"""
        return {
            'NameError': [
                {
                    'pattern': r"name '(\w+)' is not defined",
                    'fix_type': 'import_missing',
                    'fix_template': 'import {module}'
                },
                {
                    'pattern': r"name '(\w+)' is not defined",
                    'fix_type': 'define_variable',
                    'fix_template': '{variable} = None  # Define variable'
                }
            ],
            'TypeError': [
                {
                    'pattern': r"unsupported operand type\(s\)",
                    'fix_type': 'type_conversion',
                    'fix_template': 'Convert types before operation'
                }
            ],
            'AttributeError': [
                {
                    'pattern': r"'(\w+)' object has no attribute '(\w+)'",
                    'fix_type': 'check_attribute',
                    'fix_template': 'if hasattr({object}, "{attribute}"):'
                }
            ]
        }
    
    async def suggest_fixes(self, error: Exception, 
                          context: DebugContext) -> List[Dict[str, Any]]:
        """建议修复方案"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        suggestions = []
        
        # 基于错误类型的修复
        if error_type in self.fix_patterns:
            pattern_fixes = await self.get_pattern_fixes(error_type, error_msg)
            suggestions.extend(pattern_fixes)
            
        # 基于上下文的修复
        context_fixes = await self.get_context_fixes(error, context)
        suggestions.extend(context_fixes)
        
        # 基于代码分析的修复
        analysis_fixes = await self.get_analysis_fixes(error, context)
        suggestions.extend(analysis_fixes)
        
        # 排序和去重
        suggestions = self.rank_suggestions(suggestions)
        
        return suggestions[:5]  # 返回前5个建议
    
    async def get_pattern_fixes(self, error_type: str, error_msg: str) -> List[Dict[str, Any]]:
        """获取基于模式的修复"""
        fixes = []
        
        for pattern_info in self.fix_patterns.get(error_type, []):
            pattern = pattern_info['pattern']
            match = re.search(pattern, error_msg)
            
            if match:
                fix = {
                    'type': pattern_info['fix_type'],
                    'description': pattern_info['fix_template'],
                    'code': self.generate_fix_code(pattern_info, match),
                    'confidence': 0.8,
                    'automated': True
                }
                fixes.append(fix)
                
        return fixes
    
    def generate_fix_code(self, pattern_info: Dict[str, Any], 
                         match: re.Match) -> str:
        """生成修复代码"""
        template = pattern_info['fix_template']
        
        # 替换模板中的占位符
        if match.groups():
            for i, group in enumerate(match.groups()):
                template = template.replace(f'{{{i}}}', group)
                
        return template
    
    async def get_context_fixes(self, error: Exception, 
                              context: DebugContext) -> List[Dict[str, Any]]:
        """获取基于上下文的修复"""
        fixes = []
        
        # 分析变量使用
        if isinstance(error, NameError):
            undefined_var = re.search(r"name '(\w+)' is not defined", str(error))
            if undefined_var:
                var_name = undefined_var.group(1)
                
                # 检查是否在其他作用域定义
                if var_name in context.variables:
                    fixes.append({
                        'type': 'scope_fix',
                        'description': f'Variable {var_name} exists in outer scope',
                        'code': f'global {var_name}',
                        'confidence': 0.7,
                        'automated': True
                    })
                    
        return fixes
    
    async def get_analysis_fixes(self, error: Exception, 
                               context: DebugContext) -> List[Dict[str, Any]]:
        """获取基于分析的修复"""
        fixes = []
        
        # 使用代码转换器生成修复
        try:
            transformed_code = await self.code_transformer.transform(
                context.target_code,
                error
            )
            
            if transformed_code != context.target_code:
                fixes.append({
                    'type': 'automated_transform',
                    'description': 'Automated code transformation',
                    'code': transformed_code,
                    'confidence': 0.6,
                    'automated': True
                })
                
        except Exception as e:
            logging.error(f"Code transformation error: {e}")
            
        return fixes
    
    def rank_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """排序建议"""
        # 按置信度排序
        suggestions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # 去重
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            key = (suggestion['type'], suggestion.get('code', ''))
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)
                
        return unique_suggestions

# ==================== AST分析器 ====================

class ComplexityAnalyzer(ast.NodeVisitor):
    """复杂度分析器"""
    
    def __init__(self):
        self.complexity = 0
        self.cognitive_complexity = 0
        self.max_nesting = 0
        self.current_nesting = 0
        self.function_complexities = {}
        
    def visit_If(self, node):
        self.complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting
        self.current_nesting += 1
        self.generic_visit(node)
        self.current_nesting -= 1
        
    def visit_For(self, node):
        self.complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
        
    def visit_While(self, node):
        self.complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
        
    def visit_FunctionDef(self, node):
        old_complexity = self.complexity
        self.complexity = 1  # 函数本身的复杂度
        
        self.generic_visit(node)
        
        self.function_complexities[node.name] = self.complexity
        self.complexity = old_complexity

class IssueDetector(ast.NodeVisitor):
    """问题检测器"""
    
    def __init__(self):
        self.issues = []
        
    def visit_Assign(self, node):
        # 检测未使用的变量
        if isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if var_name.startswith('_'):
                return
                
            # 这里应该检查变量是否被使用
            # 简化版本：检查是否是临时变量模式
            if var_name in ['tmp', 'temp', 'x', 'i', 'j']:
                self.issues.append(DiagnosticResult(
                    issue_type='poor_naming',
                    severity='low',
                    location={'line': node.lineno, 'column': node.col_offset},
                    description=f'Poor variable name: {var_name}',
                    suggested_fix='Use descriptive variable names',
                    confidence=0.7
                ))
                
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        # 检测过长的函数
        if len(node.body) > 50:
            self.issues.append(DiagnosticResult(
                issue_type='long_function',
                severity='medium',
                location={'line': node.lineno, 'column': node.col_offset},
                description=f'Function {node.name} is too long ({len(node.body)} lines)',
                suggested_fix='Consider breaking into smaller functions',
                confidence=0.8
            ))
            
        # 检测过多的参数
        if len(node.args.args) > 5:
            self.issues.append(DiagnosticResult(
                issue_type='too_many_parameters',
                severity='medium',
                location={'line': node.lineno, 'column': node.col_offset},
                description=f'Function {node.name} has too many parameters ({len(node.args.args)})',
                suggested_fix='Consider using configuration object or builder pattern',
                confidence=0.8
            ))
            
        self.generic_visit(node)

class ASTErrorPredictor(ast.NodeVisitor):
    """AST错误预测器"""
    
    def __init__(self):
        self.predictions = []
        self.defined_vars = set()
        self.used_vars = set()
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defined_vars.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.used_vars.add(node.id)
            
            # 检查是否使用了未定义的变量
            if node.id not in self.defined_vars and not self.is_builtin(node.id):
                self.predictions.append(DiagnosticResult(
                    issue_type='potential_name_error',
                    severity='high',
                    location={'line': node.lineno, 'column': node.col_offset},
                    description=f'Potential undefined variable: {node.id}',
                    confidence=0.8
                ))
                
        self.generic_visit(node)
        
    def is_builtin(self, name: str) -> bool:
        """检查是否是内置名称"""
        import builtins
        return hasattr(builtins, name)

class AlgorithmicComplexityAnalyzer(ast.NodeVisitor):
    """算法复杂度分析器"""
    
    def __init__(self):
        self.time_complexity = 'O(1)'
        self.space_complexity = 'O(1)'
        self.loop_analysis = []
        self.recursive_analysis = []
        self.loop_depth = 0
        self.in_function = None
        
    def visit_For(self, node):
        self.loop_depth += 1
        
        # 分析循环复杂度
        loop_info = {
            'type': 'for',
            'depth': self.loop_depth,
            'line': node.lineno
        }
        
        # 检查是否是嵌套循环
        if self.loop_depth > 1:
            self.time_complexity = f'O(n^{self.loop_depth})'
            loop_info['nested'] = True
            
        self.loop_analysis.append(loop_info)
        
        self.generic_visit(node)
        self.loop_depth -= 1
        
    def visit_While(self, node):
        self.loop_depth += 1
        
        loop_info = {
            'type': 'while',
            'depth': self.loop_depth,
            'line': node.lineno
        }
        
        if self.loop_depth > 1:
            self.time_complexity = f'O(n^{self.loop_depth})'
            loop_info['nested'] = True
            
        self.loop_analysis.append(loop_info)
        
        self.generic_visit(node)
        self.loop_depth -= 1
        
    def visit_FunctionDef(self, node):
        old_function = self.in_function
        self.in_function = node.name
        
        # 检查是否是递归函数
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == node.name:
                    self.recursive_analysis.append({
                        'function': node.name,
                        'line': child.lineno,
                        'type': 'direct_recursion'
                    })
                    # 简化的复杂度估计
                    self.time_complexity = 'O(2^n) or O(n!)'
                    
        self.generic_visit(node)
        self.in_function = old_function

class BottleneckDetector(ast.NodeVisitor):
    """瓶颈检测器"""
    
    def __init__(self):
        self.bottlenecks = []
        
    def visit_For(self, node):
        # 检查循环中的列表操作
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr == 'append':
                        self.bottlenecks.append({
                            'type': 'list_append_in_loop',
                            'location': {'line': child.lineno},
                            'description': 'List append in loop can be inefficient',
                            'impact': 'medium',
                            'suggestion': 'Consider using list comprehension or preallocating list'
                        })
                        
        self.generic_visit(node)
        
    def visit_BinOp(self, node):
        # 检查字符串连接
        if isinstance(node.op, ast.Add):
            if (isinstance(node.left, ast.Str) or 
                isinstance(node.right, ast.Str)):
                self.bottlenecks.append({
                    'type': 'string_concatenation',
                    'location': {'line': node.lineno},
                    'description': 'String concatenation can be inefficient',
                    'impact': 'low',
                    'suggestion': 'Use join() or f-strings for multiple concatenations'
                })
                
        self.generic_visit(node)

# ==================== 代码转换器 ====================

class CodeTransformer:
    """代码转换器"""
    
    async def transform(self, code: str, error: Exception) -> str:
        """转换代码以修复错误"""
        try:
            tree = ast.parse(code)
            
            # 根据错误类型选择转换器
            if isinstance(error, NameError):
                transformer = NameErrorTransformer(error)
            elif isinstance(error, TypeError):
                transformer = TypeErrorTransformer(error)
            else:
                return code
                
            # 转换AST
            new_tree = transformer.visit(tree)
            
            # 转换回代码
            import astor
            return astor.to_source(new_tree)
            
        except Exception as e:
            logging.error(f"Code transformation error: {e}")
            return code

class NameErrorTransformer(ast.NodeTransformer):
    """名称错误转换器"""
    
    def __init__(self, error: NameError):
        self.error = error
        self.undefined_name = self.extract_undefined_name()
        
    def extract_undefined_name(self) -> str:
        """提取未定义的名称"""
        match = re.search(r"name '(\w+)' is not defined", str(self.error))
        return match.group(1) if match else None
        
    def visit_Module(self, node):
        # 在模块开始添加变量定义
        if self.undefined_name:
            assign = ast.Assign(
                targets=[ast.Name(id=self.undefined_name, ctx=ast.Store())],
                value=ast.Constant(value=None)
            )
            node.body.insert(0, assign)
            
        self.generic_visit(node)
        return node

# ==================== 调试环境 ====================

class DebugEnvironment:
    """调试环境"""
    
    def __init__(self, context: DebugContext):
        self.context = context
        self.breakpoints = set(context.breakpoints)
        self.current_line = 0
        self.step_mode = False
        
    def set_breakpoint(self, line: int):
        """设置断点"""
        self.breakpoints.add(line)
        
    def remove_breakpoint(self, line: int):
        """移除断点"""
        self.breakpoints.discard(line)
        
    async def execute(self, code: str, input_data: Dict[str, Any] = None) -> Any:
        """执行代码"""
        # 创建执行环境
        exec_globals = input_data.copy() if input_data else {}
        exec_globals['__debug_env__'] = self
        
        # 注入调试钩子
        instrumented_code = await self.instrument_code(code)
        
        # 执行
        try:
            exec(compile(instrumented_code, '<debug>', 'exec'), exec_globals)
            return exec_globals
        except Exception as e:
            # 记录错误信息
            self.context.execution_history.append({
                'type': 'error',
                'line': self.current_line,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
            
    async def instrument_code(self, code: str) -> str:
        """注入调试代码"""
        lines = code.split('\n')
        instrumented_lines = []
        
        for i, line in enumerate(lines, 1):
            # 在每行前添加调试钩子
            if line.strip():
                hook = f"__debug_env__.debug_hook({i}, locals())"
                instrumented_lines.append(hook)
                
            instrumented_lines.append(line)
            
        return '\n'.join(instrumented_lines)
        
    def debug_hook(self, line: int, local_vars: Dict[str, Any]):
        """调试钩子"""
        self.current_line = line
        
        # 更新变量
        self.context.variables.update(local_vars)
        
        # 记录执行历史
        self.context.execution_history.append({
            'type': 'execution',
            'line': line,
            'variables': local_vars.copy()
        })
        
        # 检查断点
        if line in self.breakpoints or self.step_mode:
            self.handle_breakpoint(line, local_vars)
            
    def handle_breakpoint(self, line: int, local_vars: Dict[str, Any]):
        """处理断点"""
        # 这里可以实现交互式调试
        print(f"Breakpoint at line {line}")
        print(f"Variables: {local_vars}")

# ==================== 数据流分析器 ====================

class DataFlowAnalyzer:
    """数据流分析器"""
    
    async def analyze(self, code: str) -> List[Dict[str, Any]]:
        """分析数据流"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # 构建控制流图
            cfg = await self.build_control_flow_graph(tree)
            
            # 分析未初始化变量
            uninitialized = await self.find_uninitialized_variables(cfg)
            issues.extend(uninitialized)
            
            # 分析未使用变量
            unused = await self.find_unused_variables(tree)
            issues.extend(unused)
            
            # 分析死代码
            dead_code = await self.find_dead_code(cfg)
            issues.extend(dead_code)
            
        except Exception as e:
            logging.error(f"Data flow analysis error: {e}")
            
        return issues
    
    async def build_control_flow_graph(self, tree: ast.AST) -> nx.DiGraph:
        """构建控制流图"""
        cfg = nx.DiGraph()
        
        # 简化的CFG构建
        # 实际实现需要更复杂的算法
        
        return cfg
    
    async def find_uninitialized_variables(self, cfg: nx.DiGraph) -> List[Dict[str, Any]]:
        """查找未初始化的变量"""
        issues = []
        
        # 简化的实现
        # 实际需要通过数据流分析
        
        return issues
    
    async def find_unused_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """查找未使用的变量"""
        issues = []
        
        defined_vars = set()
        used_vars = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined_vars.add((node.id, node.lineno))
                elif isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
                    
        for var_name, line_no in defined_vars:
            if var_name not in used_vars and not var_name.startswith('_'):
                issues.append({
                    'severity': 'low',
                    'location': {'line': line_no},
                    'description': f'Variable {var_name} is defined but never used'
                })
                
        return issues
    
    async def find_dead_code(self, cfg: nx.DiGraph) -> List[Dict[str, Any]]:
        """查找死代码"""
        issues = []
        
        # 简化的实现
        # 实际需要通过可达性分析
        
        return issues

# ==================== 诊断报告生成器 ====================

class DiagnosticReportGenerator:
    """诊断报告生成器"""
    
    async def generate_report(self, debug_session: str, 
                            debugger: IntelligentDebugger) -> Dict[str, Any]:
        """生成诊断报告"""
        if debug_session not in debugger.sessions:
            raise ValueError(f"Session {debug_session} not found")
            
        context = debugger.sessions[debug_session]
        
        report = {
            'session_id': debug_session,
            'timestamp': datetime.now(),
            'code_analysis': context.variables.get('__analysis__', {}),
            'execution_summary': await self.summarize_execution(context),
            'issues_found': await self.summarize_issues(context),
            'performance_analysis': await self.analyze_performance(context),
            'recommendations': await self.generate_recommendations(context)
        }
        
        return report
    
    async def summarize_execution(self, context: DebugContext) -> Dict[str, Any]:
        """总结执行情况"""
        execution_history = context.execution_history
        
        return {
            'total_lines_executed': len(execution_history),
            'errors_encountered': sum(1 for h in execution_history if h['type'] == 'error'),
            'breakpoints_hit': sum(1 for h in execution_history if h.get('breakpoint')),
            'execution_path': [h['line'] for h in execution_history if h['type'] == 'execution']
        }
    
    async def summarize_issues(self, context: DebugContext) -> Dict[str, Any]:
        """总结发现的问题"""
        # 从分析结果中提取问题
        analysis = context.variables.get('__analysis__', {})
        potential_issues = analysis.get('potential_issues', [])
        
        # 按严重程度分组
        issues_by_severity = defaultdict(list)
        for issue in potential_issues:
            issues_by_severity[issue.severity].append(issue)
            
        return {
            'total_issues': len(potential_issues),
            'by_severity': {
                severity: len(issues) 
                for severity, issues in issues_by_severity.items()
            },
            'top_issues': potential_issues[:5]
        }
    
    async def analyze_performance(self, context: DebugContext) -> Dict[str, Any]:
        """分析性能"""
        performance_data = context.performance_data
        
        return {
            'execution_time': performance_data.get('total_time', 0),
            'memory_usage': performance_data.get('peak_memory', 0),
            'hotspots': performance_data.get('hotspots', []),
            'bottlenecks': performance_data.get('bottlenecks', [])
        }
    
    async def generate_recommendations(self, context: DebugContext) -> List[str]:
        """生成建议"""
        recommendations = []
        
        analysis = context.variables.get('__analysis__', {})
        
        # 基于复杂度的建议
        complexity = analysis.get('complexity', {})
        if complexity.get('cyclomatic_complexity', 0) > 10:
            recommendations.append("Consider refactoring to reduce code complexity")
            
        # 基于问题的建议
        issues = analysis.get('potential_issues', [])
        if any(issue.severity == 'critical' for issue in issues):
            recommendations.append("Address critical issues before deployment")
            
        # 基于性能的建议
        if context.performance_data.get('execution_time', 0) > 1.0:
            recommendations.append("Consider optimizing performance-critical sections")
            
        return recommendations

# ==================== 集成测试和演示 ====================

async def demonstrate_intelligent_debugger():
    """演示智能调试器功能"""
    
    # 创建调试器实例
    debugger = IntelligentDebugger()
    
    # 示例代码（包含一些问题）
    sample_code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    average = total / len(numbers)  # 潜在的除零错误
    return average

def process_data(data):
    result = []
    for item in data:
        if item > threshold:  # threshold未定义
            result.append(item * 2)
    return result

# 未使用的变量
unused_var = 42

# 性能问题
def inefficient_function(n):
    result = ""
    for i in range(n):
        result += str(i)  # 字符串连接效率低
    return result
'''
    
    # 创建调试会话
    session_id = await debugger.create_debug_session(sample_code)
    print(f"Debug session created: {session_id}")
    
    # 分析代码
    analysis = await debugger.analyze_code(sample_code)
    print("\nCode Analysis:")
    print(f"- Syntax valid: {analysis['syntax_check']['valid']}")
    print(f"- Complexity: {analysis['complexity']}")
    print(f"- Potential issues: {len(analysis['potential_issues'])}")
    
    # 执行调试
    debug_result = await debugger.debug_execution(
        session_id,
        {'numbers': [1, 2, 3, 4, 5], 'data': [1, 2, 3, 4, 5]}
    )
    
    print("\nDebug Execution:")
    print(f"- Success: {debug_result['success']}")
    if not debug_result['success']:
        print(f"- Error: {debug_result['error']}")
        print(f"- Error analysis: {debug_result['error_analysis']}")
    
    # 生成诊断报告
    report_generator = DiagnosticReportGenerator()
    report = await report_generator.generate_report(session_id, debugger)
    
    print("\nDiagnostic Report:")
    print(f"- Total issues found: {report['issues_found']['total_issues']}")
    print(f"- Recommendations: {report['recommendations']}")

if __name__ == "__main__":
    # 运行演示
    asyncio.run(demonstrate_intelligent_debugger())