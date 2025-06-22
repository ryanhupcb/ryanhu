# CodeAgent核心组件实现

## 1. 代码分析引擎 (core/code_analyzer.py)
```python
import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import autopep8
import pylint.lint
from pylint.reporters.text import TextReporter
import io
import difflib
from abc import ABC, abstractmethod

@dataclass
class CodeIssue:
    """代码问题定义"""
    severity: str  # 'error', 'warning', 'info'
    line: int
    column: int
    message: str
    rule: str
    suggestion: Optional[str] = None

@dataclass
class CodeMetrics:
    """代码度量"""
    lines_of_code: int
    cyclomatic_complexity: int
    maintainability_index: float
    code_duplication: float
    test_coverage: float = 0.0

class LanguageAnalyzer(ABC):
    """语言分析器基类"""
    
    @abstractmethod
    async def analyze(self, code: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def validate(self, code: str) -> Dict[str, Any]:
        pass

class PythonAnalyzer(LanguageAnalyzer):
    """Python代码分析器"""
    
    async def analyze(self, code: str) -> Dict[str, Any]:
        """深度分析Python代码"""
        try:
            # AST分析
            tree = ast.parse(code)
            
            # 提取信息
            analysis = {
                "structure": self._analyze_structure(tree),
                "complexity": self._calculate_complexity(tree),
                "dependencies": self._extract_dependencies(tree),
                "functions": self._analyze_functions(tree),
                "classes": self._analyze_classes(tree),
                "issues": await self._run_linter(code),
                "metrics": self._calculate_metrics(code, tree),
                "security": self._security_analysis(tree)
            }
            
            return analysis
            
        except SyntaxError as e:
            return {
                "error": f"语法错误: 第{e.lineno}行 - {e.msg}",
                "valid": False
            }
            
    def _analyze_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """分析代码结构"""
        structure = {
            "imports": [],
            "functions": [],
            "classes": [],
            "global_vars": [],
            "decorators": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    structure["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    structure["imports"].append(f"{module}.{alias.name}")
            elif isinstance(node, ast.FunctionDef):
                structure["functions"].append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                    "lineno": node.lineno
                })
            elif isinstance(node, ast.ClassDef):
                structure["classes"].append({
                    "name": node.name,
                    "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                    "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    "lineno": node.lineno
                })
                
        return structure
        
    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """计算圈复杂度"""
        complexity = {"total": 0, "functions": {}}
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1
                self.current_function = None
                
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                old_complexity = self.complexity
                self.complexity = 1
                
                self.generic_visit(node)
                
                complexity["functions"][node.name] = self.complexity
                self.complexity = old_complexity
                self.current_function = old_function
                
            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        complexity["total"] = sum(complexity["functions"].values()) if complexity["functions"] else visitor.complexity
        
        return complexity
        
    async def _run_linter(self, code: str) -> List[CodeIssue]:
        """运行代码检查器"""
        issues = []
        
        # 使用pylint
        pylint_output = io.StringIO()
        reporter = TextReporter(pylint_output)
        
        # 创建临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_filename = f.name
            
        try:
            # 运行pylint
            pylint.lint.Run(
                [temp_filename, '--disable=all', '--enable=E,W'],
                reporter=reporter,
                exit=False
            )
            
            # 解析输出
            output = pylint_output.getvalue()
            for line in output.split('\n'):
                match = re.match(r'(.+):(\d+):(\d+): ([EWR]\d+): (.+)', line)
                if match:
                    issues.append(CodeIssue(
                        severity='error' if match.group(4).startswith('E') else 'warning',
                        line=int(match.group(2)),
                        column=int(match.group(3)),
                        message=match.group(5),
                        rule=match.group(4)
                    ))
                    
        finally:
            import os
            os.unlink(temp_filename)
            
        return issues
        
    def _calculate_metrics(self, code: str, tree: ast.AST) -> CodeMetrics:
        """计算代码度量"""
        lines = code.split('\n')
        loc = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        
        # 计算复杂度
        complexity = self._calculate_complexity(tree)
        avg_complexity = complexity["total"] / max(len(complexity["functions"]), 1)
        
        # 计算可维护性指数 (简化版本)
        maintainability = max(0, 171 - 5.2 * np.log(loc) - 0.23 * avg_complexity)
        
        return CodeMetrics(
            lines_of_code=loc,
            cyclomatic_complexity=complexity["total"],
            maintainability_index=maintainability / 171 * 100,
            code_duplication=self._calculate_duplication(code)
        )
        
    def _calculate_duplication(self, code: str) -> float:
        """计算代码重复率"""
        lines = [l.strip() for l in code.split('\n') if l.strip()]
        if len(lines) < 2:
            return 0.0
            
        # 简单的重复检测
        seen_lines = set()
        duplicate_lines = 0
        
        for line in lines:
            if line in seen_lines and len(line) > 10:  # 忽略短行
                duplicate_lines += 1
            seen_lines.add(line)
            
        return duplicate_lines / len(lines) * 100
        
    def _security_analysis(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """安全性分析"""
        security_issues = []
        
        dangerous_functions = {
            'eval': '避免使用eval()，可能导致代码注入',
            'exec': '避免使用exec()，可能导致代码注入',
            '__import__': '动态导入可能带来安全风险',
            'compile': '动态编译可能带来安全风险',
            'open': '检查文件操作的路径验证',
            'subprocess': '子进程调用需要谨慎处理输入'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in dangerous_functions:
                    security_issues.append({
                        'function': node.func.id,
                        'line': node.lineno,
                        'warning': dangerous_functions[node.func.id]
                    })
                    
        return security_issues
        
    async def validate(self, code: str) -> Dict[str, Any]:
        """验证代码"""
        try:
            ast.parse(code)
            issues = await self._run_linter(code)
            
            return {
                "is_valid": len([i for i in issues if i.severity == 'error']) == 0,
                "errors": [i for i in issues if i.severity == 'error'],
                "warnings": [i for i in issues if i.severity == 'warning']
            }
            
        except SyntaxError as e:
            return {
                "is_valid": False,
                "errors": [{
                    "line": e.lineno,
                    "message": e.msg
                }],
                "warnings": []
            }

class CodeAnalyzer:
    """统一的代码分析器"""
    
    def __init__(self):
        self.analyzers = {
            'python': PythonAnalyzer(),
            'javascript': JavaScriptAnalyzer(),
            'java': JavaAnalyzer(),
            'cpp': CppAnalyzer(),
            'go': GoAnalyzer()
        }
        
    async def analyze(self, code: str, language: str) -> Dict[str, Any]:
        """分析代码"""
        analyzer = self.analyzers.get(language.lower())
        if not analyzer:
            return {"error": f"不支持的语言: {language}"}
            
        return await analyzer.analyze(code)
        
    async def quick_analyze(self, code: str, language: str) -> str:
        """快速分析摘要"""
        full_analysis = await self.analyze(code, language)
        
        if 'error' in full_analysis:
            return full_analysis['error']
            
        # 生成摘要
        summary = []
        
        if 'structure' in full_analysis:
            structure = full_analysis['structure']
            summary.append(f"包含 {len(structure['functions'])} 个函数, {len(structure['classes'])} 个类")
            
        if 'metrics' in full_analysis:
            metrics = full_analysis['metrics']
            summary.append(f"代码行数: {metrics.lines_of_code}")
            summary.append(f"复杂度: {metrics.cyclomatic_complexity}")
            
        if 'issues' in full_analysis:
            issues = full_analysis['issues']
            errors = len([i for i in issues if i.severity == 'error'])
            warnings = len([i for i in issues if i.severity == 'warning'])
            summary.append(f"发现 {errors} 个错误, {warnings} 个警告")
            
        return '\n'.join(summary)
        
    async def validate_code(self, code: str, language: str) -> Dict[str, Any]:
        """验证代码"""
        analyzer = self.analyzers.get(language.lower())
        if not analyzer:
            return {"is_valid": False, "error": f"不支持的语言: {language}"}
            
        return await analyzer.validate(code)
        
    def calculate_diff(self, old_code: str, new_code: str) -> str:
        """计算代码差异"""
        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile='原始代码',
            tofile='修改后代码',
            lineterm=''
        )
        
        return ''.join(diff)
        
    def summarize_changes(self, diff: str) -> str:
        """总结变更"""
        lines = diff.split('\n')
        
        added = sum(1 for line in lines if line.startswith('+') and not line.startswith('+++'))
        removed = sum(1 for line in lines if line.startswith('-') and not line.startswith('---'))
        
        summary = f"添加了 {added} 行，删除了 {removed} 行"
        
        # 分析具体变更类型
        changes = []
        
        for line in lines:
            if 'import' in line and (line.startswith('+') or line.startswith('-')):
                changes.append("导入语句变更")
                break
                
        for line in lines:
            if 'def ' in line and line.startswith('+'):
                changes.append("新增函数")
                break
                
        for line in lines:
            if 'class ' in line and line.startswith('+'):
                changes.append("新增类")
                break
                
        if changes:
            summary += "\n主要变更: " + ", ".join(set(changes))
            
        return summary
        
    async def security_check(self, code: str, language: str) -> Dict[str, Any]:
        """安全检查"""
        # 基础安全检查
        dangerous_patterns = {
            'python': [
                (r'eval\s*\(', '检测到eval()调用'),
                (r'exec\s*\(', '检测到exec()调用'),
                (r'__import__\s*\(', '检测到动态导入'),
                (r'subprocess\.(call|run|Popen)', '检测到子进程调用'),
                (r'os\.(system|popen)', '检测到系统命令执行')
            ],
            'javascript': [
                (r'eval\s*\(', '检测到eval()调用'),
                (r'Function\s*\(', '检测到Function构造器'),
                (r'innerHTML\s*=', '可能的XSS风险'),
                (r'document\.write', '检测到document.write')
            ]
        }
        
        patterns = dangerous_patterns.get(language.lower(), [])
        issues = []
        
        for pattern, message in patterns:
            if re.search(pattern, code):
                issues.append(message)
                
        return {
            'is_safe': len(issues) == 0,
            'reason': '\n'.join(issues) if issues else '代码通过安全检查'
        }
        
    def parse_analysis(self, response: str) -> Dict[str, Any]:
        """解析分析响应"""
        # 解析AI返回的分析结果
        sections = {
            'structure': '',
            'issues': [],
            'suggestions': [],
            'quality_score': 0
        }
        
        current_section = None
        
        for line in response.split('\n'):
            if '代码结构分析' in line:
                current_section = 'structure'
            elif '潜在问题' in line:
                current_section = 'issues'
            elif '改进建议' in line or '优化建议' in line:
                current_section = 'suggestions'
            elif '评分' in line or '得分' in line:
                # 提取分数
                match = re.search(r'(\d+)/10', line)
                if match:
                    sections['quality_score'] = int(match.group(1))
            elif current_section and line.strip():
                if current_section == 'structure':
                    sections['structure'] += line + '\n'
                elif current_section in ['issues', 'suggestions']:
                    if line.strip().startswith(('-', '*', '•', '1', '2', '3')):
                        sections[current_section].append(line.strip().lstrip('-*•123456789. '))
                        
        return sections

import numpy as np  # 添加numpy导入用于计算

# 其他语言分析器的占位实现
class JavaScriptAnalyzer(LanguageAnalyzer):
    async def analyze(self, code: str) -> Dict[str, Any]:
        # JavaScript分析实现
        return {"language": "javascript", "analysis": "待实现"}
        
    async def validate(self, code: str) -> Dict[str, Any]:
        return {"is_valid": True, "errors": [], "warnings": []}

class JavaAnalyzer(LanguageAnalyzer):
    async def analyze(self, code: str) -> Dict[str, Any]:
        # Java分析实现
        return {"language": "java", "analysis": "待实现"}
        
    async def validate(self, code: str) -> Dict[str, Any]:
        return {"is_valid": True, "errors": [], "warnings": []}

class CppAnalyzer(LanguageAnalyzer):
    async def analyze(self, code: str) -> Dict[str, Any]:
        # C++分析实现
        return {"language": "cpp", "analysis": "待实现"}
        
    async def validate(self, code: str) -> Dict[str, Any]:
        return {"is_valid": True, "errors": [], "warnings": []}

class GoAnalyzer(LanguageAnalyzer):
    async def analyze(self, code: str) -> Dict[str, Any]:
        # Go分析实现
        return {"language": "go", "analysis": "待实现"}
        
    async def validate(self, code: str) -> Dict[str, Any]:
        return {"is_valid": True, "errors": [], "warnings": []}
```

## 2. 代码生成引擎 (core/code_generator.py)
```python
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import jinja2
from abc import ABC, abstractmethod

@dataclass
class CodeTemplate:
    """代码模板"""
    name: str
    language: str
    template: str
    variables: List[str]
    description: str

class LanguageGenerator(ABC):
    """语言生成器基类"""
    
    @abstractmethod
    def generate_boilerplate(self, project_type: str) -> str:
        pass
    
    @abstractmethod
    def generate_function(self, spec: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def generate_class(self, spec: Dict[str, Any]) -> str:
        pass

class PythonGenerator(LanguageGenerator):
    """Python代码生成器"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.jinja_env = jinja2.Environment(
            loader=jinja2.DictLoader(self.templates)
        )
        
    def _load_templates(self) -> Dict[str, str]:
        """加载代码模板"""
        return {
            'function': '''def {{ name }}({{ params }}):
    """{{ docstring }}"""
    {% for line in body %}
    {{ line }}
    {% endfor %}
''',
            'class': '''class {{ name }}{% if base %}({{ base }}){% endif %}:
    """{{ docstring }}"""
    
    def __init__(self{{ init_params }}):
        {% for line in init_body %}
        {{ line }}
        {% endfor %}
    
    {% for method in methods %}
    {{ method }}
    {% endfor %}
''',
            'async_function': '''async def {{ name }}({{ params }}):
    """{{ docstring }}"""
    {% for line in body %}
    {{ line }}
    {% endfor %}
''',
            'test_function': '''def test_{{ name }}():
    """Test {{ name }} function"""
    # Arrange
    {% for line in arrange %}
    {{ line }}
    {% endfor %}
    
    # Act
    {% for line in act %}
    {{ line }}
    {% endfor %}
    
    # Assert
    {% for line in assert %}
    {{ line }}
    {% endfor %}
'''
        }
        
    def generate_boilerplate(self, project_type: str) -> str:
        """生成项目模板代码"""
        boilerplates = {
            'cli': self._generate_cli_boilerplate(),
            'web': self._generate_web_boilerplate(),
            'api': self._generate_api_boilerplate(),
            'library': self._generate_library_boilerplate(),
            'ml': self._generate_ml_boilerplate()
        }
        
        return boilerplates.get(project_type, self._generate_basic_boilerplate())
        
    def _generate_cli_boilerplate(self) -> str:
        """生成CLI应用模板"""
        return '''#!/usr/bin/env python3
"""
CLI Application Template
"""

import argparse
import sys
import logging
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> int:
    """主函数"""
    logger.info(f"Starting with args: {args}")
    
    try:
        # 在这里实现主要逻辑
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            
        logger.info("Processing...")
        
        # TODO: 实现功能
        
        logger.info("Completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CLI应用描述",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="启用详细输出"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.txt",
        help="输出文件路径"
    )
    
    # TODO: 添加更多参数
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
'''

    def _generate_api_boilerplate(self) -> str:
        """生成API应用模板"""
        return '''"""
RESTful API Application
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="API应用",
    description="API应用描述",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 数据模型
class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float
    

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    price: float
    created_at: str


# 模拟数据库
items_db = []


@app.get("/")
async def root():
    """根路径"""
    return {"message": "Welcome to the API"}


@app.get("/items", response_model=List[ItemResponse])
async def get_items(skip: int = 0, limit: int = 10):
    """获取项目列表"""
    return items_db[skip : skip + limit]


@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    """获取单个项目"""
    for item in items_db:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")


@app.post("/items", response_model=ItemResponse)
async def create_item(item: Item):
    """创建新项目"""
    new_item = {
        "id": len(items_db) + 1,
        "name": item.name,
        "description": item.description,
        "price": item.price,
        "created_at": "2025-01-01T00:00:00"
    }
    items_db.append(new_item)
    return new_item


@app.put("/items/{item_id}", response_model=ItemResponse)
async def update_item(item_id: int, item: Item):
    """更新项目"""
    for idx, db_item in enumerate(items_db):
        if db_item["id"] == item_id:
            items_db[idx].update(item.dict(exclude_unset=True))
            return items_db[idx]
    raise HTTPException(status_code=404, detail="Item not found")


@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """删除项目"""
    for idx, item in enumerate(items_db):
        if item["id"] == item_id:
            del items_db[idx]
            return {"message": "Item deleted"}
    raise HTTPException(status_code=404, detail="Item not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''

    def generate_function(self, spec: Dict[str, Any]) -> str:
        """生成函数代码"""
        template = self.jinja_env.get_template(
            'async_function' if spec.get('async', False) else 'function'
        )
        
        # 处理参数
        params = []
        for param in spec.get('parameters', []):
            param_str = param['name']
            if 'type' in param:
                param_str += f": {param['type']}"
            if 'default' in param:
                param_str += f" = {param['default']}"
            params.append(param_str)
            
        # 生成函数体
        body = self._generate_function_body(spec)
        
        return template.render(
            name=spec['name'],
            params=', '.join(params),
            docstring=spec.get('description', ''),
            body=body
        )
        
    def _generate_function_body(self, spec: Dict[str, Any]) -> List[str]:
        """生成函数体"""
        body = []
        
        # 参数验证
        if spec.get('validate_params', True):
            for param in spec.get('parameters', []):
                if param.get('required', False):
                    body.append(f"if {param['name']} is None:")
                    body.append(f"    raise ValueError('{param['name']} is required')")
                    
        # 主要逻辑
        if 'logic' in spec:
            body.extend(spec['logic'])
        else:
            body.append("# TODO: 实现函数逻辑")
            body.append("pass")
            
        return body
        
    def generate_class(self, spec: Dict[str, Any]) -> str:
        """生成类代码"""
        template = self.jinja_env.get_template('class')
        
        # 处理初始化参数
        init_params = []
        init_body = []
        
        for attr in spec.get('attributes', []):
            param = attr['name']
            if 'type' in attr:
                param += f": {attr['type']}"
            if 'default' in attr:
                param += f" = {attr['default']}"
            init_params.append(param)
            
            init_body.append(f"self.{attr['name']} = {attr['name']}")
            
        # 生成方法
        methods = []
        for method_spec in spec.get('methods', []):
            method_code = self.generate_function(method_spec)
            # 缩进方法代码
            method_lines = method_code.split('\n')
            method_lines = ['    ' + line if line else line for line in method_lines]
            methods.append('\n'.join(method_lines))
            
        return template.render(
            name=spec['name'],
            base=spec.get('base', ''),
            docstring=spec.get('description', ''),
            init_params=', ' + ', '.join(init_params) if init_params else '',
            init_body=init_body,
            methods=methods
        )

class CodeGenerator:
    """统一的代码生成器"""
    
    def __init__(self):
        self.generators = {
            'python': PythonGenerator(),
            'javascript': JavaScriptGenerator(),
            'java': JavaGenerator(),
            'cpp': CppGenerator(),
            'go': GoGenerator()
        }
        
        self.prompt_templates = self._load_prompt_templates()
        
    def _load_prompt_templates(self) -> Dict[str, str]:
        """加载提示模板"""
        return {
            'generation': '''请根据以下需求生成{language}代码：

需求：
{requirements}

上下文信息：
{context}

相关代码参考：
{memory_context}

要求：
1. 代码应该清晰、高效且易于维护
2. 包含适当的错误处理
3. 添加必要的注释和文档
4. 遵循{language}的最佳实践
5. 考虑性能和安全性

请生成完整的代码实现。''',

            'optimization': '''请优化以下{language}代码：

原始代码：
```{language}
{code}
```

优化目标：
- 提高性能
- 改善可读性
- 减少复杂度
- 修复潜在问题

请提供优化后的代码，并解释主要改进。''',

            'refactoring': '''请重构以下{language}代码：

原始代码：
```{language}
{code}
```

重构要求：
{requirements}

请提供重构后的代码。'''
        }
        
    def build_generation_prompt(
        self,
        requirements: str,
        language: str,
        context: Dict[str, Any],
        memory_context: List[Dict[str, Any]]
    ) -> str:
        """构建生成提示"""
        template = self.prompt_templates['generation']
        
        # 格式化内存上下文
        memory_str = ""
        for mem in memory_context[:5]:  # 限制数量
            memory_str += f"\n---\n{mem.get('code', '')}\n"
            
        return template.format(
            language=language,
            requirements=requirements,
            context=self._format_context(context),
            memory_context=memory_str
        )
        
    def _format_context(self, context: Dict[str, Any]) -> str:
        """格式化上下文"""
        parts = []
        
        if 'project_type' in context:
            parts.append(f"项目类型: {context['project_type']}")
            
        if 'dependencies' in context:
            parts.append(f"依赖: {', '.join(context['dependencies'])}")
            
        if 'constraints' in context:
            parts.append(f"约束: {context['constraints']}")
            
        return '\n'.join(parts)
        
    def post_process(self, response: str, language: str) -> str:
        """后处理生成的代码"""
        # 提取代码块
        code = self.extract_code_from_response(response, language)
        
        # 格式化代码
        generator = self.generators.get(language.lower())
        if generator and hasattr(generator, 'format_code'):
            code = generator.format_code(code)
            
        return code
        
    def extract_code_from_response(self, response: str, language: str) -> str:
        """从响应中提取代码"""
        # 查找代码块
        code_pattern = rf'```{language}?\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
            
        # 如果没有代码块标记，尝试提取看起来像代码的部分
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # 简单的启发式方法检测代码
            if any(keyword in line for keyword in ['def ', 'class ', 'function ', 'import ', 'const ', 'var ']):
                in_code = True
                
            if in_code:
                code_lines.append(line)
                
            # 检测代码结束
            if in_code and line.strip() == '' and len(code_lines) > 5:
                # 可能是代码块结束
                break
                
        return '\n'.join(code_lines).strip()
        
    def generate_tests(self, code: str, language: str) -> str:
        """为代码生成测试"""
        generator = self.generators.get(language.lower())
        if generator and hasattr(generator, 'generate_tests'):
            return generator.generate_tests(code)
            
        return "# 测试生成功能正在开发中"
        
    def generate_documentation(self, code: str, language: str) -> str:
        """生成文档"""
        generator = self.generators.get(language.lower())
        if generator and hasattr(generator, 'generate_docs'):
            return generator.generate_docs(code)
            
        return "# 文档生成功能正在开发中"

# 其他语言生成器的占位实现
class JavaScriptGenerator(LanguageGenerator):
    def generate_boilerplate(self, project_type: str) -> str:
        # JavaScript项目模板
        return "// JavaScript boilerplate"
        
    def generate_function(self, spec: Dict[str, Any]) -> str:
        return "// JavaScript function"
        
    def generate_class(self, spec: Dict[str, Any]) -> str:
        return "// JavaScript class"

class JavaGenerator(LanguageGenerator):
    def generate_boilerplate(self, project_type: str) -> str:
        # Java项目模板
        return "// Java boilerplate"
        
    def generate_function(self, spec: Dict[str, Any]) -> str:
        return "// Java method"
        
    def generate_class(self, spec: Dict[str, Any]) -> str:
        return "// Java class"

class CppGenerator(LanguageGenerator):
    def generate_boilerplate(self, project_type: str) -> str:
        # C++项目模板
        return "// C++ boilerplate"
        
    def generate_function(self, spec: Dict[str, Any]) -> str:
        return "// C++ function"
        
    def generate_class(self, spec: Dict[str, Any]) -> str:
        return "// C++ class"

class GoGenerator(LanguageGenerator):
    def generate_boilerplate(self, project_type: str) -> str:
        # Go项目模板
        return "// Go boilerplate"
        
    def generate_function(self, spec: Dict[str, Any]) -> str:
        return "// Go function"
        
    def generate_class(self, spec: Dict[str, Any]) -> str:
        return "// Go struct"
```

## 3. 系统控制器 (core/system_controller.py)
```python
import os
import subprocess
import asyncio
import psutil
import tempfile
import docker
from typing import Dict, Any, Optional, List
from datetime import datetime
import pyautogui
import platform
from pathlib import Path
import json

class SystemController:
    """系统控制器 - 管理代码执行和系统操作"""
    
    def __init__(self):
        self.docker_client = None
        self.sandboxes = {}
        self.process_monitor = ProcessMonitor()
        self.screen_controller = ScreenController()
        self.file_controller = FileController()
        
    async def initialize(self):
        """初始化系统控制器"""
        # 初始化Docker客户端
        try:
            self.docker_client = docker.from_env()
        except:
            print("Docker未安装或未运行，沙箱功能将不可用")
            
        # 检查系统权限
        self._check_permissions()
        
    def _check_permissions(self):
        """检查系统权限"""
        self.permissions = {
            'execute': True,
            'file_access': True,
            'screen_capture': True,
            'input_control': True,
            'process_control': os.name != 'nt' or platform.system() == 'Windows'
        }
        
    async def execute_code(
        self,
        code: str,
        language: str,
        timeout: int = 30,
        memory_limit: int = 512  # MB
    ) -> Dict[str, Any]:
        """直接执行代码"""
        
        executors = {
            'python': self._execute_python,
            'javascript': self._execute_javascript,
            'java': self._execute_java,
            'cpp': self._execute_cpp,
            'go': self._execute_go
        }
        
        executor = executors.get(language.lower())
        if not executor:
            return {
                'success': False,
                'error': f'不支持的语言: {language}'
            }
            
        return await executor(code, timeout, memory_limit)
        
    async def _execute_python(
        self,
        code: str,
        timeout: int,
        memory_limit: int
    ) -> Dict[str, Any]:
        """执行Python代码"""
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name
            
        try:
            # 构建命令
            cmd = [sys.executable, temp_file]
            
            # 执行代码
            start_time = datetime.now()
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 监控内存使用
            monitor_task = asyncio.create_task(
                self._monitor_process(process.pid, memory_limit)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    'success': process.returncode == 0,
                    'output': stdout.decode('utf-8'),
                    'error': stderr.decode('utf-8'),
                    'execution_time': execution_time,
                    'exit_code': process.returncode
                }
                
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                
                return {
                    'success': False,
                    'error': f'执行超时 ({timeout}秒)',
                    'execution_time': timeout
                }
                
            finally:
                monitor_task.cancel()
                
        finally:
            # 清理临时文件
            os.unlink(temp_file)
            
    async def _monitor_process(self, pid: int, memory_limit: int):
        """监控进程内存使用"""
        try:
            process = psutil.Process(pid)
            
            while True:
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > memory_limit:
                    process.terminate()
                    raise MemoryError(f"内存使用超限: {memory_mb}MB > {memory_limit}MB")
                    
                await asyncio.sleep(0.1)
                
        except psutil.NoSuchProcess:
            pass
            
    async def execute_in_sandbox(
        self,
        code: str,
        language: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """在沙箱中执行代码"""
        
        if not self.docker_client:
            return {
                'success': False,
                'error': 'Docker未安装，无法使用沙箱功能'
            }
            
        # 选择合适的Docker镜像
        images = {
            'python': 'python:3.11-slim',
            'javascript': 'node:18-slim',
            'java': 'openjdk:17-slim',
            'cpp': 'gcc:latest',
            'go': 'golang:1.20-alpine'
        }
        
        image = images.get(language.lower())
        if not image:
            return {
                'success': False,
                'error': f'不支持的语言: {language}'
            }
            
        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 写入代码文件
                code_file = os.path.join(temp_dir, f'main.{self._get_extension(language)}')
                with open(code_file, 'w') as f:
                    f.write(code)
                    
                # 构建Docker命令
                commands = {
                    'python': f'python /code/main.py',
                    'javascript': f'node /code/main.js',
                    'java': 'cd /code && javac main.java && java Main',
                    'cpp': 'cd /code && g++ main.cpp -o main && ./main',
                    'go': 'cd /code && go run main.go'
                }
                
                command = commands.get(language.lower())
                
                # 运行容器
                container = self.docker_client.containers.run(
                    image,
                    command=command,
                    volumes={temp_dir: {'bind': '/code', 'mode': 'rw'}},
                    working_dir='/code',
                    mem_limit='512m',
                    cpu_quota=50000,  # 50% CPU
                    detach=True,
                    remove=False
                )
                
                # 等待执行完成
                start_time = datetime.now()
                
                try:
                    exit_code = container.wait(timeout=timeout)['StatusCode']
                    logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return {
                        'success': exit_code == 0,
                        'output': logs,
                        'error': '' if exit_code == 0 else logs,
                        'execution_time': execution_time,
                        'exit_code': exit_code
                    }
                    
                except Exception as e:
                    container.stop()
                    return {
                        'success': False,
                        'error': f'执行错误: {str(e)}',
                        'execution_time': (datetime.now() - start_time).total_seconds()
                    }
                    
                finally:
                    # 清理容器
                    try:
                        container.remove()
                    except:
                        pass
                        
        except Exception as e:
            return {
                'success': False,
                'error': f'沙箱错误: {str(e)}'
            }
            
    def _get_extension(self, language: str) -> str:
        """获取文件扩展名"""
        extensions = {
            'python': 'py',
            'javascript': 'js',
            'java': 'java',
            'cpp': 'cpp',
            'go': 'go'
        }
        return extensions.get(language.lower(), 'txt')
        
    async def capture_screen(self) -> bytes:
        """捕获屏幕"""
        return await self.screen_controller.capture()
        
    async def simulate_input(self, actions: List[Dict[str, Any]]):
        """模拟用户输入"""
        for action in actions:
            action_type = action.get('type')
            
            if action_type == 'click':
                await self.screen_controller.click(
                    action['x'],
                    action['y'],
                    action.get('button', 'left')
                )
            elif action_type == 'type':
                await self.screen_controller.type_text(action['text'])
            elif action_type == 'key':
                await self.screen_controller.press_key(action['key'])
            elif action_type == 'wait':
                await asyncio.sleep(action.get('duration', 1))

class ProcessMonitor:
    """进程监控器"""
    
    def __init__(self):
        self.monitored_processes = {}
        
    def start_monitoring(self, pid: int, limits: Dict[str, Any]):
        """开始监控进程"""
        self.monitored_processes[pid] = {
            'limits': limits,
            'start_time': datetime.now(),
            'stats': []
        }
        
    def get_process_stats(self, pid: int) -> Dict[str, Any]:
        """获取进程统计信息"""
        try:
            process = psutil.Process(pid)
            
            return {
                'cpu_percent': process.cpu_percent(interval=0.1),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'num_threads': process.num_threads(),
                'status': process.status(),
                'create_time': datetime.fromtimestamp(process.create_time())
            }
        except psutil.NoSuchProcess:
            return None

class ScreenController:
    """屏幕控制器"""
    
    def __init__(self):
        # 设置pyautogui安全模式
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
    async def capture(self) -> bytes:
        """捕获屏幕截图"""
        screenshot = pyautogui.screenshot()
        
        # 转换为字节
        import io
        buffer = io.BytesIO()
        screenshot.save(buffer, format='PNG')
        return buffer.getvalue()
        
    async def click(self, x: int, y: int, button: str = 'left'):
        """模拟鼠标点击"""
        pyautogui.click(x, y, button=button)
        
    async def type_text(self, text: str):
        """模拟键盘输入"""
        pyautogui.typewrite(text)
        
    async def press_key(self, key: str):
        """模拟按键"""
        pyautogui.press(key)
        
    def get_screen_size(self) -> tuple:
        """获取屏幕尺寸"""
        return pyautogui.size()
        
    def locate_image(self, image_path: str) -> Optional[tuple]:
        """在屏幕上定位图像"""
        try:
            location = pyautogui.locateOnScreen(image_path)
            if location:
                return pyautogui.center(location)
        except:
            pass
        return None

class FileController:
    """文件控制器"""
    
    def __init__(self):
        self.workspace = Path.home() / '.codeagent' / 'workspace'
        self.workspace.mkdir(parents=True, exist_ok=True)
        
    async def create_project(self, name: str, template: str) -> Path:
        """创建项目"""
        project_path = self.workspace / name
        project_path.mkdir(exist_ok=True)
        
        # TODO: 根据模板创建项目结构
        
        return project_path
        
    async def read_file(self, path: Path) -> str:
        """读取文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
            
    async def write_file(self, path: Path, content: str):
        """写入文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    async def list_files(self, directory: Path, pattern: str = '*') -> List[Path]:
        """列出文件"""
        return list(directory.glob(pattern))
        
    def watch_directory(self, directory: Path, callback: callable):
        """监视目录变化"""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                callback(event)
                
        observer = Observer()
        observer.schedule(Handler(), str(directory), recursive=True)
        observer.start()
        
        return observer

import sys  # 添加sys导入
```

## 4. LangChain集成模块 (langchain_modules/chains.py)
```python
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import BaseOutputParser
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import json

class CodeAnalysisOutput(BaseModel):
    """代码分析输出模型"""
    summary: str = Field(description="代码摘要")
    complexity: int = Field(description="复杂度评分")
    issues: List[str] = Field(description="发现的问题")
    suggestions: List[str] = Field(description="改进建议")
    security_risks: List[str] = Field(description="安全风险")

class CodeGenerationOutput(BaseModel):
    """代码生成输出模型"""
    code: str = Field(description="生成的代码")
    explanation: str = Field(description="代码解释")
    dependencies: List[str] = Field(description="依赖列表")
    usage_example: str = Field(description="使用示例")

class CodeProcessingChain:
    """代码处理链"""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 输出解析器
        self.analysis_parser = PydanticOutputParser(
            pydantic_object=CodeAnalysisOutput
        )
        self.generation_parser = PydanticOutputParser(
            pydantic_object=CodeGenerationOutput
        )
        
    def create_analysis_chain(self) -> LLMChain:
        """创建代码分析链"""
        
        # 分析提示模板
        analysis_prompt = PromptTemplate(
            input_variables=["code", "language"],
            template="""分析以下{language}代码：

{code}

请提供详细分析，包括：
1. 代码功能摘要
2. 复杂度评估（1-10分）
3. 潜在问题列表
4. 改进建议
5. 安全风险评估

{format_instructions}
""",
            partial_variables={
                "format_instructions": self.analysis_parser.get_format_instructions()
            }
        )
        
        # 创建分析链
        analysis_chain = LLMChain(
            prompt=analysis_prompt,
            output_parser=self.analysis_parser,
            memory=self.memory,
            verbose=True
        )
        
        return analysis_chain
        
    def create_generation_chain(self) -> SequentialChain:
        """创建代码生成链"""
        
        # 第一步：理解需求
        understand_prompt = PromptTemplate(
            input_variables=["requirements", "context"],
            template="""理解以下编程需求：

需求：{requirements}

上下文：{context}

请提取关键信息：
1. 主要功能点
2. 技术约束
3. 性能要求
4. 预期输入输出

输出格式：JSON
"""
        )
        
        understand_chain = LLMChain(
            prompt=understand_prompt,
            output_key="understanding"
        )
        
        # 第二步：设计架构
        design_prompt = PromptTemplate(
            input_variables=["understanding"],
            template="""基于以下理解设计代码架构：

{understanding}

请设计：
1. 整体架构
2. 主要组件
3. 数据流
4. 接口定义

输出格式：结构化文本
"""
        )
        
        design_chain = LLMChain(
            prompt=design_prompt,
            output_key="design"
        )
        
        # 第三步：生成代码
        generate_prompt = PromptTemplate(
            input_variables=["understanding", "design", "language"],
            template="""基于以下信息生成{language}代码：

需求理解：
{understanding}

架构设计：
{design}

请生成完整的、可运行的代码实现。

{format_instructions}
""",
            partial_variables={
                "format_instructions": self.generation_parser.get_format_instructions()
            }
        )
        
        generate_chain = LLMChain(
            prompt=generate_prompt,
            output_parser=self.generation_parser,
            output_key="result"
        )
        
        # 组合成顺序链
        sequential_chain = SequentialChain(
            chains=[understand_chain, design_chain, generate_chain],
            input_variables=["requirements", "context", "language"],
            output_variables=["understanding", "design", "result"],
            verbose=True
        )
        
        return sequential_chain
        
    def create_edit_chain(self) -> LLMChain:
        """创建代码编辑链"""
        
        edit_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的代码编辑助手。"),
            ("human", """请根据以下指令编辑代码：

原始代码：
```{language}
{code}
```

编辑指令：
{instructions}

代码分析：
{analysis}

请提供：
1. 编辑后的完整代码
2. 修改说明
3. 潜在影响分析
""")
        ])
        
        edit_chain = LLMChain(
            prompt=edit_prompt,
            memory=self.memory
        )
        
        return edit_chain
        
    def create_refactor_chain(self) -> SequentialChain:
        """创建重构链"""
        
        # 识别重构机会
        identify_prompt = PromptTemplate(
            input_variables=["code", "metrics"],
            template="""分析代码并识别重构机会：

代码：
{code}

代码度量：
{metrics}

识别：
1. 代码异味
2. 重复代码
3. 过长函数
4. 复杂条件
5. 不当耦合
"""
        )
        
        identify_chain = LLMChain(
            prompt=identify_prompt,
            output_key="opportunities"
        )
        
        # 制定重构计划
        plan_prompt = PromptTemplate(
            input_variables=["code", "opportunities"],
            template="""基于识别的问题制定重构计划：

代码：
{code}

重构机会：
{opportunities}

制定详细的重构步骤。
"""
        )
        
        plan_chain = LLMChain(
            prompt=plan_prompt,
            output_key="plan"
        )
        
        # 执行重构
        refactor_prompt = PromptTemplate(
            input_variables=["code", "plan"],
            template="""按照计划重构代码：

原始代码：
{code}

重构计划：
{plan}

生成重构后的代码。
"""
        )
        
        refactor_chain = LLMChain(
            prompt=refactor_prompt,
            output_key="refactored_code"
        )
        
        # 组合链
        return SequentialChain(
            chains=[identify_chain, plan_chain, refactor_chain],
            input_variables=["code", "metrics"],
            output_variables=["opportunities", "plan", "refactored_code"]
        )
        
    def create_test_generation_chain(self) -> LLMChain:
        """创建测试生成链"""
        
        test_prompt = PromptTemplate(
            input_variables=["code", "language", "framework"],
            template="""为以下代码生成单元测试：

代码：
```{language}
{code}
```

测试框架：{framework}

生成全面的测试用例，包括：
1. 正常情况测试
2. 边界条件测试
3. 异常情况测试
4. 性能测试（如适用）
"""
        )
        
        return LLMChain(
            prompt=test_prompt,
            memory=self.memory
        )
        
    def create_documentation_chain(self) -> LLMChain:
        """创建文档生成链"""
        
        doc_prompt = PromptTemplate(
            input_variables=["code", "language", "style"],
            template="""为以下代码生成文档：

代码：
```{language}
{code}
```

文档风格：{style}

生成：
1. 功能概述
2. API文档
3. 使用示例
4. 参数说明
5. 返回值说明
6. 异常说明
"""
        )
        
        return LLMChain(
            prompt=doc_prompt,
            memory=self.memory
        )
        
    def create_optimization_chain(self) -> SequentialChain:
        """创建优化链"""
        
        # 性能分析
        profile_prompt = PromptTemplate(
            input_variables=["code", "language"],
            template="""分析代码性能：

{code}

识别性能瓶颈和优化机会。
"""
        )
        
        profile_chain = LLMChain(
            prompt=profile_prompt,
            output_key="profile"
        )
        
        # 优化建议
        optimize_prompt = PromptTemplate(
            input_variables=["code", "profile"],
            template="""基于性能分析提供优化建议：

代码：
{code}

性能分析：
{profile}

提供具体的优化方案。
"""
        )
        
        optimize_chain = LLMChain(
            prompt=optimize_prompt,
            output_key="optimizations"
        )
        
        # 实施优化
        implement_prompt = PromptTemplate(
            input_variables=["code", "optimizations"],
            template="""实施优化：

原始代码：
{code}

优化建议：
{optimizations}

生成优化后的代码。
"""
        )
        
        implement_chain = LLMChain(
            prompt=implement_prompt,
            output_key="optimized_code"
        )
        
        return SequentialChain(
            chains=[profile_chain, optimize_chain, implement_chain],
            input_variables=["code", "language"],
            output_variables=["profile", "optimizations", "optimized_code"]
        )
        
    def create_debug_chain(self) -> LLMChain:
        """创建调试链"""
        
        debug_prompt = PromptTemplate(
            input_variables=["code", "error", "context"],
            template="""调试以下代码：

代码：
{code}

错误信息：
{error}

执行上下文：
{context}

请：
1. 分析错误原因
2. 提供修复方案
3. 解释如何避免类似错误
"""
        )
        
        return LLMChain(
            prompt=debug_prompt,
            memory=self.memory
        )
        
    def create_code_review_chain(self) -> LLMChain:
        """创建代码审查链"""
        
        review_prompt = PromptTemplate(
            input_variables=["code", "standards", "context"],
            template="""进行代码审查：

代码：
{code}

编码标准：
{standards}

项目上下文：
{context}

审查要点：
1. 代码质量
2. 可读性
3. 可维护性
4. 性能
5. 安全性
6. 最佳实践遵循情况

提供详细的审查报告和改进建议。
"""
        )
        
        return LLMChain(
            prompt=review_prompt,
            memory=self.memory
        )

class CustomOutputParser(BaseOutputParser):
    """自定义输出解析器"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """解析输出"""
        # 尝试提取JSON
        try:
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        # 降级到简单解析
        return {"raw_output": text}
        
    def get_format_instructions(self) -> str:
        """获取格式化指令"""
        return "请以JSON格式输出结果。"

class ChainOrchestrator:
    """链编排器"""
    
    def __init__(self):
        self.chains = {}
        self.processing_chain = CodeProcessingChain()
        
    def register_chain(self, name: str, chain: Any):
        """注册链"""
        self.chains[name] = chain
        
    async def execute_chain(
        self,
        chain_name: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行链"""
        if chain_name not in self.chains:
            raise ValueError(f"未找到链: {chain_name}")
            
        chain = self.chains[chain_name]
        
        # 执行链
        result = await chain.arun(**inputs)
        
        return result
        
    def create_custom_chain(
        self,
        steps: List[Dict[str, Any]]
    ) -> SequentialChain:
        """创建自定义链"""
        chains = []
        
        for step in steps:
            if step['type'] == 'llm':
                chain = LLMChain(
                    prompt=PromptTemplate(
                        input_variables=step['inputs'],
                        template=step['template']
                    ),
                    output_key=step['output']
                )
            elif step['type'] == 'transform':
                chain = TransformChain(
                    input_variables=step['inputs'],
                    output_variables=[step['output']],
                    transform=step['transform_func']
                )
            else:
                raise ValueError(f"未知链类型: {step['type']}")
                
            chains.append(chain)
            
        return SequentialChain(chains=chains)
```

## 5. 内存管理器 (core/memory_manager.py)
```python
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict
import pickle
import json
import faiss
import hashlib
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer

@dataclass
class MemoryEntry:
    """内存条目"""
    id: str
    type: str  # 'code', 'analysis', 'conversation', 'task'
    content: Any
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    accessed_at: datetime = None
    access_count: int = 0
    importance: float = 0.5
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.accessed_at is None:
            self.accessed_at = self.created_at
        if self.metadata is None:
            self.metadata = {}

class MemoryManager:
    """高级内存管理系统"""
    
    def __init__(self, memory_size_mb: int = 2048):
        self.memory_size_mb = memory_size_mb
        self.entries: Dict[str, MemoryEntry] = {}
        
        # 短期记忆（快速访问）
        self.short_term_memory = deque(maxlen=100)
        
        # 长期记忆（向量数据库）
        self.long_term_memory = None
        self.embedding_model = None
        
        # 工作记忆（当前上下文）
        self.working_memory: Dict[str, Any] = {}
        
        # 情景记忆（任务历史）
        self.episodic_memory: List[Dict[str, Any]] = []
        
        # 语义记忆（知识库）
        self.semantic_memory: Dict[str, Any] = defaultdict(list)
        
        # 内存统计
        self.stats = {
            'total_entries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0
        }
        
    async def initialize(self):
        """初始化内存系统"""
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 初始化向量索引
        self.long_term_memory = faiss.IndexFlatL2(384)  # 384是嵌入维度
        
        # 加载持久化数据
        await self._load_persistent_memory()
        
    async def store(
        self,
        content: Any,
        type: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> str:
        """存储内容到内存"""
        
        # 生成ID
        entry_id = self._generate_id(content)
        
        # 创建嵌入
        embedding = None
        if isinstance(content, str):
            embedding = await self._create_embedding(content)
            
        # 创建内存条目
        entry = MemoryEntry(
            id=entry_id,
            type=type,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance
        )
        
        # 存储到不同的内存系统
        self.entries[entry_id] = entry
        self.short_term_memory.append(entry_id)
        
        # 如果有嵌入，存储到长期记忆
        if embedding is not None:
            self.long_term_memory.add(np.array([embedding]))
            
        # 更新统计
        self.stats['total_entries'] += 1
        
        # 检查内存限制
        await self._check_memory_limit()
        
        return entry_id
        
    async def retrieve(
        self,
        query: str,
        type: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """检索相关内容"""
        
        # 创建查询嵌入
        query_embedding = await self._create_embedding(query)
        
        # 搜索相似内容
        if self.long_term_memory.ntotal > 0:
            distances, indices = self.long_term_memory.search(
                np.array([query_embedding]), 
                min(limit, self.long_term_memory.ntotal)
            )
            
            # 获取对应的条目
            results = []
            entry_list = list(self.entries.values())
            
            for idx in indices[0]:
                if idx < len(entry_list):
                    entry = entry_list[idx]
                    if type is None or entry.type == type:
                        # 更新访问信息
                        entry.accessed_at = datetime.now()
                        entry.access_count += 1
                        results.append(entry)
                        
            self.stats['cache_hits'] += len(results)
            return results[:limit]
            
        self.stats['cache_misses'] += 1
        return []
        
    async def update_working_memory(self, key: str, value: Any):
        """更新工作记忆"""
        self.working_memory[key] = {
            'value': value,
            'timestamp': datetime.now()
        }
        
        # 清理过期的工作记忆
        cutoff = datetime.now() - timedelta(hours=1)
        self.working_memory = {
            k: v for k, v in self.working_memory.items()
            if v['timestamp'] > cutoff
        }
        
    async def add_episode(self, episode: Dict[str, Any]):
        """添加情景记忆"""
        episode['timestamp'] = datetime.now()
        self.episodic_memory.append(episode)
        
        # 限制情景记忆大小
        if len(self.episodic_memory) > 1000:
            self.episodic_memory = self.episodic_memory[-1000:]
            
    async def store_analysis(self, code: str, analysis: Dict[str, Any]):
        """存储代码分析结果"""
        await self.store(
            content={
                'code': code,
                'analysis': analysis
            },
            type='analysis',
            metadata={
                'code_hash': hashlib.md5(code.encode()).hexdigest(),
                'timestamp': datetime.now().isoformat()
            },
            importance=0.7
        )
        
    async def get_relevant_context(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """获取相关上下文"""
        
        # 从不同内存系统检索
        results = []
        
        # 1. 工作记忆（最相关）
        for key, value in self.working_memory.items():
            if query.lower() in key.lower():
                results.append({
                    'source': 'working_memory',
                    'content': value['value'],
                    'relevance': 0.9
                })
                
        # 2. 短期记忆
        for entry_id in reversed(self.short_term_memory):
            entry = self.entries.get(entry_id)
            if entry and self._is_relevant(entry, query):
                results.append({
                    'source': 'short_term_memory',
                    'content': entry.content,
                    'relevance': 0.7
                })
                
        # 3. 长期记忆（语义搜索）
        semantic_results = await self.retrieve(query, limit=limit)
        for entry in semantic_results:
            results.append({
                'source': 'long_term_memory',
                'content': entry.content,
                'relevance': 0.5
            })
            
        # 排序并返回
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:limit]
        
    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """获取任务结果"""
        # 从情景记忆中查找
        for episode in reversed(self.episodic_memory):
            if episode.get('task_id') == task_id:
                return episode.get('result', {})
                
        return {}
        
    async def consolidate_memory(self):
        """整合记忆（将短期记忆转为长期记忆）"""
        # 评估短期记忆中的条目
        for entry_id in list(self.short_term_memory):
            entry = self.entries.get(entry_id)
            if not entry:
                continue
                
            # 计算重要性分数
            importance_score = self._calculate_importance(entry)
            
            # 如果重要性高，确保在长期记忆中
            if importance_score > 0.7:
                entry.importance = importance_score
                
                # 更新语义记忆
                if entry.type in ['code', 'analysis']:
                    self.semantic_memory[entry.type].append({
                        'content': entry.content,
                        'metadata': entry.metadata,
                        'importance': importance_score
                    })
                    
    def _calculate_importance(self, entry: MemoryEntry) -> float:
        """计算条目重要性"""
        # 基于多个因素计算重要性
        factors = []
        
        # 访问频率
        access_factor = min(entry.access_count / 10, 1.0)
        factors.append(access_factor * 0.3)
        
        # 时间因素（最近访问）
        time_since_access = (datetime.now() - entry.accessed_at).total_seconds()
        recency_factor = max(0, 1 - time_since_access / (24 * 3600))
        factors.append(recency_factor * 0.2)
        
        # 内容类型权重
        type_weights = {
            'code': 0.8,
            'analysis': 0.7,
            'task': 0.6,
            'conversation': 0.5
        }
        type_factor = type_weights.get(entry.type, 0.5)
        factors.append(type_factor * 0.3)
        
        # 原始重要性
        factors.append(entry.importance * 0.2)
        
        return sum(factors)
        
    def _is_relevant(self, entry: MemoryEntry, query: str) -> bool:
        """检查条目是否相关"""
        query_lower = query.lower()
        
        # 检查内容
        if isinstance(entry.content, str):
            if query_lower in entry.content.lower():
                return True
        elif isinstance(entry.content, dict):
            for value in entry.content.values():
                if isinstance(value, str) and query_lower in value.lower():
                    return True
                    
        # 检查元数据
        for value in entry.metadata.values():
            if isinstance(value, str) and query_lower in value.lower():
                return True
                
        return False
        
    async def _create_embedding(self, text: str) -> np.ndarray:
        """创建文本嵌入"""
        # 使用句子转换器创建嵌入
        embedding = self.embedding_model.encode(text)
        return embedding
        
    def _generate_id(self, content: Any) -> str:
        """生成唯一ID"""
        # 基于内容生成哈希
        if isinstance(content, str):
            content_str = content
        else:
            content_str = json.dumps(content, sort_keys=True)
            
        hash_obj = hashlib.sha256(content_str.encode())
        return f"mem_{hash_obj.hexdigest()[:12]}"
        
    async def _check_memory_limit(self):
        """检查内存限制"""
        # 估算当前内存使用
        estimated_size = len(pickle.dumps(self.entries)) / (1024 * 1024)
        
        if estimated_size > self.memory_size_mb * 0.9:
            # 触发内存清理
            await self._evict_memories()
            
    async def _evict_memories(self):
        """清理低重要性的记忆"""
        # 计算所有条目的重要性
        entries_with_importance = [
            (entry_id, self._calculate_importance(entry))
            for entry_id, entry in self.entries.items()
        ]
        
        # 按重要性排序
        entries_with_importance.sort(key=lambda x: x[1])
        
        # 删除最不重要的20%
        num_to_evict = len(entries_with_importance) // 5
        
        for entry_id, _ in entries_with_importance[:num_to_evict]:
            del self.entries[entry_id]
            self.stats['evictions'] += 1
            
    async def _load_persistent_memory(self):
        """加载持久化内存"""
        # TODO: 从文件系统加载
        pass
        
    async def save_persistent_memory(self):
        """保存持久化内存"""
        # TODO: 保存到文件系统
        pass
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        return {
            **self.stats,
            'short_term_size': len(self.short_term_memory),
            'long_term_size': self.long_term_memory.ntotal if self.long_term_memory else 0,
            'working_memory_size': len(self.working_memory),
            'episodic_memory_size': len(self.episodic_memory),
            'semantic_memory_size': sum(len(v) for v in self.semantic_memory.values()),
            'estimated_size_mb': len(pickle.dumps(self.entries)) / (1024 * 1024)
        }
        
    async def clear_memory(self, type: Optional[str] = None):
        """清空内存"""
        if type:
            # 清空特定类型
            self.entries = {
                k: v for k, v in self.entries.items()
                if v.type != type
            }
        else:
            # 清空所有
            self.entries.clear()
            self.short_term_memory.clear()
            self.working_memory.clear()
            self.episodic_memory.clear()
            self.semantic_memory.clear()
            
            # 重新初始化向量索引
            if self.long_term_memory:
                self.long_term_memory = faiss.IndexFlatL2(384)
```

这些核心组件提供了：

1. **代码分析引擎**：
   - 多语言支持
   - 深度AST分析
   - 安全检查
   - 代码度量计算
   - 差异分析

2. **代码生成引擎**：
   - 模板系统
   - 多种项目类型支持
   - 智能提示构建
   - 代码后处理

3. **系统控制器**：
   - 安全的代码执行
   - Docker沙箱支持
   - 屏幕控制
   - 文件管理
   - 进程监控

4. **LangChain集成**：
   - 多种处理链
   - 链编排
   - 输出解析
   - 记忆管理

5. **内存管理器**：
   - 多层内存系统
   - 向量搜索
   - 智能清理
   - 持久化支持

这个系统设计是高度模块化和可扩展的，可以轻松添加新功能和语言支持。
