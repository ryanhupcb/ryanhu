# enhanced_tools.py
# 增强版工具模块 - 支持代码开发、文件操作、系统控制等核心功能

import asyncio
import os
import sys
import json
import subprocess
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import ast
import black
import autopep8
import pylint.lint
from pylint.reporters.text import TextReporter
from io import StringIO
import difflib
import yaml
import toml

# Code analysis tools
try:
    import radon.complexity as radon_cc
    import radon.metrics as radon_metrics
except ImportError:
    radon_cc = None
    radon_metrics = None

# TypeScript support
try:
    import esprima
except ImportError:
    esprima = None

import logging
logger = logging.getLogger(__name__)

# ==================== Code Development Tools ====================

class CodeFormatter:
    """代码格式化工具"""
    
    @staticmethod
    async def format_python(code: str, line_length: int = 88) -> Dict[str, Any]:
        """格式化Python代码"""
        try:
            # Try black first
            formatted = black.format_str(code, mode=black.Mode(line_length=line_length))
            return {
                'success': True,
                'formatted_code': formatted,
                'formatter': 'black'
            }
        except Exception as e:
            # Fallback to autopep8
            try:
                formatted = autopep8.fix_code(code, options={'max_line_length': line_length})
                return {
                    'success': True,
                    'formatted_code': formatted,
                    'formatter': 'autopep8'
                }
            except Exception as e2:
                return {
                    'success': False,
                    'error': f"Black: {e}, Autopep8: {e2}"
                }
    
    @staticmethod
    async def format_javascript(code: str) -> Dict[str, Any]:
        """格式化JavaScript/TypeScript代码"""
        try:
            # Use prettier via subprocess
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ['prettier', '--write', temp_file],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                with open(temp_file, 'r') as f:
                    formatted = f.read()
                os.unlink(temp_file)
                
                return {
                    'success': True,
                    'formatted_code': formatted,
                    'formatter': 'prettier'
                }
            else:
                os.unlink(temp_file)
                return {
                    'success': False,
                    'error': result.stderr
                }
                
        except FileNotFoundError:
            return {
                'success': False,
                'error': 'Prettier not installed. Run: npm install -g prettier'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class CodeAnalyzer:
    """代码分析工具"""
    
    @staticmethod
    async def analyze_python(code: str) -> Dict[str, Any]:
        """分析Python代码"""
        analysis = {
            'syntax_valid': True,
            'complexity': {},
            'metrics': {},
            'issues': [],
            'suggestions': []
        }
        
        # Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            analysis['syntax_valid'] = False
            analysis['issues'].append(f"Syntax error at line {e.lineno}: {e.msg}")
            return analysis
        
        # Complexity analysis
        if radon_cc:
            try:
                blocks = radon_cc.cc_visit(code)
                analysis['complexity'] = {
                    'functions': []
                }
                
                for block in blocks:
                    if isinstance(block, radon_cc.Function):
                        analysis['complexity']['functions'].append({
                            'name': block.name,
                            'complexity': block.complexity,
                            'rank': radon_cc.cc_rank(block.complexity)
                        })
                        
                        if block.complexity > 10:
                            analysis['suggestions'].append(
                                f"Function '{block.name}' has high complexity ({block.complexity}). Consider refactoring."
                            )
            except:
                pass
        
        # Code metrics
        if radon_metrics:
            try:
                metrics = radon_metrics.analyze(code)
                analysis['metrics'] = {
                    'loc': metrics.loc,
                    'lloc': metrics.lloc,
                    'sloc': metrics.sloc,
                    'comments': metrics.comments,
                    'multi': metrics.multi,
                    'blank': metrics.blank
                }
            except:
                pass
        
        # Pylint analysis
        try:
            output = StringIO()
            reporter = TextReporter(output)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            pylint.lint.Run([temp_file, '--errors-only'], reporter=reporter, exit=False)
            
            lint_output = output.getvalue()
            if lint_output:
                for line in lint_output.split('\n'):
                    if line and not line.startswith('*'):
                        analysis['issues'].append(line.strip())
            
            os.unlink(temp_file)
        except:
            pass
        
        return analysis
    
    @staticmethod
    async def analyze_javascript(code: str) -> Dict[str, Any]:
        """分析JavaScript/TypeScript代码"""
        analysis = {
            'syntax_valid': True,
            'issues': [],
            'suggestions': []
        }
        
        if esprima:
            try:
                esprima.parseScript(code)
            except Exception as e:
                analysis['syntax_valid'] = False
                analysis['issues'].append(f"Syntax error: {e}")
        
        # Use ESLint via subprocess if available
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ['eslint', '--format', 'json', temp_file],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0 and result.stdout:
                eslint_results = json.loads(result.stdout)
                if eslint_results and eslint_results[0]['messages']:
                    for msg in eslint_results[0]['messages']:
                        severity = 'Error' if msg['severity'] == 2 else 'Warning'
                        analysis['issues'].append(
                            f"{severity} at line {msg['line']}: {msg['message']}"
                        )
            
            os.unlink(temp_file)
        except:
            pass
        
        return analysis

class CodeGenerator:
    """代码生成工具"""
    
    @staticmethod
    async def generate_python_class(class_name: str, attributes: List[str], 
                                  methods: List[Dict[str, Any]]) -> str:
        """生成Python类"""
        code_lines = [
            f"class {class_name}:",
            '    """Auto-generated class"""',
            '',
            '    def __init__(self):'
        ]
        
        # Add attributes
        for attr in attributes:
            code_lines.append(f'        self.{attr} = None')
        
        if not attributes:
            code_lines.append('        pass')
        
        # Add methods
        for method in methods:
            code_lines.extend([
                '',
                f"    def {method['name']}(self{', ' + method.get('params', '') if method.get('params') else ''}):",
                f'        """{method.get("docstring", "Method documentation")}"""',
                '        # TODO: Implement this method',
                '        pass'
            ])
        
        return '\n'.join(code_lines)
    
    @staticmethod
    async def generate_typescript_interface(interface_name: str, 
                                          properties: Dict[str, str]) -> str:
        """生成TypeScript接口"""
        code_lines = [
            f"interface {interface_name} {{"
        ]
        
        for prop_name, prop_type in properties.items():
            code_lines.append(f"    {prop_name}: {prop_type};")
        
        code_lines.append("}")
        
        return '\n'.join(code_lines)
    
    @staticmethod
    async def generate_rest_api_endpoint(method: str, path: str, 
                                       handler_name: str, 
                                       language: str = "python") -> str:
        """生成REST API端点代码"""
        if language == "python":
            # Flask example
            code = f'''
@app.route('{path}', methods=['{method.upper()}'])
async def {handler_name}():
    """Handle {method.upper()} {path}"""
    try:
        # Parse request data
        data = request.get_json() if request.is_json else {{}}
        
        # TODO: Implement business logic here
        result = {{
            'success': True,
            'message': 'Endpoint not implemented'
        }}
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({{'error': str(e)}}), 500
'''
        elif language == "typescript":
            # Express example
            code = f'''
app.{method.lower()}('{path}', async (req: Request, res: Response) => {{
    try {{
        // Parse request data
        const data = req.body;
        
        // TODO: Implement business logic here
        const result = {{
            success: true,
            message: 'Endpoint not implemented'
        }};
        
        res.status(200).json(result);
        
    }} catch (error) {{
        res.status(500).json({{ error: error.message }});
    }}
}});
'''
        else:
            code = f"// {method.upper()} {path} endpoint for {language}"
        
        return code

# ==================== Enhanced File Operations ====================

class EnhancedFileOperations:
    """增强版文件操作"""
    
    def __init__(self, workspace: Path = None):
        self.workspace = Path(workspace) if workspace else Path.cwd()
        
    async def create_project_structure(self, project_name: str, 
                                     structure: Dict[str, Any]) -> Dict[str, Any]:
        """创建项目结构"""
        project_path = self.workspace / project_name
        created_files = []
        
        try:
            # Create project root
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Create structure recursively
            await self._create_structure(project_path, structure, created_files)
            
            return {
                'success': True,
                'project_path': str(project_path),
                'created_files': created_files
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _create_structure(self, base_path: Path, structure: Dict[str, Any], 
                              created_files: List[str]):
        """递归创建文件结构"""
        for name, content in structure.items():
            path = base_path / name
            
            if isinstance(content, dict):
                # Directory
                path.mkdir(parents=True, exist_ok=True)
                await self._create_structure(path, content, created_files)
            else:
                # File
                path.write_text(content)
                created_files.append(str(path))
    
    async def find_files(self, pattern: str, recursive: bool = True) -> List[str]:
        """查找文件"""
        if recursive:
            files = list(self.workspace.rglob(pattern))
        else:
            files = list(self.workspace.glob(pattern))
            
        return [str(f) for f in files]
    
    async def batch_rename(self, files: List[str], 
                         rename_function: Callable[[str], str]) -> Dict[str, Any]:
        """批量重命名文件"""
        renamed = []
        errors = []
        
        for file_path in files:
            try:
                path = Path(file_path)
                new_name = rename_function(path.name)
                new_path = path.parent / new_name
                
                path.rename(new_path)
                renamed.append({
                    'old': str(path),
                    'new': str(new_path)
                })
                
            except Exception as e:
                errors.append({
                    'file': file_path,
                    'error': str(e)
                })
        
        return {
            'success': len(errors) == 0,
            'renamed': renamed,
            'errors': errors
        }
    
    async def create_backup(self, source: str, backup_dir: str = None) -> Dict[str, Any]:
        """创建备份"""
        try:
            source_path = Path(source)
            
            if not source_path.exists():
                return {
                    'success': False,
                    'error': 'Source does not exist'
                }
            
            # Create backup directory
            if backup_dir:
                backup_base = Path(backup_dir)
            else:
                backup_base = self.workspace / 'backups'
                
            backup_base.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{source_path.name}_{timestamp}"
            
            if source_path.is_file():
                backup_path = backup_base / backup_name
                shutil.copy2(source_path, backup_path)
            else:
                backup_path = backup_base / backup_name
                shutil.copytree(source_path, backup_path)
            
            return {
                'success': True,
                'backup_path': str(backup_path),
                'source': str(source_path),
                'timestamp': timestamp
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def diff_files(self, file1: str, file2: str, 
                        context_lines: int = 3) -> Dict[str, Any]:
        """比较文件差异"""
        try:
            with open(file1, 'r') as f1:
                lines1 = f1.readlines()
                
            with open(file2, 'r') as f2:
                lines2 = f2.readlines()
            
            # Generate unified diff
            diff = list(difflib.unified_diff(
                lines1, lines2,
                fromfile=file1,
                tofile=file2,
                n=context_lines
            ))
            
            # Calculate statistics
            added = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
            removed = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
            
            return {
                'success': True,
                'diff': ''.join(diff),
                'has_differences': len(diff) > 0,
                'statistics': {
                    'lines_added': added,
                    'lines_removed': removed,
                    'total_changes': added + removed
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# ==================== Development Environment Tools ====================

class DevelopmentEnvironment:
    """开发环境管理工具"""
    
    @staticmethod
    async def create_virtual_environment(env_name: str, python_version: str = None) -> Dict[str, Any]:
        """创建Python虚拟环境"""
        try:
            if python_version:
                cmd = [f'python{python_version}', '-m', 'venv', env_name]
            else:
                cmd = [sys.executable, '-m', 'venv', env_name]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'env_path': env_name,
                    'activation_command': f'source {env_name}/bin/activate' if os.name != 'nt' else f'{env_name}\\Scripts\\activate'
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    async def install_dependencies(requirements: Union[str, List[str]], 
                                 env_path: str = None) -> Dict[str, Any]:
        """安装依赖"""
        try:
            if env_path:
                pip_cmd = str(Path(env_path) / 'bin' / 'pip') if os.name != 'nt' else str(Path(env_path) / 'Scripts' / 'pip')
            else:
                pip_cmd = 'pip'
            
            if isinstance(requirements, str):
                # Requirements file
                cmd = [pip_cmd, 'install', '-r', requirements]
            else:
                # List of packages
                cmd = [pip_cmd, 'install'] + requirements
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    async def run_tests(test_path: str = "tests", 
                       framework: str = "pytest") -> Dict[str, Any]:
        """运行测试"""
        try:
            if framework == "pytest":
                cmd = ['pytest', test_path, '-v', '--json-report', '--json-report-file=test_report.json']
            elif framework == "unittest":
                cmd = [sys.executable, '-m', 'unittest', 'discover', test_path]
            else:
                return {
                    'success': False,
                    'error': f'Unsupported test framework: {framework}'
                }
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse test results if available
            test_results = {}
            if framework == "pytest" and os.path.exists('test_report.json'):
                with open('test_report.json', 'r') as f:
                    test_results = json.load(f)
                os.unlink('test_report.json')
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None,
                'test_results': test_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# ==================== Git Integration ====================

class GitOperations:
    """Git操作工具"""
    
    @staticmethod
    async def init_repository(path: str = ".") -> Dict[str, Any]:
        """初始化Git仓库"""
        try:
            result = subprocess.run(
                ['git', 'init', path],
                capture_output=True,
                text=True
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    async def commit_changes(message: str, files: List[str] = None) -> Dict[str, Any]:
        """提交更改"""
        try:
            # Add files
            if files:
                add_cmd = ['git', 'add'] + files
            else:
                add_cmd = ['git', 'add', '.']
                
            add_result = subprocess.run(add_cmd, capture_output=True, text=True)
            
            if add_result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Failed to add files: {add_result.stderr}"
                }
            
            # Commit
            commit_cmd = ['git', 'commit', '-m', message]
            commit_result = subprocess.run(commit_cmd, capture_output=True, text=True)
            
            return {
                'success': commit_result.returncode == 0,
                'output': commit_result.stdout,
                'error': commit_result.stderr if commit_result.returncode != 0 else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    async def create_branch(branch_name: str, checkout: bool = True) -> Dict[str, Any]:
        """创建分支"""
        try:
            if checkout:
                cmd = ['git', 'checkout', '-b', branch_name]
            else:
                cmd = ['git', 'branch', branch_name]
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None,
                'branch': branch_name
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# ==================== Documentation Tools ====================

class DocumentationGenerator:
    """文档生成工具"""
    
    @staticmethod
    async def generate_readme(project_info: Dict[str, Any]) -> str:
        """生成README文件"""
        readme_lines = [
            f"# {project_info.get('name', 'Project Name')}",
            "",
            project_info.get('description', 'Project description'),
            "",
            "## Installation",
            "",
            "```bash",
            project_info.get('installation', 'pip install -r requirements.txt'),
            "```",
            "",
            "## Usage",
            "",
            "```python",
            project_info.get('usage_example', '# Usage example here'),
            "```",
            ""
        ]
        
        if project_info.get('features'):
            readme_lines.extend([
                "## Features",
                ""
            ])
            for feature in project_info['features']:
                readme_lines.append(f"- {feature}")
            readme_lines.append("")
        
        if project_info.get('requirements'):
            readme_lines.extend([
                "## Requirements",
                ""
            ])
            for req in project_info['requirements']:
                readme_lines.append(f"- {req}")
            readme_lines.append("")
        
        readme_lines.extend([
            "## License",
            "",
            project_info.get('license', 'MIT License'),
            "",
            "## Contributing",
            "",
            project_info.get('contributing', 'Contributions are welcome!')
        ])
        
        return '\n'.join(readme_lines)
    
    @staticmethod
    async def extract_docstrings(file_path: str) -> Dict[str, Any]:
        """提取Python文件的文档字符串"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            docstrings = {
                'module': ast.get_docstring(tree),
                'classes': {},
                'functions': {}
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    docstrings['classes'][node.name] = {
                        'docstring': ast.get_docstring(node),
                        'methods': {}
                    }
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            docstrings['classes'][node.name]['methods'][item.name] = ast.get_docstring(item)
                            
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    docstrings['functions'][node.name] = ast.get_docstring(node)
            
            return {
                'success': True,
                'docstrings': docstrings
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# ==================== Configuration Management ====================

class ConfigurationManager:
    """配置文件管理工具"""
    
    @staticmethod
    async def load_config(file_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {
                    'success': False,
                    'error': 'Configuration file not found'
                }
            
            # Determine format by extension
            ext = path.suffix.lower()
            
            if ext == '.json':
                with open(path, 'r') as f:
                    config = json.load(f)
            elif ext in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
            elif ext == '.toml':
                with open(path, 'r') as f:
                    config = toml.load(f)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported config format: {ext}'
                }
            
            return {
                'success': True,
                'config': config,
                'format': ext[1:]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    async def save_config(config: Dict[str, Any], file_path: str, 
                        format: str = None) -> Dict[str, Any]:
        """保存配置文件"""
        try:
            path = Path(file_path)
            
            # Determine format
            if format:
                fmt = format.lower()
            else:
                fmt = path.suffix.lower()[1:] if path.suffix else 'json'
            
            # Save in appropriate format
            if fmt == 'json':
                with open(path, 'w') as f:
                    json.dump(config, f, indent=2)
            elif fmt in ['yaml', 'yml']:
                with open(path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            elif fmt == 'toml':
                with open(path, 'w') as f:
                    toml.dump(config, f)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported config format: {fmt}'
                }
            
            return {
                'success': True,
                'file_path': str(path),
                'format': fmt
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# ==================== Database Tools ====================

class DatabaseTools:
    """数据库操作工具"""
    
    @staticmethod
    async def create_sqlite_schema(db_path: str, schema: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """创建SQLite数据库架构"""
        import sqlite3
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            created_tables = []
            
            for table_name, columns in schema.items():
                # Build CREATE TABLE statement
                col_definitions = []
                
                for col in columns:
                    col_def = f"{col['name']} {col['type']}"
                    
                    if col.get('primary_key'):
                        col_def += " PRIMARY KEY"
                    if col.get('not_null'):
                        col_def += " NOT NULL"
                    if col.get('unique'):
                        col_def += " UNIQUE"
                    if 'default' in col:
                        col_def += f" DEFAULT {col['default']}"
                        
                    col_definitions.append(col_def)
                
                create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_definitions)})"
                cursor.execute(create_stmt)
                created_tables.append(table_name)
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'database': db_path,
                'created_tables': created_tables
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    async def generate_orm_models(schema: Dict[str, List[Dict[str, Any]]], 
                                orm_type: str = "sqlalchemy") -> str:
        """生成ORM模型代码"""
        if orm_type == "sqlalchemy":
            code_lines = [
                "from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text",
                "from sqlalchemy.ext.declarative import declarative_base",
                "",
                "Base = declarative_base()",
                ""
            ]
            
            for table_name, columns in schema.items():
                # Convert table name to class name
                class_name = ''.join(word.capitalize() for word in table_name.split('_'))
                
                code_lines.extend([
                    f"class {class_name}(Base):",
                    f'    __tablename__ = "{table_name}"',
                    ""
                ])
                
                for col in columns:
                    col_type = col['type'].upper()
                    
                    # Map SQL types to SQLAlchemy types
                    type_mapping = {
                        'INTEGER': 'Integer',
                        'TEXT': 'Text',
                        'VARCHAR': f"String({col.get('length', 255)})",
                        'BOOLEAN': 'Boolean',
                        'DATETIME': 'DateTime',
                        'FLOAT': 'Float',
                        'REAL': 'Float'
                    }
                    
                    sa_type = type_mapping.get(col_type, 'String(255)')
                    
                    col_def = f"    {col['name']} = Column({sa_type}"
                    
                    if col.get('primary_key'):
                        col_def += ", primary_key=True"
                    if col.get('unique'):
                        col_def += ", unique=True"
                    if col.get('not_null'):
                        col_def += ", nullable=False"
                    if 'default' in col:
                        col_def += f", default={col['default']}"
                        
                    col_def += ")"
                    code_lines.append(col_def)
                
                code_lines.append("")
            
            return '\n'.join(code_lines)
            
        else:
            return f"# ORM type {orm_type} not supported yet"

# ==================== Tool Registry ====================

class EnhancedToolRegistry:
    """增强版工具注册表"""
    
    def __init__(self):
        self.tools = {
            'code_formatter': CodeFormatter(),
            'code_analyzer': CodeAnalyzer(),
            'code_generator': CodeGenerator(),
            'file_operations': EnhancedFileOperations(),
            'dev_environment': DevelopmentEnvironment(),
            'git_operations': GitOperations(),
            'documentation': DocumentationGenerator(),
            'config_manager': ConfigurationManager(),
            'database_tools': DatabaseTools()
        }
    
    def get_tool(self, tool_name: str):
        """获取工具实例"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """列出所有可用工具"""
        return list(self.tools.keys())
    
    def get_tool_methods(self, tool_name: str) -> List[str]:
        """获取工具的所有方法"""
        tool = self.get_tool(tool_name)
        if tool:
            return [method for method in dir(tool) 
                   if not method.startswith('_') and callable(getattr(tool, method))]
        return []

# ==================== Demo and Testing ====================

async def demo_tools():
    """演示工具使用"""
    print("=== Enhanced Tools Demo ===\n")
    
    registry = EnhancedToolRegistry()
    
    # 1. Code formatting demo
    print("1. Code Formatting:")
    formatter = registry.get_tool('code_formatter')
    
    python_code = '''
def   hello_world( name   ):
    print(  f"Hello, {name}!"   )
    return   True
'''
    
    result = await formatter.format_python(python_code)
    if result['success']:
        print("Formatted Python code:")
        print(result['formatted_code'])
    
    # 2. Code analysis demo
    print("\n2. Code Analysis:")
    analyzer = registry.get_tool('code_analyzer')
    
    analysis = await analyzer.analyze_python(result['formatted_code'])
    print(f"Syntax valid: {analysis['syntax_valid']}")
    print(f"Metrics: {analysis['metrics']}")
    
    # 3. Project structure creation
    print("\n3. Project Structure Creation:")
    file_ops = registry.get_tool('file_operations')
    
    project_structure = {
        'src': {
            '__init__.py': '',
            'main.py': '# Main application file\n',
            'utils.py': '# Utility functions\n'
        },
        'tests': {
            '__init__.py': '',
            'test_main.py': '# Test cases\n'
        },
        'README.md': '# My Project\n',
        'requirements.txt': 'pytest\nblack\n'
    }
    
    result = await file_ops.create_project_structure('demo_project', project_structure)
    if result['success']:
        print(f"Created project at: {result['project_path']}")
        print(f"Files created: {len(result['created_files'])}")
    
    # 4. Documentation generation
    print("\n4. Documentation Generation:")
    doc_gen = registry.get_tool('documentation')
    
    project_info = {
        'name': 'Awesome Project',
        'description': 'A demonstration of enhanced tools',
        'features': ['Code formatting', 'Analysis', 'Documentation'],
        'installation': 'pip install awesome-project',
        'license': 'MIT'
    }
    
    readme = await doc_gen.generate_readme(project_info)
    print("Generated README preview:")
    print(readme[:200] + "...")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(demo_tools())
