# HyperAgent Debug Guide & UI Implementation

## Debug Guide

### 1. Debug Infrastructure Setup

```python
# INSERT INTO: hyperagent-core.py, Line 100 (after imports)
import inspect
import tracemalloc
from contextlib import contextmanager

class DebugConfig:
    """Global debug configuration"""
    DEBUG_MODE = False
    TRACE_MEMORY = False
    LOG_LLM_CALLS = False
    PROFILE_PERFORMANCE = False
    TRACK_AGENT_STATES = False
    
    # Debug output levels
    DEBUG_LEVELS = {
        'CRITICAL': 50,
        'ERROR': 40,
        'WARNING': 30,
        'INFO': 20,
        'DEBUG': 10,
        'TRACE': 5
    }

# Global debug instance
DEBUG = DebugConfig()
```

### 2. Enhanced Debug Logger

```python
# INSERT INTO: hyperagent-core.py, Line 150 (debug utilities)
class EnhancedDebugLogger:
    """Advanced debug logging with context tracking"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context_stack = []
        self.performance_traces = defaultdict(list)
        
    @contextmanager
    def trace_context(self, context_name: str, **kwargs):
        """Track execution context for debugging"""
        start_time = time.time()
        context_id = f"{context_name}_{uuid.uuid4().hex[:8]}"
        
        self.context_stack.append({
            'id': context_id,
            'name': context_name,
            'start_time': start_time,
            'kwargs': kwargs
        })
        
        if DEBUG.DEBUG_MODE:
            self.logger.debug(f"[ENTER] {context_name} - {kwargs}")
        
        try:
            yield context_id
        except Exception as e:
            self.logger.error(f"[ERROR] {context_name} - {e}")
            self._dump_context_stack()
            raise
        finally:
            elapsed = time.time() - start_time
            self.context_stack.pop()
            
            if DEBUG.PROFILE_PERFORMANCE:
                self.performance_traces[context_name].append(elapsed)
            
            if DEBUG.DEBUG_MODE:
                self.logger.debug(f"[EXIT] {context_name} - {elapsed:.3f}s")
    
    def _dump_context_stack(self):
        """Dump full context stack on error"""
        self.logger.error("=== Context Stack Dump ===")
        for i, ctx in enumerate(self.context_stack):
            self.logger.error(f"  [{i}] {ctx['name']} - {ctx['kwargs']}")
```

### 3. Agent State Debugger

```python
# INSERT INTO: hyperagent-core.py, Line 850 (in BaseAgent)
class AgentStateDebugger:
    """Debug tool for tracking agent states"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state_history = deque(maxlen=1000)
        self.message_trace = deque(maxlen=500)
        self.task_trace = deque(maxlen=200)
        
    def record_state_change(self, old_state: str, new_state: str, reason: str):
        """Record state transitions"""
        if DEBUG.TRACK_AGENT_STATES:
            self.state_history.append({
                'timestamp': datetime.now(),
                'old_state': old_state,
                'new_state': new_state,
                'reason': reason,
                'stack_trace': traceback.format_stack()[-5:]  # Last 5 frames
            })
    
    def record_message(self, message: AgentMessage, direction: str):
        """Track messages for debugging"""
        if DEBUG.DEBUG_MODE:
            self.message_trace.append({
                'timestamp': datetime.now(),
                'direction': direction,  # 'in' or 'out'
                'message_type': message.message_type,
                'sender': message.sender_id,
                'receiver': message.receiver_id,
                'correlation_id': message.correlation_id
            })
    
    def dump_debug_info(self) -> Dict[str, Any]:
        """Dump complete debug information"""
        return {
            'agent_id': self.agent_id,
            'state_history': list(self.state_history)[-50:],
            'message_trace': list(self.message_trace)[-50:],
            'task_trace': list(self.task_trace)[-20:],
            'current_memory_usage': self._get_memory_usage()
        }
```

### 4. LLM Call Debugger

```python
# INSERT INTO: hyperagent-core.py, Line 1550 (in LLMIntegration)
class LLMCallDebugger:
    """Debug LLM API calls for cost tracking and optimization"""
    
    def __init__(self):
        self.call_history = deque(maxlen=1000)
        self.token_usage = defaultdict(lambda: {'input': 0, 'output': 0})
        self.cost_tracker = defaultdict(float)
        self.cache_stats = {'hits': 0, 'misses': 0}
        
    async def debug_llm_call(self, prompt: str, model: ModelProvider, 
                            response: str, tokens: Dict[str, int]):
        """Log LLM call for debugging"""
        if DEBUG.LOG_LLM_CALLS:
            call_info = {
                'timestamp': datetime.now(),
                'model': model.value,
                'prompt_preview': prompt[:200] + '...' if len(prompt) > 200 else prompt,
                'response_preview': response[:200] + '...' if len(response) > 200 else response,
                'tokens': tokens,
                'estimated_cost': self._calculate_cost(model, tokens)
            }
            
            self.call_history.append(call_info)
            self.token_usage[model.value]['input'] += tokens.get('input', 0)
            self.token_usage[model.value]['output'] += tokens.get('output', 0)
            self.cost_tracker[model.value] += call_info['estimated_cost']
            
            # Log expensive calls
            if call_info['estimated_cost'] > 0.10:  # $0.10
                logging.warning(f"Expensive LLM call: {model.value} - ${call_info['estimated_cost']:.4f}")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost breakdown by model"""
        return {
            'total_cost': sum(self.cost_tracker.values()),
            'by_model': dict(self.cost_tracker),
            'token_usage': dict(self.token_usage),
            'cache_effectiveness': self.cache_stats['hits'] / max(self.cache_stats['hits'] + self.cache_stats['misses'], 1)
        }
```

### 5. Memory Leak Detector

```python
# INSERT INTO: hyperagent-core.py, Line 400 (debug utilities)
class MemoryLeakDetector:
    """Detect memory leaks in agents and components"""
    
    def __init__(self):
        self.object_counts = defaultdict(int)
        self.growth_tracking = defaultdict(list)
        
    def snapshot(self):
        """Take memory snapshot"""
        if DEBUG.TRACE_MEMORY:
            snapshot = {}
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                snapshot[obj_type] = snapshot.get(obj_type, 0) + 1
            
            # Track growth
            for obj_type, count in snapshot.items():
                if self.object_counts[obj_type] > 0:
                    growth = count - self.object_counts[obj_type]
                    if growth > 0:
                        self.growth_tracking[obj_type].append({
                            'timestamp': datetime.now(),
                            'growth': growth,
                            'total': count
                        })
            
            self.object_counts = snapshot
    
    def get_leaks(self, threshold: int = 100) -> List[Dict[str, Any]]:
        """Identify potential memory leaks"""
        leaks = []
        for obj_type, growth_history in self.growth_tracking.items():
            if len(growth_history) > 5:
                recent_growth = sum(g['growth'] for g in growth_history[-5:])
                if recent_growth > threshold:
                    leaks.append({
                        'type': obj_type,
                        'total_objects': self.object_counts[obj_type],
                        'recent_growth': recent_growth,
                        'history': growth_history[-10:]
                    })
        return leaks
```

## UI Implementation

### Text-Based UI with Model Selection

```python
# INSERT INTO: hyperagent-core.py, Line 6000 (replace HyperAgentCLI)
class HyperAgentTextUI:
    """Text-based UI with conversation history and model selection"""
    
    def __init__(self):
        self.system = None
        self.conversation_history = []
        self.current_model = ModelProvider.CLAUDE_3_7_SONNET  # Default to cheapest
        self.session_id = str(uuid.uuid4())
        self.sidebar_width = 30
        self.main_width = 80
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def draw_ui(self):
        """Draw the text-based UI"""
        self.clear_screen()
        
        # Header
        print("=" * (self.sidebar_width + self.main_width + 3))
        print(f"{'HYPERAGENT SYSTEM v1.0':^{self.sidebar_width + self.main_width + 3}}")
        print("=" * (self.sidebar_width + self.main_width + 3))
        
        # Main layout
        self._draw_layout()
        
        # Model selection
        print("-" * (self.sidebar_width + self.main_width + 3))
        print(f"Current Model: {self._get_model_display_name(self.current_model)}")
        print("Select Model: [1] Claude 4 Opus ($$$) | [2] Claude 4 Sonnet ($$) | [3] Claude 3.7 Sonnet ($)")
        print("-" * (self.sidebar_width + self.main_width + 3))
    
    def _draw_layout(self):
        """Draw sidebar and main content area"""
        # Get conversation lines
        sidebar_lines = self._format_conversation_history()
        
        # Calculate heights
        terminal_height = 20  # Fixed height for display
        
        # Draw line by line
        for i in range(terminal_height):
            # Sidebar content
            if i == 0:
                sidebar_content = f"{'CONVERSATION HISTORY':^{self.sidebar_width}}"
            elif i == 1:
                sidebar_content = "-" * self.sidebar_width
            elif i - 2 < len(sidebar_lines):
                sidebar_content = sidebar_lines[i - 2]
            else:
                sidebar_content = " " * self.sidebar_width
            
            # Main content (empty for now, will show current conversation)
            main_content = " " * self.main_width
            
            # Print combined line
            print(f"{sidebar_content} | {main_content}")
    
    def _format_conversation_history(self) -> List[str]:
        """Format conversation history for sidebar"""
        lines = []
        for conv in self.conversation_history[-10:]:  # Last 10 conversations
            timestamp = conv['timestamp'].strftime('%H:%M')
            task_preview = conv['task'][:20] + '...' if len(conv['task']) > 20 else conv['task']
            status = '[OK]' if conv['status'] == 'completed' else '[FAIL]'
            
            lines.append(f"{timestamp} {status}")
            lines.append(f"  {task_preview}")
            lines.append("")
        
        return lines
    
    def _get_model_display_name(self, model: ModelProvider) -> str:
        """Get display name with cost indicator"""
        display_names = {
            ModelProvider.CLAUDE_4_OPUS: "Claude 4 Opus ($$$)",
            ModelProvider.CLAUDE_4_SONNET: "Claude 4 Sonnet ($$)",
            ModelProvider.CLAUDE_3_7_SONNET: "Claude 3.7 Sonnet ($)",
            ModelProvider.QWEN_MAX: "Qwen Max ($$)",
            ModelProvider.QWEN_PLUS: "Qwen Plus ($)",
            ModelProvider.QWEN_TURBO: "Qwen Turbo ($)"
        }
        return display_names.get(model, model.value)
    
    async def run_interactive(self):
        """Run the interactive text UI"""
        await self.initialize()
        
        while True:
            self.draw_ui()
            
            # Get user input
            user_input = input("\nEnter command or task (or 'help' for commands): ").strip()
            
            if not user_input:
                continue
            
            # Handle model selection
            if user_input in ['1', '2', '3']:
                self._select_model(user_input)
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                await self.shutdown()
                break
            elif user_input.lower() == 'help':
                self._show_help()
                input("\nPress Enter to continue...")
                continue
            elif user_input.lower() == 'debug':
                await self._show_debug_info()
                input("\nPress Enter to continue...")
                continue
            elif user_input.lower() == 'costs':
                await self._show_cost_breakdown()
                input("\nPress Enter to continue...")
                continue
            else:
                # Submit as task
                await self._submit_task(user_input)
    
    def _select_model(self, choice: str):
        """Handle model selection"""
        model_map = {
            '1': ModelProvider.CLAUDE_4_OPUS,
            '2': ModelProvider.CLAUDE_4_SONNET,
            '3': ModelProvider.CLAUDE_3_7_SONNET
        }
        
        self.current_model = model_map.get(choice, ModelProvider.CLAUDE_3_7_SONNET)
        
        # Show cost warning for expensive models
        if self.current_model == ModelProvider.CLAUDE_4_OPUS:
            print("\nWARNING: Claude 4 Opus is the most expensive model.")
            print("Recommended only for complex code generation tasks.")
            confirm = input("Continue with this model? (y/n): ")
            if confirm.lower() != 'y':
                self.current_model = ModelProvider.CLAUDE_3_7_SONNET
                print("Switched back to Claude 3.7 Sonnet")
    
    async def _submit_task(self, task_description: str):
        """Submit task with selected model"""
        print(f"\nProcessing with {self._get_model_display_name(self.current_model)}...")
        
        # Determine if this should use OPUS (complex code tasks)
        use_opus = False
        if 'code' in task_description.lower() and any(word in task_description.lower() 
            for word in ['complex', 'advanced', 'optimize', 'architecture']):
            if self.current_model != ModelProvider.CLAUDE_4_OPUS:
                print("\nThis appears to be a complex code task.")
                print("Recommend using Claude 4 Opus for best results.")
                switch = input("Switch to Opus? (y/n): ")
                if switch.lower() == 'y':
                    use_opus = True
        
        try:
            # Configure model preference
            task_metadata = {
                'preferred_model': ModelProvider.CLAUDE_4_OPUS if use_opus else self.current_model,
                'source': 'text_ui',
                'session_id': self.session_id
            }
            
            # Submit task
            task = await self.system.submit_task(
                task_description=task_description,
                priority=5,
                metadata=task_metadata
            )
            
            # Show progress
            progress_chars = ['|', '/', '-', '\\']
            progress_idx = 0
            
            start_time = time.time()
            while task.status not in ['completed', 'failed']:
                elapsed = time.time() - start_time
                print(f"\rProcessing {progress_chars[progress_idx]} {elapsed:.1f}s", end='', flush=True)
                progress_idx = (progress_idx + 1) % 4
                await asyncio.sleep(0.25)
                
                if elapsed > 60:  # Timeout after 60 seconds
                    print("\nTask timeout!")
                    break
            
            # Add to history
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'task': task_description,
                'status': task.status,
                'model': ModelProvider.CLAUDE_4_OPUS if use_opus else self.current_model,
                'result': task.result
            })
            
            # Show result
            print(f"\n\nTask Status: {task.status}")
            if task.status == 'completed':
                print("\nResult:")
                print("-" * 80)
                self._display_result(task.result)
                print("-" * 80)
            else:
                print(f"\nError: {task.result}")
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"\nError: {e}")
            input("\nPress Enter to continue...")
    
    def _display_result(self, result: Any):
        """Display task result in readable format"""
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"\n{key.upper()}:")
                if isinstance(value, (list, dict)):
                    print(json.dumps(value, indent=2))
                else:
                    print(str(value))
        else:
            print(str(result))
    
    async def _show_debug_info(self):
        """Show debug information"""
        self.clear_screen()
        print("=" * 80)
        print("DEBUG INFORMATION")
        print("=" * 80)
        
        # Get system status
        status = await self.system.get_system_status()
        
        print("\nSYSTEM STATUS:")
        print(f"Active Agents: {status['framework']['active_agents']}")
        print(f"Total Agents: {status['framework']['total_agents']}")
        
        print("\nAGENT DETAILS:")
        for agent_id, info in status['agents'].items():
            if info['status'] != 'idle':
                print(f"  {agent_id}: {info['status']} - {info['current_tasks']} tasks")
        
        # Memory usage
        print("\nMEMORY USAGE:")
        import psutil
        process = psutil.Process()
        print(f"  RSS: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        print(f"  VMS: {process.memory_info().vms / 1024 / 1024:.2f} MB")
        
        # Performance metrics
        if hasattr(self.system, 'performance_metrics'):
            print("\nPERFORMANCE METRICS:")
            # Add performance metrics display
    
    async def _show_cost_breakdown(self):
        """Show LLM cost breakdown"""
        self.clear_screen()
        print("=" * 80)
        print("COST BREAKDOWN")
        print("=" * 80)
        
        # Get cost data from LLM debugger
        if hasattr(self.system.llm_integration, 'debugger'):
            cost_summary = self.system.llm_integration.debugger.get_cost_summary()
            
            print(f"\nTOTAL COST: ${cost_summary['total_cost']:.4f}")
            
            print("\nBY MODEL:")
            for model, cost in cost_summary['by_model'].items():
                print(f"  {model}: ${cost:.4f}")
            
            print("\nTOKEN USAGE:")
            for model, tokens in cost_summary['token_usage'].items():
                print(f"  {model}:")
                print(f"    Input: {tokens['input']:,} tokens")
                print(f"    Output: {tokens['output']:,} tokens")
            
            print(f"\nCACHE EFFECTIVENESS: {cost_summary['cache_effectiveness']:.1%}")
        else:
            print("Cost tracking not available")
    
    def _show_help(self):
        """Show help information"""
        self.clear_screen()
        print("=" * 80)
        print("HYPERAGENT HELP")
        print("=" * 80)
        print("\nCOMMANDS:")
        print("  help     - Show this help")
        print("  debug    - Show debug information")
        print("  costs    - Show cost breakdown")
        print("  exit     - Exit the system")
        print("\nMODEL SELECTION:")
        print("  1 - Claude 4 Opus (Best for complex code, most expensive)")
        print("  2 - Claude 4 Sonnet (Balanced performance)")
        print("  3 - Claude 3.7 Sonnet (Most cost-effective)")
        print("\nTASK EXAMPLES:")
        print("  - Generate optimized Python code for data processing")
        print("  - Analyze system performance and suggest improvements")
        print("  - Create detailed technical documentation")
        print("\nCOST OPTIMIZATION TIPS:")
        print("  - Use Claude 3.7 Sonnet for simple tasks")
        print("  - Reserve Opus for complex code generation only")
        print("  - Enable caching to reduce repeated API calls")
    
    async def initialize(self):
        """Initialize the system"""
        print("Initializing HyperAgent System...")
        self.system = HyperAgentSystem("config.yaml")
        
        # Enable debug features if requested
        if os.getenv('HYPERAGENT_DEBUG', '').lower() == 'true':
            DEBUG.DEBUG_MODE = True
            DEBUG.LOG_LLM_CALLS = True
            DEBUG.TRACK_AGENT_STATES = True
            print("Debug mode enabled")
        
        await self.system.initialize()
        await self.system.start()
        print("System initialized successfully!")
        await asyncio.sleep(1)
    
    async def shutdown(self):
        """Shutdown the system"""
        print("\nShutting down...")
        if self.system:
            await self.system.stop()
        print("Goodbye!")
```

## Debug Command Examples

### 1. Enable Debug Mode
```bash
# Set environment variable
export HYPERAGENT_DEBUG=true
python hyperagent-core.py
```

### 2. Debug Specific Components
```python
# In code, enable specific debug features
DEBUG.DEBUG_MODE = True
DEBUG.LOG_LLM_CALLS = True
DEBUG.TRACE_MEMORY = True
DEBUG.PROFILE_PERFORMANCE = True
```

### 3. Performance Profiling
```python
# Profile specific function
with debug_logger.trace_context("expensive_operation", task_id=task.id):
    result = await expensive_operation()
```

### 4. Memory Leak Detection
```python
# Run periodic memory leak detection
leak_detector = MemoryLeakDetector()
while True:
    leak_detector.snapshot()
    leaks = leak_detector.get_leaks()
    if leaks:
        logging.warning(f"Potential memory leaks: {leaks}")
    await asyncio.sleep(300)  # Check every 5 minutes
```

## Cost Optimization Best Practices

1. **Model Selection Strategy**
   - Use Claude 3.7 Sonnet as default
   - Upgrade to Claude 4 Sonnet for planning/analysis
   - Reserve Claude 4 Opus only for complex code generation

2. **Caching Strategy**
   - Semantic cache with 95% similarity threshold
   - Cache hit rate target: >60%
   - Monitor cache effectiveness regularly

3. **Batch Processing**
   - Batch size: 5-10 requests
   - Timeout: 0.5-1.0 seconds
   - Combine similar requests

4. **Cost Monitoring**
   - Set daily/monthly budgets
   - Alert on expensive calls (>$0.10)
   - Regular cost reports

5. **Token Optimization**
   - Compress prompts where possible
   - Use concise system messages
   - Implement prompt templates