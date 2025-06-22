"""
Planning & Execution Agent for Universal Agent System
=====================================================
Specialized agent for strategic planning, task decomposition, and execution orchestration
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import heapq
from collections import defaultdict, deque
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import pulp
import simpy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import ray
import dask
from dask import delayed, compute
from dask.distributed import Client as DaskClient
import pandas as pd
import logging
from abc import ABC, abstractmethod

# Import from core system
from core.base_agent import BaseAgent, AgentConfig, Task, Message, AgentRole
from core.memory import Memory, MemoryType
from core.reasoning import ReasoningStrategy

# ========== Planning-Specific Data Structures ==========

class PlanType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    DISTRIBUTED = "distributed"

class ExecutionStrategy(Enum):
    EAGER = "eager"
    LAZY = "lazy"
    SPECULATIVE = "speculative"
    CONSERVATIVE = "conservative"
    OPTIMISTIC = "optimistic"
    FAULT_TOLERANT = "fault_tolerant"

class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    RETRYING = "retrying"

@dataclass
class PlanNode:
    """Node in execution plan"""
    id: str
    task: Task
    dependencies: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    priority: float = 1.0
    estimated_duration: timedelta = timedelta(minutes=5)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionPlan:
    """Complete execution plan"""
    id: str
    name: str
    goal: str
    plan_type: PlanType
    nodes: Dict[str, PlanNode]
    execution_order: List[str]
    critical_path: List[str]
    estimated_duration: timedelta
    resource_allocation: Dict[str, Dict[str, float]]
    risk_assessment: Dict[str, Any]
    contingency_plans: Dict[str, 'ExecutionPlan']
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    quality_score: float = 0.0

@dataclass
class ExecutionContext:
    """Runtime execution context"""
    plan: ExecutionPlan
    current_state: Dict[str, Any]
    completed_tasks: Set[str]
    failed_tasks: Set[str]
    active_tasks: Dict[str, PlanNode]
    resource_usage: Dict[str, float]
    performance_metrics: Dict[str, Any]
    events: List[Dict[str, Any]]
    start_time: datetime
    checkpoints: List[Dict[str, Any]]

@dataclass
class WorkflowTemplate:
    """Reusable workflow template"""
    id: str
    name: str
    description: str
    category: str
    steps: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    requirements: Dict[str, Any]
    success_criteria: Dict[str, Any]
    typical_duration: timedelta
    success_rate: float

# ========== Planning & Execution Agent ==========

class PlanningExecutionAgent(BaseAgent):
    """Agent specialized in planning and executing complex tasks"""
    
    def __init__(self, agent_id: str, config: AgentConfig):
        super().__init__(agent_id, config)
        
        # Planning components
        self.task_decomposer = TaskDecomposer()
        self.plan_generator = PlanGenerator()
        self.plan_optimizer = PlanOptimizer()
        self.resource_scheduler = ResourceScheduler()
        self.risk_analyzer = RiskAnalyzer()
        
        # Execution components
        self.execution_engine = ExecutionEngine()
        self.workflow_orchestrator = WorkflowOrchestrator()
        self.monitor = ExecutionMonitor()
        self.recovery_manager = RecoveryManager()
        
        # Distributed execution support
        self._initialize_distributed_computing()
        
        # Template library
        self.workflow_templates = WorkflowTemplateLibrary()
        
        # Active executions
        self.active_plans: Dict[str, ExecutionContext] = {}
        self.execution_history: List[ExecutionContext] = []
        
        # Performance tracking
        self.metrics = {
            'plans_created': 0,
            'tasks_executed': 0,
            'success_rate': 0.0,
            'average_duration': timedelta(),
            'resource_efficiency': 0.0
        }
        
        # Initialize tools
        self._initialize_tools()
    
    def _initialize_distributed_computing(self):
        """Initialize distributed computing frameworks"""
        try:
            # Initialize Ray for distributed execution
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            self.ray_enabled = True
        except:
            self.ray_enabled = False
            self.logger.warning("Ray not available for distributed execution")
        
        try:
            # Initialize Dask for parallel computation
            self.dask_client = DaskClient(processes=False, threads_per_worker=4)
            self.dask_enabled = True
        except:
            self.dask_enabled = False
            self.logger.warning("Dask not available for parallel computation")
    
    def _initialize_tools(self):
        """Initialize planning and execution tools"""
        self.add_tool('create_plan', self.create_execution_plan)
        self.add_tool('execute_plan', self.execute_plan)
        self.add_tool('decompose_task', self.decompose_complex_task)
        self.add_tool('optimize_plan', self.optimize_execution_plan)
        self.add_tool('monitor_execution', self.monitor_execution)
        self.add_tool('analyze_risks', self.analyze_execution_risks)
        self.add_tool('schedule_resources', self.schedule_resources)
        self.add_tool('create_workflow', self.create_workflow_template)
        self.add_tool('simulate_execution', self.simulate_execution)
    
    async def process_task(self, task: Task) -> Any:
        """Process planning and execution tasks"""
        self.logger.info(f"Processing planning task: {task.type}")
        
        try:
            if task.type == "create_plan":
                return await self._create_plan_task(task)
            elif task.type == "execute_plan":
                return await self._execute_plan_task(task)
            elif task.type == "optimize_workflow":
                return await self._optimize_workflow_task(task)
            elif task.type == "manage_resources":
                return await self._manage_resources_task(task)
            elif task.type == "coordinate_agents":
                return await self._coordinate_agents_task(task)
            elif task.type == "monitor_progress":
                return await self._monitor_progress_task(task)
            elif task.type == "handle_failure":
                return await self._handle_failure_task(task)
            elif task.type == "evaluate_performance":
                return await self._evaluate_performance_task(task)
            else:
                return await self._general_planning_task(task)
                
        except Exception as e:
            self.logger.error(f"Error processing planning task: {e}")
            raise
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle planning-related messages"""
        content = message.content
        
        if isinstance(content, dict):
            message_type = content.get('type')
            
            if message_type == 'plan_request':
                plan = await self._handle_plan_request(content)
                return Message(
                    sender=self.id,
                    receiver=message.sender,
                    content={'plan': plan}
                )
            elif message_type == 'execution_status':
                status = await self._get_execution_status(content['plan_id'])
                return Message(
                    sender=self.id,
                    receiver=message.sender,
                    content={'status': status}
                )
            elif message_type == 'resource_availability':
                await self._update_resource_availability(content['resources'])
        
        return None
    
    # ========== Plan Creation ==========
    
    async def _create_plan_task(self, task: Task) -> ExecutionPlan:
        """Create execution plan for a goal"""
        goal = task.parameters.get('goal', '')
        constraints = task.parameters.get('constraints', {})
        strategy = task.parameters.get('strategy', ExecutionStrategy.BALANCED)
        
        # Use reasoning engine to understand the goal
        goal_analysis = await self.reasoning_engine.reason(
            problem=f"Create execution plan for: {goal}",
            context={
                'goal': goal,
                'constraints': constraints,
                'available_agents': self._get_available_agents(),
                'resources': self._get_available_resources()
            },
            strategy=ReasoningStrategy.TREE_OF_THOUGHT
        )
        
        # Decompose goal into tasks
        decomposed_tasks = await self.task_decomposer.decompose(
            goal=goal,
            analysis=goal_analysis,
            max_depth=3
        )
        
        # Generate initial plan
        initial_plan = await self.plan_generator.generate(
            tasks=decomposed_tasks,
            strategy=strategy,
            constraints=constraints
        )
        
        # Optimize plan
        optimized_plan = await self.plan_optimizer.optimize(
            plan=initial_plan,
            optimization_goals=['duration', 'resource_usage', 'success_probability']
        )
        
        # Analyze risks
        risk_assessment = await self.risk_analyzer.analyze(optimized_plan)
        optimized_plan.risk_assessment = risk_assessment
        
        # Create contingency plans for high-risk areas
        if risk_assessment['overall_risk'] > 0.3:
            contingency_plans = await self._create_contingency_plans(
                optimized_plan,
                risk_assessment
            )
            optimized_plan.contingency_plans = contingency_plans
        
        # Calculate quality score
        optimized_plan.quality_score = self._assess_plan_quality(optimized_plan)
        
        # Store plan
        self._store_plan(optimized_plan)
        
        # Update metrics
        self.metrics['plans_created'] += 1
        
        return optimized_plan
    
    async def create_execution_plan(
        self,
        goal: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """Public method to create execution plan"""
        task = Task(
            type="create_plan",
            parameters={
                'goal': goal,
                'constraints': constraints or {}
            }
        )
        return await self._create_plan_task(task)
    
    async def _create_contingency_plans(
        self,
        plan: ExecutionPlan,
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, ExecutionPlan]:
        """Create contingency plans for identified risks"""
        contingency_plans = {}
        
        # Identify high-risk nodes
        high_risk_nodes = [
            node_id for node_id, risk in risk_assessment['node_risks'].items()
            if risk['level'] > 0.5
        ]
        
        for node_id in high_risk_nodes:
            node = plan.nodes[node_id]
            risk_type = risk_assessment['node_risks'][node_id]['type']
            
            # Create contingency based on risk type
            if risk_type == 'failure':
                contingency = await self._create_failure_contingency(plan, node)
            elif risk_type == 'delay':
                contingency = await self._create_delay_contingency(plan, node)
            elif risk_type == 'resource':
                contingency = await self._create_resource_contingency(plan, node)
            else:
                contingency = await self._create_generic_contingency(plan, node)
            
            contingency_plans[f"{node_id}_{risk_type}"] = contingency
        
        return contingency_plans
    
    # ========== Task Decomposition ==========
    
    async def decompose_complex_task(
        self,
        task: str,
        max_subtasks: int = 10
    ) -> List[Dict[str, Any]]:
        """Decompose complex task into subtasks"""
        decomposition = await self.task_decomposer.decompose(
            goal=task,
            analysis=None,
            max_depth=2
        )
        
        # Convert to simplified format
        subtasks = []
        for node in decomposition:
            subtasks.append({
                'id': node.id,
                'description': node.task.description,
                'dependencies': node.dependencies,
                'estimated_duration': node.estimated_duration.total_seconds() / 60,  # minutes
                'priority': node.priority
            })
        
        return subtasks[:max_subtasks]
    
    # ========== Plan Execution ==========
    
    async def _execute_plan_task(self, task: Task) -> Dict[str, Any]:
        """Execute a plan"""
        plan_id = task.parameters.get('plan_id')
        execution_strategy = task.parameters.get('strategy', ExecutionStrategy.BALANCED)
        
        # Retrieve plan
        plan = self._get_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        # Create execution context
        context = ExecutionContext(
            plan=plan,
            current_state={},
            completed_tasks=set(),
            failed_tasks=set(),
            active_tasks={},
            resource_usage={},
            performance_metrics={},
            events=[],
            start_time=datetime.now(),
            checkpoints=[]
        )
        
        # Store active execution
        self.active_plans[plan.id] = context
        
        try:
            # Execute plan based on type
            if plan.plan_type == PlanType.SEQUENTIAL:
                result = await self._execute_sequential_plan(context)
            elif plan.plan_type == PlanType.PARALLEL:
                result = await self._execute_parallel_plan(context)
            elif plan.plan_type == PlanType.HIERARCHICAL:
                result = await self._execute_hierarchical_plan(context)
            elif plan.plan_type == PlanType.ADAPTIVE:
                result = await self._execute_adaptive_plan(context)
            elif plan.plan_type == PlanType.DISTRIBUTED:
                result = await self._execute_distributed_plan(context)
            else:
                result = await self._execute_generic_plan(context)
            
            # Update metrics
            self._update_execution_metrics(context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}")
            context.events.append({
                'type': 'execution_failed',
                'error': str(e),
                'timestamp': datetime.now()
            })
            raise
        finally:
            # Move to history
            self.execution_history.append(context)
            del self.active_plans[plan.id]
    
    async def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """Public method to execute plan"""
        task = Task(
            type="execute_plan",
            parameters={'plan_id': plan_id}
        )
        return await self._execute_plan_task(task)
    
    async def _execute_sequential_plan(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute plan sequentially"""
        results = {}
        
        for node_id in context.plan.execution_order:
            node = context.plan.nodes[node_id]
            
            # Check dependencies
            if not self._dependencies_satisfied(node, context):
                context.events.append({
                    'type': 'dependency_not_met',
                    'node': node_id,
                    'timestamp': datetime.now()
                })
                continue
            
            # Execute node
            try:
                result = await self._execute_node(node, context)
                results[node_id] = result
                context.completed_tasks.add(node_id)
                
                # Update state
                context.current_state[node_id] = result
                
            except Exception as e:
                self.logger.error(f"Node {node_id} execution failed: {e}")
                context.failed_tasks.add(node_id)
                
                # Check if we should continue or fail
                if node.constraints.get('critical', False):
                    raise
                
                # Try recovery
                recovery_result = await self._attempt_recovery(node, context, e)
                if recovery_result:
                    results[node_id] = recovery_result
                    context.completed_tasks.add(node_id)
        
        return {
            'status': 'completed' if not context.failed_tasks else 'partial',
            'results': results,
            'completed': list(context.completed_tasks),
            'failed': list(context.failed_tasks),
            'duration': datetime.now() - context.start_time
        }
    
    async def _execute_parallel_plan(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute plan with parallel tasks"""
        results = {}
        
        # Group tasks by dependency level
        levels = self._calculate_dependency_levels(context.plan)
        
        for level in sorted(levels.keys()):
            level_tasks = levels[level]
            
            # Execute all tasks at this level in parallel
            tasks = []
            for node_id in level_tasks:
                node = context.plan.nodes[node_id]
                if self._dependencies_satisfied(node, context):
                    tasks.append(self._execute_node_async(node, context))
            
            # Wait for all tasks at this level
            if tasks:
                level_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, node_id in enumerate(level_tasks):
                    if isinstance(level_results[i], Exception):
                        context.failed_tasks.add(node_id)
                        self.logger.error(f"Node {node_id} failed: {level_results[i]}")
                    else:
                        results[node_id] = level_results[i]
                        context.completed_tasks.add(node_id)
                        context.current_state[node_id] = level_results[i]
        
        return {
            'status': 'completed' if not context.failed_tasks else 'partial',
            'results': results,
            'completed': list(context.completed_tasks),
            'failed': list(context.failed_tasks),
            'duration': datetime.now() - context.start_time,
            'parallelism_achieved': self._calculate_parallelism_efficiency(context)
        }
    
    async def _execute_distributed_plan(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute plan using distributed computing"""
        if not self.ray_enabled:
            # Fallback to parallel execution
            return await self._execute_parallel_plan(context)
        
        results = {}
        
        # Define Ray remote function
        @ray.remote
        def execute_remote_task(task_data):
            # This would execute the task on a remote worker
            # Simplified for example
            return {'task_id': task_data['id'], 'result': 'completed'}
        
        # Submit tasks to Ray
        futures = []
        node_mapping = {}
        
        for node_id in context.plan.execution_order:
            node = context.plan.nodes[node_id]
            if self._dependencies_satisfied(node, context):
                # Serialize task data
                task_data = {
                    'id': node_id,
                    'task': node.task.type,
                    'parameters': node.task.parameters
                }
                
                future = execute_remote_task.remote(task_data)
                futures.append(future)
                node_mapping[future] = node_id
        
        # Process results as they complete
        while futures:
            ready, futures = ray.wait(futures, num_returns=1)
            
            for future in ready:
                try:
                    result = ray.get(future)
                    node_id = node_mapping[future]
                    results[node_id] = result
                    context.completed_tasks.add(node_id)
                except Exception as e:
                    node_id = node_mapping[future]
                    context.failed_tasks.add(node_id)
                    self.logger.error(f"Distributed execution failed for {node_id}: {e}")
        
        return {
            'status': 'completed' if not context.failed_tasks else 'partial',
            'results': results,
            'completed': list(context.completed_tasks),
            'failed': list(context.failed_tasks),
            'duration': datetime.now() - context.start_time,
            'execution_mode': 'distributed'
        }
    
    async def _execute_adaptive_plan(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute plan with adaptive behavior"""
        results = {}
        
        # Initialize adaptation parameters
        adaptation_state = {
            'performance_threshold': 0.8,
            'resource_threshold': 0.9,
            'failure_threshold': 0.2
        }
        
        # Execute with adaptation
        for node_id in context.plan.execution_order:
            node = context.plan.nodes[node_id]
            
            # Check if adaptation needed
            if self._needs_adaptation(context, adaptation_state):
                # Adapt the plan
                adapted_plan = await self._adapt_plan(context, node_id)
                context.plan = adapted_plan
                
                # Log adaptation
                context.events.append({
                    'type': 'plan_adapted',
                    'reason': self._get_adaptation_reason(context, adaptation_state),
                    'timestamp': datetime.now()
                })
            
            # Execute with monitoring
            start_metrics = self._capture_metrics(context)
            
            try:
                result = await self._execute_node(node, context)
                results[node_id] = result
                context.completed_tasks.add(node_id)
                
                # Learn from execution
                end_metrics = self._capture_metrics(context)
                self._update_adaptation_model(start_metrics, end_metrics, True)
                
            except Exception as e:
                context.failed_tasks.add(node_id)
                self._update_adaptation_model(start_metrics, None, False)
                
                # Adapt based on failure
                if len(context.failed_tasks) / len(context.plan.nodes) > adaptation_state['failure_threshold']:
                    recovery_plan = await self._create_recovery_plan(context)
                    return await self._execute_recovery_plan(recovery_plan, context)
        
        return {
            'status': 'completed' if not context.failed_tasks else 'partial',
            'results': results,
            'adaptations_made': len([e for e in context.events if e['type'] == 'plan_adapted']),
            'final_performance': self._calculate_performance_score(context)
        }
    
    # ========== Node Execution ==========
    
    async def _execute_node(self, node: PlanNode, context: ExecutionContext) -> Any:
        """Execute a single plan node"""
        node.status = TaskStatus.RUNNING
        node.start_time = datetime.now()
        
        # Update active tasks
        context.active_tasks[node.id] = node
        
        try:
            # Allocate resources
            allocated_resources = await self._allocate_resources(node, context)
            
            # Get assigned agent or select one
            if not node.assigned_agent:
                node.assigned_agent = await self._select_agent_for_task(node.task)
            
            # Execute task
            result = await self._dispatch_task_to_agent(
                task=node.task,
                agent_id=node.assigned_agent,
                context=context
            )
            
            # Update node
            node.status = TaskStatus.COMPLETED
            node.end_time = datetime.now()
            node.result = result
            
            # Release resources
            await self._release_resources(allocated_resources, context)
            
            # Record event
            context.events.append({
                'type': 'node_completed',
                'node': node.id,
                'duration': (node.end_time - node.start_time).total_seconds(),
                'timestamp': datetime.now()
            })
            
            return result
            
        except Exception as e:
            node.status = TaskStatus.FAILED
            node.error = str(e)
            node.end_time = datetime.now()
            
            # Record failure event
            context.events.append({
                'type': 'node_failed',
                'node': node.id,
                'error': str(e),
                'timestamp': datetime.now()
            })
            
            raise
        finally:
            # Remove from active tasks
            if node.id in context.active_tasks:
                del context.active_tasks[node.id]
    
    async def _execute_node_async(self, node: PlanNode, context: ExecutionContext) -> Any:
        """Execute node asynchronously for parallel execution"""
        return await self._execute_node(node, context)
    
    # ========== Resource Management ==========
    
    async def _allocate_resources(
        self,
        node: PlanNode,
        context: ExecutionContext
    ) -> Dict[str, float]:
        """Allocate resources for node execution"""
        allocated = {}
        
        for resource, amount in node.resource_requirements.items():
            available = self._get_available_resource(resource, context)
            
            if available >= amount:
                # Allocate resource
                context.resource_usage[resource] = context.resource_usage.get(resource, 0) + amount
                allocated[resource] = amount
            else:
                # Resource shortage
                raise ResourceError(f"Insufficient {resource}: need {amount}, have {available}")
        
        return allocated
    
    async def _release_resources(
        self,
        allocated: Dict[str, float],
        context: ExecutionContext
    ):
        """Release allocated resources"""
        for resource, amount in allocated.items():
            context.resource_usage[resource] -= amount
    
    def _get_available_resource(self, resource: str, context: ExecutionContext) -> float:
        """Get available amount of a resource"""
        total = self._get_total_resource(resource)
        used = context.resource_usage.get(resource, 0)
        return total - used
    
    def _get_total_resource(self, resource: str) -> float:
        """Get total amount of a resource"""
        # This would integrate with actual resource tracking
        resource_limits = {
            'cpu': 100.0,  # percentage
            'memory': 32768.0,  # MB
            'gpu': 4.0,  # number of GPUs
            'api_calls': 1000.0,  # per minute
            'budget': 10000.0  # dollars
        }
        return resource_limits.get(resource, float('inf'))
    
    # ========== Plan Optimization ==========
    
    async def optimize_execution_plan(
        self,
        plan_id: str,
        optimization_goals: List[str]
    ) -> ExecutionPlan:
        """Optimize an existing plan"""
        plan = self._get_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        optimized = await self.plan_optimizer.optimize(
            plan=plan,
            optimization_goals=optimization_goals
        )
        
        # Store optimized plan
        self._store_plan(optimized)
        
        return optimized
    
    # ========== Monitoring ==========
    
    async def monitor_execution(self, plan_id: str) -> Dict[str, Any]:
        """Monitor ongoing execution"""
        if plan_id not in self.active_plans:
            return {'status': 'not_running'}
        
        context = self.active_plans[plan_id]
        
        # Calculate progress
        total_nodes = len(context.plan.nodes)
        completed = len(context.completed_tasks)
        failed = len(context.failed_tasks)
        active = len(context.active_tasks)
        
        progress = (completed + failed) / total_nodes if total_nodes > 0 else 0
        
        # Estimate remaining time
        elapsed = datetime.now() - context.start_time
        if completed > 0:
            avg_task_time = elapsed / completed
            remaining_tasks = total_nodes - completed - failed
            estimated_remaining = avg_task_time * remaining_tasks
        else:
            estimated_remaining = context.plan.estimated_duration - elapsed
        
        # Get current resource usage
        resource_usage = {
            resource: f"{used}/{self._get_total_resource(resource):.1f}"
            for resource, used in context.resource_usage.items()
        }
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(context)
        
        return {
            'status': 'running',
            'progress': progress,
            'completed_tasks': completed,
            'failed_tasks': failed,
            'active_tasks': active,
            'total_tasks': total_nodes,
            'elapsed_time': elapsed,
            'estimated_remaining': estimated_remaining,
            'resource_usage': resource_usage,
            'bottlenecks': bottlenecks,
            'recent_events': context.events[-10:],  # Last 10 events
            'performance_score': self._calculate_performance_score(context)
        }
    
    def _identify_bottlenecks(self, context: ExecutionContext) -> List[Dict[str, Any]]:
        """Identify execution bottlenecks"""
        bottlenecks = []
        
        # Resource bottlenecks
        for resource, usage in context.resource_usage.items():
            total = self._get_total_resource(resource)
            if usage / total > 0.9:
                bottlenecks.append({
                    'type': 'resource',
                    'resource': resource,
                    'usage_percent': (usage / total) * 100
                })
        
        # Task bottlenecks (tasks taking longer than expected)
        for node_id, node in context.active_tasks.items():
            if node.start_time:
                actual_duration = datetime.now() - node.start_time
                if actual_duration > node.estimated_duration * 1.5:
                    bottlenecks.append({
                        'type': 'slow_task',
                        'task': node_id,
                        'expected_duration': node.estimated_duration,
                        'actual_duration': actual_duration
                    })
        
        # Dependency bottlenecks
        blocked_count = sum(
            1 for node in context.plan.nodes.values()
            if node.status == TaskStatus.BLOCKED
        )
        if blocked_count > len(context.plan.nodes) * 0.2:
            bottlenecks.append({
                'type': 'dependencies',
                'blocked_tasks': blocked_count
            })
        
        return bottlenecks
    
    # ========== Risk Analysis ==========
    
    async def analyze_execution_risks(self, plan_id: str) -> Dict[str, Any]:
        """Analyze risks in execution plan"""
        plan = self._get_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        return await self.risk_analyzer.analyze(plan)
    
    # ========== Recovery Management ==========
    
    async def _attempt_recovery(
        self,
        node: PlanNode,
        context: ExecutionContext,
        error: Exception
    ) -> Optional[Any]:
        """Attempt to recover from node failure"""
        # Check retry policy
        if node.retry_count < node.constraints.get('max_retries', 3):
            node.retry_count += 1
            
            # Wait before retry
            retry_delay = 2 ** node.retry_count  # Exponential backoff
            await asyncio.sleep(retry_delay)
            
            # Retry execution
            try:
                node.status = TaskStatus.RETRYING
                result = await self._execute_node(node, context)
                return result
            except Exception as retry_error:
                self.logger.error(f"Retry {node.retry_count} failed: {retry_error}")
        
        # Check for contingency plan
        if context.plan.contingency_plans:
            for contingency_id, contingency_plan in context.plan.contingency_plans.items():
                if node.id in contingency_id:
                    # Execute contingency plan
                    return await self._execute_contingency(contingency_plan, context)
        
        return None
    
    async def _execute_contingency(
        self,
        contingency_plan: ExecutionPlan,
        original_context: ExecutionContext
    ) -> Any:
        """Execute contingency plan"""
        # Create new context for contingency
        contingency_context = ExecutionContext(
            plan=contingency_plan,
            current_state=original_context.current_state.copy(),
            completed_tasks=set(),
            failed_tasks=set(),
            active_tasks={},
            resource_usage=original_context.resource_usage.copy(),
            performance_metrics={},
            events=[],
            start_time=datetime.now(),
            checkpoints=[]
        )
        
        # Execute contingency plan
        result = await self._execute_plan_task(
            Task(
                type="execute_plan",
                parameters={'plan_id': contingency_plan.id}
            )
        )
        
        # Merge results back
        original_context.events.extend(contingency_context.events)
        
        return result
    
    # ========== Helper Methods ==========
    
    def _dependencies_satisfied(self, node: PlanNode, context: ExecutionContext) -> bool:
        """Check if node dependencies are satisfied"""
        for dep_id in node.dependencies:
            if dep_id not in context.completed_tasks:
                return False
        return True
    
    def _calculate_dependency_levels(self, plan: ExecutionPlan) -> Dict[int, List[str]]:
        """Calculate dependency levels for parallel execution"""
        levels = defaultdict(list)
        visited = set()
        
        # Build dependency graph
        graph = nx.DiGraph()
        for node_id, node in plan.nodes.items():
            graph.add_node(node_id)
            for dep in node.dependencies:
                graph.add_edge(dep, node_id)
        
        # Calculate levels using topological sort
        for node in nx.topological_sort(graph):
            if not plan.nodes[node].dependencies:
                levels[0].append(node)
            else:
                max_dep_level = max(
                    self._get_node_level(dep, levels)
                    for dep in plan.nodes[node].dependencies
                )
                levels[max_dep_level + 1].append(node)
        
        return dict(levels)
    
    def _get_node_level(self, node_id: str, levels: Dict[int, List[str]]) -> int:
        """Get level of a node"""
        for level, nodes in levels.items():
            if node_id in nodes:
                return level
        return -1
    
    async def _select_agent_for_task(self, task: Task) -> str:
        """Select best agent for task"""
        # This would integrate with agent manager to select appropriate agent
        # For now, return a placeholder
        return "agent_1"
    
    async def _dispatch_task_to_agent(
        self,
        task: Task,
        agent_id: str,
        context: ExecutionContext
    ) -> Any:
        """Dispatch task to specific agent"""
        # This would integrate with agent manager
        # For now, simulate execution
        await asyncio.sleep(0.1)  # Simulate work
        return {"status": "completed", "result": f"Task {task.id} completed by {agent_id}"}
    
    def _store_plan(self, plan: ExecutionPlan):
        """Store plan in memory"""
        self.memory.store(
            key=f"plan_{plan.id}",
            value=plan,
            memory_type=MemoryType.LONG_TERM,
            importance=0.8
        )
    
    def _get_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Retrieve plan from memory"""
        return self.memory.retrieve(f"plan_{plan_id}", MemoryType.LONG_TERM)
    
    def _assess_plan_quality(self, plan: ExecutionPlan) -> float:
        """Assess quality of execution plan"""
        quality_score = 0.0
        
        # Efficiency (30%)
        if plan.critical_path:
            critical_path_ratio = len(plan.critical_path) / len(plan.nodes)
            efficiency = 1.0 - critical_path_ratio
            quality_score += efficiency * 0.3
        
        # Resource optimization (25%)
        total_resource_usage = sum(
            sum(alloc.values())
            for alloc in plan.resource_allocation.values()
        )
        if total_resource_usage > 0:
            resource_efficiency = 1.0 / (1.0 + total_resource_usage / 1000)
            quality_score += resource_efficiency * 0.25
        
        # Risk management (25%)
        if plan.risk_assessment:
            risk_score = 1.0 - plan.risk_assessment.get('overall_risk', 0.5)
            quality_score += risk_score * 0.25
        
        # Contingency coverage (20%)
        contingency_coverage = len(plan.contingency_plans) / max(1, len(plan.nodes) * 0.2)
        quality_score += min(1.0, contingency_coverage) * 0.2
        
        return min(1.0, quality_score)
    
    def _calculate_performance_score(self, context: ExecutionContext) -> float:
        """Calculate execution performance score"""
        if not context.plan.nodes:
            return 0.0
        
        # Success rate (40%)
        total_tasks = len(context.plan.nodes)
        success_rate = len(context.completed_tasks) / total_tasks
        
        # Time efficiency (30%)
        if context.plan.estimated_duration.total_seconds() > 0:
            actual_duration = (datetime.now() - context.start_time).total_seconds()
            estimated_duration = context.plan.estimated_duration.total_seconds()
            time_efficiency = min(1.0, estimated_duration / actual_duration)
        else:
            time_efficiency = 0.5
        
        # Resource efficiency (30%)
        resource_efficiency = 1.0
        for resource, usage in context.resource_usage.items():
            total = self._get_total_resource(resource)
            if total > 0:
                resource_efficiency *= (1.0 - usage / total)
        
        performance = (
            success_rate * 0.4 +
            time_efficiency * 0.3 +
            resource_efficiency * 0.3
        )
        
        return min(1.0, performance)
    
    def _update_execution_metrics(self, context: ExecutionContext):
        """Update agent metrics based on execution"""
        self.metrics['tasks_executed'] += len(context.completed_tasks)
        
        # Update success rate
        total_attempts = self.metrics['tasks_executed']
        successful = len(context.completed_tasks)
        self.metrics['success_rate'] = successful / total_attempts if total_attempts > 0 else 0
        
        # Update average duration
        duration = datetime.now() - context.start_time
        # Simple moving average
        self.metrics['average_duration'] = (
            self.metrics['average_duration'] * 0.9 +
            duration * 0.1
        )

# ========== Task Decomposer ==========

class TaskDecomposer:
    """Decompose complex tasks into subtasks"""
    
    async def decompose(
        self,
        goal: str,
        analysis: Optional[Dict[str, Any]],
        max_depth: int = 3
    ) -> List[PlanNode]:
        """Decompose goal into executable tasks"""
        nodes = []
        
        # Create root node
        root_node = PlanNode(
            id=str(uuid.uuid4()),
            task=Task(
                type="root",
                description=goal,
                parameters={}
            ),
            priority=1.0
        )
        
        # Decompose recursively
        await self._decompose_recursive(
            root_node,
            nodes,
            current_depth=0,
            max_depth=max_depth,
            analysis=analysis
        )
        
        return nodes
    
    async def _decompose_recursive(
        self,
        parent: PlanNode,
        nodes: List[PlanNode],
        current_depth: int,
        max_depth: int,
        analysis: Optional[Dict[str, Any]]
    ):
        """Recursively decompose tasks"""
        if current_depth >= max_depth:
            nodes.append(parent)
            return
        
        # Determine subtasks based on task type
        subtasks = await self._identify_subtasks(parent.task, analysis)
        
        if not subtasks:
            nodes.append(parent)
            return
        
        # Create nodes for subtasks
        for subtask in subtasks:
            child_node = PlanNode(
                id=str(uuid.uuid4()),
                task=subtask,
                dependencies=[parent.id] if current_depth > 0 else [],
                priority=parent.priority * 0.9,
                estimated_duration=self._estimate_duration(subtask)
            )
            
            parent.children.append(child_node.id)
            
            # Recurse
            await self._decompose_recursive(
                child_node,
                nodes,
                current_depth + 1,
                max_depth,
                analysis
            )
    
    async def _identify_subtasks(
        self,
        task: Task,
        analysis: Optional[Dict[str, Any]]
    ) -> List[Task]:
        """Identify subtasks for a given task"""
        subtasks = []
        
        # Task-specific decomposition
        if task.type == "root" or task.type == "complex":
            # Use templates for common patterns
            if "build" in task.description.lower():
                subtasks = self._decompose_build_task(task)
            elif "analyze" in task.description.lower():
                subtasks = self._decompose_analysis_task(task)
            elif "deploy" in task.description.lower():
                subtasks = self._decompose_deployment_task(task)
            else:
                subtasks = self._decompose_generic_task(task)
        
        return subtasks
    
    def _decompose_build_task(self, task: Task) -> List[Task]:
        """Decompose build-related tasks"""
        return [
            Task(
                type="design",
                description="Design architecture and components",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="implement",
                description="Implement core functionality",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="test",
                description="Test implementation",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="integrate",
                description="Integrate components",
                parameters={"parent_task": task.id}
            )
        ]
    
    def _decompose_analysis_task(self, task: Task) -> List[Task]:
        """Decompose analysis tasks"""
        return [
            Task(
                type="collect_data",
                description="Collect relevant data",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="process_data",
                description="Process and clean data",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="analyze",
                description="Perform analysis",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="visualize",
                description="Create visualizations",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="report",
                description="Generate report",
                parameters={"parent_task": task.id}
            )
        ]
    
    def _decompose_deployment_task(self, task: Task) -> List[Task]:
        """Decompose deployment tasks"""
        return [
            Task(
                type="prepare_environment",
                description="Prepare deployment environment",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="build_artifacts",
                description="Build deployment artifacts",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="deploy",
                description="Deploy to target environment",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="verify",
                description="Verify deployment",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="monitor",
                description="Monitor deployment",
                parameters={"parent_task": task.id}
            )
        ]
    
    def _decompose_generic_task(self, task: Task) -> List[Task]:
        """Generic task decomposition"""
        return [
            Task(
                type="prepare",
                description=f"Prepare for {task.description}",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="execute",
                description=f"Execute {task.description}",
                parameters={"parent_task": task.id}
            ),
            Task(
                type="verify",
                description=f"Verify results of {task.description}",
                parameters={"parent_task": task.id}
            )
        ]
    
    def _estimate_duration(self, task: Task) -> timedelta:
        """Estimate task duration"""
        # Simple estimation based on task type
        duration_map = {
            "design": timedelta(hours=4),
            "implement": timedelta(hours=8),
            "test": timedelta(hours=2),
            "analyze": timedelta(hours=3),
            "deploy": timedelta(hours=1),
            "default": timedelta(hours=1)
        }
        
        return duration_map.get(task.type, duration_map["default"])

# ========== Plan Generator ==========

class PlanGenerator:
    """Generate execution plans from tasks"""
    
    async def generate(
        self,
        tasks: List[PlanNode],
        strategy: ExecutionStrategy,
        constraints: Dict[str, Any]
    ) -> ExecutionPlan:
        """Generate execution plan"""
        plan_id = str(uuid.uuid4())
        
        # Determine plan type based on strategy
        plan_type = self._determine_plan_type(tasks, strategy)
        
        # Create node dictionary
        nodes = {node.id: node for node in tasks}
        
        # Calculate execution order
        execution_order = self._calculate_execution_order(nodes, plan_type)
        
        # Identify critical path
        critical_path = self._find_critical_path(nodes)
        
        # Allocate resources
        resource_allocation = await self._allocate_resources_to_plan(
            nodes,
            constraints.get('resources', {})
        )
        
        # Estimate total duration
        estimated_duration = self._estimate_plan_duration(nodes, critical_path)
        
        # Create plan
        plan = ExecutionPlan(
            id=plan_id,
            name=f"Plan for {tasks[0].task.description if tasks else 'Unknown'}",
            goal=tasks[0].task.description if tasks else "",
            plan_type=plan_type,
            nodes=nodes,
            execution_order=execution_order,
            critical_path=critical_path,
            estimated_duration=estimated_duration,
            resource_allocation=resource_allocation,
            risk_assessment={},
            contingency_plans={}
        )
        
        return plan
    
    def _determine_plan_type(
        self,
        tasks: List[PlanNode],
        strategy: ExecutionStrategy
    ) -> PlanType:
        """Determine appropriate plan type"""
        # Check if tasks can be parallelized
        has_parallel_potential = self._has_parallel_potential(tasks)
        
        # Check if tasks are hierarchical
        has_hierarchy = any(node.children for node in tasks)
        
        # Determine based on characteristics
        if has_hierarchy:
            return PlanType.HIERARCHICAL
        elif has_parallel_potential and strategy != ExecutionStrategy.CONSERVATIVE:
            return PlanType.PARALLEL
        elif strategy == ExecutionStrategy.ADAPTIVE:
            return PlanType.ADAPTIVE
        else:
            return PlanType.SEQUENTIAL
    
    def _has_parallel_potential(self, tasks: List[PlanNode]) -> bool:
        """Check if tasks can be executed in parallel"""
        # Build dependency graph
        dep_count = defaultdict(int)
        for node in tasks:
            for dep in node.dependencies:
                dep_count[node.id] += 1
        
        # If multiple tasks have no dependencies, they can run in parallel
        independent_tasks = sum(1 for node in tasks if dep_count[node.id] == 0)
        return independent_tasks > 1
    
    def _calculate_execution_order(
        self,
        nodes: Dict[str, PlanNode],
        plan_type: PlanType
    ) -> List[str]:
        """Calculate execution order based on plan type"""
        if plan_type == PlanType.PARALLEL:
            # Use topological sort
            return self._topological_sort(nodes)
        else:
            # Use priority-based ordering
            return self._priority_sort(nodes)
    
    def _topological_sort(self, nodes: Dict[str, PlanNode]) -> List[str]:
        """Perform topological sort on nodes"""
        # Build graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for node_id, node in nodes.items():
            for dep in node.dependencies:
                graph[dep].append(node_id)
                in_degree[node_id] += 1
        
        # Find nodes with no dependencies
        queue = [node_id for node_id in nodes if in_degree[node_id] == 0]
        result = []
        
        while queue:
            # Sort by priority
            queue.sort(key=lambda x: nodes[x].priority, reverse=True)
            node_id = queue.pop(0)
            result.append(node_id)
            
            # Update dependencies
            for child in graph[node_id]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result
    
    def _priority_sort(self, nodes: Dict[str, PlanNode]) -> List[str]:
        """Sort nodes by priority and dependencies"""
        # Simple priority-based sort
        sorted_nodes = sorted(
            nodes.items(),
            key=lambda x: (len(x[1].dependencies), -x[1].priority)
        )
        return [node_id for node_id, _ in sorted_nodes]
    
    def _find_critical_path(self, nodes: Dict[str, PlanNode]) -> List[str]:
        """Find critical path through plan"""
        # Build directed graph
        G = nx.DiGraph()
        
        for node_id, node in nodes.items():
            G.add_node(node_id, duration=node.estimated_duration.total_seconds())
            for dep in node.dependencies:
                if dep in nodes:
                    G.add_edge(dep, node_id)
        
        # Add virtual start and end nodes
        G.add_node('start', duration=0)
        G.add_node('end', duration=0)
        
        # Connect start to nodes with no dependencies
        for node_id in nodes:
            if not nodes[node_id].dependencies:
                G.add_edge('start', node_id)
        
        # Connect nodes with no children to end
        for node_id in nodes:
            if not any(node_id in nodes[n].dependencies for n in nodes):
                G.add_edge(node_id, 'end')
        
        # Find longest path (critical path)
        try:
            critical_path = nx.dag_longest_path(G, weight='duration')
            # Remove virtual nodes
            return [n for n in critical_path if n not in ['start', 'end']]
        except:
            return list(nodes.keys())
    
    async def _allocate_resources_to_plan(
        self,
        nodes: Dict[str, PlanNode],
        available_resources: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Allocate resources to plan nodes"""
        allocation = {}
        
        for node_id, node in nodes.items():
            node_allocation = {}
            
            for resource, required in node.resource_requirements.items():
                available = available_resources.get(resource, float('inf'))
                allocated = min(required, available)
                node_allocation[resource] = allocated
            
            allocation[node_id] = node_allocation
        
        return allocation
    
    def _estimate_plan_duration(
        self,
        nodes: Dict[str, PlanNode],
        critical_path: List[str]
    ) -> timedelta:
        """Estimate total plan duration"""
        if critical_path:
            # Sum durations along critical path
            total_seconds = sum(
                nodes[node_id].estimated_duration.total_seconds()
                for node_id in critical_path
                if node_id in nodes
            )
            return timedelta(seconds=total_seconds)
        else:
            # Fallback to sum of all tasks
            total_seconds = sum(
                node.estimated_duration.total_seconds()
                for node in nodes.values()
            )
            return timedelta(seconds=total_seconds)

# ========== Plan Optimizer ==========

class PlanOptimizer:
    """Optimize execution plans"""
    
    async def optimize(
        self,
        plan: ExecutionPlan,
        optimization_goals: List[str]
    ) -> ExecutionPlan:
        """Optimize plan based on goals"""
        optimized_plan = self._copy_plan(plan)
        
        for goal in optimization_goals:
            if goal == 'duration':
                optimized_plan = await self._optimize_duration(optimized_plan)
            elif goal == 'resource_usage':
                optimized_plan = await self._optimize_resources(optimized_plan)
            elif goal == 'success_probability':
                optimized_plan = await self._optimize_success(optimized_plan)
            elif goal == 'cost':
                optimized_plan = await self._optimize_cost(optimized_plan)
        
        # Recalculate metrics
        optimized_plan.estimated_duration = self._recalculate_duration(optimized_plan)
        optimized_plan.quality_score = self._recalculate_quality(optimized_plan)
        
        return optimized_plan
    
    async def _optimize_duration(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize for minimal duration"""
        # Identify parallelization opportunities
        parallel_groups = self._identify_parallel_groups(plan)
        
        # Reorder execution for maximum parallelism
        if parallel_groups:
            plan.execution_order = self._reorder_for_parallelism(plan, parallel_groups)
            plan.plan_type = PlanType.PARALLEL
        
        # Optimize resource allocation for speed
        plan.resource_allocation = self._allocate_for_speed(plan)
        
        return plan
    
    async def _optimize_resources(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize for minimal resource usage"""
        # Use linear programming for resource optimization
        if self._can_use_optimization(plan):
            optimized_allocation = self._linear_programming_optimization(plan)
            plan.resource_allocation = optimized_allocation
        
        # Reorder to minimize peak resource usage
        plan.execution_order = self._reorder_for_resources(plan)
        
        return plan
    
    async def _optimize_success(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize for maximum success probability"""
        # Add redundancy for critical tasks
        critical_nodes = self._identify_critical_nodes(plan)
        
        for node_id in critical_nodes:
            # Add retry policy
            plan.nodes[node_id].constraints['max_retries'] = 5
            plan.nodes[node_id].constraints['critical'] = True
        
        # Reorder to fail fast
        plan.execution_order = self._reorder_fail_fast(plan)
        
        return plan
    
    async def _optimize_cost(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize for minimal cost"""
        # Calculate cost for each node
        node_costs = self._calculate_node_costs(plan)
        
        # Find cost-efficient execution order
        plan.execution_order = self._optimize_cost_order(plan, node_costs)
        
        # Adjust resource allocation for cost
        plan.resource_allocation = self._allocate_for_cost(plan, node_costs)
        
        return plan
    
    def _identify_parallel_groups(self, plan: ExecutionPlan) -> List[Set[str]]:
        """Identify groups of tasks that can run in parallel"""
        groups = []
        processed = set()
        
        # Build dependency graph
        dep_graph = defaultdict(set)
        reverse_deps = defaultdict(set)
        
        for node_id, node in plan.nodes.items():
            for dep in node.dependencies:
                dep_graph[node_id].add(dep)
                reverse_deps[dep].add(node_id)
        
        # Find independent task groups
        for node_id in plan.nodes:
            if node_id not in processed:
                # Find all tasks that can run parallel to this one
                parallel_group = {node_id}
                
                for other_id in plan.nodes:
                    if other_id != node_id and other_id not in processed:
                        # Check if they have conflicting dependencies
                        if not (dep_graph[node_id] & dep_graph[other_id]):
                            parallel_group.add(other_id)
                
                if len(parallel_group) > 1:
                    groups.append(parallel_group)
                    processed.update(parallel_group)
        
        return groups
    
    def _linear_programming_optimization(self, plan: ExecutionPlan) -> Dict[str, Dict[str, float]]:
        """Use linear programming for resource optimization"""
        # Create optimization problem
        prob = pulp.LpProblem("Resource_Optimization", pulp.LpMinimize)
        
        # Decision variables
        allocations = {}
        for node_id in plan.nodes:
            for resource in ['cpu', 'memory', 'gpu']:
                var_name = f"alloc_{node_id}_{resource}"
                allocations[var_name] = pulp.LpVariable(
                    var_name,
                    lowBound=0,
                    cat='Continuous'
                )
        
        # Objective: minimize total resource usage
        prob += pulp.lpSum(allocations.values())
        
        # Constraints: meet minimum requirements
        for node_id, node in plan.nodes.items():
            for resource, required in node.resource_requirements.items():
                var_name = f"alloc_{node_id}_{resource}"
                if var_name in allocations:
                    prob += allocations[var_name] >= required
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        result = defaultdict(dict)
        for var_name, var in allocations.items():
            parts = var_name.split('_')
            node_id = parts[1]
            resource = parts[2]
            result[node_id][resource] = var.varValue or 0
        
        return dict(result)
    
    def _copy_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Create deep copy of plan"""
        import copy
        return copy.deepcopy(plan)
    
    def _recalculate_duration(self, plan: ExecutionPlan) -> timedelta:
        """Recalculate plan duration after optimization"""
        if plan.plan_type == PlanType.PARALLEL:
            # Calculate based on critical path
            critical_duration = sum(
                plan.nodes[node_id].estimated_duration.total_seconds()
                for node_id in plan.critical_path
                if node_id in plan.nodes
            )
            return timedelta(seconds=critical_duration)
        else:
            # Sum all durations
            total_duration = sum(
                node.estimated_duration.total_seconds()
                for node in plan.nodes.values()
            )
            return timedelta(seconds=total_duration)
    
    def _recalculate_quality(self, plan: ExecutionPlan) -> float:
        """Recalculate quality score"""
        # Reuse quality assessment logic
        return 0.85  # Placeholder

# ========== Supporting Components ==========

class ResourceScheduler:
    """Schedule and manage resources"""
    
    async def schedule_resources(
        self,
        tasks: List[PlanNode],
        available_resources: Dict[str, float],
        time_horizon: timedelta
    ) -> Dict[str, List[Tuple[datetime, datetime, float]]]:
        """Schedule resource usage over time"""
        schedule = defaultdict(list)
        
        # Simple scheduling algorithm
        current_time = datetime.now()
        
        for task in tasks:
            task_start = current_time
            task_end = task_start + task.estimated_duration
            
            for resource, amount in task.resource_requirements.items():
                if resource in available_resources:
                    schedule[resource].append((task_start, task_end, amount))
            
            current_time = task_end
        
        return dict(schedule)

class RiskAnalyzer:
    """Analyze risks in execution plans"""
    
    async def analyze(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Analyze risks in plan"""
        risks = {
            'overall_risk': 0.0,
            'node_risks': {},
            'resource_risks': {},
            'dependency_risks': {},
            'mitigation_strategies': []
        }
        
        # Analyze node-level risks
        for node_id, node in plan.nodes.items():
            node_risk = self._analyze_node_risk(node)
            risks['node_risks'][node_id] = node_risk
        
        # Analyze resource risks
        risks['resource_risks'] = self._analyze_resource_risks(plan)
        
        # Analyze dependency risks
        risks['dependency_risks'] = self._analyze_dependency_risks(plan)
        
        # Calculate overall risk
        all_risks = list(risks['node_risks'].values())
        if all_risks:
            risks['overall_risk'] = sum(r['level'] for r in all_risks) / len(all_risks)
        
        # Generate mitigation strategies
        risks['mitigation_strategies'] = self._generate_mitigations(risks)
        
        return risks
    
    def _analyze_node_risk(self, node: PlanNode) -> Dict[str, Any]:
        """Analyze risk for individual node"""
        risk_level = 0.0
        risk_factors = []
        
        # Complexity risk
        if node.estimated_duration > timedelta(hours=4):
            risk_level += 0.2
            risk_factors.append("Long duration task")
        
        # Dependency risk
        if len(node.dependencies) > 3:
            risk_level += 0.15
            risk_factors.append("Many dependencies")
        
        # Resource risk
        if sum(node.resource_requirements.values()) > 10:
            risk_level += 0.1
            risk_factors.append("High resource requirements")
        
        return {
            'level': min(1.0, risk_level),
            'factors': risk_factors,
            'type': self._classify_risk(risk_level)
        }
    
    def _classify_risk(self, risk_level: float) -> str:
        """Classify risk level"""
        if risk_level < 0.3:
            return 'low'
        elif risk_level < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _analyze_resource_risks(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Analyze resource-related risks"""
        resource_risks = {}
        
        # Calculate peak resource usage
        for resource in ['cpu', 'memory', 'gpu']:
            peak_usage = max(
                sum(
                    alloc.get(resource, 0)
                    for alloc in plan.resource_allocation.values()
                ),
                0
            )
            
            resource_risks[resource] = {
                'peak_usage': peak_usage,
                'risk_level': min(1.0, peak_usage / 100)  # Assuming 100 is max
            }
        
        return resource_risks
    
    def _analyze_dependency_risks(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Analyze dependency-related risks"""
        # Find long dependency chains
        max_chain_length = 0
        
        for node_id in plan.nodes:
            chain_length = self._find_dependency_depth(node_id, plan.nodes)
            max_chain_length = max(max_chain_length, chain_length)
        
        return {
            'max_dependency_chain': max_chain_length,
            'risk_level': min(1.0, max_chain_length / 10)
        }
    
    def _find_dependency_depth(self, node_id: str, nodes: Dict[str, PlanNode], visited: Set[str] = None) -> int:
        """Find maximum dependency depth"""
        if visited is None:
            visited = set()
        
        if node_id in visited or node_id not in nodes:
            return 0
        
        visited.add(node_id)
        node = nodes[node_id]
        
        if not node.dependencies:
            return 1
        
        max_depth = 0
        for dep in node.dependencies:
            depth = self._find_dependency_depth(dep, nodes, visited)
            max_depth = max(max_depth, depth)
        
        return max_depth + 1
    
    def _generate_mitigations(self, risks: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate risk mitigation strategies"""
        mitigations = []
        
        if risks['overall_risk'] > 0.5:
            mitigations.append({
                'risk': 'high_overall_risk',
                'strategy': 'Add comprehensive monitoring and checkpoints'
            })
        
        # Node-specific mitigations
        for node_id, risk in risks['node_risks'].items():
            if risk['level'] > 0.6:
                mitigations.append({
                    'risk': f'high_risk_node_{node_id}',
                    'strategy': f'Add fallback mechanism for {node_id}'
                })
        
        # Resource mitigations
        for resource, risk in risks['resource_risks'].items():
            if risk['risk_level'] > 0.7:
                mitigations.append({
                    'risk': f'resource_shortage_{resource}',
                    'strategy': f'Reserve additional {resource} or implement queuing'
                })
        
        return mitigations

class ExecutionEngine:
    """Core execution engine for plans"""
    
    def __init__(self):
        self.execution_strategies = {
            ExecutionStrategy.EAGER: self._eager_execution,
            ExecutionStrategy.LAZY: self._lazy_execution,
            ExecutionStrategy.SPECULATIVE: self._speculative_execution,
            ExecutionStrategy.FAULT_TOLERANT: self._fault_tolerant_execution
        }
    
    async def execute(
        self,
        plan: ExecutionPlan,
        context: ExecutionContext,
        strategy: ExecutionStrategy
    ) -> Any:
        """Execute plan with specified strategy"""
        executor = self.execution_strategies.get(
            strategy,
            self._default_execution
        )
        return await executor(plan, context)
    
    async def _eager_execution(self, plan: ExecutionPlan, context: ExecutionContext) -> Any:
        """Execute tasks as soon as dependencies are met"""
        # Implementation of eager execution
        pass
    
    async def _lazy_execution(self, plan: ExecutionPlan, context: ExecutionContext) -> Any:
        """Execute tasks only when results are needed"""
        # Implementation of lazy execution
        pass
    
    async def _speculative_execution(self, plan: ExecutionPlan, context: ExecutionContext) -> Any:
        """Execute likely paths speculatively"""
        # Implementation of speculative execution
        pass
    
    async def _fault_tolerant_execution(self, plan: ExecutionPlan, context: ExecutionContext) -> Any:
        """Execute with enhanced fault tolerance"""
        # Implementation with checkpointing and recovery
        pass
    
    async def _default_execution(self, plan: ExecutionPlan, context: ExecutionContext) -> Any:
        """Default execution strategy"""
        # Basic execution logic
        pass

class WorkflowOrchestrator:
    """Orchestrate complex workflows"""
    
    def __init__(self):
        self.workflows = {}
        self.running_workflows = {}
    
    async def orchestrate(
        self,
        workflow_id: str,
        context: Dict[str, Any]
    ) -> Any:
        """Orchestrate workflow execution"""
        # Workflow orchestration logic
        pass

class ExecutionMonitor:
    """Monitor execution progress and performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
    
    async def monitor(self, context: ExecutionContext) -> Dict[str, Any]:
        """Monitor execution context"""
        metrics = {
            'progress': self._calculate_progress(context),
            'performance': self._measure_performance(context),
            'health': self._check_health(context),
            'predictions': self._predict_completion(context)
        }
        
        # Check for alerts
        self._check_alerts(metrics, context)
        
        return metrics
    
    def _calculate_progress(self, context: ExecutionContext) -> float:
        """Calculate execution progress"""
        total = len(context.plan.nodes)
        completed = len(context.completed_tasks)
        return completed / total if total > 0 else 0
    
    def _measure_performance(self, context: ExecutionContext) -> Dict[str, float]:
        """Measure execution performance"""
        return {
            'throughput': self._calculate_throughput(context),
            'latency': self._calculate_latency(context),
            'efficiency': self._calculate_efficiency(context)
        }
    
    def _check_health(self, context: ExecutionContext) -> str:
        """Check execution health"""
        failure_rate = len(context.failed_tasks) / max(1, len(context.completed_tasks) + len(context.failed_tasks))
        
        if failure_rate > 0.5:
            return 'critical'
        elif failure_rate > 0.2:
            return 'warning'
        else:
            return 'healthy'
    
    def _predict_completion(self, context: ExecutionContext) -> Dict[str, Any]:
        """Predict completion time and success"""
        # Simple prediction based on current progress
        elapsed = datetime.now() - context.start_time
        progress = self._calculate_progress(context)
        
        if progress > 0:
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
        else:
            remaining = context.plan.estimated_duration
        
        return {
            'estimated_completion': datetime.now() + remaining,
            'confidence': 0.7 if progress > 0.2 else 0.3
        }
    
    def _calculate_throughput(self, context: ExecutionContext) -> float:
        """Calculate task throughput"""
        elapsed = (datetime.now() - context.start_time).total_seconds()
        completed = len(context.completed_tasks)
        return completed / elapsed if elapsed > 0 else 0
    
    def _calculate_latency(self, context: ExecutionContext) -> float:
        """Calculate average task latency"""
        latencies = []
        
        for event in context.events:
            if event['type'] == 'node_completed' and 'duration' in event:
                latencies.append(event['duration'])
        
        return sum(latencies) / len(latencies) if latencies else 0
    
    def _calculate_efficiency(self, context: ExecutionContext) -> float:
        """Calculate execution efficiency"""
        # Compare actual vs estimated duration
        if not context.completed_tasks:
            return 1.0
        
        actual_duration = (datetime.now() - context.start_time).total_seconds()
        estimated_duration = context.plan.estimated_duration.total_seconds()
        
        if actual_duration > 0:
            return min(1.0, estimated_duration / actual_duration)
        return 1.0
    
    def _check_alerts(self, metrics: Dict[str, Any], context: ExecutionContext):
        """Check for alert conditions"""
        if metrics['health'] == 'critical':
            self.alerts.append({
                'type': 'health_critical',
                'message': 'Execution health is critical',
                'timestamp': datetime.now()
            })
        
        if metrics['performance']['efficiency'] < 0.5:
            self.alerts.append({
                'type': 'low_efficiency',
                'message': 'Execution efficiency below 50%',
                'timestamp': datetime.now()
            })

class RecoveryManager:
    """Manage failure recovery"""
    
    async def recover(
        self,
        failure_context: Dict[str, Any],
        plan: ExecutionPlan
    ) -> Optional[ExecutionPlan]:
        """Attempt to recover from failure"""
        failure_type = self._classify_failure(failure_context)
        
        if failure_type == 'resource_exhaustion':
            return await self._recover_from_resource_exhaustion(plan)
        elif failure_type == 'task_failure':
            return await self._recover_from_task_failure(plan, failure_context)
        elif failure_type == 'timeout':
            return await self._recover_from_timeout(plan)
        else:
            return None
    
    def _classify_failure(self, failure_context: Dict[str, Any]) -> str:
        """Classify type of failure"""
        error = failure_context.get('error', '')
        
        if 'resource' in error.lower():
            return 'resource_exhaustion'
        elif 'timeout' in error.lower():
            return 'timeout'
        else:
            return 'task_failure'
    
    async def _recover_from_resource_exhaustion(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Recover from resource exhaustion"""
        # Reduce resource requirements or wait for resources
        recovery_plan = self._copy_plan(plan)
        
        # Reduce resource allocations by 20%
        for node_id, allocation in recovery_plan.resource_allocation.items():
            for resource, amount in allocation.items():
                recovery_plan.resource_allocation[node_id][resource] = amount * 0.8
        
        return recovery_plan
    
    async def _recover_from_task_failure(
        self,
        plan: ExecutionPlan,
        failure_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """Recover from task failure"""
        failed_node_id = failure_context.get('node_id')
        if not failed_node_id:
            return None
        
        recovery_plan = self._copy_plan(plan)
        
        # Add alternative task or skip non-critical task
        if failed_node_id in recovery_plan.nodes:
            node = recovery_plan.nodes[failed_node_id]
            if not node.constraints.get('critical', False):
                # Skip non-critical task
                recovery_plan.nodes.pop(failed_node_id)
                # Update dependencies
                for other_node in recovery_plan.nodes.values():
                    if failed_node_id in other_node.dependencies:
                        other_node.dependencies.remove(failed_node_id)
        
        return recovery_plan
    
    async def _recover_from_timeout(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Recover from timeout"""
        recovery_plan = self._copy_plan(plan)
        
        # Increase timeouts
        for node in recovery_plan.nodes.values():
            node.estimated_duration = node.estimated_duration * 1.5
        
        return recovery_plan
    
    def _copy_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Create copy of plan"""
        import copy
        return copy.deepcopy(plan)

class WorkflowTemplateLibrary:
    """Library of reusable workflow templates"""
    
    def __init__(self):
        self.templates = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default workflow templates"""
        # Software development workflow
        self.templates['software_development'] = WorkflowTemplate(
            id='software_dev_1',
            name='Software Development Workflow',
            description='Complete software development lifecycle',
            category='development',
            steps=[
                {'name': 'requirements', 'type': 'analysis'},
                {'name': 'design', 'type': 'design', 'depends_on': ['requirements']},
                {'name': 'implementation', 'type': 'development', 'depends_on': ['design']},
                {'name': 'testing', 'type': 'testing', 'depends_on': ['implementation']},
                {'name': 'deployment', 'type': 'deployment', 'depends_on': ['testing']}
            ],
            parameters={'language': 'python', 'framework': 'any'},
            requirements={'agents': ['code', 'test', 'deploy']},
            success_criteria={'test_coverage': 0.8, 'deployment_success': True},
            typical_duration=timedelta(days=5),
            success_rate=0.85
        )
        
        # Data analysis workflow
        self.templates['data_analysis'] = WorkflowTemplate(
            id='data_analysis_1',
            name='Data Analysis Workflow',
            description='Comprehensive data analysis pipeline',
            category='analytics',
            steps=[
                {'name': 'data_collection', 'type': 'collection'},
                {'name': 'data_cleaning', 'type': 'preprocessing', 'depends_on': ['data_collection']},
                {'name': 'exploration', 'type': 'analysis', 'depends_on': ['data_cleaning']},
                {'name': 'modeling', 'type': 'ml', 'depends_on': ['exploration']},
                {'name': 'visualization', 'type': 'visualization', 'depends_on': ['modeling']},
                {'name': 'reporting', 'type': 'report', 'depends_on': ['visualization']}
            ],
            parameters={'data_source': 'any', 'analysis_type': 'exploratory'},
            requirements={'agents': ['data', 'ml', 'viz']},
            success_criteria={'data_quality': 0.9, 'insights_generated': True},
            typical_duration=timedelta(days=3),
            success_rate=0.9
        )
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get workflow template"""
        return self.templates.get(template_id)
    
    def search_templates(self, category: str = None, keywords: List[str] = None) -> List[WorkflowTemplate]:
        """Search for templates"""
        results = []
        
        for template in self.templates.values():
            if category and template.category != category:
                continue
            
            if keywords:
                template_text = f"{template.name} {template.description}".lower()
                if not any(keyword.lower() in template_text for keyword in keywords):
                    continue
            
            results.append(template)
        
        return results

# ========== Custom Exceptions ==========

class ResourceError(Exception):
    """Resource allocation error"""
    pass

class PlanningError(Exception):
    """Planning error"""
    pass

class ExecutionError(Exception):
    """Execution error"""
    pass

# ========== Integration Example ==========

async def example_planning_agent_usage():
    """Example of using the planning & execution agent"""
    
    # Create planning agent
    config = AgentConfig(
        role=AgentRole.PLANNER,
        model_provider=ModelProvider.CLAUDE_4_SONNET,
        temperature=0.3,
        max_tokens=4096,
        capabilities={
            'planning': 0.95,
            'optimization': 0.9,
            'resource_management': 0.85,
            'risk_assessment': 0.85,
            'execution_monitoring': 0.9
        }
    )
    
    agent = PlanningExecutionAgent("planning_agent_1", config)
    
    # Start agent
    await agent.start()
    
    # Create execution plan
    plan = await agent.create_execution_plan(
        goal="Develop and deploy a machine learning model for customer churn prediction",
        constraints={
            'deadline': datetime.now() + timedelta(days=7),
            'budget': 5000,
            'resources': {
                'cpu': 32,
                'memory': 64000,
                'gpu': 2
            }
        }
    )
    
    print(f"Created plan: {plan.name}")
    print(f"Plan type: {plan.plan_type.value}")
    print(f"Estimated duration: {plan.estimated_duration}")
    print(f"Number of tasks: {len(plan.nodes)}")
    print(f"Critical path length: {len(plan.critical_path)}")
    print(f"Quality score: {plan.quality_score:.2f}")
    
    # Execute plan
    result = await agent.execute_plan(plan.id)
    
    print(f"\nExecution result: {result['status']}")
    print(f"Completed tasks: {len(result['completed'])}")
    print(f"Failed tasks: {len(result['failed'])}")
    print(f"Duration: {result['duration']}")

if __name__ == "__main__":
    asyncio.run(example_planning_agent_usage())
