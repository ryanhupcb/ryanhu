# LocalAgentSystem - Agent Implementation
# Agent系统实现

from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional, Set
from abc import ABC, abstractmethod
import random
import heapq
from dataclasses import dataclass
import numpy as np


# ==================== Agent基类 ====================

class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        llm: LLMInterface,
        tools: List[Tool] = None,
        memory: Memory = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.llm = llm
        self.tools = tools or []
        self.memory = memory or Memory()
        self.conversation_history: List[Message] = []
        
    @abstractmethod
    async def process(self, task: Task) -> Dict[str, Any]:
        """处理任务"""
        pass
        
    async def think(self, context: str) -> str:
        """思考步骤 - ReAct中的Thought"""
        prompt = Message(
            role="system",
            content=f"You are a {self.role.value} agent. Think about the following context and provide your reasoning:\n\n{context}"
        )
        
        messages = [prompt] + self.conversation_history[-5:]  # 包含最近5条历史
        thought = await self.llm.generate(messages, temperature=0.7)
        
        # 记录思考过程
        self.memory.add_to_short_term({
            "type": "thought",
            "agent": self.agent_id,
            "content": thought
        })
        
        return thought
        
    async def act(self, thought: str, available_tools: List[Tool]) -> Dict[str, Any]:
        """行动步骤 - ReAct中的Action"""
        if not available_tools:
            return {"action": "none", "result": "No tools available"}
            
        # 让LLM选择工具并执行
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in available_tools
        ])
        
        prompt = Message(
            role="system",
            content=f"""Based on your thought: {thought}
            
Available tools:
{tool_descriptions}

Choose the best tool and provide the arguments needed."""
        )
        
        messages = [prompt]
        response, tool_call = await self.llm.generate_with_tools(messages, available_tools)
        
        if tool_call:
            # 执行工具
            tool = next((t for t in available_tools if t.name == tool_call["name"]), None)
            if tool:
                result = await tool.execute(**tool_call.get("arguments", {}))
                
                # 记录行动
                self.memory.add_to_short_term({
                    "type": "action",
                    "agent": self.agent_id,
                    "tool": tool_call["name"],
                    "arguments": tool_call.get("arguments", {}),
                    "result": result
                })
                
                return {"action": tool_call["name"], "result": result}
                
        return {"action": "none", "result": response}
        
    async def observe(self, action_result: Dict[str, Any]) -> str:
        """观察步骤 - ReAct中的Observation"""
        observation = f"Action '{action_result['action']}' resulted in: {action_result['result']}"
        
        # 记录观察
        self.memory.add_to_short_term({
            "type": "observation",
            "agent": self.agent_id,
            "content": observation
        })
        
        return observation


# ==================== ReAct + Tree of Thoughts 混合架构 ====================

class ReactToTAgent(BaseAgent):
    """ReAct + Tree of Thoughts 混合Agent"""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        llm: LLMInterface,
        tools: List[Tool] = None,
        memory: Memory = None,
        max_thoughts_per_step: int = 3,
        max_depth: int = 5,
        exploration_constant: float = 1.414
    ):
        super().__init__(agent_id, role, llm, tools, memory)
        self.max_thoughts_per_step = max_thoughts_per_step
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.thought_tree: Dict[str, Thought] = {}
        
    async def process(self, task: Task) -> Dict[str, Any]:
        """使用ReAct+ToT处理任务"""
        try:
            # 初始化根思考节点
            root_thought = await self.initialize_thought_tree(task)
            
            # 执行树搜索
            best_solution = await self.tree_search(root_thought, task)
            
            return {
                "success": True,
                "solution": best_solution,
                "thought_tree_size": len(self.thought_tree)
            }
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def initialize_thought_tree(self, task: Task) -> Thought:
        """初始化思考树"""
        # 生成初始思考
        initial_context = f"Task: {task.description}\nType: {task.type}"
        initial_thought = await self.think(initial_context)
        
        root = Thought(
            content=initial_thought,
            score=await self.evaluate_thought(initial_thought, task),
            metadata={"task_id": task.id}
        )
        
        self.thought_tree[root.id] = root
        return root
        
    async def tree_search(self, root: Thought, task: Task) -> Dict[str, Any]:
        """执行树搜索（结合MCTS）"""
        best_solution = None
        best_score = -float('inf')
        
        # 优先队列用于最佳优先搜索
        frontier = [(-root.score, root.id)]
        visited = set()
        
        while frontier and len(visited) < 50:  # 限制搜索次数
            _, thought_id = heapq.heappop(frontier)
            
            if thought_id in visited:
                continue
                
            visited.add(thought_id)
            current_thought = self.thought_tree[thought_id]
            
            # 检查深度限制
            if current_thought.depth >= self.max_depth:
                continue
                
            # 生成子思考节点
            child_thoughts = await self.expand_thought(current_thought, task)
            
            for child in child_thoughts:
                self.thought_tree[child.id] = child
                current_thought.children.append(child.id)
                
                # 执行ReAct循环
                solution = await self.react_cycle(child, task)
                
                if solution and solution.get("score", 0) > best_score:
                    best_score = solution["score"]
                    best_solution = solution
                    
                # 添加到frontier
                priority = -self.uct_score(child, current_thought)
                heapq.heappush(frontier, (priority, child.id))
                
        return best_solution
        
    async def expand_thought(self, parent_thought: Thought, task: Task) -> List[Thought]:
        """扩展思考节点"""
        children = []
        
        # 生成多个可能的思考路径
        expansion_prompt = f"""Given the current thought: {parent_thought.content}
        
Generate {self.max_thoughts_per_step} different approaches or next steps for solving: {task.description}"""
        
        prompt = Message(role="system", content=expansion_prompt)
        response = await self.llm.generate([prompt], temperature=0.8)
        
        # 解析响应并创建子节点
        approaches = response.split("\n\n")[:self.max_thoughts_per_step]
        
        for approach in approaches:
            if approach.strip():
                child = Thought(
                    content=approach.strip(),
                    parent_id=parent_thought.id,
                    depth=parent_thought.depth + 1,
                    score=await self.evaluate_thought(approach, task)
                )
                children.append(child)
                
        return children
        
    async def react_cycle(self, thought: Thought, task: Task) -> Optional[Dict[str, Any]]:
        """执行ReAct循环"""
        max_iterations = 5
        
        for i in range(max_iterations):
            # Think
            current_context = self.build_context(thought, task)
            reasoning = await self.think(current_context)
            
            # Act
            action_result = await self.act(reasoning, self.tools)
            
            # Observe
            observation = await self.observe(action_result)
            
            # 检查是否完成
            if await self.is_task_complete(task, observation):
                return {
                    "thought_path": self.get_thought_path(thought),
                    "final_result": action_result["result"],
                    "score": thought.score,
                    "iterations": i + 1
                }
                
            # 更新思考内容
            thought.content += f"\n\nIteration {i+1}:\nReasoning: {reasoning}\nObservation: {observation}"
            
        return None
        
    async def evaluate_thought(self, thought_content: str, task: Task) -> float:
        """评估思考的质量"""
        evaluation_prompt = f"""Rate the following thought/approach for solving the task on a scale of 0-10:

Task: {task.description}
Thought: {thought_content}

Consider: relevance, feasibility, completeness, and innovation.
Respond with just a number."""
        
        prompt = Message(role="system", content=evaluation_prompt)
        response = await self.llm.generate([prompt], temperature=0.3)
        
        try:
            score = float(response.strip())
            return min(max(score, 0), 10) / 10  # 归一化到0-1
        except:
            return 0.5  # 默认分数
            
    def uct_score(self, child: Thought, parent: Thought) -> float:
        """计算UCT分数（用于树搜索）"""
        if not parent.metadata.get("visits", 0):
            return float('inf')
            
        exploitation = child.score
        exploration = self.exploration_constant * np.sqrt(
            np.log(parent.metadata.get("visits", 1)) / 
            (child.metadata.get("visits", 1) + 1)
        )
        
        return exploitation + exploration
        
    def build_context(self, thought: Thought, task: Task) -> str:
        """构建当前上下文"""
        path = self.get_thought_path(thought)
        recent_memory = self.memory.get_context(5)
        
        context = f"""Task: {task.description}
Current thought path: {' -> '.join(path)}
Current thought: {thought.content}

Recent actions and observations:
{self.format_memory(recent_memory)}"""
        
        return context
        
    def get_thought_path(self, thought: Thought) -> List[str]:
        """获取从根到当前思考的路径"""
        path = []
        current = thought
        
        while current:
            path.append(current.content[:50] + "...")  # 截断长内容
            if current.parent_id:
                current = self.thought_tree.get(current.parent_id)
            else:
                break
                
        return list(reversed(path))
        
    def format_memory(self, memory_items: List[Dict[str, Any]]) -> str:
        """格式化记忆项"""
        formatted = []
        for item in memory_items:
            if item["type"] == "action":
                formatted.append(f"- Action: {item['tool']} with {item['arguments']}")
            elif item["type"] == "observation":
                formatted.append(f"- Observation: {item['content']}")
                
        return "\n".join(formatted)
        
    async def is_task_complete(self, task: Task, observation: str) -> bool:
        """判断任务是否完成"""
        completion_prompt = f"""Task: {task.description}
Latest observation: {observation}

Has the task been completed successfully? Answer with just 'yes' or 'no'."""
        
        prompt = Message(role="system", content=completion_prompt)
        response = await self.llm.generate([prompt], temperature=0.1)
        
        return response.strip().lower() == 'yes'


# ==================== 专门化的Agent实现 ====================

class CoderAgent(ReactToTAgent):
    """代码开发Agent"""
    
    def __init__(self, agent_id: str, llm: LLMInterface, tools: List[Tool] = None):
        # 添加代码相关工具
        code_tools = [
            PythonExecutor(),
            FileOperator(),
            # 可以添加更多工具如：代码分析器、测试运行器等
        ]
        
        if tools:
            code_tools.extend(tools)
            
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.CODER,
            llm=llm,
            tools=code_tools
        )
        
    async def process(self, task: Task) -> Dict[str, Any]:
        """处理代码开发任务"""
        # 增强任务描述
        if task.type == "code_generation":
            task.description = f"""As a coding expert, {task.description}
            
Please follow best practices:
1. Write clean, modular code
2. Include proper error handling
3. Add comments and documentation
4. Consider edge cases
5. Optimize for performance when relevant"""
            
        return await super().process(task)


class ResearcherAgent(ReactToTAgent):
    """研究Agent"""
    
    def __init__(self, agent_id: str, llm: LLMInterface, tools: List[Tool] = None):
        research_tools = [
            WebSearcher(),
            FileOperator(),
        ]
        
        if tools:
            research_tools.extend(tools)
            
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.RESEARCHER,
            llm=llm,
            tools=research_tools
        )
        
    async def process(self, task: Task) -> Dict[str, Any]:
        """处理研究任务"""
        if task.type == "research":
            task.description = f"""As a research expert, {task.description}
            
Please:
1. Gather information from multiple sources
2. Verify facts and cross-reference
3. Provide citations when possible
4. Summarize key findings
5. Identify knowledge gaps"""
            
        return await super().process(task)


class ReviewerAgent(BaseAgent):
    """审核Agent - 使用简单的ReAct"""
    
    def __init__(self, agent_id: str, llm: LLMInterface):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.REVIEWER,
            llm=llm,
            tools=[]
        )
        
    async def process(self, task: Task) -> Dict[str, Any]:
        """审核其他Agent的工作"""
        review_prompt = f"""Review the following work:

Task: {task.description}
Result: {task.metadata.get('result_to_review', 'No result provided')}

Provide:
1. Quality assessment (1-10)
2. Strengths
3. Areas for improvement
4. Specific suggestions
5. Overall recommendation (approve/revise/reject)"""
        
        prompt = Message(role="system", content=review_prompt)
        review = await self.llm.generate([prompt])
        
        return {
            "success": True,
            "review": review
        }


# ==================== 多Agent协作系统 ====================

class MultiAgentOrchestrator:
    """多Agent协调器"""
    
    def __init__(self, orchestrator_llm: LLMInterface):
        self.orchestrator_llm = orchestrator_llm
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.completed_tasks: Dict[str, Task] = {}
        self.active_tasks: Dict[str, Task] = {}
        
    def register_agent(self, agent: BaseAgent):
        """注册Agent"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} with role {agent.role.value}")
        
    async def process_user_request(self, user_input: str) -> Dict[str, Any]:
        """处理用户请求"""
        # 1. 理解用户意图并拆解任务
        tasks = await self.decompose_request(user_input)
        
        # 2. 创建任务依赖图
        task_graph = await self.create_task_graph(tasks)
        
        # 3. 执行任务
        results = await self.execute_task_graph(task_graph)
        
        # 4. 综合结果
        final_result = await self.synthesize_results(results, user_input)
        
        return final_result
        
    async def decompose_request(self, user_input: str) -> List[Task]:
        """将用户请求分解为子任务"""
        decomposition_prompt = f"""Decompose the following user request into specific subtasks:

User request: {user_input}

For each subtask, provide:
1. Task type (code_generation, research, file_operation, review, etc.)
2. Clear description
3. Dependencies on other tasks (if any)
4. Suggested agent role

Format as a JSON list."""
        
        prompt = Message(role="system", content=decomposition_prompt)
        response = await self.orchestrator_llm.generate([prompt], temperature=0.3)
        
        # 解析响应并创建任务
        tasks = []
        try:
            # 简化示例 - 实际应该解析JSON
            task_descriptions = response.split("\n\n")
            for i, desc in enumerate(task_descriptions):
                if desc.strip():
                    task = Task(
                        type="general",  # 应该从响应中解析
                        description=desc.strip()
                    )
                    tasks.append(task)
        except Exception as e:
            logger.error(f"Task decomposition error: {e}")
            # 创建单个任务作为后备
            tasks = [Task(type="general", description=user_input)]
            
        return tasks
        
    async def create_task_graph(self, tasks: List[Task]) -> Dict[str, Task]:
        """创建任务依赖图"""
        task_graph = {}
        
        for task in tasks:
            task_graph[task.id] = task
            
        # 简化示例 - 实际应该分析任务间的依赖关系
        # 这里假设任务是顺序执行的
        for i in range(len(tasks) - 1):
            tasks[i+1].dependencies.append(tasks[i].id)
            
        return task_graph
        
    async def execute_task_graph(self, task_graph: Dict[str, Task]) -> Dict[str, Any]:
        """执行任务图"""
        results = {}
        
        # 拓扑排序执行任务
        while task_graph:
            # 找到没有依赖的任务
            ready_tasks = [
                task for task in task_graph.values()
                if all(dep in results for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                logger.error("Circular dependency detected in task graph")
                break
                
            # 并行执行就绪任务
            task_futures = []
            for task in ready_tasks:
                agent = await self.select_agent(task)
                if agent:
                    future = asyncio.create_task(self.execute_task(task, agent))
                    task_futures.append((task.id, future))
                    
            # 等待任务完成
            for task_id, future in task_futures:
                try:
                    result = await future
                    results[task_id] = result
                    del task_graph[task_id]
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    results[task_id] = {"error": str(e)}
                    del task_graph[task_id]
                    
        return results
        
    async def select_agent(self, task: Task) -> Optional[BaseAgent]:
        """为任务选择最合适的Agent"""
        # 基于任务类型选择Agent
        role_mapping = {
            "code_generation": AgentRole.CODER,
            "research": AgentRole.RESEARCHER,
            "review": AgentRole.REVIEWER,
            "file_operation": AgentRole.EXECUTOR,
        }
        
        preferred_role = role_mapping.get(task.type, AgentRole.EXECUTOR)
        
        # 找到匹配角色的Agent
        for agent in self.agents.values():
            if agent.role == preferred_role:
                return agent
                
        # 如果没有匹配的，选择第一个可用的Agent
        return next(iter(self.agents.values())) if self.agents else None
        
    async def execute_task(self, task: Task, agent: BaseAgent) -> Dict[str, Any]:
        """执行单个任务"""
        logger.info(f"Agent {agent.agent_id} executing task {task.id}")
        
        self.active_tasks[task.id] = task
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_agent = agent.agent_id
        
        try:
            result = await agent.process(task)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            
            self.completed_tasks[task.id] = task
            del self.active_tasks[task.id]
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            del self.active_tasks[task.id]
            raise
            
    async def synthesize_results(self, results: Dict[str, Any], original_request: str) -> Dict[str, Any]:
        """综合所有结果"""
        synthesis_prompt = f"""Original user request: {original_request}

Task results:
{json.dumps(results, indent=2)}

Please synthesize these results into a coherent final response that addresses the user's original request."""
        
        prompt = Message(role="system", content=synthesis_prompt)
        final_response = await self.orchestrator_llm.generate([prompt])
        
        return {
            "success": True,
            "response": final_response,
            "detailed_results": results,
            "tasks_completed": len(results)
        }


# ==================== 配置管理 ====================

@dataclass
class AgentSystemConfig:
    """系统配置"""
    # API Keys
    claude_api_key: str = ""
    deepseek_api_key: str = ""
    qwen_api_key: str = ""
    
    # Model Selection
    claude_model: str = "claude-opus-4-20250514"
    deepseek_model: str = "deepseek-coder"
    qwen_model: str = "qwen-plus"
    
    # Agent Settings
    max_agents: int = 10
    max_concurrent_tasks: int = 5
    
    # ToT Settings
    max_thoughts_per_step: int = 3
    max_tree_depth: int = 5
    exploration_constant: float = 1.414
    
    # Memory Settings
    max_short_term_memory: int = 100
    max_long_term_memory: int = 10000
    
    # System Settings
    workspace_dir: str = "./workspace"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'AgentSystemConfig':
        """从环境变量加载配置"""
        return cls(
            claude_api_key=os.getenv("CLAUDE_API_KEY", ""),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            qwen_api_key=os.getenv("QWEN_API_KEY", ""),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-opus-4-20250514"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-coder"),
            qwen_model=os.getenv("QWEN_MODEL", "qwen-plus"),
        )


# ==================== 主系统类 ====================

class LocalAgentSystem:
    """本地Agent系统主类"""
    
    def __init__(self, config: AgentSystemConfig):
        self.config = config
        self.orchestrator = None
        self.is_initialized = False
        
    async def initialize(self):
        """初始化系统"""
        logger.info("Initializing Local Agent System...")
        
        # 创建LLM实例
        claude_llm = ClaudeLLM(self.config.claude_api_key, self.config.claude_model)
        deepseek_llm = DeepSeekLLM(self.config.deepseek_api_key, self.config.deepseek_model)
        qwen_llm = QwenLLM(self.config.qwen_api_key, self.config.qwen_model)
        
        # 创建协调器
        self.orchestrator = MultiAgentOrchestrator(qwen_llm)
        
        # 创建并注册Agents
        agents = [
            CoderAgent("coder_1", deepseek_llm),  # 使用DeepSeek进行代码生成（成本低）
            ResearcherAgent("researcher_1", claude_llm),  # 使用Claude进行复杂研究
            ReviewerAgent("reviewer_1", claude_llm),  # 使用Claude进行审核
        ]
        
        for agent in agents:
            self.orchestrator.register_agent(agent)
            
        self.is_initialized = True
        logger.info("System initialized successfully")
        
    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """处理用户请求"""
        if not self.is_initialized:
            await self.initialize()
            
        logger.info(f"Processing user request: {user_input[:100]}...")
        
        try:
            result = await self.orchestrator.process_user_request(user_input)
            return result
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def shutdown(self):
        """关闭系统"""
        logger.info("Shutting down Local Agent System...")
        # 清理资源
        self.is_initialized = False


# ==================== 使用示例 ====================

async def main():
    """主函数示例"""
    # 加载配置
    config = AgentSystemConfig.from_env()
    
    # 创建系统
    system = LocalAgentSystem(config)
    
    # 初始化
    await system.initialize()
    
    # 处理请求示例
    requests = [
        "创建一个Python Web爬虫，爬取新闻网站的头条并保存到JSON文件",
        "开发一个简单的任务管理API，支持CRUD操作",
        "实现一个递归下降解析器来解析简单的数学表达式",
    ]
    
    for request in requests:
        print(f"\n{'='*50}")
        print(f"Request: {request}")
        print(f"{'='*50}")
        
        result = await system.process_request(request)
        
        if result["success"]:
            print(f"Response: {result['response']}")
            print(f"Tasks completed: {result['tasks_completed']}")
        else:
            print(f"Error: {result['error']}")
            
    # 关闭系统
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
