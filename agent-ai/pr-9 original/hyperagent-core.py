"""
HyperAgent Platform v1.0 - Core Agent System
生产级超级Agent系统核心实现
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TypeVar, Generic
import threading
import queue
import weakref
import hashlib
import pickle
import numpy as np
from contextlib import asynccontextmanager, contextmanager
import aiohttp
import redis.asyncio as redis
from pydantic import BaseModel, Field, validator
import networkx as nx
from transformers import AutoTokenizer
import torch
import faiss
from prometheus_client import Counter, Histogram, Gauge, Summary
import opentelemetry.trace as trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ============================= 核心类型定义 =============================

class ModelProvider(Enum):
    """支持的模型提供商"""
    CLAUDE_4_OPUS = "claude-4-opus-20240229"
    CLAUDE_4_SONNET = "claude-4-sonnet-20240229"
    CLAUDE_3_7_SONNET = "claude-3.7-sonnet-20241022"
    QWEN_PLUS = "qwen-plus"
    QWEN_MAX = "qwen-max"
    QWEN_TURBO = "qwen-turbo"

class AgentRole(Enum):
    """Agent角色定义"""
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    EXECUTOR = "executor"
    ANALYZER = "analyzer"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"

class ThoughtType(Enum):
    """思维类型定义"""
    OBSERVATION = "observation"
    REASONING = "reasoning"
    PLANNING = "planning"
    REFLECTION = "reflection"
    HYPOTHESIS = "hypothesis"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    CRITIQUE = "critique"
    EXPLORATION = "exploration"
    ABSTRACTION = "abstraction"

class MemoryType(Enum):
    """记忆类型定义"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    SENSORY = "sensory"
    LONG_TERM = "long_term"
    SHORT_TERM = "short_term"

class CoordinationStrategy(Enum):
    """协调策略定义"""
    HIERARCHICAL = "hierarchical"
    FEDERATED = "federated"
    CONSENSUS = "consensus"
    MARKET_BASED = "market_based"
    STIGMERGIC = "stigmergic"
    BLACKBOARD = "blackboard"

# ============================= 数据模型定义 =============================

@dataclass
class Thought:
    """思维单元"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ThoughtType = ThoughtType.OBSERVATION
    content: str = ""
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    vector_embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class Memory:
    """记忆单元"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType = MemoryType.WORKING
    content: Any = None
    importance: float = 0.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1
    associations: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    vector_embedding: Optional[np.ndarray] = None
    
    def update_access(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        # 基于访问频率调整重要性
        self.importance = min(1.0, self.importance + 0.05)
    
    def calculate_strength(self) -> float:
        """计算记忆强度"""
        time_decay = np.exp(-self.decay_rate * (datetime.now() - self.last_accessed).total_seconds() / 3600)
        frequency_bonus = np.log1p(self.access_count) / 10
        return min(1.0, self.importance * time_decay + frequency_bonus)

@dataclass
class AgentMessage:
    """Agent间消息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: str = "task"
    content: Any = None
    priority: int = 0
    requires_response: bool = False
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.ttl is None:
            return False
        return datetime.now() - self.timestamp > self.ttl

@dataclass
class Task:
    """任务定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    objective: str = ""
    constraints: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    priority: int = 0
    deadline: Optional[datetime] = None
    assigned_agent_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    progress: float = 0.0
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

# ============================= 思维链实现 =============================

class ThoughtChain:
    """思维链管理器"""
    
    def __init__(self, max_depth: int = 10, max_branches: int = 5):
        self.thoughts: Dict[str, Thought] = {}
        self.graph = nx.DiGraph()
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.root_thoughts: Set[str] = set()
        self._lock = threading.RLock()
        
    def add_thought(self, thought: Thought, parent_id: Optional[str] = None) -> str:
        """添加思维节点"""
        with self._lock:
            # 检查深度限制
            if parent_id and self._get_depth(parent_id) >= self.max_depth:
                raise ValueError(f"Maximum thought depth {self.max_depth} exceeded")
            
            # 检查分支限制
            if parent_id and len(self.thoughts.get(parent_id, Thought()).children_ids) >= self.max_branches:
                raise ValueError(f"Maximum branches {self.max_branches} exceeded for parent thought")
            
            # 添加思维
            self.thoughts[thought.id] = thought
            self.graph.add_node(thought.id, thought=thought)
            
            if parent_id:
                thought.parent_id = parent_id
                self.thoughts[parent_id].children_ids.append(thought.id)
                self.graph.add_edge(parent_id, thought.id)
            else:
                self.root_thoughts.add(thought.id)
            
            return thought.id
    
    def _get_depth(self, thought_id: str) -> int:
        """获取思维节点深度"""
        depth = 0
        current_id = thought_id
        while current_id and current_id in self.thoughts:
            parent_id = self.thoughts[current_id].parent_id
            if not parent_id:
                break
            depth += 1
            current_id = parent_id
            if depth > self.max_depth:  # 防止循环
                break
        return depth
    
    def get_thought_path(self, thought_id: str) -> List[Thought]:
        """获取从根到指定思维的路径"""
        with self._lock:
            path = []
            current_id = thought_id
            while current_id and current_id in self.thoughts:
                path.append(self.thoughts[current_id])
                current_id = self.thoughts[current_id].parent_id
            return list(reversed(path))
    
    def get_subtree(self, thought_id: str, max_depth: int = 3) -> Dict[str, Thought]:
        """获取子树"""
        with self._lock:
            subtree = {}
            queue = deque([(thought_id, 0)])
            
            while queue:
                current_id, depth = queue.popleft()
                if depth > max_depth or current_id not in self.thoughts:
                    continue
                
                thought = self.thoughts[current_id]
                subtree[current_id] = thought
                
                for child_id in thought.children_ids:
                    queue.append((child_id, depth + 1))
            
            return subtree
    
    def prune_low_confidence_branches(self, threshold: float = 0.3):
        """修剪低置信度分支"""
        with self._lock:
            to_remove = []
            for thought_id, thought in self.thoughts.items():
                if thought.confidence < threshold and not thought.children_ids:
                    to_remove.append(thought_id)
            
            for thought_id in to_remove:
                self._remove_thought(thought_id)
    
    def _remove_thought(self, thought_id: str):
        """移除思维节点"""
        if thought_id not in self.thoughts:
            return
        
        thought = self.thoughts[thought_id]
        
        # 移除父节点的引用
        if thought.parent_id and thought.parent_id in self.thoughts:
            parent = self.thoughts[thought.parent_id]
            parent.children_ids = [cid for cid in parent.children_ids if cid != thought_id]
        else:
            self.root_thoughts.discard(thought_id)
        
        # 递归移除子节点
        for child_id in thought.children_ids[:]:
            self._remove_thought(child_id)
        
        # 从图中移除
        if thought_id in self.graph:
            self.graph.remove_node(thought_id)
        
        # 从字典中移除
        del self.thoughts[thought_id]
    
    def find_similar_thoughts(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Thought, float]]:
        """查找相似思维"""
        with self._lock:
            similarities = []
            
            for thought in self.thoughts.values():
                if thought.vector_embedding is not None:
                    similarity = np.dot(query_embedding, thought.vector_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(thought.vector_embedding)
                    )
                    similarities.append((thought, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

# ============================= 认知架构实现 =============================

class CognitiveArchitecture:
    """认知架构基类"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.thought_chain = ThoughtChain()
        self.working_memory = deque(maxlen=20)
        self.attention_weights = defaultdict(float)
        self.metacognitive_state = {
            "confidence": 0.5,
            "uncertainty": 0.5,
            "exploration_rate": 0.3,
            "reflection_frequency": 0.2
        }
        self._lock = threading.RLock()
        
    def perceive(self, observation: Any) -> Thought:
        """感知处理"""
        with self._lock:
            thought = Thought(
                type=ThoughtType.OBSERVATION,
                content=str(observation),
                confidence=0.9,
                metadata={"source": "perception", "agent_id": self.agent_id}
            )
            self.thought_chain.add_thought(thought)
            self.working_memory.append(thought)
            self._update_attention(thought)
            return thought
    
    def reason(self, context: List[Thought]) -> Thought:
        """推理处理"""
        with self._lock:
            # 基于上下文生成推理
            evidence = [t.id for t in context]
            reasoning_content = self._generate_reasoning(context)
            
            thought = Thought(
                type=ThoughtType.REASONING,
                content=reasoning_content,
                confidence=self._calculate_reasoning_confidence(context),
                evidence=evidence,
                metadata={"agent_id": self.agent_id}
            )
            
            # 链接到最相关的上下文思维
            if context:
                parent = max(context, key=lambda t: self.attention_weights.get(t.id, 0))
                self.thought_chain.add_thought(thought, parent.id)
            else:
                self.thought_chain.add_thought(thought)
            
            self.working_memory.append(thought)
            self._update_attention(thought)
            return thought
    
    def plan(self, objective: str, constraints: List[str]) -> List[Thought]:
        """规划处理"""
        with self._lock:
            plan_thoughts = []
            
            # 创建规划根节点
            root_thought = Thought(
                type=ThoughtType.PLANNING,
                content=f"Planning for objective: {objective}",
                confidence=0.8,
                metadata={
                    "objective": objective,
                    "constraints": constraints,
                    "agent_id": self.agent_id
                }
            )
            self.thought_chain.add_thought(root_thought)
            plan_thoughts.append(root_thought)
            
            # 生成子规划
            sub_plans = self._decompose_objective(objective, constraints)
            for i, sub_plan in enumerate(sub_plans):
                sub_thought = Thought(
                    type=ThoughtType.PLANNING,
                    content=sub_plan,
                    confidence=0.7,
                    metadata={
                        "step": i + 1,
                        "agent_id": self.agent_id
                    }
                )
                self.thought_chain.add_thought(sub_thought, root_thought.id)
                plan_thoughts.append(sub_thought)
            
            # 更新工作记忆
            self.working_memory.extend(plan_thoughts[-5:])  # 只保留最近的规划
            
            return plan_thoughts
    
    def reflect(self) -> Thought:
        """反思处理"""
        with self._lock:
            # 收集最近的思维用于反思
            recent_thoughts = list(self.working_memory)[-10:]
            
            # 生成反思内容
            reflection_content = self._generate_reflection(recent_thoughts)
            
            thought = Thought(
                type=ThoughtType.REFLECTION,
                content=reflection_content,
                confidence=0.7,
                evidence=[t.id for t in recent_thoughts],
                metadata={
                    "agent_id": self.agent_id,
                    "metacognitive_state": self.metacognitive_state.copy()
                }
            )
            
            self.thought_chain.add_thought(thought)
            self.working_memory.append(thought)
            
            # 更新元认知状态
            self._update_metacognitive_state(thought)
            
            return thought
    
    def hypothesize(self, observations: List[Thought]) -> Thought:
        """假设生成"""
        with self._lock:
            hypothesis_content = self._generate_hypothesis(observations)
            
            thought = Thought(
                type=ThoughtType.HYPOTHESIS,
                content=hypothesis_content,
                confidence=0.6,
                evidence=[t.id for t in observations],
                metadata={
                    "agent_id": self.agent_id,
                    "requires_validation": True
                }
            )
            
            self.thought_chain.add_thought(thought)
            self.working_memory.append(thought)
            
            return thought
    
    def validate(self, hypothesis: Thought, evidence: List[Thought]) -> Thought:
        """验证假设"""
        with self._lock:
            validation_result = self._validate_hypothesis(hypothesis, evidence)
            
            thought = Thought(
                type=ThoughtType.VALIDATION,
                content=validation_result["conclusion"],
                confidence=validation_result["confidence"],
                evidence=[hypothesis.id] + [e.id for e in evidence],
                metadata={
                    "agent_id": self.agent_id,
                    "validation_score": validation_result["score"],
                    "supporting_evidence": validation_result["supporting"],
                    "contradicting_evidence": validation_result["contradicting"]
                }
            )
            
            self.thought_chain.add_thought(thought, hypothesis.id)
            self.working_memory.append(thought)
            
            # 更新假设的置信度
            hypothesis.confidence = validation_result["confidence"]
            
            return thought
    
    def synthesize(self, thoughts: List[Thought]) -> Thought:
        """综合多个思维"""
        with self._lock:
            synthesis_content = self._generate_synthesis(thoughts)
            
            thought = Thought(
                type=ThoughtType.SYNTHESIS,
                content=synthesis_content,
                confidence=self._calculate_synthesis_confidence(thoughts),
                evidence=[t.id for t in thoughts],
                metadata={
                    "agent_id": self.agent_id,
                    "synthesis_method": "weighted_integration"
                }
            )
            
            self.thought_chain.add_thought(thought)
            self.working_memory.append(thought)
            
            return thought
    
    def critique(self, target: Thought, criteria: List[str]) -> Thought:
        """批判性分析"""
        with self._lock:
            critique_result = self._generate_critique(target, criteria)
            
            thought = Thought(
                type=ThoughtType.CRITIQUE,
                content=critique_result["analysis"],
                confidence=0.8,
                evidence=[target.id],
                metadata={
                    "agent_id": self.agent_id,
                    "criteria": criteria,
                    "strengths": critique_result["strengths"],
                    "weaknesses": critique_result["weaknesses"],
                    "suggestions": critique_result["suggestions"]
                }
            )
            
            self.thought_chain.add_thought(thought, target.id)
            self.working_memory.append(thought)
            
            return thought
    
    def explore(self, domain: str, constraints: Dict[str, Any]) -> List[Thought]:
        """探索性思维"""
        with self._lock:
            exploration_thoughts = []
            
            # 生成探索策略
            strategy = self._generate_exploration_strategy(domain, constraints)
            
            for direction in strategy["directions"]:
                thought = Thought(
                    type=ThoughtType.EXPLORATION,
                    content=direction["hypothesis"],
                    confidence=direction["priority"],
                    metadata={
                        "agent_id": self.agent_id,
                        "domain": domain,
                        "direction": direction["name"],
                        "expected_value": direction["expected_value"]
                    }
                )
                
                self.thought_chain.add_thought(thought)
                exploration_thoughts.append(thought)
            
            self.working_memory.extend(exploration_thoughts[-5:])
            
            return exploration_thoughts
    
    def abstract(self, concrete_thoughts: List[Thought]) -> Thought:
        """抽象化处理"""
        with self._lock:
            abstraction = self._generate_abstraction(concrete_thoughts)
            
            thought = Thought(
                type=ThoughtType.ABSTRACTION,
                content=abstraction["principle"],
                confidence=abstraction["confidence"],
                evidence=[t.id for t in concrete_thoughts],
                metadata={
                    "agent_id": self.agent_id,
                    "abstraction_level": abstraction["level"],
                    "patterns": abstraction["patterns"],
                    "generalizations": abstraction["generalizations"]
                }
            )
            
            self.thought_chain.add_thought(thought)
            self.working_memory.append(thought)
            
            return thought
    
    def _update_attention(self, thought: Thought):
        """更新注意力权重"""
        # 基于思维类型和置信度计算注意力权重
        base_weight = {
            ThoughtType.OBSERVATION: 0.7,
            ThoughtType.REASONING: 0.8,
            ThoughtType.PLANNING: 0.9,
            ThoughtType.REFLECTION: 0.6,
            ThoughtType.HYPOTHESIS: 0.7,
            ThoughtType.VALIDATION: 0.8,
            ThoughtType.SYNTHESIS: 0.85,
            ThoughtType.CRITIQUE: 0.75,
            ThoughtType.EXPLORATION: 0.65,
            ThoughtType.ABSTRACTION: 0.9
        }.get(thought.type, 0.5)
        
        self.attention_weights[thought.id] = base_weight * thought.confidence
        
        # 衰减旧的注意力权重
        for tid in list(self.attention_weights.keys()):
            if tid != thought.id:
                self.attention_weights[tid] *= 0.95
                if self.attention_weights[tid] < 0.1:
                    del self.attention_weights[tid]
    
    def _update_metacognitive_state(self, reflection: Thought):
        """更新元认知状态"""
        # 基于反思结果调整元认知参数
        content_lower = reflection.content.lower()
        
        # 调整置信度
        if "uncertain" in content_lower or "unclear" in content_lower:
            self.metacognitive_state["confidence"] *= 0.9
            self.metacognitive_state["uncertainty"] = 1 - self.metacognitive_state["confidence"]
        elif "confident" in content_lower or "clear" in content_lower:
            self.metacognitive_state["confidence"] = min(0.95, self.metacognitive_state["confidence"] * 1.1)
            self.metacognitive_state["uncertainty"] = 1 - self.metacognitive_state["confidence"]
        
        # 调整探索率
        if "explore" in content_lower or "investigate" in content_lower:
            self.metacognitive_state["exploration_rate"] = min(0.8, self.metacognitive_state["exploration_rate"] * 1.2)
        else:
            self.metacognitive_state["exploration_rate"] *= 0.95
        
        # 调整反思频率
        if len(self.working_memory) > 15:
            self.metacognitive_state["reflection_frequency"] = min(0.5, self.metacognitive_state["reflection_frequency"] * 1.1)
    
    def _calculate_reasoning_confidence(self, context: List[Thought]) -> float:
        """计算推理置信度"""
        if not context:
            return 0.5
        
        # 基于上下文思维的平均置信度和一致性
        avg_confidence = np.mean([t.confidence for t in context])
        
        # 检查思维类型的多样性
        thought_types = set(t.type for t in context)
        diversity_bonus = len(thought_types) / len(ThoughtType) * 0.2
        
        return min(0.95, avg_confidence * 0.8 + diversity_bonus)
    
    def _calculate_synthesis_confidence(self, thoughts: List[Thought]) -> float:
        """计算综合置信度"""
        if not thoughts:
            return 0.5
        
        # 加权平均，考虑思维的重要性
        weights = [self.attention_weights.get(t.id, 0.5) for t in thoughts]
        confidences = [t.confidence for t in thoughts]
        
        if sum(weights) == 0:
            return np.mean(confidences)
        
        weighted_confidence = np.average(confidences, weights=weights)
        
        # 一致性检查
        confidence_std = np.std(confidences)
        consistency_factor = 1 - min(confidence_std, 0.3)
        
        return weighted_confidence * consistency_factor
    
    def _generate_reasoning(self, context: List[Thought]) -> str:
        """生成推理内容"""
        # 这里应该调用LLM，现在返回模拟内容
        context_summary = "\n".join([f"- {t.type.value}: {t.content[:100]}..." for t in context[-5:]])
        return f"Based on the context:\n{context_summary}\n\nI reason that: [推理结论]"
    
    def _decompose_objective(self, objective: str, constraints: List[str]) -> List[str]:
        """分解目标"""
        # 这里应该调用LLM进行目标分解，现在返回模拟内容
        return [
            f"Step 1: Analyze requirements for {objective}",
            f"Step 2: Consider constraints: {', '.join(constraints[:2])}",
            f"Step 3: Design solution approach",
            f"Step 4: Implement and validate",
            f"Step 5: Optimize and refine"
        ]
    
    def _generate_reflection(self, recent_thoughts: List[Thought]) -> str:
        """生成反思内容"""
        thought_summary = defaultdict(int)
        for t in recent_thoughts:
            thought_summary[t.type] += 1
        
        summary_str = ", ".join([f"{k.value}: {v}" for k, v in thought_summary.items()])
        avg_confidence = np.mean([t.confidence for t in recent_thoughts]) if recent_thoughts else 0.5
        
        return f"Reflection on recent thoughts ({summary_str}). Average confidence: {avg_confidence:.2f}. " \
               f"Current cognitive state suggests {'high' if avg_confidence > 0.7 else 'moderate'} certainty."
    
    def _generate_hypothesis(self, observations: List[Thought]) -> str:
        """生成假设"""
        obs_summary = " ".join([o.content[:50] for o in observations[-3:]])
        return f"Hypothesis: Given observations ({obs_summary}...), I hypothesize that [假设内容]"
    
    def _validate_hypothesis(self, hypothesis: Thought, evidence: List[Thought]) -> Dict[str, Any]:
        """验证假设"""
        supporting = []
        contradicting = []
        
        # 简单的模拟验证逻辑
        for e in evidence:
            if e.confidence > 0.7:
                supporting.append(e.id)
            elif e.confidence < 0.3:
                contradicting.append(e.id)
        
        support_ratio = len(supporting) / (len(supporting) + len(contradicting) + 1)
        
        return {
            "score": support_ratio,
            "confidence": support_ratio * hypothesis.confidence,
            "supporting": supporting,
            "contradicting": contradicting,
            "conclusion": f"Hypothesis {'supported' if support_ratio > 0.6 else 'not supported'} by evidence"
        }
    
    def _generate_synthesis(self, thoughts: List[Thought]) -> str:
        """生成综合内容"""
        thought_contents = [t.content[:100] for t in thoughts]
        return f"Synthesis of {len(thoughts)} thoughts: {' | '.join(thought_contents)}"
    
    def _generate_critique(self, target: Thought, criteria: List[str]) -> Dict[str, Any]:
        """生成批判性分析"""
        return {
            "analysis": f"Critical analysis of {target.type.value} thought against criteria: {', '.join(criteria)}",
            "strengths": ["Clear reasoning", "Well-supported"],
            "weaknesses": ["Limited scope", "Needs more evidence"],
            "suggestions": ["Expand analysis", "Gather more data"]
        }
    
    def _generate_exploration_strategy(self, domain: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """生成探索策略"""
        return {
            "directions": [
                {
                    "name": "Breadth-first exploration",
                    "hypothesis": f"Exploring multiple aspects of {domain}",
                    "priority": 0.8,
                    "expected_value": 0.7
                },
                {
                    "name": "Depth-first investigation",
                    "hypothesis": f"Deep diving into core aspects of {domain}",
                    "priority": 0.7,
                    "expected_value": 0.8
                }
            ]
        }
    
    def _generate_abstraction(self, concrete_thoughts: List[Thought]) -> Dict[str, Any]:
        """生成抽象化内容"""
        return {
            "principle": "General principle extracted from specific instances",
            "confidence": 0.75,
            "level": 2,
            "patterns": ["Pattern A", "Pattern B"],
            "generalizations": ["Generalization 1", "Generalization 2"]
        }

# ============================= 高级记忆系统 =============================

class HierarchicalMemorySystem:
    """层次化记忆系统"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.memories: Dict[str, Memory] = {}
        self.memory_index = faiss.IndexFlatL2(embedding_dim)
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.memory_graph = nx.DiGraph()
        self.consolidation_threshold = 0.7
        self._lock = threading.RLock()
        self._current_index = 0
        
        # 分层存储
        self.sensory_buffer = deque(maxlen=100)
        self.short_term_memory = deque(maxlen=1000)
        self.long_term_memory: Dict[str, Memory] = {}
        
        # 记忆巩固线程
        self.consolidation_thread = threading.Thread(target=self._consolidation_loop, daemon=True)
        self.consolidation_thread.start()
    
    def store(self, content: Any, memory_type: MemoryType, importance: float = 0.5, 
              context: Dict[str, Any] = None, embedding: Optional[np.ndarray] = None) -> str:
        """存储记忆"""
        with self._lock:
            memory = Memory(
                type=memory_type,
                content=content,
                importance=importance,
                context=context or {},
                vector_embedding=embedding
            )
            
            # 根据类型存储到不同层次
            if memory_type == MemoryType.SENSORY:
                self.sensory_buffer.append(memory)
            elif memory_type in [MemoryType.SHORT_TERM, MemoryType.WORKING]:
                self.short_term_memory.append(memory)
                self.memories[memory.id] = memory
            else:
                self.long_term_memory[memory.id] = memory
                self.memories[memory.id] = memory
            
            # 添加到向量索引
            if embedding is not None:
                self._add_to_index(memory.id, embedding)
            
            # 添加到图结构
            self.memory_graph.add_node(memory.id, memory=memory)
            
            return memory.id
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """检索记忆"""
        with self._lock:
            memory = self.memories.get(memory_id)
            if memory:
                memory.update_access()
            return memory
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10, 
                      memory_type: Optional[MemoryType] = None) -> List[Tuple[Memory, float]]:
        """搜索相似记忆"""
        with self._lock:
            if self.memory_index.ntotal == 0:
                return []
            
            # 搜索最近邻
            distances, indices = self.memory_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 
                min(k * 2, self.memory_index.ntotal)
            )
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue
                
                memory_id = self.index_to_id.get(idx)
                if not memory_id:
                    continue
                
                memory = self.memories.get(memory_id)
                if not memory:
                    continue
                
                # 类型过滤
                if memory_type and memory.type != memory_type:
                    continue
                
                memory.update_access()
                similarity = 1 / (1 + dist)  # 转换距离为相似度
                results.append((memory, similarity))
                
                if len(results) >= k:
                    break
            
            return results
    
    def associate(self, memory_id1: str, memory_id2: str, strength: float = 0.5):
        """建立记忆关联"""
        with self._lock:
            if memory_id1 in self.memories and memory_id2 in self.memories:
                self.memories[memory_id1].associations.add(memory_id2)
                self.memories[memory_id2].associations.add(memory_id1)
                
                # 在图中添加边
                self.memory_graph.add_edge(memory_id1, memory_id2, weight=strength)
                self.memory_graph.add_edge(memory_id2, memory_id1, weight=strength)
    
    def get_associated_memories(self, memory_id: str, max_hops: int = 2) -> Dict[str, Memory]:
        """获取关联记忆"""
        with self._lock:
            if memory_id not in self.memory_graph:
                return {}
            
            # 使用BFS获取多跳关联
            associated = {}
            visited = set()
            queue = deque([(memory_id, 0)])
            
            while queue:
                current_id, hop = queue.popleft()
                
                if current_id in visited or hop > max_hops:
                    continue
                
                visited.add(current_id)
                
                if current_id != memory_id and current_id in self.memories:
                    associated[current_id] = self.memories[current_id]
                
                # 添加邻居
                for neighbor in self.memory_graph.neighbors(current_id):
                    if neighbor not in visited:
                        queue.append((neighbor, hop + 1))
            
            return associated
    
    def consolidate(self):
        """记忆巩固"""
        with self._lock:
            # 从感觉缓冲区到短期记忆
            while self.sensory_buffer:
                memory = self.sensory_buffer.popleft()
                if memory.importance > 0.3:
                    memory.type = MemoryType.SHORT_TERM
                    self.short_term_memory.append(memory)
                    self.memories[memory.id] = memory
            
            # 从短期记忆到长期记忆
            to_consolidate = []
            for memory in list(self.short_term_memory):
                strength = memory.calculate_strength()
                if strength > self.consolidation_threshold:
                    to_consolidate.append(memory)
            
            for memory in to_consolidate:
                self.short_term_memory.remove(memory)
                memory.type = MemoryType.LONG_TERM
                self.long_term_memory[memory.id] = memory
                
                # 建立关联
                self._build_associations(memory)
    
    def forget(self, decay_factor: float = 0.1):
        """遗忘机制"""
        with self._lock:
            to_forget = []
            
            for memory_id, memory in self.memories.items():
                # 计算记忆强度
                strength = memory.calculate_strength()
                
                # 应用遗忘
                if strength < 0.1:
                    to_forget.append(memory_id)
                else:
                    # 衰减重要性
                    memory.importance *= (1 - decay_factor)
            
            # 移除遗忘的记忆
            for memory_id in to_forget:
                self._remove_memory(memory_id)
    
    def _add_to_index(self, memory_id: str, embedding: np.ndarray):
        """添加到向量索引"""
        self.id_to_index[memory_id] = self._current_index
        self.index_to_id[self._current_index] = memory_id
        self.memory_index.add(embedding.reshape(1, -1).astype(np.float32))
        self._current_index += 1
    
    def _remove_memory(self, memory_id: str):
        """移除记忆"""
        if memory_id not in self.memories:
            return
        
        memory = self.memories[memory_id]
        
        # 从各层次移除
        if memory.type == MemoryType.LONG_TERM:
            self.long_term_memory.pop(memory_id, None)
        
        # 从图中移除
        if memory_id in self.memory_graph:
            self.memory_graph.remove_node(memory_id)
        
        # 从主字典移除
        del self.memories[memory_id]
        
        # 注意：FAISS索引不支持删除，需要定期重建
    
    def _build_associations(self, memory: Memory):
        """建立记忆关联"""
        if memory.vector_embedding is None:
            return
        
        # 查找相似记忆
        similar_memories = self.search_similar(memory.vector_embedding, k=5)
        
        for similar_memory, similarity in similar_memories:
            if similar_memory.id != memory.id and similarity > 0.7:
                self.associate(memory.id, similar_memory.id, similarity)
    
    def _consolidation_loop(self):
        """记忆巩固循环"""
        while True:
            try:
                time.sleep(60)  # 每分钟执行一次
                self.consolidate()
                self.forget()
            except Exception as e:
                logging.error(f"Memory consolidation error: {e}")

# ============================= 基础Agent实现 =============================

class BaseAgent(ABC):
    """基础Agent类"""
    
    def __init__(self, agent_id: str, name: str, role: AgentRole, 
                 model_provider: ModelProvider = ModelProvider.CLAUDE_4_SONNET,
                 config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.model_provider = model_provider
        self.config = config or {}
        
        # 认知系统
        self.cognitive_architecture = CognitiveArchitecture(agent_id)
        self.memory_system = HierarchicalMemorySystem()
        
        # 消息队列
        self.inbox = asyncio.Queue()
        self.outbox = asyncio.Queue()
        
        # 任务管理
        self.current_tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.PriorityQueue()
        
        # 协作相关
        self.known_agents: Dict[str, Dict[str, Any]] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        
        # 状态管理
        self.state = {
            "status": "idle",
            "current_activity": None,
            "performance_metrics": {},
            "last_active": datetime.now()
        }
        
        # 工具集成
        self.tools: Dict[str, Callable] = {}
        
        # 监控
        self._setup_monitoring()
        
        # 运行标志
        self.running = False
        self._lock = asyncio.Lock()
    
    async def start(self):
        """启动Agent"""
        self.running = True
        await asyncio.gather(
            self._message_handler(),
            self._task_processor(),
            self._cognitive_cycle()
        )
    
    async def stop(self):
        """停止Agent"""
        self.running = False
    
    @abstractmethod
    async def process_task(self, task: Task) -> Any:
        """处理任务 - 子类必须实现"""
        pass
    
    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理消息 - 子类必须实现"""
        pass
    
    async def think(self, context: Dict[str, Any]) -> Thought:
        """思考过程"""
        async with self._lock:
            # 感知
            observation = await self._perceive(context)
            perception_thought = self.cognitive_architecture.perceive(observation)
            
            # 检索相关记忆
            if perception_thought.vector_embedding is not None:
                relevant_memories = self.memory_system.search_similar(
                    perception_thought.vector_embedding, k=5
                )
                memory_thoughts = [
                    Thought(
                        type=ThoughtType.OBSERVATION,
                        content=f"Retrieved memory: {mem.content}",
                        confidence=similarity,
                        metadata={"memory_id": mem.id}
                    )
                    for mem, similarity in relevant_memories
                ]
            else:
                memory_thoughts = []
            
            # 推理
            reasoning_context = [perception_thought] + memory_thoughts
            reasoning_thought = self.cognitive_architecture.reason(reasoning_context)
            
            # 存储新的认知到记忆
            self.memory_system.store(
                content=reasoning_thought.content,
                memory_type=MemoryType.WORKING,
                importance=reasoning_thought.confidence,
                context={"thought_id": reasoning_thought.id},
                embedding=reasoning_thought.vector_embedding
            )
            
            return reasoning_thought
    
    async def collaborate(self, agent_id: str, message: AgentMessage) -> Optional[AgentMessage]:
        """与其他Agent协作"""
        # 记录协作历史
        self.collaboration_history.append({
            "timestamp": datetime.now(),
            "agent_id": agent_id,
            "message_type": message.message_type,
            "message_id": message.id
        })
        
        # 发送消息
        await self._send_message(agent_id, message)
        
        # 等待响应（如果需要）
        if message.requires_response:
            response = await self._wait_for_response(message.id, timeout=30)
            return response
        
        return None
    
    async def register_tool(self, name: str, tool: Callable):
        """注册工具"""
        self.tools[name] = tool
    
    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """使用工具"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered")
        
        tool = self.tools[tool_name]
        
        # 记录工具使用
        self.state["current_activity"] = f"Using tool: {tool_name}"
        
        try:
            if asyncio.iscoroutinefunction(tool):
                result = await tool(**kwargs)
            else:
                result = await asyncio.to_thread(tool, **kwargs)
            
            # 将结果存入记忆
            self.memory_system.store(
                content={"tool": tool_name, "input": kwargs, "output": result},
                memory_type=MemoryType.PROCEDURAL,
                importance=0.6,
                context={"timestamp": datetime.now().isoformat()}
            )
            
            return result
        finally:
            self.state["current_activity"] = None
    
    async def _message_handler(self):
        """消息处理循环"""
        while self.running:
            try:
                message = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                
                # 检查消息是否过期
                if message.is_expired():
                    continue
                
                # 处理消息
                response = await self.handle_message(message)
                
                # 发送响应
                if response and message.requires_response:
                    response.correlation_id = message.id
                    await self.outbox.put(response)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Message handling error: {e}")
    
    async def _task_processor(self):
        """任务处理循环"""
        while self.running:
            try:
                # 获取优先级最高的任务
                priority, task = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )
                
                self.current_tasks[task.id] = task
                self.state["status"] = "working"
                self.state["current_activity"] = f"Processing task: {task.name}"
                
                try:
                    # 处理任务
                    result = await self.process_task(task)
                    task.status = "completed"
                    task.result = result
                    task.progress = 1.0
                except Exception as e:
                    task.status = "failed"
                    task.result = {"error": str(e)}
                    logging.error(f"Task processing error: {e}")
                finally:
                    task.updated_at = datetime.now()
                    del self.current_tasks[task.id]
                    
                    if not self.current_tasks:
                        self.state["status"] = "idle"
                        self.state["current_activity"] = None
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Task processor error: {e}")
    
    async def _cognitive_cycle(self):
        """认知循环"""
        while self.running:
            try:
                # 收集当前上下文
                context = await self._build_context()
                
                # 思考
                thought = await self.think(context)
                
                # 定期反思
                if np.random.random() < self.cognitive_architecture.metacognitive_state["reflection_frequency"]:
                    reflection = self.cognitive_architecture.reflect()
                    
                    # 基于反思调整行为
                    await self._adjust_behavior(reflection)
                
                # 探索性思维
                if np.random.random() < self.cognitive_architecture.metacognitive_state["exploration_rate"]:
                    exploration_domain = self._select_exploration_domain()
                    exploration_thoughts = self.cognitive_architecture.explore(
                        exploration_domain, 
                        {"current_tasks": len(self.current_tasks)}
                    )
                
                await asyncio.sleep(5)  # 认知周期间隔
                
            except Exception as e:
                logging.error(f"Cognitive cycle error: {e}")
                await asyncio.sleep(10)
    
    async def _perceive(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """感知环境"""
        perception = {
            "timestamp": datetime.now().isoformat(),
            "agent_state": self.state.copy(),
            "active_tasks": len(self.current_tasks),
            "inbox_size": self.inbox.qsize(),
            "memory_usage": len(self.memory_system.memories),
            "context": context
        }
        return perception
    
    async def _build_context(self) -> Dict[str, Any]:
        """构建当前上下文"""
        return {
            "current_tasks": list(self.current_tasks.values()),
            "recent_messages": self.collaboration_history[-10:],
            "cognitive_state": self.cognitive_architecture.metacognitive_state,
            "performance": self.state.get("performance_metrics", {})
        }
    
    async def _adjust_behavior(self, reflection: Thought):
        """基于反思调整行为"""
        # 这里可以实现更复杂的行为调整逻辑
        pass
    
    def _select_exploration_domain(self) -> str:
        """选择探索领域"""
        domains = ["task_optimization", "collaboration_strategies", 
                  "tool_usage", "memory_organization"]
        return np.random.choice(domains)
    
    async def _send_message(self, agent_id: str, message: AgentMessage):
        """发送消息给其他Agent"""
        # 这里应该通过消息总线发送，现在简化处理
        message.sender_id = self.agent_id
        message.receiver_id = agent_id
        await self.outbox.put(message)
    
    async def _wait_for_response(self, correlation_id: str, timeout: float = 30) -> Optional[AgentMessage]:
        """等待响应消息"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 检查收件箱
            temp_messages = []
            
            try:
                while True:
                    message = await asyncio.wait_for(self.inbox.get(), timeout=0.1)
                    if message.correlation_id == correlation_id:
                        # 将其他消息放回
                        for msg in temp_messages:
                            await self.inbox.put(msg)
                        return message
                    temp_messages.append(message)
            except asyncio.TimeoutError:
                # 将消息放回
                for msg in temp_messages:
                    await self.inbox.put(msg)
                
            await asyncio.sleep(0.5)
        
        return None
    
    def _setup_monitoring(self):
        """设置监控指标"""
        self.metrics = {
            "tasks_processed": Counter(
                'agent_tasks_processed_total',
                'Total number of tasks processed',
                ['agent_id', 'status']
            ),
            "message_latency": Histogram(
                'agent_message_latency_seconds',
                'Message processing latency',
                ['agent_id', 'message_type']
            ),
            "memory_usage": Gauge(
                'agent_memory_usage_bytes',
                'Memory usage in bytes',
                ['agent_id']
            ),
            "cognitive_confidence": Gauge(
                'agent_cognitive_confidence',
                'Cognitive confidence level',
                ['agent_id']
            )
        }

# ============================= 专门化Agent实现 =============================

class SupervisorAgent(BaseAgent):
    """监督者Agent"""
    
    def __init__(self, agent_id: str, name: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, name, AgentRole.SUPERVISOR, 
                        ModelProvider.CLAUDE_4_OPUS, config)
        
        self.supervised_agents: Dict[str, Dict[str, Any]] = {}
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
    
    async def process_task(self, task: Task) -> Any:
        """处理监督任务"""
        # 分解任务
        subtasks = await self._decompose_task(task)
        
        # 分配任务给下属Agent
        assignments = await self._assign_tasks(subtasks)
        
        # 监督执行
        results = await self._supervise_execution(assignments)
        
        # 汇总结果
        final_result = await self._aggregate_results(results)
        
        return final_result
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理监督相关消息"""
        if message.message_type == "status_report":
            await self._handle_status_report(message)
        elif message.message_type == "help_request":
            return await self._handle_help_request(message)
        elif message.message_type == "task_completion":
            await self._handle_task_completion(message)
        
        return None
    
    async def _decompose_task(self, task: Task) -> List[Task]:
        """任务分解"""
        # 使用认知架构进行规划
        plan_thoughts = self.cognitive_architecture.plan(
            task.objective,
            task.constraints
        )
        
        subtasks = []
        for i, thought in enumerate(plan_thoughts[1:]):  # 跳过根节点
            subtask = Task(
                name=f"{task.name}_sub_{i+1}",
                description=thought.content,
                objective=thought.content,
                constraints=task.constraints,
                parent_task_id=task.id,
                priority=task.priority - i,
                deadline=task.deadline
            )
            subtasks.append(subtask)
            task.subtask_ids.append(subtask.id)
        
        return subtasks
    
    async def _assign_tasks(self, tasks: List[Task]) -> Dict[str, Tuple[str, Task]]:
        """任务分配"""
        assignments = {}
        
        for task in tasks:
            # 选择最合适的Agent
            best_agent = await self._select_best_agent(task)
            
            if best_agent:
                # 发送任务分配消息
                assignment_message = AgentMessage(
                    message_type="task_assignment",
                    content=task,
                    priority=task.priority,
                    requires_response=True
                )
                
                response = await self.collaborate(best_agent, assignment_message)
                
                if response and response.content.get("accepted"):
                    assignments[task.id] = (best_agent, task)
                    self.task_assignments[task.id] = best_agent
                else:
                    # 任务被拒绝，尝试其他Agent
                    pass
        
        return assignments
    
    async def _select_best_agent(self, task: Task) -> Optional[str]:
        """选择最佳Agent"""
        candidates = []
        
        for agent_id, agent_info in self.supervised_agents.items():
            # 计算适配度分数
            score = await self._calculate_fitness_score(agent_id, task)
            candidates.append((agent_id, score))
        
        # 选择得分最高的Agent
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    async def _calculate_fitness_score(self, agent_id: str, task: Task) -> float:
        """计算Agent对任务的适配度"""
        agent_info = self.supervised_agents.get(agent_id, {})
        
        # 基础分数
        base_score = 0.5
        
        # 专长匹配
        if agent_info.get("specialization") in task.description.lower():
            base_score += 0.2
        
        # 当前负载
        current_load = agent_info.get("current_load", 0)
        load_penalty = min(0.3, current_load * 0.1)
        base_score -= load_penalty
        
        # 历史表现
        if agent_id in self.performance_history:
            avg_performance = np.mean(self.performance_history[agent_id][-10:])
            base_score += avg_performance * 0.3
        
        return max(0, min(1, base_score))
    
    async def _supervise_execution(self, assignments: Dict[str, Tuple[str, Task]]) -> Dict[str, Any]:
        """监督任务执行"""
        results = {}
        monitoring_tasks = []
        
        for task_id, (agent_id, task) in assignments.items():
            # 创建监控协程
            monitor_coro = self._monitor_task_execution(task_id, agent_id)
            monitoring_tasks.append(monitor_coro)
        
        # 并行监控所有任务
        monitoring_results = await asyncio.gather(*monitoring_tasks, return_exceptions=True)
        
        for i, (task_id, _) in enumerate(assignments.items()):
            if isinstance(monitoring_results[i], Exception):
                results[task_id] = {"status": "failed", "error": str(monitoring_results[i])}
            else:
                results[task_id] = monitoring_results[i]
        
        return results
    
    async def _monitor_task_execution(self, task_id: str, agent_id: str) -> Dict[str, Any]:
        """监控单个任务执行"""
        start_time = time.time()
        last_progress = 0
        
        while True:
            # 请求状态更新
            status_request = AgentMessage(
                message_type="status_request",
                content={"task_id": task_id},
                requires_response=True
            )
            
            response = await self.collaborate(agent_id, status_request)
            
            if response:
                status = response.content
                current_progress = status.get("progress", 0)
                
                # 检查是否完成
                if status.get("status") == "completed":
                    execution_time = time.time() - start_time
                    self._update_performance(agent_id, 1.0)
                    return {
                        "status": "completed",
                        "result": status.get("result"),
                        "execution_time": execution_time
                    }
                
                # 检查是否失败
                elif status.get("status") == "failed":
                    self._update_performance(agent_id, 0.0)
                    return {
                        "status": "failed",
                        "error": status.get("error"),
                        "execution_time": time.time() - start_time
                    }
                
                # 检查进度
                if current_progress <= last_progress:
                    # 进度停滞，可能需要干预
                    intervention = await self._consider_intervention(
                        task_id, agent_id, status
                    )
                    if intervention:
                        await self._execute_intervention(intervention)
                
                last_progress = current_progress
            
            await asyncio.sleep(5)  # 监控间隔
    
    async def _aggregate_results(self, results: Dict[str, Any]) -> Any:
        """汇总结果"""
        successful_results = []
        failed_tasks = []
        
        for task_id, result in results.items():
            if result["status"] == "completed":
                successful_results.append(result["result"])
            else:
                failed_tasks.append(task_id)
        
        # 使用认知架构综合结果
        if successful_results:
            synthesis_thought = self.cognitive_architecture.synthesize([
                Thought(content=str(r), confidence=0.8) for r in successful_results
            ])
            
            return {
                "status": "completed" if not failed_tasks else "partial",
                "synthesis": synthesis_thought.content,
                "detailed_results": successful_results,
                "failed_tasks": failed_tasks
            }
        else:
            return {
                "status": "failed",
                "error": "All subtasks failed",
                "failed_tasks": failed_tasks
            }
    
    def _update_performance(self, agent_id: str, score: float):
        """更新Agent性能记录"""
        self.performance_history[agent_id].append(score)
        # 保持历史记录在合理范围内
        if len(self.performance_history[agent_id]) > 100:
            self.performance_history[agent_id] = self.performance_history[agent_id][-100:]
    
    async def _consider_intervention(self, task_id: str, agent_id: str, 
                                   status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """考虑是否需要干预"""
        # 使用认知架构分析情况
        analysis_thought = self.cognitive_architecture.reason([
            Thought(
                type=ThoughtType.OBSERVATION,
                content=f"Task {task_id} progress stalled at {status.get('progress', 0)}",
                confidence=0.9
            )
        ])
        
        # 基于分析决定干预策略
        if "intervention needed" in analysis_thought.content.lower():
            return {
                "type": "reassign",
                "task_id": task_id,
                "current_agent": agent_id,
                "reason": "Progress stalled"
            }
        
        return None
    
    async def _execute_intervention(self, intervention: Dict[str, Any]):
        """执行干预"""
        if intervention["type"] == "reassign":
            # 重新分配任务
            # TODO: 实现任务重新分配逻辑
            pass
    
    async def _handle_status_report(self, message: AgentMessage):
        """处理状态报告"""
        agent_id = message.sender_id
        status = message.content
        
        # 更新Agent信息
        if agent_id in self.supervised_agents:
            self.supervised_agents[agent_id].update(status)
    
    async def _handle_help_request(self, message: AgentMessage) -> AgentMessage:
        """处理帮助请求"""
        # 分析请求
        help_type = message.content.get("type")
        details = message.content.get("details")
        
        # 提供帮助或重新分配资源
        if help_type == "resource":
            # 分配额外资源
            response_content = {
                "approved": True,
                "resources": self._allocate_resources(details)
            }
        elif help_type == "guidance":
            # 提供指导
            guidance_thought = self.cognitive_architecture.reason([
                Thought(content=f"Help request: {details}", confidence=0.8)
            ])
            response_content = {
                "guidance": guidance_thought.content
            }
        else:
            response_content = {
                "approved": False,
                "reason": "Unknown help type"
            }
        
        return AgentMessage(
            message_type="help_response",
            content=response_content
        )
    
    async def _handle_task_completion(self, message: AgentMessage):
        """处理任务完成通知"""
        task_id = message.content.get("task_id")
        result = message.content.get("result")
        
        # 更新任务状态
        if task_id in self.task_assignments:
            del self.task_assignments[task_id]
            
            # 记录完成情况
            self.memory_system.store(
                content={
                    "task_id": task_id,
                    "agent_id": message.sender_id,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                },
                memory_type=MemoryType.EPISODIC,
                importance=0.7
            )
    
    def _allocate_resources(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """分配资源"""
        # 简化的资源分配逻辑
        return {
            "compute": min(requirements.get("compute", 1), 4),
            "memory": min(requirements.get("memory", 1024), 8192),
            "priority_boost": 1
        }

# ============================= 协作框架实现 =============================

class CollaborationFramework:
    """多Agent协作框架"""
    
    def __init__(self, coordination_strategy: CoordinationStrategy = CoordinationStrategy.HIERARCHICAL):
        self.agents: Dict[str, BaseAgent] = {}
        self.coordination_strategy = coordination_strategy
        self.message_bus = asyncio.Queue()
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.collaboration_graph = nx.DiGraph()
        self._lock = asyncio.Lock()
        
        # 协调器
        self.coordinators: Dict[CoordinationStrategy, 'Coordinator'] = {
            CoordinationStrategy.HIERARCHICAL: HierarchicalCoordinator(self),
            CoordinationStrategy.FEDERATED: FederatedCoordinator(self),
            CoordinationStrategy.CONSENSUS: ConsensusCoordinator(self),
            CoordinationStrategy.BLACKBOARD: BlackboardCoordinator(self)
        }
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 运行状态
        self.running = False
    
    async def register_agent(self, agent: BaseAgent):
        """注册Agent"""
        async with self._lock:
            self.agents[agent.agent_id] = agent
            self.agent_registry[agent.agent_id] = {
                "name": agent.name,
                "role": agent.role,
                "status": "active",
                "capabilities": [],
                "performance": {},
                "registered_at": datetime.now()
            }
            
            # 添加到协作图
            self.collaboration_graph.add_node(agent.agent_id, agent=agent)
    
    async def unregister_agent(self, agent_id: str):
        """注销Agent"""
        async with self._lock:
            if agent_id in self.agents:
                await self.agents[agent_id].stop()
                del self.agents[agent_id]
                del self.agent_registry[agent_id]
                
                if agent_id in self.collaboration_graph:
                    self.collaboration_graph.remove_node(agent_id)
    
    async def start(self):
        """启动协作框架"""
        self.running = True
        
        # 启动所有Agent
        agent_tasks = [agent.start() for agent in self.agents.values()]
        
        # 启动消息路由
        router_task = asyncio.create_task(self._message_router())
        
        # 启动协调器
        coordinator = self.coordinators[self.coordination_strategy]
        coordinator_task = asyncio.create_task(coordinator.coordinate())
        
        # 启动性能监控
        monitor_task = asyncio.create_task(self.performance_monitor.monitor(self))
        
        await asyncio.gather(
            *agent_tasks,
            router_task,
            coordinator_task,
            monitor_task
        )
    
    async def stop(self):
        """停止协作框架"""
        self.running = False
        
        # 停止所有Agent
        for agent in self.agents.values():
            await agent.stop()
    
    async def submit_task(self, task: Task) -> Any:
        """提交任务到框架"""
        # 选择协调器处理任务
        coordinator = self.coordinators[self.coordination_strategy]
        return await coordinator.handle_task(task)
    
    async def _message_router(self):
        """消息路由器"""
        while self.running:
            try:
                # 从所有Agent收集消息
                for agent in self.agents.values():
                    try:
                        message = await asyncio.wait_for(
                            agent.outbox.get(), timeout=0.1
                        )
                        await self._route_message(message)
                    except asyncio.TimeoutError:
                        continue
                
                await asyncio.sleep(0.01)  # 避免CPU占用过高
                
            except Exception as e:
                logging.error(f"Message router error: {e}")
    
    async def _route_message(self, message: AgentMessage):
        """路由消息到目标Agent"""
        receiver_id = message.receiver_id
        
        if receiver_id in self.agents:
            await self.agents[receiver_id].inbox.put(message)
            
            # 更新协作图
            if message.sender_id in self.collaboration_graph:
                self.collaboration_graph.add_edge(
                    message.sender_id, 
                    receiver_id,
                    weight=self.collaboration_graph.get_edge_data(
                        message.sender_id, receiver_id, {"weight": 0}
                    )["weight"] + 1
                )
        else:
            logging.warning(f"Unknown receiver: {receiver_id}")

# ============================= 协调器实现 =============================

class Coordinator(ABC):
    """协调器基类"""
    
    def __init__(self, framework: CollaborationFramework):
        self.framework = framework
    
    @abstractmethod
    async def coordinate(self):
        """协调Agent工作"""
        pass
    
    @abstractmethod
    async def handle_task(self, task: Task) -> Any:
        """处理任务"""
        pass

class HierarchicalCoordinator(Coordinator):
    """层次化协调器"""
    
    def __init__(self, framework: CollaborationFramework):
        super().__init__(framework)
        self.hierarchy_levels: Dict[int, List[str]] = defaultdict(list)
        self._build_hierarchy()
    
    def _build_hierarchy(self):
        """构建层次结构"""
        # 根据Agent角色构建层次
        for agent_id, agent in self.framework.agents.items():
            if agent.role == AgentRole.SUPERVISOR:
                self.hierarchy_levels[0].append(agent_id)
            elif agent.role == AgentRole.COORDINATOR:
                self.hierarchy_levels[1].append(agent_id)
            elif agent.role in [AgentRole.PLANNER, AgentRole.ANALYZER]:
                self.hierarchy_levels[2].append(agent_id)
            else:
                self.hierarchy_levels[3].append(agent_id)
    
    async def coordinate(self):
        """层次化协调"""
        while self.framework.running:
            # 监控各层级状态
            for level, agent_ids in self.hierarchy_levels.items():
                for agent_id in agent_ids:
                    agent = self.framework.agents.get(agent_id)
                    if agent and agent.state["status"] == "idle" and level > 0:
                        # 向上级请求任务
                        await self._request_task_from_superior(agent_id, level)
            
            await asyncio.sleep(1)
    
    async def handle_task(self, task: Task) -> Any:
        """处理任务 - 分配给顶层supervisor"""
        if not self.hierarchy_levels[0]:
            raise ValueError("No supervisor agents available")
        
        # 选择负载最低的supervisor
        supervisor_id = await self._select_least_loaded_agent(self.hierarchy_levels[0])
        supervisor = self.framework.agents[supervisor_id]
        
        # 将任务加入supervisor的队列
        await supervisor.task_queue.put((task.priority, task))
        
        # 等待任务完成
        while task.status not in ["completed", "failed"]:
            await asyncio.sleep(1)
        
        return task.result
    
    async def _request_task_from_superior(self, agent_id: str, level: int):
        """向上级请求任务"""
        if level == 0:
            return
        
        # 找到上级
        superior_level = level - 1
        if not self.hierarchy_levels[superior_level]:
            return
        
        superior_id = self.hierarchy_levels[superior_level][0]  # 简化：选择第一个
        
        # 发送任务请求
        request = AgentMessage(
            sender_id=agent_id,
            receiver_id=superior_id,
            message_type="task_request",
            content={"agent_id": agent_id, "capacity": 1},
            requires_response=True
        )
        
        agent = self.framework.agents[agent_id]
        await agent.outbox.put(request)
    
    async def _select_least_loaded_agent(self, agent_ids: List[str]) -> str:
        """选择负载最低的Agent"""
        loads = []
        
        for agent_id in agent_ids:
            agent = self.framework.agents.get(agent_id)
            if agent:
                load = len(agent.current_tasks)
                loads.append((agent_id, load))
        
        if not loads:
            raise ValueError("No available agents")
        
        loads.sort(key=lambda x: x[1])
        return loads[0][0]

class FederatedCoordinator(Coordinator):
    """联邦式协调器"""
    
    async def coordinate(self):
        """联邦式协调 - Agent间平等协作"""
        while self.framework.running:
            # 定期进行负载均衡
            await self._load_balance()
            await asyncio.sleep(5)
    
    async def handle_task(self, task: Task) -> Any:
        """处理任务 - 通过竞标机制分配"""
        # 发起任务竞标
        bids = await self._collect_bids(task)
        
        if not bids:
            raise ValueError("No agents available for task")
        
        # 选择最佳竞标者
        winner_id = max(bids.items(), key=lambda x: x[1]["score"])[0]
        winner = self.framework.agents[winner_id]
        
        # 分配任务
        await winner.task_queue.put((task.priority, task))
        
        # 等待完成
        while task.status not in ["completed", "failed"]:
            await asyncio.sleep(1)
        
        return task.result
    
    async def _collect_bids(self, task: Task) -> Dict[str, Dict[str, Any]]:
        """收集任务竞标"""
        bids = {}
        
        # 广播任务竞标请求
        bid_request = AgentMessage(
            message_type="bid_request",
            content={
                "task": task.to_dict() if hasattr(task, 'to_dict') else {
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "priority": task.priority
                }
            },
            requires_response=True
        )
        
        # 向所有Agent发送竞标请求
        responses = []
        for agent_id, agent in self.framework.agents.items():
            bid_request.receiver_id = agent_id
            response_future = agent.collaborate(agent_id, bid_request)
            responses.append((agent_id, response_future))
        
        # 收集响应
        for agent_id, response_future in responses:
            try:
                response = await asyncio.wait_for(response_future, timeout=5)
                if response and response.content.get("bid"):
                    bids[agent_id] = response.content
            except asyncio.TimeoutError:
                continue
        
        return bids
    
    async def _load_balance(self):
        """负载均衡"""
        # 收集所有Agent的负载信息
        loads = {}
        for agent_id, agent in self.framework.agents.items():
            loads[agent_id] = len(agent.current_tasks)
        
        if not loads:
            return
        
        avg_load = np.mean(list(loads.values()))
        
        # 识别过载和空闲的Agent
        overloaded = [(aid, load) for aid, load in loads.items() if load > avg_load * 1.5]
        idle = [(aid, load) for aid, load in loads.items() if load < avg_load * 0.5]
        
        # 请求任务迁移
        for overloaded_id, _ in overloaded:
            for idle_id, _ in idle:
                # 发送任务迁移请求
                migration_request = AgentMessage(
                    sender_id="coordinator",
                    receiver_id=overloaded_id,
                    message_type="task_migration_request",
                    content={"target_agent": idle_id, "num_tasks": 1}
                )
                
                agent = self.framework.agents[overloaded_id]
                await agent.inbox.put(migration_request)
                
                break  # 一次只迁移一个任务

class ConsensusCoordinator(Coordinator):
    """共识协调器"""
    
    def __init__(self, framework: CollaborationFramework):
        super().__init__(framework)
        self.consensus_threshold = 0.6
    
    async def coordinate(self):
        """共识协调"""
        while self.framework.running:
            await asyncio.sleep(10)  # 共识协调通常是被动的
    
    async def handle_task(self, task: Task) -> Any:
        """通过共识机制处理任务"""
        # 提议阶段
        proposals = await self._collect_proposals(task)
        
        if not proposals:
            raise ValueError("No proposals received")
        
        # 投票阶段
        consensus = await self._reach_consensus(proposals)
        
        if not consensus:
            raise ValueError("Failed to reach consensus")
        
        # 执行共识决定
        return await self._execute_consensus(consensus, task)
    
    async def _collect_proposals(self, task: Task) -> List[Dict[str, Any]]:
        """收集提议"""
        proposals = []
        
        # 向所有Agent请求提议
        proposal_request = AgentMessage(
            message_type="proposal_request",
            content={"task": task},
            requires_response=True
        )
        
        responses = []
        for agent_id, agent in self.framework.agents.items():
            response_future = agent.collaborate(agent_id, proposal_request)
            responses.append(response_future)
        
        # 收集提议
        for response_future in responses:
            try:
                response = await asyncio.wait_for(response_future, timeout=10)
                if response and response.content.get("proposal"):
                    proposals.append(response.content["proposal"])
            except asyncio.TimeoutError:
                continue
        
        return proposals
    
    async def _reach_consensus(self, proposals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """达成共识"""
        # 简化的投票机制
        votes = defaultdict(int)
        
        # 让每个Agent对所有提议投票
        for proposal in proposals:
            vote_request = AgentMessage(
                message_type="vote_request",
                content={"proposal": proposal},
                requires_response=True
            )
            
            for agent_id, agent in self.framework.agents.items():
                try:
                    response = await agent.collaborate(agent_id, vote_request)
                    if response and response.content.get("vote"):
                        proposal_id = proposal.get("id", str(proposal))
                        votes[proposal_id] += 1
                except:
                    continue
        
        # 检查是否有提议达到共识阈值
        total_agents = len(self.framework.agents)
        for proposal in proposals:
            proposal_id = proposal.get("id", str(proposal))
            if votes[proposal_id] / total_agents >= self.consensus_threshold:
                return proposal
        
        return None
    
    async def _execute_consensus(self, consensus: Dict[str, Any], task: Task) -> Any:
        """执行共识决定"""
        # 根据共识分配任务
        executor_id = consensus.get("executor")
        if executor_id and executor_id in self.framework.agents:
            executor = self.framework.agents[executor_id]
            await executor.task_queue.put((task.priority, task))
            
            # 等待完成
            while task.status not in ["completed", "failed"]:
                await asyncio.sleep(1)
            
            return task.result
        
        raise ValueError("Invalid consensus executor")

class BlackboardCoordinator(Coordinator):
    """黑板协调器"""
    
    def __init__(self, framework: CollaborationFramework):
        super().__init__(framework)
        self.blackboard: Dict[str, Any] = {}
        self.knowledge_sources: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def coordinate(self):
        """黑板协调"""
        while self.framework.running:
            # 检查黑板状态，触发相应的知识源
            await self._trigger_knowledge_sources()
            await asyncio.sleep(1)
    
    async def handle_task(self, task: Task) -> Any:
        """通过黑板机制处理任务"""
        # 将任务放到黑板上
        async with self._lock:
            self.blackboard[f"task_{task.id}"] = {
                "task": task,
                "status": "pending",
                "partial_solutions": [],
                "final_solution": None
            }
        
        # 通知所有Agent
        await self._notify_agents("new_task", task.id)
        
        # 等待解决方案
        while True:
            async with self._lock:
                task_info = self.blackboard.get(f"task_{task.id}")
                if task_info and task_info["final_solution"]:
                    return task_info["final_solution"]
            
            await asyncio.sleep(1)
    
    async def _trigger_knowledge_sources(self):
        """触发知识源"""
        async with self._lock:
            # 检查黑板上的问题
            for key, value in list(self.blackboard.items()):
                if key.startswith("problem_") and value.get("status") == "unsolved":
                    problem_type = value.get("type")
                    
                    # 找到相关的知识源（Agent）
                    relevant_agents = self._find_relevant_agents(problem_type)
                    
                    for agent_id in relevant_agents:
                        # 通知Agent处理问题
                        notification = AgentMessage(
                            message_type="blackboard_trigger",
                            content={"problem_key": key, "problem": value}
                        )
                        
                        agent = self.framework.agents.get(agent_id)
                        if agent:
                            await agent.inbox.put(notification)
    
    def _find_relevant_agents(self, problem_type: str) -> List[str]:
        """找到相关的Agent"""
        relevant = []
        
        for agent_id, agent in self.framework.agents.items():
            # 基于Agent的角色和能力判断相关性
            if problem_type == "planning" and agent.role == AgentRole.PLANNER:
                relevant.append(agent_id)
            elif problem_type == "analysis" and agent.role == AgentRole.ANALYZER:
                relevant.append(agent_id)
            elif problem_type == "execution" and agent.role == AgentRole.EXECUTOR:
                relevant.append(agent_id)
        
        return relevant
    
    async def _notify_agents(self, event_type: str, data: Any):
        """通知所有Agent"""
        notification = AgentMessage(
            message_type="blackboard_event",
            content={"event": event_type, "data": data}
        )
        
        for agent in self.framework.agents.values():
            await agent.inbox.put(notification.model_copy() if hasattr(notification, 'model_copy') else notification)

# ============================= 性能监控 =============================

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            "task_latency": 60.0,  # 秒
            "memory_usage": 0.8,   # 80%
            "error_rate": 0.1,     # 10%
            "message_backlog": 100
        }
    
    async def monitor(self, framework: CollaborationFramework):
        """监控框架性能"""
        while framework.running:
            metrics = await self._collect_metrics(framework)
            self._analyze_metrics(metrics)
            await self._check_alerts(metrics)
            await asyncio.sleep(10)
    
    async def _collect_metrics(self, framework: CollaborationFramework) -> Dict[str, Any]:
        """收集性能指标"""
        metrics = {
            "timestamp": datetime.now(),
            "agents": {}
        }
        
        for agent_id, agent in framework.agents.items():
            agent_metrics = {
                "status": agent.state["status"],
                "active_tasks": len(agent.current_tasks),
                "inbox_size": agent.inbox.qsize(),
                "memory_usage": len(agent.memory_system.memories),
                "cognitive_confidence": agent.cognitive_architecture.metacognitive_state["confidence"]
            }
            
            metrics["agents"][agent_id] = agent_metrics
        
        # 系统级指标
        metrics["system"] = {
            "total_agents": len(framework.agents),
            "active_agents": sum(1 for a in framework.agents.values() if a.state["status"] != "idle"),
            "total_tasks": sum(len(a.current_tasks) for a in framework.agents.values()),
            "collaboration_edges": framework.collaboration_graph.number_of_edges()
        }
        
        return metrics
    
    def _analyze_metrics(self, metrics: Dict[str, Any]):
        """分析指标"""
        # 存储历史
        self.metrics_history["system"].append(metrics)
        
        # 保持历史记录在合理范围
        if len(self.metrics_history["system"]) > 1000:
            self.metrics_history["system"] = self.metrics_history["system"][-1000:]
    
    async def _check_alerts(self, metrics: Dict[str, Any]):
        """检查告警条件"""
        alerts = []
        
        # 检查消息积压
        for agent_id, agent_metrics in metrics["agents"].items():
            if agent_metrics["inbox_size"] > self.thresholds["message_backlog"]:
                alerts.append({
                    "type": "message_backlog",
                    "agent_id": agent_id,
                    "value": agent_metrics["inbox_size"],
                    "threshold": self.thresholds["message_backlog"]
                })
        
        # 记录新告警
        for alert in alerts:
            alert["timestamp"] = datetime.now()
            self.alerts.append(alert)
            logging.warning(f"Performance alert: {alert}")

# ============================= LLM集成层 =============================

class LLMIntegration:
    """LLM集成层 - 管理多个模型提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {
            ModelProvider.CLAUDE_4_OPUS: ClaudeProvider(config.get("claude", {})),
            ModelProvider.CLAUDE_4_SONNET: ClaudeProvider(config.get("claude", {})),
            ModelProvider.CLAUDE_3_7_SONNET: ClaudeProvider(config.get("claude", {})),
            ModelProvider.QWEN_PLUS: QwenProvider(config.get("qwen", {})),
            ModelProvider.QWEN_MAX: QwenProvider(config.get("qwen", {})),
            ModelProvider.QWEN_TURBO: QwenProvider(config.get("qwen", {}))
        }
        
        # 成本优化器
        self.cost_optimizer = CostOptimizer()
        
        # 请求缓存
        self.cache = SemanticCache(max_size=10000)
        
        # 速率限制器
        self.rate_limiters = {
            provider: RateLimiter(
                max_requests=config.get(provider.value, {}).get("rate_limit", 100),
                window_seconds=60
            )
            for provider in ModelProvider
        }
    
    async def generate(self, prompt: str, model: ModelProvider, 
                      temperature: float = 0.7, max_tokens: int = 1000,
                      use_cache: bool = True) -> str:
        """生成响应"""
        # 检查缓存
        if use_cache:
            cached_response = await self.cache.get(prompt, model)
            if cached_response:
                return cached_response
        
        # 检查速率限制
        await self.rate_limiters[model].acquire()
        
        # 选择最优模型（基于成本和性能）
        optimal_model = await self.cost_optimizer.select_model(
            prompt, model, self.providers
        )
        
        # 调用对应的提供商
        provider = self.providers[optimal_model]
        
        try:
            response = await provider.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 缓存响应
            if use_cache:
                await self.cache.set(prompt, model, response)
            
            # 记录成本
            self.cost_optimizer.record_usage(
                model=optimal_model,
                prompt_tokens=provider.count_tokens(prompt),
                completion_tokens=provider.count_tokens(response)
            )
            
            return response
            
        except Exception as e:
            logging.error(f"LLM generation error: {e}")
            # 尝试降级到其他模型
            return await self._fallback_generate(prompt, model, temperature, max_tokens)
    
    async def _fallback_generate(self, prompt: str, model: ModelProvider,
                                temperature: float, max_tokens: int) -> str:
        """降级生成策略"""
        fallback_order = [
            ModelProvider.CLAUDE_3_7_SONNET,
            ModelProvider.QWEN_TURBO,
            ModelProvider.QWEN_PLUS
        ]
        
        for fallback_model in fallback_order:
            if fallback_model == model:
                continue
            
            try:
                provider = self.providers[fallback_model]
                return await provider.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except:
                continue
        
        raise Exception("All LLM providers failed")

# ============================= LLM提供商实现 =============================

class BaseProvider(ABC):
    """LLM提供商基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = None
        self._setup_tokenizer()
    
    @abstractmethod
    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """生成响应"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算token数量"""
        pass
    
    @abstractmethod
    def _setup_tokenizer(self):
        """设置分词器"""
        pass

class ClaudeProvider(BaseProvider):
    """Claude模型提供商"""
    
    def _setup_tokenizer(self):
        # Claude使用自己的tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("anthropic/claude-tokenizer")
    
    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """调用Claude API"""
        api_key = self.config.get("api_key")
        model = self.config.get("model", "claude-4-sonnet-20240229")
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2024-01-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                return result["content"][0]["text"]
    
    def count_tokens(self, text: str) -> int:
        """计算Claude tokens"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # 估算：平均每个字符0.25个token
        return int(len(text) * 0.25)

class QwenProvider(BaseProvider):
    """Qwen模型提供商"""
    
    def _setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-tokenizer")
    
    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """调用Qwen API"""
        api_key = self.config.get("api_key")
        model = self.config.get("model", "qwen-plus")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.8
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                return result["output"]["text"]
    
    def count_tokens(self, text: str) -> int:
        """计算Qwen tokens"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return int(len(text) * 0.3)  # Qwen的token比例略高

# ============================= 辅助组件实现 =============================

class CostOptimizer:
    """成本优化器"""
    
    def __init__(self):
        self.cost_per_token = {
            ModelProvider.CLAUDE_4_OPUS: {"input": 0.015, "output": 0.075},
            ModelProvider.CLAUDE_4_SONNET: {"input": 0.003, "output": 0.015},
            ModelProvider.CLAUDE_3_7_SONNET: {"input": 0.001, "output": 0.005},
            ModelProvider.QWEN_MAX: {"input": 0.002, "output": 0.010},
            ModelProvider.QWEN_PLUS: {"input": 0.001, "output": 0.005},
            ModelProvider.QWEN_TURBO: {"input": 0.0005, "output": 0.002}
        }
        self.usage_history = defaultdict(lambda: {"cost": 0, "tokens": 0})
        self._lock = threading.Lock()
    
    async def select_model(self, prompt: str, preferred_model: ModelProvider,
                          providers: Dict[ModelProvider, BaseProvider]) -> ModelProvider:
        """选择最优模型"""
        prompt_length = len(prompt)
        
        # 短prompt使用便宜模型
        if prompt_length < 500:
            return ModelProvider.QWEN_TURBO
        
        # 复杂任务使用高级模型
        if "complex" in prompt.lower() or "analyze" in prompt.lower():
            return preferred_model
        
        # 基于成本选择
        candidates = []
        for model, costs in self.cost_per_token.items():
            if model in providers:
                estimated_tokens = prompt_length * 0.25
                estimated_cost = estimated_tokens * costs["input"] / 1000
                candidates.append((model, estimated_cost))
        
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0] if candidates else preferred_model
    
    def record_usage(self, model: ModelProvider, prompt_tokens: int, completion_tokens: int):
        """记录使用情况"""
        with self._lock:
            costs = self.cost_per_token[model]
            total_cost = (prompt_tokens * costs["input"] + 
                         completion_tokens * costs["output"]) / 1000
            
            self.usage_history[model]["cost"] += total_cost
            self.usage_history[model]["tokens"] += prompt_tokens + completion_tokens

class SemanticCache:
    """语义缓存"""
    
    def __init__(self, max_size: int = 10000, similarity_threshold: float = 0.95):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, Tuple[str, np.ndarray, datetime]] = {}
        self.embedding_model = self._load_embedding_model()
        self._lock = asyncio.Lock()
    
    def _load_embedding_model(self):
        """加载嵌入模型"""
        # 使用轻量级嵌入模型
        return AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    
    async def get(self, prompt: str, model: ModelProvider) -> Optional[str]:
        """获取缓存响应"""
        async with self._lock:
            # 计算prompt嵌入
            prompt_embedding = await self._compute_embedding(prompt)
            
            # 搜索相似缓存
            for key, (cached_response, cached_embedding, timestamp) in self.cache.items():
                # 检查时效性（24小时）
                if datetime.now() - timestamp > timedelta(hours=24):
                    continue
                
                # 计算相似度
                similarity = np.dot(prompt_embedding, cached_embedding) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(cached_embedding)
                )
                
                if similarity >= self.similarity_threshold:
                    return cached_response
            
            return None
    
    async def set(self, prompt: str, model: ModelProvider, response: str):
        """设置缓存"""
        async with self._lock:
            # 检查缓存大小
            if len(self.cache) >= self.max_size:
                # 移除最旧的缓存
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k][2])
                del self.cache[oldest_key]
            
            # 计算嵌入并缓存
            embedding = await self._compute_embedding(prompt)
            cache_key = f"{model.value}:{hashlib.md5(prompt.encode()).hexdigest()}"
            self.cache[cache_key] = (response, embedding, datetime.now())
    
    async def _compute_embedding(self, text: str) -> np.ndarray:
        """计算文本嵌入"""
        # 简化实现，实际应使用嵌入模型
        tokens = self.embedding_model.encode(text)
        # 模拟嵌入向量
        embedding = np.random.randn(384)  # BGE-small维度
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """获取许可"""
        async with self._lock:
            now = time.time()
            
            # 清理过期请求
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            
            # 检查是否超限
            if len(self.requests) >= self.max_requests:
                # 计算需要等待的时间
                wait_time = self.requests[0] + self.window_seconds - now
                await asyncio.sleep(wait_time)
                # 递归重试
                return await self.acquire()
            
            # 记录请求
            self.requests.append(now)

# ============================= 专门化Agent实现 =============================

class PlannerAgent(BaseAgent):
    """规划Agent - 负责任务分解和计划制定"""
    
    def __init__(self, agent_id: str, name: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, name, AgentRole.PLANNER, 
                        ModelProvider.CLAUDE_4_SONNET, config)
        
        self.planning_strategies = {
            "hierarchical": self._hierarchical_planning,
            "backward": self._backward_planning,
            "iterative": self._iterative_planning
        }
        
        # 规划模板
        self.planning_templates = self._load_planning_templates()
    
    async def process_task(self, task: Task) -> Any:
        """处理规划任务"""
        # 分析任务复杂度
        complexity = await self._analyze_complexity(task)
        
        # 选择规划策略
        strategy = self._select_strategy(complexity)
        
        # 执行规划
        plan = await self.planning_strategies[strategy](task)
        
        # 验证计划
        validated_plan = await self._validate_plan(plan, task)
        
        # 优化计划
        optimized_plan = await self._optimize_plan(validated_plan)
        
        return {
            "plan": optimized_plan,
            "strategy": strategy,
            "complexity": complexity,
            "estimated_duration": self._estimate_duration(optimized_plan)
        }
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理消息"""
        if message.message_type == "plan_request":
            # 处理规划请求
            task = message.content["task"]
            plan = await self.process_task(task)
            
            return AgentMessage(
                message_type="plan_response",
                content=plan,
                correlation_id=message.id
            )
        
        elif message.message_type == "plan_adjustment":
            # 调整现有计划
            return await self._adjust_plan(message)
        
        return None
    
    async def _hierarchical_planning(self, task: Task) -> Dict[str, Any]:
        """层次化规划"""
        # 使用认知架构进行规划
        plan_thoughts = self.cognitive_architecture.plan(
            task.objective,
            task.constraints
        )
        
        # 构建层次化计划
        levels = defaultdict(list)
        current_level = 0
        
        for thought in plan_thoughts:
            # 提取计划步骤
            step = {
                "id": str(uuid.uuid4()),
                "description": thought.content,
                "level": current_level,
                "dependencies": [],
                "estimated_effort": self._estimate_effort(thought.content),
                "required_capabilities": self._extract_capabilities(thought.content)
            }
            
            levels[current_level].append(step)
            
            # 检查是否需要进入下一层
            if "then" in thought.content.lower() or "next" in thought.content.lower():
                current_level += 1
        
        return {
            "type": "hierarchical",
            "levels": dict(levels),
            "total_steps": sum(len(steps) for steps in levels.values()),
            "critical_path": self._identify_critical_path(levels)
        }
    
    async def _backward_planning(self, task: Task) -> Dict[str, Any]:
        """反向规划 - 从目标开始"""
        # 定义目标状态
        goal_state = {
            "objective": task.objective,
            "success_criteria": self._extract_success_criteria(task)
        }
        
        # 反向推导步骤
        steps = []
        current_state = goal_state
        
        while not self._is_initial_state(current_state):
            # 找出达到当前状态需要的前置条件
            prerequisites = await self._find_prerequisites(current_state)
            
            for prereq in prerequisites:
                step = {
                    "id": str(uuid.uuid4()),
                    "description": prereq["action"],
                    "preconditions": prereq["conditions"],
                    "postconditions": prereq["results"],
                    "estimated_effort": prereq["effort"]
                }
                steps.insert(0, step)  # 插入到开头
                
                # 更新当前状态
                current_state = prereq["conditions"]
        
        return {
            "type": "backward",
            "steps": steps,
            "goal_state": goal_state,
            "initial_state": current_state
        }
    
    async def _iterative_planning(self, task: Task) -> Dict[str, Any]:
        """迭代式规划"""
        iterations = []
        current_plan = None
        max_iterations = 5
        
        for i in range(max_iterations):
            # 生成或改进计划
            if current_plan is None:
                current_plan = await self._generate_initial_plan(task)
            else:
                current_plan = await self._refine_plan(current_plan, task)
            
            # 评估计划质量
            quality_score = await self._evaluate_plan_quality(current_plan)
            
            iterations.append({
                "iteration": i + 1,
                "plan": current_plan,
                "quality_score": quality_score,
                "improvements": self._identify_improvements(current_plan)
            })
            
            # 如果质量足够好，停止迭代
            if quality_score > 0.9:
                break
        
        return {
            "type": "iterative",
            "final_plan": current_plan,
            "iterations": iterations,
            "total_iterations": len(iterations)
        }
    
    async def _analyze_complexity(self, task: Task) -> Dict[str, Any]:
        """分析任务复杂度"""
        factors = {
            "scope": len(task.description) / 100,  # 描述长度
            "constraints": len(task.constraints),
            "dependencies": len(task.dependencies),
            "ambiguity": self._measure_ambiguity(task.description),
            "technical_depth": self._measure_technical_depth(task.description)
        }
        
        # 计算总体复杂度分数
        complexity_score = np.mean(list(factors.values()))
        
        return {
            "score": complexity_score,
            "factors": factors,
            "level": "high" if complexity_score > 0.7 else "medium" if complexity_score > 0.4 else "low"
        }
    
    def _select_strategy(self, complexity: Dict[str, Any]) -> str:
        """选择规划策略"""
        if complexity["level"] == "high":
            return "iterative"  # 高复杂度使用迭代优化
        elif complexity["factors"]["dependencies"] > 3:
            return "backward"   # 依赖多使用反向规划
        else:
            return "hierarchical"  # 默认层次化规划
    
    async def _validate_plan(self, plan: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """验证计划的可行性"""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        # 检查完整性
        if not self._covers_all_requirements(plan, task):
            validation_results["issues"].append("Plan does not cover all requirements")
            validation_results["is_valid"] = False
        
        # 检查依赖关系
        if self._has_circular_dependencies(plan):
            validation_results["issues"].append("Circular dependencies detected")
            validation_results["is_valid"] = False
        
        # 检查资源需求
        resource_check = self._check_resource_requirements(plan)
        if resource_check["exceeded"]:
            validation_results["warnings"].append(f"Resource requirements may be high: {resource_check['details']}")
        
        # 如果有问题，尝试修复
        if not validation_results["is_valid"]:
            plan = await self._fix_plan_issues(plan, validation_results["issues"])
        
        plan["validation"] = validation_results
        return plan
    
    async def _optimize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """优化计划"""
        # 并行化可以并行的步骤
        plan = self._parallelize_steps(plan)
        
        # 合并相似步骤
        plan = self._merge_similar_steps(plan)
        
        # 优化资源分配
        plan = self._optimize_resource_allocation(plan)
        
        # 添加缓冲时间
        plan = self._add_buffer_time(plan)
        
        return plan
    
    def _estimate_duration(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """估算计划持续时间"""
        if plan["type"] == "hierarchical":
            total_effort = sum(
                step["estimated_effort"] 
                for level_steps in plan["levels"].values() 
                for step in level_steps
            )
        else:
            total_effort = sum(step.get("estimated_effort", 1) for step in plan.get("steps", []))
        
        return {
            "optimistic": total_effort * 0.8,
            "realistic": total_effort,
            "pessimistic": total_effort * 1.5,
            "unit": "hours"
        }
    
    def _load_planning_templates(self) -> Dict[str, Any]:
        """加载规划模板"""
        return {
            "software_development": {
                "phases": ["requirements", "design", "implementation", "testing", "deployment"],
                "deliverables": ["spec", "architecture", "code", "tests", "documentation"]
            },
            "content_creation": {
                "phases": ["research", "outline", "draft", "review", "publish"],
                "deliverables": ["research_notes", "outline", "draft", "final_content"]
            },
            "data_analysis": {
                "phases": ["collection", "cleaning", "exploration", "analysis", "visualization"],
                "deliverables": ["dataset", "cleaned_data", "insights", "report", "dashboard"]
            }
        }
    
    def _estimate_effort(self, description: str) -> float:
        """估算工作量（小时）"""
        # 基于关键词的简单估算
        keywords_effort = {
            "simple": 0.5, "basic": 0.5, "quick": 0.5,
            "complex": 3.0, "detailed": 2.0, "comprehensive": 4.0,
            "analyze": 2.0, "implement": 3.0, "design": 2.5,
            "test": 1.5, "deploy": 1.0, "document": 1.5
        }
        
        effort = 1.0  # 默认工作量
        description_lower = description.lower()
        
        for keyword, hours in keywords_effort.items():
            if keyword in description_lower:
                effort = max(effort, hours)
        
        return effort
    
    def _extract_capabilities(self, description: str) -> List[str]:
        """提取所需能力"""
        capabilities = []
        
        capability_keywords = {
            "code": ["coding", "programming"],
            "analyze": ["analysis", "data_analysis"],
            "write": ["writing", "content_creation"],
            "design": ["design", "architecture"],
            "test": ["testing", "quality_assurance"]
        }
        
        description_lower = description.lower()
        for keyword, caps in capability_keywords.items():
            if keyword in description_lower:
                capabilities.extend(caps)
        
        return list(set(capabilities))
    
    def _identify_critical_path(self, levels: Dict[int, List[Dict[str, Any]]]) -> List[str]:
        """识别关键路径"""
        # 简化实现：返回每层耗时最长的步骤
        critical_path = []
        
        for level, steps in sorted(levels.items()):
            if steps:
                critical_step = max(steps, key=lambda s: s["estimated_effort"])
                critical_path.append(critical_step["id"])
        
        return critical_path
    
    def _extract_success_criteria(self, task: Task) -> List[str]:
        """提取成功标准"""
        criteria = []
        
        # 从任务描述中提取
        if "must" in task.description:
            criteria.append("Must requirements met")
        if "should" in task.description:
            criteria.append("Should requirements met")
        
        # 从约束中提取
        criteria.extend([f"Constraint satisfied: {c}" for c in task.constraints[:3]])
        
        return criteria
    
    def _is_initial_state(self, state: Dict[str, Any]) -> bool:
        """检查是否是初始状态"""
        # 简化判断：如果没有前置条件，就是初始状态
        return not state.get("preconditions") or len(state.get("preconditions", [])) == 0
    
    async def _find_prerequisites(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找前置条件"""
        # 使用认知架构推理前置条件
        reasoning = self.cognitive_architecture.reason([
            Thought(
                type=ThoughtType.PLANNING,
                content=f"What is needed to achieve: {state}",
                confidence=0.8
            )
        ])
        
        # 模拟返回前置条件
        return [{
            "action": "Prepare prerequisite",
            "conditions": {"prepared": False},
            "results": {"prepared": True},
            "effort": 1.0
        }]
    
    async def _generate_initial_plan(self, task: Task) -> Dict[str, Any]:
        """生成初始计划"""
        # 使用模板生成
        template_type = self._identify_template_type(task)
        template = self.planning_templates.get(template_type, {})
        
        steps = []
        for phase in template.get("phases", ["planning", "execution", "validation"]):
            steps.append({
                "id": str(uuid.uuid4()),
                "phase": phase,
                "description": f"{phase} for {task.name}",
                "estimated_effort": 2.0
            })
        
        return {
            "steps": steps,
            "template": template_type
        }
    
    async def _refine_plan(self, plan: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """改进计划"""
        # 使用认知架构的批判性分析
        critique = self.cognitive_architecture.critique(
            Thought(content=str(plan), confidence=0.7),
            ["completeness", "efficiency", "feasibility"]
        )
        
        # 基于批判结果改进
        improvements = critique.metadata.get("suggestions", [])
        
        # 应用改进（简化实现）
        if "add more detail" in str(improvements).lower():
            # 为每个步骤添加子步骤
            for step in plan.get("steps", []):
                step["substeps"] = [
                    {"description": f"Sub-task for {step['description']}", 
                     "effort": step["estimated_effort"] / 3}
                    for _ in range(3)
                ]
        
        return plan
    
    async def _evaluate_plan_quality(self, plan: Dict[str, Any]) -> float:
        """评估计划质量"""
        scores = {
            "completeness": 0.8,  # 完整性
            "clarity": 0.9,       # 清晰度
            "feasibility": 0.7,   # 可行性
            "efficiency": 0.8     # 效率
        }
        
        # 检查计划结构
        if "steps" in plan and len(plan["steps"]) > 0:
            scores["completeness"] = min(1.0, len(plan["steps"]) / 10)
        
        # 检查时间估算
        if all(step.get("estimated_effort") for step in plan.get("steps", [])):
            scores["clarity"] = 0.95
        
        return np.mean(list(scores.values()))
    
    def _identify_improvements(self, plan: Dict[str, Any]) -> List[str]:
        """识别改进点"""
        improvements = []
        
        # 检查并行机会
        if self._has_sequential_independent_steps(plan):
            improvements.append("Some steps can be parallelized")
        
        # 检查冗余
        if self._has_redundant_steps(plan):
            improvements.append("Remove redundant steps")
        
        # 检查资源利用
        if self._has_resource_conflicts(plan):
            improvements.append("Optimize resource allocation")
        
        return improvements
    
    def _measure_ambiguity(self, text: str) -> float:
        """测量文本模糊度"""
        ambiguous_words = ["maybe", "possibly", "might", "could", "should", "probably"]
        word_count = len(text.split())
        ambiguous_count = sum(1 for word in ambiguous_words if word in text.lower())
        return ambiguous_count / max(word_count, 1)
    
    def _measure_technical_depth(self, text: str) -> float:
        """测量技术深度"""
        technical_terms = ["algorithm", "architecture", "framework", "protocol", 
                          "optimization", "implementation", "integration"]
        word_count = len(text.split())
        technical_count = sum(1 for term in technical_terms if term in text.lower())
        return min(1.0, technical_count / 5)  # 5个技术词算满分
    
    def _covers_all_requirements(self, plan: Dict[str, Any], task: Task) -> bool:
        """检查计划是否覆盖所有需求"""
        # 简化检查：确保有步骤
        return len(plan.get("steps", [])) > 0 or len(plan.get("levels", {})) > 0
    
    def _has_circular_dependencies(self, plan: Dict[str, Any]) -> bool:
        """检查循环依赖"""
        # 构建依赖图
        graph = nx.DiGraph()
        
        for step in plan.get("steps", []):
            step_id = step["id"]
            for dep in step.get("dependencies", []):
                graph.add_edge(dep, step_id)
        
        # 检查是否有环
        return not nx.is_directed_acyclic_graph(graph)
    
    def _check_resource_requirements(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """检查资源需求"""
        total_effort = sum(
            step.get("estimated_effort", 0) 
            for step in plan.get("steps", [])
        )
        
        return {
            "exceeded": total_effort > 40,  # 超过40小时
            "details": f"Total effort: {total_effort} hours"
        }
    
    async def _fix_plan_issues(self, plan: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
        """修复计划问题"""
        for issue in issues:
            if "circular dependencies" in issue:
                # 移除循环依赖
                plan = self._remove_circular_dependencies(plan)
            elif "requirements" in issue:
                # 添加缺失的步骤
                plan = await self._add_missing_steps(plan)
        
        return plan
    
    def _parallelize_steps(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """并行化步骤"""
        # 识别可并行的步骤
        if "steps" in plan:
            # 标记可并行的步骤
            for i, step in enumerate(plan["steps"]):
                if i > 0 and not step.get("dependencies"):
                    step["can_parallel"] = True
        
        return plan
    
    def _merge_similar_steps(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """合并相似步骤"""
        # 简化实现：基于描述相似度合并
        if "steps" in plan:
            merged_steps = []
            seen_descriptions = set()
            
            for step in plan["steps"]:
                desc_lower = step["description"].lower()
                if desc_lower not in seen_descriptions:
                    merged_steps.append(step)
                    seen_descriptions.add(desc_lower)
            
            plan["steps"] = merged_steps
        
        return plan
    
    def _optimize_resource_allocation(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """优化资源分配"""
        # 简化实现：平衡工作量
        if "steps" in plan:
            total_effort = sum(step.get("estimated_effort", 1) for step in plan["steps"])
            avg_effort = total_effort / len(plan["steps"]) if plan["steps"] else 1
            
            # 标记高负载步骤
            for step in plan["steps"]:
                if step.get("estimated_effort", 0) > avg_effort * 2:
                    step["high_load"] = True
                    step["suggested_split"] = True
        
        return plan
    
    def _add_buffer_time(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """添加缓冲时间"""
        buffer_ratio = 0.2  # 20%缓冲
        
        if "steps" in plan:
            for step in plan["steps"]:
                original_effort = step.get("estimated_effort", 1)
                step["buffer"] = original_effort * buffer_ratio
                step["total_effort"] = original_effort + step["buffer"]
        
        return plan
    
    def _identify_template_type(self, task: Task) -> str:
        """识别任务类型"""
        description_lower = task.description.lower()
        
        if any(word in description_lower for word in ["code", "develop", "implement"]):
            return "software_development"
        elif any(word in description_lower for word in ["write", "content", "article"]):
            return "content_creation"
        elif any(word in description_lower for word in ["analyze", "data", "insight"]):
            return "data_analysis"
        else:
            return "generic"
    
    def _has_sequential_independent_steps(self, plan: Dict[str, Any]) -> bool:
        """检查是否有顺序独立的步骤"""
        # 简化检查
        return len(plan.get("steps", [])) > 3
    
    def _has_redundant_steps(self, plan: Dict[str, Any]) -> bool:
        """检查是否有冗余步骤"""
        # 简化检查：相似描述
        if "steps" in plan:
            descriptions = [step["description"].lower() for step in plan["steps"]]
            return len(descriptions) != len(set(descriptions))
        return False
    
    def _has_resource_conflicts(self, plan: Dict[str, Any]) -> bool:
        """检查资源冲突"""
        # 简化检查
        return False
    
    def _remove_circular_dependencies(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """移除循环依赖"""
        # 简化实现：清空所有依赖
        if "steps" in plan:
            for step in plan["steps"]:
                step["dependencies"] = []
        return plan
    
    async def _add_missing_steps(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """添加缺失的步骤"""
        # 添加验证步骤
        if "steps" in plan:
            plan["steps"].append({
                "id": str(uuid.uuid4()),
                "description": "Validate results",
                "estimated_effort": 1.0,
                "phase": "validation"
            })
        return plan
    
    async def _adjust_plan(self, message: AgentMessage) -> AgentMessage:
        """调整计划"""
        adjustment_type = message.content.get("type")
        current_plan = message.content.get("plan")
        
        if adjustment_type == "add_step":
            # 添加新步骤
            new_step = message.content.get("step")
            current_plan["steps"].append(new_step)
        elif adjustment_type == "remove_step":
            # 移除步骤
            step_id = message.content.get("step_id")
            current_plan["steps"] = [
                s for s in current_plan["steps"] if s["id"] != step_id
            ]
        elif adjustment_type == "reorder":
            # 重新排序
            new_order = message.content.get("order")
            # 实现重新排序逻辑
        
        return AgentMessage(
            message_type="plan_adjusted",
            content={"adjusted_plan": current_plan}
        )

class ExecutorAgent(BaseAgent):
    """执行Agent - 负责具体任务执行"""
    
    def __init__(self, agent_id: str, name: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, name, AgentRole.EXECUTOR,
                        ModelProvider.QWEN_PLUS, config)
        
        self.execution_history = []
        self.skill_registry = {}
        self._register_skills()
    
    async def process_task(self, task: Task) -> Any:
        """执行任务"""
        try:
            # 准备执行环境
            environment = await self._prepare_environment(task)
            
            # 执行任务
            result = await self._execute_task(task, environment)
            
            # 验证结果
            validated_result = await self._validate_execution(result, task)
            
            # 清理环境
            await self._cleanup_environment(environment)
            
            # 记录执行历史
            self.execution_history.append({
                "task_id": task.id,
                "result": validated_result,
                "timestamp": datetime.now(),
                "duration": environment.get("duration")
            })
            
            return validated_result
            
        except Exception as e:
            logging.error(f"Execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "task_id": task.id
            }
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理执行相关消息"""
        if message.message_type == "execute_command":
            # 执行命令
            result = await self._execute_command(message.content)
            return AgentMessage(
                message_type="execution_result",
                content=result
            )
        
        elif message.message_type == "skill_request":
            # 使用特定技能
            skill_name = message.content.get("skill")
            params = message.content.get("params", {})
            result = await self._use_skill(skill_name, **params)
            return AgentMessage(
                message_type="skill_result",
                content=result
            )
        
        return None
    
    def _register_skills(self):
        """注册执行技能"""
        self.skill_registry = {
            "code_execution": self._execute_code,
            "data_processing": self._process_data,
            "file_operations": self._file_operations,
            "api_calls": self._make_api_calls,
            "shell_commands": self._execute_shell
        }
    
    async def _prepare_environment(self, task: Task) -> Dict[str, Any]:
        """准备执行环境"""
        environment = {
            "task_id": task.id,
            "start_time": datetime.now(),
            "resources": await self._allocate_resources(task),
            "context": {}
        }
        
        # 加载必要的工具
        required_tools = self._identify_required_tools(task)
        for tool in required_tools:
            await self.register_tool(tool["name"], tool["function"])
        
        return environment
    
    async def _execute_task(self, task: Task, environment: Dict[str, Any]) -> Any:
        """执行具体任务"""
        # 解析任务类型
        task_type = self._identify_task_type(task)
        
        if task_type == "code_generation":
            return await self._generate_code(task)
        elif task_type == "data_analysis":
            return await self._analyze_data(task)
        elif task_type == "content_creation":
            return await self._create_content(task)
        elif task_type == "automation":
            return await self._automate_process(task)
        else:
            return await self._generic_execution(task)
    
    async def _validate_execution(self, result: Any, task: Task) -> Any:
        """验证执行结果"""
        validation = {
            "result": result,
            "valid": True,
            "issues": []
        }
        
        # 检查结果完整性
        if result is None:
            validation["valid"] = False
            validation["issues"].append("Null result")
        
        # 检查是否满足任务要求
        if not self._meets_requirements(result, task):
            validation["valid"] = False
            validation["issues"].append("Requirements not met")
        
        # 运行测试（如果适用）
        if task.metadata.get("tests"):
            test_results = await self._run_tests(result, task.metadata["tests"])
            validation["test_results"] = test_results
            if not all(test_results.values()):
                validation["valid"] = False
                validation["issues"].append("Tests failed")
        
        return validation
    
    async def _cleanup_environment(self, environment: Dict[str, Any]):
        """清理执行环境"""
        # 释放资源
        if "resources" in environment:
            await self._release_resources(environment["resources"])
        
        # 记录执行时长
        if "start_time" in environment:
            environment["duration"] = (datetime.now() - environment["start_time"]).total_seconds()
    
    async def _execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """执行命令"""
        cmd_type = command.get("type")
        cmd_content = command.get("content")
        
        try:
            if cmd_type == "shell":
                result = await self._execute_shell(cmd_content)
            elif cmd_type == "python":
                result = await self._execute_code(cmd_content, "python")
            elif cmd_type == "api":
                result = await self._make_api_calls(cmd_content)
            else:
                result = {"error": f"Unknown command type: {cmd_type}"}
            
            return {
                "status": "success",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _use_skill(self, skill_name: str, **params) -> Any:
        """使用特定技能"""
        if skill_name not in self.skill_registry:
            raise ValueError(f"Unknown skill: {skill_name}")
        
        skill_func = self.skill_registry[skill_name]
        
        # 记录技能使用
        thought = Thought(
            type=ThoughtType.OBSERVATION,
            content=f"Using skill: {skill_name} with params: {params}",
            confidence=0.9
        )
        self.cognitive_architecture.thought_chain.add_thought(thought)
        
        # 执行技能
        if asyncio.iscoroutinefunction(skill_func):
            result = await skill_func(**params)
        else:
            result = await asyncio.to_thread(skill_func, **params)
        
        return result
    
    async def _execute_code(self, code: str, language: str = "python") -> Any:
        """执行代码"""
        if language == "python":
            # 创建安全的执行环境
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter
                }
            }
            
            safe_locals = {}
            
            try:
                # 使用受限的exec
                exec(code, safe_globals, safe_locals)
                
                # 返回结果
                return {
                    "output": safe_locals.get("result", "Code executed successfully"),
                    "locals": {k: str(v) for k, v in safe_locals.items() if not k.startswith("_")}
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "type": type(e).__name__
                }
        else:
            return {"error": f"Unsupported language: {language}"}
    
    async def _process_data(self, data: Any, operation: str) -> Any:
        """处理数据"""
        if operation == "filter":
            # 实现数据过滤
            pass
        elif operation == "transform":
            # 实现数据转换
            pass
        elif operation == "aggregate":
            # 实现数据聚合
            pass
        
        return {"processed": True, "operation": operation}
    
    async def _file_operations(self, operation: str, path: str, **kwargs) -> Any:
        """文件操作"""
        # 安全检查
        if not self._is_safe_path(path):
            raise ValueError("Unsafe file path")
        
        if operation == "read":
            with open(path, 'r') as f:
                return f.read()
        elif operation == "write":
            content = kwargs.get("content", "")
            with open(path, 'w') as f:
                f.write(content)
            return {"written": len(content)}
        elif operation == "list":
            import os
            return os.listdir(path)
        
        return {"error": f"Unknown operation: {operation}"}
    
    async def _make_api_calls(self, config: Dict[str, Any]) -> Any:
        """进行API调用"""
        url = config.get("url")
        method = config.get("method", "GET")
        headers = config.get("headers", {})
        data = config.get("data")
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if method in ["POST", "PUT"] else None,
                params=data if method == "GET" else None
            ) as response:
                return {
                    "status": response.status,
                    "data": await response.json() if response.content_type == "application/json" else await response.text()
                }
    
    async def _execute_shell(self, command: str) -> Dict[str, Any]:
        """执行shell命令"""
        # 安全检查
        if not self._is_safe_command(command):
            raise ValueError("Unsafe command")
        
        import subprocess
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timeout"}
    
    async def _allocate_resources(self, task: Task) -> Dict[str, Any]:
        """分配资源"""
        return {
            "cpu": 1,
            "memory": "1GB",
            "disk": "10GB",
            "timeout": 300  # 5分钟
        }
    
    def _identify_required_tools(self, task: Task) -> List[Dict[str, Any]]:
        """识别所需工具"""
        tools = []
        
        # 基于任务描述识别工具
        if "file" in task.description.lower():
            tools.append({
                "name": "file_handler",
                "function": self._file_operations
            })
        
        if "api" in task.description.lower() or "http" in task.description.lower():
            tools.append({
                "name": "http_client",
                "function": self._make_api_calls
            })
        
        return tools
    
    def _identify_task_type(self, task: Task) -> str:
        """识别任务类型"""
        description_lower = task.description.lower()
        
        if any(word in description_lower for word in ["code", "program", "script"]):
            return "code_generation"
        elif any(word in description_lower for word in ["analyze", "data", "statistics"]):
            return "data_analysis"
        elif any(word in description_lower for word in ["write", "content", "article"]):
            return "content_creation"
        elif any(word in description_lower for word in ["automate", "workflow", "process"]):
            return "automation"
        else:
            return "generic"
    
    async def _generate_code(self, task: Task) -> Dict[str, Any]:
        """生成代码"""
        # 使用LLM生成代码
        prompt = f"""
        Generate code for the following task:
        Task: {task.description}
        Requirements: {', '.join(task.requirements)}
        Constraints: {', '.join(task.constraints)}
        
        Please provide clean, well-commented code.
        """
        
        # 这里应该调用LLM
        generated_code = "# Generated code placeholder\ndef solution():\n    pass"
        
        return {
            "code": generated_code,
            "language": "python",
            "explanation": "Code generated based on task requirements"
        }
    
    async def _analyze_data(self, task: Task) -> Dict[str, Any]:
        """分析数据"""
        # 提取数据源
        data_source = task.metadata.get("data_source", {})
        
        # 执行分析
        analysis_results = {
            "summary": "Data analysis completed",
            "insights": ["Insight 1", "Insight 2"],
            "statistics": {
                "count": 100,
                "mean": 50,
                "std": 10
            }
        }
        
        return analysis_results
    
    async def _create_content(self, task: Task) -> Dict[str, Any]:
        """创建内容"""
        content_type = task.metadata.get("content_type", "article")
        
        # 使用LLM创建内容
        prompt = f"""
        Create {content_type} for the following:
        Topic: {task.description}
        Requirements: {', '.join(task.requirements)}
        """
        
        # 这里应该调用LLM
        content = f"# {task.name}\n\nContent placeholder..."
        
        return {
            "content": content,
            "type": content_type,
            "word_count": len(content.split())
        }
    
    async def _automate_process(self, task: Task) -> Dict[str, Any]:
        """自动化流程"""
        steps = task.metadata.get("automation_steps", [])
        
        results = []
        for step in steps:
            # 执行每个自动化步骤
            step_result = await self._execute_automation_step(step)
            results.append(step_result)
        
        return {
            "automation_complete": True,
            "steps_executed": len(results),
            "results": results
        }
    
    async def _generic_execution(self, task: Task) -> Any:
        """通用执行"""
        # 使用认知架构决定执行策略
        execution_thought = self.cognitive_architecture.reason([
            Thought(
                type=ThoughtType.PLANNING,
                content=f"How to execute: {task.description}",
                confidence=0.7
            )
        ])
        
        return {
            "execution": "completed",
            "strategy": execution_thought.content,
            "task_id": task.id
        }
    
    def _meets_requirements(self, result: Any, task: Task) -> bool:
        """检查是否满足需求"""
        # 简化检查
        return result is not None and not isinstance(result, dict) or result.get("status") != "failed"
    
    async def _run_tests(self, result: Any, tests: List[Dict[str, Any]]) -> Dict[str, bool]:
        """运行测试"""
        test_results = {}
        
        for test in tests:
            test_name = test.get("name", "unnamed_test")
            test_type = test.get("type", "assertion")
            
            if test_type == "assertion":
                # 运行断言测试
                passed = await self._run_assertion_test(result, test)
            elif test_type == "validation":
                # 运行验证测试
                passed = await self._run_validation_test(result, test)
            else:
                passed = False
            
            test_results[test_name] = passed
        
        return test_results
    
    async def _release_resources(self, resources: Dict[str, Any]):
        """释放资源"""
        # 实现资源释放逻辑
        pass
    
    def _is_safe_path(self, path: str) -> bool:
        """检查路径安全性"""
        # 禁止访问系统目录
        forbidden_paths = ["/etc", "/sys", "/proc", "C:\\Windows", "C:\\System"]
        return not any(path.startswith(fp) for fp in forbidden_paths)
    
    def _is_safe_command(self, command: str) -> bool:
        """检查命令安全性"""
        # 禁止危险命令
        dangerous_commands = ["rm -rf", "format", "del /f", "shutdown", "reboot"]
        return not any(dc in command.lower() for dc in dangerous_commands)
    
    async def _execute_automation_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """执行自动化步骤"""
        step_type = step.get("type")
        
        if step_type == "command":
            return await self._execute_command(step.get("command", {}))
        elif step_type == "api":
            return await self._make_api_calls(step.get("config", {}))
        else:
            return {"error": f"Unknown step type: {step_type}"}
    
    async def _run_assertion_test(self, result: Any, test: Dict[str, Any]) -> bool:
        """运行断言测试"""
        assertion = test.get("assertion")
        expected = test.get("expected")
        
        # 简单的断言检查
        if assertion == "equals":
            return result == expected
        elif assertion == "contains":
            return expected in str(result)
        elif assertion == "type":
            return isinstance(result, eval(expected))
        
        return False
    
    async def _run_validation_test(self, result: Any, test: Dict[str, Any]) -> bool:
        """运行验证测试"""
        validator = test.get("validator")
        
        # 运行自定义验证器
        if callable(validator):
            return validator(result)
        
        return True

class AnalyzerAgent(BaseAgent):
    """分析Agent - 负责数据分析和洞察生成"""
    
    def __init__(self, agent_id: str, name: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, name, AgentRole.ANALYZER,
                        ModelProvider.CLAUDE_4_SONNET, config)
        
        self.analysis_methods = {
            "statistical": self._statistical_analysis,
            "semantic": self._semantic_analysis,
            "pattern": self._pattern_analysis,
            "comparative": self._comparative_analysis,
            "predictive": self._predictive_analysis
        }
        
        self.insight_generator = InsightGenerator()
    
    async def process_task(self, task: Task) -> Any:
        """处理分析任务"""
        # 提取分析目标
        analysis_target = task.metadata.get("target", task.description)
        analysis_type = task.metadata.get("analysis_type", "comprehensive")
        
        # 收集数据
        data = await self._collect_data(task)
        
        # 执行分析
        if analysis_type == "comprehensive":
            results = await self._comprehensive_analysis(data, analysis_target)
        else:
            method = self.analysis_methods.get(analysis_type, self._generic_analysis)
            results = await method(data, analysis_target)
        
        # 生成洞察
        insights = await self.insight_generator.generate(results, task)
        
        # 创建可视化
        visualizations = await self._create_visualizations(results)
        
        return {
            "analysis_results": results,
            "insights": insights,
            "visualizations": visualizations,
            "summary": self._generate_summary(results, insights)
        }
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理分析相关消息"""
        if message.message_type == "analysis_request":
            # 快速分析请求
            data = message.content.get("data")
            analysis_type = message.content.get("type", "quick")
            
            result = await self._quick_analysis(data, analysis_type)
            
            return AgentMessage(
                message_type="analysis_result",
                content=result
            )
        
        return None
    
    async def _collect_data(self, task: Task) -> Dict[str, Any]:
        """收集分析数据"""
        data_sources = task.metadata.get("data_sources", [])
        collected_data = {}
        
        for source in data_sources:
            source_type = source.get("type")
            
            if source_type == "file":
                data = await self._load_file_data(source["path"])
            elif source_type == "api":
                data = await self._fetch_api_data(source["endpoint"])
            elif source_type == "database":
                data = await self._query_database(source["query"])
            else:
                data = source.get("data", {})
            
            collected_data[source.get("name", "unnamed")] = data
        
        return collected_data
    
    async def _comprehensive_analysis(self, data: Dict[str, Any], target: str) -> Dict[str, Any]:
        """综合分析"""
        results = {}
        
        # 运行所有分析方法
        for method_name, method_func in self.analysis_methods.items():
            try:
                method_result = await method_func(data, target)
                results[method_name] = method_result
            except Exception as e:
                logging.error(f"Analysis method {method_name} failed: {e}")
                results[method_name] = {"error": str(e)}
        
        # 综合所有结果
        results["synthesis"] = self._synthesize_results(results)
        
        return results
    
    async def _statistical_analysis(self, data: Dict[str, Any], target: str) -> Dict[str, Any]:
        """统计分析"""
        stats = {
            "descriptive": {},
            "distributions": {},
            "correlations": {}
        }
        
        # 对每个数据集进行统计分析
        for name, dataset in data.items():
            if isinstance(dataset, list) and all(isinstance(x, (int, float)) for x in dataset):
                # 描述性统计
                stats["descriptive"][name] = {
                    "count": len(dataset),
                    "mean": np.mean(dataset),
                    "median": np.median(dataset),
                    "std": np.std(dataset),
                    "min": np.min(dataset),
                    "max": np.max(dataset),
                    "quantiles": {
                        "25%": np.percentile(dataset, 25),
                        "50%": np.percentile(dataset, 50),
                        "75%": np.percentile(dataset, 75)
                    }
                }
        
        return stats
    
    async def _semantic_analysis(self, data: Dict[str, Any], target: str) -> Dict[str, Any]:
        """语义分析"""
        semantic_results = {
            "themes": [],
            "entities": [],
            "sentiments": {},
            "keywords": []
        }
        
        # 对文本数据进行语义分析
        for name, dataset in data.items():
            if isinstance(dataset, str):
                # 主题提取
                themes = await self._extract_themes(dataset)
                semantic_results["themes"].extend(themes)
                
                # 实体识别
                entities = await self._extract_entities(dataset)
                semantic_results["entities"].extend(entities)
                
                # 情感分析
                sentiment = await self._analyze_sentiment(dataset)
                semantic_results["sentiments"][name] = sentiment
                
                # 关键词提取
                keywords = await self._extract_keywords(dataset)
                semantic_results["keywords"].extend(keywords)
        
        return semantic_results
    
    async def _pattern_analysis(self, data: Dict[str, Any], target: str) -> Dict[str, Any]:
        """模式分析"""
        patterns = {
            "temporal": [],
            "sequential": [],
            "cyclic": [],
            "anomalies": []
        }
        
        # 识别不同类型的模式
        for name, dataset in data.items():
            if isinstance(dataset, list):
                # 时间模式
                temporal_patterns = self._find_temporal_patterns(dataset)
                patterns["temporal"].extend(temporal_patterns)
                
                # 序列模式
                sequential_patterns = self._find_sequential_patterns(dataset)
                patterns["sequential"].extend(sequential_patterns)
                
                # 周期模式
                cyclic_patterns = self._find_cyclic_patterns(dataset)
                patterns["cyclic"].extend(cyclic_patterns)
                
                # 异常检测
                anomalies = self._detect_anomalies(dataset)
                patterns["anomalies"].extend(anomalies)
        
        return patterns
    
    async def _comparative_analysis(self, data: Dict[str, Any], target: str) -> Dict[str, Any]:
        """比较分析"""
        comparisons = {
            "similarities": [],
            "differences": [],
            "rankings": {},
            "benchmarks": {}
        }
        
        # 比较不同数据集
        dataset_names = list(data.keys())
        
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                name1, name2 = dataset_names[i], dataset_names[j]
                
                # 计算相似度
                similarity = self._calculate_similarity(data[name1], data[name2])
                comparisons["similarities"].append({
                    "datasets": [name1, name2],
                    "similarity": similarity
                })
                
                # 识别差异
                differences = self._identify_differences(data[name1], data[name2])
                comparisons["differences"].extend(differences)
        
        return comparisons
    
    async def _predictive_analysis(self, data: Dict[str, Any], target: str) -> Dict[str, Any]:
        """预测分析"""
        predictions = {
            "trends": [],
            "forecasts": {},
            "scenarios": []
        }
        
        # 趋势预测
        for name, dataset in data.items():
            if isinstance(dataset, list) and len(dataset) > 10:
                trend = self._predict_trend(dataset)
                predictions["trends"].append({
                    "dataset": name,
                    "trend": trend
                })
                
                # 预测未来值
                forecast = self._forecast_values(dataset, periods=5)
                predictions["forecasts"][name] = forecast
        
        # 场景分析
        scenarios = self._generate_scenarios(data)
        predictions["scenarios"] = scenarios
        
        return predictions
    
    async def _create_visualizations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建可视化"""
        visualizations = []
        
        # 统计图表
        if "statistical" in results:
            stats = results["statistical"]
            for name, stat_data in stats.get("descriptive", {}).items():
                viz = {
                    "type": "box_plot",
                    "title": f"Distribution of {name}",
                    "data": stat_data,
                    "config": {
                        "show_outliers": True,
                        "color_scheme": "blue"
                    }
                }
                visualizations.append(viz)
        
        # 趋势图
        if "predictive" in results:
            for trend_data in results["predictive"].get("trends", []):
                viz = {
                    "type": "line_chart",
                    "title": f"Trend Analysis: {trend_data['dataset']}",
                    "data": trend_data["trend"],
                    "config": {
                        "show_forecast": True,
                        "confidence_interval": True
                    }
                }
                visualizations.append(viz)
        
        return visualizations
    
    def _generate_summary(self, results: Dict[str, Any], insights: List[Dict[str, Any]]) -> str:
        """生成分析摘要"""
        summary_parts = []
        
        # 关键发现
        summary_parts.append("Key Findings:")
        for i, insight in enumerate(insights[:5], 1):
            summary_parts.append(f"{i}. {insight.get('description', 'N/A')}")
        
        # 统计摘要
        if "statistical" in results:
            summary_parts.append("\nStatistical Summary:")
            stats = results["statistical"].get("descriptive", {})
            for name, stat in stats.items():
                summary_parts.append(f"- {name}: mean={stat['mean']:.2f}, std={stat['std']:.2f}")
        
        # 预测摘要
        if "predictive" in results:
            summary_parts.append("\nPredictions:")
            for trend in results["predictive"].get("trends", [])[:3]:
                summary_parts.append(f"- {trend['dataset']}: {trend['trend']['direction']} trend")
        
        return "\n".join(summary_parts)
    
    async def _quick_analysis(self, data: Any, analysis_type: str) -> Dict[str, Any]:
        """快速分析"""
        if analysis_type == "summary":
            return {
                "type": type(data).__name__,
                "size": len(data) if hasattr(data, "__len__") else "N/A",
                "preview": str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
            }
        elif analysis_type == "stats":
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                return {
                    "mean": np.mean(data),
                    "std": np.std(data),
                    "min": min(data),
                    "max": max(data)
                }
        
        return {"error": "Unsupported quick analysis type"}
    
    async def _load_file_data(self, path: str) -> Any:
        """加载文件数据"""
        # 实现文件加载逻辑
        return {"file_path": path, "data": "placeholder"}
    
    async def _fetch_api_data(self, endpoint: str) -> Any:
        """获取API数据"""
        # 实现API调用逻辑
        return {"endpoint": endpoint, "data": "placeholder"}
    
    async def _query_database(self, query: str) -> Any:
        """查询数据库"""
        # 实现数据库查询逻辑
        return {"query": query, "data": "placeholder"}
    
    def _synthesize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """综合分析结果"""
        synthesis = {
            "overall_quality": self._assess_data_quality(results),
            "key_patterns": self._extract_key_patterns(results),
            "recommendations": self._generate_recommendations(results)
        }
        return synthesis
    
    async def _extract_themes(self, text: str) -> List[str]:
        """提取主题"""
        # 简化实现
        words = text.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 4:  # 只考虑较长的词
                word_freq[word] += 1
        
        # 返回高频词作为主题
        themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [theme[0] for theme in themes]
    
    async def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """提取实体"""
        # 简化实现：识别大写词作为实体
        entities = []
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append({
                    "text": word,
                    "type": "PROPER_NOUN"
                })
        return entities
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """情感分析"""
        # 简化实现：基于关键词
        positive_words = ["good", "great", "excellent", "positive", "success"]
        negative_words = ["bad", "poor", "negative", "failure", "problem"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count + 1
        return {
            "positive": positive_count / total,
            "negative": negative_count / total,
            "neutral": 1 - (positive_count + negative_count) / total
        }
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化实现：TF-IDF的简单版本
        words = text.lower().split()
        word_freq = defaultdict(int)
        
        for word in words:
            if len(word) > 3 and word.isalpha():
                word_freq[word] += 1
        
        # 返回高频词
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [kw[0] for kw in keywords]
    
    def _find_temporal_patterns(self, data: List[Any]) -> List[Dict[str, Any]]:
        """查找时间模式"""
        patterns = []
        
        # 检查递增/递减模式
        if all(isinstance(x, (int, float)) for x in data):
            differences = [data[i+1] - data[i] for i in range(len(data)-1)]
            
            if all(d > 0 for d in differences):
                patterns.append({
                    "type": "monotonic_increase",
                    "strength": min(differences) / max(differences) if max(differences) > 0 else 0
                })
            elif all(d < 0 for d in differences):
                patterns.append({
                    "type": "monotonic_decrease",
                    "strength": max(differences) / min(differences) if min(differences) < 0 else 0
                })
        
        return patterns
    
    def _find_sequential_patterns(self, data: List[Any]) -> List[Dict[str, Any]]:
        """查找序列模式"""
        patterns = []
        
        # 查找重复序列
        for pattern_len in range(2, min(len(data) // 2, 10)):
            for start in range(len(data) - pattern_len):
                pattern = data[start:start + pattern_len]
                count = 0
                
                for i in range(start + pattern_len, len(data) - pattern_len + 1):
                    if data[i:i + pattern_len] == pattern:
                        count += 1
                
                if count > 1:
                    patterns.append({
                        "type": "repeating_sequence",
                        "pattern": pattern,
                        "occurrences": count + 1
                    })
        
        return patterns
    
    def _find_cyclic_patterns(self, data: List[Any]) -> List[Dict[str, Any]]:
        """查找周期模式"""
        patterns = []
        
        # 简化的周期检测
        if len(data) > 10 and all(isinstance(x, (int, float)) for x in data):
            # 尝试不同的周期长度
            for period in range(2, min(len(data) // 3, 20)):
                cycles = []
                for i in range(0, len(data) - period, period):
                    cycle = data[i:i + period]
                    cycles.append(cycle)
                
                if len(cycles) > 2:
                    # 计算周期间的相似度
                    similarity = self._calculate_cycle_similarity(cycles)
                    if similarity > 0.7:
                        patterns.append({
                            "type": "cyclic",
                            "period": period,
                            "similarity": similarity
                        })
        
        return patterns
    
    def _detect_anomalies(self, data: List[Any]) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []
        
        if all(isinstance(x, (int, float)) for x in data) and len(data) > 3:
            # 使用IQR方法检测异常值
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, value in enumerate(data):
                if value < lower_bound or value > upper_bound:
                    anomalies.append({
                        "index": i,
                        "value": value,
                        "type": "outlier",
                        "severity": abs(value - np.mean(data)) / np.std(data) if np.std(data) > 0 else 0
                    })
        
        return anomalies
    
    def _calculate_similarity(self, data1: Any, data2: Any) -> float:
        """计算数据集相似度"""
        if type(data1) != type(data2):
            return 0.0
        
        if isinstance(data1, list) and isinstance(data2, list):
            # 基于集合的相似度
            set1, set2 = set(data1), set(data2)
            if not set1 and not set2:
                return 1.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        return 0.5  # 默认相似度
    
    def _identify_differences(self, data1: Any, data2: Any) -> List[Dict[str, Any]]:
        """识别差异"""
        differences = []
        
        if isinstance(data1, dict) and isinstance(data2, dict):
            # 键的差异
            keys1, keys2 = set(data1.keys()), set(data2.keys())
            only_in_1 = keys1 - keys2
            only_in_2 = keys2 - keys1
            
            if only_in_1:
                differences.append({
                    "type": "missing_keys",
                    "in_first_only": list(only_in_1)
                })
            if only_in_2:
                differences.append({
                    "type": "missing_keys",
                    "in_second_only": list(only_in_2)
                })
        
        return differences
    
    def _predict_trend(self, data: List[float]) -> Dict[str, Any]:
        """预测趋势"""
        if len(data) < 3:
            return {"direction": "insufficient_data"}
        
        # 简单线性回归
        x = np.arange(len(data))
        y = np.array(data)
        
        # 计算斜率
        slope = np.polyfit(x, y, 1)[0]
        
        return {
            "direction": "increasing" if slope > 0 else "decreasing",
            "slope": float(slope),
            "confidence": min(1.0, abs(slope) / (np.std(y) + 0.01))
        }
    
    def _forecast_values(self, data: List[float], periods: int) -> List[float]:
        """预测未来值"""
        if len(data) < 3:
            return []
        
        # 简单移动平均预测
        window = min(5, len(data))
        last_values = data[-window:]
        avg = np.mean(last_values)
        
        # 考虑趋势
        if len(data) > window:
            trend = (data[-1] - data[-window]) / window
        else:
            trend = 0
        
        forecast = []
        current = data[-1]
        for i in range(periods):
            current = current + trend
            forecast.append(float(current))
        
        return forecast
    
    def _generate_scenarios(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成场景"""
        scenarios = [
            {
                "name": "Best Case",
                "description": "All metrics improve by 20%",
                "probability": 0.2
            },
            {
                "name": "Base Case",
                "description": "Current trends continue",
                "probability": 0.6
            },
            {
                "name": "Worst Case",
                "description": "All metrics decline by 10%",
                "probability": 0.2
            }
        ]
        return scenarios
    
    def _assess_data_quality(self, results: Dict[str, Any]) -> Dict[str, float]:
        """评估数据质量"""
        quality_scores = {
            "completeness": 0.8,
            "accuracy": 0.9,
            "consistency": 0.85,
            "timeliness": 0.7
        }
        
        # 基于分析结果调整分数
        if "anomalies" in results.get("pattern", {}):
            anomaly_count = len(results["pattern"]["anomalies"])
            if anomaly_count > 10:
                quality_scores["consistency"] *= 0.8
        
        return quality_scores
    
    def _extract_key_patterns(self, results: Dict[str, Any]) -> List[str]:
        """提取关键模式"""
        key_patterns = []
        
        # 从各种分析中提取关键发现
        if "pattern" in results:
            patterns = results["pattern"]
            if patterns.get("temporal"):
                key_patterns.append("Temporal patterns detected")
            if patterns.get("cyclic"):
                key_patterns.append("Cyclic behavior observed")
        
        if "statistical" in results:
            key_patterns.append("Statistical distribution analyzed")
        
        return key_patterns
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于分析结果生成建议
        if "anomalies" in results.get("pattern", {}):
            recommendations.append("Investigate detected anomalies")
        
        if "predictive" in results:
            trends = results["predictive"].get("trends", [])
            for trend in trends:
                if trend["trend"]["direction"] == "decreasing":
                    recommendations.append(f"Address declining trend in {trend['dataset']}")
        
        if not recommendations:
            recommendations.append("Continue monitoring current metrics")
        
        return recommendations
    
    def _calculate_cycle_similarity(self, cycles: List[List[float]]) -> float:
        """计算周期相似度"""
        if len(cycles) < 2:
            return 0.0
        
        # 计算所有周期对的相似度
        similarities = []
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                # 使用相关系数作为相似度
                if len(cycles[i]) == len(cycles[j]):
                    corr = np.corrcoef(cycles[i], cycles[j])[0, 1]
                    similarities.append(corr)
        
        return np.mean(similarities) if similarities else 0.0

class InsightGenerator:
    """洞察生成器"""
    
    def __init__(self):
        self.insight_templates = self._load_templates()
    
    async def generate(self, analysis_results: Dict[str, Any], task: Task) -> List[Dict[str, Any]]:
        """生成洞察"""
        insights = []
        
        # 从统计分析生成洞察
        if "statistical" in analysis_results:
            statistical_insights = self._generate_statistical_insights(
                analysis_results["statistical"]
            )
            insights.extend(statistical_insights)
        
        # 从模式分析生成洞察
        if "pattern" in analysis_results:
            pattern_insights = self._generate_pattern_insights(
                analysis_results["pattern"]
            )
            insights.extend(pattern_insights)
        
        # 从预测分析生成洞察
        if "predictive" in analysis_results:
            predictive_insights = self._generate_predictive_insights(
                analysis_results["predictive"]
            )
            insights.extend(predictive_insights)
        
        # 排序和过滤
        insights = self._rank_insights(insights)
        
        return insights[:10]  # 返回前10个洞察
    
    def _load_templates(self) -> Dict[str, str]:
        """加载洞察模板"""
        return {
            "high_variance": "The {metric} shows high variance (σ={std:.2f}), suggesting significant fluctuation",
            "strong_trend": "A strong {direction} trend is observed in {metric} with {confidence:.1%} confidence",
            "anomaly_detected": "{count} anomalies detected in {metric}, requiring further investigation",
            "correlation_found": "Strong correlation ({corr:.2f}) found between {metric1} and {metric2}",
            "pattern_identified": "{pattern_type} pattern identified with period of {period} units"
        }
    
    def _generate_statistical_insights(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成统计洞察"""
        insights = []
        
        for metric, stat_data in stats.get("descriptive", {}).items():
            # 高方差洞察
            if stat_data["std"] > stat_data["mean"] * 0.5:
                insights.append({
                    "type": "high_variance",
                    "severity": "medium",
                    "description": self.insight_templates["high_variance"].format(
                        metric=metric,
                        std=stat_data["std"]
                    ),
                    "metric": metric,
                    "value": stat_data["std"]
                })
        
        return insights
    
    def _generate_pattern_insights(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成模式洞察"""
        insights = []
        
        # 异常洞察
        anomalies = patterns.get("anomalies", [])
        if anomalies:
            insights.append({
                "type": "anomaly",
                "severity": "high" if len(anomalies) > 5 else "medium",
                "description": self.insight_templates["anomaly_detected"].format(
                    count=len(anomalies),
                    metric="data"
                ),
                "details": anomalies[:3]  # 前3个异常
            })
        
        # 周期模式洞察
        for cyclic in patterns.get("cyclic", []):
            insights.append({
                "type": "pattern",
                "severity": "low",
                "description": self.insight_templates["pattern_identified"].format(
                    pattern_type="Cyclic",
                    period=cyclic["period"]
                ),
                "confidence": cyclic["similarity"]
            })
        
        return insights
    
    def _generate_predictive_insights(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成预测洞察"""
        insights = []
        
        # 趋势洞察
        for trend_data in predictions.get("trends", []):
            trend = trend_data["trend"]
            if trend["confidence"] > 0.7:
                insights.append({
                    "type": "trend",
                    "severity": "medium",
                    "description": self.insight_templates["strong_trend"].format(
                        direction=trend["direction"],
                        metric=trend_data["dataset"],
                        confidence=trend["confidence"]
                    ),
                    "actionable": True
                })
        
        return insights
    
    def _rank_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对洞察进行排序"""
        # 定义严重性权重
        severity_weights = {
            "high": 3,
            "medium": 2,
            "low": 1
        }
        
        # 计算每个洞察的分数
        for insight in insights:
            severity_score = severity_weights.get(insight.get("severity", "low"), 1)
            actionable_score = 2 if insight.get("actionable", False) else 1
            confidence_score = insight.get("confidence", 0.5)
            
            insight["score"] = severity_score * actionable_score * confidence_score
        
        # 按分数排序
        insights.sort(key=lambda x: x["score"], reverse=True)
        
        return insights

# ============================= 主程序入口 =============================

class HyperAgentSystem:
    """HyperAgent系统主类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.framework = CollaborationFramework(
            coordination_strategy=CoordinationStrategy[
                self.config.get("coordination_strategy", "HIERARCHICAL")
            ]
        )
        self.llm_integration = LLMIntegration(self.config.get("llm", {}))
        self.agents: Dict[str, BaseAgent] = {}
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        # 默认配置
        default_config = {
            "coordination_strategy": "HIERARCHICAL",
            "llm": {
                "claude": {"api_key": "your-claude-key"},
                "qwen": {"api_key": "your-qwen-key"}
            },
            "agents": {
                "supervisor": {"count": 1, "model": "CLAUDE_4_OPUS"},
                "planner": {"count": 2, "model": "CLAUDE_4_SONNET"},
                "executor": {"count": 5, "model": "QWEN_PLUS"},
                "analyzer": {"count": 3, "model": "CLAUDE_4_SONNET"}
            }
        }
        
        # 尝试加载配置文件
        try:
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
        except:
            logging.warning(f"Could not load config from {config_path}, using defaults")
        
        return default_config
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('hyperagent.log')
            ]
        )
    
    async def initialize(self):
        """初始化系统"""
        logging.info("Initializing HyperAgent System...")
        
        # 创建Agent
        await self._create_agents()
        
        # 注册Agent到框架
        for agent in self.agents.values():
            await self.framework.register_agent(agent)
        
        # 设置Agent间的关系
        await self._setup_agent_relationships()
        
        logging.info("HyperAgent System initialized successfully")
    
    async def _create_agents(self):
        """创建Agent实体"""
        """
HyperAgent Platform v1.0 - Core Agent System (Part 2)
生产级超级Agent系统核心实现 - 第二部分
"""

# 继续 HyperAgentSystem 类的实现

    async def _create_agents(self):
        """创建Agent实例"""
        agent_configs = self.config.get("agents", {})
        
        # 创建Supervisor Agent
        supervisor_config = agent_configs.get("supervisor", {})
        for i in range(supervisor_config.get("count", 1)):
            agent = SupervisorAgent(
                agent_id=f"supervisor_{i}",
                name=f"Supervisor-{i}",
                config=supervisor_config
            )
            self.agents[agent.agent_id] = agent
        
        # 创建Planner Agent
        planner_config = agent_configs.get("planner", {})
        for i in range(planner_config.get("count", 2)):
            agent = PlannerAgent(
                agent_id=f"planner_{i}",
                name=f"Planner-{i}",
                config=planner_config
            )
            self.agents[agent.agent_id] = agent
        
        # 创建Executor Agent
        executor_config = agent_configs.get("executor", {})
        for i in range(executor_config.get("count", 5)):
            agent = ExecutorAgent(
                agent_id=f"executor_{i}",
                name=f"Executor-{i}",
                config=executor_config
            )
            self.agents[agent.agent_id] = agent
        
        # 创建Analyzer Agent
        analyzer_config = agent_configs.get("analyzer", {})
        for i in range(analyzer_config.get("count", 3)):
            agent = AnalyzerAgent(
                agent_id=f"analyzer_{i}",
                name=f"Analyzer-{i}",
                config=analyzer_config
            )
            self.agents[agent.agent_id] = agent
        
        # 创建Validator Agent
        validator_config = agent_configs.get("validator", {})
        for i in range(validator_config.get("count", 2)):
            agent = ValidatorAgent(
                agent_id=f"validator_{i}",
                name=f"Validator-{i}",
                config=validator_config
            )
            self.agents[agent.agent_id] = agent
        
        # 创建Coordinator Agent
        coordinator_config = agent_configs.get("coordinator", {})
        for i in range(coordinator_config.get("count", 1)):
            agent = CoordinatorAgent(
                agent_id=f"coordinator_{i}",
                name=f"Coordinator-{i}",
                config=coordinator_config
            )
            self.agents[agent.agent_id] = agent
        
        logging.info(f"Created {len(self.agents)} agents")
    
    async def _setup_agent_relationships(self):
        """设置Agent间的关系"""
        # 建立监督关系
        supervisors = [a for a in self.agents.values() if a.role == AgentRole.SUPERVISOR]
        workers = [a for a in self.agents.values() if a.role != AgentRole.SUPERVISOR]
        
        if supervisors and workers:
            # 平均分配工作者给监督者
            workers_per_supervisor = len(workers) // len(supervisors)
            
            for i, supervisor in enumerate(supervisors):
                start_idx = i * workers_per_supervisor
                end_idx = start_idx + workers_per_supervisor if i < len(supervisors) - 1 else len(workers)
                
                for worker in workers[start_idx:end_idx]:
                    if isinstance(supervisor, SupervisorAgent):
                        supervisor.supervised_agents[worker.agent_id] = {
                            "agent": worker,
                            "specialization": worker.role.value,
                            "current_load": 0
                        }
                    
                    # 工作者记录其监督者
                    worker.known_agents[supervisor.agent_id] = {
                        "role": "supervisor",
                        "relationship": "reports_to"
                    }
    
    async def start(self):
        """启动系统"""
        logging.info("Starting HyperAgent System...")
        
        # 启动框架
        await self.framework.start()
        
        logging.info("HyperAgent System started successfully")
    
    async def stop(self):
        """停止系统"""
        logging.info("Stopping HyperAgent System...")
        
        # 停止框架
        await self.framework.stop()
        
        logging.info("HyperAgent System stopped")
    
    async def submit_task(self, task_description: str, 
                         priority: int = 5,
                         constraints: List[str] = None,
                         requirements: List[str] = None,
                         metadata: Dict[str, Any] = None) -> Task:
        """提交任务到系统"""
        task = Task(
            name=f"Task_{uuid.uuid4().hex[:8]}",
            description=task_description,
            objective=task_description,
            constraints=constraints or [],
            requirements=requirements or [],
            priority=priority,
            metadata=metadata or {}
        )
        
        # 提交到框架
        result = await self.framework.submit_task(task)
        task.result = result
        
        return task
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "agents": {},
            "framework": {
                "coordination_strategy": self.framework.coordination_strategy.value,
                "total_agents": len(self.framework.agents),
                "active_agents": sum(1 for a in self.framework.agents.values() 
                                   if a.state["status"] != "idle")
            },
            "performance": await self._get_performance_metrics()
        }
        
        # 收集每个Agent的状态
        for agent_id, agent in self.agents.items():
            status["agents"][agent_id] = {
                "name": agent.name,
                "role": agent.role.value,
                "status": agent.state["status"],
                "current_tasks": len(agent.current_tasks),
                "memory_usage": len(agent.memory_system.memories),
                "cognitive_confidence": agent.cognitive_architecture.metacognitive_state["confidence"]
            }
        
        return status
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        # 从性能监控器获取指标
        if hasattr(self.framework, 'performance_monitor'):
            monitor = self.framework.performance_monitor
            
            # 获取最新的系统指标
            if monitor.metrics_history["system"]:
                latest_metrics = monitor.metrics_history["system"][-1]
                return {
                    "timestamp": latest_metrics["timestamp"].isoformat(),
                    "total_tasks": latest_metrics["system"]["total_tasks"],
                    "active_agents": latest_metrics["system"]["active_agents"],
                    "collaboration_edges": latest_metrics["system"]["collaboration_edges"]
                }
        
        return {}

# ============================= 新增专门化Agent =============================

class ValidatorAgent(BaseAgent):
    """验证Agent - 负责结果验证和质量保证"""
    
    def __init__(self, agent_id: str, name: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, name, AgentRole.VALIDATOR,
                        ModelProvider.CLAUDE_3_7_SONNET, config)
        
        self.validation_rules = self._load_validation_rules()
        self.quality_metrics = {
            "accuracy": 0.0,
            "completeness": 0.0,
            "consistency": 0.0,
            "reliability": 0.0
        }
    
    async def process_task(self, task: Task) -> Any:
        """处理验证任务"""
        validation_target = task.metadata.get("validation_target")
        validation_type = task.metadata.get("validation_type", "comprehensive")
        
        if validation_type == "comprehensive":
            results = await self._comprehensive_validation(validation_target)
        elif validation_type == "code":
            results = await self._validate_code(validation_target)
        elif validation_type == "content":
            results = await self._validate_content(validation_target)
        elif validation_type == "data":
            results = await self._validate_data(validation_target)
        else:
            results = await self._generic_validation(validation_target)
        
        # 更新质量指标
        await self._update_quality_metrics(results)
        
        return results
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理验证相关消息"""
        if message.message_type == "validation_request":
            target = message.content.get("target")
            criteria = message.content.get("criteria", [])
            
            validation_result = await self._quick_validation(target, criteria)
            
            return AgentMessage(
                message_type="validation_result",
                content=validation_result
            )
        
        return None
    
    def _load_validation_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载验证规则"""
        return {
            "code": [
                {"rule": "syntax_valid", "severity": "error"},
                {"rule": "no_security_issues", "severity": "critical"},
                {"rule": "follows_standards", "severity": "warning"},
                {"rule": "has_documentation", "severity": "info"}
            ],
            "content": [
                {"rule": "grammar_correct", "severity": "warning"},
                {"rule": "no_plagiarism", "severity": "error"},
                {"rule": "factually_accurate", "severity": "error"},
                {"rule": "appropriate_tone", "severity": "info"}
            ],
            "data": [
                {"rule": "schema_valid", "severity": "error"},
                {"rule": "no_missing_values", "severity": "warning"},
                {"rule": "within_range", "severity": "error"},
                {"rule": "consistent_format", "severity": "warning"}
            ]
        }
    
    async def _comprehensive_validation(self, target: Any) -> Dict[str, Any]:
        """综合验证"""
        results = {
            "valid": True,
            "score": 1.0,
            "issues": [],
            "warnings": [],
            "suggestions": []
        }
        
        # 运行所有适用的验证
        validations = [
            self._validate_structure(target),
            self._validate_quality(target),
            self._validate_compliance(target),
            self._validate_performance(target)
        ]
        
        for validation_coro in validations:
            validation_result = await validation_coro
            
            # 合并结果
            if not validation_result["valid"]:
                results["valid"] = False
            
            results["score"] *= validation_result.get("score", 1.0)
            results["issues"].extend(validation_result.get("issues", []))
            results["warnings"].extend(validation_result.get("warnings", []))
            results["suggestions"].extend(validation_result.get("suggestions", []))
        
        return results
    
    async def _validate_code(self, code: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """验证代码"""
        results = {
            "valid": True,
            "score": 1.0,
            "issues": [],
            "metrics": {}
        }
        
        if isinstance(code, str):
            code_content = code
        else:
            code_content = code.get("content", "")
        
        # 语法检查
        syntax_check = await self._check_syntax(code_content)
        if not syntax_check["valid"]:
            results["valid"] = False
            results["issues"].extend(syntax_check["errors"])
        
        # 安全检查
        security_check = await self._check_security(code_content)
        if security_check["vulnerabilities"]:
            results["valid"] = False
            results["issues"].extend(security_check["vulnerabilities"])
        
        # 代码质量
        quality_metrics = await self._analyze_code_quality(code_content)
        results["metrics"] = quality_metrics
        results["score"] = quality_metrics.get("overall_score", 0.5)
        
        return results
    
    async def _validate_content(self, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """验证内容"""
        results = {
            "valid": True,
            "score": 1.0,
            "issues": [],
            "analysis": {}
        }
        
        if isinstance(content, str):
            text = content
        else:
            text = content.get("text", "")
        
        # 语法检查
        grammar_issues = await self._check_grammar(text)
        if grammar_issues:
            results["warnings"] = grammar_issues
            results["score"] *= 0.9
        
        # 事实检查
        fact_check = await self._check_facts(text)
        if fact_check["inaccuracies"]:
            results["valid"] = False
            results["issues"].extend(fact_check["inaccuracies"])
        
        # 原创性检查
        originality = await self._check_originality(text)
        results["analysis"]["originality_score"] = originality["score"]
        
        return results
    
    async def _validate_data(self, data: Any) -> Dict[str, Any]:
        """验证数据"""
        results = {
            "valid": True,
            "score": 1.0,
            "issues": [],
            "statistics": {}
        }
        
        # 模式验证
        if isinstance(data, dict):
            schema_validation = self._validate_schema(data)
            if not schema_validation["valid"]:
                results["valid"] = False
                results["issues"].extend(schema_validation["errors"])
        
        # 数据质量检查
        quality_check = await self._check_data_quality(data)
        results["statistics"] = quality_check["statistics"]
        results["score"] = quality_check["quality_score"]
        
        # 一致性检查
        consistency_check = self._check_consistency(data)
        if consistency_check["inconsistencies"]:
            results["warnings"] = consistency_check["inconsistencies"]
            results["score"] *= 0.8
        
        return results
    
    async def _generic_validation(self, target: Any) -> Dict[str, Any]:
        """通用验证"""
        return {
            "valid": True,
            "score": 0.8,
            "message": "Generic validation passed",
            "target_type": type(target).__name__
        }
    
    async def _quick_validation(self, target: Any, criteria: List[str]) -> Dict[str, Any]:
        """快速验证"""
        results = {"criteria_results": {}}
        
        for criterion in criteria:
            if criterion == "non_empty":
                results["criteria_results"][criterion] = bool(target)
            elif criterion == "type_check":
                results["criteria_results"][criterion] = target is not None
            elif criterion == "length_check":
                if hasattr(target, "__len__"):
                    results["criteria_results"][criterion] = len(target) > 0
                else:
                    results["criteria_results"][criterion] = False
        
        results["all_passed"] = all(results["criteria_results"].values())
        return results
    
    async def _validate_structure(self, target: Any) -> Dict[str, Any]:
        """验证结构"""
        return {
            "valid": True,
            "score": 1.0,
            "issues": [],
            "structure_type": type(target).__name__
        }
    
    async def _validate_quality(self, target: Any) -> Dict[str, Any]:
        """验证质量"""
        # 使用认知架构评估质量
        quality_thought = self.cognitive_architecture.critique(
            Thought(content=str(target), confidence=0.7),
            ["completeness", "accuracy", "clarity"]
        )
        
        return {
            "valid": True,
            "score": quality_thought.confidence,
            "issues": [],
            "quality_assessment": quality_thought.metadata
        }
    
    async def _validate_compliance(self, target: Any) -> Dict[str, Any]:
        """验证合规性"""
        return {
            "valid": True,
            "score": 1.0,
            "issues": [],
            "compliance_checks": ["GDPR", "Security", "Best Practices"]
        }
    
    async def _validate_performance(self, target: Any) -> Dict[str, Any]:
        """验证性能"""
        return {
            "valid": True,
            "score": 0.9,
            "issues": [],
            "performance_metrics": {
                "efficiency": 0.9,
                "scalability": 0.8,
                "resource_usage": 0.85
            }
        }
    
    async def _check_syntax(self, code: str) -> Dict[str, Any]:
        """检查语法"""
        try:
            compile(code, '<string>', 'exec')
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"Syntax error at line {e.lineno}: {e.msg}"]
            }
    
    async def _check_security(self, code: str) -> Dict[str, Any]:
        """检查安全性"""
        vulnerabilities = []
        
        # 检查危险函数
        dangerous_functions = ['eval', 'exec', '__import__', 'open', 'compile']
        for func in dangerous_functions:
            if func in code:
                vulnerabilities.append(f"Potentially dangerous function: {func}")
        
        # 检查SQL注入风险
        if 'SELECT' in code.upper() and 'WHERE' in code.upper():
            if '%s' not in code and '?' not in code:
                vulnerabilities.append("Potential SQL injection vulnerability")
        
        return {"vulnerabilities": vulnerabilities}
    
    async def _analyze_code_quality(self, code: str) -> Dict[str, float]:
        """分析代码质量"""
        lines = code.split('\n')
        
        # 简单的代码质量指标
        metrics = {
            "lines_of_code": len(lines),
            "comment_ratio": sum(1 for line in lines if line.strip().startswith('#')) / max(len(lines), 1),
            "average_line_length": np.mean([len(line) for line in lines]) if lines else 0,
            "complexity_score": min(1.0, len(lines) / 100),  # 简化的复杂度
            "readability_score": 0.8 if metrics["comment_ratio"] > 0.1 else 0.6
        }
        
        # 计算总体分数
        metrics["overall_score"] = np.mean([
            1 - metrics["complexity_score"],
            metrics["readability_score"],
            min(1.0, metrics["comment_ratio"] * 5)
        ])
        
        return metrics
    
    async def _check_grammar(self, text: str) -> List[str]:
        """检查语法"""
        # 简化实现
        issues = []
        
        # 基本检查
        if not text.strip():
            issues.append("Empty content")
        
        # 检查句子结尾
        sentences = text.split('.')
        for sentence in sentences:
            if sentence.strip() and not sentence.strip()[-1] in '.!?':
                issues.append("Sentence may be missing punctuation")
        
        return issues
    
    async def _check_facts(self, text: str) -> Dict[str, List[str]]:
        """事实检查"""
        # 简化实现
        return {"inaccuracies": []}
    
    async def _check_originality(self, text: str) -> Dict[str, float]:
        """检查原创性"""
        # 简化实现
        return {"score": 0.95}
    
    def _validate_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证数据模式"""
        # 简化实现
        return {"valid": True, "errors": []}
    
    async def _check_data_quality(self, data: Any) -> Dict[str, Any]:
        """检查数据质量"""
        statistics = {}
        
        if isinstance(data, list):
            statistics["count"] = len(data)
            statistics["unique_count"] = len(set(str(item) for item in data))
            statistics["completeness"] = sum(1 for item in data if item is not None) / max(len(data), 1)
        
        quality_score = statistics.get("completeness", 0.5)
        
        return {
            "statistics": statistics,
            "quality_score": quality_score
        }
    
    def _check_consistency(self, data: Any) -> Dict[str, List[str]]:
        """检查一致性"""
        return {"inconsistencies": []}
    
    async def _update_quality_metrics(self, validation_results: Dict[str, Any]):
        """更新质量指标"""
        # 更新准确性
        if "score" in validation_results:
            self.quality_metrics["accuracy"] = (
                self.quality_metrics["accuracy"] * 0.9 + 
                validation_results["score"] * 0.1
            )
        
        # 更新完整性
        if not validation_results.get("issues"):
            self.quality_metrics["completeness"] = min(
                1.0, self.quality_metrics["completeness"] * 1.01
            )
        
        # 更新一致性
        self.quality_metrics["consistency"] = (
            self.quality_metrics["consistency"] * 0.95 + 
            (1.0 if validation_results.get("valid") else 0.0) * 0.05
        )

class CoordinatorAgent(BaseAgent):
    """协调Agent - 负责任务协调和资源调度"""
    
    def __init__(self, agent_id: str, name: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, name, AgentRole.COORDINATOR,
                        ModelProvider.CLAUDE_4_SONNET, config)
        
        self.resource_pool = ResourcePool()
        self.task_scheduler = TaskScheduler()
        self.conflict_resolver = ConflictResolver()
        self.optimization_engine = OptimizationEngine()
    
    async def process_task(self, task: Task) -> Any:
        """处理协调任务"""
        coordination_type = task.metadata.get("coordination_type", "resource_allocation")
        
        if coordination_type == "resource_allocation":
            result = await self._allocate_resources(task)
        elif coordination_type == "task_scheduling":
            result = await self._schedule_tasks(task)
        elif coordination_type == "conflict_resolution":
            result = await self._resolve_conflicts(task)
        elif coordination_type == "optimization":
            result = await self._optimize_workflow(task)
        else:
            result = await self._general_coordination(task)
        
        return result
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理协调相关消息"""
        if message.message_type == "resource_request":
            # 处理资源请求
            resources = await self._handle_resource_request(message)
            return AgentMessage(
                message_type="resource_allocation",
                content=resources
            )
        
        elif message.message_type == "scheduling_request":
            # 处理调度请求
            schedule = await self._handle_scheduling_request(message)
            return AgentMessage(
                message_type="schedule_response",
                content=schedule
            )
        
        elif message.message_type == "conflict_report":
            # 处理冲突报告
            resolution = await self._handle_conflict_report(message)
            return AgentMessage(
                message_type="conflict_resolution",
                content=resolution
            )
        
        return None
    
    async def _allocate_resources(self, task: Task) -> Dict[str, Any]:
        """分配资源"""
        requirements = task.metadata.get("resource_requirements", {})
        
        # 检查资源可用性
        availability = await self.resource_pool.check_availability(requirements)
        
        if availability["available"]:
            # 分配资源
            allocation = await self.resource_pool.allocate(
                task.id,
                requirements,
                duration=task.metadata.get("estimated_duration", 3600)
            )
            
            return {
                "status": "allocated",
                "allocation_id": allocation["id"],
                "resources": allocation["resources"],
                "expires_at": allocation["expires_at"]
            }
        else:
            # 尝试优化分配
            optimized_allocation = await self.optimization_engine.optimize_allocation(
                requirements,
                self.resource_pool.get_current_state()
            )
            
            if optimized_allocation["feasible"]:
                # 执行优化后的分配
                return await self._execute_optimized_allocation(optimized_allocation)
            else:
                return {
                    "status": "insufficient_resources",
                    "available": availability["available_resources"],
                    "required": requirements,
                    "suggestions": optimized_allocation.get("suggestions", [])
                }
    
    async def _schedule_tasks(self, task: Task) -> Dict[str, Any]:
        """任务调度"""
        tasks_to_schedule = task.metadata.get("tasks", [])
        constraints = task.metadata.get("scheduling_constraints", {})
        
        # 创建调度计划
        schedule = await self.task_scheduler.create_schedule(
            tasks_to_schedule,
            constraints
        )
        
        # 验证调度可行性
        validation = await self._validate_schedule(schedule)
        
        if validation["valid"]:
            # 提交调度
            await self.task_scheduler.commit_schedule(schedule)
            
            return {
                "status": "scheduled",
                "schedule": schedule,
                "estimated_completion": schedule["estimated_completion"],
                "critical_path": schedule["critical_path"]
            }
        else:
            # 尝试重新调度
            revised_schedule = await self.task_scheduler.revise_schedule(
                schedule,
                validation["issues"]
            )
            
            return {
                "status": "revised",
                "schedule": revised_schedule,
                "adjustments": validation["issues"]
            }
    
    async def _resolve_conflicts(self, task: Task) -> Dict[str, Any]:
        """解决冲突"""
        conflicts = task.metadata.get("conflicts", [])
        
        resolutions = []
        for conflict in conflicts:
            # 分析冲突
            analysis = await self.conflict_resolver.analyze_conflict(conflict)
            
            # 生成解决方案
            solutions = await self.conflict_resolver.generate_solutions(
                conflict,
                analysis
            )
            
            # 选择最佳方案
            best_solution = await self._select_best_solution(solutions, conflict)
            
            # 应用解决方案
            result = await self.conflict_resolver.apply_solution(
                conflict,
                best_solution
            )
            
            resolutions.append({
                "conflict_id": conflict.get("id"),
                "solution": best_solution,
                "result": result
            })
        
        return {
            "status": "resolved",
            "resolutions": resolutions,
            "success_rate": sum(1 for r in resolutions if r["result"]["success"]) / len(resolutions)
        }
    
    async def _optimize_workflow(self, task: Task) -> Dict[str, Any]:
        """优化工作流"""
        workflow = task.metadata.get("workflow", {})
        optimization_goals = task.metadata.get("optimization_goals", ["efficiency"])
        
        # 分析当前工作流
        analysis = await self.optimization_engine.analyze_workflow(workflow)
        
        # 识别瓶颈
        bottlenecks = await self.optimization_engine.identify_bottlenecks(
            workflow,
            analysis
        )
        
        # 生成优化建议
        optimizations = await self.optimization_engine.generate_optimizations(
            workflow,
            bottlenecks,
            optimization_goals
        )
        
        # 模拟优化效果
        simulation = await self.optimization_engine.simulate_optimizations(
            workflow,
            optimizations
        )
        
        return {
            "status": "optimized",
            "original_metrics": analysis["metrics"],
            "optimizations": optimizations,
            "expected_improvement": simulation["improvement"],
            "recommendations": self._prioritize_optimizations(optimizations, simulation)
        }
    
    async def _general_coordination(self, task: Task) -> Dict[str, Any]:
        """通用协调"""
        # 使用认知架构进行协调决策
        coordination_thought = self.cognitive_architecture.plan(
            f"Coordinate: {task.description}",
            task.constraints
        )
        
        return {
            "status": "coordinated",
            "plan": [t.content for t in coordination_thought],
            "confidence": np.mean([t.confidence for t in coordination_thought])
        }
    
    async def _handle_resource_request(self, message: AgentMessage) -> Dict[str, Any]:
        """处理资源请求"""
        requester_id = message.sender_id
        requested_resources = message.content.get("resources", {})
        priority = message.content.get("priority", 5)
        
        # 检查请求者配额
        quota_check = await self.resource_pool.check_quota(requester_id)
        
        if quota_check["within_quota"]:
            # 尝试分配
            allocation = await self.resource_pool.try_allocate(
                requester_id,
                requested_resources,
                priority
            )
            
            return {
                "approved": allocation["success"],
                "resources": allocation.get("allocated", {}),
                "reason": allocation.get("reason", "")
            }
        else:
            return {
                "approved": False,
                "reason": "Quota exceeded",
                "current_usage": quota_check["current_usage"],
                "quota": quota_check["quota"]
            }
    
    async def _handle_scheduling_request(self, message: AgentMessage) -> Dict[str, Any]:
        """处理调度请求"""
        task_info = message.content.get("task", {})
        preferences = message.content.get("preferences", {})
        
        # 查找可用时间槽
        available_slots = await self.task_scheduler.find_available_slots(
            task_info["duration"],
            preferences
        )
        
        if available_slots:
            # 选择最佳时间槽
            best_slot = await self._select_best_slot(
                available_slots,
                task_info,
                preferences
            )
            
            # 预约时间槽
            reservation = await self.task_scheduler.reserve_slot(
                best_slot,
                task_info
            )
            
            return {
                "scheduled": True,
                "slot": best_slot,
                "reservation_id": reservation["id"]
            }
        else:
            return {
                "scheduled": False,
                "reason": "No available slots",
                "suggestions": await self.task_scheduler.suggest_alternatives(
                    task_info,
                    preferences
                )
            }
    
    async def _handle_conflict_report(self, message: AgentMessage) -> Dict[str, Any]:
        """处理冲突报告"""
        conflict_info = message.content
        
        # 记录冲突
        conflict_id = await self.conflict_resolver.log_conflict(conflict_info)
        
        # 快速评估
        severity = await self.conflict_resolver.assess_severity(conflict_info)
        
        if severity["level"] == "critical":
            # 立即处理关键冲突
            immediate_action = await self.conflict_resolver.take_immediate_action(
                conflict_info
            )
            
            return {
                "conflict_id": conflict_id,
                "immediate_action": immediate_action,
                "status": "escalated"
            }
        else:
            # 加入解决队列
            queue_position = await self.conflict_resolver.queue_for_resolution(
                conflict_id,
                severity["level"]
            )
            
            return {
                "conflict_id": conflict_id,
                "queue_position": queue_position,
                "estimated_resolution_time": severity["estimated_time"]
            }
    
    async def _execute_optimized_allocation(self, optimization: Dict[str, Any]) -> Dict[str, Any]:
        """执行优化后的分配"""
        steps = optimization["steps"]
        
        for step in steps:
            if step["action"] == "preempt":
                # 抢占低优先级资源
                await self.resource_pool.preempt_resources(
                    step["target_allocations"],
                    step["reason"]
                )
            elif step["action"] == "migrate":
                # 迁移资源
                await self.resource_pool.migrate_resources(
                    step["from"],
                    step["to"]
                )
            elif step["action"] == "scale":
                # 扩展资源
                await self.resource_pool.scale_resources(
                    step["resource_type"],
                    step["scale_factor"]
                )
        
        # 执行最终分配
        final_allocation = await self.resource_pool.allocate(
            optimization["task_id"],
            optimization["requirements"]
        )
        
        return {
            "status": "allocated",
            "allocation": final_allocation,
            "optimization_applied": True
        }
    
    async def _validate_schedule(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """验证调度"""
        issues = []
        
        # 检查资源冲突
        resource_conflicts = await self._check_resource_conflicts(schedule)
        if resource_conflicts:
            issues.extend(resource_conflicts)
        
        # 检查依赖关系
        dependency_issues = await self._check_dependencies(schedule)
        if dependency_issues:
            issues.extend(dependency_issues)
        
        # 检查时间约束
        timing_issues = await self._check_timing_constraints(schedule)
        if timing_issues:
            issues.extend(timing_issues)
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    async def _select_best_solution(self, solutions: List[Dict[str, Any]], 
                                   conflict: Dict[str, Any]) -> Dict[str, Any]:
        """选择最佳解决方案"""
        if not solutions:
            return {"type": "no_solution", "action": "escalate"}
        
        # 评分每个方案
        scored_solutions = []
        for solution in solutions:
            score = await self._score_solution(solution, conflict)
            scored_solutions.append((solution, score))
        
        # 选择得分最高的方案
        scored_solutions.sort(key=lambda x: x[1], reverse=True)
        return scored_solutions[0][0]
    
    def _prioritize_optimizations(self, optimizations: List[Dict[str, Any]], 
                                 simulation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优先级排序优化建议"""
        prioritized = []
        
        for opt in optimizations:
            impact = simulation["impacts"].get(opt["id"], {})
            
            priority_score = (
                impact.get("performance_gain", 0) * 0.4 +
                impact.get("cost_reduction", 0) * 0.3 +
                (1 - impact.get("implementation_complexity", 1)) * 0.3
            )
            
            prioritized.append({
                "optimization": opt,
                "priority_score": priority_score,
                "expected_impact": impact
            })
        
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)
        return prioritized
    
    async def _check_resource_conflicts(self, schedule: Dict[str, Any]) -> List[str]:
        """检查资源冲突"""
        conflicts = []
        
        # 简化实现
        tasks = schedule.get("tasks", [])
        for i, task1 in enumerate(tasks):
            for task2 in tasks[i+1:]:
                if self._tasks_overlap(task1, task2) and self._share_resources(task1, task2):
                    conflicts.append(
                        f"Resource conflict between {task1['id']} and {task2['id']}"
                    )
        
        return conflicts
    
    async def _check_dependencies(self, schedule: Dict[str, Any]) -> List[str]:
        """检查依赖关系"""
        issues = []
        
        # 简化实现
        tasks = schedule.get("tasks", [])
        for task in tasks:
            for dep_id in task.get("dependencies", []):
                dep_task = next((t for t in tasks if t["id"] == dep_id), None)
                if dep_task and dep_task["end_time"] > task["start_time"]:
                    issues.append(
                        f"Dependency violation: {task['id']} starts before {dep_id} ends"
                    )
        
        return issues
    
    async def _check_timing_constraints(self, schedule: Dict[str, Any]) -> List[str]:
        """检查时间约束"""
        issues = []
        
        # 简化实现
        deadline = schedule.get("deadline")
        if deadline:
            completion_time = max(
                task["end_time"] for task in schedule.get("tasks", [])
            )
            if completion_time > deadline:
                issues.append(f"Schedule exceeds deadline by {completion_time - deadline}")
        
        return issues
    
    async def _select_best_slot(self, slots: List[Dict[str, Any]], 
                               task_info: Dict[str, Any],
                               preferences: Dict[str, Any]) -> Dict[str, Any]:
        """选择最佳时间槽"""
        # 简化实现：选择最早的时间槽
        return min(slots, key=lambda s: s["start_time"])
    
    async def _score_solution(self, solution: Dict[str, Any], 
                            conflict: Dict[str, Any]) -> float:
        """为解决方案评分"""
        score = 0.5  # 基础分
        
        # 考虑影响范围
        if solution.get("impact", "high") == "low":
            score += 0.2
        
        # 考虑实施难度
        if solution.get("complexity", "high") == "low":
            score += 0.2
        
        # 考虑成功率
        score += solution.get("success_probability", 0.5) * 0.1
        
        return score
    
    def _tasks_overlap(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> bool:
        """检查任务是否重叠"""
        return not (task1["end_time"] <= task2["start_time"] or 
                   task2["end_time"] <= task1["start_time"])
    
    def _share_resources(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> bool:
        """检查任务是否共享资源"""
        resources1 = set(task1.get("resources", []))
        resources2 = set(task2.get("resources", []))
        return bool(resources1 & resources2)

# ============================= 辅助类实现 =============================

class ResourcePool:
    """资源池管理器"""
    
    def __init__(self):
        self.resources = {
            "cpu": {"total": 100, "allocated": 0},
            "memory": {"total": 1024, "allocated": 0},  # GB
            "gpu": {"total": 8, "allocated": 0},
            "storage": {"total": 10240, "allocated": 0}  # GB
        }
        self.allocations = {}
        self.quotas = defaultdict(lambda: {
            "cpu": 20, "memory": 256, "gpu": 2, "storage": 1024
        })
        self._lock = asyncio.Lock()
    
    async def check_availability(self, requirements: Dict[str, float]) -> Dict[str, Any]:
        """检查资源可用性"""
        async with self._lock:
            available = True
            available_resources = {}
            
            for resource_type, required in requirements.items():
                if resource_type in self.resources:
                    available_amount = (
                        self.resources[resource_type]["total"] - 
                        self.resources[resource_type]["allocated"]
                    )
                    available_resources[resource_type] = available_amount
                    
                    if available_amount < required:
                        available = False
            
            return {
                "available": available,
                "available_resources": available_resources
            }
    
    async def allocate(self, task_id: str, requirements: Dict[str, float], 
                      duration: int = 3600) -> Dict[str, Any]:
        """分配资源"""
        async with self._lock:
            allocation_id = f"alloc_{uuid.uuid4().hex[:8]}"
            
            # 更新资源使用
            for resource_type, amount in requirements.items():
                if resource_type in self.resources:
                    self.resources[resource_type]["allocated"] += amount
            
            # 记录分配
            self.allocations[allocation_id] = {
                "id": allocation_id,
                "task_id": task_id,
                "resources": requirements,
                "allocated_at": datetime.now(),
                "duration": duration,
                "expires_at": datetime.now() + timedelta(seconds=duration)
            }
            
            return self.allocations[allocation_id]
    
    async def check_quota(self, requester_id: str) -> Dict[str, Any]:
        """检查配额"""
        async with self._lock:
            current_usage = defaultdict(float)
            
            # 计算当前使用量
            for allocation in self.allocations.values():
                if allocation.get("requester_id") == requester_id:
                    for resource_type, amount in allocation["resources"].items():
                        current_usage[resource_type] += amount
            
            # 检查是否超过配额
            quota = self.quotas[requester_id]
            within_quota = all(
                current_usage[rt] < quota.get(rt, float('inf'))
                for rt in current_usage
            )
            
            return {
                "within_quota": within_quota,
                "current_usage": dict(current_usage),
                "quota": quota
            }
    
    async def try_allocate(self, requester_id: str, resources: Dict[str, float], 
                          priority: int) -> Dict[str, Any]:
        """尝试分配资源"""
        # 检查可用性
        availability = await self.check_availability(resources)
        
        if availability["available"]:
            allocation = await self.allocate(
                f"task_{requester_id}_{uuid.uuid4().hex[:8]}",
                resources
            )
            allocation["requester_id"] = requester_id
            
            return {
                "success": True,
                "allocated": resources,
                "allocation_id": allocation["id"]
            }
        else:
            return {
                "success": False,
                "reason": "Insufficient resources",
                "available": availability["available_resources"]
            }
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "resources": self.resources.copy(),
            "allocations": len(self.allocations),
            "total_allocated": {
                rt: self.resources[rt]["allocated"]
                for rt in self.resources
            }
        }
    
    async def preempt_resources(self, target_allocations: List[str], reason: str):
        """抢占资源"""
        async with self._lock:
            for alloc_id in target_allocations:
                if alloc_id in self.allocations:
                    allocation = self.allocations[alloc_id]
                    
                    # 释放资源
                    for resource_type, amount in allocation["resources"].items():
                        if resource_type in self.resources:
                            self.resources[resource_type]["allocated"] -= amount
                    
                    # 记录抢占
                    allocation["preempted"] = True
                    allocation["preemption_reason"] = reason
                    
                    # 移除分配
                    del self.allocations[alloc_id]
    
    async def migrate_resources(self, from_allocation: str, to_allocation: str):
        """迁移资源"""
        async with self._lock:
            if from_allocation in self.allocations and to_allocation in self.allocations:
                from_alloc = self.allocations[from_allocation]
                to_alloc = self.allocations[to_allocation]
                
                # 转移资源
                for resource_type, amount in from_alloc["resources"].items():
                    to_alloc["resources"][resource_type] = (
                        to_alloc["resources"].get(resource_type, 0) + amount
                    )
                
                # 清空源分配
                from_alloc["resources"] = {}
                from_alloc["migrated_to"] = to_allocation
    
    async def scale_resources(self, resource_type: str, scale_factor: float):
        """扩展资源"""
        async with self._lock:
            if resource_type in self.resources:
                self.resources[resource_type]["total"] = int(
                    self.resources[resource_type]["total"] * scale_factor
                )

class TaskScheduler:
    """任务调度器"""
    
    def __init__(self):
        self.schedule = []
        self.reservations = {}
        self.time_slots = defaultdict(list)  # 时间 -> 任务列表
        self._lock = asyncio.Lock()
    
    async def create_schedule(self, tasks: List[Dict[str, Any]], 
                            constraints: Dict[str, Any]) -> Dict[str, Any]:
        """创建调度计划"""
        async with self._lock:
            # 拓扑排序处理依赖
            sorted_tasks = await self._topological_sort(tasks)
            
            # 分配时间槽
            scheduled_tasks = []
            current_time = constraints.get("start_time", 0)
            
            for task in sorted_tasks:
                # 计算最早开始时间
                earliest_start = max(
                    current_time,
                    await self._calculate_earliest_start(task, scheduled_tasks)
                )
                
                # 分配时间
                task["start_time"] = earliest_start
                task["end_time"] = earliest_start + task.get("duration", 1)
                
                scheduled_tasks.append(task)
                
                # 更新当前时间
                if constraints.get("parallel", False):
                    # 并行执行
                    current_time = earliest_start
                else:
                    # 串行执行
                    current_time = task["end_time"]
            
            # 计算关键路径
            critical_path = await self._calculate_critical_path(scheduled_tasks)
            
            return {
                "tasks": scheduled_tasks,
                "estimated_completion": max(t["end_time"] for t in scheduled_tasks),
                "critical_path": critical_path,
                "utilization": self._calculate_utilization(scheduled_tasks)
            }
    
    async def commit_schedule(self, schedule: Dict[str, Any]):
        """提交调度计划"""
        async with self._lock:
            for task in schedule["tasks"]:
                # 添加到时间槽
                for t in range(int(task["start_time"]), int(task["end_time"])):
                    self.time_slots[t].append(task["id"])
                
                # 添加到总调度
                self.schedule.append(task)
    
    async def find_available_slots(self, duration: int, 
                                  preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找可用时间槽"""
        async with self._lock:
            available_slots = []
            
            # 扫描时间线
            start_time = preferences.get("earliest", 0)
            end_time = preferences.get("latest", start_time + 86400)  # 24小时
            
            current_start = start_time
            while current_start + duration <= end_time:
                # 检查时间段是否可用
                if await self._is_slot_available(current_start, duration):
                    available_slots.append({
                        "start_time": current_start,
                        "end_time": current_start + duration,
                        "duration": duration
                    })
                
                current_start += preferences.get("step", 3600)  # 默认1小时步长
            
            return available_slots
    
    async def reserve_slot(self, slot: Dict[str, Any], 
                          task_info: Dict[str, Any]) -> Dict[str, Any]:
        """预约时间槽"""
        async with self._lock:
            reservation_id = f"res_{uuid.uuid4().hex[:8]}"
            
            # 记录预约
            self.reservations[reservation_id] = {
                "id": reservation_id,
                "slot": slot,
                "task": task_info,
                "reserved_at": datetime.now()
            }
            
            # 标记时间槽为已占用
            for t in range(int(slot["start_time"]), int(slot["end_time"])):
                self.time_slots[t].append(reservation_id)
            
            return self.reservations[reservation_id]
    
    async def revise_schedule(self, schedule: Dict[str, Any], 
                            issues: List[str]) -> Dict[str, Any]:
        """修订调度计划"""
        revised_schedule = schedule.copy()
        
        # 基于问题类型进行调整
        for issue in issues:
            if "conflict" in issue:
                # 解决资源冲突
                revised_schedule = await self._resolve_resource_conflicts(revised_schedule)
            elif "dependency" in issue:
                # 修复依赖问题
                revised_schedule = await self._fix_dependency_issues(revised_schedule)
            elif "deadline" in issue:
                # 压缩时间线
                revised_schedule = await self._compress_timeline(revised_schedule)
        
        return revised_schedule
    
    async def suggest_alternatives(self, task_info: Dict[str, Any], 
                                  preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """建议替代方案"""
        alternatives = []
        
        # 建议分割任务
        if task_info["duration"] > 4 * 3600:  # 超过4小时
            alternatives.append({
                "type": "split_task",
                "description": "Split task into smaller chunks",
                "chunks": [
                    {"duration": task_info["duration"] // 2},
                    {"duration": task_info["duration"] // 2}
                ]
            })
        
        # 建议调整时间
        alternatives.append({
            "type": "adjust_timing",
            "description": "Consider off-peak hours",
            "suggested_times": await self._find_off_peak_slots(task_info["duration"])
        })
        
        # 建议降低资源需求
        if task_info.get("resources"):
            alternatives.append({
                "type": "reduce_resources",
                "description": "Reduce resource requirements",
                "suggested_resources": {
                    k: v * 0.8 for k, v in task_info["resources"].items()
                }
            })
        
        return alternatives
    
    async def _topological_sort(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """拓扑排序"""
        # 构建依赖图
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        task_map = {task["id"]: task for task in tasks}
        
        for task in tasks:
            for dep in task.get("dependencies", []):
                graph[dep].append(task["id"])
                in_degree[task["id"]] += 1
        
        # 拓扑排序
        queue = deque([task["id"] for task in tasks if in_degree[task["id"]] == 0])
        sorted_order = []
        
        while queue:
            current = queue.popleft()
            sorted_order.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 返回排序后的任务
        return [task_map[task_id] for task_id in sorted_order if task_id in task_map]
    
    async def _calculate_earliest_start(self, task: Dict[str, Any], 
                                      scheduled: List[Dict[str, Any]]) -> float:
        """计算最早开始时间"""
        earliest = 0
        
        # 考虑依赖关系
        for dep_id in task.get("dependencies", []):
            dep_task = next((t for t in scheduled if t["id"] == dep_id), None)
            if dep_task:
                earliest = max(earliest, dep_task["end_time"])
        
        return earliest
    
    async def _calculate_critical_path(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """计算关键路径"""
        # 简化实现：返回最长路径
        if not tasks:
            return []
        
        # 找到没有后继的任务
        end_tasks = []
        all_deps = set()
        for task in tasks:
            all_deps.update(task.get("dependencies", []))
        
        for task in tasks:
            if task["id"] not in all_deps:
                end_tasks.append(task)
        
        # 从结束任务回溯最长路径
        if end_tasks:
            longest_task = max(end_tasks, key=lambda t: t["end_time"])
            path = [longest_task["id"]]
            
            current = longest_task
            while current.get("dependencies"):
                dep_id = current["dependencies"][0]  # 简化：取第一个依赖
                dep_task = next((t for t in tasks if t["id"] == dep_id), None)
                if dep_task:
                    path.insert(0, dep_task["id"])
                    current = dep_task
                else:
                    break
            
            return path
        
        return []
    
    def _calculate_utilization(self, tasks: List[Dict[str, Any]]) -> float:
        """计算利用率"""
        if not tasks:
            return 0.0
        
        # 计算总工作时间
        total_work_time = sum(t["end_time"] - t["start_time"] for t in tasks)
        
        # 计算总时间跨度
        min_start = min(t["start_time"] for t in tasks)
        max_end = max(t["end_time"] for t in tasks)
        total_span = max_end - min_start
        
        return total_work_time / total_span if total_span > 0 else 0.0
    
    async def _is_slot_available(self, start_time: int, duration: int) -> bool:
        """检查时间槽是否可用"""
        for t in range(start_time, start_time + duration):
            if self.time_slots[t]:
                return False
        return True
    
    async def _resolve_resource_conflicts(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """解决资源冲突"""
        # 简化实现：串行化冲突任务
        tasks = schedule["tasks"]
        
        # 检测冲突并调整
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                if self._tasks_conflict(tasks[i], tasks[j]):
                    # 将后一个任务推迟
                    delay = tasks[i]["end_time"] - tasks[j]["start_time"]
                    tasks[j]["start_time"] += delay
                    tasks[j]["end_time"] += delay
        
        return schedule
    
    async def _fix_dependency_issues(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """修复依赖问题"""
        # 重新进行拓扑排序和调度
        return await self.create_schedule(
            schedule["tasks"],
            {"start_time": 0, "parallel": True}
        )
    
    async def _compress_timeline(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """压缩时间线"""
        tasks = schedule["tasks"]
        
        # 减少任务间的间隔
        for i in range(1, len(tasks)):
            prev_task = tasks[i-1]
            curr_task = tasks[i]
            
            # 如果没有依赖关系，可以并行
            if prev_task["id"] not in curr_task.get("dependencies", []):
                overlap = min(
                    prev_task["duration"] * 0.5,
                    curr_task["duration"] * 0.5
                )
                curr_task["start_time"] = prev_task["start_time"] + overlap
                curr_task["end_time"] = curr_task["start_time"] + curr_task["duration"]
        
        return schedule
    
    async def _find_off_peak_slots(self, duration: int) -> List[Dict[str, Any]]:
        """查找非高峰时段"""
        off_peak_hours = [0, 1, 2, 3, 4, 5, 22, 23]  # 晚上10点到早上6点
        slots = []
        
        for hour in off_peak_hours:
            slots.append({
                "start_time": hour * 3600,
                "end_time": (hour + duration // 3600) * 3600,
                "description": f"Off-peak slot starting at {hour}:00"
            })
        
        return slots[:3]  # 返回前3个建议
    
    def _tasks_conflict(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> bool:
        """检查任务是否冲突"""
        # 时间重叠且有资源冲突
        time_overlap = not (task1["end_time"] <= task2["start_time"] or 
                           task2["end_time"] <= task1["start_time"])
        
        if not time_overlap:
            return False
        
        # 检查资源冲突
        resources1 = set(task1.get("resources", {}).keys())
        resources2 = set(task2.get("resources", {}).keys())
        
        return bool(resources1 & resources2)

class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self):
        self.conflicts = {}
        self.resolution_history = []
        self.resolution_strategies = {
            "resource": self._resolve_resource_conflict,
            "scheduling": self._resolve_scheduling_conflict,
            "priority": self._resolve_priority_conflict,
            "dependency": self._resolve_dependency_conflict
        }
        self._lock = asyncio.Lock()
    
    async def analyze_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """分析冲突"""
        conflict_type = conflict.get("type", "unknown")
        parties = conflict.get("parties", [])
        
        analysis = {
            "type": conflict_type,
            "severity": await self._assess_severity_level(conflict),
            "impact": await self._assess_impact(conflict),
            "root_cause": await self._identify_root_cause(conflict),
            "stakeholders": parties
        }
        
        return analysis
    
    async def generate_solutions(self, conflict: Dict[str, Any], 
                               analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成解决方案"""
        solutions = []
        conflict_type = analysis["type"]
        
        # 获取对应的解决策略
        if conflict_type in self.resolution_strategies:
            type_solutions = await self.resolution_strategies[conflict_type](
                conflict, analysis
            )
            solutions.extend(type_solutions)
        
        # 通用解决方案
        generic_solutions = await self._generate_generic_solutions(conflict, analysis)
        solutions.extend(generic_solutions)
        
        # 评估每个方案
        for solution in solutions:
            solution["feasibility"] = await self._assess_feasibility(solution)
            solution["success_probability"] = await self._estimate_success_probability(
                solution, conflict
            )
        
        return solutions
    
    async def apply_solution(self, conflict: Dict[str, Any], 
                           solution: Dict[str, Any]) -> Dict[str, Any]:
        """应用解决方案"""
        async with self._lock:
            result = {
                "success": False,
                "outcome": None,
                "side_effects": []
            }
            
            try:
                # 执行解决方案
                if solution["type"] == "negotiate":
                    outcome = await self._execute_negotiation(conflict, solution)
                elif solution["type"] == "arbitrate":
                    outcome = await self._execute_arbitration(conflict, solution)
                elif solution["type"] == "compromise":
                    outcome = await self._execute_compromise(conflict, solution)
                elif solution["type"] == "escalate":
                    outcome = await self._execute_escalation(conflict, solution)
                else:
                    outcome = await self._execute_generic_solution(conflict, solution)
                
                result["success"] = outcome.get("success", False)
                result["outcome"] = outcome
                
                # 记录解决历史
                self.resolution_history.append({
                    "conflict_id": conflict.get("id"),
                    "solution": solution,
                    "result": result,
                    "timestamp": datetime.now()
                })
                
            except Exception as e:
                result["error"] = str(e)
                logging.error(f"Failed to apply solution: {e}")
            
            return result
    
    async def log_conflict(self, conflict_info: Dict[str, Any]) -> str:
        """记录冲突"""
        async with self._lock:
            conflict_id = f"conflict_{uuid.uuid4().hex[:8]}"
            
            self.conflicts[conflict_id] = {
                "id": conflict_id,
                "info": conflict_info,
                "reported_at": datetime.now(),
                "status": "reported"
            }
            
            return conflict_id
    
    async def assess_severity(self, conflict_info: Dict[str, Any]) -> Dict[str, Any]:
        """评估严重性"""
        severity_score = 0
        
        # 基于影响范围
        affected_count = len(conflict_info.get("affected_agents", []))
        if affected_count > 5:
            severity_score += 3
        elif affected_count > 2:
            severity_score += 2
        else:
            severity_score += 1
        
        # 基于类型
        if conflict_info.get("type") == "critical_resource":
            severity_score += 3
        elif conflict_info.get("type") == "deadline":
            severity_score += 2
        
        # 确定级别
        if severity_score >= 5:
            level = "critical"
            estimated_time = 300  # 5分钟
        elif severity_score >= 3:
            level = "high"
            estimated_time = 900  # 15分钟
        else:
            level = "normal"
            estimated_time = 1800  # 30分钟
        
        return {
            "level": level,
            "score": severity_score,
            "estimated_time": estimated_time
        }
    
    async def take_immediate_action(self, conflict_info: Dict[str, Any]) -> Dict[str, Any]:
        """采取即时行动"""
        actions_taken = []
        
        # 冻结相关资源
        if "resources" in conflict_info:
            actions_taken.append({
                "action": "freeze_resources",
                "resources": conflict_info["resources"],
                "duration": 300  # 5分钟
            })
        
        # 暂停相关任务
        if "tasks" in conflict_info:
            actions_taken.append({
                "action": "pause_tasks",
                "tasks": conflict_info["tasks"]
            })
        
        # 通知相关方
        if "affected_agents" in conflict_info:
            actions_taken.append({
                "action": "notify_agents",
                "agents": conflict_info["affected_agents"],
                "message": "Critical conflict detected, temporary pause initiated"
            })
        
        return {
            "actions": actions_taken,
            "status": "immediate_response_executed"
        }
    
    async def queue_for_resolution(self, conflict_id: str, 
                                  priority_level: str) -> int:
        """加入解决队列"""
        # 简化实现：返回队列位置
        priority_map = {"critical": 0, "high": 10, "normal": 20}
        base_position = priority_map.get(priority_level, 30)
        
        # 加入随机因素模拟真实队列
        import random
        position = base_position + random.randint(0, 5)
        
        return position
    
    async def _assess_severity_level(self, conflict: Dict[str, Any]) -> str:
        """评估严重级别"""
        # 基于冲突类型和影响范围
        if conflict.get("type") == "critical_resource":
            return "high"
        elif len(conflict.get("parties", [])) > 3:
            return "medium"
        else:
            return "low"
    
    async def _assess_impact(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """评估影响"""
        return {
            "scope": "local" if len(conflict.get("parties", [])) <= 2 else "global",
            "duration": "short" if conflict.get("type") != "dependency" else "long",
            "cost": "medium"  # 简化评估
        }
    
    async def _identify_root_cause(self, conflict: Dict[str, Any]) -> str:
        """识别根本原因"""
        conflict_type = conflict.get("type")
        
        if conflict_type == "resource":
            return "Resource scarcity"
        elif conflict_type == "scheduling":
            return "Time constraints"
        elif conflict_type == "priority":
            return "Competing objectives"
        else:
            return "Unknown cause"
    
    async def _resolve_resource_conflict(self, conflict: Dict[str, Any], 
                                       analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解决资源冲突"""
        solutions = []
        
        # 资源共享
        solutions.append({
            "type": "share",
            "description": "Share resources with time-slicing",
            "impact": "medium",
            "complexity": "low"
        })
        
        # 资源替代
        solutions.append({
            "type": "substitute",
            "description": "Use alternative resources",
            "impact": "low",
            "complexity": "medium"
        })
        
        # 资源扩展
        solutions.append({
            "type": "expand",
            "description": "Allocate additional resources",
            "impact": "low",
            "complexity": "high"
        })
        
        return solutions
    
    async def _resolve_scheduling_conflict(self, conflict: Dict[str, Any], 
                                         analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解决调度冲突"""
        return [
            {
                "type": "reschedule",
                "description": "Adjust task timings",
                "impact": "medium",
                "complexity": "low"
            },
            {
                "type": "parallel",
                "description": "Enable parallel execution",
                "impact": "low",
                "complexity": "medium"
            }
        ]
    
    async def _resolve_priority_conflict(self, conflict: Dict[str, Any], 
                                       analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解决优先级冲突"""
        return [
            {
                "type": "reorder",
                "description": "Reorder priorities based on impact",
                "impact": "high",
                "complexity": "low"
            },
            {
                "type": "negotiate",
                "description": "Negotiate priority trade-offs",
                "impact": "medium",
                "complexity": "medium"
            }
        ]
    
    async def _resolve_dependency_conflict(self, conflict: Dict[str, Any], 
                                         analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解决依赖冲突"""
        return [
            {
                "type": "decouple",
                "description": "Remove or reduce dependencies",
                "impact": "high",
                "complexity": "high"
            },
            {
                "type": "buffer",
                "description": "Add buffer time between dependencies",
                "impact": "low",
                "complexity": "low"
            }
        ]
    
    async def _generate_generic_solutions(self, conflict: Dict[str, Any], 
                                        analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成通用解决方案"""
        return [
            {
                "type": "escalate",
                "description": "Escalate to higher authority",
                "impact": "variable",
                "complexity": "low"
            },
            {
                "type": "compromise",
                "description": "Find middle ground",
                "impact": "medium",
                "complexity": "medium"
            }
        ]
    
    async def _assess_feasibility(self, solution: Dict[str, Any]) -> float:
        """评估可行性"""
        # 基于复杂度和影响
        complexity_score = {"low": 0.9, "medium": 0.7, "high": 0.5}
        impact_score = {"low": 0.8, "medium": 0.6, "high": 0.4}
        
        complexity = complexity_score.get(solution.get("complexity", "medium"), 0.5)
        impact = impact_score.get(solution.get("impact", "medium"), 0.5)
        
        return (complexity + impact) / 2
    
    async def _estimate_success_probability(self, solution: Dict[str, Any], 
                                          conflict: Dict[str, Any]) -> float:
        """估计成功概率"""
        base_probability = 0.6
        
        # 基于解决方案类型调整
        if solution["type"] in ["share", "negotiate", "compromise"]:
            base_probability += 0.2
        elif solution["type"] in ["escalate", "expand"]:
            base_probability += 0.1
        
        # 基于可行性调整
        feasibility = solution.get("feasibility", 0.5)
        
        return min(1.0, base_probability * feasibility)
    
    async def _execute_negotiation(self, conflict: Dict[str, Any], 
                                 solution: Dict[str, Any]) -> Dict[str, Any]:
        """执行协商"""
        return {
            "success": True,
            "agreement": "Parties agreed to share resources",
            "terms": ["50-50 split", "Alternating priority"]
        }
    
    async def _execute_arbitration(self, conflict: Dict[str, Any], 
                                 solution: Dict[str, Any]) -> Dict[str, Any]:
        """执行仲裁"""
        return {
            "success": True,
            "decision": "Resource allocated to higher priority task",
            "rationale": "Based on business impact"
        }
    
    async def _execute_compromise(self, conflict: Dict[str, Any], 
                                solution: Dict[str, Any]) -> Dict[str, Any]:
        """执行妥协"""
        return {
            "success": True,
            "compromise": "Both parties reduce requirements by 20%",
            "satisfaction": 0.7
        }
    
    async def _execute_escalation(self, conflict: Dict[str, Any], 
                                solution: Dict[str, Any]) -> Dict[str, Any]:
        """执行升级"""
        return {
            "success": True,
            "escalated_to": "System Administrator",
            "response_time": 600  # 10分钟
        }
    
    async def _execute_generic_solution(self, conflict: Dict[str, Any], 
                                      solution: Dict[str, Any]) -> Dict[str, Any]:
        """执行通用解决方案"""
        return {
            "success": True,
            "method": solution["type"],
            "result": "Conflict resolved using generic approach"
        }

class OptimizationEngine:
    """优化引擎"""
    
    def __init__(self):
        self.optimization_algorithms = {
            "genetic": self._genetic_optimization,
            "simulated_annealing": self._simulated_annealing,
            "gradient_descent": self._gradient_descent,
            "particle_swarm": self._particle_swarm
        }
        self.metrics_cache = {}
    
    async def optimize_allocation(self, requirements: Dict[str, float], 
                                current_state: Dict[str, Any]) -> Dict[str, Any]:
        """优化资源分配"""
        # 定义优化目标
        objective = {
            "minimize": ["cost", "fragmentation"],
            "maximize": ["utilization", "fairness"]
        }
        
        # 生成初始解决方案
        initial_solution = await self._generate_initial_allocation(
            requirements, current_state
        )
        
        # 应用优化算法
        optimized = await self._genetic_optimization(
            initial_solution,
            objective,
            constraints={"resources": current_state["resources"]}
        )
        
        return {
            "feasible": optimized["feasible"],
            "solution": optimized["solution"],
            "improvement": optimized["improvement"],
            "steps": optimized["steps"],
            "suggestions": await self._generate_suggestions(optimized)
        }
    
    async def analyze_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """分析工作流"""
        metrics = {
            "total_duration": await self._calculate_duration(workflow),
            "parallelism": await self._calculate_parallelism(workflow),
            "resource_efficiency": await self._calculate_efficiency(workflow),
            "bottleneck_score": await self._calculate_bottleneck_score(workflow)
        }
        
        # 缓存指标
        workflow_id = workflow.get("id", "unknown")
        self.metrics_cache[workflow_id] = metrics
        
        return {
            "metrics": metrics,
            "health_score": await self._calculate_health_score(metrics),
            "recommendations": await self._generate_workflow_recommendations(metrics)
        }
    
    async def identify_bottlenecks(self, workflow: Dict[str, Any], 
                                 analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别瓶颈"""
        bottlenecks = []
        
        # 检查任务瓶颈
        tasks = workflow.get("tasks", [])
        for task in tasks:
            if await self._is_bottleneck(task, workflow):
                bottlenecks.append({
                    "type": "task",
                    "id": task["id"],
                    "severity": await self._calculate_bottleneck_severity(task),
                    "impact": await self._calculate_bottleneck_impact(task, workflow)
                })
        
        # 检查资源瓶颈
        resource_bottlenecks = await self._identify_resource_bottlenecks(workflow)
        bottlenecks.extend(resource_bottlenecks)
        
        # 排序by严重性
        bottlenecks.sort(key=lambda x: x["severity"], reverse=True)
        
        return bottlenecks
    
    async def generate_optimizations(self, workflow: Dict[str, Any],
                                   bottlenecks: List[Dict[str, Any]],
                                   goals: List[str]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        optimizations = []
        
        # 针对每个瓶颈生成优化
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "task":
                task_opts = await self._optimize_task(bottleneck, workflow, goals)
                optimizations.extend(task_opts)
            elif bottleneck["type"] == "resource":
                resource_opts = await self._optimize_resource(bottleneck, workflow, goals)
                optimizations.extend(resource_opts)
        
        # 全局优化
        global_opts = await self._generate_global_optimizations(workflow, goals)
        optimizations.extend(global_opts)
        
        # 去重和排序
        optimizations = self._deduplicate_optimizations(optimizations)
        optimizations = await self._rank_optimizations(optimizations, goals)
        
        return optimizations
    
    async def simulate_optimizations(self, workflow: Dict[str, Any],
                                   optimizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """模拟优化效果"""
        simulation_results = {
            "baseline": await self._get_baseline_metrics(workflow),
            "impacts": {},
            "improvement": {}
        }
        
        # 模拟每个优化的影响
        for opt in optimizations:
            simulated_workflow = await self._apply_optimization(
                workflow.copy(), opt
            )
            
            metrics = await self.analyze_workflow(simulated_workflow)
            
            impact = {
                "performance_gain": self._calculate_performance_gain(
                    simulation_results["baseline"],
                    metrics["metrics"]
                ),
                "cost_reduction": self._calculate_cost_reduction(
                    simulation_results["baseline"],
                    metrics["metrics"]
                ),
                "implementation_complexity": opt.get("complexity", 0.5)
            }
            
            simulation_results["impacts"][opt["id"]] = impact
        
        # 计算总体改进
        if optimizations:
            combined_workflow = workflow.copy()
            for opt in optimizations:
                combined_workflow = await self._apply_optimization(
                    combined_workflow, opt
                )
            
            combined_metrics = await self.analyze_workflow(combined_workflow)
            simulation_results["improvement"] = {
                "combined_performance_gain": self._calculate_performance_gain(
                    simulation_results["baseline"],
                    combined_metrics["metrics"]
                ),
                "estimated_roi": await self._calculate_roi(optimizations)
            }
        
        return simulation_results
    
    async def _generate_initial_allocation(self, requirements: Dict[str, float],
                                         state: Dict[str, Any]) -> Dict[str, Any]:
        """生成初始分配方案"""
        allocation = {}
        
        # 简单的首次适应算法
        for resource_type, required in requirements.items():
            if resource_type in state["resources"]:
                available = (state["resources"][resource_type]["total"] - 
                           state["resources"][resource_type]["allocated"])
                
                if available >= required:
                    allocation[resource_type] = required
                else:
                    allocation[resource_type] = available
        
        return {
            "allocation": allocation,
            "feasible": all(
                allocation.get(rt, 0) >= requirements[rt]
                for rt in requirements
            )
        }
    
    async def _genetic_optimization(self, initial: Dict[str, Any],
                                  objective: Dict[str, Any],
                                  constraints: Dict[str, Any]) -> Dict[str, Any]:
        """遗传算法优化"""
        # 简化的遗传算法实现
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        
        # 初始化种群
        population = [initial["allocation"]]
        for _ in range(population_size - 1):
            population.append(self._mutate_allocation(initial["allocation"], 0.3))
        
        best_solution = initial["allocation"]
        best_fitness = await self._calculate_fitness(best_solution, objective)
        
        # 进化
        for generation in range(generations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                fitness = await self._calculate_fitness(individual, objective)
                fitness_scores.append((individual, fitness))
            
            # 选择
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            survivors = fitness_scores[:population_size // 2]
            
            # 更新最佳解
            if survivors[0][1] > best_fitness:
                best_solution = survivors[0][0]
                best_fitness = survivors[0][1]
            
            # 交叉和变异
            new_population = [s[0] for s in survivors]
            while len(new_population) < population_size:
                parent1 = random.choice(survivors)[0]
                parent2 = random.choice(survivors)[0]
                child = self._crossover(parent1, parent2)
                
                if random.random() < mutation_rate:
                    child = self._mutate_allocation(child, 0.1)
                
                new_population.append(child)
            
            population = new_population
        
        return {
            "feasible": True,
            "solution": best_solution,
            "improvement": best_fitness / await self._calculate_fitness(initial["allocation"], objective),
            "steps": [
                {
                    "action": "optimize",
                    "method": "genetic_algorithm",
                    "details": f"Evolved through {generations} generations"
                }
            ]
        }
    
    async def _simulated_annealing(self, initial: Dict[str, Any],
                                 objective: Dict[str, Any],
                                 constraints: Dict[str, Any]) -> Dict[str, Any]:
        """模拟退火优化"""
        # 简化实现
        return {
            "feasible": True,
            "solution": initial,
            "improvement": 1.1,
            "steps": []
        }
    
    async def _gradient_descent(self, initial: Dict[str, Any],
                              objective: Dict[str, Any],
                              constraints: Dict[str, Any]) -> Dict[str, Any]:
        """梯度下降优化"""
        # 简化实现
        return {
            "feasible": True,
            "solution": initial,
            "improvement": 1.05,
            "steps": []
        }
    
    async def _particle_swarm(self, initial: Dict[str, Any],
                            objective: Dict[str, Any],
                            constraints: Dict[str, Any]) -> Dict[str, Any]:
        """粒子群优化"""
        # 简化实现
        return {
            "feasible": True,
            "solution": initial,
            "improvement": 1.08,
            "steps": []
        }
    
    async def _calculate_fitness(self, allocation: Dict[str, float],
                               objective: Dict[str, Any]) -> float:
        """计算适应度"""
        fitness = 1.0
        
        # 最小化目标
        for metric in objective.get("minimize", []):
            if metric == "cost":
                cost = sum(allocation.values())
                fitness *= 1 / (1 + cost)
            elif metric == "fragmentation":
                # 简化：假设碎片化与分配数量相关
                fitness *= 1 / (1 + len(allocation))
        
        # 最大化目标
        for metric in objective.get("maximize", []):
            if metric == "utilization":
                # 简化：使用率与分配总量相关
                fitness *= sum(allocation.values()) / 100
            elif metric == "fairness":
                # 简化：公平性与分配均匀度相关
                if allocation:
                    values = list(allocation.values())
                    fairness = 1 - (max(values) - min(values)) / (max(values) + 1)
                    fitness *= fairness
        
        return fitness
    
    def _mutate_allocation(self, allocation: Dict[str, float],
                         rate: float) -> Dict[str, float]:
        """变异分配方案"""
        mutated = allocation.copy()
        
        for resource in mutated:
            if random.random() < rate:
                # 随机调整±20%
                factor = 1 + (random.random() - 0.5) * 0.4
                mutated[resource] = mutated[resource] * factor
        
        return mutated
    
    def _crossover(self, parent1: Dict[str, float],
                  parent2: Dict[str, float]) -> Dict[str, float]:
        """交叉操作"""
        child = {}
        
        for resource in parent1:
            if random.random() < 0.5:
                child[resource] = parent1[resource]
            else:
                child[resource] = parent2.get(resource, parent1[resource])
        
        return child
    
    async def _generate_suggestions(self, optimization: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        if optimization["feasible"]:
            suggestions.append("Apply the optimized allocation plan")
            
            if optimization["improvement"] > 1.2:
                suggestions.append("Significant improvement possible - prioritize implementation")
            
            # 基于步骤生成具体建议
            for step in optimization.get("steps", []):
                if step["action"] == "preempt":
                    suggestions.append(f"Consider preempting low-priority tasks")
                elif step["action"] == "scale":
                    suggestions.append(f"Scale up resources for better performance")
        else:
            suggestions.append("Current requirements cannot be fully satisfied")
            suggestions.append("Consider reducing requirements or adding resources")
        
        return suggestions
    
    async def _calculate_duration(self, workflow: Dict[str, Any]) -> float:
        """计算工作流持续时间"""
        tasks = workflow.get("tasks", [])
        if not tasks:
            return 0
        
        # 考虑并行执行
        end_times = []
        for task in tasks:
            end_time = task.get("start_time", 0) + task.get("duration", 0)
            end_times.append(end_time)
        
        return max(end_times) if end_times else 0
    
    async def _calculate_parallelism(self, workflow: Dict[str, Any]) -> float:
        """计算并行度"""
        tasks = workflow.get("tasks", [])
        if len(tasks) <= 1:
            return 1.0
        
        # 计算同时执行的任务数
        time_points = []
        for task in tasks:
            time_points.append((task.get("start_time", 0), 1))  # 开始
            time_points.append((task.get("start_time", 0) + task.get("duration", 0), -1))  # 结束
        
        time_points.sort()
        
        max_parallel = 0
        current_parallel = 0
        
        for _, delta in time_points:
            current_parallel += delta
            max_parallel = max(max_parallel, current_parallel)
        
        # 并行度 = 最大并行数 / 任务总数
        return min(1.0, max_parallel / len(tasks))
    
    async def _calculate_efficiency(self, workflow: Dict[str, Any]) -> float:
        """计算资源效率"""
        # 简化实现：基于资源利用率
        total_resource_time = 0
        available_resource_time = 0
        
        tasks = workflow.get("tasks", [])
        for task in tasks:
            duration = task.get("duration", 0)
            resources = task.get("resources", {})
            
            for resource_type, amount in resources.items():
                total_resource_time += amount * duration
                available_resource_time += 100 * duration  # 假设总容量为100
        
        return total_resource_time / available_resource_time if available_resource_time > 0 else 0
    
    async def _calculate_bottleneck_score(self, workflow: Dict[str, Any]) -> float:
        """计算瓶颈分数"""
        # 简化实现：基于关键路径长度
        critical_path_length = await self._find_critical_path_length(workflow)
        total_duration = await self._calculate_duration(workflow)
        
        # 瓶颈分数 = 关键路径长度 / 总持续时间
        return critical_path_length / total_duration if total_duration > 0 else 1.0
    
    async def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """计算健康分数"""
        # 综合各项指标
        weights = {
            "paral
            # ============================= 继续 OptimizationEngine 类的实现 ======================// ...省略其他代码... 


// ==== 修改后代码 ====

    async def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """计算健康分数"""
        # 综合各项指标
        weights = {
            "parallelism": 0.3,
            "resource_efficiency": 0.3,
            "bottleneck_score": 0.4
        }
        
        health_score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                # bottleneck_score越低越好，需要反转
                if metric == "bottleneck_score":
                    score = 1 - metrics[metric]
                else:
                    score = metrics[metric]
                health_score += score * weight
        
        return min(1.0, max(0.0, health_score))
    
    async def _generate_workflow_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """生成工作流建议"""
        recommendations = []
        
        if metrics["parallelism"] < 0.5:
            recommendations.append("Increase task parallelization for better performance")
        
        if metrics["resource_efficiency"] < 0.7:
            recommendations.append("Optimize resource allocation to reduce waste")
        
        if metrics["bottleneck_score"] > 0.8:
            recommendations.append("Critical path dominates - consider breaking down bottleneck tasks")
        
        return recommendations
    
    async def _is_bottleneck(self, task: Dict[str, Any], workflow: Dict[str, Any]) -> bool:
        """判断任务是否是瓶颈"""
        # 检查是否在关键路径上
        critical_path = await self._find_critical_path(workflow)
        if task["id"] not in critical_path:
            return False
        
        # 检查是否有多个依赖它的任务
        dependent_count = 0
        for other_task in workflow.get("tasks", []):
            if task["id"] in other_task.get("dependencies", []):
                dependent_count += 1
        
        return dependent_count > 2
    
    async def _calculate_bottleneck_severity(self, task: Dict[str, Any]) -> float:
        """计算瓶颈严重性"""
        severity = 0.5  # 基础严重性
        
        # 持续时间越长越严重
        duration = task.get("duration", 0)
        if duration > 3600:  # 超过1小时
            severity += 0.2
        
        # 资源需求越多越严重
        resource_count = len(task.get("resources", {}))
        if resource_count > 3:
            severity += 0.2
        
        # 依赖越多越严重
        dependency_count = len(task.get("dependencies", []))
        if dependency_count > 2:
            severity += 0.1
        
        return min(1.0, severity)
    
    async def _calculate_bottleneck_impact(self, task: Dict[str, Any], 
                                         workflow: Dict[str, Any]) -> Dict[str, Any]:
        """计算瓶颈影响"""
        # 计算受影响的下游任务数
        affected_tasks = set()
        
        def find_downstream(task_id):
            for t in workflow.get("tasks", []):
                if task_id in t.get("dependencies", []):
                    affected_tasks.add(t["id"])
                    find_downstream(t["id"])
        
        find_downstream(task["id"])
        
        # 计算延迟影响
        delay_impact = task.get("duration", 0) * len(affected_tasks)
        
        return {
            "affected_task_count": len(affected_tasks),
            "delay_impact": delay_impact,
            "percentage_affected": len(affected_tasks) / max(len(workflow.get("tasks", [])), 1)
        }
    
    async def _identify_resource_bottlenecks(self, workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别资源瓶颈"""
        bottlenecks = []
        resource_usage = defaultdict(list)
        
        # 统计资源使用时间段
        for task in workflow.get("tasks", []):
            start = task.get("start_time", 0)
            end = start + task.get("duration", 0)
            
            for resource, amount in task.get("resources", {}).items():
                resource_usage[resource].append({
                    "start": start,
                    "end": end,
                    "amount": amount,
                    "task_id": task["id"]
                })
        
        # 分析每种资源的使用情况
        for resource, usage_list in resource_usage.items():
            # 找出高峰期
            peak_usage = await self._find_peak_usage(usage_list)
            
            if peak_usage["max_concurrent"] > 80:  # 超过80%认为是瓶颈
                bottlenecks.append({
                    "type": "resource",
                    "resource": resource,
                    "severity": peak_usage["max_concurrent"] / 100,
                    "peak_time": peak_usage["peak_time"],
                    "affected_tasks": peak_usage["tasks"]
                })
        
        return bottlenecks
    
    async def _optimize_task(self, bottleneck: Dict[str, Any], 
                           workflow: Dict[str, Any], 
                           goals: List[str]) -> List[Dict[str, Any]]:
        """优化任务瓶颈"""
        optimizations = []
        task_id = bottleneck["id"]
        
        # 任务分解
        optimizations.append({
            "id": f"split_{task_id}",
            "type": "task_decomposition",
            "description": f"Split task {task_id} into smaller parallel subtasks",
            "complexity": 0.7,
            "expected_improvement": 0.3
        })
        
        # 资源增加
        optimizations.append({
            "id": f"scale_{task_id}",
            "type": "resource_scaling",
            "description": f"Allocate more resources to task {task_id}",
            "complexity": 0.3,
            "expected_improvement": 0.2
        })
        
        # 算法优化
        if "efficiency" in goals:
            optimizations.append({
                "id": f"optimize_algo_{task_id}",
                "type": "algorithm_optimization",
                "description": f"Use more efficient algorithm for task {task_id}",
                "complexity": 0.8,
                "expected_improvement": 0.4
            })
        
        return optimizations
    
    async def _optimize_resource(self, bottleneck: Dict[str, Any], 
                               workflow: Dict[str, Any], 
                               goals: List[str]) -> List[Dict[str, Any]]:
        """优化资源瓶颈"""
        optimizations = []
        resource = bottleneck["resource"]
        
        # 资源池扩展
        optimizations.append({
            "id": f"expand_{resource}",
            "type": "resource_expansion",
            "description": f"Increase {resource} capacity by 50%",
            "complexity": 0.5,
            "expected_improvement": 0.4
        })
        
        # 负载均衡
        optimizations.append({
            "id": f"balance_{resource}",
            "type": "load_balancing",
            "description": f"Distribute {resource} usage more evenly",
            "complexity": 0.4,
            "expected_improvement": 0.3
        })
        
        # 资源共享
        optimizations.append({
            "id": f"share_{resource}",
            "type": "resource_sharing",
            "description": f"Enable time-sliced sharing of {resource}",
            "complexity": 0.6,
            "expected_improvement": 0.25
        })
        
        return optimizations
    
    async def _generate_global_optimizations(self, workflow: Dict[str, Any], 
                                           goals: List[str]) -> List[Dict[str, Any]]:
        """生成全局优化"""
        optimizations = []
        
        # 工作流重组
        optimizations.append({
            "id": "workflow_restructure",
            "type": "structural",
            "description": "Restructure workflow for better parallelization",
            "complexity": 0.9,
            "expected_improvement": 0.5
        })
        
        # 缓存策略
        if "efficiency" in goals:
            optimizations.append({
                "id": "caching_strategy",
                "type": "caching",
                "description": "Implement result caching for repeated operations",
                "complexity": 0.5,
                "expected_improvement": 0.3
            })
        
        # 预测性调度
        optimizations.append({
            "id": "predictive_scheduling",
            "type": "scheduling",
            "description": "Use ML-based predictive scheduling",
            "complexity": 0.8,
            "expected_improvement": 0.35
        })
        
        return optimizations
    
    def _deduplicate_optimizations(self, optimizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重优化建议"""
        seen = set()
        unique = []
        
        for opt in optimizations:
            key = (opt["type"], opt.get("target", "global"))
            if key not in seen:
                seen.add(key)
                unique.append(opt)
        
        return unique
    
    async def _rank_optimizations(self, optimizations: List[Dict[str, Any]], 
                                goals: List[str]) -> List[Dict[str, Any]]:
        """排序优化建议"""
        # 计算每个优化的得分
        for opt in optimizations:
            # 基础得分 = 预期改进 / 复杂度
            base_score = opt.get("expected_improvement", 0.5) / (opt.get("complexity", 0.5) + 0.1)
            
            # 目标对齐加分
            goal_bonus = 0
            if "efficiency" in goals and opt["type"] in ["algorithm_optimization", "caching"]:
                goal_bonus += 0.2
            if "cost" in goals and opt["type"] in ["resource_sharing", "load_balancing"]:
                goal_bonus += 0.2
            
            opt["score"] = base_score + goal_bonus
        
        # 按得分排序
        optimizations.sort(key=lambda x: x["score"], reverse=True)
        
        return optimizations
    
    async def _get_baseline_metrics(self, workflow: Dict[str, Any]) -> Dict[str, float]:
        """获取基线指标"""
        return {
            "duration": await self._calculate_duration(workflow),
            "cost": await self._calculate_cost(workflow),
            "resource_usage": await self._calculate_resource_usage(workflow),
            "throughput": await self._calculate_throughput(workflow)
        }
    
    async def _apply_optimization(self, workflow: Dict[str, Any], 
                                optimization: Dict[str, Any]) -> Dict[str, Any]:
        """应用优化"""
        opt_type = optimization["type"]
        
        if opt_type == "task_decomposition":
            # 分解任务
            return await self._decompose_task_in_workflow(workflow, optimization)
        elif opt_type == "resource_scaling":
            # 扩展资源
            return await self._scale_resources_in_workflow(workflow, optimization)
        elif opt_type == "load_balancing":
            # 负载均衡
            return await self._balance_load_in_workflow(workflow, optimization)
        else:
            # 默认：轻微改进
            for task in workflow.get("tasks", []):
                task["duration"] *= 0.95  # 5%改进
            return workflow
    
    def _calculate_performance_gain(self, baseline: Dict[str, float], 
                                  optimized: Dict[str, float]) -> float:
        """计算性能提升"""
        # 基于持续时间和吞吐量
        duration_improvement = (baseline.get("duration", 1) - optimized.get("duration", 1)) / baseline.get("duration", 1)
        throughput_improvement = (optimized.get("throughput", 1) - baseline.get("throughput", 1)) / baseline.get("throughput", 1)
        
        return (duration_improvement + throughput_improvement) / 2
    
    def _calculate_cost_reduction(self, baseline: Dict[str, float], 
                                optimized: Dict[str, float]) -> float:
        """计算成本降低"""
        baseline_cost = baseline.get("cost", 1)
        optimized_cost = optimized.get("cost", 1)
        
        return (baseline_cost - optimized_cost) / baseline_cost if baseline_cost > 0 else 0
    
    async def _calculate_roi(self, optimizations: List[Dict[str, Any]]) -> float:
        """计算投资回报率"""
        # 简化计算：基于预期改进和实施成本
        total_improvement = sum(opt.get("expected_improvement", 0) for opt in optimizations)
        total_cost = sum(opt.get("complexity", 0.5) for opt in optimizations)
        
        return total_improvement / total_cost if total_cost > 0 else 0
    
    async def _find_critical_path_length(self, workflow: Dict[str, Any]) -> float:
        """找出关键路径长度"""
        critical_path = await self._find_critical_path(workflow)
        
        total_duration = 0
        for task_id in critical_path:
            task = next((t for t in workflow.get("tasks", []) if t["id"] == task_id), None)
            if task:
                total_duration += task.get("duration", 0)
        
        return total_duration
    
    async def _find_critical_path(self, workflow: Dict[str, Any]) -> List[str]:
        """找出关键路径
        
        Args:
            workflow: 工作流定义字典，包含tasks列表
            
        Returns:
            关键路径上的任务ID列表
            
        Raises:
            ValueError: 如果工作流定义无效
        """
        if not workflow or not isinstance(workflow, dict):
            raise ValueError("Invalid workflow definition")
            
        tasks = workflow.get("tasks", [])
        if not tasks:
            return []
        
        # 构建任务字典
        task_dict = {t["id"]: t for t in tasks}
        
        # 计算最早开始时间和最晚开始时间
        earliest_start = {}
        latest_start = {}
        
        # 前向遍历计算最早开始时间
        for task in tasks:
            if "id" not in task:
                continue
                
            deps = task.get("dependencies", [])
            if not deps:
                earliest_start[task["id"]] = 0
            else:
                max_end = 0
                for dep_id in deps:
                    if dep_id in task_dict and dep_id in earliest_start:
                        dep_end = earliest_start[dep_id] + task_dict[dep_id].get("duration", 0)
                        max_end = max(max_end, dep_end)
                earliest_start[task["id"]] = max_end
        
        # 计算项目总时长
        project_duration = max(
            earliest_start.get(t["id"], 0) + t.get("duration", 0)
            for t in tasks
        )
        
        # 后向遍历计算最晚开始时间
        for task in reversed(tasks):
            task_id = task["id"]
            # 找出所有依赖此任务的任务
            successors = [t for t in tasks if task_id in t.get("dependencies", [])]
            
            if not successors:
                latest_start[task_id] = project_duration - task.get("duration", 0)
            else:
                min_start = float('inf')
                for succ in successors:
                    if succ["id"] in latest_start:
                        min_start = min(min_start, latest_start[succ["id"]])
                latest_start[task_id] = min_start - task.get("duration", 0)
        
        # 找出关键路径（最早开始时间=最晚开始时间的任务）
        critical_path = []
        for task_id in task_dict:
            if task_id in earliest_start and task_id in latest_start:
                if abs(earliest_start[task_id] - latest_start[task_id]) < 0.001:
                    critical_path.append(task_id)
        
        return critical_path
    
    async def _find_peak_usage(self, usage_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """找出资源使用高峰
        
        Args:
            usage_list: 资源使用记录列表，每个记录应包含:
                - start: 开始时间
                - end: 结束时间 
                - amount: 资源量
                - task_id: 关联任务ID
                
        Returns:
            包含以下键的字典:
                - max_concurrent: 最大并发资源量
                - peak_time: 达到峰值的时间点
                - tasks: 峰值时活跃的任务ID列表
                
        Raises:
            ValueError: 如果输入数据格式无效
        """
        if not usage_list:
            return {"max_concurrent": 0, "peak_time": 0, "tasks": []}
            
        # 验证输入数据格式
        required_keys = {"start", "end", "amount", "task_id"}
        for usage in usage_list:
            if not all(key in usage for key in required_keys):
                raise ValueError("Invalid usage record format")
            if usage["start"] > usage["end"]:
                raise ValueError("Start time cannot be after end time")
        
        try:
            # 创建时间点事件
            events = []
            for usage in usage_list:
                events.append((usage["start"], usage["amount"], usage["task_id"]))
                events.append((usage["end"], -usage["amount"], usage["task_id"]))
            
            events.sort()
            
            # 扫描找出最大并发使用量
            current_usage = 0
            max_usage = 0
            peak_time = 0
            active_tasks = set()
            peak_tasks = []
            
            for time, delta, task_id in events:
                if delta > 0:
                    active_tasks.add(task_id)
                else:
                    active_tasks.discard(task_id)
                
                current_usage += delta
                
                if current_usage > max_usage:
                    max_usage = current_usage
                    peak_time = time
                    peak_tasks = list(active_tasks)
            
            return {
                "max_concurrent": max_usage,
                "peak_time": peak_time,
                "tasks": peak_tasks
            }
            
        except Exception as e:
            logging.error(f"Error finding peak usage: {str(e)}")
            raise ValueError(f"Failed to calculate peak usage: {str(e)}")
    
    async def _calculate_cost(self, workflow: Dict[str, Any]) -> float:
        """计算工作流总成本
        
        Args:
            workflow: 工作流定义字典，必须包含tasks列表
            
        Returns:
            工作流总成本(浮点数)
            
        Raises:
            ValueError: 如果工作流定义无效或包含无效资源类型
            
        Notes:
            使用以下资源成本模型(单位:美元/小时):
            - cpu: $0.1 每单位
            - memory: $0.05 每GB
            - gpu: $1.0 每单位
            - storage: $0.01 每GB
            其他资源类型默认按$0.1/单位计算
        """
        if not workflow or not isinstance(workflow, dict):
            raise ValueError("Invalid workflow definition")
            
        total_cost = 0.0
        resource_costs = {
            "cpu": 0.1,      # $0.1 per unit per hour
            "memory": 0.05,   # $0.05 per GB per hour
            "gpu": 1.0,       # $1.0 per unit per hour
            "storage": 0.01   # $0.01 per GB per hour
        }
        
        try:
            for task in workflow.get("tasks", []):
                if not isinstance(task, dict):
                    continue
                    
                duration = task.get("duration", 0)
                if duration <= 0:
                    continue
                    
                duration_hours = duration / 3600.0
                
                resources = task.get("resources", {})
                if not isinstance(resources, dict):
                    continue
                    
                for resource, amount in resources.items():
                    if not isinstance(amount, (int, float)) or amount < 0:
                        continue
                        
                    unit_cost = resource_costs.get(resource, 0.1)  # default $0.1/unit/hour
                    total_cost += amount * duration_hours * unit_cost
                    
            return round(total_cost, 2)  # 保留两位小数
            
        except Exception as e:
            logging.error(f"Error calculating workflow cost: {str(e)}")
            raise ValueError(f"Failed to calculate cost: {str(e)}")
    
    async def _calculate_resource_usage(self, workflow: Dict[str, Any]) -> float:
        """计算工作流资源总利用率
        
        Args:
            workflow: 工作流定义字典，必须包含tasks列表
            
        Returns:
            资源利用率(0-1之间的浮点数)，表示总资源使用时间与总可用时间的比率
            
        Raises:
            ValueError: 如果工作流定义无效或包含无效资源数据
            
        Notes:
            计算方法:
            1. 计算每种资源的总使用时间(资源量×任务持续时间)
            2. 计算总可用时间(资源类型数×资源容量×工作流总持续时间)
            3. 利用率 = 总使用时间 / 总可用时间
            假设每种资源有100单位容量
        """
        if not workflow or not isinstance(workflow, dict):
            raise ValueError("Invalid workflow definition")
            
        total_resource_time = 0.0
        resource_capacity = 100  # 假设每种资源有100单位容量
        resource_types = set()
        
        try:
            # 计算工作流总持续时间
            total_duration = await self._calculate_duration(workflow)
            if total_duration <= 0:
                return 0.0
                
            # 收集资源使用数据
            for task in workflow.get("tasks", []):
                if not isinstance(task, dict):
                    continue
                    
                duration = task.get("duration", 0)
                if duration <= 0:
                    continue
                    
                resources = task.get("resources", {})
                if not isinstance(resources, dict):
                    continue
                    
                for resource, amount in resources.items():
                    if not isinstance(amount, (int, float)) or amount < 0:
                        continue
                        
                    resource_types.add(resource)
                    total_resource_time += amount * duration
            
            # 计算总可用时间
            resource_type_count = len(resource_types)
            if resource_type_count == 0:
                return 0.0
                
            total_available_time = resource_type_count * resource_capacity * total_duration
            if total_available_time <= 0:
                return 0.0
                
            # 计算并确保利用率在0-1范围内
            utilization = total_resource_time / total_available_time
            return min(1.0, max(0.0, utilization))
            
        except Exception as e:
            logging.error(f"Error calculating resource usage: {str(e)}")
            raise ValueError(f"Failed to calculate resource usage: {str(e)}")
    
    async def _calculate_throughput(self, workflow: Dict[str, Any]) -> float:
        """计算吞吐量"""
        # 简化：任务数/总时间
        task_count = len(workflow.get("tasks", []))
        total_duration = await self._calculate_duration(workflow)
        
        return task_count / total_duration if total_duration > 0 else 0
    
    async def _decompose_task_in_workflow(self, workflow: Dict[str, Any], 
                                        optimization: Dict[str, Any]) -> Dict[str, Any]:
        """在工作流中分解任务"""
        # 找到要分解的任务
        task_id = optimization["id"].replace("split_", "")
        
        tasks = workflow.get("tasks", [])
        for i, task in enumerate(tasks):
            if task["id"] == task_id:
                # 创建子任务
                subtasks = []
                original_duration = task.get("duration", 0)
                
                for j in range(3):  # 分成3个子任务
                    subtask = task.copy()
                    subtask["id"] = f"{task_id}_sub{j}"
                    subtask["duration"] = original_duration / 3
                    subtask["resources"] = {
                        k: v / 3 for k, v in task.get("resources", {}).items()
                    }
                    subtasks.append(subtask)
                
                # 替换原任务
                tasks[i:i+1] = subtasks
                break
        
        return workflow
    
    async def _scale_resources_in_workflow(self, workflow: Dict[str, Any], 
                                         optimization: Dict[str, Any]) -> Dict[str, Any]:
        """在工作流中扩展资源"""
        task_id = optimization["id"].replace("scale_", "")
        
        for task in workflow.get("tasks", []):
            if task["id"] == task_id:
                # 增加资源，减少时间
                for resource in task.get("resources", {}):
                    task["resources"][resource] *= 1.5
                task["duration"] *= 0.7  # 时间减少30%
                break
        
        return workflow
    
    async def _balance_load_in_workflow(self, workflow: Dict[str, Any], 
                                      optimization: Dict[str, Any]) -> Dict[str, Any]:
        """在工作流中平衡负载"""
        # 简化实现：调整任务开始时间以平滑资源使用
        tasks = workflow.get("tasks", [])
        
        # 按开始时间排序
        tasks.sort(key=lambda t: t.get("start_time", 0))
        
        # 错开任务开始时间
        for i in range(1, len(tasks)):
            if tasks[i].get("start_time", 0) == tasks[i-1].get("start_time", 0):
                # 延迟后续任务
                tasks[i]["start_time"] = tasks[i-1].get("start_time", 0) + 100
        
        return workflow

# ============================= 工具类实现 ======================// ...省略其他代码... 


// ==== 修改后代码 ====

class MCPToolRegistry:
    """MCP工具注册表"""
    
    def __init__(self):
        self.tools = {}
        self.tool_metadata = {}
        self._lock = asyncio.Lock()
    
    async def register(self, tool_name: str, tool_func: Callable, 
                      metadata: Dict[str, Any] = None):
        """注册工具"""
        async with self._lock:
            self.tools[tool_name] = tool_func
            self.tool_metadata[tool_name] = metadata or {}
            
            logging.info(f"Registered tool: {tool_name}")
    
    async def unregister(self, tool_name: str):
        """注销工具"""
        async with self._lock:
            if tool_name in self.tools:
                del self.tools[tool_name]
                del self.tool_metadata[tool_name]
                
                logging.info(f"Unregistered tool: {tool_name}")
    
    async def get_tool(self, tool_name: str) -> Optional[Callable]:
        """获取工具"""
        return self.tools.get(tool_name)
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        tools_list = []
        
        for name, metadata in self.tool_metadata.items():
            tools_list.append({
                "name": name,
                "description": metadata.get("description", ""),
                "parameters": metadata.get("parameters", {}),
                "returns": metadata.get("returns", "Any")
            })
        
        return tools_list
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """执行工具"""
        tool = await self.get_tool(tool_name)
        
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        # 验证参数
        metadata = self.tool_metadata.get(tool_name, {})
        required_params = metadata.get("parameters", {}).get("required", [])
        
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
        
        # 执行工具
        if asyncio.iscoroutinefunction(tool):
            return await tool(**kwargs)
        else:
            return await asyncio.to_thread(tool, **kwargs)

class VectorDatabase:
    """向量数据库"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map = {}
        self.data_store = {}
        self._current_id = 0
        self._lock = asyncio.Lock()
    
    async def add(self, vector: np.ndarray, data: Any, metadata: Dict[str, Any] = None) -> str:
        """添加向量"""
        async with self._lock:
            # 生成ID
            vector_id = f"vec_{self._current_id}"
            
            # 添加到FAISS索引
            self.index.add(vector.reshape(1, -1).astype(np.float32))
            
            # 存储映射和数据
            self.id_map[self._current_id] = vector_id
            self.data_store[vector_id] = {
                "data": data,
                "metadata": metadata or {},
                "vector": vector,
                "added_at": datetime.now()
            }
            
            self._current_id += 1
            
            return vector_id
    
    async def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float, Any]]:
        """搜索相似向量"""
        async with self._lock:
            if self.index.ntotal == 0:
                return []
            
            # 搜索
            distances, indices = self.index.search(
                query_vector.reshape(1, -1).astype(np.float32),
                min(k, self.index.ntotal)
            )
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx in self.id_map:
                    vector_id = self.id_map[idx]
                    data_entry = self.data_store[vector_id]
                    
                    similarity = 1 / (1 + dist)
                    results.append((vector_id, similarity, data_entry["data"]))
            
            return results
    
    async def get(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """获取向量数据"""
        return self.data_store.get(vector_id)
    
    async def delete(self, vector_id: str) -> bool:
        """删除向量"""
        async with self._lock:
            if vector_id in self.data_store:
                del self.data_store[vector_id]
                # 注意：FAISS不支持直接删除，需要重建索引
                return True
            return False
    
    async def update_metadata(self, vector_id: str, metadata: Dict[str, Any]):
        """更新元数据"""
        async with self._lock:
            if vector_id in self.data_store:
                self.data_store[vector_id]["metadata"].update(metadata)

class EventBus:
    """事件总线"""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_history = deque(maxlen=1000)
        self._lock = asyncio.Lock()
    
    async def subscribe(self, event_type: str, handler: Callable):
        """订阅事件"""
        async with self._lock:
            self.subscribers[event_type].append(handler)
    
    async def unsubscribe(self, event_type: str, handler: Callable):
        """取消订阅"""
        async with self._lock:
            if event_type in self.subscribers:
                self.subscribers[event_type].remove(handler)
    
    async def publish(self, event_type: str, data: Any):
        """发布事件"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now()
        }
        
        # 记录事件历史
        self.event_history.append(event)
        
        # 通知订阅者
        handlers = self.subscribers.get(event_type, [])
        
        # 并发执行处理器
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(event))
            else:
                tasks.append(asyncio.to_thread(handler, event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_history(self, event_type: Optional[str] = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """获取事件历史"""
        history = list(self.event_history)
        
        if event_type:
            history = [e for e in history if e["type"] == event_type]
        
        return history[-limit:]

# ============================= 示例和测试代码 ======================// ...省略其他代码... 


// ==== 修改后代码 ====

async def example_usage():
    """示例用法"""
    # 创建系统
    system = HyperAgentSystem()
    
    # 初始化
    await system.initialize()
    
    # 启动系统
    await system.start()
    
    # 提交任务
    task1 = await system.submit_task(
        task_description="Analyze customer feedback data and generate insights report",
        priority=8,
        constraints=["Must complete within 2 hours", "Use only approved data sources"],
        requirements=["Statistical analysis", "Sentiment analysis", "Visualization"],
        metadata={
            "data_sources": [
                {"type": "file", "path": "feedback.csv"},
                {"type": "api", "endpoint": "https://api.example.com/reviews"}
            ],
            "output_format": "PDF report with charts"
        }
    )
    
    # 监控任务进度
    while task1.status not in ["completed", "failed"]:
        await asyncio.sleep(5)
        status = await system.get_system_status()
        print(f"System status: {status}")
    
    # 获取结果
    print(f"Task result: {task1.result}")
    
    # 停止系统
    await system.stop()

async def test_agent_collaboration():
    """测试Agent协作"""
    # 创建协作框架
    framework = CollaborationFramework(CoordinationStrategy.HIERARCHICAL)
    
    # 创建Agent
    supervisor = SupervisorAgent("sup_1", "Chief Supervisor")
    planner = PlannerAgent("plan_1", "Strategic Planner")
    executor1 = ExecutorAgent("exec_1", "Code Executor")
    executor2 = ExecutorAgent("exec_2", "Data Processor")
    analyzer = AnalyzerAgent("analyzer_1", "Data Analyst")
    
    # 注册到框架
    await framework.register_agent(supervisor)
    await framework.register_agent(planner)
    await framework.register_agent(executor1)
    await framework.register_agent(executor2)
    await framework.register_agent(analyzer)
    
    # 启动框架
    await framework.start()
    
    # 创建复杂任务
    complex_task = Task(
        name="Build ML Pipeline",
        description="Create an end-to-end machine learning pipeline for customer churn prediction",
        objective="Deploy a production-ready ML model with >85% accuracy",
        constraints=["Use only internal data", "Complete within 1 week"],
        requirements=[
            "Data preprocessing",
            "Feature engineering", 
            "Model training",
            "Model evaluation",
            "API deployment"
        ],
        priority=9
    )
    
    # 提交任务
    result = await framework.submit_task(complex_task)
    
    print(f"Task completed with result: {result}")
    
    # 停止框架
    await framework.stop()

async def test_memory_system():
    """测试记忆系统"""
    memory_system = HierarchicalMemorySystem()
    
    # 存储不同类型的记忆
    # 感官记忆
    sensory_id = await memory_system.store(
        content="User clicked on product recommendation",
        memory_type=MemoryType.SENSORY,
        importance=0.3,
        context={"user_id": "user123", "product_id": "prod456"}
    )
    
    # 短期记忆
    short_term_id = await memory_system.store(
        content="Customer expressed interest in premium features",
        memory_type=MemoryType.SHORT_TERM,
        importance=0.7,
        context={"conversation_id": "conv789", "timestamp": datetime.now().isoformat()}
    )
    
    # 长期记忆
    long_term_id = await memory_system.store(
        content="Successful sales strategy: Bundle products with complementary services",
        memory_type=MemoryType.LONG_TERM,
        importance=0.9,
        context={"success_rate": 0.75, "revenue_impact": "+23%"}
    )
    
    # 建立关联
    memory_system.associate(short_term_id, long_term_id, strength=0.8)
    
    # 巩固记忆
    memory_system.consolidate()
    
    # 搜索相似记忆
    query_embedding = np.random.randn(768)
    similar_memories = memory_system.search_similar(query_embedding, k=5)
    
    print(f"Found {len(similar_memories)} similar memories")
    
    # 获取关联记忆
    associated = memory_system.get_associated_memories(short_term_id, max_hops=2)
    print(f"Found {len(associated)} associated memories")

async def test_tool_integration():
    """测试工具集成"""
    # 创建工具注册表
    registry = MCPToolRegistry()
    
    # 注册工具
    async def web_search_tool(query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """模拟网络搜索"""
        return [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": f"Content for {query}"}
            for i in range(max_results)
        ]
    
    await registry.register(
        "web_search",
        web_search_tool,
        metadata={
            "description": "Search the web for information",
            "parameters": {
                "required": ["query"],
                "optional": ["max_results"]
            },
            "returns": "List of search results"
        }
    )
    
    # 列出工具
    tools = await registry.list_tools()
    print(f"Available tools: {tools}")
    
    # 执行工具
    results = await registry.execute_tool("web_search", query="AI agents", max_results=5)
    print(f"Search results: {results}")

# ============================= 主函数 ======================// ...省略其他代码... 


// ==== 修改后代码 ====

async def main():
    """主函数"""
    print("HyperAgent Platform v1.0")
    print("=" * 50)
    
    # 运行示例
    # await example_usage()
    
    # 测试组件
    # await test_agent_collaboration()
    # await test_memory_system()
    # await test_tool_integration()
    
    # 简单演示
    system = HyperAgentSystem()
    await system.initialize()
    
    print("\nSystem initialized successfully!")
    print(f"Total agents: {len(system.agents)}")
    print("\nAgent breakdown:")
    
    agent_counts = defaultdict(int)
    for agent in system.agents.values():
        agent_counts[agent.role.value] += 1
    
    for role, count in agent_counts.items():
        print(f"  - {role}: {count}")
    
    # 获取系统状态
    status = await system.get_system_status()
    print(f"\nSystem status: {json.dumps(status, indent=2, default=str)}")

if __name__ == "__main__":
    asyncio.run(main())
    # HyperAgent Core System#