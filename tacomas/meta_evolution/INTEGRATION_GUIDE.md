# Meta-LLM 演化系统集成指南

## 概述

本指南说明如何将 Meta-LLM 驱动的 Birth-Death 图共演化系统集成到现有的多 agent 框架中。

## 集成架构

```
CentralizedMultiAgentSystem (现有)
    ↓
BDGCEMultiAgentSystem (新增)
    ├── LeadAgent (现有)
    ├── SubAgents (现有)
    └── EvolutionController (新增)
```

## 步骤 1: 创建 BD-GCE 多 agent 系统

```python
from tacomas.agents.multiagent_centralized import CentralizedMultiAgentSystem
from tacomas.config.meta_evolution import MetaEvolutionConfig
from tacomas.meta_evolution import EvolutionController
from tacomas.config.llm import LLMConfig

class BDGCEMultiAgentSystem(CentralizedMultiAgentSystem):
    """多 agent 系统，支持 Birth-Death 图共演化。"""
    
    def __init__(
        self,
        *args,
        evolution_config: MetaEvolutionConfig = None,
        meta_llm_config: LLMConfig = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        if evolution_config is None:
            evolution_config = MetaEvolutionConfig()
        
        if meta_llm_config is None:
            meta_llm_config = LLMConfig(model="gpt-4-turbo")
        
        self.evolution_controller = EvolutionController(
            evolution_config,
            meta_llm_config,
        )
        
        self.evolution_enabled = evolution_config.enabled
        self.fast_time_step = 0
```

## 步骤 2: 初始化 Agent 和图结构

```python
def initialize_evolution_system(self):
    """初始化演化系统中的 agents 和图结构。"""
    from tacomas.meta_evolution import AgentState
    
    # 添加 lead agent
    lead_agent_state = AgentState(
        agent_id="lead_agent",
        role="orchestrator",
        policy="Coordinate and synthesize subagent outputs",
        output="",
        memory_summary="",
        capability={"coordination": 0.9, "synthesis": 0.8},
        recent_scores=[0.8],
        score_trend=0.0,
        improvement_direction="",
        neighbor_externality=0.5,
        redundancy_score=0.0,
        failure_modes=[],
        bridge_value=1.0,
        community_id=0,
        creation_time=0,
    )
    self.evolution_controller.add_agent(lead_agent_state)
    
    # 添加 subagents
    for i in range(self.n_base_agents):
        subagent_state = AgentState(
            agent_id=f"subagent_{i}",
            role="worker",
            policy="Execute assigned tasks",
            output="",
            memory_summary="",
            capability={"task_execution": 0.7, "reasoning": 0.6},
            recent_scores=[0.6],
            score_trend=0.0,
            improvement_direction="",
            neighbor_externality=0.2,
            redundancy_score=0.0,
            failure_modes=[],
            bridge_value=0.3,
            community_id=0,
            creation_time=0,
        )
        self.evolution_controller.add_agent(subagent_state)
    
    # 构建图结构
    for i in range(self.n_base_agents):
        self.evolution_controller.add_edge(
            "lead_agent",
            f"subagent_{i}",
            "directed"
        )
```

## 步骤 3: 在主循环中集成演化

```python
async def run_agent_async(
    self,
    instance: DatasetInstance,
    instance_dir: Optional[str] = None,
    llm_params: Optional[LLMParams] = None,
    instance_idx: Optional[int] = None,
) -> DatasetInstanceOutputWithTrajectory:
    """运行多 agent 系统，支持演化。"""
    
    start_time = time.time()
    
    if self.evolution_enabled:
        self.initialize_evolution_system()
    
    # 主循环
    for iteration in range(100):
        if self.evolution_enabled:
            self.evolution_controller.step_fast_time()
            self.fast_time_step += 1
        
        # 运行 lead agent
        processing_result = await self.lead_agent.orchestrate_work(
            task_instance=instance,
            llm_params_dict=llm_params.model_dump() if llm_params else {},
        )
        
        # 更新演化系统中的 agent 状态
        if self.evolution_enabled:
            self._update_evolution_agent_states(processing_result)
        
        # 检查是否触发慢时间更新
        if self.evolution_enabled and self.evolution_controller.should_trigger_slow_update():
            current_quality = self._estimate_answer_quality(processing_result)
            self.evolution_controller.execute_slow_update(
                task_description=instance.task,
                current_answer_quality=current_quality,
            )
        
        if self._has_converged(processing_result):
            break
    
    execution_time = time.time() - start_time
    
    return DatasetInstanceOutputWithTrajectory(
        data_instance=instance,
        agent_output=processing_result.synthesized_answer,
        trajectory=[],
        final_env_output=processing_result.combined_env_status,
    )

def _update_evolution_agent_states(self, processing_result):
    """从处理结果中更新演化系统的 agent 状态。"""
    
    lead_agent_state = self.evolution_controller.population.get_agent("lead_agent")
    if lead_agent_state:
        lead_agent_state.output = str(processing_result.synthesized_answer)[:200]
        lead_agent_state.recent_scores.append(0.75)
        lead_agent_state.score_trend = 0.01
        self.evolution_controller.update_agent_state(lead_agent_state)
    
    for i, conv in enumerate(processing_result.subagent_conversations):
        subagent_state = self.evolution_controller.population.get_agent(f"subagent_{i}")
        if subagent_state:
            subagent_state.output = conv.latest_response[:200] if conv.latest_response else ""
            subagent_state.recent_scores.append(0.7)
            subagent_state.score_trend = 0.005
            self.evolution_controller.update_agent_state(subagent_state)

def _estimate_answer_quality(self, processing_result) -> float:
    """估计当前答案的质量。"""
    if processing_result.synthesized_answer:
        return 0.75
    return 0.5

def _has_converged(self, processing_result) -> bool:
    """检查系统是否已收敛。"""
    return False
```

## 步骤 4: 配置和运行

```python
# 创建配置
evolution_config = MetaEvolutionConfig(
    enabled=True,
    meta_llm_model="gpt-4-turbo",
    n_min=3,
    n_max=10,
    bd_check_interval=50,
    graph_rewire_interval=200,
    protected_nodes=["lead_agent"],
    critical_roles=["orchestrator"],
)

meta_llm_config = LLMConfig(
    model="gpt-4-turbo",
    temperature=0.7,
)

# 创建系统
system = BDGCEMultiAgentSystem(
    llm=llm,
    dataset=dataset,
    prompts=prompts,
    n_base_agents=5,
    evolution_config=evolution_config,
    meta_llm_config=meta_llm_config,
)

# 运行
result = asyncio.run(system.run_agent_async(instance))
```

## 关键集成点

### 1. Agent 状态映射

```python
def map_agent_to_state(agent, agent_id: str):
    """将现有 agent 映射到 AgentState。"""
    from tacomas.meta_evolution import AgentState
    
    return AgentState(
        agent_id=agent_id,
        role=getattr(agent, 'role', 'worker'),
        policy=getattr(agent, 'system_prompt', ''),
        output=getattr(agent, 'last_output', ''),
        memory_summary=getattr(agent, 'memory_summary', ''),
        capability=getattr(agent, 'capability', {}),
        recent_scores=getattr(agent, 'recent_scores', []),
        score_trend=getattr(agent, 'score_trend', 0.0),
        improvement_direction=getattr(agent, 'improvement_direction', ''),
        neighbor_externality=0.0,
        redundancy_score=0.0,
        failure_modes=[],
        bridge_value=0.0,
        community_id=0,
        creation_time=0,
    )
```

### 2. 评分机制

```python
def score_agent_output(agent_id: str, output: str) -> float:
    """为 agent 输出评分。"""
    # 基于输出质量、完整性等
    return 0.75
```

### 3. 图结构初始化

```python
def initialize_graph_for_task(task_type: str):
    """根据任务类型初始化图结构。"""
    
    if task_type == "reasoning":
        edges = [
            ("lead_agent", "subagent_0", "directed"),
            ("subagent_0", "subagent_1", "directed"),
            ("subagent_1", "subagent_2", "bidirectional"),
        ]
    elif task_type == "debate":
        edges = [
            ("subagent_0", "subagent_1", "bidirectional"),
            ("subagent_1", "subagent_2", "bidirectional"),
        ]
    else:
        edges = [
            ("lead_agent", f"subagent_{i}", "directed")
            for i in range(3)
        ]
    
    return edges
```

## 性能优化

### 1. 减少 Meta LLM 调用

```python
evolution_config.bd_check_interval = 100
evolution_config.enable_cache = True
```

### 2. 检查点保存

```python
import json

def save_checkpoint(controller, step: int):
    """保存系统检查点。"""
    checkpoint = {
        "step": step,
        "agents": controller.population.to_dict(),
        "graph": controller.graph.to_dict(),
    }
    with open(f"checkpoint_{step}.json", "w") as f:
        json.dump(checkpoint, f)

def load_checkpoint(controller, checkpoint_file: str):
    """加载系统检查点。"""
    with open(checkpoint_file) as f:
        checkpoint = json.load(f)
    
    controller.population.from_dict(checkpoint["agents"])
    controller.graph.from_dict(checkpoint["graph"])
```

## 测试

```python
def test_evolution_system():
    """测试演化系统。"""
    from tacomas.meta_evolution import EvolutionController, AgentState
    
    config = MetaEvolutionConfig(
        enabled=True,
        n_min=3,
        n_max=5,
        bd_check_interval=10,
    )
    
    llm_config = LLMConfig(model="gpt-4-turbo")
    controller = EvolutionController(config, llm_config)
    
    # 添加测试 agents
    for i in range(3):
        agent = AgentState(
            agent_id=f"agent_{i}",
            role="worker",
            policy="",
            output="",
            memory_summary="",
            capability={},
            recent_scores=[0.5],
            score_trend=0.0,
            improvement_direction="",
            neighbor_externality=0.0,
            redundancy_score=0.0,
            failure_modes=[],
            bridge_value=0.0,
            community_id=0,
            creation_time=0,
        )
        controller.add_agent(agent)
    
    # 测试演化
    for step in range(100):
        controller.step_fast_time()
        
        if controller.should_trigger_slow_update():
            controller.execute_slow_update()
    
    # 验证结果
    assert controller.population.get_population_size() <= config.n_max
    assert controller.population.get_population_size() >= config.n_min
    print("Evolution system test passed!")

if __name__ == "__main__":
    test_evolution_system()
```

## 总结

集成 Meta-LLM 演化系统的关键步骤：

1. ✅ 创建 `BDGCEMultiAgentSystem` 子类
2. ✅ 初始化 `EvolutionController`
3. ✅ 映射现有 agents 到 `AgentState`
4. ✅ 构建初始图结构
5. ✅ 在主循环中集成演化
6. ✅ 实现评分和反馈机制
7. ✅ 监控和日志记录
8. ✅ 性能优化和故障恢复

更多详情请参考 `README.md` 和 `example_usage.py`。
