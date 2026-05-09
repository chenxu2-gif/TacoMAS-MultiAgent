# Meta-LLM 驱动的 Birth-Death 图共演化算法

## 概述

这是一个实现了 Meta-LLM 驱动的 Birth-Death 图共演化（BD-GCE）算法的系统。核心思想是引入一个高层的 meta LLM 作为"种群调度器"和"结构编辑器"，让它隐式决定：

1. **哪个 agent 应该 birth**（复制/扩张）
2. **哪个 agent 应该 death**（淘汰）
3. **哪些边或局部子图应该被替换**
4. **快时间尺度与慢时间尺度如何协同更新**

## 系统架构

### 三层时间尺度

```
快时间尺度 (Fast):     底层 agents 通过 LLM 和 memory 持续学习
    ↓
中慢时间尺度 (Medium):  meta LLM 判断是否触发 Birth-Death
    ↓
更慢时间尺度 (Slow):   meta LLM 判断是否触发图结构调整
```

### 核心模块

```
tacomas/meta_evolution/
├── schemas.py              # 数据结构定义
├── graph_manager.py        # 图结构管理（支持双向和单向边）
├── population_manager.py   # 种群管理（agent 生命周期）
├── summarizer.py           # 系统摘要生成（压缩状态）
├── meta_llm.py            # Meta LLM 接口（调用 LLM）
├── executor.py            # 约束投影和执行器
├── evolution_controller.py # 主控制器
└── example_usage.py       # 使用示例
```

## 快速开始

### 1. 基本配置

```python
from tacomas.config.llm import LLMConfig
from tacomas.config.meta_evolution import MetaEvolutionConfig
from tacomas.meta_evolution import EvolutionController

# 配置 LLM
llm_config = LLMConfig(
    model="gpt-4-turbo",
    temperature=0.7,
    api_key="your-api-key",
)

# 配置演化系统
evolution_config = MetaEvolutionConfig(
    enabled=True,
    meta_llm_model="gpt-4-turbo",
    n_min=5,
    n_max=15,
    bd_check_interval=50,
    graph_rewire_interval=200,
)

# 初始化控制器
controller = EvolutionController(evolution_config, llm_config)
```

### 2. 添加 Agents

```python
from tacomas.meta_evolution import AgentState

agent = AgentState(
    agent_id="planner",
    role="planner",
    policy="Break down the task into subproblems",
    output="",
    memory_summary="",
    capability={"planning": 0.8, "reasoning": 0.7},
    recent_scores=[0.7, 0.75, 0.8],
    score_trend=0.05,
    improvement_direction="Focus on clearer decomposition",
    neighbor_externality=0.3,
    redundancy_score=0.0,
    failure_modes=["Incomplete decomposition"],
    bridge_value=0.8,
    community_id=0,
    creation_time=0,
)

controller.add_agent(agent)
```

### 3. 构建图结构

```python
# 添加单向边（serial handoff）
controller.add_edge("planner", "proposer", "directed")

# 添加双向边（debate）
controller.add_edge("proposer", "verifier", "bidirectional")
```

### 4. 运行演化循环

```python
for step in range(1000):
    # 快时间步
    controller.step_fast_time()
    
    # 更新 agent 状态
    for agent_id in controller.get_agent_ids():
        agent = controller.population.get_agent(agent_id)
        # ... 更新 agent 的 scores, memory 等
        controller.update_agent_state(agent)
    
    # 检查是否触发慢时间更新
    if controller.should_trigger_slow_update():
        controller.execute_slow_update(
            task_description="Solve complex reasoning tasks",
            current_answer_quality=0.75,
        )
```

## 关键概念

### 边类型

系统支持两种边类型：

1. **双向边 (Bidirectional)**: 用于 debate 交互
   - 两个 agent 针对同一问题展开双向讨论
   - 支持批评、反驳、修正

2. **单向边 (Directed)**: 用于 serial handoff
   - 上游 agent 的输出作为下游 agent 的输入
   - 适合流水线任务、逐步推理

### Agent 状态

每个 agent 维护以下状态：

```python
AgentState:
  - agent_id: 唯一标识
  - role: 角色（planner, proposer, verifier 等）
  - policy: 当前策略/行为描述
  - output: 最新输出
  - memory_summary: 压缩的历史记忆
  - capability: 能力向量
  - recent_scores: 最近的评分
  - score_trend: 评分趋势
  - improvement_direction: 改进方向（来自环境反馈）
  - neighbor_externality: 对邻居的正/负外部性
  - redundancy_score: 与其他 agent 的冗余度
  - failure_modes: 常见失败模式
  - bridge_value: 桥接价值（对图连通性的重要性）
```

### Meta LLM 决策

Meta LLM 在每个慢时间步输出一个结构化决策：

```json
{
  "slow_update": true,
  "confidence": 0.8,
  "birth_death_pairs": [
    {
      "parent": "agent_id",
      "death_target": "agent_id",
      "birth_reason": "...",
      "death_reason": "...",
      "child_plan": {
        "inherit_type": "pure_clone|mutate_role|hybrid",
        "memory_from_parent_ratio": 0.8,
        "memory_from_target_ratio": 0.2,
        "capability_noise_scale": 0.1,
        "policy_mutation": "...",
        "role_prompt_update": "..."
      },
      "improvement_direction": "..."
    }
  ],
  "graph_edit": {
    "remove_edges": [["a","b"]],
    "add_edges": [["c","d"]],
    "type_changes": [{"edge": ["a","b"], "new_type": "bidirectional"}],
    "rewire_notes": ["..."],
    "subgraph_rewrites": ["..."]
  },
  "final_synthesis": {
    "contributors": ["agent_1", "agent_3"],
    "strategy": "...",
    "conflict_resolution": "...",
    "final_answer_spec": "..."
  },
  "time_control": {
    "birth_death_value": 0.6,
    "graph_rewire_value": 0.2,
    "fast_steps_next": 20,
    "trigger_birth_death": true,
    "trigger_graph_rewire": false,
    "trigger_next_slow_rule": "adaptive",
    "cooldown": 0
  },
  "global_rationale": ["reason 1", "reason 2"]
}
```

## 配置参数

### 时间尺度控制

```yaml
fast_steps_per_window: 10        # 每个窗口的快步数
bd_check_interval: 50            # Birth-Death 检查间隔
graph_rewire_interval: 200       # 图重连间隔
```

### 种群控制

```yaml
n_min: 5                         # 最小 agent 数
n_max: 15                        # 最大 agent 数
specialization_level: "medium"   # 分工强度：low, medium, high
```

### 演化控制

```yaml
max_birth_death_pairs: 2         # 每次最多替换的 agent 对数
max_edge_edits: 5                # 每次最多编辑的边数
protected_nodes: []              # 受保护的节点（不能删除）
critical_roles: []               # 关键角色（必须保留）
```

### 约束

```yaml
preserve_connectivity: true      # 保持图连通性
max_degree: 10                   # 最大节点度数
```

## 系统摘要

Meta LLM 接收的系统摘要包含以下信息：

### 全局统计

- 平均评分
- 评分趋势
- 图密度
- 模块性
- 瓶颈节点数
- 多样性指数
- 系统稳定性
- 学习效率

### 节点摘要

- 最近评分
- 评分趋势
- 改进方向
- 邻域外部性
- 冗余度
- 失败模式
- 桥接价值

### 边摘要

- 交互质量
- 信息冗余度
- 冲突率
- 桥接标志

### 子图摘要

- 收敛状态
- 质量趋势
- 内部密度
- 外部桥接数
- 瓶颈节点

## 约束投影

Meta LLM 的输出可能违反系统约束，约束投影器会自动修复：

1. **种群大小约束**: 确保 agent 数量在 [n_min, n_max] 范围内
2. **Birth-Death 有效性**: 检查父代和死亡目标是否存在
3. **图编辑有效性**: 检查边操作是否合法
4. **角色覆盖**: 确保关键角色不被完全删除
5. **连通性**: 确保图保持连通（可选）

## 日志和监控

系统会记录每次演化决策：

```
meta_evolution_logs/
├── evolution_0_2024-01-15T10-30-45.json
├── evolution_1_2024-01-15T10-35-20.json
└── ...
```

每个日志文件包含：

```json
{
  "timestamp": "2024-01-15T10:30:45",
  "slow_update_step": 0,
  "fast_time_step": 50,
  "decision": {
    "confidence": 0.8,
    "birth_death_pairs": 1,
    "graph_edits": {...},
    "rationale": [...]
  },
  "system_state": {
    "num_agents": 8,
    "num_edges": 12,
    "mean_score": 0.75,
    ...
  }
}
```

## 继承类型

当创建新 agent 时，可以选择不同的继承类型：

### 1. Pure Clone (纯克隆)

完全复制父代 agent：

```python
ChildInheritancePlan(
    inherit_type=InheritanceType.PURE_CLONE,
    memory_from_parent_ratio=1.0,
    memory_from_target_ratio=0.0,
    capability_noise_scale=0.0,
)
```

### 2. Mutate Role (角色突变)

复制父代但改变角色：

```python
ChildInheritancePlan(
    inherit_type=InheritanceType.MUTATE_ROLE,
    memory_from_parent_ratio=0.8,
    memory_from_target_ratio=0.0,
    capability_noise_scale=0.1,
    role_prompt_update="New role prompt here",
)
```

### 3. Hybrid (混合)

混合继承父代和被替换 agent 的特性：

```python
ChildInheritancePlan(
    inherit_type=InheritanceType.HYBRID,
    memory_from_parent_ratio=0.6,
    memory_from_target_ratio=0.4,
    capability_noise_scale=0.15,
)
```

## 最佳实践

### 1. 初始化

- 从较小的 agent 数量开始（5-8 个）
- 确保关键角色被保护
- 设置合理的时间尺度间隔

### 2. 监控

- 定期检查日志文件
- 监控系统稳定性和学习效率
- 调整参数以适应任务

### 3. 调优

- 增加 `bd_check_interval` 以减少演化频率
- 增加 `max_birth_death_pairs` 以加快演化
- 调整 `specialization_level` 以控制分工程度

### 4. 约束

- 始终保护关键角色
- 设置合理的最大度数限制
- 启用连通性保护以避免图碎片化

## 与现有系统的集成

### 集成到 CentralizedMultiAgentSystem

```python
from tacomas.agents.multiagent_centralized import CentralizedMultiAgentSystem
from tacomas.meta_evolution import EvolutionController

# 创建演化控制器
evolution_controller = EvolutionController(evolution_config, llm_config)

# 在多 agent 系统中使用
class BDGCEMultiAgentSystem(CentralizedMultiAgentSystem):
    def __init__(self, *args, evolution_controller=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evolution_controller = evolution_controller
    
    def run_agent(self, instance, instance_dir=None, llm_params=None, instance_idx=None):
        # ... 运行底层 agents
        
        # 检查是否触发演化
        if self.evolution_controller.should_trigger_slow_update():
            self.evolution_controller.execute_slow_update(
                task_description=instance.task,
                current_answer_quality=current_quality,
            )
        
        # ... 返回结果
```

## 故障排除

### 问题：Meta LLM 调用失败

**解决方案**：
- 检查 API 密钥是否正确设置
- 检查网络连接
- 查看日志中的错误信息

### 问题：Agent 数量不变

**解决方案**：
- 增加 `bd_check_interval` 的值
- 检查 `trigger_birth_death` 是否为 true
- 检查是否有受保护的节点阻止了演化

### 问题：图变得不连通

**解决方案**：
- 启用 `preserve_connectivity=True`
- 减少 `max_edge_edits`
- 检查受保护节点的配置

## 性能考虑

1. **摘要压缩**: 系统自动压缩摘要以减少 token 使用
2. **异步调用**: Meta LLM 调用不阻塞底层 agents
3. **缓存**: 可以启用决策缓存以减少 API 调用
4. **约束投影**: 快速检查决策可行性

## 参考文献

- 原始 TS-BD-GCE 框架
- Meta-learning 和种群演化算法
- 多 agent 系统和图神经网络

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 issue 或 pull request。
