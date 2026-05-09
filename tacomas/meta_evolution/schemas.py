"""
Core data structures for Meta-LLM driven Birth-Death Graph Co-Evolution.
"""

from typing import Dict, List, Optional, Literal, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class EdgeType(str, Enum):
    """Edge types in the mixed graph."""
    BIDIRECTIONAL = "bidirectional"  # debate edge
    DIRECTED = "directed"  # serial handoff edge
    EVIDENCE_FLOW = "evidence_flow"
    VERIFICATION_FLOW = "verification_flow"
    COMPUTATION_FLOW = "computation_flow"
    REFLECTION_FEEDBACK = "reflection_feedback"

    @classmethod
    def from_value(cls, value: Union[str, "EdgeType", None]) -> "EdgeType":
        """Parse an edge type value with backward-compatible fallback."""
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        if text in cls._value2member_map_:
            return cls(text)
        return cls.DIRECTED

    def is_bidirectional(self) -> bool:
        return self == EdgeType.BIDIRECTIONAL

    def is_auxiliary(self) -> bool:
        return self == EdgeType.REFLECTION_FEEDBACK

    def is_mainline(self) -> bool:
        return self in {
            EdgeType.BIDIRECTIONAL,
            EdgeType.DIRECTED,
            EdgeType.EVIDENCE_FLOW,
            EdgeType.VERIFICATION_FLOW,
            EdgeType.COMPUTATION_FLOW,
        }

    def semantic_group(self) -> str:
        if self == EdgeType.BIDIRECTIONAL:
            return "debate"
        if self == EdgeType.EVIDENCE_FLOW:
            return "evidence_handoff"
        if self == EdgeType.VERIFICATION_FLOW:
            return "verification_handoff"
        if self == EdgeType.COMPUTATION_FLOW:
            return "computation_handoff"
        if self == EdgeType.REFLECTION_FEEDBACK:
            return "corrective_feedback"
        return "generic_handoff"

    def importance_weight(self) -> float:
        if self == EdgeType.EVIDENCE_FLOW:
            return 1.0
        if self == EdgeType.VERIFICATION_FLOW:
            return 0.95
        if self == EdgeType.COMPUTATION_FLOW:
            return 0.9
        if self == EdgeType.BIDIRECTIONAL:
            return 0.85
        if self == EdgeType.REFLECTION_FEEDBACK:
            return 0.45
        return 0.75


class InheritanceType(str, Enum):
    """Types of inheritance for child agents."""
    PURE_CLONE = "pure_clone"
    MUTATE_ROLE = "mutate_role"
    HYBRID = "hybrid"


@dataclass
class AgentState:
    """State of a single agent at time t."""
    agent_id: str
    role: str
    
    # Core state variables
    policy: str  # Current policy or behavior description
    output: str  # Latest output y_i^t
    memory_summary: str  # Compressed memory M_i^t
    capability: Dict[str, float]  # Capability vector c_i^t
    
    # Performance tracking
    recent_scores: List[float] = field(default_factory=list)
    score_trend: float = 0.0  # Slope of recent scores
    improvement_direction: str = ""  # m_i^(t) from environment / meta controller
    meta_score: float = 0.0
    workflow_correction: str = ""
    next_round_workflow: str = ""
    score_reason: str = ""
    
    # Interaction quality
    neighbor_externality: float = 0.0  # Positive/negative spillover to neighbors
    redundancy_score: float = 0.0  # How redundant with other agents
    failure_modes: List[str] = field(default_factory=list)  # Common failure patterns
    
    # Graph properties
    bridge_value: float = 0.0  # How much this agent bridges communities
    community_id: Optional[int] = None
    
    # Metadata
    creation_time: int = 0
    last_update_time: int = 0
    prompt_variant: str = ""


@dataclass
class EdgeSummary:
    """Summary of an edge in the graph."""
    source: str
    target: str
    edge_type: EdgeType
    semantic_group: str
    importance_weight: float
    
    # Quality metrics
    interaction_quality: float  # How well do these agents collaborate
    redundancy: float  # Information redundancy on this edge
    conflict_rate: float  # How often they disagree
    bridge_flag: bool  # Is this a critical bridge edge
    
    # History
    interaction_count: int = 0
    last_interaction_time: int = 0


@dataclass
class NodeSummary:
    """Summary of a node for meta LLM."""
    agent_id: str
    role: str
    recent_scores: List[float]
    score_trend: float
    latest_output: str
    current_improvement_direction: str
    neighbor_externality: float
    redundancy_cluster: Optional[str]
    memory_summary: str
    failure_modes: List[str]
    bridge_value: float
    community_id: Optional[int]
    last_meta_score: float = 0.0
    workflow_correction: str = ""
    next_round_workflow: str = ""


@dataclass
class SubgraphSummary:
    """Summary of a subgraph/community."""
    community_id: int
    agent_ids: List[str]
    convergence_status: str  # "converged", "learning", "stuck"
    quality_trend: float
    internal_density: float
    external_bridges: int
    bottleneck_nodes: List[str]


@dataclass
class SystemSummary:
    """Complete system summary for meta LLM decision."""
    slow_update_step: int
    fast_time_window: tuple  # (start_step, end_step)
    task_objective: str
    
    # Global statistics
    mean_score: float
    score_trend: float
    graph_density: float
    modularity: float
    bottleneck_count: int
    diversity_index: float
    current_answer_quality: float
    
    # Detailed summaries
    node_summaries: List[NodeSummary]
    edge_summaries: List[EdgeSummary]
    subgraph_summaries: List[SubgraphSummary]
    
    # System state
    is_stable: bool
    has_bottleneck: bool
    learning_efficiency: float  # Recent improvement rate
    
    # Metadata
    total_agents: int
    total_edges: int
    graph_connectivity_score: float = 0.0


@dataclass
class ChildInheritancePlan:
    """Plan for how a child agent inherits from parent."""
    inherit_type: InheritanceType
    memory_from_parent_ratio: float  # 0.0 to 1.0
    memory_from_target_ratio: float  # 0.0 to 1.0
    capability_noise_scale: float  # Mutation strength
    policy_mutation: str  # Description of policy changes
    role_prompt_update: str  # Updated role prompt
    capability_need: str = ""  # abstract capability demand decided by meta
    capability_variant: str = ""  # variant / niche under the capability


@dataclass
class BirthDeathPair:
    """A birth-death pair in the evolution decision.

    Three operating modes:
    - Default (death_only=False, birth_only=False): strict 1-for-1 replacement.
      parent_id + death_target_id + child_plan are all required.
    - death_only=True: remove death_target_id with no replacement (shrinks population by 1).
      parent_id and child_plan are ignored.
    - birth_only=True: add a new child from parent_id without killing anyone (grows population by 1).
      death_target_id is ignored.

    At most 1 pure-kill and 1 pure-birth are allowed per slow-update cycle.
    """
    parent_id: str
    death_target_id: str
    birth_reason: str
    death_reason: str
    child_plan: ChildInheritancePlan
    improvement_direction: str
    death_only: bool = False   # pure kill — shrink by 1
    birth_only: bool = False   # pure birth — grow by 1


@dataclass
class GraphEdit:
    """Graph structure edits."""
    remove_edges: List[tuple]  # List of (source, target)
    add_edges: List[tuple]  # List of (source, target)
    type_changes: List[Dict[str, Any]]  # Edge type changes
    rewire_notes: List[str]
    subgraph_rewrites: List[str]


@dataclass
class FinalSynthesis:
    """Plan for synthesizing final answer."""
    contributors: List[str]  # Which agents contribute to final answer
    strategy: str  # How to merge their outputs
    conflict_resolution: str  # How to handle disagreements
    final_answer_spec: str  # Specification of final answer format


@dataclass
class TimeControl:
    """Time scale control decisions."""
    birth_death_value: float  # Expected benefit of BD
    graph_rewire_value: float  # Expected benefit of graph rewiring
    fast_steps_next: int  # Suggested fast steps before next slow update
    trigger_birth_death: bool
    trigger_graph_rewire: bool
    trigger_next_slow_rule: Literal["fixed", "adaptive"]
    cooldown: int  # Steps to wait before next slow update
    continue_evolution: bool = True  # Whether MAS should keep running fast rounds
    stop_reason: str = ""  # Optional stop reason from meta controller


@dataclass
class AgentEvolutionFeedback:
    """Meta-LLM guidance and score for one agent."""
    agent_id: str
    score: float
    score_reason: str
    workflow_correction: str
    next_round_workflow: str


@dataclass
class MetaDecision:
    """Complete decision from meta LLM."""
    slow_update: bool
    confidence: float  # 0.0 to 1.0
    
    birth_death_pairs: List[BirthDeathPair]
    graph_edit: GraphEdit
    final_synthesis: FinalSynthesis
    time_control: TimeControl
    agent_feedback: List[AgentEvolutionFeedback]
    
    global_rationale: List[str]  # Explanation of decisions
    
    def to_json(self) -> str:
        """Convert to JSON for logging."""
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)


@dataclass
class ControlParams:
    """Control parameters for meta LLM."""
    # Population control
    n_min: int
    n_max: int
    specialization_level: Literal["low", "medium", "high"]
    
    # Evolution control
    max_birth_death_pairs: int
    max_edge_edits: int
    protected_nodes: List[str]
    critical_roles: List[str]
    
    # Time scale control
    k_bd_min: int  # Minimum steps between BD updates
    k_graph_min: int  # Minimum steps between graph updates
    
    # Constraints
    min_role_coverage: Dict[str, int]  # Minimum count for each critical role
    max_degree: int  # Maximum node degree
    preserve_connectivity: bool


@dataclass
class MetaLLMConfig:
    """Configuration for meta LLM."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 4000
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    
    # Prompt templates
    system_prompt_template: str = ""
    developer_prompt_template: str = ""
    
    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class EvolutionConfig:
    """Configuration for the evolution system."""
    enabled: bool = True
    meta_llm_config: MetaLLMConfig = field(default_factory=MetaLLMConfig)
    control_params: ControlParams = field(default_factory=ControlParams)
    
    # Timescale
    fast_steps_per_window: int = 10
    bd_check_interval: int = 50
    graph_rewire_interval: int = 200
    
    # Logging
    log_decisions: bool = True
    log_dir: str = "./meta_evolution_logs"
