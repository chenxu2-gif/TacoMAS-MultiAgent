"""
Meta-LLM driven Birth-Death Graph Co-Evolution system.
"""

# Core data structures and graph/population managers can always be imported
# without the full project dependencies (langfuse, langchain, etc.).
from .schemas import (
    EdgeType,
    InheritanceType,
    AgentState,
    EdgeSummary,
    NodeSummary,
    SubgraphSummary,
    SystemSummary,
    ChildInheritancePlan,
    BirthDeathPair,
    GraphEdit,
    FinalSynthesis,
    TimeControl,
    MetaDecision,
    ControlParams,
    MetaLLMConfig,
    EvolutionConfig,
)

from .graph_manager import GraphManager
from .population_manager import PopulationManager
from .summarizer import SystemSummarizer

# MetaLLMInterface, ConstraintProjector, EvolutionExecutor and
# EvolutionController depend on tacomas.llm (which pulls in langfuse /
# langchain). Import them lazily so the module can be used without those
# packages installed.
def __getattr__(name: str):
    if name == "MetaLLMInterface":
        from .meta_llm import MetaLLMInterface
        return MetaLLMInterface
    if name in ("ConstraintProjector", "EvolutionExecutor"):
        from .executor import ConstraintProjector, EvolutionExecutor
        return locals()[name]
    if name == "EvolutionController":
        from .evolution_controller import EvolutionController
        return EvolutionController
    if name == "MetaEvolutionMASRuntime":
        from .mas_runtime import MetaEvolutionMASRuntime
        return MetaEvolutionMASRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Enums
    "EdgeType",
    "InheritanceType",
    # Data structures
    "AgentState",
    "EdgeSummary",
    "NodeSummary",
    "SubgraphSummary",
    "SystemSummary",
    "ChildInheritancePlan",
    "BirthDeathPair",
    "GraphEdit",
    "FinalSynthesis",
    "TimeControl",
    "MetaDecision",
    "ControlParams",
    "MetaLLMConfig",
    "EvolutionConfig",
    # Managers (always available)
    "GraphManager",
    "PopulationManager",
    "SystemSummarizer",
    # LLM-dependent (lazy)
    "MetaLLMInterface",
    "ConstraintProjector",
    "EvolutionExecutor",
    "EvolutionController",
    "MetaEvolutionMASRuntime",
]
