"""
Configuration for meta evolution system.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel


class MetaEvolutionConfig(BaseModel):
    """Configuration for meta evolution system."""
    
    # Enable/disable
    enabled: bool = True
    
    # Meta LLM configuration
    meta_llm_model: str = "gpt-4-turbo"
    meta_llm_temperature: float = 0.7
    meta_llm_max_tokens: int = 4000
    meta_llm_api_base: Optional[str] = None
    meta_llm_api_key: Optional[str] = None
    
    # Time scale control
    fast_steps_per_window: int = 10
    bd_check_interval: int = 50
    graph_rewire_interval: int = 200
    
    # Population control
    n_min: int = 5
    n_max: int = 15
    specialization_level: str = "medium"  # low, medium, high
    
    # Evolution control
    max_birth_death_pairs: int = 2
    max_edge_edits: int = 5
    protected_nodes: List[str] = []
    critical_roles: List[str] = ["planner", "proposer", "verifier"]
    
    # Constraints
    preserve_connectivity: bool = True
    max_degree: int = 10
    
    # Logging
    log_decisions: bool = True
    log_dir: str = "./meta_evolution_logs"
    
    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600
    
    class Config:
        """Pydantic config."""
        extra = "allow"
