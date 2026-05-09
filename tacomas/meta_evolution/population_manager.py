"""
Population manager for agent lifecycle and inheritance.
"""

from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import uuid
import re

from .schemas import (
    AgentState,
    ChildInheritancePlan,
    InheritanceType,
)

_VALID_ROLE_NAMES = {
    "planner", "searcher", "researcher", "analyst", "verifier", "auditor",
    "schema_verifier", "calculator", "forecaster", "reflector", "synthesizer",
}


class PopulationManager:
    """Manages agent population and lifecycle."""
    
    def __init__(self):
        """Initialize the population manager."""
        self.agents: Dict[str, AgentState] = {}
        self.agent_history: Dict[str, List[AgentState]] = {}  # Track history for each agent
        self.lineage: Dict[str, Optional[str]] = {}  # agent_id -> parent_id
        self.children: Dict[str, List[str]] = {}  # parent_id -> list of children
        self.creation_time: Dict[str, int] = {}
        self.death_time: Dict[str, Optional[int]] = {}
    
    def add_agent(self, agent: AgentState) -> None:
        """Add an agent to the population."""
        self.agents[agent.agent_id] = agent
        self.agent_history[agent.agent_id] = [deepcopy(agent)]
        self.lineage[agent.agent_id] = None
        self.children[agent.agent_id] = []
        self.creation_time[agent.agent_id] = agent.creation_time
        self.death_time[agent.agent_id] = None

    def _split_role_variant(self, role_prompt_update: str) -> Tuple[str, str]:
        """Parse role + optional prompt variant from a meta-specified string."""
        raw = (role_prompt_update or "").strip()
        if not raw:
            return "", ""
        for sep in ("|", ":", "#", "@"):
            if sep in raw:
                left, right = raw.split(sep, 1)
                role = left.strip().lower()
                variant_tag = right.strip().lower()
                if variant_tag.startswith("alt"):
                    suffix = variant_tag.replace("alt", "")
                    key = f"{role}_alt{suffix}" if suffix else f"{role}_alt"
                    if role in _VALID_ROLE_NAMES:
                        return role, key
                if role in _VALID_ROLE_NAMES:
                    return role, ""
        lowered = raw.strip().lower()
        if lowered in _VALID_ROLE_NAMES:
            return lowered, ""
        role_match = None
        for role in sorted(_VALID_ROLE_NAMES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(role)}\b", lowered):
                role_match = role
                break
        return (role_match or "", "")

    def _resolve_capability_to_role(self, capability_need: str, fallback_role: str) -> str:
        capability = str(capability_need or "").strip().lower()
        mapping = {
            "broad_retrieval": "searcher",
            "targeted_extraction": "researcher",
            "verification": "verifier",
            "schema_validation": "schema_verifier",
            "computation": "calculator",
            "synthesis": "synthesizer",
            "conflict_resolution": "verifier",
            "coordination": "reflector",
        }
        resolved = mapping.get(capability, "")
        if resolved in _VALID_ROLE_NAMES:
            return resolved
        fallback = str(fallback_role or "").strip().lower()
        return fallback if fallback in _VALID_ROLE_NAMES else "searcher"
    
    def remove_agent(self, agent_id: str) -> Optional[AgentState]:
        """Remove an agent from the population."""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents.pop(agent_id)
        self.death_time[agent_id] = agent.last_update_time
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def update_agent(self, agent: AgentState) -> None:
        """Update an agent's state."""
        if agent.agent_id not in self.agents:
            self.add_agent(agent)
        else:
            self.agents[agent.agent_id] = agent
            self.agent_history[agent.agent_id].append(deepcopy(agent))
    
    def get_all_agents(self) -> List[AgentState]:
        """Get all agents in the population."""
        return list(self.agents.values())
    
    def get_agents_by_role(self, role: str) -> List[AgentState]:
        """Get all agents with a specific role."""
        return [a for a in self.agents.values() if a.role == role]
    
    def get_population_size(self) -> int:
        """Get current population size."""
        return len(self.agents)
    
    def spawn_child(
        self,
        parent_id: str,
        death_target_id: str,
        inheritance_plan: ChildInheritancePlan,
        current_time: int,
    ) -> str:
        """Spawn a child agent to replace a dead one.
        
        Args:
            parent_id: ID of the parent agent
            death_target_id: ID of the agent being replaced
            inheritance_plan: Plan for inheritance
            current_time: Current time step
        
        Returns:
            ID of the new child agent
        """
        if parent_id not in self.agents:
            raise ValueError(f"Parent agent {parent_id} not found")

        parent = self.agents[parent_id]
        target = self.agents.get(death_target_id)

        # Generate new agent ID
        child_id = f"{parent_id}_child_{uuid.uuid4().hex[:8]}"

        # Determine the child's intended role (post-mutation or inherited from parent).
        intended_role_raw = str(inheritance_plan.role_prompt_update or parent.role or "").strip()
        intended_role, prompt_variant = self._split_role_variant(intended_role_raw)
        intended_role = self._resolve_capability_to_role(
            getattr(inheritance_plan, "capability_need", ""),
            intended_role or parent.role,
        )
        if not intended_role:
            intended_role = parent.role
        if not prompt_variant:
            capability_variant = str(getattr(inheritance_plan, "capability_variant", "") or "").strip().lower()
            if capability_variant:
                safe_variant = re.sub(r"[^a-z0-9_]+", "_", capability_variant).strip("_")
                if safe_variant:
                    prompt_variant = f"{intended_role}_{safe_variant}"

        # Create child based on (possibly updated) inheritance plan
        if inheritance_plan.inherit_type == InheritanceType.PURE_CLONE:
            child = self._create_pure_clone(parent, child_id, current_time)
        elif inheritance_plan.inherit_type == InheritanceType.MUTATE_ROLE:
            child = self._create_role_mutant(
                parent, child_id, current_time,
                intended_role,
                policy_override=(
                    (
                        f"[capability_need={getattr(inheritance_plan, 'capability_need', '')}; "
                        f"capability_variant={getattr(inheritance_plan, 'capability_variant', '')}] "
                        f"{getattr(inheritance_plan, 'policy_mutation', '')}"
                    ).strip()
                    or None
                ),
            )
        elif inheritance_plan.inherit_type == InheritanceType.HYBRID:
            child = self._create_hybrid_child(
                parent, target, child_id, current_time, inheritance_plan
            )
        else:
            raise ValueError(f"Unknown inheritance type: {inheritance_plan.inherit_type}")

        if prompt_variant:
            child.prompt_variant = prompt_variant

        # Add to population first (add_agent initialises lineage to None)
        self.add_agent(child)

        # Record lineage AFTER add_agent so it is not overwritten
        self.lineage[child_id] = parent_id
        self.children[parent_id].append(child_id)

        return child_id
    
    def _create_pure_clone(
        self, parent: AgentState, child_id: str, current_time: int
    ) -> AgentState:
        """Create a pure clone of the parent."""
        child = deepcopy(parent)
        child.agent_id = child_id
        child.creation_time = current_time
        child.last_update_time = current_time
        child.recent_scores = []  # Reset scores
        child.score_trend = 0.0
        return child
    
    def _create_role_mutant(
        self,
        parent: AgentState,
        child_id: str,
        current_time: int,
        role_prompt_update: str,
        policy_override: Optional[str] = None,
    ) -> AgentState:
        """Create a role mutant of the parent."""
        child = deepcopy(parent)
        child.agent_id = child_id
        child.creation_time = current_time
        child.last_update_time = current_time
        # Mutate both the declared role (used for routing/metrics) and policy prompt.
        # Some callers pass canonical role strings (e.g. "searcher"); keep it simple here.
        child.role = str(role_prompt_update or child.role)
        child.policy = policy_override or role_prompt_update  # Update policy/role prompt
        child.recent_scores = []
        child.score_trend = 0.0
        return child
    
    def _create_hybrid_child(
        self,
        parent: AgentState,
        target: Optional[AgentState],
        child_id: str,
        current_time: int,
        inheritance_plan: ChildInheritancePlan,
    ) -> AgentState:
        """Create a hybrid child combining parent and target."""
        child = deepcopy(parent)
        child.agent_id = child_id
        child.creation_time = current_time
        child.last_update_time = current_time
        
        if target is not None:
            # Blend memory
            if inheritance_plan.memory_from_parent_ratio + inheritance_plan.memory_from_target_ratio > 0:
                total_ratio = (
                    inheritance_plan.memory_from_parent_ratio
                    + inheritance_plan.memory_from_target_ratio
                )
                parent_weight = (
                    inheritance_plan.memory_from_parent_ratio / total_ratio
                )
                target_weight = (
                    inheritance_plan.memory_from_target_ratio / total_ratio
                )
                child.memory_summary = (
                    f"[Hybrid: {parent_weight:.1%} from parent, "
                    f"{target_weight:.1%} from target] "
                    f"Parent: {parent.memory_summary[:100]}... "
                    f"Target: {target.memory_summary[:100]}..."
                )
        
        # Apply capability noise
        if inheritance_plan.capability_noise_scale > 0:
            import random
            for key in child.capability:
                noise = random.gauss(0, inheritance_plan.capability_noise_scale)
                child.capability[key] = max(0, min(1, child.capability[key] + noise))
        
        # Update policy if specified
        if inheritance_plan.policy_mutation:
            child.policy = inheritance_plan.policy_mutation
        
        child.recent_scores = []
        child.score_trend = 0.0
        
        return child
    
    def get_agent_history(self, agent_id: str) -> List[AgentState]:
        """Get the history of an agent."""
        return self.agent_history.get(agent_id, [])
    
    def get_lineage_tree(self, agent_id: str) -> Dict:
        """Get the lineage tree of an agent."""
        tree = {
            "agent_id": agent_id,
            "parent": self.lineage.get(agent_id),
            "children": self.children.get(agent_id, []),
        }
        return tree
    
    def get_all_descendants(self, agent_id: str) -> List[str]:
        """Get all descendants of an agent."""
        descendants = []
        to_visit = [agent_id]
        
        while to_visit:
            current = to_visit.pop(0)
            children = self.children.get(current, [])
            descendants.extend(children)
            to_visit.extend(children)
        
        return descendants
    
    def get_all_ancestors(self, agent_id: str) -> List[str]:
        """Get all ancestors of an agent."""
        ancestors = []
        current = agent_id
        
        while current is not None:
            parent = self.lineage.get(current)
            if parent is None:
                break
            ancestors.append(parent)
            current = parent
        
        return ancestors
    
    def get_role_distribution(self) -> Dict[str, int]:
        """Get the distribution of roles in the population."""
        distribution = {}
        for agent in self.agents.values():
            distribution[agent.role] = distribution.get(agent.role, 0) + 1
        return distribution
    
    def get_high_performers(self, top_k: int = 3) -> List[AgentState]:
        """Get the top-k performing agents."""
        agents_with_scores = [
            (a, a.recent_scores[-1] if a.recent_scores else 0)
            for a in self.agents.values()
        ]
        agents_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [a for a, _ in agents_with_scores[:top_k]]
    
    def get_low_performers(self, bottom_k: int = 3) -> List[AgentState]:
        """Get the bottom-k performing agents."""
        agents_with_scores = [
            (a, a.recent_scores[-1] if a.recent_scores else 0)
            for a in self.agents.values()
        ]
        agents_with_scores.sort(key=lambda x: x[1])
        return [a for a, _ in agents_with_scores[:bottom_k]]
    
    def get_redundant_agents(self, threshold: float = 0.8) -> List[Tuple[str, str]]:
        """Get pairs of redundant agents.
        
        Args:
            threshold: Redundancy score threshold
        
        Returns:
            List of (agent_id1, agent_id_2) pairs
        """
        redundant_pairs = []
        agents = list(self.agents.values())
        
        for i, a1 in enumerate(agents):
            for a2 in agents[i + 1 :]:
                # Simple heuristic: same role and similar recent scores
                if a1.role == a2.role and a1.redundancy_score >= threshold:
                    redundant_pairs.append((a1.agent_id, a2.agent_id))
        
        return redundant_pairs
    
    def get_critical_agents(self, bridge_threshold: float = 0.5) -> List[AgentState]:
        """Get agents that are critical for graph connectivity."""
        return [
            a
            for a in self.agents.values()
            if a.bridge_value >= bridge_threshold
        ]
    
    def to_dict(self) -> Dict:
        """Convert population to dictionary."""
        return {
            "agents": {
                aid: {
                    "role": a.role,
                    "recent_scores": a.recent_scores,
                    "score_trend": a.score_trend,
                }
                for aid, a in self.agents.items()
            },
            "lineage": self.lineage,
            "children": self.children,
        }
