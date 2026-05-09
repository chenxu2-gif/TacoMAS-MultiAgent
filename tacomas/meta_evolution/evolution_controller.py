"""
Main evolution controller that orchestrates the entire system.
"""

import logging
from typing import Dict, List, Optional
import os
import json
from datetime import datetime
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

from tacomas.config.llm import LLMConfig
from tacomas.config.meta_evolution import MetaEvolutionConfig

from .schemas import (
    AgentState,
    ControlParams,
    MetaLLMConfig,
    SystemSummary,
    AgentEvolutionFeedback,
)
from .graph_manager import GraphManager
from .population_manager import PopulationManager
from .summarizer import SystemSummarizer
from .meta_llm import MetaLLMInterface
from .executor import ConstraintProjector, EvolutionExecutor

logger = logging.getLogger(__name__)


class EvolutionController:
    """Main controller for meta-LLM driven evolution."""
    
    def __init__(
        self,
        config: MetaEvolutionConfig,
        llm_config: LLMConfig,
    ):
        """Initialize the evolution controller.
        
        Args:
            config: Meta evolution configuration
            llm_config: LLM configuration for meta LLM
        """
        self.config = config
        self.llm_config = llm_config
        
        # Initialize managers
        self.graph = GraphManager()
        self.population = PopulationManager()
        self.summarizer = SystemSummarizer(self.graph, self.population)
        
        # Initialize meta LLM
        meta_llm_config = LLMConfig(
            model=config.meta_llm_model,
            temperature=config.meta_llm_temperature,
            api_base=config.meta_llm_api_base,
            api_key=config.meta_llm_api_key,
        )
        self.meta_llm = MetaLLMInterface(meta_llm_config)
        
        # Initialize executor
        control_params = ControlParams(
            n_min=config.n_min,
            n_max=config.n_max,
            specialization_level=config.specialization_level,
            max_birth_death_pairs=config.max_birth_death_pairs,
            max_edge_edits=config.max_edge_edits,
            protected_nodes=config.protected_nodes,
            critical_roles=config.critical_roles,
            k_bd_min=config.bd_check_interval,
            k_graph_min=config.graph_rewire_interval,
            min_role_coverage={role: 1 for role in config.critical_roles},
            max_degree=config.max_degree,
            preserve_connectivity=config.preserve_connectivity,
        )
        self.control_params = control_params
        
        self.constraint_projector = ConstraintProjector(
            self.graph, self.population, control_params
        )
        self.executor = EvolutionExecutor(self.graph, self.population)
        
        # State tracking
        self.slow_update_step = 0
        self.fast_time_step = 0
        self.last_bd_time = 0
        self.last_graph_rewire_time = 0
        self.last_decision = None
        self.latest_agent_feedback: Dict[str, AgentEvolutionFeedback] = {}
        self.next_slow_update_at = config.bd_check_interval
        self.cooldown_until = 0
        self.no_change_streak = 0
        
        # Logging
        if config.log_decisions:
            os.makedirs(config.log_dir, exist_ok=True)
            self.log_dir = config.log_dir
        else:
            self.log_dir = None
        self.log_run_id: Optional[str] = None
        self.log_dataset_id: Optional[str] = None
        self.log_instance_idx: Optional[int] = None
        
        masked_cfg = config.model_dump() if hasattr(config, "model_dump") else dict(config)
        if masked_cfg.get("meta_llm_api_key"):
            key = str(masked_cfg["meta_llm_api_key"])
            masked_cfg["meta_llm_api_key"] = f"{key[:6]}***{key[-4:]}" if len(key) > 12 else "***"
        logger.info(f"EvolutionController initialized with config: {masked_cfg}")

    def _decision_has_structural_change(self, decision: Optional["MetaDecision"]) -> bool:
        if not decision:
            return False
        bd_pairs = list(getattr(decision, "birth_death_pairs", []) or [])
        graph_edit = getattr(decision, "graph_edit", None)
        has_graph_edit = bool(
            getattr(graph_edit, "add_edges", [])
            or getattr(graph_edit, "remove_edges", [])
            or getattr(graph_edit, "type_changes", [])
            or getattr(graph_edit, "subgraph_rewrites", [])
        )
        return bool(bd_pairs) or has_graph_edit

    def _force_min_graph_change(
        self,
        decision: "MetaDecision",
        system_summary: SystemSummary,
        reason: str,
    ) -> "MetaDecision":
        """Force a minimal graph change to make evolution visible."""
        graph_edit = decision.graph_edit
        nodes = list(system_summary.node_summaries or [])
        if len(nodes) < 2:
            return decision

        existing = {(e.source, e.target) for e in system_summary.edge_summaries or []}

        def _has_edge(a: str, b: str) -> bool:
            return (a, b) in existing

        nodes_sorted = sorted(nodes, key=lambda n: float(getattr(n, "last_meta_score", 0.0)))
        best = nodes_sorted[-1]
        second = nodes_sorted[-2] if len(nodes_sorted) > 1 else nodes_sorted[-1]
        worst = nodes_sorted[0]

        max_edits = int(getattr(self.control_params, "max_edge_edits", 0) or 0)

        def _try_add(a: str, b: str) -> bool:
            if max_edits and len(graph_edit.add_edges) >= max_edits:
                return False
            if not _has_edge(a, b):
                graph_edit.add_edges.append((a, b, "evidence_flow"))
                return True
            return False

        added = (
            _try_add(best.agent_id, second.agent_id)
            or _try_add(second.agent_id, best.agent_id)
            or _try_add(worst.agent_id, best.agent_id)
            or _try_add(best.agent_id, worst.agent_id)
        )

        if not added and system_summary.edge_summaries:
            candidate = None
            for edge in system_summary.edge_summaries:
                if {edge.source, edge.target} & {best.agent_id, second.agent_id}:
                    candidate = edge
                    break
            if candidate is None:
                candidate = system_summary.edge_summaries[0]
            graph_edit.type_changes.append(
                {
                    "edge": [candidate.source, candidate.target],
                    "new_type": "bidirectional",
                }
            )
            added = True

        if added:
            graph_edit.rewire_notes.append(f"forced_min_change: {reason}")
            decision.time_control.trigger_graph_rewire = True
            decision.time_control.graph_rewire_value = max(
                0.6,
                float(getattr(decision.time_control, "graph_rewire_value", 0.0)),
            )
        return decision
    
    def add_agent(self, agent: AgentState) -> None:
        """Add an agent to the system."""
        self.population.add_agent(agent)
        self.graph.add_node(agent.agent_id, agent.role)
        logger.info(f"Added agent {agent.agent_id} with role {agent.role}")
    
    def add_edge(self, source: str, target: str, edge_type: str = "directed") -> None:
        """Add an edge between two agents."""
        from .schemas import EdgeType
        self.graph.add_edge(source, target, EdgeType(edge_type))
        logger.info(f"Added edge ({source}, {target}) of type {edge_type}")
    
    def update_agent_state(self, agent: AgentState) -> None:
        """Update an agent's state."""
        self.population.update_agent(agent)
    
    def should_trigger_slow_update(self) -> bool:
        """Check if slow update should be triggered."""
        if not self.config.enabled:
            return False

        # Respect cooldown
        if self.fast_time_step < self.cooldown_until:
            return False

        # Adaptive schedule target
        return self.fast_time_step >= self.next_slow_update_at
    
    def execute_slow_update(
        self,
        task_description: str = "",
        current_answer_quality: float = 0.0,
        protected_graph_paths: dict | None = None,
    ) -> dict:
        """Execute a slow-time update.
        
        Args:
            task_description: Description of the task
            current_answer_quality: Quality of current system answer
        """
        if not self.config.enabled:
            return {"decision": None, "system_summary": None}
        
        logger.info(f"Executing slow update at step {self.fast_time_step}")
        
        # Generate system summary
        system_summary = self.summarizer.summarize(
            slow_update_step=self.slow_update_step,
            fast_time_window=(
                max(0, self.fast_time_step - self.config.fast_steps_per_window),
                self.fast_time_step,
            ),
            task_objective=task_description,
            current_answer_quality=current_answer_quality,
        )
        
        # Call meta LLM
        decision = self.meta_llm.call_meta_llm(
            system_summary,
            self.control_params,
            task_description,
        )
        meta_llm_request = getattr(self.meta_llm, "last_request_context", None)
        self.latest_agent_feedback = {
            fb.agent_id: fb for fb in getattr(decision, "agent_feedback", []) or [] if fb.agent_id
        }
        
        # Project decision to feasible space
        decision = self.constraint_projector.project_decision(decision)

        # Second-chance gated-greedy fallback (post-projection).
        # Rationale: we observed runs where meta saw positive candidate Delta(U_t) but still returned
        # an empty structural decision; applying fallback again here guarantees the slow update
        # does not silently remain empty when an executable positive-gain candidate exists.
        try:
            decision = self.meta_llm._apply_gated_greedy_fallback(  # noqa: SLF001
                decision=decision,
                system_summary=system_summary,
                control_params=self.control_params,
            )
            decision = self.constraint_projector.project_decision(decision)
        except Exception as e:
            logger.warning(f"Post-projection gated-greedy fallback failed: {e}")
        self.last_decision = decision

        # If we are on a slow cadence but quality is still low, ensure at least one graph edit
        # survives the projection to keep exploration pressure.
        try:
            if current_answer_quality < 0.6:
                graph_edit = decision.graph_edit
                has_graph_edit = bool(
                    graph_edit.add_edges
                    or graph_edit.remove_edges
                    or graph_edit.type_changes
                    or graph_edit.subgraph_rewrites
                )
                if not has_graph_edit:
                    nodes = list(system_summary.node_summaries or [])
                    if len(nodes) >= 2:
                        nodes_sorted = sorted(nodes, key=lambda n: float(getattr(n, "last_meta_score", 0.0)))
                        worst = nodes_sorted[0]
                        best = nodes_sorted[-1]
                        second_best = nodes_sorted[-2] if len(nodes_sorted) >= 2 else None
                        existing = {(e.source, e.target) for e in system_summary.edge_summaries}

                        def _has_any_edge(a: str, b: str) -> bool:
                            return (a, b) in existing or (b, a) in existing

                        if second_best and not _has_any_edge(best.agent_id, second_best.agent_id):
                            graph_edit.add_edges.append((best.agent_id, second_best.agent_id, "directed"))
                        elif not _has_any_edge(worst.agent_id, best.agent_id):
                            graph_edit.add_edges.append((worst.agent_id, best.agent_id, "directed"))
                        elif second_best and not _has_any_edge(worst.agent_id, second_best.agent_id):
                            graph_edit.add_edges.append((worst.agent_id, second_best.agent_id, "directed"))

                        decision.time_control.trigger_graph_rewire = True
                        decision.time_control.graph_rewire_value = max(
                            0.5,
                            float(getattr(decision.time_control, "graph_rewire_value", 0.0)),
                        )
                        decision = self.constraint_projector.project_decision(decision)
        except Exception as e:
            logger.warning(f"Post-projection low-quality graph rewire fallback failed: {e}")

        # Anti-stagnation: enforce a minimal structural change if meta returns none.
        try:
            force_no_change_streak = int(os.getenv("EVOLVE_FORCE_CHANGE_NO_CHANGE_STREAK", "1"))
            force_until_step = int(os.getenv("EVOLVE_FORCE_CHANGE_UNTIL_SLOW_STEP", "2"))
            score_trend = float(getattr(system_summary, "score_trend", 0.0))
            learning_eff = float(getattr(system_summary, "learning_efficiency", 0.0))
            is_stagnant = abs(score_trend) < 0.02 and learning_eff <= 0.05

            has_structural_change = self._decision_has_structural_change(decision)
            if has_structural_change:
                self.no_change_streak = 0
            else:
                self.no_change_streak += 1

            should_force = (
                (self.no_change_streak >= max(1, force_no_change_streak))
                or (self.slow_update_step < force_until_step)
                or is_stagnant
            )
            if should_force and not has_structural_change:
                decision = self._force_min_graph_change(
                    decision,
                    system_summary,
                    reason=(
                        "no_structural_change"
                        if not is_stagnant
                        else "stagnant_scores"
                    ),
                )
                decision = self.constraint_projector.project_decision(decision)
                self.no_change_streak = 0
        except Exception as e:
            logger.warning(f"Anti-stagnation forced-change fallback failed: {e}")

        if protected_graph_paths:
            decision = self._apply_high_score_path_protection(decision, protected_graph_paths)
        
        # Execute decision
        self.executor.execute_decision(decision, self.fast_time_step)
        
        # Update time tracking
        if decision.time_control.trigger_birth_death:
            self.last_bd_time = self.fast_time_step
        
        if decision.time_control.trigger_graph_rewire:
            self.last_graph_rewire_time = self.fast_time_step
        
        # Fixed schedule: always follow configured bd_check_interval.
        # This guarantees predictable cadence (e.g., every 2 fast rounds).
        applied_fast_steps_next = max(1, int(self.config.bd_check_interval))
        self.next_slow_update_at = self.fast_time_step + applied_fast_steps_next

        # Apply cooldown if requested
        if decision.time_control.cooldown > 0:
            self.cooldown_until = self.fast_time_step + int(decision.time_control.cooldown)
        
        self.slow_update_step += 1
        
        # Log decision
        if self.log_dir:
            self._log_decision(decision, system_summary, meta_llm_request=meta_llm_request)

        return {
            "decision": decision,
            "system_summary": system_summary,
            "meta_llm_request": meta_llm_request,
            "applied_fast_steps_next": applied_fast_steps_next,
        }

    def _apply_high_score_path_protection(self, decision, protected_graph_paths: dict):
        """Soft-protect high-scoring answer paths.

        Meta may delete a protected edge only when it provides an explicit
        protected_edge_override_reason in rewire_notes/subgraph_rewrites and
        proposes an alternative main-path edge among protected agents.
        """
        try:
            graph_edit = getattr(decision, "graph_edit", None)
            if graph_edit is None:
                return decision
            protected_edges = {
                (str(edge.get("source")), str(edge.get("target")))
                for edge in protected_graph_paths.get("protected_edges", []) or []
                if edge.get("source") and edge.get("target")
            }
            protected_agents = {
                str(agent_id)
                for agent_id in protected_graph_paths.get("protected_agents", []) or []
                if agent_id
            }
            protected_types = set(
                protected_graph_paths.get(
                    "protected_edge_types",
                    ["evidence_flow", "verification_flow", "computation_flow", "bidirectional"],
                )
                or []
            )
            notes = list(getattr(graph_edit, "rewire_notes", []) or [])
            subgraph_rewrites = list(getattr(graph_edit, "subgraph_rewrites", []) or [])

            def _has_override_reason(src: str, tgt: str) -> bool:
                haystack = "\n".join(str(item) for item in notes + subgraph_rewrites).lower()
                if "protected_edge_override_reason" not in haystack:
                    return False
                # Prefer edge-specific reasons, but allow a global override
                # note if it explicitly mentions alternative backbone/path.
                edge_mentions = (
                    f"{src}->{tgt}".lower() in haystack
                    or f"{src}, {tgt}".lower() in haystack
                    or f"{src} {tgt}".lower() in haystack
                )
                alternative_mentions = any(
                    marker in haystack
                    for marker in [
                        "alternative_main_path",
                        "alternative backbone",
                        "replacement path",
                        "new main path",
                        "替代主路径",
                    ]
                )
                return edge_mentions or alternative_mentions

            def _has_alternative_main_path(src: str, tgt: str) -> bool:
                for edge in list(getattr(graph_edit, "add_edges", []) or []):
                    if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                        continue
                    add_src, add_tgt = str(edge[0]), str(edge[1])
                    add_type = str(edge[2] if len(edge) >= 3 else "directed")
                    if add_type not in protected_types:
                        continue
                    if add_src not in protected_agents and add_tgt not in protected_agents:
                        continue
                    # A replacement path should preserve at least one endpoint
                    # of the removed protected edge, or connect two protected
                    # backbone agents.
                    if (
                        add_src in {src, tgt}
                        or add_tgt in {src, tgt}
                        or (add_src in protected_agents and add_tgt in protected_agents)
                    ):
                        return True
                for change in list(getattr(graph_edit, "type_changes", []) or []):
                    edge = change.get("edge", []) if isinstance(change, dict) else []
                    new_type = str(change.get("new_type", "") or "") if isinstance(change, dict) else ""
                    if len(edge) != 2 or new_type not in protected_types:
                        continue
                    ch_src, ch_tgt = str(edge[0]), str(edge[1])
                    if ch_src in protected_agents and ch_tgt in protected_agents:
                        return True
                return False

            kept_remove = []
            blocked_remove = []
            allowed_override_remove = []
            for edge in list(getattr(graph_edit, "remove_edges", []) or []):
                if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                    continue
                src, tgt = str(edge[0]), str(edge[1])
                edge_type = self.graph.get_edge_type(src, tgt)
                edge_type_value = getattr(edge_type, "value", str(edge_type or ""))
                is_protected_edge = (src, tgt) in protected_edges
                is_protected_backbone = (
                    src in protected_agents
                    and tgt in protected_agents
                    and edge_type_value in protected_types
                )
                if is_protected_edge or is_protected_backbone:
                    if _has_override_reason(src, tgt) and _has_alternative_main_path(src, tgt):
                        allowed_override_remove.append((src, tgt, edge_type_value))
                        kept_remove.append(edge)
                        continue
                    blocked_remove.append((src, tgt, edge_type_value))
                    continue
                kept_remove.append(edge)
            if blocked_remove:
                notes.append(f"high_score_path_protection_blocked_remove={blocked_remove}")
            if allowed_override_remove:
                notes.append(f"high_score_path_protection_allowed_override_remove={allowed_override_remove}")
            graph_edit.remove_edges = kept_remove

            kept_type_changes = []
            blocked_type_changes = []
            allowed_override_type_changes = []
            for change in list(getattr(graph_edit, "type_changes", []) or []):
                edge = change.get("edge", []) if isinstance(change, dict) else []
                if len(edge) != 2:
                    kept_type_changes.append(change)
                    continue
                src, tgt = str(edge[0]), str(edge[1])
                current_type = self.graph.get_edge_type(src, tgt)
                current_value = getattr(current_type, "value", str(current_type or ""))
                new_type = str(change.get("new_type", "") or "")
                if (
                    ((src, tgt) in protected_edges or (src in protected_agents and tgt in protected_agents))
                    and current_value in protected_types
                    and new_type == "reflection_feedback"
                ):
                    if _has_override_reason(src, tgt) and _has_alternative_main_path(src, tgt):
                        allowed_override_type_changes.append((src, tgt, current_value, new_type))
                        kept_type_changes.append(change)
                        continue
                    blocked_type_changes.append((src, tgt, current_value, new_type))
                    continue
                kept_type_changes.append(change)
            if blocked_type_changes:
                notes.append(f"high_score_path_protection_blocked_type_change={blocked_type_changes}")
            if allowed_override_type_changes:
                notes.append(f"high_score_path_protection_allowed_override_type_change={allowed_override_type_changes}")
            graph_edit.type_changes = kept_type_changes

            if not graph_edit.remove_edges and not graph_edit.add_edges and not graph_edit.type_changes:
                decision.time_control.trigger_graph_rewire = False
            graph_edit.rewire_notes = notes
        except Exception as exc:
            logger.warning("High-score path protection failed: %s", exc)
        return decision

    def execute_meta_feedback_only(
        self,
        task_description: str = "",
        current_answer_quality: float = 0.0,
    ) -> dict:
        """Call meta-LLM to produce per-agent feedback only (no structural edits, no executor side-effects).

        This is intentionally optional because it can be expensive to run every fast round.
        """
        if not self.config.enabled:
            return {"decision": None, "system_summary": None}

        # Use current slow_update_step for context, but do not advance slow-time counters here.
        system_summary = self.summarizer.summarize(
            slow_update_step=self.slow_update_step,
            fast_time_window=(
                max(0, self.fast_time_step - self.config.fast_steps_per_window),
                self.fast_time_step,
            ),
            task_objective=task_description,
            current_answer_quality=current_answer_quality,
        )

        decision = self.meta_llm.call_meta_llm(
            system_summary,
            self.control_params,
            task_description,
        )
        meta_llm_request = getattr(self.meta_llm, "last_request_context", None)

        # Hard-disable any structural actions for feedback-only mode.
        try:
            decision.birth_death_pairs = []
            decision.graph_edit.remove_edges = []
            decision.graph_edit.add_edges = []
            decision.graph_edit.type_changes = []
            decision.graph_edit.rewire_notes = []
            decision.graph_edit.subgraph_rewrites = []
            decision.slow_update = False
            decision.time_control.trigger_birth_death = False
            decision.time_control.trigger_graph_rewire = False
        except Exception:
            # If schema differs, keep best-effort feedback; structural actions won't be executed anyway.
            pass

        self.latest_agent_feedback = {
            fb.agent_id: fb for fb in getattr(decision, "agent_feedback", []) or [] if fb.agent_id
        }

        # Optional logging (separate from slow-update logs).
        if self.log_dir:
            try:
                timestamp = datetime.now().isoformat()
                log_data = {
                    "event_type": "meta_feedback_fast",
                    "timestamp": timestamp,
                    "fast_time_step": self.fast_time_step,
                    "slow_update_step": self.slow_update_step,
                    "decision": {
                        "confidence": getattr(decision, "confidence", 0.0),
                        "slow_update": getattr(decision, "slow_update", False),
                        "agent_feedback_count": len(getattr(decision, "agent_feedback", []) or []),
                    },
                    "system_summary": self._to_jsonable(system_summary),
                    "meta_llm_request": self._to_jsonable(meta_llm_request) if meta_llm_request is not None else None,
                }
                self._write_log_event(log_data)
            except Exception as e:
                logger.warning(f"Error logging meta feedback-only decision: {e}")

        return {
            "decision": decision,
            "system_summary": system_summary,
            "meta_llm_request": meta_llm_request,
        }
    
    def step_fast_time(self) -> None:
        """Advance fast time step."""
        self.fast_time_step += 1
    
    def get_system_state(self) -> Dict:
        """Get current system state."""
        return {
            "fast_time_step": self.fast_time_step,
            "slow_update_step": self.slow_update_step,
            "num_agents": self.population.get_population_size(),
            "num_edges": len(self.graph.get_all_edges()),
            "graph_density": self.graph.get_density(),
            "modularity": self.graph.get_modularity(),
            "role_distribution": self.population.get_role_distribution(),
        }
    
    def get_agent_ids(self) -> List[str]:
        """Get all agent IDs."""
        return self.graph.get_all_nodes()
    
    def get_agent_neighbors(self, agent_id: str) -> List[str]:
        """Get neighbors of an agent."""
        return self.graph.get_neighbors(agent_id)
    
    def _log_decision(self, decision, system_summary, meta_llm_request=None) -> None:
        """Log evolutionary decision."""
        timestamp = datetime.now().isoformat()
        log_data = {
            "event_type": "decision",
            "timestamp": timestamp,
            "slow_update_step": self.slow_update_step,
            "fast_time_step": self.fast_time_step,
            "decision": {
                "confidence": decision.confidence,
                "slow_update": decision.slow_update,
                "birth_death_pairs": len(decision.birth_death_pairs),
                "agent_feedback_count": len(getattr(decision, "agent_feedback", []) or []),
                "graph_edits": {
                    "remove_edges": len(decision.graph_edit.remove_edges),
                    "add_edges": len(decision.graph_edit.add_edges),
                    "type_changes": len(decision.graph_edit.type_changes),
                },
                "time_control": {
                    "birth_death_value": decision.time_control.birth_death_value,
                    "graph_rewire_value": decision.time_control.graph_rewire_value,
                    "trigger_birth_death": decision.time_control.trigger_birth_death,
                    "trigger_graph_rewire": decision.time_control.trigger_graph_rewire,
                    "continue_evolution": decision.time_control.continue_evolution,
                    "stop_reason": decision.time_control.stop_reason,
                },
                "rationale": decision.global_rationale,
                "agent_feedback": [
                    {
                        "agent_id": fb.agent_id,
                        "score": fb.score,
                        "score_reason": fb.score_reason,
                        "workflow_correction": fb.workflow_correction,
                        "next_round_workflow": fb.next_round_workflow,
                    }
                    for fb in (getattr(decision, "agent_feedback", []) or [])
                ],
            },
            "system_state": {
                "num_agents": system_summary.total_agents,
                "num_edges": system_summary.total_edges,
                "mean_score": system_summary.mean_score,
                "score_trend": system_summary.score_trend,
                "graph_density": system_summary.graph_density,
                "modularity": system_summary.modularity,
                "is_stable": system_summary.is_stable,
            },
            "system_summary": self._to_jsonable(system_summary),
            "meta_llm_request": self._to_jsonable(meta_llm_request) if meta_llm_request is not None else None,
        }
        self._write_log_event(log_data)

    def set_logging_context(
        self,
        *,
        run_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        instance_idx: Optional[int] = None,
    ) -> None:
        self.log_run_id = run_id
        self.log_dataset_id = dataset_id
        self.log_instance_idx = instance_idx

    def _current_log_path(self) -> Optional[Path]:
        if not self.log_dir:
            return None
        if self.log_run_id is not None and self.log_dataset_id is not None and self.log_instance_idx is not None:
            base = Path(self.log_dir) / self.log_run_id / self.log_dataset_id
            base.mkdir(parents=True, exist_ok=True)
            return base / f"instance_{self.log_instance_idx}.jsonl"
        return None

    def _write_log_event(self, payload: Dict) -> None:
        try:
            log_path = self._current_log_path()
            if log_path is not None:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                logger.debug("Appended meta evolution event to %s", log_path)
                return

            timestamp = str(payload.get("timestamp", datetime.now().isoformat())).replace(":", "-")
            event_type = str(payload.get("event_type", "event"))
            step = payload.get("slow_update_step", self.slow_update_step)
            log_file = os.path.join(self.log_dir, f"{event_type}_{step}_{timestamp}.json")
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info("Logged decision to %s", log_file)
        except Exception as e:
            logger.error("Error logging decision: %s", e)

    def _to_jsonable(self, value):
        """Convert dataclass/enum-rich structures into JSON-safe objects."""
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_jsonable(v) for v in value]
        return value
