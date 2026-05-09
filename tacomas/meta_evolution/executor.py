"""
Constraint projection and evolution executor.
"""

from typing import List, Tuple, Optional
import logging
import os
import re

from .schemas import (
    MetaDecision,
    ControlParams,
    GraphEdit,
    BirthDeathPair,
    EdgeType,
)
from .graph_manager import GraphManager
from .population_manager import PopulationManager

logger = logging.getLogger(__name__)


_ALLOWED_EDGE_TYPE_VALUES = {edge.value for edge in EdgeType}


def _sanitize_role_update(raw_value: str, allowed_roles: set[str]) -> str:
    raw = str(raw_value or "").strip().lower()
    if not raw:
        return ""
    base = raw.split("|", 1)[0].strip()
    if base in allowed_roles:
        return raw
    for role in sorted(allowed_roles, key=len, reverse=True):
        if re.search(rf"\b{re.escape(role)}\b", raw):
            suffix = ""
            if "|" in raw:
                maybe_suffix = raw.split("|", 1)[1].strip()
                if maybe_suffix.startswith("alt"):
                    suffix = f"|{maybe_suffix}"
            return f"{role}{suffix}"
    return ""


class ConstraintProjector:
    """Projects meta LLM decisions to feasible space."""
    
    def __init__(
        self,
        graph_manager: GraphManager,
        population_manager: PopulationManager,
        control_params: ControlParams,
    ):
        """Initialize the constraint projector."""
        self.graph = graph_manager
        self.population = population_manager
        self.control_params = control_params
    
    def project_decision(self, decision: MetaDecision) -> MetaDecision:
        """Project decision to feasible space."""
        relax_constraints = os.environ.get("META_RELAX_CONSTRAINTS", "0") == "1"
        allowed_roles = self._load_allowed_roles()

        # Check population size constraints
        if not relax_constraints:
            decision = self._project_population_size(decision)
        
        # Check birth-death pairs validity
        decision = self._project_birth_death_pairs(
            decision,
            relax_constraints=relax_constraints,
            allowed_roles=allowed_roles,
        )
        
        # Check graph edit validity
        decision = self._project_graph_edit(decision, relax_constraints=relax_constraints)
        
        # Check role coverage
        if not relax_constraints:
            decision = self._project_role_coverage(decision)
        
        return decision

    def _load_allowed_roles(self) -> set[str]:
        raw = os.environ.get("ALLOWED_ROLES", "").strip()
        if not raw:
            return {"planner", "searcher", "verifier", "calculator", "reflector"}
        roles = {r.strip().lower() for r in raw.split(",") if r.strip()}
        return roles or {"planner", "searcher", "verifier", "calculator", "reflector"}

    def _search_capable_roles(self) -> set[str]:
        return {
            "searcher",
            "researcher",
            "analyst",
            "curator",
            "verifier",
            "auditor",
            "schema_verifier",
            "critic",
            "worker",
            "subagent",
        }

    def _is_search_capable_role(self, role: str) -> bool:
        return str(role or "").strip().lower() in self._search_capable_roles()

    def _memory_has_concrete_evidence(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        lowered = raw.lower()
        failure_markers = [
            "no evidence",
            "not found",
            "unable to",
            "cannot answer",
            "missing",
            "insufficient",
            "failed to retrieve",
            "无法",
            "没有证据",
            "未找到",
            "缺失",
        ]
        source_markers = [
            "primary_filing",
            "source",
            "document",
            "filing",
            "table",
            "snippet",
            "sec.gov",
            "http://",
            "https://",
            "www.",
        ]
        has_numeric_signal = any(ch.isdigit() for ch in raw)
        has_source_signal = any(marker in lowered for marker in source_markers)
        if any(marker in lowered for marker in failure_markers) and not (has_numeric_signal or has_source_signal):
            return False
        return has_numeric_signal or has_source_signal

    def _estimate_retrieval_bottleneck(self) -> bool:
        agents = list(self.population.get_all_agents() or [])
        if not agents:
            return False
        search_capable_agents = [agent for agent in agents if self._is_search_capable_role(getattr(agent, "role", ""))]
        if not search_capable_agents:
            return False

        failure_mode_hits = 0
        no_evidence_like_hits = 0
        concrete_upstream_memory = 0
        for agent in search_capable_agents:
            failure_modes = {str(item or "").strip().lower() for item in getattr(agent, "failure_modes", []) or []}
            if failure_modes & {"missing_key", "no_evidence", "not_finished", "no_reconciliation"}:
                failure_mode_hits += 1
            memory = str(getattr(agent, "memory_summary", "") or "")
            lowered = memory.lower()
            if any(tok in lowered for tok in ["no evidence", "not found", "missing", "insufficient", "没有证据", "未找到", "缺失"]):
                no_evidence_like_hits += 1
            if self._memory_has_concrete_evidence(memory):
                concrete_upstream_memory += 1

        if concrete_upstream_memory == 0 and (failure_mode_hits > 0 or no_evidence_like_hits > 0):
            return True
        if concrete_upstream_memory <= 1 and (failure_mode_hits + no_evidence_like_hits) >= 2:
            return True
        return False

    def _would_reduce_search_capacity_too_far(
        self,
        pair: BirthDeathPair,
        allowed_roles: set[str],
    ) -> bool:
        death_only = getattr(pair, "death_only", False)
        birth_only = getattr(pair, "birth_only", False)
        if birth_only and not death_only:
            return False

        death_agent = self.population.get_agent(getattr(pair, "death_target_id", ""))
        if death_agent is None or not self._is_search_capable_role(getattr(death_agent, "role", "")):
            return False

        if not self._estimate_retrieval_bottleneck():
            return False

        current_search_capable = [
            agent for agent in self.population.get_all_agents()
            if self._is_search_capable_role(getattr(agent, "role", ""))
        ]
        current_count = len(current_search_capable)

        child_role = ""
        if not death_only:
            plan = getattr(pair, "child_plan", None)
            role_update_raw = str(getattr(plan, "role_prompt_update", "") or "").strip().lower()
            sanitized = _sanitize_role_update(role_update_raw, allowed_roles)
            child_role = sanitized.split("|", 1)[0].strip() if sanitized else ""

        child_is_search_capable = self._is_search_capable_role(child_role)

        # Soft guard:
        # - during retrieval bottleneck, never eliminate the last search-capable agent
        # - and avoid shrinking the count of search-capable agents below 2 while evidence acquisition is still weak
        if current_count <= 1:
            return True
        if current_count <= 2 and not child_is_search_capable:
            return True
        return False
    
    def _project_population_size(self, decision: MetaDecision) -> MetaDecision:
        """Ensure population size stays within bounds.

        Pair accounting:
          - replacement (default): -1 death, +1 birth  → net 0
          - death_only=True:       -1 death, +0 birth  → net -1
          - birth_only=True:       -0 death, +1 birth  → net +1
        """
        current_size = self.population.get_population_size()
        num_deaths = sum(
            1 for p in decision.birth_death_pairs if not getattr(p, "birth_only", False)
        )
        num_births = sum(
            1 for p in decision.birth_death_pairs if not getattr(p, "death_only", False)
        )

        projected_size = current_size - num_deaths + num_births

        if projected_size > self.control_params.n_max:
            # Remove pure-birth pairs first (they increase population), then replacements.
            excess = projected_size - self.control_params.n_max
            decision.birth_death_pairs = decision.birth_death_pairs[:-excess]
            logger.warning(f"Removed {excess} birth-death pairs to stay within population bounds")

        if projected_size < self.control_params.n_min:
            decision.birth_death_pairs = []
            decision.time_control.trigger_birth_death = False
            logger.warning("Disabled birth-death to maintain minimum population size")

        return decision
    
    def _project_birth_death_pairs(
        self,
        decision: MetaDecision,
        relax_constraints: bool = False,
        allowed_roles: Optional[set[str]] = None,
    ) -> MetaDecision:
        """Ensure birth-death pairs are valid.

        Handles three modes:
          - replacement (default): validate parent + death_target + role
          - death_only=True: validate death_target only (pure kill)
          - birth_only=True: validate parent + role only (pure birth)
        """
        valid_pairs = []
        allowed = allowed_roles or {"planner", "searcher", "verifier", "calculator", "reflector"}

        for pair in decision.birth_death_pairs:
            death_only = getattr(pair, "death_only", False)
            birth_only = getattr(pair, "birth_only", False)

            # --- validate parent (needed unless death_only) ---
            if not death_only:
                if self.population.get_agent(pair.parent_id) is None:
                    logger.warning(f"Parent {pair.parent_id} not found, skipping pair")
                    continue

            # --- validate death target (needed unless birth_only) ---
            if not birth_only:
                if self.population.get_agent(pair.death_target_id) is None:
                    logger.warning(f"Death target {pair.death_target_id} not found, skipping pair")
                    continue
                if pair.death_target_id in self.control_params.protected_nodes:
                    logger.warning(f"Death target {pair.death_target_id} is protected, skipping pair")
                    continue

            # --- validate child role (needed unless death_only) ---
            if not death_only:
                try:
                    plan = getattr(pair, "child_plan", None)
                    role_update_raw = str(getattr(plan, "role_prompt_update", "") or "").strip().lower()
                    sanitized = _sanitize_role_update(role_update_raw, allowed)
                    role_update = sanitized.split("|")[0].strip() if sanitized else ""
                    if role_update_raw and not sanitized:
                        logger.warning(
                            "Role '%s' not in ALLOWED_ROLES=%s, skipping birth-death pair",
                            role_update_raw,
                            sorted(list(allowed)),
                        )
                        continue
                    if plan is not None and sanitized and sanitized != role_update_raw:
                        plan.role_prompt_update = sanitized
                except Exception:
                    pass

            if self._would_reduce_search_capacity_too_far(pair, allowed):
                logger.warning(
                    "Retrieval bottleneck guard skipped pair parent=%s death_target=%s: "
                    "would reduce search-capable capacity too far while upstream evidence is still incomplete",
                    getattr(pair, "parent_id", ""),
                    getattr(pair, "death_target_id", ""),
                )
                continue

            valid_pairs.append(pair)

        # Cap pure-kill and pure-birth at 1 each per slow update to limit disruption.
        pure_kills = [p for p in valid_pairs if getattr(p, "death_only", False) and not getattr(p, "birth_only", False)]
        pure_births = [p for p in valid_pairs if getattr(p, "birth_only", False) and not getattr(p, "death_only", False)]
        replacements = [p for p in valid_pairs if not getattr(p, "death_only", False) and not getattr(p, "birth_only", False)]

        if len(pure_kills) > 1:
            logger.warning(f"Capping pure-kills from {len(pure_kills)} to 1 per slow update")
            pure_kills = pure_kills[:1]
        if len(pure_births) > 1:
            logger.warning(f"Capping pure-births from {len(pure_births)} to 1 per slow update")
            pure_births = pure_births[:1]

        valid_pairs = replacements + pure_kills + pure_births

        # Limit to max_birth_death_pairs
        if len(valid_pairs) > self.control_params.max_birth_death_pairs:
            valid_pairs = valid_pairs[:self.control_params.max_birth_death_pairs]
            logger.warning(f"Limited to {self.control_params.max_birth_death_pairs} birth-death pairs")

        decision.birth_death_pairs = valid_pairs

        if not valid_pairs:
            decision.time_control.trigger_birth_death = False

        return decision
    
    def _project_graph_edit(self, decision: MetaDecision, relax_constraints: bool = False) -> MetaDecision:
        """Ensure graph edits are valid."""
        graph_edit = decision.graph_edit
        
        # Validate remove_edges
        valid_remove = []
        for edge in graph_edit.remove_edges:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            source, target = edge[0], edge[1]
            if self.graph.get_edge_type(source, target) is not None:
                # In strict mode, protect connectivity of protected nodes.
                if relax_constraints or not self._would_disconnect_protected(source, target):
                    valid_remove.append((source, target))
        
        # Validate add_edges
        valid_add = []
        for edge in graph_edit.add_edges:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            source, target = edge[0], edge[1]
            edge_type = edge[2] if len(edge) >= 3 else "directed"
            if (
                source in self.graph.nodes
                and target in self.graph.nodes
                and self.graph.get_edge_type(source, target) is None
            ):
                if str(edge_type).lower() not in _ALLOWED_EDGE_TYPE_VALUES:
                    edge_type = "directed"
                valid_add.append((source, target, str(edge_type).lower()))
        
        if relax_constraints:
            # Relaxed mode: keep only per-direction upper bounds.
            if len(valid_remove) > self.control_params.max_edge_edits:
                valid_remove = valid_remove[:self.control_params.max_edge_edits]
                logger.warning(
                    f"Limited remove_edges to {self.control_params.max_edge_edits} in relaxed mode"
                )
            if len(valid_add) > self.control_params.max_edge_edits:
                valid_add = valid_add[:self.control_params.max_edge_edits]
                logger.warning(
                    f"Limited add_edges to {self.control_params.max_edge_edits} in relaxed mode"
                )
        else:
            # Strict mode: keep global total edge edit budget.
            total_edits = len(valid_remove) + len(valid_add)
            if total_edits > self.control_params.max_edge_edits:
                max_removes = min(len(valid_remove), self.control_params.max_edge_edits // 2)
                max_adds = self.control_params.max_edge_edits - max_removes
                valid_remove = valid_remove[:max_removes]
                valid_add = valid_add[:max_adds]
                logger.warning(f"Limited to {self.control_params.max_edge_edits} total edge edits")
        
        graph_edit.remove_edges = valid_remove
        graph_edit.add_edges = valid_add
        
        if not valid_remove and not valid_add:
            decision.time_control.trigger_graph_rewire = False
        else:
            # If edits disconnect any node, add minimal edges to restore connectivity.
            self._repair_isolated_nodes(graph_edit)
        
        return decision

    def _repair_isolated_nodes(self, graph_edit: GraphEdit) -> None:
        """Ensure no nodes become isolated after proposed edits."""
        try:
            nodes = list(self.graph.nodes)
            if len(nodes) < 2:
                return

            # Simulate edge set after edits.
            # Use get_all_edges() which returns (src, tgt, EdgeType) 3-tuples;
            # self.graph.edges is a dict keyed by (src,tgt) 2-tuples.
            edges = set()
            for src, tgt, etype in self.graph.get_all_edges():
                edges.add((src, tgt, etype))
                if str(etype).lower() == "bidirectional":
                    edges.add((tgt, src, etype))

            for src, tgt in graph_edit.remove_edges:
                edges = {e for e in edges if not (e[0] == src and e[1] == tgt)}

            for item in graph_edit.add_edges:
                if len(item) >= 2:
                    src, tgt = item[0], item[1]
                    etype = item[2] if len(item) >= 3 else "directed"
                    edges.add((src, tgt, str(etype).lower()))

            connected = set()
            for src, tgt, _ in edges:
                connected.add(src)
                connected.add(tgt)

            isolated = [n for n in nodes if n not in connected]
            if not isolated:
                return

            # Connect isolated nodes to best-scoring node (or any node if missing).
            try:
                nodes_sorted = sorted(
                    (self.population.get_agent(n) for n in nodes),
                    key=lambda a: float(getattr(a, "meta_score", 0.0)),
                )
                best = nodes_sorted[-1].agent_id if nodes_sorted else nodes[0]
            except Exception:
                best = nodes[0]

            for iso in isolated:
                if iso == best:
                    continue
                graph_edit.add_edges.append((iso, best, "evidence_flow"))
        except Exception as e:
            logger.warning(f"Failed to repair isolated nodes: {e}")
    
    def _project_role_coverage(self, decision: MetaDecision) -> MetaDecision:
        """Ensure critical role coverage is maintained."""
        # Simulate the effect of birth-death on role distribution
        role_dist = self.population.get_role_distribution()
        
        for pair in decision.birth_death_pairs:
            death_agent = self.population.get_agent(pair.death_target_id)
            if death_agent:
                role_dist[death_agent.role] = role_dist.get(death_agent.role, 1) - 1
        
        # Check if any critical role would be eliminated
        for role in self.control_params.critical_roles:
            if role_dist.get(role, 0) <= 0:
                logger.warning(f"Critical role {role} would be eliminated, disabling birth-death")
                decision.birth_death_pairs = []
                decision.time_control.trigger_birth_death = False
                break
        
        return decision
    
    def _would_disconnect_protected(self, source: str, target: str) -> bool:
        """Check if removing an edge would disconnect protected nodes."""
        if not self.control_params.preserve_connectivity:
            return False
        
        # Check if either endpoint is protected
        if source in self.control_params.protected_nodes or target in self.control_params.protected_nodes:
            # Check if this is a bridge edge
            bridge_edges = self.graph.get_bridge_edges()
            if (source, target) in bridge_edges or (target, source) in bridge_edges:
                return True
        
        return False


class EvolutionExecutor:
    """Executes evolutionary decisions."""
    
    def __init__(
        self,
        graph_manager: GraphManager,
        population_manager: PopulationManager,
    ):
        """Initialize the evolution executor."""
        self.graph = graph_manager
        self.population = population_manager
        self._graph_edit_aliases: dict[str, str] = {}

    def _load_allowed_roles(self) -> set[str]:
        raw = os.environ.get("ALLOWED_ROLES", "").strip()
        if not raw:
            return {"planner", "searcher", "verifier", "calculator", "reflector"}
        roles = {r.strip().lower() for r in raw.split(",") if r.strip()}
        return roles or {"planner", "searcher", "verifier", "calculator", "reflector"}
    
    def execute_decision(
        self,
        decision: MetaDecision,
        current_time: int,
    ) -> None:
        """Execute the evolutionary decision."""
        if not decision.slow_update:
            logger.info("No slow update triggered")
            return
        
        logger.info(f"Executing evolutionary decision with confidence {decision.confidence:.2f}")
        self._graph_edit_aliases = {}
        
        # Execute birth-death
        if decision.time_control.trigger_birth_death:
            self._execute_birth_death(decision.birth_death_pairs, current_time)
        
        # Execute graph edit
        if decision.time_control.trigger_graph_rewire:
            self._execute_graph_edit(decision.graph_edit)
    
    def _execute_birth_death(
        self,
        birth_death_pairs: List[BirthDeathPair],
        current_time: int,
    ) -> None:
        """Execute birth-death operations.

        Three modes:
          - death_only=True: pure kill — remove agent, no replacement.
          - birth_only=True: pure birth — spawn new child from parent, no removal.
          - default: standard replacement (spawn child then remove death target).
        """
        for pair in birth_death_pairs:
            death_only = getattr(pair, "death_only", False)
            birth_only = getattr(pair, "birth_only", False)

            try:
                # ----------------------------------------------------------
                # Mode 1: pure kill
                # ----------------------------------------------------------
                if death_only and not birth_only:
                    logger.info(
                        f"Pure-Kill: death_target={pair.death_target_id}, reason={pair.death_reason}"
                    )
                    self.population.remove_agent(pair.death_target_id)
                    self.graph.remove_node(pair.death_target_id)
                    logger.info(f"Removed agent {pair.death_target_id} (pure-kill)")
                    continue

                # ----------------------------------------------------------
                # Mode 2: pure birth
                # ----------------------------------------------------------
                if birth_only and not death_only:
                    logger.info(
                        f"Pure-Birth: parent={pair.parent_id}, reason={pair.birth_reason}"
                    )
                    child_id = self.population.spawn_child(
                        pair.parent_id,
                        "",           # no death target — spawn_child uses target only for HYBRID
                        pair.child_plan,
                        current_time,
                    )
                    child_agent = self.population.get_agent(child_id)
                    self.graph.add_node(child_id, child_agent.role)
                    self._register_graph_edit_aliases(pair, child_id, child_agent.role)
                    # Wire: directed edge from parent to new child so it is reachable.
                    if self.graph.get_edge_type(pair.parent_id, child_id) is None:
                        self.graph.add_edge(pair.parent_id, child_id, edge_type=EdgeType.DIRECTED)
                    logger.info(
                        "Spawned fresh agent %s from parent %s (pure-birth), role=%s, prompt_variant=%s, capability_need=%s, capability_variant=%s",
                        child_id,
                        pair.parent_id,
                        getattr(child_agent, "role", ""),
                        getattr(child_agent, "prompt_variant", ""),
                        getattr(getattr(pair, "child_plan", None), "capability_need", ""),
                        getattr(getattr(pair, "child_plan", None), "capability_variant", ""),
                    )
                    continue

                # ----------------------------------------------------------
                # Mode 3: standard replacement (default)
                # ----------------------------------------------------------
                logger.info(
                    f"Birth-Death: parent={pair.parent_id}, "
                    f"death_target={pair.death_target_id}, "
                    f"reason={pair.death_reason}"
                )

                inherited_edges: list[tuple[str, str, EdgeType]] = []
                death_target_id = pair.death_target_id
                for neighbor_id in self.graph.get_neighbors(death_target_id, direction="both"):
                    bidirectional_edge = self.graph.get_edge_type(death_target_id, neighbor_id)
                    reverse_edge = self.graph.get_edge_type(neighbor_id, death_target_id)

                    if bidirectional_edge == EdgeType.BIDIRECTIONAL or reverse_edge == EdgeType.BIDIRECTIONAL:
                        inherited_edges.append((death_target_id, neighbor_id, EdgeType.BIDIRECTIONAL))
                        continue

                    if bidirectional_edge is not None:
                        inherited_edges.append((death_target_id, neighbor_id, bidirectional_edge))
                    if reverse_edge is not None:
                        inherited_edges.append((neighbor_id, death_target_id, reverse_edge))

                # Spawn child before removing the death target so hybrid inheritance
                # can still read the target agent state.
                child_id = self.population.spawn_child(
                    pair.parent_id,
                    pair.death_target_id,
                    pair.child_plan,
                    current_time,
                )

                # Remove old agent after inheritance is materialized.
                self.population.remove_agent(pair.death_target_id)
                self.graph.remove_node(pair.death_target_id)

                # Add child to graph
                child_agent = self.population.get_agent(child_id)
                self.graph.add_node(child_id, child_agent.role)
                self._register_graph_edit_aliases(pair, child_id, child_agent.role)

                for source, target, edge_type in inherited_edges:
                    new_source = child_id if source == death_target_id else source
                    new_target = child_id if target == death_target_id else target
                    if new_source == new_target:
                        continue
                    if self.graph.get_edge_type(new_source, new_target) is not None:
                        continue
                    self.graph.add_edge(new_source, new_target, edge_type=edge_type)

                logger.info(
                    "Spawned child agent %s via replacement, parent=%s, replaced=%s, role=%s, prompt_variant=%s, capability_need=%s, capability_variant=%s",
                    child_id,
                    pair.parent_id,
                    pair.death_target_id,
                    getattr(child_agent, "role", ""),
                    getattr(child_agent, "prompt_variant", ""),
                    getattr(getattr(pair, "child_plan", None), "capability_need", ""),
                    getattr(getattr(pair, "child_plan", None), "capability_variant", ""),
                )

            except Exception as e:
                logger.error(f"Error executing birth-death pair: {e}")

    def _register_graph_edit_aliases(
        self,
        pair: BirthDeathPair,
        child_id: str,
        child_role: str,
    ) -> None:
        """Register placeholder aliases so post-BD graph edits can target new nodes.

        Meta often writes edges using placeholders like `new_searcher` before the executor
        knows the real child id. We preserve a tiny alias map from those placeholders to the
        actual spawned node id so graph edits can still be applied after birth/death.
        """
        aliases = {
            "new_child": child_id,
            f"new_{child_role}": child_id,
            f"child_{child_role}": child_id,
        }
        allowed = self._load_allowed_roles()
        role_update = _sanitize_role_update(
            str(getattr(getattr(pair, "child_plan", None), "role_prompt_update", "") or "").strip().lower(),
            allowed,
        )
        role_base = role_update.split("|", 1)[0].strip() if role_update else ""
        if role_base:
            aliases[f"new_{role_base}"] = child_id
            aliases[f"child_{role_base}"] = child_id

        for alias, value in aliases.items():
            self._graph_edit_aliases[alias] = value

    def _normalize_graph_edit_endpoint(self, node_id: str) -> Optional[str]:
        """Resolve placeholder ids and drop stale graph-edit endpoints."""
        text = str(node_id or "").strip()
        if not text:
            return None
        resolved = self._graph_edit_aliases.get(text, text)
        if resolved in self.graph.nodes:
            return resolved
        # Drop placeholder-like ids after BD if they were never materialized.
        lowered = text.lower()
        if lowered.startswith("new_") or lowered.startswith("child_"):
            return None
        return resolved if resolved in self.graph.nodes else None
    
    def _execute_graph_edit(self, graph_edit: GraphEdit) -> None:
        """Execute graph structure edits."""
        # Remove edges
        for edge in graph_edit.remove_edges:
            try:
                if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                    continue
                source = self._normalize_graph_edit_endpoint(edge[0])
                target = self._normalize_graph_edit_endpoint(edge[1])
                if not source or not target:
                    logger.info(f"Skipped stale remove_edge {edge}")
                    continue
                self.graph.remove_edge(source, target)
                logger.info(f"Removed edge ({source}, {target})")
            except Exception as e:
                logger.error(f"Error removing edge ({edge}): {e}")
        
        # Add edges
        for edge in graph_edit.add_edges:
            try:
                if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                    continue
                source = self._normalize_graph_edit_endpoint(edge[0])
                target = self._normalize_graph_edit_endpoint(edge[1])
                if not source or not target or source == target:
                    logger.info(f"Skipped stale add_edge {edge}")
                    continue
                edge_type = edge[2] if len(edge) >= 3 else "directed"
                edge_type_enum = EdgeType.from_value(edge_type)
                if self.graph.get_edge_type(source, target) is not None:
                    logger.info(f"Skipped duplicate add_edge ({source}, {target})")
                    continue
                self.graph.add_edge(source, target, edge_type=edge_type_enum)
                logger.info(f"Added edge ({source}, {target}) type={edge_type_enum.value}")
            except Exception as e:
                logger.error(f"Error adding edge ({edge}): {e}")
        
        # Change edge types
        for change in graph_edit.type_changes:
            try:
                edge = change.get("edge", [])
                new_type = change.get("new_type", "directed")
                if len(edge) == 2:
                    source = self._normalize_graph_edit_endpoint(edge[0])
                    target = self._normalize_graph_edit_endpoint(edge[1])
                    if not source or not target:
                        logger.info(f"Skipped stale type_change {change}")
                        continue
                    edge_type = EdgeType.from_value(new_type)
                    if self.graph.get_edge_type(source, target) is None:
                        logger.info(f"Skipped missing edge type_change ({source}, {target})")
                        continue
                    self.graph.change_edge_type(source, target, edge_type)
                    logger.info(f"Changed edge type ({source}, {target}) to {edge_type.value}")
            except Exception as e:
                logger.error(f"Error changing edge type: {e}")
