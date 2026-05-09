"""
Meta LLM interface for making evolutionary decisions.
"""

import json
import math
import os
import time
import re
from typing import Optional, TYPE_CHECKING
import logging

import requests

if TYPE_CHECKING:
    from tacomas.llm import ChatLiteLLMLC
    from tacomas.config.llm import LLMConfig

from .schemas import (
    SystemSummary,
    ControlParams,
    MetaDecision,
    BirthDeathPair,
    GraphEdit,
    FinalSynthesis,
    TimeControl,
    ChildInheritancePlan,
    InheritanceType,
    AgentEvolutionFeedback,
)

logger = logging.getLogger(__name__)


class MetaLLMInterface:
    """Interface for calling meta LLM to make evolutionary decisions."""
    
    def __init__(self, llm_config: "LLMConfig"):
        """Initialize the meta LLM interface."""
        # Lazy import to avoid pulling in langfuse/langchain at module load time
        from tacomas.config.llm import LLMConfig as _LLMConfig  # noqa: F401
        self.llm_config = llm_config

        # DMX non-stream mode for meta LLM (explicitly requested)
        self.use_dmx_nonstream = os.environ.get("USE_DMX_META_NONSTREAM", "0") == "1"
        self.dmx_api_key = os.environ.get("DMX_API_KEY", "")
        self.dmx_model = os.environ.get("DMX_MODEL", "gemini-2.5-flash")

        if not self.use_dmx_nonstream:
            self.llm = llm_config.get_llm()
        else:
            self.llm = None
        self.last_request_context: dict | None = None
        self.last_response_text: str = ""

        # Allowed roles for structural mutations (hard gated). Comma-separated.
        # Example: "planner,searcher,verifier" or "planner,searcher,verifier,calculator,reflector"
        self.allowed_roles = self._load_allowed_roles()

    def _load_allowed_roles(self) -> list[str]:
        raw = os.getenv("ALLOWED_ROLES", "").strip()
        if not raw:
            return [
                "planner",
                "searcher",
                "researcher",
                "analyst",
                "verifier",
                "auditor",
                "schema_verifier",
                "calculator",
                "forecaster",
                "reflector",
                "synthesizer",
            ]
        roles = [r.strip().lower() for r in raw.split(",") if r.strip()]
        # Keep only non-empty unique roles, stable order.
        seen = set()
        out: list[str] = []
        for r in roles:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out or [
            "planner",
            "searcher",
            "researcher",
            "analyst",
            "verifier",
            "auditor",
            "schema_verifier",
            "calculator",
            "forecaster",
            "reflector",
            "synthesizer",
        ]

    def _default_capability_for_role(self, role: str) -> tuple[str, str]:
        role = str(role or "").strip().lower()
        mapping = {
            "searcher": ("broad_retrieval", "source_exploration"),
            "researcher": ("targeted_extraction", "document_drilldown"),
            "analyst": ("targeted_extraction", "structured_slot_extraction"),
            "verifier": ("verification", "fact_crosscheck"),
            "auditor": ("verification", "source_audit"),
            "schema_verifier": ("schema_validation", "field_semantics"),
            "calculator": ("computation", "numeric_derivation"),
            "forecaster": ("computation", "trend_projection"),
            "reflector": ("coordination", "process_diagnosis"),
            "synthesizer": ("synthesis", "concise_merge"),
            "planner": ("coordination", "task_routing"),
        }
        return mapping.get(role, ("coordination", "generalist"))
    
    def call_meta_llm(
        self,
        system_summary: SystemSummary,
        control_params: ControlParams,
        task_description: str = "",
    ) -> MetaDecision:
        """Call meta LLM to get evolutionary decision."""
        system_prompt = self._get_system_prompt()
        developer_prompt = self._get_developer_prompt(control_params)
        user_payload = self._construct_user_payload(
            system_summary, control_params, task_description
        )
        self.last_request_context = {
            "system_prompt": system_prompt,
            "developer_prompt": developer_prompt,
            "user_payload": user_payload,
        }
        
        try:
            if self.use_dmx_nonstream:
                response_text = self._call_dmx_nonstream(
                    system_prompt=system_prompt,
                    developer_prompt=developer_prompt,
                    user_payload=user_payload,
                )
            else:
                response = self.llm.invoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "system", "content": developer_prompt},
                        {"role": "user", "content": user_payload},
                    ]
                )
                response_text = response.content
            self.last_response_text = str(response_text)
            decision = self._parse_meta_decision(response_text, system_summary)
            decision = self._apply_gated_greedy_fallback(
                decision=decision,
                system_summary=system_summary,
                control_params=control_params,
            )
            logger.info(f"Meta LLM decision: confidence={decision.confidence:.2f}")
            return decision
            
        except Exception as e:
            logger.error(f"Error calling meta LLM: {e}")
            return self._get_default_decision(system_summary)
    
    def _call_dmx_nonstream(
        self,
        system_prompt: str,
        developer_prompt: str,
        user_payload: str,
    ) -> str:
        """Call DMX Gemini generateContent (non-stream) for meta decision."""
        if not self.dmx_api_key:
            raise RuntimeError("USE_DMX_META_NONSTREAM=1 but DMX_API_KEY is empty")

        url = (
            f"https://www.dmxapi.cn/v1beta/models/{self.dmx_model}:generateContent"
            f"?key={self.dmx_api_key}"
        )
        headers = {"Content-Type": "application/json"}

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt + "\n\n" + developer_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_payload}],
                }
            ],
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"DMX API failed: {resp.status_code}, {resp.text[:500]}")

        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return json.dumps(data, ensure_ascii=False)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for meta LLM."""
        return """You are a meta-level evolutionary controller for a multi-agent graph system.

Your job is to improve the whole agent population over time and produce the final system answer by synthesizing all agent outputs.

At each slow-time decision point, you must decide:
1. which agent lineage should reproduce (birth),
2. which agent should be removed or replaced (death),
3. how the child agent should inherit memory, capability, and role,
4. how the graph structure should be edited, including edge types,
5. how to synthesize multiple agent outputs into the final answer.

You must separately judge whether local learning, birth-death, or graph rewiring should happen.
Graph rewiring must be slower and more conservative than birth-death.

You must optimize for long-horizon system performance.
You must balance exploration, exploitation, diversity, robustness, and communication efficiency.

Edge types carry generalized coordination semantics:
- bidirectional: debate or mutual challenge.
- directed: generic serial handoff when no stronger semantic is known.
- evidence_flow: primary evidence or source handoff.
- verification_flow: validation, correction, or consistency checking.
- computation_flow: numeric aggregation, derivation, or structured transformation.
- reflection_feedback: corrective process feedback; useful, but usually auxiliary rather than the main evidence backbone.

Prefer evidence_flow / verification_flow / computation_flow for mainline collaboration.
Use reflection_feedback as a lighter corrective edge, and avoid letting it dominate the graph backbone unless the task truly requires that.
Treat reflection_feedback as auxiliary process feedback: it can improve future search, verification, or calculation, but it should not be selected as the final-answer evidence path unless there is also a mainline edge carrying concrete evidence.
When graph rewiring is enabled, make sure the main answer path is carried by non-reflection edge types such as evidence_flow, verification_flow, or computation_flow.

Before choosing a child role, first reason about the missing capability niche.
Use role as a lightweight prototype / bias, not as a rigid job title.
When proposing a child, prefer to specify:
- capability_need: one of broad_retrieval, targeted_extraction, verification, schema_validation, computation, synthesis, conflict_resolution, coordination
- capability_variant: a short niche such as source_exploration, source_curation, slot_filling, contradiction_check, range_check, table_extraction, concise_synthesis
- role_prompt_update: the closest allowed role prototype to host that capability

The system should evolve toward the missing capability, not merely duplicate role labels.
You output a structured decision object in valid JSON only."""

    def _estimate_agent_round_cost(self, num_agents: int) -> dict:
        """Estimate per-fast-round token/cost burden from agent count."""
        # Optional overrides for rough budgeting (per agent per fast round).
        # These are intentionally conservative defaults to expose scale pressure.
        per_agent_input_tokens = int(os.getenv("META_COST_EST_INPUT_TOKENS_PER_AGENT", "650"))
        per_agent_output_tokens = int(os.getenv("META_COST_EST_OUTPUT_TOKENS_PER_AGENT", "220"))
        # Cost unit here is abstract unless caller sets concrete pricing via env.
        per_1k_input_cost = float(os.getenv("META_COST_EST_INPUT_COST_PER_1K", "0.0"))
        per_1k_output_cost = float(os.getenv("META_COST_EST_OUTPUT_COST_PER_1K", "0.0"))
        total_input_tokens = max(0, num_agents) * max(0, per_agent_input_tokens)
        total_output_tokens = max(0, num_agents) * max(0, per_agent_output_tokens)
        est_cost = (
            (total_input_tokens / 1000.0) * max(0.0, per_1k_input_cost)
            + (total_output_tokens / 1000.0) * max(0.0, per_1k_output_cost)
        )
        return {
            "per_agent_input_tokens": per_agent_input_tokens,
            "per_agent_output_tokens": per_agent_output_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "estimated_cost_per_fast_round": round(est_cost, 6),
        }




    def _get_developer_prompt(self, control_params: ControlParams) -> str:
        """Get the developer prompt for meta LLM."""
        return f"""Return exactly one structured decision in valid JSON.

Hard constraints:
- Replace at most {control_params.max_birth_death_pairs} nodes in one slow update.
- Add or remove at most {control_params.max_edge_edits} edges.
- Do not disconnect protected core nodes: {control_params.protected_nodes}
- Do NOT introduce isolated nodes (every node must have at least one incident edge after edits).
- Preserve at least these critical roles: {control_params.critical_roles}
- Allowed agent roles for child creation/mutation: {self.allowed_roles}
- Keep total agents within [{control_params.n_min}, {control_params.n_max}].

General principle:
- First decide the missing capability niche; then choose the closest role prototype.
- Role names are lightweight priors, not fixed jobs.
- Prefer capability-aware differentiation over naive role duplication.
- If a role is duplicated, make sure the capability_variant is different.

Birth-death pair modes (set exactly one flag or neither):
- Default (death_only=false, birth_only=false): 1-for-1 replacement. parent_id + death_target_id + child_plan required.
- death_only=true: pure kill — remove death_target_id, no replacement (shrinks pop by 1). parent_id/child_plan ignored. Max 1 per update.
- birth_only=true: pure birth — spawn new child from parent_id, no removal (grows pop by 1). death_target_id ignored. Max 1 per update.

Decision principles — base decisions directly on observed evidence, not any formula:
- Look at each agent's contribution_score and score_trend. Agents with persistent low scores AND negative trend are candidates for replacement or removal.
- Look at current answer_quality and its recent trajectory. If quality is already high and stable, prefer conservative or no structural edits.
- Consider role coverage: if a critical role (searcher, verifier) is missing or underperforming, add or replace toward it.
- Consider whether adding an agent (birth_only) is worth the extra cost per fast round — only add when a genuinely missing capability would unblock the workflow.
- Consider whether removing an agent (death_only) makes sense when population is large and a role is clearly redundant or blocking.
- Prefer birth_only over replacement when current_N < N_max and the problem is missing capacity / missing role coverage rather than a clearly bad incumbent.
- Prefer death_only over replacement when current_N > N_min and one agent is low-signal, redundant, and removing it would not break critical role coverage.
- Prefer replacement when the system is at capacity or when a specific low-score agent should be swapped into a more useful role.
- Distinguish "low score but unique role" from "low score AND replaceable" — only replace the latter.
- Treat search-capable roles as a protected soft resource while the system is still in a retrieval bottleneck. Do not remove or mutate away the last search-capable lineage unless there is already strong upstream evidence and another search-capable node remains.
- When upstream evidence is still incomplete, prefer birth_only of a complementary role or rewire over deleting a search-capable node.
- When editing the graph, look at bridge_value and interaction_quality on edges. Prefer adding edges between high-score agents that are currently disconnected.
- If confidence is low, prefer conservative edits (graph rewire over BD, or no edits at all).
- Use `schema_verifier` only when the task appears schema-sensitive (for example range comparison, structured bundle filling, symbolic/numeric derivation, or citation-bound QA) and the current failure is due to missing schema fields, field-semantic drift, or malformed bundles rather than generic retrieval failure.
- If the runtime/task description mentions missing schema fields, incomplete bundle, or field semantic errors, prefer adding or mutating toward `schema_verifier` before adding more generic reflector capacity.
- Do NOT force the population into a rigid pipeline. Treat task schema as environment bias, not a fixed execution order.
- Prefer agents and edges that create reusable validated information objects (for example claims, values, relations, derivations, critiques, repair hints) over agents that only repeat process narration.
- When deciding birth/death/rewire, reason about bottleneck type: retrieval bottleneck, structuring bottleneck, validation bottleneck, reasoning bottleneck, or coordination bottleneck. Use weak_or_missing_object_types and object status mix as evidence.
- If the runtime reports a dominant bottleneck, treat that as a stronger signal than the static task label. Retrieval bottleneck means protect or expand evidence acquisition capacity; structuring bottleneck means reward agents that turn raw evidence into reusable objects; validation bottleneck means add checking/repair capacity.
- Role prompts are biases, not hard job descriptions. Allow niche specialization to emerge through prompt variants and graph position if it improves validated object production.

Anti-stagnation rules (must follow):
- If learning_efficiency <= 0.05 AND bottleneck_count > 0, you MUST NOT return an empty structural decision.
- If one role dominates (>60% of agents) and at least one non-dominant role has low score trend, you MUST trigger either birth-death or graph rewire.
- If the same role dominates (>60%) for two consecutive slow updates and answer_quality < 0.9, you MUST trigger structural change.
- Avoid empty structural decisions. If uncertain, propose at least one minimal graph edit between two high-score nodes using a light but meaningful edge type (for example evidence_flow, verification_flow, directed, or bidirectional) and justify it.
- If reflection_feedback edges become the dominant path around the highest-degree node, rewire toward a mainline evidence/verification/computation path unless the current answer_quality is already high and stable.
- Do not use reflection_feedback as the only bridge between evidence-producing agents and the final synthesis contributors.
- High-score path protection: if the runtime reports protected high-score backbone edges, avoid deleting them. You may override a protected edge ONLY if graph_edit.rewire_notes or subgraph_rewrites contains `protected_edge_override_reason: ...` and graph_edit.add_edges/type_changes creates an alternative main path using evidence_flow, verification_flow, computation_flow, or bidirectional edges.
- In these cases, set at least one of:
  - time_control.trigger_birth_death = true, with non-empty birth_death_pairs
  - time_control.trigger_graph_rewire = true, with non-empty graph_edit changes
- Only keep both triggers false when a hard constraint blocks all edits; then explain it in global_rationale.

Workflow guidance quality rules:
- For every low-score agent (<0.35), workflow_correction and next_round_workflow must be concrete and tool-aware (mention exact tool sequence or verification checklist).
- Avoid generic advice like "be more structured"; provide executable instructions with 2-4 steps and expected evidence output.
- If multiple agents report failure_modes like "missing_key", "no_reconciliation", or "no_evidence", you MUST instruct the searcher to re-search with a different query and re-parse new documents (do not reuse those documents).
- When the searcher must re-search, include one explicit rewritten query in next_round_workflow using the exact format `SEARCH_QUERY: ...` so runtime can pass it directly to search tools.

Return JSON with this schema:
{{"slow_update": true, "confidence": 0.8, "agent_feedback": [{{"agent_id": "...", "score": 0.0, "score_reason": "...", "workflow_correction": "How the current workflow should be corrected", "next_round_workflow": "What this agent should do next round to evolve"}}], "birth_death_pairs": [{{"parent_id": "agent_1", "death_target_id": "agent_3", "birth_reason": "...", "death_reason": "...", "death_only": false, "birth_only": false, "child_plan": {{"inherit_type": "mutate_role", "memory_from_parent_ratio": 0.8, "memory_from_target_ratio": 0.2, "capability_noise_scale": 0.1, "policy_mutation": "...", "role_prompt_update": "searcher", "capability_need": "broad_retrieval", "capability_variant": "source_exploration"}}, "improvement_direction": "..."}}], "graph_edit": {{"remove_edges": [], "add_edges": [], "type_changes": [], "rewire_notes": [], "subgraph_rewrites": []}}, "final_synthesis": {{"contributors": [], "strategy": "", "conflict_resolution": "", "final_answer_spec": ""}}, "time_control": {{"birth_death_value": 0.6, "graph_rewire_value": 0.2, "fast_steps_next": 20, "trigger_birth_death": true, "trigger_graph_rewire": false, "trigger_next_slow_rule": "adaptive", "cooldown": 0, "continue_evolution": true, "stop_reason": ""}}, "global_rationale": []}}"""
    
    def _construct_user_payload(
        self,
        system_summary: SystemSummary,
        control_params: ControlParams,
        task_description: str,
    ) -> str:
        """Construct the user payload for meta LLM."""
        role_counts: dict[str, int] = {}
        low_signal_agents = 0
        for node_summary in system_summary.node_summaries:
            role_counts[node_summary.role] = role_counts.get(node_summary.role, 0) + 1
            if node_summary.last_meta_score < 0.25:
                low_signal_agents += 1

        if system_summary.node_summaries:
            mean_contribution = sum(
                max(0.0, min(1.0, float(n.last_meta_score)))
                for n in system_summary.node_summaries
            ) / len(system_summary.node_summaries)
        else:
            mean_contribution = 0.0

        connected = set()
        for e in system_summary.edge_summaries or []:
            connected.add(e.source)
            connected.add(e.target)
        isolated_ids = [n.agent_id for n in system_summary.node_summaries if n.agent_id not in connected]

        self._latest_payload_ctx = {
            "isolated_nodes": isolated_ids,
            "mean_contribution": mean_contribution,
        }

        payload = f"""Current slow-time step: {system_summary.slow_update_step}
Task objective: {task_description or system_summary.task_objective}

Global statistics:
- answer_quality (judge score): {system_summary.current_answer_quality:.3f}
- mean_agent_contribution_score: {mean_contribution:.3f}
- score_trend: {system_summary.score_trend:.3f}
- graph_density: {system_summary.graph_density:.3f}
- modularity: {system_summary.modularity:.3f}
- graph_connectivity_score: {getattr(system_summary, "graph_connectivity_score", 0.0):.3f}
- bottleneck_count: {system_summary.bottleneck_count}
- diversity_index: {system_summary.diversity_index:.3f}
- system_stable: {system_summary.is_stable}
- learning_efficiency: {system_summary.learning_efficiency:.3f}
- role_distribution: {role_counts}
- low_signal_agents (contribution_score < 0.25): {low_signal_agents}
- isolated_nodes (must be connected): {isolated_ids}
- object_ecology_total: {getattr(system_summary, "object_ecology_total", 0)}
- object_ecology_validated: {getattr(system_summary, "object_ecology_validated", 0)}
- object_ecology_tentative: {getattr(system_summary, "object_ecology_tentative", 0)}
- weak_or_missing_object_types: {getattr(system_summary, "weak_or_missing_object_types", [])}

Constraints:
- max_birth_death_pairs = {control_params.max_birth_death_pairs}
- max_edge_edits = {control_params.max_edge_edits}
- N_min = {control_params.n_min}, N_max = {control_params.n_max}
- current_N = {system_summary.total_agents}
- protected_nodes = {control_params.protected_nodes}
- critical_roles = {control_params.critical_roles}

Agent summaries:
For each agent, score the current contribution and decide how its workflow should evolve next.
"""
        for node_summary in system_summary.node_summaries[:8]:
            payload += f"""
  - id: {node_summary.agent_id}, role: {node_summary.role}
    contribution_score: {node_summary.last_meta_score:.3f}
    recent_scores: {node_summary.recent_scores[-3:]}
    score_trend: {node_summary.score_trend:.3f}
    failure_modes: {getattr(node_summary, "failure_modes", [])}
    neighbor_externality: {node_summary.neighbor_externality:.3f}
    bridge_value: {node_summary.bridge_value:.3f}
    latest_workflow: {node_summary.memory_summary}
    latest_output: {node_summary.latest_output}
    workflow_correction: {node_summary.workflow_correction}
    next_round_workflow: {node_summary.next_round_workflow}
"""

        payload += "\nEdge summaries:\n"
        for edge_summary in system_summary.edge_summaries[:5]:
            payload += f"""
  - [{edge_summary.source}, {edge_summary.target}]: type={edge_summary.edge_type.value}, semantic_group={edge_summary.semantic_group}, weight={edge_summary.importance_weight:.2f}, quality={edge_summary.interaction_quality:.3f}, bridge={edge_summary.bridge_flag}
"""

        payload += "\nPlease produce one slow-time evolutionary decision in valid JSON."
        return payload
    
    def _parse_meta_decision(
        self, response_text: str, system_summary: SystemSummary
    ) -> MetaDecision:
        """Parse meta LLM response into MetaDecision."""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                return self._get_default_decision(system_summary)
            
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            # Parse agent feedback
            agent_feedback = []
            for feedback_data in data.get("agent_feedback", []):
                agent_feedback.append(
                    AgentEvolutionFeedback(
                        agent_id=str(feedback_data.get("agent_id", "")),
                        score=max(0.0, min(1.0, float(feedback_data.get("score", 0.0)))),
                        score_reason=str(feedback_data.get("score_reason", "")),
                        workflow_correction=str(feedback_data.get("workflow_correction", "")),
                        next_round_workflow=str(feedback_data.get("next_round_workflow", "")),
                    )
                )

            # Parse birth-death pairs
            bd_pairs = []
            for pair_data in data.get("birth_death_pairs", []):
                # Support two output styles from the meta-LLM:
                # 1) Schema-aligned: parent/death_target + child_plan.{inherit_type,...}
                # 2) "Human-ish" style (seen in logs): death/birth/inherit_from + child_role/inheritance_type/memory_transfer/...
                def _coerce_float(val: object, default: float) -> float:
                    try:
                        return float(val)
                    except Exception:
                        return default

                def _coerce_inherit_type(raw: object) -> InheritanceType:
                    if isinstance(raw, InheritanceType):
                        return raw
                    if raw is None:
                        return InheritanceType.PURE_CLONE
                    s = str(raw).strip().lower()
                    if s in {"pure_clone", "pure-clone", "clone"}:
                        return InheritanceType.PURE_CLONE
                    if s in {"mutate_role", "mutate-role", "mutate"}:
                        return InheritanceType.MUTATE_ROLE
                    if s in {"hybrid", "mix"}:
                        return InheritanceType.HYBRID
                    # Default if unknown
                    return InheritanceType.PURE_CLONE

                parent_id = (
                    pair_data.get("parent")
                    or pair_data.get("parent_id")
                    or pair_data.get("inherit_from")
                    or ""
                )
                death_target_id = (
                    pair_data.get("death_target")
                    or pair_data.get("death_target_id")
                    or pair_data.get("death")
                    or ""
                )

                birth_reason = (
                    pair_data.get("birth_reason")
                    or pair_data.get("reason")
                    or pair_data.get("rationale")
                    or ""
                )
                death_reason = (
                    pair_data.get("death_reason")
                    or pair_data.get("death_reason_text")
                    or ""
                )
                improvement_direction = (
                    pair_data.get("improvement_direction")
                    or pair_data.get("capability_transfer")
                    or pair_data.get("rationale")
                    or ""
                )

                # Prefer schema-aligned child_plan if provided; otherwise synthesize from alias fields.
                child_plan_data = pair_data.get("child_plan") or {}
                inherit_type = _coerce_inherit_type(
                    child_plan_data.get("inherit_type", pair_data.get("inheritance_type"))
                )

                memory_from_parent_ratio = child_plan_data.get(
                    "memory_from_parent_ratio"
                )
                memory_from_target_ratio = child_plan_data.get(
                    "memory_from_target_ratio"
                )
                capability_noise_scale = child_plan_data.get(
                    "capability_noise_scale"
                )

                # If ratios are not numeric (e.g. "partial"), map common string values.
                if memory_from_parent_ratio is None or memory_from_target_ratio is None:
                    mem_transfer = str(pair_data.get("memory_transfer", "")).strip().lower()
                    if mem_transfer == "partial":
                        memory_from_parent_ratio = 0.7
                        memory_from_target_ratio = 0.3
                    elif mem_transfer in {"from_parent", "parent_only", "mostly_parent"}:
                        memory_from_parent_ratio = 0.9
                        memory_from_target_ratio = 0.1
                    elif mem_transfer in {"from_target", "target_only", "mostly_target"}:
                        memory_from_parent_ratio = 0.1
                        memory_from_target_ratio = 0.9
                    else:
                        memory_from_parent_ratio = 0.8
                        memory_from_target_ratio = 0.2

                # capability_noise_scale falls back to a safe default.
                if capability_noise_scale is None:
                    capability_noise_scale = 0.1

                policy_mutation = child_plan_data.get("policy_mutation", "") or ""
                role_prompt_update = child_plan_data.get("role_prompt_update", "") or ""
                capability_need = child_plan_data.get("capability_need", "") or ""
                capability_variant = child_plan_data.get("capability_variant", "") or ""
                if not role_prompt_update:
                    # "child_role" is what the meta-LLM output uses in the log.
                    role_prompt_update = pair_data.get("child_role", "") or ""
                if role_prompt_update:
                    role_text = str(role_prompt_update).strip()
                    chosen_role = ""
                    for role in self.allowed_roles:
                        if re.search(rf"\b{re.escape(role)}\b", role_text.lower()):
                            chosen_role = role
                            break
                    if chosen_role:
                        role_prompt_update = chosen_role
                    elif len(role_text.split()) > 3:
                        role_prompt_update = ""

                # If the meta output specifies a target role or an abstract capability
                # but omits inheritance_type, defaulting to PURE_CLONE would create a
                # no-op structural change. Treat either signal as a role mutation.
                if (role_prompt_update or capability_need) and inherit_type == InheritanceType.PURE_CLONE:
                    inherit_type = InheritanceType.MUTATE_ROLE

                child_plan = ChildInheritancePlan(
                    inherit_type=inherit_type,
                    memory_from_parent_ratio=_coerce_float(
                        memory_from_parent_ratio, 0.8
                    ),
                    memory_from_target_ratio=_coerce_float(
                        memory_from_target_ratio, 0.2
                    ),
                    capability_noise_scale=_coerce_float(
                        capability_noise_scale, 0.1
                    ),
                    policy_mutation=str(policy_mutation),
                    role_prompt_update=str(role_prompt_update),
                    capability_need=str(capability_need),
                    capability_variant=str(capability_variant),
                )

                death_only = bool(pair_data.get("death_only", False))
                birth_only = bool(pair_data.get("birth_only", False))

                bd_pair = BirthDeathPair(
                    parent_id=str(parent_id),
                    death_target_id=str(death_target_id),
                    birth_reason=str(birth_reason),
                    death_reason=str(death_reason),
                    child_plan=child_plan,
                    improvement_direction=str(improvement_direction),
                    death_only=death_only,
                    birth_only=birth_only,
                )
                bd_pairs.append(bd_pair)
            
            # Parse graph edit
            graph_edit_data = data.get("graph_edit", {})
            graph_edit = GraphEdit(
                remove_edges=graph_edit_data.get("remove_edges", []),
                add_edges=graph_edit_data.get("add_edges", []),
                type_changes=graph_edit_data.get("type_changes", []),
                rewire_notes=graph_edit_data.get("rewire_notes", []),
                subgraph_rewrites=graph_edit_data.get("subgraph_rewrites", []),
            )
            
            # Parse final synthesis
            synthesis_data = data.get("final_synthesis", {})
            final_synthesis = FinalSynthesis(
                contributors=synthesis_data.get("contributors", []),
                strategy=synthesis_data.get("strategy", ""),
                conflict_resolution=synthesis_data.get("conflict_resolution", ""),
                final_answer_spec=synthesis_data.get("final_answer_spec", ""),
            )
            
            # Parse time control
            time_control_data = data.get("time_control", {})
            time_control = TimeControl(
                birth_death_value=float(time_control_data.get("birth_death_value", 0.0)),
                graph_rewire_value=float(time_control_data.get("graph_rewire_value", 0.0)),
                fast_steps_next=int(time_control_data.get("fast_steps_next", 10)),
                trigger_birth_death=bool(time_control_data.get("trigger_birth_death", False)),
                trigger_graph_rewire=bool(time_control_data.get("trigger_graph_rewire", False)),
                trigger_next_slow_rule=time_control_data.get("trigger_next_slow_rule", "fixed"),
                cooldown=int(time_control_data.get("cooldown", 0)),
                continue_evolution=bool(time_control_data.get("continue_evolution", True)),
                stop_reason=str(time_control_data.get("stop_reason", "")),
            )
            
            decision = MetaDecision(
                slow_update=bool(data.get("slow_update", True)),
                confidence=float(data.get("confidence", 0.5)),
                birth_death_pairs=bd_pairs,
                graph_edit=graph_edit,
                final_synthesis=final_synthesis,
                time_control=time_control,
                agent_feedback=agent_feedback,
                global_rationale=data.get("global_rationale", []),
            )

            decision = self._enforce_prompt_variants(decision, system_summary)
            decision = self._enforce_graph_rewire(decision, system_summary)
            decision = self._ensure_no_isolated_nodes(decision, system_summary)
            decision = self._ensure_graph_edge_if_none(decision, system_summary)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error parsing meta LLM response: {e}")
            return self._get_default_decision(system_summary)

    def _enforce_graph_rewire(
        self,
        decision: MetaDecision,
        system_summary: SystemSummary,
    ) -> MetaDecision:
        """If connectivity/collaboration score is low, force a graph edit."""
        g_t = float(getattr(system_summary, "graph_connectivity_score", 0.0))
        if g_t >= 0.35:
            return decision

        decision.time_control.trigger_graph_rewire = True
        decision.time_control.graph_rewire_value = max(0.6, float(getattr(decision.time_control, "graph_rewire_value", 0.0)))

        # If meta didn't propose edges, add a minimal connectivity fix.
        if not (decision.graph_edit.add_edges or decision.graph_edit.remove_edges or decision.graph_edit.type_changes):
            nodes = list(system_summary.node_summaries or [])
            if len(nodes) < 2:
                return decision
            # pick best and worst by last_meta_score
            nodes_sorted = sorted(nodes, key=lambda n: float(getattr(n, "last_meta_score", 0.0)))
            worst = nodes_sorted[0]
            best = nodes_sorted[-1]
            second_best = nodes_sorted[-2] if len(nodes_sorted) >= 2 else None
            existing = {(e.source, e.target) for e in system_summary.edge_summaries}
            def _has_any_edge(a: str, b: str) -> bool:
                return (a, b) in existing or (b, a) in existing
            if second_best and not _has_any_edge(best.agent_id, second_best.agent_id):
                decision.graph_edit.add_edges.append((best.agent_id, second_best.agent_id, "evidence_flow"))
            elif not _has_any_edge(worst.agent_id, best.agent_id):
                decision.graph_edit.add_edges.append((worst.agent_id, best.agent_id, "evidence_flow"))
            elif second_best and not _has_any_edge(worst.agent_id, second_best.agent_id):
                decision.graph_edit.add_edges.append((worst.agent_id, second_best.agent_id, "verification_flow"))
        return decision

    def _ensure_no_isolated_nodes(
        self,
        decision: MetaDecision,
        system_summary: SystemSummary,
    ) -> MetaDecision:
        """If current graph has isolated nodes, force edges to connect them."""
        nodes = list(system_summary.node_summaries or [])
        if len(nodes) < 2:
            return decision

        connected = set()
        for e in system_summary.edge_summaries or []:
            connected.add(e.source)
            connected.add(e.target)

        isolated = [n for n in nodes if n.agent_id not in connected]
        if not isolated:
            return decision

        # Ensure a graph edit will be executed.
        decision.time_control.trigger_graph_rewire = True
        decision.time_control.graph_rewire_value = max(
            0.5,
            float(getattr(decision.time_control, "graph_rewire_value", 0.0)),
        )

        nodes_sorted = sorted(nodes, key=lambda n: float(getattr(n, "last_meta_score", 0.0)))
        best = nodes_sorted[-1]
        second_best = nodes_sorted[-2] if len(nodes_sorted) >= 2 else None
        existing = {(e.source, e.target) for e in system_summary.edge_summaries}

        def _has_any_edge(a: str, b: str) -> bool:
            return (a, b) in existing or (b, a) in existing

        for iso in isolated:
            if iso.agent_id == best.agent_id and second_best:
                if not _has_any_edge(iso.agent_id, second_best.agent_id):
                    decision.graph_edit.add_edges.append((iso.agent_id, second_best.agent_id, "evidence_flow"))
            else:
                if not _has_any_edge(iso.agent_id, best.agent_id):
                    decision.graph_edit.add_edges.append((iso.agent_id, best.agent_id, "evidence_flow"))
        return decision

    def _ensure_graph_edge_if_none(
        self,
        decision: MetaDecision,
        system_summary: SystemSummary,
    ) -> MetaDecision:
        """If meta provides no graph edits, add one safe edge to improve connectivity."""
        if (
            decision.graph_edit.add_edges
            or decision.graph_edit.remove_edges
            or decision.graph_edit.type_changes
            or decision.graph_edit.subgraph_rewrites
        ):
            return decision

        nodes = list(system_summary.node_summaries or [])
        if len(nodes) < 2:
            return decision

        # Ensure graph edit will execute.
        decision.time_control.trigger_graph_rewire = True
        decision.time_control.graph_rewire_value = max(
            0.3,
            float(getattr(decision.time_control, "graph_rewire_value", 0.0)),
        )

        nodes_sorted = sorted(nodes, key=lambda n: float(getattr(n, "last_meta_score", 0.0)))
        worst = nodes_sorted[0]
        best = nodes_sorted[-1]
        second_best = nodes_sorted[-2] if len(nodes_sorted) >= 2 else None

        existing = {(e.source, e.target) for e in system_summary.edge_summaries}
        def _has_any_edge(a: str, b: str) -> bool:
            return (a, b) in existing or (b, a) in existing

        # Prefer connecting two high-scoring nodes if they're disconnected.
        if second_best and not _has_any_edge(best.agent_id, second_best.agent_id):
            decision.graph_edit.add_edges.append((best.agent_id, second_best.agent_id, "evidence_flow"))
            return decision

        # Otherwise connect the lowest score to the highest score.
        if not _has_any_edge(worst.agent_id, best.agent_id):
            decision.graph_edit.add_edges.append((worst.agent_id, best.agent_id, "evidence_flow"))
            return decision

        # Fallback: connect lowest to second-best if needed.
        if second_best and not _has_any_edge(worst.agent_id, second_best.agent_id):
            decision.graph_edit.add_edges.append((worst.agent_id, second_best.agent_id, "verification_flow"))
        return decision

    def _enforce_prompt_variants(
        self,
        decision: MetaDecision,
        system_summary: SystemSummary,
    ) -> MetaDecision:
        """Force meta to specify prompt variants for duplicate roles."""
        nodes = {n.agent_id: n.role for n in (system_summary.node_summaries or [])}
        for pair in decision.birth_death_pairs:
            plan = getattr(pair, "child_plan", None)
            if plan is None:
                continue
            role_update = str(getattr(plan, "role_prompt_update", "") or "").strip()
            role_base = role_update.split("|", 1)[0].strip().lower() if role_update else ""
            if not role_base:
                continue
            # Count survivors with same role (exclude death target).
            existing = [
                rid for rid, role in nodes.items()
                if rid != pair.death_target_id and role == role_base
            ]
            if existing:
                # If no explicit variant specified, assign one deterministically.
                if "|" not in role_update:
                    variant_idx = len(existing) + 1
                    plan.role_prompt_update = f"{role_base}|alt{variant_idx}"
                    if plan.inherit_type == InheritanceType.PURE_CLONE:
                        plan.inherit_type = InheritanceType.MUTATE_ROLE
        return decision

    def _apply_gated_greedy_fallback(
        self,
        decision: MetaDecision,
        system_summary: SystemSummary,
        control_params: Optional[ControlParams] = None,
    ) -> MetaDecision:
        """Score-based greedy fallback: when meta LLM returns a structural no-op, force a
        minimal improvement using one of three generic structural moves:
        pure birth, pure death, or 1-for-1 replacement.

        Why only_birth / only_death were not appearing before:
        - the executor supported them,
        - the schema/prompt mentioned them,
        - but this fallback always emitted a replacement pair.
        So once meta returned a no-op, the runtime itself erased any chance of observing
        pure birth / pure death. This fallback now chooses among all three modes."""
        def _is_noop_birth_death(dec: MetaDecision) -> bool:
            if not dec.birth_death_pairs:
                return False
            nodes_by_id = {n.agent_id: n for n in (system_summary.node_summaries or [])}
            for pair in dec.birth_death_pairs:
                plan = getattr(pair, "child_plan", None)
                inherit_type = getattr(plan, "inherit_type", None)
                role_update = str(getattr(plan, "role_prompt_update", "") or "").strip()
                if inherit_type == InheritanceType.MUTATE_ROLE and role_update:
                    parent_role = getattr(nodes_by_id.get(pair.parent_id), "role", "") or ""
                    if role_update and role_update != parent_role:
                        return False
                if inherit_type == InheritanceType.HYBRID:
                    return False
            return True

        has_graph_action = bool(
            decision.graph_edit.remove_edges or decision.graph_edit.add_edges or decision.graph_edit.type_changes
        )
        has_structural_action = bool(decision.birth_death_pairs) or has_graph_action
        if has_structural_action and not _is_noop_birth_death(decision):
            return decision
        if has_structural_action and _is_noop_birth_death(decision):
            decision.global_rationale = list(decision.global_rationale or []) + [
                "Detected no-op birth-death (pure_clone / no role change); applying score-based replacement."
            ]

        nodes = list(system_summary.node_summaries or [])
        if len(nodes) < 2:
            return decision

        protected = set(getattr(system_summary, "protected_nodes", []) or [])
        candidates_for_death = [n for n in nodes if n.agent_id not in protected]
        if not candidates_for_death:
            return decision

        # Kill the agent with the lowest contribution score (break ties by worst trend).
        worst = min(candidates_for_death, key=lambda n: (n.last_meta_score, n.score_trend))

        # Pick parent: highest-scoring agent that is not the death target.
        remaining = [n for n in nodes if n.agent_id != worst.agent_id]
        if not remaining:
            return decision
        parent = max(remaining, key=lambda n: (n.last_meta_score, n.score_trend))

        allowed_roles = list(self.allowed_roles or ["planner", "searcher", "verifier", "calculator", "reflector"])
        role_counts: dict[str, int] = {}
        for n in nodes:
            role_counts[n.role] = role_counts.get(n.role, 0) + 1

        if control_params is not None:
            n_min = int(control_params.n_min)
            n_max = int(control_params.n_max)
            critical_roles = list(control_params.critical_roles or [])
            min_role_coverage = dict(control_params.min_role_coverage or {})
        else:
            n_min = max(1, len(allowed_roles))
            n_max = max(n_min, len(nodes) + 1)
            critical_roles = ["planner", "searcher", "verifier"]
            min_role_coverage = {role: 1 for role in critical_roles}

        current_n = len(nodes)
        role_counts_after_removing_worst = dict(role_counts)
        role_counts_after_removing_worst[worst.role] = max(
            0,
            role_counts_after_removing_worst.get(worst.role, 0) - 1,
        )

        missing_critical_roles = [
            role for role in critical_roles
            if role_counts.get(role, 0) < max(1, int(min_role_coverage.get(role, 1)))
        ]
        underrepresented_roles = sorted(
            allowed_roles,
            key=lambda role: (
                role_counts.get(role, 0),
                0 if role in critical_roles else 1,
                role,
            ),
        )
        child_role = underrepresented_roles[0]

        def _make_child_plan(role: str, target_role: str) -> ChildInheritancePlan:
            capability_need, capability_variant = self._default_capability_for_role(role)
            return ChildInheritancePlan(
                inherit_type=InheritanceType.MUTATE_ROLE,
                memory_from_parent_ratio=0.8,
                memory_from_target_ratio=0.2,
                capability_noise_scale=0.12,
                policy_mutation=(
                    f"Shift child focus toward capability={capability_need} variant={capability_variant} using {role} as the prototype role. "
                    f"Keep useful workflow fragments from parent={parent.role} and target={target_role}."
                ),
                role_prompt_update=role,
                capability_need=capability_need,
                capability_variant=capability_variant,
            )

        # Generic pure-birth trigger:
        # if the system is below max size and a critical or clearly underrepresented role is missing,
        # add capacity without deleting any survivor.
        birth_gap_role = missing_critical_roles[0] if missing_critical_roles else None
        can_pure_birth = current_n < n_max
        if can_pure_birth and (
            birth_gap_role is not None
            or (
                system_summary.current_answer_quality < 0.75
                and system_summary.learning_efficiency <= 0.05
                and role_counts.get(child_role, 0) == 0
            )
        ):
            birth_role = birth_gap_role or child_role
            fallback_pair = BirthDeathPair(
                parent_id=parent.agent_id,
                death_target_id="",
                birth_reason=(
                    f"score_fallback_birth_only: add missing_or_sparse role={birth_role} "
                    f"while preserving current graph state"
                ),
                death_reason="",
                child_plan=_make_child_plan(birth_role, worst.role),
                improvement_direction=(
                    f"Pure birth to expand coverage with role={birth_role} without losing any current capability."
                ),
                death_only=False,
                birth_only=True,
            )
            decision.birth_death_pairs = [fallback_pair]
            decision.time_control.trigger_birth_death = True
            decision.time_control.trigger_graph_rewire = False
            decision.global_rationale = list(decision.global_rationale or []) + [
                f"Score-based fallback selected pure birth: add {birth_role} from {parent.agent_id} "
                f"because current_N={current_n} < N_max={n_max} and role coverage is insufficient."
            ]
            return decision

        # Generic pure-death trigger:
        # if the system is above minimum size and the worst agent is both weak and safely redundant,
        # shrink instead of immediately replacing.
        can_pure_death = current_n > n_min
        preserves_critical_roles = all(
            role_counts_after_removing_worst.get(role, 0) >= max(1, int(min_role_coverage.get(role, 1)))
            for role in critical_roles
        )
        worst_role_count = role_counts.get(worst.role, 0)
        if can_pure_death and preserves_critical_roles and (
            worst.last_meta_score <= 0.12
            or (
                worst.last_meta_score <= 0.2
                and worst.score_trend <= 0.0
                and worst_role_count >= 2
                and system_summary.diversity_index < 0.8
            )
        ):
            fallback_pair = BirthDeathPair(
                parent_id=parent.agent_id,
                death_target_id=worst.agent_id,
                birth_reason="",
                death_reason=(
                    f"score_fallback_death_only: remove redundant low-signal agent "
                    f"role={worst.role}, score={worst.last_meta_score:.3f}"
                ),
                child_plan=_make_child_plan(parent.role, worst.role),
                improvement_direction=(
                    "Pure death to reduce redundancy/noise and simplify the coordination graph."
                ),
                death_only=True,
                birth_only=False,
            )
            decision.birth_death_pairs = [fallback_pair]
            decision.time_control.trigger_birth_death = True
            decision.time_control.trigger_graph_rewire = False
            decision.global_rationale = list(decision.global_rationale or []) + [
                f"Score-based fallback selected pure death: remove {worst.agent_id} "
                f"(role={worst.role}, score={worst.last_meta_score:.3f}) because it is weak, redundant, "
                f"and current_N={current_n} > N_min={n_min}."
            ]
            return decision

        # Default generic replacement: remove the weakest replaceable agent and redirect capacity
        # toward the most underrepresented role.
        fallback_pair = BirthDeathPair(
            parent_id=parent.agent_id,
            death_target_id=worst.agent_id,
            birth_reason=f"score_fallback_replacement: add under-represented role={child_role}",
            death_reason=f"score_fallback_replacement: lowest contribution_score={worst.last_meta_score:.3f}",
            child_plan=_make_child_plan(child_role, worst.role),
            improvement_direction=f"Replace low-scorer with {child_role} to improve coverage.",
            death_only=False,
            birth_only=False,
        )
        decision.birth_death_pairs = [fallback_pair]
        decision.time_control.trigger_birth_death = True
        decision.time_control.trigger_graph_rewire = False
        decision.global_rationale = list(decision.global_rationale or []) + [
            f"Score-based fallback selected replacement: replace {worst.agent_id} "
            f"(score={worst.last_meta_score:.3f}) with {child_role} child from {parent.agent_id} "
            f"(score={parent.last_meta_score:.3f})."
        ]
        return decision
    
    def _get_default_decision(self, system_summary: SystemSummary) -> MetaDecision:
        """Get a conservative default decision."""
        return MetaDecision(
            slow_update=False,
            confidence=0.0,
            birth_death_pairs=[],
            graph_edit=GraphEdit(
                remove_edges=[],
                add_edges=[],
                type_changes=[],
                rewire_notes=[],
                subgraph_rewrites=[],
            ),
            final_synthesis=FinalSynthesis(
                contributors=[],
                strategy="No synthesis",
                conflict_resolution="",
                final_answer_spec="",
            ),
            time_control=TimeControl(
                birth_death_value=0.0,
                graph_rewire_value=0.0,
                fast_steps_next=10,
                trigger_birth_death=False,
                trigger_graph_rewire=False,
                trigger_next_slow_rule="fixed",
                cooldown=0,
                continue_evolution=True,
                stop_reason="",
            ),
            agent_feedback=[
                AgentEvolutionFeedback(
                    agent_id=node.agent_id,
                    score=node.last_meta_score,
                    score_reason="default_conservative_decision",
                    workflow_correction="Keep current workflow until meta feedback is available.",
                    next_round_workflow=node.next_round_workflow or node.memory_summary,
                )
                for node in system_summary.node_summaries
            ],
            global_rationale=["Error in meta LLM call, using conservative default"],
        )
