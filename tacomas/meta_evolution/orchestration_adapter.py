"""
Orchestration Adapter
Bridges CentralizedMultiAgentSystem / OrchestrationResult with EvolutionController.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from tacomas.agents.multiagent_components.conversation import OrchestrationResult
    from tacomas.meta_evolution.evolution_controller import EvolutionController

from tacomas.meta_evolution.schemas import AgentState, EdgeType

logger = logging.getLogger(__name__)


_CANONICAL_ROLES = {
    "planner",
    "searcher",
    "calculator",
    "verifier",
    "reflector",
}

_ROLE_ALIASES = {
    "lead_agent": "planner",
    "lead": "planner",
    "coordinator": "planner",
    "orchestrator": "planner",
    "researcher": "searcher",
    "analyst": "searcher",
    "question_handler": "searcher",
    "worker": "searcher",
    "subagent": "searcher",
}


def _canonicalize_role(role: str) -> str:
    normalized = str(role or "").strip().lower()
    if normalized in _CANONICAL_ROLES:
        return normalized
    if normalized in _ROLE_ALIASES:
        return _ROLE_ALIASES[normalized]
    logger.warning("InitializationMetaLLM produced unknown role '%s', fallback to 'searcher'", role)
    return "searcher"


def update_from_orchestration_result(
    controller: "EvolutionController",
    result: "OrchestrationResult",
    overall_score: float,
    iteration: int,
) -> float:
    """Sync real run outputs back into the EvolutionController's population."""
    answer_quality = _score_from_answer(result.synthesized_answer, overall_score)
    _update_lead_agent(controller, result, answer_quality, iteration)
    _update_subagents(controller, result, answer_quality, iteration)
    return answer_quality


def _score_from_answer(answer: Optional[str], fallback: float) -> float:
    if not answer:
        return max(0.0, fallback)
    length_score = min(1.0, len(answer) / 600)
    if fallback > 0:
        return 0.6 * fallback + 0.4 * length_score
    return length_score


def _update_lead_agent(controller: "EvolutionController", result: "OrchestrationResult", answer_quality: float, iteration: int) -> None:
    agent = controller.population.get_agent("lead_agent")
    if agent is None:
        return
    agent.output = str(result.synthesized_answer or "")[:300]
    agent.recent_scores.append(answer_quality)
    agent.score_trend = _compute_trend(agent.recent_scores)
    agent.last_update_time = iteration
    agent.failure_modes = [] if len(agent.output) >= 50 else ["very_short_answer"]
    agent.meta_score = answer_quality
    agent.score_reason = "overall synthesized answer quality"
    controller.update_agent_state(agent)
    logger.debug(f"Updated lead_agent: score={answer_quality:.3f}")


def _update_subagents(controller: "EvolutionController", result: "OrchestrationResult", answer_quality: float, iteration: int) -> None:
    conv_items: List[Tuple[str, any]] = list(result.subagent_conversations.items())
    for idx, (raw_id, conv) in enumerate(conv_items):
        evo_id = f"subagent_{idx}"
        agent = controller.population.get_agent(evo_id)
        if agent is None:
            continue
        agent.output = (conv.last_outgoing_external_message or "")[:300]
        n_turns = conv.total_iterations
        agent_score = max(0.1, answer_quality - 0.04 * max(0, n_turns - 6))
        agent.recent_scores.append(agent_score)
        agent.score_trend = _compute_trend(agent.recent_scores)
        agent.last_update_time = iteration
        env_status = result.subagent_env_status.get(raw_id)
        if env_status and not env_status.success:
            agent.improvement_direction = "Task not completed — improve tool use."
            agent.failure_modes = ["task_not_completed"]
        else:
            agent.improvement_direction = "Maintain quality; reduce iterations."
            agent.failure_modes = []
        agent.meta_score = agent.recent_scores[-1] if agent.recent_scores else 0.0
        agent.score_reason = "runtime contribution estimate"
        findings = result.subagent_findings.get(raw_id, [])
        agent.neighbor_externality = min(1.0, len(findings) * 0.2)
        controller.update_agent_state(agent)


def _compute_trend(scores: List[float], window: int = 5) -> float:
    recent = scores[-window:]
    if len(recent) < 2:
        return 0.0
    n = len(recent)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(recent) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, recent))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den else 0.0


class InitializationMetaLLM:
    """Calls meta LLM at t=0 to design initial population, roles, and graph."""

    SYSTEM_PROMPT = """You are designing the initial multi-agent population and communication graph.
Your goal is to choose the initial number of agents, role assignment, role prompts, and graph topology.
Infer the division of labour and topology from the task itself. Output valid JSON only.

IMPORTANT:
- If you include duplicate roles, you MUST assign a distinct prompt variant to each duplicate.
- Use the format: "role|alt2", "role|alt3", ... (e.g. "searcher|alt2")."""

    DEVELOPER_PROMPT = """Return exactly one JSON object:
{
  "total_agents": <int>,
  "agents": [{"id": "agent_0", "role": "<role>", "init_prompt": "<prompt>"}],
  "edges": [{"source": "agent_0", "target": "agent_1", "type": "directed"}],
  "rationale": ["..."]
}
Constraints:
- total_agents within [N_min, N_max], which will be provided by user later
- You DO NOT need to use all canonical roles in one initialization
- Any role may appear multiple times if task decomposition needs it
- Prioritize choosing the right role mix under [N_min, N_max] instead of maximizing role variety
- Use canonical roles only: planner, searcher, calculator, verifier, reflector
 - If you repeat a role, include a prompt variant suffix: role|alt2, role|alt3, ...

Role design guide:
- planner: decompose task, define evidence plan, assign execution order
- searcher: retrieve external facts/documents with tools
- calculator: compute metrics and verify numerical consistency
- verifier: challenge claims, detect conflicts, enforce source-grounded checks
- reflector: inspect failure modes and propose next-round strategy patches

Edge mechanism:
- directed edge A->B means generic one-way handoff when no stronger semantic is known
- bidirectional edge A<->B means debate/cross-check loop (both can critique each other)
- evidence_flow means primary evidence or source handoff
- verification_flow means validation or correction handoff
- computation_flow means numerical or structured transformation handoff
- reflection_feedback means auxiliary corrective feedback, not usually the main backbone
- use semantic edge types when helpful, but keep them general so they transfer across tasks"""

    def __init__(self, llm_config):
        self.llm_config = llm_config
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self.llm_config.get_llm()
        return self._llm

    def initialize(self, controller: "EvolutionController", task_description: str, n_min: int = 3, n_max: int = 8) -> Dict:
        user_payload = f"""Task: {task_description}
Agents: [{n_min}, {n_max}]
Please design the initial multi-agent system."""
        try:
            response = self.llm.invoke([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "system", "content": self.DEVELOPER_PROMPT},
                {"role": "user", "content": user_payload},
            ])
            init_dict = self._parse(response.content)
        except Exception as e:
            logger.error(f"InitializationMetaLLM failed ({e}), using default")
            init_dict = self._default_init(n_min)
        raw_n = init_dict.get("total_agents")
        init_dict = self._sanitize_init_dict(init_dict, n_min=n_min, n_max=n_max)
        fixed_n = init_dict.get("total_agents")
        if raw_n != fixed_n:
            logger.warning(
                "InitializationMetaLLM agent count adjusted by sanitizer: raw_total_agents=%s, enforced_total_agents=%s, range=[%s,%s]",
                raw_n,
                fixed_n,
                n_min,
                n_max,
            )
        self._register(controller, init_dict)
        logger.info(f"InitializationMetaLLM: {init_dict.get('total_agents')} agents")
        return init_dict

    def _sanitize_init_dict(self, init_dict: Dict, n_min: int, n_max: int) -> Dict:
        """Enforce initialization constraints even when LLM output violates them."""
        raw_agents = init_dict.get("agents") or []
        if not isinstance(raw_agents, list):
            raw_agents = []

        desired_n = init_dict.get("total_agents", len(raw_agents))
        try:
            desired_n = int(desired_n)
        except Exception:
            desired_n = len(raw_agents)
        desired_n = max(n_min, min(n_max, desired_n))

        normalized_agents: List[Dict[str, str]] = []
        used_ids: set[str] = set()
        role_counts: Dict[str, int] = {}
        for idx, spec in enumerate(raw_agents):
            if not isinstance(spec, dict):
                continue
            raw_role = str(spec.get("role", "searcher")).strip()
            role_base = _canonicalize_role(raw_role.split("|", 1)[0])
            aid = str(spec.get("id", f"agent_{idx}")).strip() or f"agent_{idx}"
            if aid in {"lead_agent", "orchestrator"}:
                aid = f"agent_{idx}"
            if aid in used_ids:
                aid = f"agent_{idx}"
            used_ids.add(aid)
            role_counts[role_base] = role_counts.get(role_base, 0) + 1
            count = role_counts[role_base]
            if count > 1 and "|" not in raw_role:
                role = f"{role_base}|alt{count}"
            else:
                role = raw_role or role_base
            normalized_agents.append(
                {
                    "id": aid,
                    "role": role,
                    "init_prompt": str(spec.get("init_prompt", "")).strip(),
                }
            )

        fallback_roles = ["planner", "searcher", "verifier", "calculator", "reflector"]
        while len(normalized_agents) < desired_n:
            idx = len(normalized_agents)
            role_base = fallback_roles[idx % len(fallback_roles)]
            role_counts[role_base] = role_counts.get(role_base, 0) + 1
            count = role_counts[role_base]
            role = f"{role_base}|alt{count}" if count > 1 else role_base
            normalized_agents.append(
                {
                    "id": f"agent_{idx}",
                    "role": role,
                    "init_prompt": f"You are the {role}.",
                }
            )

        normalized_agents = normalized_agents[:desired_n]

        alive_ids = {a["id"] for a in normalized_agents}

        raw_edges = init_dict.get("edges") or []
        sanitized_edges: List[Dict[str, str]] = []
        if isinstance(raw_edges, list):
            for edge in raw_edges:
                if not isinstance(edge, dict):
                    continue
                src = str(edge.get("source", "")).strip()
                tgt = str(edge.get("target", "")).strip()
                et = str(edge.get("type", "directed")).strip().lower() or "directed"
                if src in alive_ids and tgt in alive_ids and src != tgt and et in {edge.value for edge in EdgeType}:
                    sanitized_edges.append({"source": src, "target": tgt, "type": et})

        # Serial add: if a node has no edge with any prior node, connect it to maintain flow.
        seen: List[str] = []
        for agent in normalized_agents:
            aid = agent["id"]
            if not seen:
                seen.append(aid)
                continue
            has_edge = any(
                (e["source"] == aid and e["target"] in seen)
                or (e["target"] == aid and e["source"] in seen)
                for e in sanitized_edges
            )
            if not has_edge:
                sanitized_edges.append({"source": seen[-1], "target": aid, "type": "evidence_flow"})
            seen.append(aid)

        return {
            "total_agents": len(normalized_agents),
            "agents": normalized_agents,
            "edges": sanitized_edges,
            "rationale": init_dict.get("rationale", []),
        }

    def _parse(self, text: str) -> Dict:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        return json.loads(text[start:end])

    def _register(self, controller: "EvolutionController", init_dict: Dict) -> None:
        import time as _time
        now = int(_time.time())
        raw_agents = init_dict.get("agents", [])
        id_remap: Dict[str, str] = {}
        reserved = {"lead_agent", "orchestrator"}

        for idx, spec in enumerate(raw_agents):
            old_id = str(spec.get("id", f"agent_{idx}"))
            if old_id in reserved:
                id_remap[old_id] = f"agent_{idx}"

        for idx, spec in enumerate(raw_agents):
            old_id = str(spec.get("id", f"agent_{idx}"))
            new_id = id_remap.get(old_id, old_id)
            raw_role = str(spec.get("role", "searcher")).strip()
            role_base = _canonicalize_role(raw_role.split("|", 1)[0])
            prompt_variant = ""
            if "|" in raw_role:
                _, right = raw_role.split("|", 1)
                tag = right.strip().lower()
                if tag:
                    prompt_variant = f"{role_base}_{tag}" if not tag.startswith(role_base) else tag
            agent = AgentState(
                agent_id=new_id, role=role_base,
                policy=spec.get("init_prompt", ""), output="", memory_summary="",
                capability={"reasoning": 0.7, "tool_use": 0.6},
                recent_scores=[0.6], score_trend=0.0, improvement_direction="",
                neighbor_externality=0.0, redundancy_score=0.0, failure_modes=[],
                bridge_value=0.0, community_id=0, creation_time=now, last_update_time=now,
            )
            agent.prompt_variant = prompt_variant
            controller.add_agent(agent)

        for e in init_dict.get("edges", []):
            try:
                src = id_remap.get(str(e.get("source", "")), str(e.get("source", "")))
                tgt = id_remap.get(str(e.get("target", "")), str(e.get("target", "")))
                controller.add_edge(src, tgt, e.get("type", "directed"))
            except Exception:
                pass

    def _default_init(self, n_agents: int) -> Dict:
        if os.getenv("UNREASONABLE_INIT", "0") == "1":
            # Extremely unreasonable topology:
            # - 1 searcher node isolated from the rest
            # - remaining planners arranged in a ring without evidence flow to verifier/reflector
            n = max(int(n_agents), 3)
            agents: List[Dict[str, str]] = []
            agents.append(
                {
                    "id": f"agent_{n-1}",
                    "role": "searcher",
                    "init_prompt": "You are the only evidence retriever, but you are isolated from the reasoning chain.",
                }
            )
            for aid_i in range(0, n - 1):
                role = "planner" if aid_i < n - 2 else "verifier"
                agents.append(
                    {
                        "id": f"agent_{aid_i}",
                        "role": role,
                        "init_prompt": f"You are the {role}. Work with limited evidence flow.",
                    }
                )

            planner_like_ids = [a["id"] for a in agents if a["role"] in {"planner", "verifier"}]
            edges: List[Dict[str, str]] = []
            if len(planner_like_ids) >= 2:
                for idx, src in enumerate(planner_like_ids):
                    tgt = planner_like_ids[(idx + 1) % len(planner_like_ids)]
                    if src != tgt:
                        edges.append({"source": src, "target": tgt, "type": "directed"})

            return {
                "total_agents": len(agents),
                "agents": agents,
                "edges": edges,
                "rationale": ["Unreasonable init: isolated searcher and weak evidence flow."],
            }

        import random as _random

        # Fixed base pool: 5 diverse roles, no duplicates.
        # Each covers a distinct responsibility so there is no wasted overlap.
        _BASE_ROLES = [
            {
                "id": "agent_0",
                "role": "planner",
                "init_prompt": (
                    "Decompose the task into concrete subtasks. Identify required data sources "
                    "(SEC EDGAR 10-K/10-Q, earnings press releases). "
                    "Output an ordered execution plan with expected tool calls and evidence targets."
                ),
            },
            {
                "id": "agent_1",
                "role": "searcher",
                "init_prompt": (
                    "Retrieve high-signal evidence. For financial tasks: use edgar_search to find "
                    "the correct filing URL, then parse_html_page to store it, then retrieve_information "
                    "with targeted prompts. Always extract concrete numbers and cite the source section."
                ),
            },
            {
                "id": "agent_2",
                "role": "verifier",
                "init_prompt": (
                    "Cross-validate the searcher's findings against the raw document. "
                    "Confirm each claimed adjustment item exists in the filing. "
                    "Output PASS/FAIL with the exact quoted text as evidence."
                ),
            },
            {
                "id": "agent_3",
                "role": "calculator",
                "init_prompt": (
                    "Given extracted financial data, compute and reconcile numeric figures. "
                    "Use retrieve_information to pull specific tables or line items from stored documents. "
                    "Show step-by-step arithmetic and flag any discrepancies."
                ),
            },
            {
                "id": "agent_4",
                "role": "reflector",
                "init_prompt": (
                    "Review the current system output and identify gaps or errors. "
                    "If the answer is incomplete, specify which exact document section is missing "
                    "and suggest the precise tool call sequence to fill the gap next round."
                ),
            },
        ]
        # Generic prompts for extra slots when n_agents > 5.
        # Roles are sampled without replacement from the canonical set to avoid duplicates.
        _EXTRA_ROLE_PROMPTS: Dict[str, str] = {
            "planner": (
                "Decompose the task into subtasks with alternative decompositions. "
                "Focus on cross-checking the primary planner's logic and identifying missed angles."
            ),
            "searcher": (
                "Search for evidence the primary searcher may have missed. "
                "Prefer sources the first searcher has not yet queried (e.g. if primary used EDGAR, try web/press)."
            ),
            "verifier": (
                "Verify consistency between numeric figures from different sources. "
                "Flag any conflict between the primary verifier's conclusion and raw document text."
            ),
            "calculator": (
                "Perform alternative numerical reconciliation to double-check earlier computations. "
                "Show work step-by-step and note any discrepancy."
            ),
            "reflector": (
                "Reflect on the overall reasoning chain for logical gaps or missing evidence. "
                "Propose a concrete corrective action for the next round."
            ),
        }

        n = max(int(n_agents), 3)
        agents: List[Dict[str, str]] = _BASE_ROLES[:min(n, len(_BASE_ROLES))]

        if n > len(_BASE_ROLES):
            # Roles already used in the base pool.
            used_roles = {a["role"] for a in agents}
            # Sample extra roles from canonical set, preferring under-represented ones.
            available = [r for r in _EXTRA_ROLE_PROMPTS if r not in used_roles]
            if not available:
                available = list(_EXTRA_ROLE_PROMPTS.keys())
            extra_roles = _random.sample(available, min(n - len(_BASE_ROLES), len(available)))
            for idx, role in enumerate(extra_roles):
                agents.append(
                    {
                        "id": f"agent_{len(_BASE_ROLES) + idx}",
                        "role": role,
                        "init_prompt": _EXTRA_ROLE_PROMPTS[role],
                    }
                )

        # Build edges: planner → all others (directed handoff).
        # searcher ↔ verifier (bidirectional debate).
        # calculator reads from searcher.
        # reflector reads from verifier.
        agent_ids = [a["id"] for a in agents]
        roles = {a["id"]: a["role"] for a in agents}

        edges: List[Dict[str, str]] = []
        planner_ids = [aid for aid in agent_ids if roles[aid] == "planner"]
        searcher_ids = [aid for aid in agent_ids if roles[aid] == "searcher"]
        verifier_ids = [aid for aid in agent_ids if roles[aid] == "verifier"]
        calculator_ids = [aid for aid in agent_ids if roles[aid] == "calculator"]
        reflector_ids = [aid for aid in agent_ids if roles[aid] == "reflector"]

        # Planner → searchers + verifier (generic setup handoff)
        for p in planner_ids:
            for s in searcher_ids + verifier_ids:
                edges.append({"source": p, "target": s, "type": "directed"})

        # Searcher ↔ Verifier (bidirectional debate)
        for s in searcher_ids:
            for v in verifier_ids:
                edges.append({"source": s, "target": v, "type": "bidirectional"})

        # Searcher → Calculator (evidence handoff)
        for s in searcher_ids[:1]:
            for c in calculator_ids:
                edges.append({"source": s, "target": c, "type": "evidence_flow"})

        # Verifier → Reflector (verification handoff)
        for v in verifier_ids[:1]:
            for r in reflector_ids:
                edges.append({"source": v, "target": r, "type": "verification_flow"})

        # Deduplicate edges
        seen = set()
        unique_edges = []
        for e in edges:
            key = (e["source"], e["target"], e["type"])
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        return {
            "total_agents": len(agents),
            "agents": agents,
            "edges": unique_edges,
            "rationale": [f"Default init with {len(agents)} agents across diverse roles."],
        }


class FinalSynthesisAdapter:
    """Extracts meta LLM final_synthesis.final_answer_spec."""

    def __init__(self, controller: "EvolutionController"):
        self.controller = controller

    def get_final_answer(self, meta_decision, orchestration_result, fallback_answer: str = "") -> str:
        spec = getattr(getattr(meta_decision, "final_synthesis", None), "final_answer_spec", "")
        if spec and spec.strip():
            logger.info("Using meta LLM final_answer_spec")
            return spec.strip()
        if orchestration_result is not None:
            synth = getattr(orchestration_result, "synthesized_answer", None)
            if synth and synth.strip():
                return synth.strip()
        return fallback_answer

    def should_override(self, meta_decision) -> bool:
        spec = getattr(getattr(meta_decision, "final_synthesis", None), "final_answer_spec", "")
        return bool(spec and len(spec.strip()) > 20)


class AdaptiveTimeScaleManager:
    """Manages cooldown periods and adaptive fast_steps_next."""

    def __init__(self, initial_bd_interval: int = 5, initial_graph_interval: int = 20):
        self.bd_interval = initial_bd_interval
        self.graph_interval = initial_graph_interval
        self.bd_cooldown = 0
        self.graph_cooldown = 0

    def update_after_slow_update(self, meta_decision, perf_before: float, perf_after: float) -> None:
        """Update cooldown based on performance change."""
        delta = perf_after - perf_before
        if meta_decision.time_control.trigger_birth_death:
            self.bd_cooldown = self.bd_interval * 2 if delta < -0.05 else 0
        if meta_decision.time_control.trigger_graph_rewire:
            self.graph_cooldown = self.graph_interval * 3 if delta < -0.05 else 0

    def step(self) -> None:
        if self.bd_cooldown > 0:
            self.bd_cooldown -= 1
        if self.graph_cooldown > 0:
            self.graph_cooldown -= 1

    def should_allow_bd(self) -> bool:
        return self.bd_cooldown <= 0

    def should_allow_graph(self) -> bool:
        return self.graph_cooldown <= 0

    def get_next_bd_interval(self, meta_suggestion: int) -> int:
        return int(0.7 * self.bd_interval + 0.3 * meta_suggestion) if meta_suggestion > 0 else self.bd_interval

    def get_next_graph_interval(self, meta_suggestion: int) -> int:
        return int(0.7 * self.graph_interval + 0.3 * meta_suggestion) if meta_suggestion > 0 else self.graph_interval
