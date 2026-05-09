"""
Runtime MAS executor driven by EvolutionController and BaseWorkerAgent.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from enum import Enum

from tacomas.agents.base import BaseAgentWithTools
from tacomas.config.llm import LLMConfig
from tacomas.datasets import DatasetInstance
from tacomas.datasets.base import DatasetEnvStatus, DatasetInstanceOutputWithTrajectory
from tacomas.config.prompts import Prompt

from .mas_base_agent import BaseWorkerAgent
from .schemas import EdgeType, AgentEvolutionFeedback

logger = logging.getLogger(__name__)


_DEFAULT_ROLE_PROMPT_PATHS = {
    "planner": "prompts/tacomas/planner.yaml",
    "planner_alt": "prompts/tacomas/planner_alt.yaml",
    "searcher": "prompts/tacomas/searcher.yaml",
    "searcher_alt": "prompts/tacomas/searcher_alt.yaml",
    "researcher": "prompts/tacomas/searcher.yaml",
    "calculator": "prompts/tacomas/calculator.yaml",
    "calculator_alt": "prompts/tacomas/calculator_alt.yaml",
    "verifier": "prompts/tacomas/verifier.yaml",
    "verifier_alt": "prompts/tacomas/verifier_alt.yaml",
    "schema_verifier": "prompts/tacomas/verifier_alt.yaml",
    "reflector": "prompts/tacomas/reflector.yaml",
    "reflector_alt": "prompts/tacomas/reflector_alt.yaml",
    "researcher": "prompts/tacomas/searcher.yaml",
    "analyst": "prompts/tacomas/searcher.yaml",
    "curator": "prompts/tacomas/searcher_alt.yaml",
    "auditor": "prompts/tacomas/verifier.yaml",
    "critic": "prompts/tacomas/verifier_alt.yaml",
    "forecaster": "prompts/tacomas/calculator_alt.yaml",
    "synthesizer": "prompts/tacomas/reflector_alt.yaml",
    "worker": "prompts/multi-agent/independent_worker.yaml",
    "subagent": "prompts/multi-agent/independent_worker.yaml",
}

_DEFAULT_ROLE_PERMISSIONS = {
    "planner": [],
    "searcher": ["prepare_primary_filing", "web_search", "edgar_search", "parse_html_page", "retrieve_information"],
    "researcher": ["prepare_primary_filing", "web_search", "edgar_search", "parse_html_page", "retrieve_information"],
    "calculator": ["retrieve_information"],
    "verifier": ["retrieve_information"],
    "schema_verifier": ["retrieve_information"],
    "reflector": [],
    "worker": ["prepare_primary_filing", "web_search", "edgar_search", "parse_html_page", "retrieve_information"],
    "subagent": ["prepare_primary_filing", "web_search", "edgar_search", "parse_html_page", "retrieve_information"],
}

_CANONICAL_ROLE_MAP = {
    "lead_agent": "planner",
    "lead": "planner",
    "coordinator": "planner",
    "orchestrator": "planner",
    "question_handler": "searcher",
    "analyst": "searcher",
    "researcher": "searcher",
    "curator": "searcher",
    "source_curator": "searcher",
    "evidence_curator": "searcher",
    "worker": "searcher",
    "subagent": "searcher",
    "auditor": "verifier",
    "schema_verifier": "verifier",
    "critic": "verifier",
    "fact_checker": "verifier",
    "validator": "verifier",
    "forecaster": "calculator",
    "modeler": "calculator",
    "synthesizer": "reflector",
    "writer": "reflector",
    "summarizer": "reflector",
}


def _canonicalize_role(role: str) -> str:
    normalized = str(role or "").strip().lower()
    if not normalized:
        return "searcher"
    if normalized in _DEFAULT_ROLE_PERMISSIONS:
        return normalized
    if normalized in _CANONICAL_ROLE_MAP:
        return _CANONICAL_ROLE_MAP[normalized]
    logger.warning("Unknown role '%s' mapped to 'searcher'", role)
    return "searcher"


def _load_yaml_file(path: str) -> Dict[str, Any]:
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except FileNotFoundError:
        logger.warning("Role permissions file not found: %s", path)
        return {}
    except Exception as exc:
        logger.warning("Failed to load YAML %s: %s", path, exc)
        return {}


def _build_role_prompt_map(prompt_paths: Dict[str, str]) -> Dict[str, Prompt]:
    prompt_map: Dict[str, Prompt] = {}
    for role, local_path in prompt_paths.items():
        try:
            prompt_map[role] = Prompt(name=role, local_path=local_path)
        except Exception as exc:
            logger.warning("Failed to load role prompt %s from %s: %s", role, local_path, exc)
    return prompt_map


def _merge_worker_prompt_map(role_prompts: Dict[str, Prompt]) -> Dict[str, Prompt]:
    """Ensure BaseWorkerAgent always receives a base_agent prompt key."""
    merged = dict(role_prompts)
    if "base_agent" not in merged:
        fallback = merged.get("subagent") or merged.get("searcher")
        if fallback is not None:
            merged["base_agent"] = fallback
    return merged


class MetaEvolutionMASRuntime:
    """Execute dynamic MAS rounds directly from the controller's population and graph."""

    def __init__(
        self,
        controller,
        template_agent: BaseAgentWithTools,
        task_instance: DatasetInstance,
        task_description: str,
        min_iterations_per_agent: int = 1,
        max_iterations_per_agent: int = 2,
        max_fast_rounds: int = 8,
        target_answer_quality: float = 0.82,
        role_prompt_paths: Optional[Dict[str, str]] = None,
        role_permissions_path: str = "run_conf/agent/tacomas-role-permissions.yaml",
        role_permissions: Optional[Dict[str, list[str]]] = None,
    ):
        self.controller = controller
        self.template_agent = template_agent
        self.task_instance = task_instance
        self.task_description = task_description
        self.dataset_id = os.getenv("DATASET_ID", "").strip() or "finance-benchmark"
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.max_fast_rounds = max_fast_rounds
        self.target_answer_quality = target_answer_quality
        prompt_paths = role_prompt_paths or _DEFAULT_ROLE_PROMPT_PATHS
        self.role_prompts = _merge_worker_prompt_map(_build_role_prompt_map(prompt_paths))
        loaded_role_cfg = _load_yaml_file(role_permissions_path)
        loaded_permissions = loaded_role_cfg.get("role_permissions") or {}
        # Merge order: defaults < yaml < explicit args
        self.role_permissions = {
            **_DEFAULT_ROLE_PERMISSIONS,
            **loaded_permissions,
            **(role_permissions or {}),
        }

        self.runtime_agents: Dict[str, BaseWorkerAgent] = {}
        self.latest_outputs: Dict[str, str] = {}
        self.latest_artifacts: Dict[str, Dict[str, Any]] = {}
        self.latest_queries: Dict[str, str] = {}
        self.latest_env_status: Dict[str, DatasetEnvStatus] = {}
        self.latest_tool_names: Dict[str, list[str]] = {}

        self.round_idx = 0
        self.current_answer_quality = 0.0
        self.final_answer = ""
        self.best_final_answer = ""        # best answer seen across all rounds
        self.best_answer_quality = 0.0     # quality score of best_final_answer
        self.best_graph_snapshot: Dict[str, Any] = {}
        self.best_protected_agent_ids: set[str] = set()
        self.best_protected_edges: list[Dict[str, Any]] = []
        self.final_notes = ""
        self.last_meta_decision = None
        self.latest_agent_feedback: Dict[str, AgentEvolutionFeedback] = {}
        self.stop_reason = ""
        self.last_answer_eval_metrics: Dict[str, Any] = {"score": 0.0, "reason": "init"}
        self.fast_round_logs: list[Dict[str, Any]] = []
        self.slow_update_logs: list[Dict[str, Any]] = []
        self.judge_llm = self._build_judge_llm()
        self.task_profile = self._infer_task_profile()
        self.task_schema = self._infer_task_schema()
        self.direct_stop_quality = float(self.task_profile.get("direct_stop_quality", 0.999) or 0.999)
        self.target_answer_quality = max(
            float(self.target_answer_quality),
            float(self.task_profile.get("target_quality_floor", self.target_answer_quality) or self.target_answer_quality),
        )
        self.coverage_schema = self._infer_coverage_schema()
        self.coverage_memory: Dict[str, Dict[str, Any]] = {
            slot["slot_id"]: dict(slot)
            for slot in self.coverage_schema.get("required_slots", [])
        }
        self.source_feedback_memory: Dict[str, Any] = {
            "preferred_source_keys": [],
            "rejected_sources": [],
            "rejection_reasons": [],
            "next_search_hints": [],
            "supported_claims": [],
            "unsupported_claims": [],
            "contradicted_claims": [],
            "missing_support": [],
            "required_constraints": {},
            "avoid_constraints": {},
            "prefer_source_patterns": [],
            "avoid_source_patterns": [],
            "missing_targets": [],
        }
        self.information_objects: list[Dict[str, Any]] = []
        self.agent_object_memory: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
        self.dataset_skill_text = self._load_dataset_skill_text()

        self._sync_runtime_agents()

    def _load_dataset_skill_text(self) -> str:
        skill_map = {
            "finance-benchmark": "skill/finance_benchmark_playbook/SKILL.md",
            "workbench": "skill/workbench_playbook/SKILL.md",
            "plancraft": "skill/plancraft_playbook/SKILL.md",
            "browsecomp-plus": "skill/browsecomp_plus_playbook/SKILL.md",
        }
        rel_path = skill_map.get(self.dataset_id)
        if not rel_path:
            return ""
        try:
            root = Path(__file__).resolve().parents[2]
            skill_path = root / rel_path
            if not skill_path.exists():
                return ""
            text = skill_path.read_text(encoding="utf-8").strip()
            return text[:6000]
        except Exception as exc:
            logger.warning("Failed to load dataset skill for %s: %s", self.dataset_id, exc)
            return ""

    def _slot_id(self, dimensions: Dict[str, Any]) -> str:
        parts = []
        for key in sorted((dimensions or {}).keys()):
            value = str(dimensions.get(key, "")).strip()
            if value:
                parts.append(f"{key}={value}")
        return "|".join(parts) if parts else "slot"

    def _heuristic_task_profile(self) -> Dict[str, Any]:
        text = self._instance_text().lower()
        years = set(re.findall(r"\b20\d{2}\b", text))
        for start, end in re.findall(r"\b(20\d{2})\s*(?:-|–|—|to|through|until)\s*(20\d{2})\b", text, flags=re.IGNORECASE):
            lo, hi = sorted((int(start), int(end)))
            if hi - lo <= 15:
                years.update(str(y) for y in range(lo, hi + 1))
        has_compare = any(tok in text for tok in ["compare", "versus", "vs", "beat", "exceed", "difference", "delta", "change", "trend"])
        has_compute = any(tok in text for tok in ["calculate", "compute", "ratio", "margin", "growth", "bps", "percentage", "average"])
        has_extract = any(tok in text for tok in ["list", "extract", "identify", "reconciliation", "adjustments", "components"])
        has_multi_hop = any(tok in text for tok in ["why", "how did", "explain", "drivers", "because", "impact"])
        if len(years) >= 3 or has_extract:
            primary = "structured_extraction"
            complexity = "high" if len(years) >= 4 else "medium"
            bias = "exploratory"
        elif has_multi_hop:
            primary = "multi_hop_synthesis"
            complexity = "high"
            bias = "balanced"
        elif has_compare and has_compute:
            primary = "quantitative_reasoning"
            complexity = "medium"
            bias = "balanced"
        elif has_compare:
            primary = "comparative_analysis"
            complexity = "medium"
            bias = "balanced"
        elif has_compute:
            primary = "quantitative_reasoning"
            complexity = "medium"
            bias = "balanced"
        else:
            primary = "fact_lookup"
            complexity = "low"
            bias = "conservative"
        return {
            "primary_type": primary,
            "complexity": complexity,
            "evidence_shape": "tabular_or_multi_slot" if primary == "structured_extraction" else "point_or_narrative",
            "evolution_bias": bias,
            "target_quality_floor": 0.9 if primary == "fact_lookup" else 0.85,
            "direct_stop_quality": float(os.getenv("DIRECT_STOP_QUALITY", "0.999")),
            "profile_source": "heuristic",
            "reason": "generic heuristic task typing",
        }

    def _coerce_profile_choice(
        self,
        raw_value: Any,
        allowed: set[str],
        fallback: str,
    ) -> str:
        value = str(raw_value or "").strip().lower()
        return value if value in allowed else fallback

    def _coerce_profile_float(
        self,
        raw_value: Any,
        fallback: float,
        *,
        field_name: str,
    ) -> float:
        if raw_value in (None, ""):
            return float(fallback)
        try:
            return max(0.0, min(1.0, float(raw_value)))
        except (TypeError, ValueError):
            pass

        normalized = str(raw_value).strip().lower().replace("-", "_").replace(" ", "_")
        field_maps = {
            "target_quality_floor": {
                "exact_match": 0.98,
                "precise": 0.95,
                "high_confidence": 0.93,
                "supported_answer": 0.9,
                "answer_with_reasoning": 0.9,
                "first_valid_answer": 0.85,
            },
            "direct_stop_quality": {
                "exact_match": 0.995,
                "precise": 0.99,
                "high_confidence": 0.985,
                "supported_answer": float(fallback),
                "answer_with_reasoning": float(fallback),
                "first_valid_answer": float(fallback),
            },
        }
        mapped = field_maps.get(field_name, {}).get(normalized)
        if mapped is not None:
            return max(0.0, min(1.0, float(mapped)))
        return float(fallback)

    def _infer_task_profile(self) -> Dict[str, Any]:
        fallback = self._heuristic_task_profile()
        llm = getattr(getattr(self.controller, "meta_llm", None), "llm", None) or self.judge_llm
        if llm is None:
            return fallback
        prompt = (
            "Classify the task into a general workflow profile. "
            "Use only these labels: fact_lookup, structured_extraction, quantitative_reasoning, "
            "comparative_analysis, multi_hop_synthesis. "
            "Return strict JSON with keys primary_type, complexity, evidence_shape, evolution_bias, "
            "target_quality_floor, direct_stop_quality, reason.\n\n"
            f"Task:\n{self._instance_text()}"
        )
        try:
            resp = llm.invoke([{"role": "user", "content": prompt}], num_retries=0)
            payload = self._extract_json_payload(getattr(resp, "content", "") or "")
            if payload:
                profile = {
                    "primary_type": self._coerce_profile_choice(
                        payload.get("primary_type", fallback["primary_type"]),
                        {"fact_lookup", "structured_extraction", "quantitative_reasoning", "comparative_analysis", "multi_hop_synthesis"},
                        fallback["primary_type"],
                    ),
                    "complexity": self._coerce_profile_choice(
                        payload.get("complexity", fallback["complexity"]),
                        {"low", "medium", "high"},
                        fallback["complexity"],
                    ),
                    "evidence_shape": self._coerce_profile_choice(
                        payload.get("evidence_shape", fallback["evidence_shape"]),
                        {"point_or_narrative", "tabular_or_multi_slot", "mixed"},
                        fallback["evidence_shape"],
                    ),
                    "evolution_bias": self._coerce_profile_choice(
                        payload.get("evolution_bias", fallback["evolution_bias"]),
                        {"conservative", "balanced", "exploratory"},
                        fallback["evolution_bias"],
                    ),
                    "target_quality_floor": self._coerce_profile_float(
                        payload.get("target_quality_floor", fallback["target_quality_floor"]),
                        float(fallback["target_quality_floor"]),
                        field_name="target_quality_floor",
                    ),
                    "direct_stop_quality": self._coerce_profile_float(
                        payload.get("direct_stop_quality", fallback["direct_stop_quality"]),
                        float(fallback["direct_stop_quality"]),
                        field_name="direct_stop_quality",
                    ),
                    "profile_source": "llm",
                    "reason": str(payload.get("reason", ""))[:240],
                }
                return profile
        except Exception as exc:
            logger.warning("Task profile classification failed; using heuristic fallback: %s", exc)
        return fallback

    def _heuristic_task_schema(self) -> Dict[str, Any]:
        text = self._instance_text().lower()
        profile = self.task_profile or self._heuristic_task_profile()
        primary = str(profile.get("primary_type", "fact_lookup"))
        explicit_range = any(tok in text for tok in ["guidance", "range", "low end", "high end", "upper end", "lower end"])
        comparison_words = any(tok in text for tok in ["beat", "above", "below", "difference", "delta", "vs", "versus", "exceed"])
        distinct_years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))

        if explicit_range and comparison_words:
            family = "range_comparison"
            memory_mode = "relational_bundle"
            answer_shape = "dual_boundary_comparison"
            roles = ["researcher", "schema_verifier", "verifier", "calculator"]
            required_fields = ["metric", "actual_value", "range_low", "range_high", "direction", "unit", "source"]
            fallback_family = "symbolic_or_numeric"
            object_types = ["relation", "value", "critique"]
        elif (
            (primary == "structured_extraction" and re.search(r"\b(19|20)\d{2}\b", text))
            or (len(distinct_years) >= 2 and not explicit_range)
        ):
            family = "slot_table"
            memory_mode = "multi_slot_coverage"
            answer_shape = "table_then_summary"
            roles = ["searcher", "researcher", "verifier", "calculator"]
            required_fields = ["dimensions", "value", "unit", "source"]
            fallback_family = "set_extraction"
            object_types = ["value", "source_map", "critique"]
        elif any(tok in text for tok in ["list", "identify", "which", "what are", "components", "adjustments", "factors"]):
            family = "set_extraction"
            memory_mode = "deduped_item_set"
            answer_shape = "bullet_list"
            roles = ["searcher", "verifier", "reflector"]
            required_fields = ["item", "source", "evidence"]
            fallback_family = "direct_qa"
            object_types = ["set_item", "support", "critique"]
        elif any(tok in text for tok in ["rank", "top", "best", "largest", "smallest", "highest", "lowest"]):
            family = "ranking_selection"
            memory_mode = "candidate_scoring"
            answer_shape = "ranked_list"
            roles = ["searcher", "verifier", "calculator"]
            required_fields = ["candidate", "score_or_criterion", "source"]
            fallback_family = "direct_qa"
            object_types = ["candidate", "criterion", "ranking"]
        elif primary == "multi_hop_synthesis":
            family = "multi_hop_explanation"
            memory_mode = "claim_evidence_chain"
            answer_shape = "claim_then_support"
            roles = ["searcher", "verifier", "reflector"]
            required_fields = ["claim", "supporting_evidence", "source"]
            fallback_family = "open_ended_synthesis"
            object_types = ["claim", "relation", "support"]
        elif primary == "quantitative_reasoning":
            family = "symbolic_or_numeric"
            memory_mode = "structured_values_plus_derivation"
            answer_shape = "derivation_then_result"
            roles = ["searcher", "calculator", "verifier"]
            required_fields = ["input_values", "formula_or_operation", "result", "unit", "source"]
            fallback_family = "direct_qa"
            object_types = ["input_value", "derivation", "result"]
        elif primary == "comparative_analysis":
            family = "direct_qa"
            memory_mode = "fact_with_comparison_note"
            answer_shape = "short_answer_with_comparison"
            roles = ["searcher", "verifier"]
            required_fields = ["answer", "comparison_target", "source"]
            fallback_family = "open_ended_synthesis"
            object_types = ["claim", "relation", "support"]
        elif primary == "fact_lookup":
            family = "direct_qa"
            memory_mode = "single_fact"
            answer_shape = "short_answer"
            roles = ["searcher", "verifier"]
            required_fields = ["answer", "source"]
            fallback_family = "open_ended_synthesis"
            object_types = ["claim", "support"]
        else:
            family = "open_ended_synthesis"
            memory_mode = "theme_and_evidence"
            answer_shape = "structured_paragraph"
            roles = ["searcher", "reflector", "verifier"]
            required_fields = ["main_points", "support", "source"]
            fallback_family = "direct_qa"
            object_types = ["claim", "support", "theme", "critique"]
        return {
            "schema_family": family,
            "memory_mode": memory_mode,
            "final_answer_shape": answer_shape,
            "priority_roles": roles,
            "required_fields": required_fields,
            "preferred_object_types": object_types,
            "fallback_schema_family": fallback_family,
            "schema_source": "heuristic",
            "reason": "generic schema-family heuristic",
        }

    def _infer_task_schema(self) -> Dict[str, Any]:
        fallback = self._heuristic_task_schema()
        llm = getattr(getattr(self.controller, "meta_llm", None), "llm", None) or self.judge_llm
        if llm is None:
            return fallback
        prompt = (
            "Choose the best generic task schema family for the task. "
            "Use only these schema_family labels: direct_qa, slot_table, range_comparison, set_extraction, "
            "symbolic_or_numeric, multi_hop_explanation, ranking_selection, open_ended_synthesis. "
            "Return strict JSON with keys: schema_family, memory_mode, final_answer_shape, priority_roles, "
            "required_fields, preferred_object_types, fallback_schema_family, reason. "
            "Keep it general so it could also apply to QA, math problems, open-ended questions, extraction, and comparison tasks.\n\n"
            f"Task:\n{self._instance_text()}\n\n"
            f"Task profile:\n{json.dumps(self.task_profile, ensure_ascii=False)}"
        )
        try:
            resp = llm.invoke([{"role": "user", "content": prompt}], num_retries=0)
            payload = self._extract_json_payload(getattr(resp, "content", "") or "")
            if payload:
                requested_family = self._coerce_profile_choice(
                    payload.get("schema_family", fallback["schema_family"]),
                    {
                        "direct_qa", "slot_table", "range_comparison", "set_extraction",
                        "symbolic_or_numeric", "multi_hop_explanation", "ranking_selection", "open_ended_synthesis",
                    },
                    fallback["schema_family"],
                )
                allowed_families = {
                    "direct_qa", "slot_table", "range_comparison", "set_extraction",
                    "symbolic_or_numeric", "multi_hop_explanation", "ranking_selection", "open_ended_synthesis",
                }
                family = requested_family
                text = self._instance_text().lower()
                explicit_range = any(tok in text for tok in ["guidance", "range", "low end", "high end", "upper end", "lower end"])
                distinct_years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
                if family == "range_comparison" and not explicit_range:
                    if len(distinct_years) >= 2:
                        family = "slot_table"
                    else:
                        family = fallback["schema_family"]
                if family != requested_family:
                    guarded = dict(fallback)
                    guarded["schema_family"] = family
                    guarded["schema_source"] = "llm_guarded"
                    guarded["reason"] = f"guarded_from_{requested_family}: {fallback.get('reason', '')}"[:240]
                    return guarded
                roles = payload.get("priority_roles", fallback["priority_roles"])
                if not isinstance(roles, list):
                    roles = fallback["priority_roles"]
                roles = [str(r).strip().lower() for r in roles if str(r).strip()][:5] or list(fallback["priority_roles"])
                fields = payload.get("required_fields", fallback["required_fields"])
                if not isinstance(fields, list):
                    fields = fallback["required_fields"]
                fields = [str(f).strip() for f in fields if str(f).strip()][:8] or list(fallback["required_fields"])
                object_types = payload.get("preferred_object_types", fallback.get("preferred_object_types", []))
                if not isinstance(object_types, list):
                    object_types = fallback.get("preferred_object_types", [])
                object_types = [str(x).strip().lower() for x in object_types if str(x).strip()][:6] or list(fallback.get("preferred_object_types", []))
                return {
                    "schema_family": family,
                    "memory_mode": str(payload.get("memory_mode", fallback["memory_mode"]))[:80] or fallback["memory_mode"],
                    "final_answer_shape": str(payload.get("final_answer_shape", fallback["final_answer_shape"]))[:80] or fallback["final_answer_shape"],
                    "priority_roles": roles,
                    "required_fields": fields,
                    "preferred_object_types": object_types,
                    "fallback_schema_family": self._coerce_profile_choice(
                        payload.get("fallback_schema_family", fallback["fallback_schema_family"]),
                        allowed_families,
                        fallback["fallback_schema_family"],
                    ),
                    "schema_source": "llm",
                    "reason": str(payload.get("reason", ""))[:240],
                }
        except Exception as exc:
            logger.warning("Task schema proposal failed; using heuristic schema: %s", exc)
        return fallback

    def _task_profile_block(self) -> str:
        profile = self.task_profile or {}
        bottleneck = self._infer_runtime_bottleneck_state()
        return (
            "\n[Task profile]\n"
            f"primary_type: {profile.get('primary_type', 'unknown')}\n"
            f"complexity: {profile.get('complexity', 'unknown')}\n"
            f"evidence_shape: {profile.get('evidence_shape', 'unknown')}\n"
            f"evolution_bias: {profile.get('evolution_bias', 'balanced')}\n"
            f"profile_reason: {profile.get('reason', '')}\n"
            f"runtime_bottleneck: {bottleneck.get('dominant_bottleneck', 'unknown')}\n"
            f"bottleneck_reason: {bottleneck.get('reason', '')}\n"
        )

    def _task_schema_block(self) -> str:
        schema = self.task_schema or {}
        return (
            "\n[Task schema]\n"
            f"schema_family: {schema.get('schema_family', 'unknown')}\n"
            f"memory_mode: {schema.get('memory_mode', 'unknown')}\n"
            f"final_answer_shape: {schema.get('final_answer_shape', 'unknown')}\n"
            f"priority_roles: {', '.join(schema.get('priority_roles', []) or [])}\n"
            f"required_fields: {', '.join(schema.get('required_fields', []) or [])}\n"
            f"preferred_object_types: {', '.join(schema.get('preferred_object_types', []) or [])}\n"
            f"fallback_schema_family: {schema.get('fallback_schema_family', 'unknown')}\n"
            f"schema_reason: {schema.get('reason', '')}\n"
        )

    def _task_description_for_meta(self) -> str:
        profile = self.task_profile or {}
        schema = self.task_schema or {}
        schema_notes = str(self.final_notes or "").strip()
        object_summary = self._object_ecology_summary()
        bottleneck = self._infer_runtime_bottleneck_state()
        return (
            f"{self.task_description}\n\n"
            "[Task profile]\n"
            f"- primary_type: {profile.get('primary_type', 'unknown')}\n"
            f"- complexity: {profile.get('complexity', 'unknown')}\n"
            f"- evidence_shape: {profile.get('evidence_shape', 'unknown')}\n"
            f"- evolution_bias: {profile.get('evolution_bias', 'balanced')}\n"
            "[Task schema]\n"
            f"- schema_family: {schema.get('schema_family', 'unknown')}\n"
            f"- memory_mode: {schema.get('memory_mode', 'unknown')}\n"
            f"- final_answer_shape: {schema.get('final_answer_shape', 'unknown')}\n"
            f"- priority_roles: {', '.join(schema.get('priority_roles', []) or [])}\n"
            f"- preferred_object_types: {', '.join(schema.get('preferred_object_types', []) or [])}\n"
            "[Object ecology]\n"
            f"- total_objects: {object_summary.get('total_objects', 0)}\n"
            f"- validated_objects: {object_summary.get('validated_objects', 0)}\n"
            f"- tentative_objects: {object_summary.get('tentative_objects', 0)}\n"
            f"- rejected_objects: {object_summary.get('rejected_objects', 0)}\n"
            f"- top_object_types: {object_summary.get('top_object_types', {})}\n"
            f"- weak_or_missing_object_types: {object_summary.get('weak_or_missing_object_types', [])}\n"
            "[Runtime bottleneck]\n"
            f"- dominant_bottleneck: {bottleneck.get('dominant_bottleneck', 'unknown')}\n"
            f"- retrieval_pressure: {bottleneck.get('retrieval_pressure', 0)}\n"
            f"- structuring_pressure: {bottleneck.get('structuring_pressure', 0)}\n"
            f"- validation_pressure: {bottleneck.get('validation_pressure', 0)}\n"
            f"- reasoning_pressure: {bottleneck.get('reasoning_pressure', 0)}\n"
            f"- action_bias: {bottleneck.get('action_bias', '')}\n"
            f"- bottleneck_reason: {bottleneck.get('reason', '')}\n"
            f"{('- runtime_schema_notes: ' + schema_notes + chr(10)) if schema_notes else ''}"
            "Use this profile to choose whether to stop early, stay conservative, or explore structure.\n"
        )

    def _normalize_object_status(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"validated", "confirmed", "pass", "accepted", "final"}:
            return "validated"
        if text in {"rejected", "fail", "invalid", "discarded"}:
            return "rejected"
        if text in {"tentative", "candidate", "draft", "proposed"}:
            return "tentative"
        return "candidate"

    def _normalize_candidate_object(
        self,
        raw: Dict[str, Any],
        *,
        agent_id: str,
        role: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        object_type = str(
            raw.get("object_type")
            or raw.get("type")
            or raw.get("kind")
            or ""
        ).strip().lower()
        if not object_type:
            return None
        summary = str(
            raw.get("summary")
            or raw.get("content")
            or raw.get("claim")
            or raw.get("value")
            or ""
        ).strip()
        fields = raw.get("fields")
        if not isinstance(fields, dict):
            fields = {}
        source = str(raw.get("source", "") or fields.get("source", "") or "").strip()
        confidence = raw.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except Exception:
            confidence = 0.5
        return {
            "object_type": object_type[:40],
            "status": self._normalize_object_status(raw.get("status")),
            "summary": summary[:240],
            "source": source[:120],
            "confidence": confidence,
            "producer_id": agent_id,
            "producer_role": role,
            "fields": self._to_jsonable(fields),
        }

    def _extract_candidate_objects(self, role: str, output: str, agent_id: str) -> list[Dict[str, Any]]:
        text = str(output or "").strip()
        if not text or self._is_low_signal_answer(text):
            return []

        extracted: list[Dict[str, Any]] = []
        payload = self._extract_json_payload(text)
        for key in ("candidate_objects", "info_objects", "objects"):
            raw_objects = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(raw_objects, list):
                for raw in raw_objects:
                    normalized = self._normalize_candidate_object(raw, agent_id=agent_id, role=role)
                    if normalized:
                        extracted.append(normalized)

        if isinstance(payload, dict):
            schema_facts = payload.get("schema_facts")
            if isinstance(schema_facts, dict):
                if isinstance(schema_facts.get("range_comparison"), dict):
                    fact = dict(schema_facts["range_comparison"])
                    extracted.append(
                        {
                            "object_type": "relation",
                            "status": "validated" if str(fact.get("source", "")).startswith("primary_filing") else "tentative",
                            "summary": str(fact.get("metric", "range comparison"))[:240],
                            "source": str(fact.get("source", ""))[:120],
                            "confidence": 0.9 if str(fact.get("source", "")).startswith("primary_filing") else 0.7,
                            "producer_id": agent_id,
                            "producer_role": role,
                            "fields": self._to_jsonable(fact),
                        }
                    )
            for key in ("slot_facts", "coverage_facts", "validated_schema_facts"):
                raw_facts = payload.get(key)
                if isinstance(raw_facts, dict):
                    for fact_key, fact_value in raw_facts.items():
                        if isinstance(fact_value, dict):
                            source = str(fact_value.get("source", ""))[:120]
                            extracted.append(
                                {
                                    "object_type": "value",
                                    "status": "validated" if source.startswith("primary_filing") else "tentative",
                                    "summary": str(fact_key)[:240],
                                    "source": source,
                                    "confidence": 0.85 if source.startswith("primary_filing") else 0.65,
                                    "producer_id": agent_id,
                                    "producer_role": role,
                                    "fields": self._to_jsonable(fact_value),
                                }
                            )
            if role == "schema_verifier":
                verdict = str(payload.get("schema_verdict", "") or "").strip().lower()
                if verdict:
                    extracted.append(
                        {
                            "object_type": "critique" if verdict != "pass" else "repair_hint",
                            "status": "validated" if verdict == "pass" else "candidate",
                            "summary": str(payload.get("corrective_hint", "") or payload.get("field_semantic_errors", ""))[:240],
                            "source": "",
                            "confidence": 0.8,
                            "producer_id": agent_id,
                            "producer_role": role,
                            "fields": self._to_jsonable(payload),
                        }
                    )

        deduped: list[Dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for obj in extracted:
            key = (
                str(obj.get("object_type", "")),
                str(obj.get("status", "")),
                str(obj.get("summary", "")),
                str(obj.get("source", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(obj)
        return deduped[:10]

    def _update_information_objects(self, role: str, output: str, agent_id: str) -> None:
        new_objects = self._extract_candidate_objects(role, output, agent_id)
        if not new_objects:
            return
        existing_keys = {
            (
                str(obj.get("object_type", "")),
                str(obj.get("status", "")),
                str(obj.get("summary", "")),
                str(obj.get("source", "")),
            )
            for obj in self.information_objects
        }
        for obj in new_objects:
            key = (
                str(obj.get("object_type", "")),
                str(obj.get("status", "")),
                str(obj.get("summary", "")),
                str(obj.get("source", "")),
            )
            if key in existing_keys:
                continue
            existing_keys.add(key)
            self.information_objects.append(obj)
            self.agent_object_memory[agent_id].append(obj)
        self.information_objects = self.information_objects[-120:]
        self.agent_object_memory[agent_id] = self.agent_object_memory[agent_id][-20:]

    def _object_ecology_summary(self) -> Dict[str, Any]:
        objects = list(self.information_objects)
        if not objects:
            preferred = list((self.task_schema or {}).get("preferred_object_types", []) or [])
            return {
                "total_objects": 0,
                "validated_objects": 0,
                "tentative_objects": 0,
                "rejected_objects": 0,
                "top_object_types": {},
                "weak_or_missing_object_types": preferred[:4],
            }
        type_counts: Dict[str, int] = defaultdict(int)
        status_counts: Dict[str, int] = defaultdict(int)
        validated_type_counts: Dict[str, int] = defaultdict(int)
        for obj in objects:
            object_type = str(obj.get("object_type", "") or "unknown")
            status = self._normalize_object_status(obj.get("status"))
            type_counts[object_type] += 1
            status_counts[status] += 1
            if status == "validated":
                validated_type_counts[object_type] += 1
        preferred = list((self.task_schema or {}).get("preferred_object_types", []) or [])
        weak_or_missing = [
            object_type for object_type in preferred
            if validated_type_counts.get(object_type, 0) == 0
        ]
        top_types = dict(sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))[:5])
        return {
            "total_objects": len(objects),
            "validated_objects": int(status_counts.get("validated", 0)),
            "tentative_objects": int(status_counts.get("tentative", 0) + status_counts.get("candidate", 0)),
            "rejected_objects": int(status_counts.get("rejected", 0)),
            "top_object_types": top_types,
            "weak_or_missing_object_types": weak_or_missing[:5],
        }

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

    def _object_signal_stats(self, objects: list[Dict[str, Any]]) -> Dict[str, Any]:
        factual_types = {
            "claim",
            "value",
            "relation",
            "derivation",
            "set_item",
            "criterion",
            "candidate",
            "result",
            "support",
            "source_map",
            "input_value",
        }
        critique_types = {"critique", "repair_hint"}
        factual = 0
        grounded = 0
        critique = 0
        validated = 0
        preferred_hits: set[str] = set()
        weak_types = set((self._object_ecology_summary() or {}).get("weak_or_missing_object_types", []) or [])
        for obj in objects:
            object_type = str(obj.get("object_type", "") or "").strip().lower()
            status = self._normalize_object_status(obj.get("status"))
            source = str(obj.get("source", "") or "").strip()
            fields = obj.get("fields") if isinstance(obj.get("fields"), dict) else {}
            if status == "validated":
                validated += 1
            if object_type in factual_types:
                factual += 1
                if source or str(fields.get("source", "") or "").strip():
                    grounded += 1
            elif object_type in critique_types:
                critique += 1
            if object_type in weak_types:
                preferred_hits.add(object_type)
        return {
            "factual_objects": factual,
            "grounded_factual_objects": grounded,
            "critique_objects": critique,
            "validated_objects": validated,
            "preferred_hits": sorted(preferred_hits),
        }

    def _infer_runtime_bottleneck_state(self) -> Dict[str, Any]:
        object_summary = self._object_ecology_summary()
        agents = [self.controller.population.get_agent(aid) for aid in self.controller.get_agent_ids()]
        agents = [agent for agent in agents if agent is not None]
        search_agents = [agent for agent in agents if self._is_search_capable_role(getattr(agent, "role", ""))]

        retrieval_pressure = 0
        validation_pressure = 0
        reasoning_pressure = 0
        search_failures = 0
        concrete_search_outputs = 0
        validation_markers = {
            "inconsistent",
            "conflict",
            "mismatch",
            "semantic",
            "wrong unit",
            "field_semantic",
            "rejected",
            "schema",
            "bundle",
        }

        for agent in agents:
            role = _canonicalize_role(getattr(agent, "role", ""))
            failure_modes = {str(item or "").strip().lower() for item in getattr(agent, "failure_modes", []) or []}
            memory = str(getattr(agent, "memory_summary", "") or "")
            latest_output = str(self.latest_outputs.get(getattr(agent, "agent_id", ""), "") or "")
            combined = f"{memory}\n{latest_output}".lower()

            if self._is_search_capable_role(role):
                if failure_modes & {"missing_key", "no_evidence", "not_finished", "no_reconciliation"}:
                    search_failures += 1
                if self._has_grounded_evidence(memory, [], role) or self._has_grounded_evidence(latest_output, [], role):
                    concrete_search_outputs += 1

            if any(marker in combined for marker in validation_markers):
                validation_pressure += 1

        retrieval_pressure += search_failures
        if int(object_summary.get("validated_objects", 0) or 0) == 0:
            retrieval_pressure += 1
        if concrete_search_outputs == 0 and search_agents:
            retrieval_pressure += 1

        weak_missing = len(object_summary.get("weak_or_missing_object_types", []) or [])
        tentative_objects = int(object_summary.get("tentative_objects", 0) or 0)
        validated_objects = int(object_summary.get("validated_objects", 0) or 0)
        structuring_pressure = weak_missing + (1 if tentative_objects > validated_objects + 1 else 0)

        if validated_objects >= 2 and max(self.current_answer_quality, self.best_answer_quality) < 0.5:
            reasoning_pressure += 2
        elif validated_objects >= 1 and max(self.current_answer_quality, self.best_answer_quality) < 0.35:
            reasoning_pressure += 1

        bottlenecks = {
            "retrieval": retrieval_pressure,
            "structuring": structuring_pressure,
            "validation": validation_pressure,
            "reasoning": reasoning_pressure,
        }
        dominant = max(bottlenecks.items(), key=lambda item: item[1])[0] if bottlenecks else "coordination"
        if bottlenecks.get(dominant, 0) <= 0:
            dominant = "coordination"

        bias_map = {
            "retrieval": "preserve or add evidence-acquisition capacity; reward new grounded evidence over critique-only output",
            "structuring": "favor agents that turn raw evidence into reusable objects or validated summaries",
            "validation": "favor checking/repair agents that disambiguate malformed or conflicting evidence",
            "reasoning": "favor agents that combine grounded objects into a final answer",
            "coordination": "favor minimal rewiring and avoid unnecessary replacement",
        }
        return {
            "dominant_bottleneck": dominant,
            "retrieval_pressure": retrieval_pressure,
            "structuring_pressure": structuring_pressure,
            "validation_pressure": validation_pressure,
            "reasoning_pressure": reasoning_pressure,
            "action_bias": bias_map.get(dominant, ""),
            "reason": (
                f"search_failures={search_failures}, concrete_search_outputs={concrete_search_outputs}, "
                f"validated_objects={validated_objects}, tentative_objects={tentative_objects}, "
                f"weak_missing_types={weak_missing}, validation_signals={validation_pressure}"
            ),
        }

    def _object_ecology_block(self) -> str:
        summary = self._object_ecology_summary()
        bottleneck = self._infer_runtime_bottleneck_state()
        preferred = ", ".join((self.task_schema or {}).get("preferred_object_types", []) or []) or "none"
        weak_types = ", ".join(summary.get("weak_or_missing_object_types", []) or []) or "none"
        top_types = json.dumps(summary.get("top_object_types", {}), ensure_ascii=False)
        return (
            "\n[Object ecology]\n"
            "There is NO required fixed pipeline. You may solve the task in your own way.\n"
            "However, if you discover a reusable information object, you MAY emit compact JSON like:\n"
            "{\"candidate_objects\":[{\"object_type\":\"claim|value|relation|derivation|set_item|criterion|critique|repair_hint\","
            "\"status\":\"candidate|tentative|validated|rejected\",\"summary\":\"...\",\"source\":\"primary_filing_1\","
            "\"confidence\":0.72,\"fields\":{...}}]}\n"
            "These objects are used as a lightweight shared ecology for credit assignment and evolution pressure, not as a mandatory workflow.\n"
            f"Preferred object types for this task: {preferred}\n"
            f"Current top object types: {top_types}\n"
            f"Weak or missing preferred object types: {weak_types}\n"
            f"Current dominant bottleneck: {bottleneck.get('dominant_bottleneck', 'coordination')}\n"
            f"Current action bias: {bottleneck.get('action_bias', '')}\n"
        )

    def _infer_coverage_schema(self) -> Dict[str, Any]:
        """Infer a lightweight, task-specific slot schema without hardcoding domains.

        The schema is intentionally generic: dimensions are open strings such as
        time_period/entity/metric/category.  Agents can fill these slots from any
        dataset, while the runtime uses the slots to avoid repeated retrieval.
        """
        question = self._instance_text()
        q = re.sub(r"\s+", " ", question).strip()
        if not q:
            return {"slot_keys": [], "required_slots": [], "schema_reason": "empty_task"}

        years: list[str] = []
        for start, end in re.findall(r"\b(20\d{2})\s*(?:-|–|—|to|through|until)\s*(20\d{2})\b", q, flags=re.IGNORECASE):
            lo, hi = int(start), int(end)
            if lo > hi:
                lo, hi = hi, lo
            if hi - lo <= 15:
                years.extend(str(y) for y in range(lo, hi + 1))
        years.extend(re.findall(r"\b(20\d{2})\b", q))

        quarters = []
        for qtr, year in re.findall(r"\b(Q[1-4])\s*(?:FY)?\s*(20\d{2})\b", q, flags=re.IGNORECASE):
            quarters.append(f"{qtr.upper()} {year}")
        for year, qtr in re.findall(r"\b(20\d{2})\s*(Q[1-4])\b", q, flags=re.IGNORECASE):
            quarters.append(f"{qtr.upper()} {year}")

        def _dedupe(values: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for value in values:
                if value not in seen:
                    seen.add(value)
                    out.append(value)
            return out

        years = _dedupe(years)
        quarters = _dedupe(quarters)

        metric = q
        metric = re.sub(r"^(what|how|which|when|where|why|calculate|compute|find|list|show)\b", "", metric, flags=re.IGNORECASE).strip(" ?")
        metric = re.sub(r"\bfrom\s+20\d{2}\s*(?:-|–|—|to|through|until)\s*20\d{2}\b", "", metric, flags=re.IGNORECASE)
        metric = re.sub(r"\bbetween\s+20\d{2}\s+and\s+20\d{2}\b", "", metric, flags=re.IGNORECASE)
        metric = re.sub(r"\b(in|for|during)\s+(?:Q[1-4]\s*)?20\d{2}\b", "", metric, flags=re.IGNORECASE)
        metric = re.sub(r"\([^)]{1,40}\)", "", metric)
        metric = re.sub(r"\s+", " ", metric).strip(" ?:,;")
        if len(metric) > 90:
            metric = metric[:90].rsplit(" ", 1)[0].strip()
        metric = metric or "target_value"

        slot_periods = quarters if quarters else years
        slots: list[Dict[str, Any]] = []
        if slot_periods:
            for period in slot_periods:
                dimensions = {"time_period": period, "metric": metric}
                slots.append(
                    {
                        "slot_id": self._slot_id(dimensions),
                        "dimensions": dimensions,
                        "status": "missing",
                        "value": None,
                        "unit": "",
                        "source": "",
                        "evidence_text": "",
                        "confidence": 0.0,
                        "updated_by": "",
                        "locked": False,
                        "locked_by": "",
                        "support_signals": {},
                    }
                )

        return {
            "slot_keys": ["time_period", "metric"] if slots else [],
            "required_slots": slots,
            "schema_reason": "time_series_or_periodic_question" if slots else "no_explicit_coverage_slots_detected",
        }

    def _coverage_status(self) -> Dict[str, Any]:
        slots = list(self.coverage_memory.values())
        found = [s for s in slots if s.get("status") == "found" and s.get("value") not in (None, "")]
        tentative = [s for s in slots if s.get("status") == "tentative" and s.get("value") not in (None, "")]
        missing = [s for s in slots if s.get("status") not in ("found", "tentative") or s.get("value") in (None, "")]
        return {
            "total": len(slots),
            "found": len(found),
            "tentative": len(tentative),
            "missing": len(missing),
            "coverage": (len(found) / len(slots)) if slots else 0.0,
            "found_slots": found,
            "tentative_slots": tentative,
            "missing_slots": missing,
        }

    def _is_complex_evidence_task(self) -> bool:
        profile = self.task_profile or {}
        schema = self.task_schema or {}
        primary = str(profile.get("primary_type", "") or "").lower()
        complexity = str(profile.get("complexity", "") or "").lower()
        evidence_shape = str(profile.get("evidence_shape", "") or "").lower()
        schema_family = str(schema.get("schema_family", "") or "").lower()
        total_slots = len(self.coverage_memory or {})
        if complexity == "high":
            return True
        if evidence_shape in {"tabular_or_multi_slot", "mixed"}:
            return True
        if primary in {"structured_extraction", "multi_hop_synthesis", "quantitative_reasoning"}:
            return True
        if schema_family in {"slot_table", "range_comparison", "set_extraction", "multi_hop_explanation"}:
            return True
        return total_slots >= 3

    def _should_treat_bundle_as_routing_only(self) -> bool:
        return self._is_complex_evidence_task()

    def _format_slot_brief(self, slot: Dict[str, Any]) -> str:
        dims = slot.get("dimensions", {}) or {}
        dim_text = ", ".join(f"{k}={v}" for k, v in dims.items())
        value = slot.get("value")
        if value not in (None, ""):
            return f"{slot.get('slot_id')}: {dim_text}; value={value}; source={slot.get('source') or 'unknown'}"
        return f"{slot.get('slot_id')}: {dim_text}"

    def _format_coverage_block(self, role: str) -> str:
        if not self.coverage_memory:
            return ""
        status = self._coverage_status()
        found_lines = [self._format_slot_brief(s) for s in status["found_slots"][:12]]
        tentative_lines = [self._format_slot_brief(s) for s in status["tentative_slots"][:12]]
        missing_lines = [self._format_slot_brief(s) for s in status["missing_slots"][:12]]
        role = _canonicalize_role(role)

        role_directive = ""
        if role == "searcher":
            role_directive = (
                "Searcher: search only missing slots unless a verifier explicitly rejected a found slot. "
                "Rewrite queries around the missing slot dimensions; do not repeat already found slots.\n"
                "When you find slot values, include a compact JSON object like "
                "{\"coverage_facts\":[{\"dimensions\":{\"time_period\":\"2024\"},\"value\":\"11.70\",\"unit\":\"...\",\"source\":\"primary_filing_2\",\"evidence_text\":\"...\",\"confidence\":0.9}]}.\n"
                "Only emit coverage_facts for values you directly extracted from a source; do not include page numbers, result ranks, or unrelated dates.\n"
            )
        elif role == "verifier":
            role_directive = (
                "Verifier: validate found slots against primary_filing/source keys and identify only truly missing or rejected slots. "
                "Do not turn one missing slot into a global failure narrative. "
                "Do NOT fill missing slots with estimates, approximations, interpolations, or calculator-style derivations.\n"
            )
        elif role == "calculator":
            role_directive = (
                "Calculator: do not call retrieve_information blindly when required slots are still missing. "
                "If slots are missing, return CALCULATOR_WAITING_FOR_STRUCTURED_FACTS and list missing slots. "
                "If slots are found, compute trends only from the structured slot table below. "
                "Do NOT create new slot values or overwrite missing slots.\n"
            )
        else:
            role_directive = (
                "Use found slots as high-priority memory. Treat missing slots as targeted gaps rather than restarting the task.\n"
            )

        return (
            "\n[Coverage targets / structured memory]\n"
            f"Schema reason: {self.coverage_schema.get('schema_reason')}\n"
            f"Confirmed coverage: {status['found']}/{status['total']} ({status['coverage']:.0%}); tentative candidates: {status['tentative']}\n"
            + ("Confirmed slots:\n- " + "\n- ".join(found_lines) + "\n" if found_lines else "Confirmed slots: none yet\n")
            + ("Tentative candidates (verify before using in final answer):\n- " + "\n- ".join(tentative_lines) + "\n" if tentative_lines else "")
            + ("Missing slots:\n- " + "\n- ".join(missing_lines) + "\n" if missing_lines else "Missing slots: none\n")
            + role_directive
        )

    def _strip_tool_plan_text_for_coverage(self, text: str) -> str:
        """Remove tool-call plans/query JSON before heuristic slot extraction."""
        cleaned_lines = []
        skip_jsonish = False
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            low = line.lower()
            if not line:
                continue
            if any(marker in low for marker in [
                '"tool_name"', '"tool_calls"', '"parameters"', '"query"', '"filing_type"',
                "tool called with arguments", "[[tool call", "function_calls", "edgar_search",
                "web_search", "parse_html_page", "prepare_primary_filing",
            ]):
                continue
            if re.match(r"^[\{\}\[\],]+$", line):
                continue
            if re.search(r'"\w+"\s*:', line) and any(tok in low for tok in ["query", "tool", "parameter", "plan"]):
                skip_jsonish = True
                continue
            if skip_jsonish and re.match(r"^[\]\}],?$", line):
                skip_jsonish = False
                continue
            if skip_jsonish:
                continue
            cleaned_lines.append(raw_line)
        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"```(?:json)?[\s\S]*?```", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\{[\s\S]{0,1200}?(?:tool_name|tool_calls|parameters|query)[\s\S]{0,1200}?\}", " ", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _parse_numeric_value_near_slot(self, text: str, target: str, metric: str = "") -> Optional[Dict[str, Any]]:
        if not target:
            return None
        metric_tokens = [
            tok.lower()
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", metric or "")
            if tok.lower() not in {
                "the", "and", "for", "from", "with", "has", "have", "changed",
                "change", "average", "revenue", "user", "users", "paying",
                "how", "what", "which", "nasdaq",
            }
        ]
        metric_tokens = metric_tokens[:8]
        pattern = re.compile(re.escape(str(target)), flags=re.IGNORECASE)
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 180): match.end() + 220]
            after = text[match.end(): match.end() + 140]
            window_low = window.lower()
            has_metric_context = (
                any(tok in window_low for tok in metric_tokens)
                or any(tok in window_low for tok in ["arpu", "arppu", "per paying", "per membership", "per user"])
                or re.search(r"[|:]\s*(?:\$|US\$)?\s*-?\d", after) is not None
            )
            if not has_metric_context:
                continue
            candidates = list(re.finditer(
                r"(?P<prefix>\$|US\$)?\s*(?P<value>-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*(?P<suffix>%|percent|million|billion|shares|per user|per member|per membership|ARPU|ARPPU|x)?",
                after,
                flags=re.IGNORECASE,
            ))
            for cand in candidates[:4]:
                value = cand.group("value").replace(",", "")
                if value == str(target):
                    continue
                # Bare integers are often result ranks/page numbers.  Accept
                # them only when units are explicit or the metric asks for
                # count-like values.
                suffix = cand.group("suffix") or ""
                prefix = cand.group("prefix") or ""
                is_decimal = "." in value
                count_like_metric = any(tok in window_low for tok in ["shares", "employees", "subscribers", "customers"])
                if not (is_decimal or suffix or prefix or count_like_metric):
                    continue
                return {
                    "value": value,
                    "unit": suffix or ("USD" if prefix else ""),
                    "evidence_text": re.sub(r"\s+", " ", window).strip()[:300],
                }
        return None

    def _extract_coverage_facts(self, output: str, role: str) -> list[Dict[str, Any]]:
        if not self.coverage_memory or not output:
            return []
        # Hard role boundary: only search-capacity roles may populate slot-table
        # memory from intermediate outputs. Verifiers/calculators/reflectors can
        # critique or request missing evidence, but they must not backfill values.
        if role not in {"searcher", "synthesizer"}:
            return []
        facts: list[Dict[str, Any]] = []
        payload = self._extract_json_payload(output)
        lowered_output = str(output).lower()
        if any(token in lowered_output for token in [
            "approximation",
            "approximate",
            "estimated",
            "estimate",
            "inferred",
            "interpolated",
            "back-solved",
            "back solved",
        ]):
            return []
        if isinstance(payload.get("coverage_facts"), list):
            for item in payload.get("coverage_facts") or []:
                if not isinstance(item, dict):
                    continue
                dims = item.get("dimensions") if isinstance(item.get("dimensions"), dict) else {}
                if not dims:
                    continue
                value = item.get("value")
                value_text = str(value if value is not None else "").strip()
                if not value_text or not re.search(r"-?\d", value_text):
                    continue
                facts.append(
                    {
                        "dimensions": dims,
                        "value": value_text,
                        "unit": str(item.get("unit", "") or ""),
                        "source": str(item.get("source", "") or ""),
                        "evidence_text": str(item.get("evidence_text", "") or "")[:300],
                        "confidence": float(item.get("confidence", 0.7) or 0.7),
                    }
                )

        # Text heuristics are deliberately conservative.  They provide weak
        # slot fill candidates only when a slot label is near metric/table
        # context and the value shape is plausible.  This avoids poisoning
        # memory with search ranks, page numbers, or years.
        text = self._strip_tool_plan_text_for_coverage(output)
        low = text.lower()
        for slot in self.coverage_memory.values():
            dims = slot.get("dimensions", {}) or {}
            period = str(dims.get("time_period", "") or "")
            metric = str(dims.get("metric", "") or "")
            if not period or period.lower() not in low:
                continue
            parsed = self._parse_numeric_value_near_slot(text, period, metric)
            if not parsed:
                continue
            source_match = re.search(r"primary_filing(?:_[1-5])?", text, flags=re.IGNORECASE)
            facts.append(
                {
                    "dimensions": dims,
                    "value": parsed["value"],
                    "unit": parsed.get("unit", ""),
                    "source": source_match.group(0) if source_match else "",
                    "evidence_text": parsed.get("evidence_text", ""),
                    "confidence": 0.45,
                    "heuristic": True,
                }
            )
        return facts

    def _update_coverage_memory(self, role: str, output: str, agent_id: str) -> None:
        if not self.coverage_memory:
            return
        for fact in self._extract_coverage_facts(output, role):
            self._upsert_coverage_fact(
                fact=fact,
                agent_id=agent_id,
                allow_locked_override=False,
                force_lock=False,
            )

    def _upsert_coverage_fact(
        self,
        fact: Dict[str, Any],
        agent_id: str,
        *,
        allow_locked_override: bool = False,
        force_lock: bool = False,
    ) -> bool:
        if not self.coverage_memory:
            return False
        dims = dict(fact.get("dimensions", {}) or {})
        matched_id = ""
        for slot_id, slot in self.coverage_memory.items():
            slot_dims = slot.get("dimensions", {}) or {}
            if all(str(slot_dims.get(k, "")).lower() == str(v).lower() for k, v in dims.items() if v):
                matched_id = slot_id
                break
        if not matched_id:
            return False
        existing = self.coverage_memory[matched_id]
        if existing.get("locked") and existing.get("value") not in (None, "") and not allow_locked_override:
            return False
        old_conf = float(existing.get("confidence", 0.0) or 0.0)
        new_conf = max(0.0, min(1.0, float(fact.get("confidence", 0.6) or 0.6)))
        confirmed_threshold = float(os.getenv("COVERAGE_CONFIRMED_THRESHOLD", "0.7"))
        new_status = "found" if new_conf >= confirmed_threshold and not fact.get("heuristic") else "tentative"
        if force_lock:
            new_status = "found"
            new_conf = max(new_conf, float(os.getenv("LOCKED_COVERAGE_MIN_CONFIDENCE", "0.8")))
        if (
            existing.get("status") == "found"
            and old_conf > new_conf
            and existing.get("value") not in (None, "")
        ):
            return False
        if existing.get("status") == "found" and new_status != "found":
            return False
        existing.update(
            {
                "status": new_status,
                "value": fact.get("value"),
                "unit": fact.get("unit", "") or existing.get("unit", ""),
                "source": fact.get("source", "") or existing.get("source", ""),
                "evidence_text": fact.get("evidence_text", "") or existing.get("evidence_text", ""),
                "confidence": new_conf,
                "updated_by": agent_id,
                "locked": bool(existing.get("locked")) or bool(force_lock),
                "locked_by": str(existing.get("locked_by", "") or (agent_id if force_lock else ""))[:80],
                "support_signals": self._to_jsonable(fact.get("support_signals", {}) or existing.get("support_signals", {})),
            }
        )
        return True

    def _synthesize_from_coverage_memory(self) -> str:
        if not self.coverage_memory:
            return ""
        status = self._coverage_status()
        if status["found"] == 0:
            return ""
        complex_task = self._is_complex_evidence_task()
        found_slots = status["found_slots"]
        protected_slots = [
            slot for slot in found_slots
            if bool((slot.get("support_signals", {}) or {}).get("protected"))
            or bool(slot.get("locked"))
        ]
        if complex_task:
            if len(protected_slots) < max(2, int(len(found_slots) * 0.6)):
                return ""
            if status["coverage"] < 0.8 and status["missing"] > 0:
                return ""
        else:
            if status["coverage"] < 0.6:
                return ""
        found_slots = sorted(
            protected_slots if complex_task else found_slots,
            key=lambda s: str((s.get("dimensions", {}) or {}).get("time_period", s.get("slot_id", ""))),
        )
        lines = []
        for slot in found_slots:
            dims = slot.get("dimensions", {}) or {}
            label = dims.get("time_period") or slot.get("slot_id")
            value = slot.get("value")
            unit = f" {slot.get('unit')}" if slot.get("unit") else ""
            source = f" (source: {slot.get('source')})" if slot.get("source") else ""
            lines.append(f"{label}: {value}{unit}{source}")
        missing = [self._format_slot_brief(s) for s in status["missing_slots"]]
        answer = "\n".join(lines)
        if missing:
            answer += "\nMissing/needs verification: " + "; ".join(missing[:6])
        return answer

    def _conservative_min_score(self) -> float:
        return float(os.getenv("CONSERVATIVE_MODE_MIN_SCORE", "0.75"))

    def _conservative_delta(self) -> float:
        return float(os.getenv("CONSERVATIVE_MODE_DELTA", "0.05"))

    def _in_conservative_mode(self) -> bool:
        if not str(self.best_final_answer or "").strip():
            return False
        if float(self.best_answer_quality) < self._conservative_min_score():
            return False
        return float(self.current_answer_quality) >= float(self.best_answer_quality) - self._conservative_delta()

    def _defer_slow_update_for_conservative_mode(self) -> int:
        extra = int(os.getenv("CONSERVATIVE_MODE_SLOW_SKIP", "2"))
        extra = max(1, extra)
        current_next = int(getattr(self.controller, "next_slow_update_at", 0))
        self.controller.next_slow_update_at = max(
            current_next + extra,
            int(getattr(self.controller, "fast_time_step", 0)) + extra,
        )
        return extra

    def _high_score_path_min_score(self) -> float:
        return float(os.getenv("HIGH_SCORE_PATH_PROTECT_MIN_SCORE", "0.75"))

    def _high_score_path_min_agent_score(self) -> float:
        return float(os.getenv("HIGH_SCORE_PATH_PROTECT_MIN_AGENT_SCORE", "0.3"))

    def _main_path_edge_types(self) -> set[str]:
        raw = os.getenv(
            "HIGH_SCORE_PATH_PROTECTED_EDGE_TYPES",
            "evidence_flow,verification_flow,computation_flow,bidirectional",
        )
        return {item.strip() for item in raw.split(",") if item.strip()}

    def _capture_high_score_path(self, graph_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Capture the current answer backbone when a new high score is reached."""
        if float(self.best_answer_quality) < self._high_score_path_min_score():
            return
        graph = graph_snapshot or self._snapshot_graph()
        protected_agents: set[str] = set()
        min_agent_score = self._high_score_path_min_agent_score()

        if self.fast_round_logs:
            for agent in self.fast_round_logs[-1].get("agents", []) or []:
                aid = str(agent.get("agent_id", "") or "")
                if not aid or not self._eligible_for_final_answer_path(aid):
                    continue
                score = float(agent.get("contribution_score", 0.0) or 0.0)
                output = str(agent.get("output_text", "") or "")
                if score >= min_agent_score and output and not self._is_low_signal_answer(output):
                    protected_agents.add(aid)

        for aid, output in self.latest_outputs.items():
            if aid and output and not self._is_low_signal_answer(output) and self._eligible_for_final_answer_path(aid):
                state = self.controller.population.get_agent(aid)
                recent = state.recent_scores[-1] if state and state.recent_scores else 0.0
                if float(recent or 0.0) >= min_agent_score:
                    protected_agents.add(aid)

        # If all agents scored weakly but the final answer was strong, protect
        # the non-reflection main-path agents in the current graph as a fallback.
        if not protected_agents:
            for node in graph.get("nodes", []) or []:
                aid = str(node.get("agent_id", "") or "")
                role = _canonicalize_role(node.get("role", ""))
                if role in {"searcher", "verifier", "calculator"} and self._eligible_for_final_answer_path(aid):
                    protected_agents.add(aid)

        main_types = self._main_path_edge_types()
        protected_edges = []
        for edge in graph.get("edges", []) or []:
            etype = str(edge.get("type", "") or "")
            src = str(edge.get("source", "") or "")
            tgt = str(edge.get("target", "") or "")
            if not src or not tgt or etype not in main_types:
                continue
            if src in protected_agents or tgt in protected_agents:
                protected_edges.append(dict(edge))

        self.best_graph_snapshot = graph
        self.best_protected_agent_ids = protected_agents
        self.best_protected_edges = protected_edges
        logger.info(
            "Captured high-score protected path: best=%.3f agents=%s edges=%s",
            self.best_answer_quality,
            sorted(protected_agents),
            len(protected_edges),
        )

    def _protected_graph_paths(self) -> Dict[str, Any]:
        if float(self.best_answer_quality) < self._high_score_path_min_score():
            return {}
        return {
            "best_answer_quality": float(self.best_answer_quality),
            "protected_agents": sorted(self.best_protected_agent_ids),
            "protected_edges": list(self.best_protected_edges),
            "protected_edge_types": sorted(self._main_path_edge_types()),
        }

    def _record_best_answer_if_improved(
        self,
        graph_snapshot: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> bool:
        candidate = self._sanitize_final_answer_text(self.final_answer)
        if candidate and not self._answer_satisfies_task_schema(candidate):
            logger.info(
                "Skipped best-answer update%s: candidate does not satisfy task schema",
                f" ({source})" if source else "",
            )
            return False
        if self.current_answer_quality > self.best_answer_quality and candidate.strip():
            self.best_answer_quality = self.current_answer_quality
            self.best_final_answer = candidate
            self._promote_best_answer_to_stable_memory(candidate, source=source)
            self._capture_high_score_path(graph_snapshot=graph_snapshot)
            logger.info(
                "New best answer%s: quality=%.3f, len=%d",
                f" ({source})" if source else "",
                self.best_answer_quality,
                len(self.best_final_answer),
            )
            return True
        return False

    def _slot_support_metadata(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        dims = dict(fact.get("dimensions", {}) or {})
        period = str(dims.get("time_period", "") or "").strip().lower()
        metric = str(dims.get("metric", "") or "").strip().lower()
        value = str(fact.get("value", "") or "").strip().lower()
        source = str(fact.get("source", "") or "").strip()
        source_low = source.lower()

        support_roles = {"verifier", "schema_verifier", "auditor", "critic"}
        supporter_ids: list[str] = []
        schema_supporter_ids: list[str] = []
        for aid in self.controller.get_agent_ids():
            state = self.controller.population.get_agent(aid)
            role = _canonicalize_role(state.role if state else "")
            if role not in support_roles:
                continue
            text = "\n".join(
                [
                    str(getattr(state, "memory_summary", "") or ""),
                    str(self.latest_outputs.get(aid, "") or ""),
                    str(getattr(state, "workflow_correction", "") or ""),
                    str(getattr(state, "next_round_workflow", "") or ""),
                ]
            ).lower()
            if not text.strip():
                continue
            value_hit = bool(value and value in text)
            period_hit = bool(period and period in text)
            source_hit = bool(source_low and source_low in text)
            metric_tokens = [tok for tok in re.findall(r"[a-z][a-z0-9_-]{2,}", metric) if tok not in {"the", "and", "for", "from", "with"}][:4]
            metric_hit = any(tok in text for tok in metric_tokens) if metric_tokens else False
            if (value_hit and (period_hit or source_hit)) or (period_hit and source_hit) or (period_hit and metric_hit and role == "schema_verifier"):
                supporter_ids.append(aid)
                if role == "schema_verifier":
                    schema_supporter_ids.append(aid)

        preferred_keys = {str(v).strip().lower() for v in (self.source_feedback_memory.get("preferred_source_keys", []) or []) if str(v).strip()}
        preferred_source = bool(source_low and source_low in preferred_keys)
        grounded_source = bool(source)
        primary_like_source = bool(source_low.startswith("primary_filing"))
        supported_by_verifier = bool(supporter_ids)
        supported_by_schema_verifier = bool(schema_supporter_ids)
        protected = grounded_source and (
            supported_by_verifier
            or supported_by_schema_verifier
            or preferred_source
        )
        return {
            "has_source": grounded_source,
            "primary_like_source": primary_like_source,
            "preferred_source": preferred_source,
            "supported_by_verifier": supported_by_verifier,
            "supported_by_schema_verifier": supported_by_schema_verifier,
            "protected": protected,
            "supporter_ids": supporter_ids[:6],
            "schema_supporter_ids": schema_supporter_ids[:6],
        }

    def _promote_best_answer_to_stable_memory(self, answer_text: str, source: str = "") -> None:
        if not self.coverage_memory:
            return
        quality = float(self.current_answer_quality or 0.0)
        if quality <= 0.0:
            return
        facts = self._extract_coverage_facts(answer_text, role="synthesizer")
        if not facts:
            return
        for fact in facts:
            fact = dict(fact)
            fact["heuristic"] = False
            support = self._slot_support_metadata(fact)
            base_conf = max(float(fact.get("confidence", 0.0) or 0.0), min(0.95, max(0.65, quality)))
            if support.get("has_source") and (support.get("supported_by_verifier") or support.get("supported_by_schema_verifier")):
                base_conf = max(base_conf, 0.85)
            elif support.get("has_source") and support.get("preferred_source"):
                base_conf = max(base_conf, 0.8)
            fact["confidence"] = base_conf
            fact["support_signals"] = support
            self._upsert_coverage_fact(
                fact=fact,
                agent_id=f"best_answer:{source or 'runtime'}",
                allow_locked_override=False,
                force_lock=bool(support.get("protected")),
            )

    def _build_judge_llm(self):
        judge_model = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini").strip()
        judge_api_key = (
            os.getenv("JUDGE_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        judge_api_base = (
            os.getenv("JUDGE_API_BASE", "").strip()
            or os.getenv("OPENAI_API_BASE", "").strip()
        )

        if not judge_api_key:
            logger.info(
                "Judge LLM key not configured; falling back to template agent LLM for contribution scoring."
            )
            return self.template_agent.llm

        try:
            judge_config = LLMConfig(
                model=judge_model,
                api_key=judge_api_key,
                api_base=judge_api_base or None,
                temperature=0.0,
            )
            logger.info(
                "Judge LLM configured: model=%s api_base=%s",
                judge_model,
                judge_api_base or "default",
            )
            return judge_config.get_llm()
        except Exception as exc:
            logger.warning(
                "Failed to initialize dedicated judge LLM (%s); falling back to template agent LLM.",
                exc,
            )
            return self.template_agent.llm

    def _snapshot_graph(self) -> Dict[str, Any]:
        edges = []
        for src, tgt, edge_type in self.controller.graph.get_all_edges():
            edges.append({
                "source": src,
                "target": tgt,
                "type": edge_type.value,
            })
        edges = sorted(edges, key=lambda x: (x["source"], x["target"], x["type"]))
        nodes = []
        for aid in sorted(self.controller.get_agent_ids()):
            state = self.controller.population.get_agent(aid)
            nodes.append(
                {
                    "agent_id": aid,
                    "role": _canonicalize_role(state.role if state else "searcher"),
                }
            )
        return {"nodes": nodes, "edges": edges}

    def _diff_graph(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        before_nodes = {n["agent_id"]: n.get("role", "") for n in before.get("nodes", [])}
        after_nodes = {n["agent_id"]: n.get("role", "") for n in after.get("nodes", [])}
        added_nodes = [
            {"agent_id": aid, "role": role}
            for aid, role in after_nodes.items()
            if aid not in before_nodes
        ]
        removed_nodes = [
            {"agent_id": aid, "role": role}
            for aid, role in before_nodes.items()
            if aid not in after_nodes
        ]
        role_changes = [
            {"agent_id": aid, "from": before_nodes[aid], "to": after_nodes[aid]}
            for aid in before_nodes.keys() & after_nodes.keys()
            if before_nodes[aid] != after_nodes[aid]
        ]

        def _edge_key(edge: Dict[str, Any]) -> tuple[str, str, str]:
            return (edge["source"], edge["target"], edge["type"])

        before_edges = {_edge_key(e): e for e in before.get("edges", [])}
        after_edges = {_edge_key(e): e for e in after.get("edges", [])}
        added_edges = [after_edges[k] for k in sorted(after_edges.keys() - before_edges.keys())]
        removed_edges = [before_edges[k] for k in sorted(before_edges.keys() - after_edges.keys())]

        return {
            "added_nodes": added_nodes,
            "removed_nodes": removed_nodes,
            "role_changes": role_changes,
            "added_edges": added_edges,
            "removed_edges": removed_edges,
        }

    def _to_jsonable(self, value: Any) -> Any:
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

    def _sync_runtime_agents(self) -> None:
        """Keep runtime worker set aligned with current population ids."""
        target_ids = set(self.controller.get_agent_ids())
        current_ids = set(self.runtime_agents.keys())
        role_rank: dict[str, int] = {}
        role_seen: dict[str, int] = defaultdict(int)
        for aid in sorted(target_ids):
            state = self.controller.population.get_agent(aid)
            role = _canonicalize_role(state.role if state is not None else "searcher")
            role_seen[role] += 1
            role_rank[aid] = role_seen[role]

        for removed_id in current_ids - target_ids:
            self.runtime_agents.pop(removed_id, None)
            self.latest_outputs.pop(removed_id, None)
            self.latest_artifacts.pop(removed_id, None)
            self.latest_queries.pop(removed_id, None)
            self.latest_env_status.pop(removed_id, None)
            self.latest_tool_names.pop(removed_id, None)

        for new_id in target_ids - current_ids:
            state = self.controller.population.get_agent(new_id)
            role = _canonicalize_role(state.role if state is not None else "searcher")
            policy = state.policy if state is not None else ""
            rank = role_rank.get(new_id, 1)
            prompt_map = self._build_prompt_map_for_agent(role, state, rank)
            self.runtime_agents[new_id] = BaseWorkerAgent.init_from_agent(
                agent=self.template_agent,
                agent_id=new_id,
                task_instance=self.task_instance,
                role=role,
                policy=policy,
                allowed_tools=self.role_permissions.get(role, self.template_agent.tools),
                tool_budget=None,
                prompts=prompt_map,
                min_iterations_per_agent=self.min_iterations_per_agent,
                max_iterations_per_agent=self.max_iterations_per_agent,
            )

    def _edge_allows_artifact_handoff(self, edge_type: Any) -> bool:
        normalized = EdgeType.from_value(edge_type)
        return normalized != EdgeType.REFLECTION_FEEDBACK

    def _export_worker_artifacts(
        self,
        worker: BaseWorkerAgent,
        query_hint: str = "",
        include_full_docs: bool = False,
    ) -> Dict[str, Any]:
        env = getattr(worker, "env", None)
        if env is None or not hasattr(env, "export_storage_artifacts"):
            return {}
        try:
            payload = env.export_storage_artifacts(
                query_hint=query_hint,
                include_full_docs=include_full_docs,
            )
        except Exception as exc:
            logger.warning("Failed to export worker artifacts for %s: %s", worker.agent_id, exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    def _hydrate_worker_with_incoming_artifacts(
        self,
        agent_id: str,
        worker: BaseWorkerAgent,
        query_hint: str = "",
    ) -> None:
        env = getattr(worker, "env", None)
        if env is None or not hasattr(env, "import_storage_artifacts"):
            return
        payloads: list[Dict[str, Any]] = []
        if hasattr(env, "clear_storage_artifacts"):
            pass
        for src in self.controller.graph.get_neighbors(agent_id, direction="in"):
            edge_type = self.controller.graph.get_edge_type(src, agent_id) or EdgeType.DIRECTED
            if not self._edge_allows_artifact_handoff(edge_type):
                continue
            src_worker = self.runtime_agents.get(src)
            if src_worker is not None:
                payload = self._export_worker_artifacts(src_worker, query_hint=query_hint)
            else:
                payload = self.latest_artifacts.get(src) or {}
            if not payload:
                continue
            payloads.append(payload)

        if not payloads:
            return

        if hasattr(env, "clear_storage_artifacts"):
            try:
                env.clear_storage_artifacts()
            except Exception as exc:
                logger.warning("Failed to clear prior artifacts for %s: %s", agent_id, exc)

        for payload in payloads:
            try:
                env.import_storage_artifacts(payload)
            except Exception as exc:
                logger.warning(
                    "Failed to import artifacts into %s: %s",
                    agent_id,
                    exc,
                )

    def _build_prompt_map_for_agent(
        self,
        role: str,
        state: Optional[Any],
        role_rank: int,
    ) -> Dict[str, Prompt]:
        prompt_map = dict(self.role_prompts)
        # Meta can specify a prompt variant (e.g. "searcher_alt2") via AgentState.
        if state is not None:
            variant = str(getattr(state, "prompt_variant", "") or "").strip()
            if variant and variant in self.role_prompts:
                prompt_map[role] = self.role_prompts[variant]
                return prompt_map
            if variant and variant.startswith(f"{role}_alt") and f"{role}_alt" in self.role_prompts:
                prompt_map[role] = self.role_prompts[f"{role}_alt"]
                return prompt_map
        # Automatic variant selection for duplicate roles: cycle through available variants.
        variants = sorted(
            key for key in self.role_prompts.keys()
            if key.startswith(f"{role}_alt")
        )
        if role_rank > 1 and variants:
            pick = variants[(role_rank - 2) % len(variants)]
            prompt_map[role] = self.role_prompts[pick]
        return prompt_map

    def _instance_text(self) -> str:
        prompt_info = self.task_instance.get_prompt_info()
        if isinstance(prompt_info, dict):
            if "question" in prompt_info:
                return str(prompt_info["question"])
            if prompt_info:
                return "\n".join(f"{k}: {v}" for k, v in prompt_info.items())
        return str(prompt_info)

    def _sanitize_agent_output_for_context(self, text: str, max_chars: int = 1200) -> str:
        """Strip noisy tool-call markup and degenerate loops before passing between agents."""
        raw = str(text or "")
        if not raw:
            return ""
        # Normalize common escaped/full-width tokens first.
        cleaned = raw.replace("\\<", "<").replace("\\>", ">").replace("｜", "|")
        # Remove textual function-call blocks that often pollute peer context.
        cleaned = re.sub(
            r"<(?:\|DSML\|)?function_calls>[\s\S]*?</(?:\|DSML\|)?function_calls>",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"<function_calls>[\s\S]*?</function_calls>",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"</?\|[^>\n]{1,80}\|>",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"</?\s*[A-Za-z0-9_:-]{1,40}\s*/?>",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\{[\s]*(tool|function|call_id|arguments)[\s\S]*?\}",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Collapse extreme repeated "Let" corruption.
        cleaned = re.sub(r"(?:\bLet\b[\s]*){20,}", " ", cleaned, flags=re.IGNORECASE)
        # Also collapse "LetLetLet..." without whitespace (common degenerate loop).
        cleaned = re.sub(r"(?i)(?:let){20,}", " ", cleaned)
        # Detect degenerate outputs that repeatedly spill internal agent ids/tokens
        # (e.g., "agent_agent_agent_...") and collapse them aggressively.
        if cleaned.count("agent_") >= 40:
            # Keep only a tiny excerpt to avoid flooding downstream prompts/logs.
            cleaned = cleaned[:200] + " ...[degenerate agent-token repetition truncated]"
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + " ...[truncated]"
        return cleaned

    def _is_low_signal_answer(self, text: str) -> bool:
        t = str(text or "").strip().lower()
        if not t:
            return True
        # Treat degenerate repeated tokens as low-signal.
        if t.count("agent_") >= 30:
            return True
        # Treat repeated "let" loops (with or without spaces) as low-signal.
        if "letletletletletletletletletlet" in t:
            return True
        low_signal_phrases = [
            "i need to know what keys are available",
            "i need to understand what documents are available",
            "let me start by searching",
            "failed to generate a valid answer",
            "data storage system",
            "<function_calls>",
            "<invoke ",
            "|dsml|",
            # Synthesis fallback marker
            "insufficient_data:",
            # Procedural / in-progress statements (English)
            "i am currently",
            "i am now",
            "i will now",
            "i will retrieve",
            "i will search",
            "i will parse",
            "i will extract",
            "let me now",
            "let me retrieve",
            "let me search",
            "let me parse",
            "let me extract",
            "to answer this question, i need to",
            "before answering, i need to",
            "first, i need to",
            "i should search",
            "i need to look up",
            "currently extracting",
            "currently searching",
            "currently processing",
            "currently retrieving",
            "currently analyzing",
            "extracting now",
            "retrieving now",
            "processing now",
            "searching now",
            "i need to search",
            "i need to retrieve",
            "i need to extract",
            # Procedural / in-progress statements (Chinese)
            "正在抽取",
            "正在检索",
            "正在处理",
            "正在搜索",
            "正在分析",
            "正在提取",
            "正在获取",
            "正在计算",
            "无法继续",
            "需要提供",
            "正在尝试",
            "需要数据",
            "处理中",
            "检索中",
            "我先去检索",
            "我先去搜索",
            "我先提取",
            # Explicit "no evidence" / "not found" reports should not be summarized as answers.
            "no evidence",
            "no reconciliation",
            "not contain the reconciliation",
            "does not contain the reconciliation",
            "reconciliation not found",
            "no relevant reconciliation",
            "key was not found",
            "key not found",
            "was not found in storage",
            "not found in storage",
            "没有证据",
            "未找到对账",
            "未找到调节项",
            "不包含对账",
            "不包含调整",
        ]
        return any(p in t for p in low_signal_phrases)

    def _is_no_evidence_output(self, text: str) -> bool:
        t = str(text or "").strip().lower()
        if not t:
            return True
        no_evidence_phrases = [
            "no evidence",
            "no reconciliation",
            "not contain the reconciliation",
            "does not contain the reconciliation",
            "reconciliation not found",
            "no relevant reconciliation",
            "key was not found",
            "key not found",
            "was not found in storage",
            "not found in storage",
            "没有证据",
            "未找到对账",
            "未找到调节项",
            "不包含对账",
            "不包含调整",
        ]
        return any(p in t for p in no_evidence_phrases)

    def _extract_failure_modes(self, output: str, env_status: DatasetEnvStatus) -> list[str]:
        failures: list[str] = []
        if not env_status.success:
            failures.append("not_finished")
        text = str(output or "").lower()
        if "key was not found" in text or "key not found" in text or "not found in storage" in text:
            failures.append("missing_key")
        if (
            "no reconciliation" in text
            or "does not contain the reconciliation" in text
            or "not contain the reconciliation" in text
            or "reconciliation not found" in text
            or "no relevant reconciliation" in text
            or "未找到对账" in text
            or "不包含对账" in text
        ):
            failures.append("no_reconciliation")
        if "no evidence" in text or "没有证据" in text:
            failures.append("no_evidence")
        return failures

    def _is_procedural_output(self, text: str) -> bool:
        """Detect plan/procedural outputs that should not be treated as final answers."""
        t = str(text or "").strip().lower()
        if not t:
            return True
        if "insufficient_data:" in t:
            return True
        # JSON plan patterns
        if "\"plan\"" in t and ("\"task_id\"" in t or "\"tool_calls\"" in t):
            return True
        # Common plan headings
        if t.startswith("plan:") or t.startswith("步骤:") or t.startswith("步骤："):
            return True
        # Obvious procedural phrasing (English / Chinese)
        procedural_phrases = [
            "here's the plan",
            "i will now",
            "i will retrieve",
            "i will search",
            "i will parse",
            "i will extract",
            "let me retrieve",
            "let me search",
            "let me parse",
            "let me extract",
            "to answer this question, i need to",
            "before answering, i need to",
            "first, i need to",
            "next step",
            "next, i will",
            "plan:",
            "步骤",
            "计划",
            "下一步",
            "我将",
            "我会",
            "先",
            "然后",
        ]
        return any(p in t for p in procedural_phrases)

    def _build_agent_query(
        self,
        agent_id: str,
        current_round_outputs: Optional[Dict[str, str]] = None,
    ) -> str:
        """Construct query with directed/debate context according to graph edges.

        ``current_round_outputs`` contains the outputs of agents that have already
        run **this round** (earlier in the sequential execution order).  Injecting
        this enables lightweight cross-agent coordination: an agent can see what
        peers have already retrieved or computed and avoid duplicating that work.
        """
        state = self.controller.population.get_agent(agent_id)
        role = _canonicalize_role(state.role if state is not None else "searcher")
        policy = state.policy if state is not None else ""
        workflow_guidance = state.next_round_workflow if state is not None else ""
        workflow_correction = state.workflow_correction if state is not None else ""
        meta_score = state.meta_score if state is not None else 0.0
        score_reason = state.score_reason if state is not None else ""
        conservative_mode = self._in_conservative_mode()

        incoming_msgs = []
        for src in self.controller.graph.get_neighbors(agent_id, direction="in"):
            src_state = self.controller.population.get_agent(src)
            src_output = self._sanitize_agent_output_for_context(self.latest_outputs.get(src, ""))
            if not src_output:
                continue
            # Also carry the peer's latest meta-guidance into agent-agent comms.
            # This is especially important for bidirectional (debate) edges to avoid "debating"
            # on outdated/unguided behavior.
            if src_state is not None:
                src_role = _canonicalize_role(src_state.role)
                src_meta_score = float(getattr(src_state, "meta_score", 0.0) or 0.0)
                src_score_reason = str(getattr(src_state, "score_reason", "") or "")
                src_wf_corr = self._sanitize_agent_output_for_context(
                    getattr(src_state, "workflow_correction", "") or "",
                    max_chars=260,
                )
                src_wf_next = self._sanitize_agent_output_for_context(
                    getattr(src_state, "next_round_workflow", "") or "",
                    max_chars=260,
                )
                src_guidance_block = (
                    f"\n[Peer meta guidance]\n"
                    f"- peer_role: {src_role}\n"
                    f"- peer_meta_score: {src_meta_score:.3f}\n"
                    f"- peer_score_reason: {src_score_reason or 'N/A'}\n"
                    f"- peer_workflow_correction: {src_wf_corr or 'N/A'}\n"
                    f"- peer_next_round_workflow: {src_wf_next or 'N/A'}\n"
                )
            else:
                src_guidance_block = ""
            edge_type = self.controller.graph.get_edge_type(src, agent_id) or EdgeType.DIRECTED
            if edge_type == EdgeType.BIDIRECTIONAL:
                incoming_msgs.append(
                    f"[Debate input from {src}] Critique and integrate this peer view (follow their meta guidance when evaluating drift):\n"
                    f"{src_output}{src_guidance_block}"
                )
            elif edge_type == EdgeType.EVIDENCE_FLOW:
                incoming_msgs.append(
                    f"[Evidence handoff from {src}] Treat this as upstream evidence or source context. Reuse, refine, or extend it rather than restarting from scratch:\n"
                    f"{src_output}{src_guidance_block}"
                )
            elif edge_type == EdgeType.VERIFICATION_FLOW:
                incoming_msgs.append(
                    f"[Verification handoff from {src}] Treat this as validation or correction on prior evidence. Prioritize consistency fixes and factual alignment:\n"
                    f"{src_output}{src_guidance_block}"
                )
            elif edge_type == EdgeType.COMPUTATION_FLOW:
                incoming_msgs.append(
                    f"[Computation handoff from {src}] Treat this as derived values, aggregations, or structured calculations to incorporate carefully:\n"
                    f"{src_output}{src_guidance_block}"
                )
            elif edge_type == EdgeType.REFLECTION_FEEDBACK:
                incoming_msgs.append(
                    f"[Reflection feedback from {src}] Treat this as corrective process feedback, not primary evidence. Use it to avoid drift or duplication:\n"
                    f"{src_output}{src_guidance_block}"
                )
            else:
                incoming_msgs.append(
                    f"[Directed handoff from {src}] Treat this as upstream context:\n{src_output}{src_guidance_block}"
                )

        own_prev = self._sanitize_agent_output_for_context(self.latest_outputs.get(agent_id, ""), max_chars=800)
        own_prev_block = (
            f"\n[Your previous output]\n{own_prev}\n" if own_prev else ""
        )
        incoming_block = "\n\n".join(incoming_msgs) if incoming_msgs else "[No incoming messages this round]"
        object_ecology_block = self._object_ecology_block()
        outgoing_targets = list(self.controller.graph.get_neighbors(agent_id, direction="out"))
        if outgoing_targets:
            outgoing_roles = []
            for tgt in outgoing_targets:
                tgt_state = self.controller.population.get_agent(tgt)
                tgt_role = _canonicalize_role(tgt_state.role if tgt_state else "searcher")
                out_edge_type = self.controller.graph.get_edge_type(agent_id, tgt) or EdgeType.DIRECTED
                outgoing_roles.append(f"{tgt}({tgt_role}, {out_edge_type.value})")
            handoff_block = (
                "\n[Handoff expectation]\n"
                "Provide concise, evidence-backed output that downstream peers can reuse. "
                f"Your output will be consumed by: {', '.join(outgoing_roles)}.\n"
            )
        else:
            handoff_block = ""

        # ── Intra-round coordination block ────────────────────────────────────
        # Show what other agents have already done THIS round so this agent can
        # take a complementary approach rather than duplicating their work.
        coord_block = ""
        if current_round_outputs:
            coord_lines = []
            for peer_id, peer_out in current_round_outputs.items():
                if peer_id == agent_id:
                    continue
                peer_state = self.controller.population.get_agent(peer_id)
                peer_role = _canonicalize_role(peer_state.role if peer_state else "searcher")
                peer_tools = ", ".join(self.latest_tool_names.get(peer_id, [])) or "none"
                snippet = self._sanitize_agent_output_for_context(peer_out, max_chars=300)
                coord_lines.append(
                    f"  - {peer_id} [{peer_role}] tools_used=[{peer_tools}]\n"
                    f"    output_snippet: {snippet}"
                )
            if coord_lines:
                same_role_peers = [
                    pid for pid in current_round_outputs
                    if pid != agent_id
                    and _canonicalize_role(
                        getattr(self.controller.population.get_agent(pid), "role", "")
                    ) == role
                ]
                directive = (
                    "IMPORTANT: peers with the SAME role as you have already run this round "
                    f"({', '.join(same_role_peers)}). You MUST take a different approach — "
                    "use different sources, different search queries, or verify/extend their findings "
                    "rather than repeating the same tool calls.\n\n"
                    if same_role_peers else ""
                )
                coord_block = (
                    f"\n[Peer work completed this round — DO NOT duplicate]\n"
                    f"{directive}"
                    + "\n".join(coord_lines)
                    + "\n"
                )

        key_convention_block = (
            "\n[Storage key convention]\n"
            "If you are a SEARCHER: first call prepare_primary_filing(search_query=..., top_n_results=10, max_valid_docs=5). "
            "This deterministically searches, parses, filters, and stores up to 5 relevant documents as "
            "primary_filing_1 ... primary_filing_5, and stores a combined multi-document bundle in primary_filing.\n"
            "Only fall back to manual edgar_search/web_search/parse_html_page if prepare_primary_filing fails.\n"
            "If you are NOT a searcher: inspect {{primary_filing}} first, then explicitly drill into the most relevant "
            "{{primary_filing_1}} ... {{primary_filing_5}} document(s) for full-detail retrieval.\n"
            "If you are NOT a searcher, DO NOT call prepare_primary_filing, web_search, edgar_search, or parse_html_page.\n"
            "Do not invent unrelated key names, but you MAY use primary_filing and primary_filing_i keys that were handed off upstream.\n"
        )
        if self._should_treat_bundle_as_routing_only():
            key_convention_block += (
                "For this task, treat {{primary_filing}} mainly as a routing/index artifact. "
                "Use it to choose the right document, but prefer grounding concrete multi-field or multi-slot facts in "
                "{{primary_filing_1}} ... {{primary_filing_5}} rather than relying only on the bundle summary.\n"
            )
        else:
            key_convention_block += (
                "For this task, if {{primary_filing}} already contains a clear, single-hop supported answer, "
                "you may answer directly from the bundle without unnecessary drill-down.\n"
            )

        searcher_block = ""
        schema_family = str((self.task_schema or {}).get("schema_family", ""))
        schema_block = ""
        if schema_family == "range_comparison":
            schema_block = (
                "\n[Schema-family guidance: range_comparison]\n"
                "This task is best handled as one ACTUAL value compared against a LOW/HIGH range. "
                "Do not collapse the range into a midpoint unless explicitly asked.\n"
                "If you find the needed values, emit compact JSON such as "
                "{\"schema_facts\":{\"range_comparison\":{\"metric\":\"...\",\"actual_value\":\"11.6\",\"actual_unit\":\"%\","
                "\"range_low\":\"10.8\",\"range_high\":\"10.9\",\"range_unit\":\"%\",\"direction\":\"above\","
                "\"source\":\"primary_filing_1\",\"evidence_text\":\"...\"}}}.\n"
                "The final answer should include BOTH delta vs low end and delta vs high end whenever both boundaries are available.\n"
                "Required bundle fields for this schema: actual_value, range_low, range_high, source. "
                "If any required field is missing, DO NOT submit a final answer claim yet. "
                "Instead, output schema_facts plus missing_schema_fields so downstream peers can fill the gap.\n"
            )
            if role == "verifier":
                schema_block += (
                    "Verifier structured-comparison responsibilities: verify that ACTUAL is a true observed value rather than a beat-by/delta phrase; "
                    "verify that reference bounds come from the correct comparison context; "
                    "and verify that derived comparisons are directionally consistent with the extracted values. "
                    "If the structure is semantically wrong, output compact JSON including "
                    "{\"missing_schema_fields\":[...],\"field_semantic_errors\":[...],\"corrective_hint\":\"...\"}.\n"
                )
            if role == "schema_verifier":
                schema_block += (
                    "Schema verifier responsibilities: validate that each bundle field has the correct semantic meaning. "
                    "Specifically distinguish observed values from deltas or derived values, and distinguish reference bounds from observed values. "
                    "Output PASS only if the bundle is complete and semantically consistent. "
                    "If invalid, output a compact JSON object like "
                    "{\"schema_verdict\":\"FAIL\",\"missing_schema_fields\":[...],\"field_semantic_errors\":[...],"
                    "\"corrective_hint\":\"...\",\"validated_schema_facts\":{...}}.\n"
                )
        elif schema_family == "slot_table":
            schema_block = (
                "\n[Schema-family guidance: slot_table]\n"
                "Treat this as a structured table-fill problem. "
                "Prefer complete per-slot extraction over a vague narrative summary.\n"
            )
        elif schema_family == "symbolic_or_numeric":
            schema_block = (
                "\n[Schema-family guidance: symbolic_or_numeric]\n"
                "Store explicit inputs, the operation/formula, and the computed result. "
                "If inputs are missing, identify the exact missing variable rather than giving a generic failure narrative.\n"
            )
        elif schema_family == "set_extraction":
            schema_block = (
                "\n[Schema-family guidance: set_extraction]\n"
                "Produce a deduplicated set of supported items with source-backed evidence. "
                "Do not turn one missing item into a global failure statement.\n"
            )

        if role == "searcher":
            memory_text = state.memory_summary if state is not None else ""
            has_adjustments = self._memory_has_adjustments(memory_text)
            missing_items = self._missing_adjustments_from_memory(memory_text) if has_adjustments else []
            memory_complete = self._memory_covers_all_adjustments(memory_text) if has_adjustments else False
            verifier_hints = self._extract_verifier_source_hints()
            generic_missing_targets = self._derive_missing_search_targets(role)
            query_rewrite_hint = ""
            if missing_items:
                query_terms = " OR ".join(missing_items)
                query_rewrite_hint = (
                    "Suggested rewritten query (use this or improve it): "
                    f"\"{query_terms}\" + \"official source evidence details\".\n"
                )
            generic_query_intent = self._build_search_query_intent(generic_missing_targets, verifier_hints)
            search_constraints_payload = {
                "required_constraints": verifier_hints.get("required_constraints", {}) or {},
                "avoid_constraints": verifier_hints.get("avoid_constraints", {}) or {},
                "prefer_source_patterns": verifier_hints.get("prefer_source_patterns", []) or [],
                "avoid_source_patterns": verifier_hints.get("avoid_source_patterns", []) or [],
                "missing_targets": generic_missing_targets[:8],
            }
            verifier_block = ""
            if any(verifier_hints.values()):
                verifier_block = (
                    "\n[Verifier source hints]\n"
                    + (f"Preferred source keys: {', '.join(verifier_hints['preferred_source_keys'])}\n" if verifier_hints["preferred_source_keys"] else "")
                    + (f"Rejected source keys: {', '.join(verifier_hints['rejected_source_keys'])}\n" if verifier_hints["rejected_source_keys"] else "")
                    + ("Rejected source memory:\n" + "\n".join(
                        f"- {item.get('source_key') or 'unknown'} :: {item.get('reason') or item.get('details') or 'rejected'}"
                        for item in (verifier_hints.get("rejected_sources", []) or [])[:5]
                        if isinstance(item, dict)
                    ) + "\n" if verifier_hints.get("rejected_sources") else "")
                    + (f"Rejection reasons: {' | '.join(verifier_hints['rejection_reasons'])}\n" if verifier_hints["rejection_reasons"] else "")
                    + (f"Next search hints: {' | '.join(verifier_hints['next_search_hints'])}\n" if verifier_hints["next_search_hints"] else "")
                    + (self._format_search_constraints_block() + "\n" if self._format_search_constraints_block() else "")
                )
            searcher_block = (
                "\n[Searcher guidance]\n"
                "Do not hardcode form_types or ticker in the query. "
                "Start with prepare_primary_filing so the program, not the model, handles top-10 search, valid-5 filtering, "
                "document summaries, and creation of the multi-document primary_filing bundle.\n"
                "Treat rejected sources and search constraints as memory. Do NOT keep revisiting source families that have already been rejected for scope/entity/time mismatch.\n"
                "Treat locked/confirmed slots as stable memory: do not overwrite them with weaker later guesses; only search for truly missing targets or explicit verifier-flagged corrections.\n"
                "If prepare_primary_filing fails, ask the meta-LLM to rewrite the query into a broader or alternative phrasing, then retry.\n"
                + (
                    "Because this task appears evidence-complex, do not stop at the bundle summary once you have candidate documents. "
                    "Drill into specific primary_filing_i documents to extract grounded facts before claiming a multi-field answer.\n"
                    if self._should_treat_bundle_as_routing_only() else
                    "If the bundle already contains a direct, high-confidence single-hop answer, you may answer from it without forced extra drill-down.\n"
                )
                +
                "If the prepared bundle still does NOT contain the required evidence, "
                "report it explicitly as NO_REQUIRED_EVIDENCE_FOUND and request a new search; "
                "do NOT summarize such documents as final evidence.\n"
                + (
                    "Missing targets for the next search pass:\n"
                    + "\n".join(f"- {item}" for item in generic_missing_targets[:6])
                    + "\n"
                    if generic_missing_targets else ""
                )
                + (
                    "[Search constraints JSON]\n"
                    + json.dumps(search_constraints_payload, ensure_ascii=False)
                    + "\n"
                    if any(search_constraints_payload.values()) else ""
                )
                + (f"{generic_query_intent}\n" if generic_query_intent else "")
                + (
                    "Memory already contains adjustment items. "
                    "Do NOT re-fetch the same items; instead search for missing adjustments only.\n"
                    + (f"Missing items (prioritize these): {', '.join(missing_items)}\n" if missing_items else "")
                    + (query_rewrite_hint if query_rewrite_hint else "")
                    if has_adjustments else ""
                )
                + (
                    "Memory already covers ALL required adjustments. "
                    "DO NOT call web_search/edgar_search/parse_html_page. "
                    "Instead, verify consistency and summarize the adjustments from memory.\n"
                    if memory_complete else ""
                )
                + verifier_block
            )
            if conservative_mode:
                searcher_block += (
                    "\n[Conservative mode]\n"
                    "The system already has a historically high-scoring answer. "
                    "Prioritize reusing memory, {{primary_filing}}, and verifier-preferred source keys. "
                    "Do NOT re-run broad search unless there is a specific missing fact or a verifier rejection that identifies a concrete evidence gap.\n"
                )
        elif role == "verifier":
            if self.dataset_id == "finance-benchmark":
                searcher_block = (
                    "\n[Verifier guidance]\n"
                    "First read {{primary_filing}} to inspect the bundle summaries, source manifest, and high-relevance excerpts. "
                    "Then, if you need more detail, explicitly read the relevant underlying document key(s) such as "
                    "{{primary_filing_1}} ... {{primary_filing_5}} by name in a follow-up retrieve_information call.\n"
                    "Do not stop at saying the bundle is insufficient; use it to choose which detailed document to drill into.\n"
                    "You are a verifier, not a searcher or calculator. Do NOT search for new sources, do NOT call prepare_primary_filing/web_search/edgar_search/parse_html_page, "
                    "do NOT estimate missing values, and do NOT compute substitute figures from partial evidence.\n"
                    "If a value is not directly supported by {{primary_filing}} or {{primary_filing_i}}, mark it missing or rejected instead of filling the gap.\n"
                    "For multi-year or multi-period answers, verify each year/period independently. If a value is only supported for a different year, flag it as a year-value misalignment and mark that claim unsupported or contradicted.\n"
                )
            else:
                searcher_block = (
                    "\n[Verifier guidance]\n"
                    "Check whether proposed answers, actions, or claims are actually supported by the available upstream context, handed-off documents, passages, or tool outputs.\n"
                    "When upstream source material or document identifiers are available, inspect those first and verify candidate claims against them before passing anything downstream.\n"
                    "Use only the tools available to your role. Do not invent unavailable tools, and do not add new evidence claims unless they are directly supported by the context you can inspect.\n"
                    "If a proposed action or answer is malformed, contradictory, or unsupported, state the smallest correction needed.\n"
                    "Prefer compact supported_claims / unsupported_claims / contradicted_claims style feedback over long narrative.\n"
                )
            if conservative_mode:
                searcher_block += (
                    "Conservative mode is active: prioritize validating the current best evidence chain rather than requesting broad re-search.\n"
                )
        elif role == "calculator":
            if self.dataset_id == "finance-benchmark":
                searcher_block = (
                    "\n[Calculator guidance]\n"
                    "You are a calculator, not a searcher or verifier. Use retrieve_information only on {{primary_filing}} / {{primary_filing_i}} facts already gathered upstream.\n"
                    "Do NOT search for new sources, do NOT call prepare_primary_filing/web_search/edgar_search/parse_html_page, and do NOT invent missing inputs.\n"
                    "If required inputs are missing, return CALCULATOR_WAITING_FOR_STRUCTURED_FACTS and list the exact missing fields.\n"
                    "For multi-year or multi-period numeric tasks, first extract a single source-grounded table or series covering all supported years/periods from the chosen primary_filing_i document, then validate each year against that table.\n"
                    "Do not interrogate the same document with separate single-year guesses when one consolidated extraction is possible.\n"
                    "Do not copy or reuse a nearby year's value to fill a missing year. If a year is not directly supported from source, leave that year missing.\n"
                )
            elif self.dataset_id == "workbench":
                searcher_block = (
                    "\n[Action compiler guidance]\n"
                    "Convert the best available task understanding into the most direct executable tool invocation or action sequence.\n"
                    "Do not output retrieval placeholders, evidence-bundle keys, or waiting messages as the final deliverable.\n"
                    "If enough parameters are present, commit to a concrete invocation.\n"
                )
            elif self.dataset_id == "plancraft":
                searcher_block = (
                    "\n[Action compiler guidance]\n"
                    "Convert recipe knowledge and inventory facts into the shortest valid crafting action sequence, or exactly IMPOSSIBLE.\n"
                    "Prefer executable crafting steps over recipe explanation.\n"
                )
            else:
                searcher_block = (
                    "\n[Calculator guidance]\n"
                    "Use the upstream context to derive the most concrete result you can. "
                    "If the task is action-like, prefer an executable action or invocation over a narrative plan.\n"
                )
        elif role == "reflector":
            searcher_block = (
                "\n[Reflector guidance]\n"
                "You are a reflector. Produce post-mortem analysis only.\n"
                "Do NOT search, do NOT verify facts, do NOT calculate substitute values, and do NOT introduce new evidence claims.\n"
                "Your job is to diagnose workflow failures, role confusion, and missing evidence handoffs.\n"
            )
        elif conservative_mode:
            searcher_block = (
                "\n[Conservative mode]\n"
                "A historically high-scoring answer already exists. "
                "Favor reusing verified memory and existing source keys. "
                "Avoid introducing new searches or new graph-wide hypotheses unless the current evidence chain is contradicted or clearly incomplete.\n"
            )

        coverage_block = self._format_coverage_block(role)
        skill_block = ""
        if self.dataset_skill_text:
            skill_block = f"\n[Task skill]\n{self.dataset_skill_text}\n"

        memory_block = ""
        if state is not None and state.memory_summary:
            memory_text = str(state.memory_summary or "")
            if self._has_concrete_adjustment_facts(memory_text):
                memory_parts = [part.strip() for part in memory_text.split(" || ") if part.strip()]
                positive_parts = [part for part in memory_parts if self._has_concrete_adjustment_facts(part)]
                if positive_parts:
                    memory_text = " || ".join(positive_parts)
            memory_block = (
                "\n[Your memory summary]\n"
                f"{memory_text}\n"
                "If your memory already contains concrete adjustments, DO NOT re-search; "
                "focus on verification or filling missing items only.\n"
            )
            if conservative_mode:
                memory_block += (
                    "This round is in conservative mode: preserve and refine the strongest existing evidence instead of exploring broadly.\n"
                )

        return (
            f"[Task]\n{self._instance_text()}\n\n"
            f"{self._task_profile_block()}\n"
            f"{self._task_schema_block()}\n"
            f"[Agent role]\n{role}\n\n"
            f"[Role policy]\n{policy}\n\n"
            f"[Meta score]\n{meta_score:.3f}\nReason: {score_reason or 'N/A'}\n\n"
            f"[Meta workflow correction]\n{workflow_correction or 'No correction yet.'}\n\n"
            f"[Meta next-round workflow]\n{workflow_guidance or 'No next-round workflow yet. Continue current policy.'}\n"
            f"{own_prev_block}"
            f"{key_convention_block}"
            f"{schema_block}"
            f"{skill_block}"
            f"{object_ecology_block}"
            f"{searcher_block}"
            f"{coverage_block}"
            f"{memory_block}"
            f"{coord_block}\n"
            f"{handoff_block}"
            f"[Incoming graph messages]\n{incoming_block}\n\n"
            f"Please solve/improve the answer for this round. Follow the meta next-round workflow if provided, and use the workflow correction to evolve your behavior.\n"
            f"When workflow guidance exists, execute it as a concrete 2-4 step plan with tool names and expected evidence snippets."
        )

    def _normalize_env_status(self, value: Any) -> DatasetEnvStatus:
        if isinstance(value, DatasetEnvStatus):
            return value
        if isinstance(value, dict):
            return DatasetEnvStatus(
                success=bool(value.get("success", False)),
                num_steps=int(value.get("num_steps", 0)),
            )
        return DatasetEnvStatus(success=False, num_steps=0)

    def _score_text(self, text: str, env_status: Optional[DatasetEnvStatus] = None) -> float:
        if not text:
            return 0.0
        length_score = min(1.0, len(text) / 800.0)
        signal_score = 0.15 if env_status and env_status.success else 0.0
        return min(1.0, 0.85 * length_score + signal_score)

    def _extract_json_payload(self, text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            return {}
        block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
        if block:
            raw = block.group(1).strip()
        try:
            loaded = json.loads(raw)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            brace_match = re.search(r"\{[\s\S]*\}", raw)
            if not brace_match:
                return {}
            try:
                loaded = json.loads(brace_match.group(0))
                return loaded if isinstance(loaded, dict) else {}
            except Exception:
                return {}

    def _score_agent_contribution_llm(
        self,
        agent_id: str,
        role: str,
        query: str,
        output: str,
        tool_names: Optional[list[str]] = None,
        env_status: Optional[DatasetEnvStatus] = None,
    ) -> tuple[float, str]:
        """LLM judge for a single agent's potential contribution to solving the task."""
        # Always sanitize and truncate before sending to the judge LLM.
        output_for_judge = self._sanitize_agent_output_for_context(output, max_chars=800)
        if not str(output_for_judge or "").strip():
            return 0.0, "empty output"
        if self._is_low_signal_answer(output_for_judge):
            return 0.0, "low_signal output"

        tool_names = list(tool_names or [])
        evidence_ok = self._has_grounded_evidence(output_for_judge, tool_names, role)
        evidence_note = (
            "High scores (>0.5) are allowed ONLY when the output clearly cites or reflects grounded evidence "
            "from parsed documents, retrieved sources, structured source-backed objects, or explicit evidence snippets.\n"
            if role != "planner" else
            "Planner outputs may score for workflow quality, but they are not document evidence.\n"
        )

        prompt = (
            "You are an evaluator for multi-agent collaboration. "
            "Score how much this agent output can contribute to solving the task.\n"
            "Return strict JSON only: {\"score\": <0~1 float>, \"reason\": \"...\"}.\n"
            "Scoring rubric:\n"
            "- 0.0: irrelevant/wrong/no useful signal\n"
            "- 0.3: weak but partially relevant\n"
            "- 0.5: moderately useful evidence or decomposition\n"
            "- 0.7: strong useful contribution with concrete progress\n"
            "- 1.0: critical and directly decisive contribution\n\n"
            + evidence_note + "\n"
            f"Task:\n{self._instance_text()}\n\n"
            f"Round Query Given To Agent:\n{query}\n\n"
            f"Agent ID: {agent_id}\n"
            f"Agent Role: {role}\n\n"
            f"Tools Used: {tool_names}\n"
            f"Evidence Gate Precheck: {'PASS' if evidence_ok else 'FAIL'}\n"
            f"Agent Output:\n{output_for_judge}\n"
        )

        try:
            judge_resp = self.judge_llm.invoke(
                [{"role": "user", "content": prompt}],
                num_retries=0,
            )
            payload = self._extract_json_payload(getattr(judge_resp, "content", ""))
            score = float(payload.get("score", 0.0)) if payload else 0.0
            score = max(0.0, min(1.0, score))
            reason = str(payload.get("reason", "")) if payload else ""
            score, reason = self._apply_contribution_score_gates(
                role=role,
                output=output_for_judge,
                tool_names=tool_names,
                env_status=env_status,
                raw_score=score,
                raw_reason=reason,
            )
            return score, reason
        except Exception as exc:
            logger.warning(
                "LLM contribution scoring failed for %s in round %s: %s",
                agent_id,
                self.round_idx,
                exc,
            )
            fallback = self._score_text(output)
            return fallback, "fallback:text_heuristic"

    def _has_grounded_evidence(self, output: str, tool_names: list[str], role: str) -> bool:
        if role == "planner":
            return False
        t = str(output or "").lower()
        if self._is_no_evidence_output(t):
            return False
        has_doc_trace = (
            "primary_filing" in t
            or "retrieved content" in t
            or "the document provides" in t
            or "the filing" in t
            or "the provided snippet" in t
            or "for the three months ended" in t
            or "for the year ended" in t
            or "source:" in t
            or "citation" in t
            or "evidence:" in t
            or "from the document" in t
            or "from the source" in t
            or "according to the" in t
            or "http://" in t
            or "https://" in t
        )
        has_extraction_tools = any(
            name in tool_names
            for name in ["retrieve_information", "parse_html_page", "web_search", "edgar_search", "prepare_primary_filing"]
        )
        object_probe = self._extract_candidate_objects(role, output, agent_id="__evidence_probe__")
        object_stats = self._object_signal_stats(object_probe)
        has_grounded_objects = int(object_stats.get("grounded_factual_objects", 0)) > 0
        return (has_doc_trace and has_extraction_tools) or has_grounded_objects

    def _has_primary_filing_evidence(self, output: str, tool_names: list[str], role: str) -> bool:
        return self._has_grounded_evidence(output, tool_names, role)

    def _looks_like_generic_finance_answer(self, output: str) -> bool:
        t = str(output or "").lower()
        generic_markers = [
            "typically makes the following adjustments",
            "usually found in",
            "these adjustments are usually found",
            "non-cash expense",
            "core operational activities",
            "one-time or infrequent costs",
            "other non-operating or non-recurring items",
        ]
        return any(marker in t for marker in generic_markers)

    def _apply_contribution_score_gates(
        self,
        role: str,
        output: str,
        tool_names: list[str],
        env_status: Optional[DatasetEnvStatus],
        raw_score: float,
        raw_reason: str,
    ) -> tuple[float, str]:
        score = max(0.0, min(1.0, float(raw_score)))
        reason = str(raw_reason or "")

        if self._is_no_evidence_output(output):
            return 0.0, "no_evidence output"

        if role == "planner":
            return min(score, 0.7), reason or "planner score capped at 0.7"

        bottleneck = self._infer_runtime_bottleneck_state()
        dominant_bottleneck = str(bottleneck.get("dominant_bottleneck", "coordination"))
        is_search_capable = self._is_search_capable_role(role)
        evidence_ok = self._has_grounded_evidence(output, tool_names, role)
        object_summary = self._extract_candidate_objects(role, output, agent_id="__score_probe__")
        object_stats = self._object_signal_stats(object_summary)
        validated_objects = int(object_stats.get("validated_objects", 0))
        total_objects = len(object_summary)
        grounded_factual_objects = int(object_stats.get("grounded_factual_objects", 0))
        factual_objects = int(object_stats.get("factual_objects", 0))
        critique_objects = int(object_stats.get("critique_objects", 0))
        preferred_hits = list(object_stats.get("preferred_hits", []))

        if self._is_procedural_output(output) and not evidence_ok:
            return 0.0, "procedural output without grounded evidence"

        if not evidence_ok and score > 0.5:
            score = 0.3
            suffix = "capped: no clear grounded evidence"
            reason = f"{reason}; {suffix}" if reason else suffix

        if self._looks_like_generic_finance_answer(output) and not evidence_ok:
            score = min(score, 0.2)
            suffix = "generic finance answer without document-grounded evidence"
            reason = f"{reason}; {suffix}" if reason else suffix

        if evidence_ok and validated_objects > 0:
            boosted = min(1.0, score + 0.1)
            if boosted > score:
                score = boosted
                suffix = f"boosted: {validated_objects} validated information object(s)"
                reason = f"{reason}; {suffix}" if reason else suffix
        elif evidence_ok and total_objects > 0 and score < 0.4:
            score = max(score, 0.4)
            suffix = f"floor: reusable information objects detected ({total_objects})"
            reason = f"{reason}; {suffix}" if reason else suffix

        if evidence_ok and grounded_factual_objects > 0:
            boosted = min(1.0, score + 0.08)
            if boosted > score:
                score = boosted
                suffix = f"boosted: grounded factual objects={grounded_factual_objects}"
                reason = f"{reason}; {suffix}" if reason else suffix

        if preferred_hits and (grounded_factual_objects > 0 or validated_objects > 0):
            boosted = min(1.0, score + 0.08)
            if boosted > score:
                score = boosted
                suffix = f"boosted: fills weak preferred object types {preferred_hits}"
                reason = f"{reason}; {suffix}" if reason else suffix

        if critique_objects > 0 and factual_objects == 0 and grounded_factual_objects == 0 and score > 0.3:
            score = min(score, 0.25)
            suffix = "capped: critique-only output without new grounded objects"
            reason = f"{reason}; {suffix}" if reason else suffix

        if dominant_bottleneck == "retrieval":
            if is_search_capable and evidence_ok and grounded_factual_objects > 0 and score < 0.55:
                score = max(score, 0.55)
                suffix = "floor: retrieval bottleneck rewards new grounded evidence"
                reason = f"{reason}; {suffix}" if reason else suffix
            if not is_search_capable and critique_objects > 0 and grounded_factual_objects == 0 and score > 0.25:
                score = min(score, 0.25)
                suffix = "capped: retrieval bottleneck prioritizes evidence acquisition over critique-only output"
                reason = f"{reason}; {suffix}" if reason else suffix
        elif dominant_bottleneck == "structuring":
            if preferred_hits and factual_objects > 0 and score < 0.5:
                score = max(score, 0.5)
                suffix = "floor: structuring bottleneck rewards reusable object formation"
                reason = f"{reason}; {suffix}" if reason else suffix

        if env_status is not None and not getattr(env_status, "success", False) and score > 0.5:
            score = 0.3
            suffix = "capped: env/tool execution was not successful"
            reason = f"{reason}; {suffix}" if reason else suffix

        return score, reason

    def _evaluate_final_answer_quality(self, answer: str) -> tuple[float, Dict[str, Any]]:
        """Reuse dataset final-answer evaluator (groundtruth-based) to score current final answer."""
        final_answer = str(answer or "")
        if not final_answer.strip():
            return 0.0, {"score": 0.0, "reason": "empty final answer"}

        dataset = getattr(self.template_agent, "dataset", None)
        if dataset is None or not hasattr(dataset, "get_instance_eval_metrics"):
            return self._score_text(final_answer), {"score": self._score_text(final_answer), "reason": "dataset evaluator unavailable"}

        try:
            output = DatasetInstanceOutputWithTrajectory(
                data_instance=self.task_instance,
                agent_output=final_answer,
                trajectory=[],
                final_env_output=DatasetEnvStatus(
                    success=bool(final_answer.strip()),
                    num_steps=int(self.round_idx),
                ),
            )
            metrics = dataset.get_instance_eval_metrics(output) or {}
            raw_score = metrics.get("score", metrics.get("exact_match", metrics.get("avg_score", None)))
            score = float(raw_score) if raw_score is not None else 0.0
            return max(0.0, min(1.0, score)), metrics
        except Exception as exc:
            logger.warning(
                "Dataset final-answer evaluation failed in round %s: %s",
                self.round_idx,
                exc,
            )
            fallback = self._score_text(final_answer)
            return fallback, {"score": fallback, "reason": "fallback:text_heuristic"}

    def _compute_trend(self, scores: list[float]) -> float:
        if len(scores) < 2:
            return 0.0
        return (scores[-1] - scores[0]) / max(1, len(scores) - 1)

    def _extract_memory_summary(self, role: str, output: str) -> str:
        """Extract concise, high-signal memory from agent output."""
        if not output or self._is_low_signal_answer(output) or self._is_procedural_output(output):
            return ""
        text = str(output)
        if self._is_failure_narrative_output(text) and not self._has_concrete_adjustment_facts(text):
            return ""
        # Prefer lines with numbers, adjustment keywords, or clear bullet points.
        keywords = [
            "adjusted ebitda", "net income", "reconciliation", "adjustment",
            "depreciation", "amortization", "stock-based", "stock based",
            "interest", "income tax", "taxes", "other income", "acquisition",
            "lodging", "withholding", "transactional",
            "调整", "净利润", "调节", "折旧", "摊销", "股权激励", "利息", "税", "其他收益",
        ]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        picked = []
        for ln in lines:
            low = ln.lower()
            if any(k in low for k in keywords) or any(ch.isdigit() for ch in ln) or ln.startswith(("-", "*", "•")):
                picked.append(ln)
            if len(picked) >= 6:
                break
        summary = " | ".join(picked) if picked else ""
        if summary:
            role_tag = role.upper()
            summary = f"{role_tag}: {summary}"
        return summary[:400]

    def _has_concrete_adjustment_facts(self, text: str) -> bool:
        t = str(text or "").lower()
        markers = [
            "provision for (benefit from) income taxes",
            "other income (expense), net",
            "other (income) expense, net",
            "other expense, net",
            "interest income",
            "interest expense",
            "depreciation and amortization",
            "stock-based compensation",
            "acquisition-related impacts",
            "contingent consideration",
            "lodging taxes",
            "withholding taxes",
            "transactional taxes",
            "stock-settlement obligations",
            "restructuring charges",
        ]
        matched = sum(1 for marker in markers if marker in t)
        list_like = t.count("*") + t.count("•") + t.count("|")
        return matched >= 3 or (matched >= 2 and list_like >= 2)

    def _is_failure_narrative_output(self, text: str) -> bool:
        t = str(text or "").lower()
        failure_markers = [
            "insufficient",
            "not found",
            "unable to",
            "cannot answer",
            "cannot fulfill",
            "primary_filing bundle is insufficient",
            "no relevant filings were found",
            "data pipeline failure",
            "tool misuse",
            "critical failure",
            "failed to retrieve",
            "missing primary_filing",
            "i am sorry, but i cannot",
            "fail:",
            " fail.",
            "could not be found",
            "remains elusive",
            "not available to determine",
            "cannot complete the verification",
            "same issue",
        ]
        return any(marker in t for marker in failure_markers)

    def _merge_memory(self, existing: str, new: str) -> str:
        if not new:
            return existing or ""
        if not existing:
            return new
        existing_positive = self._has_concrete_adjustment_facts(existing)
        new_positive = self._has_concrete_adjustment_facts(new)
        existing_failure = self._is_failure_narrative_output(existing)
        new_failure = self._is_failure_narrative_output(new)

        # Once we have concrete positive evidence, do not let stale failure
        # narratives dominate the next-round memory context.
        if new_positive and existing_failure and not existing_positive:
            return new
        if existing_positive and new_failure and not new_positive:
            return existing
        if existing_positive and new_positive:
            existing_parts = [part.strip() for part in existing.split(" || ") if part.strip()]
            new_parts = [part.strip() for part in new.split(" || ") if part.strip()]
            merged_parts: list[str] = []
            seen: set[str] = set()
            for part in existing_parts + new_parts:
                if part in seen:
                    continue
                seen.add(part)
                merged_parts.append(part)
            return " || ".join(merged_parts)[:800]
        if existing_failure and new_failure:
            return existing[:800]
        if new in existing:
            return existing
        merged = f"{existing} || {new}"
        return merged[:800]

    def _extract_verifier_source_hints(self) -> dict[str, Any]:
        preferred: list[str] = list(self.source_feedback_memory.get("preferred_source_keys", []) or [])
        rejected: list[str] = []
        reasons: list[str] = list(self.source_feedback_memory.get("rejection_reasons", []) or [])
        hints: list[str] = list(self.source_feedback_memory.get("next_search_hints", []) or [])
        supported_claims: list[str] = list(self.source_feedback_memory.get("supported_claims", []) or [])
        unsupported_claims: list[str] = list(self.source_feedback_memory.get("unsupported_claims", []) or [])
        contradicted_claims: list[str] = list(self.source_feedback_memory.get("contradicted_claims", []) or [])
        missing_support: list[str] = list(self.source_feedback_memory.get("missing_support", []) or [])
        rejected_sources: list[dict[str, str]] = [
            dict(item) for item in (self.source_feedback_memory.get("rejected_sources", []) or [])
            if isinstance(item, dict)
        ]
        required_constraints: dict[str, list[str]] = {
            str(k): list(v) for k, v in (self.source_feedback_memory.get("required_constraints", {}) or {}).items()
            if isinstance(v, list)
        }
        avoid_constraints: dict[str, list[str]] = {
            str(k): list(v) for k, v in (self.source_feedback_memory.get("avoid_constraints", {}) or {}).items()
            if isinstance(v, list)
        }
        prefer_source_patterns: list[str] = list(self.source_feedback_memory.get("prefer_source_patterns", []) or [])
        avoid_source_patterns: list[str] = list(self.source_feedback_memory.get("avoid_source_patterns", []) or [])
        missing_targets: list[str] = list(self.source_feedback_memory.get("missing_targets", []) or [])

        def _add_unique(target: list[str], value: str, limit: int = 5) -> None:
            value = str(value or "").strip()
            if not value or value in target or len(target) >= limit:
                return
            target.append(value)

        def _add_unique_dict(target: list[dict[str, str]], item: dict[str, str], limit: int = 8) -> None:
            source_key = str(item.get("source_key", "") or "").strip()
            reason = str(item.get("reason", "") or "").strip()
            details = str(item.get("details", "") or "").strip()
            if not (source_key or reason or details):
                return
            normalized = {"source_key": source_key[:120], "reason": reason[:80], "details": details[:180]}
            if normalized in target or len(target) >= limit:
                return
            target.append(normalized)

        def _merge_constraint_map(target: dict[str, list[str]], raw: Any) -> None:
            if not isinstance(raw, dict):
                return
            for key, values in raw.items():
                key_text = str(key or "").strip()
                if not key_text:
                    continue
                current = target.setdefault(key_text, [])
                if isinstance(values, list):
                    for value in values:
                        _add_unique(current, str(value), limit=8)
                elif values not in (None, ""):
                    _add_unique(current, str(values), limit=8)

        for aid in self.controller.get_agent_ids():
            state = self.controller.population.get_agent(aid)
            role = _canonicalize_role(state.role if state else "")
            if role not in {"verifier", "schema_verifier", "auditor", "critic"}:
                continue
            text = "\n".join(
                [
                    str(getattr(state, "memory_summary", "") or ""),
                    str(self.latest_outputs.get(aid, "") or ""),
                    str(getattr(state, "workflow_correction", "") or ""),
                    str(getattr(state, "next_round_workflow", "") or ""),
                ]
            )
            if not text.strip():
                continue
            payload = self._extract_json_payload(text)
            if payload:
                source_feedback = payload.get("source_feedback")
                if isinstance(source_feedback, dict):
                    for key in source_feedback.get("preferred_source_keys", []) or []:
                        _add_unique(preferred, key)
                    for item in source_feedback.get("rejected_sources", []) or []:
                        if isinstance(item, dict):
                            _add_unique_dict(rejected_sources, item)
                            _add_unique(rejected, str(item.get("source_key", "") or ""))
                            _add_unique(reasons, str(item.get("reason", "") or item.get("details", "") or ""))
                    for reason in source_feedback.get("rejection_reasons", []) or []:
                        _add_unique(reasons, str(reason))
                    for hint in source_feedback.get("next_search_hints", []) or []:
                        _add_unique(hints, str(hint))
                    _merge_constraint_map(required_constraints, source_feedback.get("required_constraints"))
                    _merge_constraint_map(avoid_constraints, source_feedback.get("avoid_constraints"))
                    for value in source_feedback.get("prefer_source_patterns", []) or []:
                        _add_unique(prefer_source_patterns, str(value), limit=8)
                    for value in source_feedback.get("avoid_source_patterns", []) or []:
                        _add_unique(avoid_source_patterns, str(value), limit=8)
                    for value in source_feedback.get("missing_targets", []) or []:
                        _add_unique(missing_targets, str(value), limit=8)
                search_constraints = payload.get("next_search_constraints")
                if isinstance(search_constraints, dict):
                    _merge_constraint_map(required_constraints, search_constraints)
                for key in payload.get("preferred_source_keys", []) or []:
                    _add_unique(preferred, key)
                for item in payload.get("rejected_source_keys", []) or []:
                    if isinstance(item, dict):
                        _add_unique(rejected, str(item.get("key", "") or ""))
                        _add_unique(reasons, str(item.get("reason", "") or ""))
                        _add_unique_dict(
                            rejected_sources,
                            {
                                "source_key": str(item.get("key", "") or ""),
                                "reason": str(item.get("reason", "") or ""),
                                "details": str(item.get("details", "") or ""),
                            },
                        )
                    else:
                        _add_unique(rejected, str(item))
                for reason in payload.get("rejection_reasons", []) or []:
                    _add_unique(reasons, str(reason))
                _add_unique(hints, str(payload.get("next_search_hint", "") or ""))
                _add_unique(hints, str(payload.get("corrective_hint", "") or ""))
                for err in payload.get("field_semantic_errors", []) or []:
                    _add_unique(reasons, f"semantic_error:{err}")
                for claim in payload.get("supported_claims", []) or []:
                    _add_unique(supported_claims, str(claim), limit=8)
                for claim in payload.get("unsupported_claims", []) or []:
                    _add_unique(unsupported_claims, str(claim), limit=8)
                for claim in payload.get("contradicted_claims", []) or []:
                    _add_unique(contradicted_claims, str(claim), limit=8)
                for claim in payload.get("missing_support", []) or []:
                    _add_unique(missing_support, str(claim), limit=8)
                for target in payload.get("missing_targets", []) or []:
                    _add_unique(missing_targets, str(target), limit=8)
                for field in payload.get("missing_schema_fields", []) or []:
                    _add_unique(missing_targets, f"missing_field:{field}", limit=8)

            for match in re.findall(r"primary_filing_[1-5]", text):
                low = text.lower()
                around = low[max(0, low.find(match) - 80): low.find(match) + 160] if match in low else low
                if any(word in around for word in ["reject", "irrelevant", "wrong", "bad", "not relevant", "skip"]):
                    _add_unique(rejected, match)
                    _add_unique_dict(
                        rejected_sources,
                        {"source_key": match, "reason": "rejected_in_verifier_text", "details": around[:180]},
                    )
                elif any(word in around for word in ["prefer", "best", "relevant", "trust", "use", "chosen"]):
                    _add_unique(preferred, match)
            low_text = text.lower()
            if "north america only" in low_text or "regional" in low_text:
                _merge_constraint_map(required_constraints, {"scope": ["global"]})
                _merge_constraint_map(avoid_constraints, {"scope": ["north america", "regional"]})
                _add_unique(reasons, "scope_mismatch")
                _add_unique(hints, "Avoid regional scope; search for global metric/source instead.")
                _add_unique(prefer_source_patterns, "global", limit=8)
                _add_unique(avoid_source_patterns, "north america", limit=8)

        return {
            "preferred_source_keys": preferred,
            "rejected_source_keys": rejected,
            "rejected_sources": rejected_sources,
            "rejection_reasons": reasons,
            "next_search_hints": hints,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "contradicted_claims": contradicted_claims,
            "missing_support": missing_support,
            "required_constraints": required_constraints,
            "avoid_constraints": avoid_constraints,
            "prefer_source_patterns": prefer_source_patterns,
            "avoid_source_patterns": avoid_source_patterns,
            "missing_targets": missing_targets,
        }

    def _memory_has_adjustments(self, memory_text: str) -> bool:
        if not memory_text:
            return False
        t = memory_text.lower()
        return any(k in t for k in [
            "adjusted ebitda", "reconciliation", "adjustment",
            "depreciation", "amortization", "stock-based", "interest",
            "income tax", "other income", "acquisition", "lodging",
            "调整", "净利润", "折旧", "摊销", "股权激励", "利息", "税", "其他收益",
        ])

    def _missing_adjustments_from_memory(self, memory_text: str) -> list[str]:
        if not memory_text:
            return []
        t = memory_text.lower()
        checklist = [
            ("provision for income taxes", ["income tax", "taxes", "税"]),
            ("other income (expense), net", ["other income", "other expense", "其他收益", "其他费用"]),
            ("interest income", ["interest income", "利息收入"]),
            ("depreciation and amortization", ["depreciation", "amortization", "折旧", "摊销"]),
            ("stock-based compensation", ["stock-based", "股权激励"]),
            ("acquisition-related impacts", ["acquisition", "contingent consideration", "收购"]),
            ("lodging/withholding/transactional taxes", ["lodging", "withholding", "transactional", "交易税", "住宿税", "代扣税"]),
        ]
        missing = []
        for label, keys in checklist:
            if not any(k in t for k in keys):
                missing.append(label)
        return missing[:5]

    def _memory_covers_all_adjustments(self, memory_text: str) -> bool:
        return len(self._missing_adjustments_from_memory(memory_text)) == 0 if memory_text else False

    def _update_source_feedback_memory(self, role: str, output: str) -> None:
        role = _canonicalize_role(role)
        if role not in {"verifier", "schema_verifier", "auditor", "critic"}:
            return
        hints = self._extract_verifier_source_hints()
        if not hints:
            return
        self.source_feedback_memory = {
            "preferred_source_keys": list(hints.get("preferred_source_keys", []) or [])[:8],
            "rejected_sources": list(hints.get("rejected_sources", []) or [])[:8],
            "rejection_reasons": list(hints.get("rejection_reasons", []) or [])[:8],
            "next_search_hints": list(hints.get("next_search_hints", []) or [])[:8],
            "supported_claims": list(hints.get("supported_claims", []) or [])[:8],
            "unsupported_claims": list(hints.get("unsupported_claims", []) or [])[:8],
            "contradicted_claims": list(hints.get("contradicted_claims", []) or [])[:8],
            "missing_support": list(hints.get("missing_support", []) or [])[:8],
            "required_constraints": dict(hints.get("required_constraints", {}) or {}),
            "avoid_constraints": dict(hints.get("avoid_constraints", {}) or {}),
            "prefer_source_patterns": list(hints.get("prefer_source_patterns", []) or [])[:8],
            "avoid_source_patterns": list(hints.get("avoid_source_patterns", []) or [])[:8],
            "missing_targets": list(hints.get("missing_targets", []) or [])[:8],
        }

    def _derive_missing_search_targets(self, role: str = "searcher") -> list[str]:
        targets: list[str] = []
        coverage = self._coverage_status()
        for slot in coverage.get("missing_slots", [])[:6]:
            dims = slot.get("dimensions", {}) or {}
            target_text = ", ".join(f"{k}={v}" for k, v in dims.items() if v)
            if target_text and target_text not in targets:
                targets.append(target_text)
        for raw in self.source_feedback_memory.get("missing_targets", []) or []:
            text = str(raw or "").strip()
            if text and text not in targets and len(targets) < 8:
                targets.append(text)
        notes = str(self.final_notes or "")
        for match in re.findall(r"incomplete_range_bundle_missing_fields=([^\n|]+)", notes):
            for field in re.split(r"[,;/]", match):
                token = str(field or "").strip()
                if token and token not in targets and len(targets) < 8:
                    targets.append(f"missing_field:{token}")
        return targets[:8]

    def _format_search_constraints_block(self) -> str:
        memory = self.source_feedback_memory or {}
        required_constraints = memory.get("required_constraints", {}) or {}
        avoid_constraints = memory.get("avoid_constraints", {}) or {}
        prefer_patterns = memory.get("prefer_source_patterns", []) or []
        avoid_patterns = memory.get("avoid_source_patterns", []) or []
        preferred_keys = memory.get("preferred_source_keys", []) or []
        supported_claims = memory.get("supported_claims", []) or []
        unsupported_claims = memory.get("unsupported_claims", []) or []
        contradicted_claims = memory.get("contradicted_claims", []) or []
        missing_support = memory.get("missing_support", []) or []
        rejected_sources = memory.get("rejected_sources", []) or []
        rejected_preview = [
            f"{item.get('source_key') or 'unknown'} ({item.get('reason') or item.get('details') or 'rejected'})"
            for item in rejected_sources[:5]
            if isinstance(item, dict)
        ]
        lines = []
        if required_constraints:
            lines.append("Required constraints:")
            for key, values in required_constraints.items():
                if values:
                    lines.append(f"- {key}: {', '.join(str(v) for v in values[:6])}")
        if avoid_constraints:
            lines.append("Avoid constraints:")
            for key, values in avoid_constraints.items():
                if values:
                    lines.append(f"- {key}: {', '.join(str(v) for v in values[:6])}")
        if prefer_patterns:
            lines.append(f"Prefer source/query patterns: {', '.join(str(v) for v in prefer_patterns[:6])}")
        if avoid_patterns:
            lines.append(f"Avoid source/query patterns: {', '.join(str(v) for v in avoid_patterns[:6])}")
        if preferred_keys:
            lines.append(f"Preferred stored source keys: {', '.join(str(v) for v in preferred_keys[:6])}")
        if supported_claims:
            lines.append("Verifier-supported claims:")
            lines.extend(f"- {item}" for item in supported_claims[:5])
        if unsupported_claims:
            lines.append("Unsupported claims to avoid repeating:")
            lines.extend(f"- {item}" for item in unsupported_claims[:5])
        if contradicted_claims:
            lines.append("Contradicted claims:")
            lines.extend(f"- {item}" for item in contradicted_claims[:5])
        if missing_support:
            lines.append("Claims still missing direct support:")
            lines.extend(f"- {item}" for item in missing_support[:5])
        if rejected_preview:
            lines.append("Rejected sources in memory:")
            lines.extend(f"- {item}" for item in rejected_preview)
        return "\n".join(lines).strip()

    def _build_search_query_intent(self, missing_targets: list[str], verifier_hints: dict[str, Any]) -> str:
        if not missing_targets and not any(verifier_hints.get(key) for key in [
            "required_constraints", "avoid_constraints", "prefer_source_patterns", "avoid_source_patterns"
        ]):
            return ""
        required_constraints = verifier_hints.get("required_constraints", {}) or {}
        avoid_constraints = verifier_hints.get("avoid_constraints", {}) or {}
        prefer_patterns = verifier_hints.get("prefer_source_patterns", []) or []
        avoid_patterns = verifier_hints.get("avoid_source_patterns", []) or []

        intent_parts: list[str] = []
        if missing_targets:
            intent_parts.append("targets: " + " ; ".join(str(v) for v in missing_targets[:5]))
        for key, values in required_constraints.items():
            if values:
                intent_parts.append(f"must_match_{key}: " + ", ".join(str(v) for v in values[:4]))
        for key, values in avoid_constraints.items():
            if values:
                intent_parts.append(f"avoid_{key}: " + ", ".join(str(v) for v in values[:4]))
        if prefer_patterns:
            intent_parts.append("prefer_patterns: " + ", ".join(str(v) for v in prefer_patterns[:4]))
        if avoid_patterns:
            intent_parts.append("avoid_patterns: " + ", ".join(str(v) for v in avoid_patterns[:4]))
        if not intent_parts:
            return ""
        return "SEARCH_QUERY_INTENT: " + " | ".join(intent_parts)

    def _is_slow_update_round(self) -> bool:
        """Whether current fast round will immediately trigger slow update."""
        next_slow = int(getattr(self.controller, "next_slow_update_at", 0))
        projected_fast_time = int(getattr(self.controller, "fast_time_step", 0)) + 1
        return projected_fast_time >= next_slow

    def _update_agent_state(
        self,
        agent_id: str,
        output: str,
        env_status: DatasetEnvStatus,
        query: str = "",
        local_score: Optional[float] = None,
    ) -> None:
        state = self.controller.population.get_agent(agent_id)
        if state is None:
            return

        if local_score is None:
            local_score = self._score_text(output, env_status)
        state.output = output[:1200]
        state.recent_scores.append(local_score)
        state.meta_score = max(0.0, min(1.0, float(local_score)))
        state.score_reason = state.score_reason or "runtime local contribution score"
        state.recent_scores = state.recent_scores[-10:]
        state.score_trend = self._compute_trend(state.recent_scores)
        state.last_update_time = self.controller.fast_time_step
        role = _canonicalize_role(state.role)
        memory_update = self._extract_memory_summary(role, output)
        if memory_update:
            state.memory_summary = self._merge_memory(state.memory_summary, memory_update)
        self._update_coverage_memory(role, output, agent_id)
        self._update_source_feedback_memory(role, output)
        self._update_information_objects(role, output, agent_id)
        state.improvement_direction = "keep refining" if not env_status.success else "validated by env"
        state.failure_modes = self._extract_failure_modes(output, env_status)
        self.controller.update_agent_state(state)

    def _synthesize_top_k_with_llm(self, candidates: list) -> str:
        """Ask meta LLM to summarize top-k agent outputs into a final answer.

        candidates: list of (aid, score, output) sorted by score descending.
        Returns the summarized answer string, or "" on failure.
        """
        task_text = self._instance_text()
        filtered = [
            (aid, score, output)
            for (aid, score, output) in candidates
            if output
            and not self._is_low_signal_answer(output)
            and not self._is_procedural_output(output)
        ]
        if not filtered:
            return ""
        if any(self._has_concrete_adjustment_facts(output) for _, _, output in filtered):
            filtered = [
                (aid, score, output)
                for (aid, score, output) in filtered
                if self._has_concrete_adjustment_facts(output)
                or not self._is_failure_narrative_output(output)
            ]
        parts = []
        for rank, (aid, score, output) in enumerate(filtered, 1):
            state = self.controller.population.get_agent(aid)
            role = _canonicalize_role(state.role if state else "searcher")
            parts.append(
                f"[Agent {rank} | role={role} | contribution_score={score:.3f}]\n{output}"
            )
        combined = "\n\n---\n\n".join(parts)
        prompt = (
            "The following agents have worked on the task below. "
            "Their outputs are listed in order of contribution score (highest first).\n\n"
            f"Task:\n{task_text}\n\n"
            f"Agent Outputs:\n{combined}\n\n"
            "Please synthesize the above agent outputs into a single, concise final answer for the task. "
            "Rules:\n"
            "1. Prefer information from higher-scored agents when there is a conflict.\n"
            "2. IGNORE any agent output that is purely procedural (e.g. 'I am currently searching', "
            "'extracting now', 'processing', planning steps, or similar in-progress descriptions). "
            "Only use outputs that contain concrete facts, numbers, or verified findings.\n"
            "3. If all agent outputs are procedural or empty, reply with: "
            "'INSUFFICIENT_DATA: no concrete findings available.'\n"
            "Return ONLY the final answer — no preamble, no explanation."
        )
        # Prefer meta LLM (gemini-2.5-flash); fall back to judge LLM (gpt-4o-mini)
        meta_llm = getattr(getattr(self.controller, "meta_llm", None), "llm", None)
        llm = meta_llm if meta_llm is not None else self.judge_llm
        llm_name = "meta_llm" if meta_llm is not None else "judge_llm"
        try:
            resp = llm.invoke(
                [{"role": "user", "content": prompt}],
                num_retries=0,
            )
            answer = getattr(resp, "content", "") or ""
            answer = self._sanitize_final_answer_text(answer)
            if answer and not self._is_low_signal_answer(answer) and not self._is_procedural_output(answer):
                logger.info(
                    "Top-k synthesis via %s succeeded (k=%d, len=%d)", llm_name, len(candidates), len(answer)
                )
                return answer
        except Exception as exc:
            logger.warning("Top-k synthesis via %s failed: %s", llm_name, exc)
        return ""

    def _synthesize_from_memory(self) -> str:
        memories = []
        positive_memories = []
        for aid in self.controller.get_agent_ids():
            if not self._eligible_for_final_answer_path(aid):
                continue
            state = self.controller.population.get_agent(aid)
            if not state:
                continue
            mem = str(state.memory_summary or "").strip()
            if not mem:
                continue
            memories.append(mem)
            if self._has_concrete_adjustment_facts(mem):
                positive_memories.append(mem)
        if not memories:
            return ""
        selected_memories = positive_memories if positive_memories else memories
        prompt = (
            "You are synthesizing the final answer from agent memory summaries. "
            "Use ONLY concrete facts from memory (adjustment items, numbers, sources). "
            "If positive concrete memories are present, DO NOT be overly conservative and do NOT output INSUFFICIENT_DATA. "
            "Summarize the strongest supported answer from those memories. "
            "If memory is insufficient, reply exactly: INSUFFICIENT_DATA: memory lacks concrete findings.\n\n"
            f"Task:\n{self._instance_text()}\n\n"
            "Memory summaries:\n"
            + "\n".join(f"- {m}" for m in selected_memories)
        )
        meta_llm = getattr(getattr(self.controller, "meta_llm", None), "llm", None)
        llm = meta_llm if meta_llm is not None else self.judge_llm
        try:
            resp = llm.invoke([{"role": "user", "content": prompt}], num_retries=0)
            answer = getattr(resp, "content", "") or ""
            answer = self._sanitize_final_answer_text(answer)
            if answer and not self._is_low_signal_answer(answer) and not self._is_procedural_output(answer):
                return answer
        except Exception:
            return ""
        return ""

    def _extract_range_comparison_bundles(self) -> list[Dict[str, Any]]:
        bundles: list[Dict[str, Any]] = []
        texts: list[str] = []
        texts.extend(str(v or "") for v in self.latest_outputs.values())
        for aid in self.controller.get_agent_ids():
            state = self.controller.population.get_agent(aid)
            if state and str(getattr(state, "memory_summary", "") or "").strip():
                texts.append(str(state.memory_summary))

        for text in texts:
            if not text:
                continue
            payload = self._extract_json_payload(text)
            schema_facts = payload.get("schema_facts") if isinstance(payload, dict) else None
            if isinstance(schema_facts, dict) and isinstance(schema_facts.get("range_comparison"), dict):
                fact = dict(schema_facts.get("range_comparison") or {})
                if fact:
                    fact["_confidence"] = 1.0 if str(fact.get("source", "")).startswith("primary_filing") else 0.75
                    bundles.append(fact)
                    continue

            lowered = text.lower()
            actual_match = re.search(
                r"(?:actual|reported|was|margin was|value was)[^\\n]{0,80}?(-?\d+(?:\.\d+)?)\s*(%|percent|percentage points|bps|basis points)\b",
                lowered,
            )
            range_match = re.search(
                r"(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points)\s*(?:to|-|–|—)\s*(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points)",
                lowered,
            )
            if not actual_match or not range_match:
                continue
            unit = actual_match.group(2)
            low = float(range_match.group(1))
            high = float(range_match.group(3))
            if low > high:
                low, high = high, low
            bundles.append(
                {
                    "metric": "",
                    "actual_value": actual_match.group(1),
                    "actual_unit": unit,
                    "range_low": f"{low:g}",
                    "range_high": f"{high:g}",
                    "range_unit": range_match.group(2),
                    "direction": "above" if "above" in lowered or "beat" in lowered else ("below" if "below" in lowered or "miss" in lowered else "compare"),
                    "source": "primary_filing_heuristic" if "primary_filing" in lowered else "",
                    "evidence_text": text[:300],
                    "_confidence": 0.65 if "primary_filing" in lowered else 0.5,
                }
            )
        return bundles

    def _bundle_missing_required_fields(self, bundle: Dict[str, Any]) -> list[str]:
        schema_family = str((self.task_schema or {}).get("schema_family", ""))
        if schema_family != "range_comparison":
            return []
        missing: list[str] = []
        field_aliases = {
            "actual_value": ["actual_value"],
            "range_low": ["range_low"],
            "range_high": ["range_high"],
            "source": ["source"],
        }
        for field_name, aliases in field_aliases.items():
            if not any(str(bundle.get(alias, "") or "").strip() for alias in aliases):
                missing.append(field_name)
        return missing

    def _bundle_satisfies_task_schema(self, bundle: Dict[str, Any]) -> bool:
        return len(self._bundle_missing_required_fields(bundle)) == 0

    def _answer_satisfies_task_schema(self, text: str) -> bool:
        schema_family = str((self.task_schema or {}).get("schema_family", ""))
        if not text:
            return False
        if schema_family != "range_comparison":
            return True
        payload = self._extract_json_payload(text)
        schema_facts = payload.get("schema_facts") if isinstance(payload, dict) else None
        if isinstance(schema_facts, dict) and isinstance(schema_facts.get("range_comparison"), dict):
            return self._bundle_satisfies_task_schema(dict(schema_facts.get("range_comparison") or {}))
        lowered = str(text).lower()
        return (
            ("versus low end" in lowered or "low end:" in lowered or "from low end" in lowered)
            and ("versus high end" in lowered or "high end:" in lowered or "from high end" in lowered)
        )

    def _synthesize_from_task_schema(self) -> str:
        schema_family = str((self.task_schema or {}).get("schema_family", ""))
        if schema_family != "range_comparison":
            return ""
        bundles = self._extract_range_comparison_bundles()
        if not bundles:
            return ""
        complete_bundles = [bundle for bundle in bundles if self._bundle_satisfies_task_schema(bundle)]
        if not complete_bundles:
            missing_sets = [self._bundle_missing_required_fields(bundle) for bundle in bundles[:3]]
            missing_text = "; ".join(", ".join(items) for items in missing_sets if items)
            if missing_text:
                existing = str(self.final_notes or "").strip()
                note = f"incomplete_range_bundle_missing_fields={missing_text}"
                self.final_notes = (existing + " || " + note).strip(" |")[:1200] if existing else note[:1200]
            return ""
        bundles = sorted(
            complete_bundles,
            key=lambda b: (
                float(b.get("_confidence", 0.0) or 0.0),
                1 if str(b.get("source", "")).startswith("primary_filing") else 0,
                len(str(b.get("evidence_text", "") or "")),
            ),
            reverse=True,
        )
        best = bundles[0]
        try:
            actual = float(best.get("actual_value"))
            low = float(best.get("range_low"))
            high = float(best.get("range_high"))
        except (TypeError, ValueError):
            return ""

        actual_unit = str(best.get("actual_unit", "") or "").strip() or str(best.get("range_unit", "") or "").strip()
        source = str(best.get("source", "") or "").strip()
        metric = str(best.get("metric", "") or "").strip()
        label = metric or "Actual value"
        delta_low = actual - low
        delta_high = actual - high

        if actual_unit in {"%", "percent"}:
            delta_low_text = f"{delta_low * 100:.0f} bps"
            delta_high_text = f"{delta_high * 100:.0f} bps"
            actual_text = f"{actual:g}%"
            low_text = f"{low:g}%"
            high_text = f"{high:g}%"
        elif actual_unit in {"bps", "basis points"}:
            delta_low_text = f"{delta_low:g} bps"
            delta_high_text = f"{delta_high:g} bps"
            actual_text = f"{actual:g} bps"
            low_text = f"{low:g} bps"
            high_text = f"{high:g} bps"
        else:
            suffix = f" {actual_unit}".rstrip()
            delta_low_text = f"{delta_low:g}{suffix}"
            delta_high_text = f"{delta_high:g}{suffix}"
            actual_text = f"{actual:g}{suffix}"
            low_text = f"{low:g}{suffix}"
            high_text = f"{high:g}{suffix}"

        source_text = f" (source: {source})" if source else ""
        lines = [
            f"{label}: {actual_text}{source_text}",
            f"Reference range: {low_text} to {high_text}",
            f"Versus low end: {delta_low_text}",
            f"Versus high end: {delta_high_text}",
        ]
        return "\n".join(lines)

    def _prefer_historical_best_answer(self, candidate: str) -> str:
        candidate_text = self._sanitize_final_answer_text(candidate)
        best_text = str(self.best_final_answer or "").strip()
        candidate_schema_ok = self._answer_satisfies_task_schema(candidate_text)
        best_schema_ok = self._answer_satisfies_task_schema(best_text) if best_text else False
        if not candidate_schema_ok:
            if best_schema_ok:
                return best_text
            return ""
        if not best_text:
            return candidate_text
        if not candidate_text:
            return best_text
        if (
            self._is_low_signal_answer(candidate_text)
            or self._is_procedural_output(candidate_text)
            or self._is_no_evidence_output(candidate_text)
            or self._is_failure_narrative_output(candidate_text)
        ):
            return best_text
        if self._in_conservative_mode() and len(candidate_text) < max(40, int(len(best_text) * 0.35)):
            return best_text
        return candidate_text

    def _sanitize_final_answer_text(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        notes_parts: list[str] = []
        main = raw
        # Normalize inline bullet blobs into line-wise items so we can keep facts
        # while dropping trailing failure narrative.
        main = re.sub(r"\s+\*\s+(?=\*\*?\d{4}\b)", "\n* ", main)
        main = re.sub(r"\s+\*\s+(?=\d{4}\b)", "\n* ", main)
        cut_markers = [
            "Verdict:", "**Verdict:**", "Reasoning:", "**Reasoning:**",
            "Conflict List:", "**Conflict List:**", "Notes:", "**Notes:**",
        ]
        cut_positions = [main.find(marker) for marker in cut_markers if marker in main]
        if cut_positions:
            cut = min(pos for pos in cut_positions if pos >= 0)
            notes_parts.append(main[cut:].strip())
            main = main[:cut].strip()

        failure_sentence_patterns = [
            r"(?i)\btherefore[, ]+i cannot definitively answer[^.]*\.",
            r"(?i)\binsufficient data[^.]*\.",
            r"(?i)\bi cannot definitively answer[^.]*\.",
            r"(?i)\bi am unable to (?:proceed|complete|gather|verify)[^.]*\.",
            r"(?i)\bi am still encountering[^.]*\.",
            r"(?i)\bthe available (?:documents|evidence|sources)[^.]*do not provide[^.]*\.",
            r"(?i)\bthe document does not specify[^.]*\.",
            r"(?i)\bdata for [^.]* is not available to determine[^.]*\.",
            r"(?i)\bwithout this data point[^.]*\.",
            r"(?i)\bthe [^.]* remains elusive[^.]*\.",
            r"(?i)\bthe [^.]* could not be found[^.]*\.",
            r"(?i)\bi cannot complete the verification[^.]*\.",
            r"(?i)\bmaking it impossible to fully determine[^.]*\.",
            r"(?i)\bimpossible to fully determine[^.]*\.",
            r"(?i)\bverdict:\s*fail\b[^.]*\.?",
            r"(?i)\bfail\b\.?",
        ]
        cleaned = main
        for pattern in failure_sentence_patterns:
            cleaned, removed = re.subn(pattern, " ", cleaned)
            if removed:
                notes_parts.append(pattern)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

        # Drop remaining line fragments that are clearly failure/process narration,
        # while keeping compact factual lines/bullets.
        failure_line_markers = [
            "encountering the same issue",
            "remains elusive",
            "could not be found",
            "not available to determine",
            "cannot complete the verification",
            "no concrete findings available",
            "insufficient_data:",
            "fail",
        ]
        kept_lines: list[str] = []
        dropped_lines: list[str] = []
        for line in re.split(r"\n+", cleaned):
            ln = line.strip()
            if not ln:
                continue
            low = ln.lower()
            if any(marker in low for marker in failure_line_markers):
                dropped_lines.append(ln[:160])
                continue
            kept_lines.append(ln)
        if dropped_lines:
            notes_parts.extend(dropped_lines)
        cleaned = "\n".join(kept_lines).strip()

        if notes_parts:
            merged_notes = " | ".join(part for part in notes_parts if part)
            if merged_notes:
                existing = str(self.final_notes or "").strip()
                self.final_notes = (existing + " || " + merged_notes).strip(" |")[:1200] if existing else merged_notes[:1200]

        factual_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not factual_lines and notes_parts:
            return ""
        if len(" ".join(factual_lines)) < 24 and self._is_failure_narrative_output(raw):
            return ""
        candidate = cleaned or raw
        if self._is_procedural_output(candidate) or self._is_low_signal_answer(candidate) or self._is_no_evidence_output(candidate):
            return ""
        return candidate

    def _is_reflection_feedback_only_agent(self, agent_id: str) -> bool:
        """True when an agent is connected only through auxiliary reflection edges."""
        incident_types: list[EdgeType] = []
        try:
            for neighbor in self.controller.graph.get_neighbors(agent_id, direction="in"):
                edge_type = self.controller.graph.get_edge_type(neighbor, agent_id)
                if edge_type is not None:
                    incident_types.append(edge_type)
            for neighbor in self.controller.graph.get_neighbors(agent_id, direction="out"):
                edge_type = self.controller.graph.get_edge_type(agent_id, neighbor)
                if edge_type is not None:
                    incident_types.append(edge_type)
        except Exception:
            return False
        return bool(incident_types) and all(
            edge_type == EdgeType.REFLECTION_FEEDBACK for edge_type in incident_types
        )

    def _eligible_for_final_answer_path(self, agent_id: str) -> bool:
        """Reflection-only nodes provide process feedback, not answer evidence."""
        return not self._is_reflection_feedback_only_agent(agent_id)

    def _synthesize_answer(self, meta_decision=None) -> str:
        complex_task = self._is_complex_evidence_task()
        schema_answer = self._synthesize_from_task_schema()
        if schema_answer:
            return self._prefer_historical_best_answer(schema_answer)

        coverage_answer = self._synthesize_from_coverage_memory()
        if coverage_answer and not complex_task:
            return self._prefer_historical_best_answer(coverage_answer)

        if meta_decision is not None:
            contrib = getattr(getattr(meta_decision, "final_synthesis", None), "contributors", [])
            if contrib:
                picked = [
                    self.latest_outputs.get(aid, "")
                    for aid in contrib
                    if self.latest_outputs.get(aid)
                    and self._eligible_for_final_answer_path(aid)
                ]
                if picked:
                    return self._prefer_historical_best_answer("\n\n".join(picked))

        memory_answer = self._synthesize_from_memory()
        if memory_answer:
            return self._prefer_historical_best_answer(memory_answer)

        if coverage_answer:
            return self._prefer_historical_best_answer(coverage_answer)

        # Prefer an answer explicitly submitted via env (submit_final_result).
        successful_answers = []
        for aid, status in self.latest_env_status.items():
            if not self._eligible_for_final_answer_path(aid):
                continue
            if not getattr(status, "success", False):
                continue
            output = self.latest_outputs.get(aid, "")
            if not output:
                continue
            if self._is_low_signal_answer(output) or self._is_procedural_output(output) or self._is_no_evidence_output(output):
                continue
            successful_answers.append(output)
        if successful_answers:
            for ans in reversed(successful_answers):
                return self._prefer_historical_best_answer(ans)
            # All submitted answers are low-signal; fall through to tool-backed scoring
            # rather than returning a procedural placeholder.
            logger.warning(
                "All %d submitted answer(s) are low-signal; falling through to tool-backed scoring.",
                len(successful_answers),
            )

        tool_backed_ids = []
        for aid in self.controller.get_agent_ids():
            if not self._eligible_for_final_answer_path(aid):
                continue
            output = self.latest_outputs.get(aid, "")
            if not output or self._is_low_signal_answer(output) or self._is_procedural_output(output):
                continue
            if self._is_no_evidence_output(output):
                continue
            state = self.controller.population.get_agent(aid)
            role = _canonicalize_role(state.role if state else "searcher")
            tool_names = self.latest_tool_names.get(aid, [])
            if tool_names and role != "planner":
                tool_backed_ids.append(aid)
        if tool_backed_ids:
            def _contrib_score(aid):
                state = self.controller.population.get_agent(aid)
                return state.recent_scores[-1] if state and state.recent_scores else 0.0
            sorted_ids = sorted(
                tool_backed_ids,
                key=lambda aid: (
                    _contrib_score(aid),
                    len(self.latest_tool_names.get(aid, [])),
                    len(self.latest_outputs.get(aid, "")),
                ),
                reverse=True,
            )
            top_k = int(os.getenv("SYNTHESIS_TOP_K", "3"))
            candidates = [(aid, _contrib_score(aid), self.latest_outputs.get(aid, "")) for aid in sorted_ids[:top_k]]
            # If multiple candidates with positive score, synthesize via LLM
            if len(candidates) > 1 and candidates[0][1] > 0.0:
                synthesized = self._synthesize_top_k_with_llm(candidates)
                if synthesized:
                    return self._prefer_historical_best_answer(synthesized)
            return self._prefer_historical_best_answer(candidates[0][2])

        best_id = None
        best_score = -1.0
        for aid in self.controller.get_agent_ids():
            if not self._eligible_for_final_answer_path(aid):
                continue
            state = self.controller.population.get_agent(aid)
            score = state.recent_scores[-1] if state and state.recent_scores else 0.0
            if score > best_score and self.latest_outputs.get(aid):
                if (
                    self._is_low_signal_answer(self.latest_outputs.get(aid, ""))
                    or self._is_procedural_output(self.latest_outputs.get(aid, ""))
                    or self._is_no_evidence_output(self.latest_outputs.get(aid, ""))
                ):
                    continue
                role = _canonicalize_role(state.role if state else "searcher")
                if role == "planner" and any(
                    self.latest_outputs.get(other, "") and not self._is_low_signal_answer(self.latest_outputs.get(other, ""))
                    for other in self.controller.get_agent_ids()
                    if other != aid
                ):
                    continue
                best_score = score
                best_id = aid
        if best_id is not None:
            return self._prefer_historical_best_answer(self.latest_outputs.get(best_id, ""))
        return self._prefer_historical_best_answer("")

    def _apply_agent_feedback(self, decision) -> None:
        self.latest_agent_feedback = {}
        if decision is None:
            return

        for feedback in getattr(decision, "agent_feedback", []) or []:
            if not feedback.agent_id:
                continue
            self.latest_agent_feedback[feedback.agent_id] = feedback
            state = self.controller.population.get_agent(feedback.agent_id)
            if state is None:
                continue
            state.meta_score = max(0.0, min(1.0, float(feedback.score)))
            state.score_reason = feedback.score_reason
            state.workflow_correction = feedback.workflow_correction
            state.next_round_workflow = feedback.next_round_workflow
            state.improvement_direction = feedback.next_round_workflow or feedback.workflow_correction or state.improvement_direction
            self.controller.update_agent_state(state)

    def run_fast_round(self) -> float:
        """Run one fast-timescale MAS round."""
        self.round_idx += 1
        self._sync_runtime_agents()
        graph_before = self._snapshot_graph()
        logger.info(
            "=== Fast round %s start: agents=%s, edges=%s ===",
            self.round_idx,
            len(self.controller.get_agent_ids()),
            len(self.controller.graph.edges),
        )

        round_outputs: Dict[str, str] = {}
        round_agent_details: list[Dict[str, Any]] = []
        run_expensive_eval = self._is_slow_update_round()
        for agent_id in self.controller.get_agent_ids():
            worker = self.runtime_agents[agent_id]
            incoming_src = list(self.controller.graph.get_neighbors(agent_id, direction="in"))
            state = self.controller.population.get_agent(agent_id)
            logger.info(
                "Fast round %s agent=%s incoming_sources=%s",
                self.round_idx,
                agent_id,
                incoming_src,
            )
            query = self._build_agent_query(agent_id, current_round_outputs=round_outputs)
            self.latest_queries[agent_id] = query
            self._hydrate_worker_with_incoming_artifacts(agent_id, worker, query_hint=query)
            try:
                result = worker.solve_task(query)
                output = str(result.findings or "")
                env_status = self._normalize_env_status(result.env_status)
            except Exception as e:
                logger.warning(
                    "Worker %s failed in round %s: %s — retrying once.",
                    agent_id, self.round_idx, e,
                )
                try:
                    result = worker.solve_task(query)
                    output = str(result.findings or "")
                    env_status = self._normalize_env_status(result.env_status)
                except Exception as e2:
                    logger.warning("Worker %s retry also failed: %s", agent_id, e2)
                    output = f"ERROR: {e2}"
                    env_status = DatasetEnvStatus(success=False, num_steps=0)
                    result = None

            # If the output looks procedural / low-signal, retry once with a more directive prompt.
            if self._is_low_signal_answer(output):
                logger.warning(
                    "Worker %s round %s returned low-signal output; retrying with directive prompt.",
                    agent_id, self.round_idx,
                )
                directive_query = (
                    query
                    + "\n\nIMPORTANT: Your previous response was incomplete or procedural. "
                    "Do NOT output planning statements, progress descriptions, or 'currently doing X' phrases. "
                    "Respond ONLY with concrete, factual findings: specific numbers, verified sources, "
                    "and direct answers. If data is unavailable, state the best estimate with confidence level."
                )
                try:
                    result = worker.solve_task(directive_query)
                    output = str(result.findings or "")
                    env_status = self._normalize_env_status(result.env_status)
                except Exception as e3:
                    logger.warning("Worker %s directive retry failed: %s", agent_id, e3)
                    # keep previous output/env_status

            sanitized_output = self._sanitize_agent_output_for_context(output, max_chars=1200)
            round_outputs[agent_id] = sanitized_output
            self.latest_artifacts[agent_id] = self._export_worker_artifacts(worker, query_hint=query)
            self.latest_env_status[agent_id] = env_status
            self.latest_tool_names[agent_id] = list(getattr(result, "tool_names", []) or [])
            if run_expensive_eval:
                contribution_score, contribution_reason = self._score_agent_contribution_llm(
                    agent_id=agent_id,
                    role=_canonicalize_role(state.role if state else "searcher"),
                    query=query,
                    output=sanitized_output,
                    tool_names=list(getattr(result, "tool_names", []) or []),
                    env_status=env_status,
                )
            else:
                contribution_score = self._score_text(sanitized_output, env_status)
                contribution_reason = "heuristic_only_non_slow_round"
            self._update_agent_state(
                agent_id,
                sanitized_output,
                env_status,
                query=query,
                local_score=contribution_score,
            )
            round_agent_details.append(
                {
                    "agent_id": agent_id,
                    "role": _canonicalize_role(state.role if state else "searcher"),
                    "incoming_sources": incoming_src,
                    "query_chars": len(query or ""),
                    "output_text": sanitized_output,
                    "output_chars": len(sanitized_output or ""),
                    "contribution_score": float(contribution_score),
                    "contribution_reason": contribution_reason,
                    "env_success": bool(getattr(env_status, "success", False)),
                    "env_steps": int(getattr(env_status, "num_steps", 0)),
                    "tool_names": list(getattr(result, "tool_names", []) or []),
                    "tool_calls": list(getattr(result, "tool_calls", []) or []),
                    "total_iterations": int(getattr(result, "total_iterations", 0) or 0),
                }
            )

        self.latest_outputs = round_outputs
        self.final_answer = self._synthesize_answer()
        if run_expensive_eval or self.round_idx >= self.max_fast_rounds:
            self.current_answer_quality, answer_eval_metrics = self._evaluate_final_answer_quality(self.final_answer)
            self.last_answer_eval_metrics = answer_eval_metrics
        else:
            heuristic_score = self._score_text(self.final_answer)
            self.current_answer_quality = heuristic_score
            answer_eval_metrics = {
                "score": heuristic_score,
                "reason": "heuristic_only_non_slow_round",
            }
        graph_after = self._snapshot_graph()
        # Keep the best answer across all rounds (by quality score).
        self._record_best_answer_if_improved(graph_snapshot=graph_after, source=f"fast_round_{self.round_idx}")
        self.fast_round_logs.append(
            {
                "round": self.round_idx,
                "answer_quality": self.current_answer_quality,
                "best_answer_quality": self.best_answer_quality,
                "answer_eval_metrics": answer_eval_metrics,
                "expensive_eval_round": run_expensive_eval,
                "final_answer_chars": len(self.final_answer or ""),
                "best_answer_chars": len(self.best_final_answer or ""),
                "coverage_status": self._to_jsonable(self._coverage_status()),
                "object_ecology": self._to_jsonable(self._object_ecology_summary()),
                "agents": round_agent_details,
                "graph_before": graph_before,
                "graph_after": graph_after,
                "graph_diff": self._diff_graph(graph_before, graph_after),
            }
        )
        logger.info(
            "=== Fast round %s end: quality=%.3f, final_answer_len=%s ===",
            self.round_idx,
            self.current_answer_quality,
            len(self.final_answer or ""),
        )
        return self.current_answer_quality

    def should_continue(self, meta_decision) -> bool:
        if self.current_answer_quality >= self.direct_stop_quality:
            self.stop_reason = f"direct_stop_quality={self.direct_stop_quality:.3f} reached"
            return False
        if self.round_idx >= self.max_fast_rounds:
            self.stop_reason = f"max_fast_rounds={self.max_fast_rounds} reached"
            return False

        if meta_decision is None:
            return True

        tc = meta_decision.time_control
        if not tc.continue_evolution:
            self.stop_reason = tc.stop_reason or "meta requested to stop evolution"
            return False

        if (
            not meta_decision.slow_update
            and not tc.trigger_birth_death
            and not tc.trigger_graph_rewire
            and self.current_answer_quality >= self.target_answer_quality
        ):
            self.stop_reason = "meta marked stable and answer quality reached target"
            return False

        return True

    def run_until_stable(self) -> Dict[str, Any]:
        """Run fast/slow loop until meta decides to stop or max rounds reached."""
        while True:
            answer_quality = self.run_fast_round()
            self.controller.step_fast_time()

            decision = None
            if self.controller.should_trigger_slow_update():
                if self._in_conservative_mode():
                    deferred = self._defer_slow_update_for_conservative_mode()
                    logger.info(
                        "Conservative mode active at round %s: deferring slow update by %s fast steps (best=%.3f current=%.3f)",
                        self.round_idx,
                        deferred,
                        self.best_answer_quality,
                        self.current_answer_quality,
                    )
                else:
                    graph_before_slow = self._snapshot_graph()
                    slow_ret = self.controller.execute_slow_update(
                        task_description=self._task_description_for_meta(),
                        current_answer_quality=answer_quality,
                        protected_graph_paths=self._protected_graph_paths(),
                    )
                    decision = slow_ret.get("decision") if isinstance(slow_ret, dict) else None
                    system_summary = slow_ret.get("system_summary") if isinstance(slow_ret, dict) else None
                    meta_llm_request = slow_ret.get("meta_llm_request") if isinstance(slow_ret, dict) else None
                    applied_fast_steps_next = int(slow_ret.get("applied_fast_steps_next", 0)) if isinstance(slow_ret, dict) else 0
                    self.last_meta_decision = decision
                    self._apply_agent_feedback(decision)
                    self._sync_runtime_agents()
                    self.final_answer = self._synthesize_answer(decision)
                    self.current_answer_quality, _ = self._evaluate_final_answer_quality(self.final_answer)
                    graph_after_slow = self._snapshot_graph()
                    self._record_best_answer_if_improved(
                        graph_snapshot=graph_after_slow,
                        source=f"slow_update_after_round_{self.round_idx}",
                    )
                    self.slow_update_logs.append(
                        {
                            "after_fast_round": self.round_idx,
                            "slow_update_step": self.controller.slow_update_step,
                            "decision_confidence": float(getattr(decision, "confidence", 0.0)) if decision else 0.0,
                            "trigger_birth_death": bool(getattr(getattr(decision, "time_control", None), "trigger_birth_death", False)) if decision else False,
                            "trigger_graph_rewire": bool(getattr(getattr(decision, "time_control", None), "trigger_graph_rewire", False)) if decision else False,
                            "fast_steps_next": applied_fast_steps_next if applied_fast_steps_next > 0 else (int(getattr(getattr(decision, "time_control", None), "fast_steps_next", 0)) if decision else 0),
                            "continue_evolution": bool(getattr(getattr(decision, "time_control", None), "continue_evolution", True)) if decision else True,
                            "stop_reason": str(getattr(getattr(decision, "time_control", None), "stop_reason", "")) if decision else "",
                            "global_rationale": list(getattr(decision, "global_rationale", []) or []) if decision else [],
                            "agent_feedback": [
                                {
                                    "agent_id": fb.agent_id,
                                    "score": fb.score,
                                    "score_reason": fb.score_reason,
                                    "workflow_correction": fb.workflow_correction,
                                    "next_round_workflow": fb.next_round_workflow,
                                }
                                for fb in (getattr(decision, "agent_feedback", []) or [])
                            ] if decision else [],
                            "birth_death_pairs": [
                                {
                                    "parent_id": p.parent_id,
                                    "death_target_id": p.death_target_id,
                                    "birth_reason": p.birth_reason,
                                    "death_reason": p.death_reason,
                                    "inherit_type": p.child_plan.inherit_type.value,
                                }
                                for p in (getattr(decision, "birth_death_pairs", []) or [])
                            ] if decision else [],
                            "graph_edit": {
                                "remove_edges": list(getattr(getattr(decision, "graph_edit", None), "remove_edges", []) or []),
                                "add_edges": list(getattr(getattr(decision, "graph_edit", None), "add_edges", []) or []),
                                "type_changes": list(getattr(getattr(decision, "graph_edit", None), "type_changes", []) or []),
                            } if decision else {"remove_edges": [], "add_edges": [], "type_changes": []},
                            "graph_before": graph_before_slow,
                            "graph_after": graph_after_slow,
                            "graph_diff": self._diff_graph(graph_before_slow, graph_after_slow),
                            "system_summary": self._to_jsonable(system_summary) if system_summary is not None else None,
                            "meta_llm_request": self._to_jsonable(meta_llm_request) if meta_llm_request is not None else None,
                            "conservative_mode": False,
                            "high_score_path_protection": self._to_jsonable(self._protected_graph_paths()),
                        }
                    )
            else:
                # Optional: meta feedback every fast round (no structural actions).
                # Controlled by env var because it can be expensive.
                feedback_every = os.getenv("META_FEEDBACK_EVERY_FAST_ROUND", "").strip().lower()
                if feedback_every in ("1", "true", "yes", "on"):
                    fb_ret = self.controller.execute_meta_feedback_only(
                        task_description=self._task_description_for_meta(),
                        current_answer_quality=answer_quality,
                    )
                    fb_decision = fb_ret.get("decision") if isinstance(fb_ret, dict) else None
                    if fb_decision is not None:
                        self._apply_agent_feedback(fb_decision)

            if not self.should_continue(decision):
                break

        # Use the best answer seen across all rounds, not just the last round.
        reported_answer = self.best_final_answer if self.best_final_answer.strip() else self.final_answer
        reported_quality = self.best_answer_quality if self.best_final_answer.strip() else self.current_answer_quality
        logger.info(
            "Run complete: best_quality=%.3f (round best), last_quality=%.3f, reporting best answer (len=%d)",
            self.best_answer_quality, self.current_answer_quality, len(reported_answer),
        )
        return {
            "final_answer": reported_answer,
            "best_final_answer": self.best_final_answer,
            "answer_quality": reported_quality,
            "last_round_answer": self.final_answer,
            "last_round_quality": self.current_answer_quality,
            "rounds": self.round_idx,
            "stop_reason": self.stop_reason,
            "agent_outputs": dict(self.latest_outputs),
            "meta_decision": self.last_meta_decision,
            "fast_round_logs": list(self.fast_round_logs),
            "slow_update_logs": list(self.slow_update_logs),
            "final_graph": self._snapshot_graph(),
            "coverage_schema": self._to_jsonable(self.coverage_schema),
            "coverage_memory": self._to_jsonable(self.coverage_memory),
            "source_feedback_memory": self._to_jsonable(self.source_feedback_memory),
            "task_profile": self._to_jsonable(self.task_profile),
            "task_schema": self._to_jsonable(self.task_schema),
            "object_ecology": self._to_jsonable(self._object_ecology_summary()),
            "information_objects": self._to_jsonable(self.information_objects[-40:]),
            "final_notes": self.final_notes,
            "high_score_path_protection": self._to_jsonable(self._protected_graph_paths()),
        }
