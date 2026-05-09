"""TacoMAS entry point.

Run a single instance or a range of instances of one of the four
benchmarks (finance-benchmark, browsecomp-plus, plancraft, workbench)
through the test-time co-evolution pipeline.

By default this script enforces SLURM execution. Set ALLOW_DIRECT_RUN=1
to run from a regular shell (intended for debugging only). For batch
jobs, prefer scripts/run_demo.slurm via sbatch.
"""


# 注：每个问题之间必须重置独立的演化状态（controller / graph / population / memory）。
import logging
import os
import time
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tacomas.meta_evolution.orchestration_adapter import (
    InitializationMetaLLM,
)

# ── key.env / keys.env（GT-GEO 风格）───────────────────────────────
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs):
        return False


def _find_project_env(start: Path) -> Path | None:
    current = start.resolve()
    search_roots = (current,) + tuple(current.parents)
    for parent in search_roots:
        for filename in ("key.env", "keys.env"):
            candidate = parent / filename
            if candidate.exists():
                return candidate
    return None


def _sync_api_aliases() -> None:
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    claude_key = os.getenv("CLAUDE_API_KEY", "").strip()
    if google_key and not gemini_key:
        os.environ["GEMINI_API_KEY"] = google_key
    if gemini_key and not google_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
    if claude_key and not anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = claude_key
    if anthropic_key and not claude_key:
        os.environ["CLAUDE_API_KEY"] = anthropic_key


def _load_project_env() -> None:
    env_file = _find_project_env(Path(__file__).resolve().parent)
    if env_file is not None:
        load_dotenv(env_file, override=False)
    _sync_api_aliases()


def _infer_provider_from_model(model_name: str) -> str:
    model = str(model_name or "").strip().lower()
    if not model:
        return ""
    if "/" in model:
        return model.split("/", 1)[0]
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return "openai"
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("deepseek"):
        return "deepseek"
    return ""


def _default_api_key_for_model(model_name: str) -> str:
    provider = _infer_provider_from_model(model_name)
    provider_key_map = {
        "openai": os.getenv("OPENAI_API_KEY", "").strip(),
        "gemini": os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip(),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", os.getenv("CLAUDE_API_KEY", "")).strip(),
        "deepseek": os.getenv("DEEPSEEK_API_KEY", "").strip(),
    }
    return provider_key_map.get(provider, "")


_load_project_env()

# ── 环境变量 ─────────────────────────────────────────────────────
os.environ.setdefault("TAVILY_API_KEY", "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")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-09bcca0e-4de4-4551-9c6c-0d23a88af2b6")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-d46e9aea-a861-441c-a6d5-771e142d37b4")
os.environ.setdefault("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# ── Gemini API Key ────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── DMX Meta LLM 非流式开关（仅用于 meta 决策层） ─────────────────
#USE_DMX_META_NONSTREAM = os.environ.get("USE_DMX_META_NONSTREAM", "0") == "1"
# 你可以直接把 key 写在这里，或者用环境变量 DMX_API_KEY 覆盖
#DMX_API_KEY = os.environ.get("DMX_API_KEY", "sk-dsCBJ5BgF6oXwmhQyNgUhSXlXWnLSeksJCQ6ijc3wW31F9Ni")
#DMX_MODEL = os.environ.get("DMX_MODEL", "gemini-2.5-flash")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_evolution_gemini")


def _instance_task_text(instance: Optional[Any]) -> str:
    """Extract a concrete task description from a dataset instance."""
    if instance is None:
        return "Analyze the task and answer it accurately using the available tools and evidence."
    try:
        prompt_info = instance.get_prompt_info()
        if isinstance(prompt_info, dict):
            question = str(prompt_info.get("question", "")).strip()
            if question:
                return question
            if prompt_info:
                return "\n".join(f"{k}: {v}" for k, v in prompt_info.items())
        text = str(prompt_info).strip()
        if text:
            return text
    except Exception:
        pass
    fallback = str(instance).strip()
    return fallback or "Analyze the task and answer it accurately using the available tools and evidence."


def _dataset_runtime_profile(dataset_id: str) -> dict[str, Any]:
    dataset_id = (dataset_id or "").strip()
    generic_template = "prompts/dataset-shared/generic_role_aware.yaml"
    profiles: dict[str, dict[str, Any]] = {
        "finance-benchmark": {
            "local_path": "datasets/finance-benchmark.json",
            "templates_local_path": "prompts/dataset-shared/finance_role_aware.yaml",
            "env_name": "finance",
            "tools": [
                "submit_final_result",
                "prepare_primary_filing",
                "web_search",
                "edgar_search",
                "parse_html_page",
                "retrieve_information",
            ],
        },
        "browsecomp-plus": {
            "local_path": "datasets/browsecomp_plus_converted.json",
            "templates_local_path": "prompts/dataset-shared/browsecomp_role_aware.yaml",
            "env_name": "browsecomp-plus",
            "tools": ["search_documents", "retrieve_document", "done"],
        },
        "plancraft": {
            "local_path": "datasets/plancraft_converted.json",
            "templates_local_path": "prompts/dataset-shared/plancraft_role_aware.yaml",
            "env_name": "plancraft-converted",
            "tools": ["search", "move", "smelt", "impossible"],
        },
        "workbench": {
            "local_path": "datasets/workbench_converted.json",
            "templates_local_path": "prompts/dataset-shared/workbench_role_aware.yaml",
            "env_name": "basic",
            "tools": ["done"],
        },
    }
    return profiles.get(
        dataset_id,
        {
            "local_path": f"datasets/{dataset_id}.json",
            "templates_local_path": generic_template,
            "env_name": "basic",
            "tools": ["done"],
        },
    )


def _derive_role_permissions_for_tools(dataset_tools: list[str]) -> dict[str, list[str]]:
    tool_set = {tool.strip() for tool in dataset_tools if tool and tool.strip()}

    def keep(*tools: str) -> list[str]:
        return [tool for tool in tools if tool in tool_set]

    search_acquire = keep(
        "prepare_primary_filing",
        "web_search",
        "edgar_search",
        "parse_html_page",
        "search_documents",
        "retrieve_document",
        "retrieve_information",
    )
    action_lookup = keep("search")
    action_exec = keep("move", "smelt", "impossible")
    read_only = keep("retrieve_information", "retrieve_document")
    finisher = keep("submit_final_result", "done")
    search_capable = list(dict.fromkeys(search_acquire + action_lookup + action_exec))
    verifier_tools = list(dict.fromkeys(read_only + action_lookup))
    summary_tools = list(dict.fromkeys(read_only + finisher))

    return {
        "planner": [],
        "searcher": list(search_capable),
        "researcher": list(search_capable),
        "analyst": list(search_capable),
        "curator": list(search_capable),
        "verifier": list(verifier_tools),
        "schema_verifier": list(verifier_tools),
        "auditor": list(verifier_tools),
        "critic": list(verifier_tools),
        "calculator": list(dict.fromkeys(read_only + action_lookup + action_exec)),
        "forecaster": list(dict.fromkeys(read_only + action_lookup + action_exec)),
        "reflector": [],
        "summary": list(summary_tools),
        "synthesizer": list(summary_tools),
        "worker": list(search_capable),
        "subagent": list(search_capable),
    }


def _export_readable_instance_views(
    output_dir: Path,
    instance_idx: int,
    runtime_result: dict[str, Any],
) -> None:
    """Export compact per-role/per-agent views to simplify trace inspection."""
    fast_round_logs = runtime_result.get("fast_round_logs", []) or []
    slow_update_logs = runtime_result.get("slow_update_logs", []) or []

    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_agent: dict[str, dict[str, Any]] = {}

    for round_log in fast_round_logs:
        round_id = int(round_log.get("round", 0))
        for agent_log in round_log.get("agents", []) or []:
            agent_id = str(agent_log.get("agent_id", ""))
            role = str(agent_log.get("role", "unknown"))
            step_record = {
                "round": round_id,
                "agent_id": agent_id,
                "incoming_sources": agent_log.get("incoming_sources", []),
                "output_text": agent_log.get("output_text", ""),
                "tool_calls": agent_log.get("tool_calls", []),
                "contribution_score": agent_log.get("contribution_score", None),
                "contribution_reason": agent_log.get("contribution_reason", ""),
                "query_chars": agent_log.get("query_chars", 0),
                "output_chars": agent_log.get("output_chars", 0),
            }

            by_role[role].append(step_record)
            by_agent.setdefault(agent_id, {"role": role, "steps": []})
            by_agent[agent_id]["steps"].append(step_record)

    meta_requests = []
    for slow_log in slow_update_logs:
        req = slow_log.get("meta_llm_request", {}) or {}
        decision_time_control = {
            "decision_confidence": slow_log.get("decision_confidence", 0.0),
            "trigger_birth_death": slow_log.get("trigger_birth_death", False),
            "trigger_graph_rewire": slow_log.get("trigger_graph_rewire", False),
            "fast_steps_next": slow_log.get("fast_steps_next", 0),
            "continue_evolution": slow_log.get("continue_evolution", True),
            "stop_reason": slow_log.get("stop_reason", ""),
        }
        meta_requests.append(
            {
                "after_fast_round": slow_log.get("after_fast_round", 0),
                "slow_update_step": slow_log.get("slow_update_step", 0),
                "system_prompt": req.get("system_prompt", ""),
                "developer_prompt": req.get("developer_prompt", ""),
                "user_payload": req.get("user_payload", ""),
                "global_rationale": slow_log.get("global_rationale", []),
                "time_control": decision_time_control,
                "birth_death_pairs": slow_log.get("birth_death_pairs", []),
                "graph_edit": slow_log.get("graph_edit", {}),
                "graph_diff": slow_log.get("graph_diff", {}),
                "agent_feedback": slow_log.get("agent_feedback", []),
            }
        )

    with open(output_dir / f"instance_{instance_idx}_by_role.json", "w", encoding="utf-8") as f:
        json.dump(by_role, f, ensure_ascii=False, indent=2)
    with open(output_dir / f"instance_{instance_idx}_by_agent.json", "w", encoding="utf-8") as f:
        json.dump(by_agent, f, ensure_ascii=False, indent=2)
    with open(output_dir / f"instance_{instance_idx}_meta_requests.json", "w", encoding="utf-8") as f:
        json.dump(meta_requests, f, ensure_ascii=False, indent=2)

    # Compact per-round score summary — easy to read without parsing the full instance JSON.
    round_scores = []
    for round_log in fast_round_logs:
        round_entry = {
            "round": round_log.get("round", 0),
            "answer_quality": round_log.get("answer_quality"),
            "best_answer_quality": round_log.get("best_answer_quality"),
            "final_answer_chars": round_log.get("final_answer_chars", 0),
            "best_answer_chars": round_log.get("best_answer_chars", 0),
            "expensive_eval": round_log.get("expensive_eval_round", False),
            "agents": [
                {
                    "agent_id": ag.get("agent_id", ""),
                    "role": ag.get("role", ""),
                    "contribution_score": ag.get("contribution_score"),
                    "contribution_reason": ag.get("contribution_reason", ""),
                    "tools": ag.get("tool_names", []),
                    "output_chars": ag.get("output_chars", 0),
                    "output_preview": (ag.get("output_text", "") or "")[:300],
                }
                for ag in (round_log.get("agents", []) or [])
            ],
        }
        round_scores.append(round_entry)
    with open(output_dir / f"instance_{instance_idx}_round_scores.json", "w", encoding="utf-8") as f:
        json.dump(round_scores, f, ensure_ascii=False, indent=2)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _instance_field(instance: Optional[Any], field_name: str, default: Any = "") -> Any:
    if instance is None:
        return default
    value = getattr(instance, field_name, default)
    return default if value is None else value


def _instance_ground_truth(instance: Optional[Any]) -> str:
    return str(_instance_field(instance, "expected_answer", "") or "").strip()


def _instance_rubric(instance: Optional[Any]) -> Any:
    return _instance_field(instance, "rubric", "")


def _build_evaluation_record(
    instance_idx: int,
    instance: Optional[Any],
    final_answer: str,
    metrics: dict[str, Any],
    runtime_result: dict[str, Any],
) -> dict[str, Any]:
    judge_score = metrics.get(
        "score",
        metrics.get("exact_match", metrics.get("avg_score", None)),
    )
    judge_score = float(judge_score) if judge_score is not None else None
    return {
        "instance_idx": instance_idx,
        "question": _instance_task_text(instance),
        "prediction": final_answer,
        "ground_truth": _instance_ground_truth(instance),
        "rubric": _instance_rubric(instance),
        "judge_similarity_score": judge_score,
        "judge_success": metrics.get("success"),
        "judge_reasoning": metrics.get("reasoning", ""),
        "runtime_answer_quality": float(runtime_result.get("answer_quality", 0.0) or 0.0),
        "rounds": int(runtime_result.get("rounds", 0) or 0),
        "stop_reason": runtime_result.get("stop_reason", ""),
    }


# ════════════════════════════════════════════════════
# 1. 构建 EvolutionController
# ════════════════════════════════════════════════════

'''
def build_evolution_controller():
    """构建演化控制器，meta LLM 使用 Gemini。"""
    from tacomas.config.llm import LLMConfig
    from tacomas.config.meta_evolution import MetaEvolutionConfig
    from tacomas.meta_evolution.evolution_controller import EvolutionController

    meta_llm_config = LLMConfig(
        model="gemini/gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.3,  # meta 决策用低温度，JSON 输出更稳定
    )

    evol_config = MetaEvolutionConfig(
        enabled=True,
        meta_llm_model=meta_llm_config.model,
        meta_llm_api_key=GEMINI_API_KEY,
        meta_llm_temperature=0.3,

        # 种群大小
        n_min=3,
        n_max=8,

        # 时间尺度（小实验调小，快速看到演化效果）
    fast_steps_per_window=1,
    bd_check_interval=1,      # 每 1 次迭代检查一次 birth-death
        graph_rewire_interval=20, # 每 20 次迭代检查一次图重连

        # 演化约束
        max_birth_death_pairs=2,
        max_edge_edits=4,
        specialization_level="medium",

        # 保护关键节点（lead agent 不能被杀掉）
        protected_nodes=["lead_agent"],
        critical_roles=["planner", "searcher", "verifier"],

        preserve_connectivity=True,
        max_degree=6,

        log_decisions=True,
        log_dir="./meta_evolution_logs",
    )

    return EvolutionController(evol_config, meta_llm_config)
'''

def build_evolution_controller_with_config(
    meta_model: str,
    meta_api_key: str,
    meta_api_base: str = "",
    meta_temperature: float = 0.3,
    n_min: int = 3,
    n_max: int = 8,
    max_birth_death_pairs: int = 2,
    max_edge_edits: int = 8,
    fast_steps_per_window: int = 1,
    bd_check_interval: int = 1,
    graph_rewire_interval: int = 20,
    critical_roles: Optional[list[str]] = None,
):
    """构建演化控制器，可外部传入 meta LLM 与演化约束参数。"""
    from tacomas.config.llm import LLMConfig
    from tacomas.config.meta_evolution import MetaEvolutionConfig
    from tacomas.meta_evolution.evolution_controller import EvolutionController

    meta_llm_config = LLMConfig(
        model=meta_model,
        api_key=meta_api_key,
        api_base=meta_api_base or None,
        temperature=meta_temperature,
    )

    evol_config = MetaEvolutionConfig(
        enabled=True,
        meta_llm_model=meta_llm_config.model,
        meta_llm_api_key=meta_api_key,
        meta_llm_api_base=(meta_api_base or None),
        meta_llm_temperature=meta_temperature,
        n_min=n_min,
        n_max=n_max,
        fast_steps_per_window=fast_steps_per_window,
        bd_check_interval=bd_check_interval,
        graph_rewire_interval=graph_rewire_interval,
        max_birth_death_pairs=max_birth_death_pairs,
        max_edge_edits=max_edge_edits,
        specialization_level="medium",
        # Agent ids are dynamically generated by meta initialization, so protect by role coverage instead.
        protected_nodes=[],
        critical_roles=critical_roles or ["searcher"],
        preserve_connectivity=True,
        max_degree=6,
        log_decisions=True,
        log_dir="./meta_evolution_logs",
    )

    return EvolutionController(evol_config, meta_llm_config)


# ════════════════════════════════════════════════════
# 2. 初始化种群和图结构（meta LLM 自动初始化）
# ════════════════════════════════════════════════════

def initialize_population_with_meta_llm(
    controller,
    task_description: str,
    dataset_id: str = "",
    skip_llm_init: bool = False,
    fixed_topology_file: str = "",
    init_n_min: int = 0,
    init_n_max: int = 0,
):
    """由初始化版 meta LLM 自动决定初始 agents + roles + graph。

    init_n_min / init_n_max: desired initial population size range.
    When 0, falls back to controller.control_params.n_min / n_max
    (which are the *evolution* bounds, potentially wider).
    """
    # Resolve effective init range (separate from evolution bounds).
    eff_init_min = init_n_min if init_n_min > 0 else controller.control_params.n_min
    eff_init_max = init_n_max if init_n_max > 0 else controller.control_params.n_max

    def _dataset_default_init_result(curr_dataset_id: str, n_agents: int) -> dict[str, Any]:
        did = (curr_dataset_id or "").strip()
        n = max(int(n_agents), 4)
        if did == "workbench":
            agents = [
                {"id": "agent_0", "role": "planner", "init_prompt": "Decompose the workplace request into the minimal executable action needed to satisfy it."},
                {"id": "agent_1", "role": "calculator", "init_prompt": "Compile the required parameters into the most direct executable tool invocation."},
                {"id": "agent_2", "role": "verifier", "init_prompt": "Check whether the proposed action or invocation matches the request constraints exactly."},
                {"id": "agent_3", "role": "summary", "init_prompt": "Output the final answer as the cleanest executable invocation or shortest valid action sequence."},
                {"id": "agent_4", "role": "reflector", "init_prompt": "Only provide brief corrective hints if the action is malformed or incomplete."},
            ][:n]
            edges = [
                {"source": "agent_0", "target": "agent_1", "type": "directed"},
                {"source": "agent_0", "target": "agent_2", "type": "directed"},
                {"source": "agent_1", "target": "agent_2", "type": "bidirectional"},
                {"source": "agent_1", "target": "agent_3", "type": "computation_flow"},
                {"source": "agent_2", "target": "agent_3", "type": "verification_flow"},
                {"source": "agent_4", "target": "agent_3", "type": "reflection_feedback"},
            ]
            return {
                "total_agents": len(agents),
                "agents": agents,
                "edges": [e for e in edges if e["source"] in {a["id"] for a in agents} and e["target"] in {a["id"] for a in agents}],
                "rationale": ["Dataset-aware default init for action-first workplace tasks."],
            }
        if did == "plancraft":
            agents = [
                {"id": "agent_0", "role": "planner", "init_prompt": "Translate the crafting goal into a minimal executable action plan or an IMPOSSIBLE judgment."},
                {"id": "agent_1", "role": "searcher", "init_prompt": "Use recipe lookup and environment actions to discover the exact crafting dependencies needed."},
                {"id": "agent_2", "role": "calculator", "init_prompt": "Reason about counts, intermediate items, and whether the action sequence is feasible."},
                {"id": "agent_3", "role": "verifier", "init_prompt": "Check that the candidate crafting sequence or IMPOSSIBLE claim is consistent with recipes and inventory."},
                {"id": "agent_4", "role": "summary", "init_prompt": "Return a short executable action sequence or exactly IMPOSSIBLE."},
            ][:n]
            edges = [
                {"source": "agent_0", "target": "agent_1", "type": "directed"},
                {"source": "agent_0", "target": "agent_2", "type": "directed"},
                {"source": "agent_1", "target": "agent_2", "type": "evidence_flow"},
                {"source": "agent_1", "target": "agent_3", "type": "verification_flow"},
                {"source": "agent_2", "target": "agent_4", "type": "computation_flow"},
                {"source": "agent_3", "target": "agent_4", "type": "verification_flow"},
            ]
            return {
                "total_agents": len(agents),
                "agents": agents,
                "edges": [e for e in edges if e["source"] in {a["id"] for a in agents} and e["target"] in {a["id"] for a in agents}],
                "rationale": ["Dataset-aware default init for crafting/action tasks."],
            }
        if did == "browsecomp-plus":
            agents = [
                {"id": "agent_0", "role": "planner", "init_prompt": "Break the question into concrete retrieval targets and answer constraints."},
                {"id": "agent_1", "role": "searcher", "init_prompt": "Retrieve high-signal passages that directly constrain the answer."},
                {"id": "agent_2", "role": "researcher", "init_prompt": "Cross-check candidate passages and chase the most specific remaining clue."},
                {"id": "agent_3", "role": "verifier", "init_prompt": "Reject unsupported candidate answers and keep only source-grounded claims."},
                {"id": "agent_4", "role": "summary", "init_prompt": "Return the shortest exact answer span supported by the retrieved passages."},
            ][:n]
            edges = [
                {"source": "agent_0", "target": "agent_1", "type": "directed"},
                {"source": "agent_0", "target": "agent_2", "type": "directed"},
                {"source": "agent_1", "target": "agent_3", "type": "verification_flow"},
                {"source": "agent_2", "target": "agent_3", "type": "verification_flow"},
                {"source": "agent_1", "target": "agent_4", "type": "evidence_flow"},
                {"source": "agent_3", "target": "agent_4", "type": "verification_flow"},
            ]
            return {
                "total_agents": len(agents),
                "agents": agents,
                "edges": [e for e in edges if e["source"] in {a["id"] for a in agents} and e["target"] in {a["id"] for a in agents}],
                "rationale": ["Dataset-aware default init for retrieval-heavy QA tasks."],
            }
        return {}

    if fixed_topology_file:
        logger.info("Loading fixed topology from %s", fixed_topology_file)
        init = InitializationMetaLLM(controller.llm_config)
        with open(fixed_topology_file, "r", encoding="utf-8") as f:
            init_result = json.load(f)
        init._register(controller, init_result)
    elif skip_llm_init:
        logger.warning("Skip meta LLM initialization (network-sensitive mode), using default topology.")
        init = InitializationMetaLLM(controller.llm_config)
        n_agents = (
            eff_init_max
            if os.getenv("UNREASONABLE_INIT", "0") == "1"
            else eff_init_min
        )
        init_result = _dataset_default_init_result(dataset_id, n_agents)
        if not init_result:
            init_result = init._default_init(n_agents)
        init._register(controller, init_result)
    else:
        init = InitializationMetaLLM(controller.llm_config)
        init_result = init.initialize(
            controller=controller,
            task_description=task_description,
            n_min=eff_init_min,
            n_max=eff_init_max,
        )
    state = controller.get_system_state()
    logger.info(
        f"Meta initialization done: {state['num_agents']} agents, "
        f"{state['num_edges']} edges, roles={state['role_distribution']}"
    )
    return init_result


# ════════════════════════════════════════════════════
# 3. 主实验循环
# ════════════════════════════════════════════════════

def run_evolution_experiment(
    start_idx: int = 38,
    end_idx: int = 40,
    skip_meta_init: bool = False,
    agent_name: str = "multi-agent-independent",
    n_base_agents: int = 1,
    min_iterations_per_agent: int = 1,
    max_iterations_per_agent: int = 2,
    agent_model_override: str = "",
    agent_api_base: str = "",
    agent_api_key: str = "",
    agent_temperature: float = 0.0,
    meta_model: str = "gemini/gemini-2.5-pro",
    meta_api_base: str = "",
    meta_api_key: str = "",
    meta_temperature: float = 0.3,
    relax_constraints: bool = False,
    max_birth_death_pairs: int = 2,
    max_edge_edits: int = 4,
    init_n_min: int = 3,
    init_n_max: int = 8,
    pop_n_min: int = 0,
    pop_n_max: int = 0,
    fast_steps_per_window: int = 1,
    bd_check_interval: int = 1,
    graph_rewire_interval: int = 20,
    fixed_topology_file: str = "",
):
    """
    主循环：基于 Meta 初始化得到的图结构动态创建 BaseWorkerAgent 集合，
    按 directed/bidirectional 边进行定向消息传递，
    并由 meta LLM 决定是否继续 fast rounds、是否触发 birth-death/图重连，
    直到 meta 认为结构稳定且答案足够。

    init_n_min/init_n_max: initial topology size (how many agents to spawn at start).
    pop_n_min/pop_n_max:   evolution population bounds (hard constraints during BD).
                           Defaults to init_n_min/init_n_max when not specified.
    """
    if end_idx <= start_idx:
        logger.warning(f"Empty range: start={start_idx}, end={end_idx}. Nothing to run.")
        return
    if init_n_min <= 0 or init_n_max <= 0:
        raise ValueError(f"init_n_min/init_n_max must be > 0, got {init_n_min}/{init_n_max}")
    if init_n_min > init_n_max:
        raise ValueError(f"init_n_min must be <= init_n_max, got {init_n_min}>{init_n_max}")

    # Evolution bounds fall back to init range when not explicitly set.
    eff_pop_n_min = pop_n_min if pop_n_min > 0 else init_n_min
    eff_pop_n_max = pop_n_max if pop_n_max > 0 else init_n_max

    from dotenv import load_dotenv
    load_dotenv(override=True)

    import tacomas.datasets.finance  # noqa: F401 — 注册 finance 数据集
    import tacomas.env.finance       # noqa: F401 — 注册 finance 环境
    from tacomas.config.llm import LLMConfig
    from tacomas.config.run import RunConfig
    from tacomas.datasets.base import DatasetEnvStatus, DatasetInstanceOutputWithTrajectory
    from tacomas.meta_evolution.mas_runtime import MetaEvolutionMASRuntime

    # Respect env var already set by SLURM/caller; only override if not already configured.
    if "META_RELAX_CONSTRAINTS" not in os.environ:
        os.environ["META_RELAX_CONSTRAINTS"] = "1" if relax_constraints else "0"

    # 演化控制器 / 调用后端开关
    # --use_dmx_meta 打开时，meta + agent 全部走 DMX 非流式
    '''
    if USE_DMX_META_NONSTREAM:
        os.environ["USE_DMX_META_NONSTREAM"] = "1"
        os.environ["USE_DMX_ALL_NONSTREAM"] = "1"
        os.environ["DMX_MODEL"] = DMX_MODEL
        os.environ.setdefault("DMX_TIMEOUT", "60")
        os.environ.setdefault("DMX_HTTP_RETRIES", "5")
        os.environ.setdefault("DMX_HTTP_BACKOFF", "1.5")
        if DMX_API_KEY:
            os.environ["DMX_API_KEY"] = DMX_API_KEY
        logger.info(f"LLM backend: DMX non-stream (meta+agent, model={DMX_MODEL})")
        agent_model = f"gemini/{DMX_MODEL}"
    else:
        os.environ.pop("USE_DMX_META_NONSTREAM", None)
        os.environ.pop("USE_DMX_ALL_NONSTREAM", None)
        logger.info("LLM backend: default litellm backend")
        agent_model = agent_model_override or "gemini/gemini-2.0-flash"
    '''
    logger.info("LLM backend: default litellm backend")
    agent_model = agent_model_override or "gemini/gemini-2.5-flash-lite"

    final_agent_api_key = agent_api_key or _default_api_key_for_model(agent_model)
    final_meta_api_key = meta_api_key or _default_api_key_for_model(meta_model)

    # 底层 agents 的 LLM
    agent_llm_config = LLMConfig(
        model=agent_model,
        api_key=final_agent_api_key or None,
        api_base=agent_api_base or None,
        temperature=agent_temperature,
    )

    dataset_id = os.environ.get("DATASET_ID", "finance-benchmark").strip()
    dataset_profile = _dataset_runtime_profile(dataset_id)
    dataset_local_path = os.environ.get("DATASET_LOCAL_PATH", dataset_profile["local_path"]).strip()
    dataset_templates_path = os.environ.get("DATASET_TEMPLATE_PATH", dataset_profile["templates_local_path"]).strip()
    if dataset_templates_path and not Path(dataset_templates_path).exists():
        logger.warning(
            "DATASET_TEMPLATE_PATH=%s not found; falling back to dataset profile template %s",
            dataset_templates_path,
            dataset_profile["templates_local_path"],
        )
        dataset_templates_path = dataset_profile["templates_local_path"]
    dataset_env_name = os.environ.get("DATASET_ENV_NAME", dataset_profile["env_name"]).strip()
    dataset_tools_raw = os.environ.get("DATASET_TOOLS", ",".join(dataset_profile["tools"]))
    dataset_tools = [tool.strip() for tool in dataset_tools_raw.split(",") if tool.strip()]
    runtime_role_permissions = _derive_role_permissions_for_tools(dataset_tools)

    if dataset_id == "workbench":
        critical_roles = ["planner", "calculator", "verifier", "reflector"]
    elif dataset_id == "plancraft":
        critical_roles = ["planner", "calculator", "verifier"]
    elif dataset_id == "finance-benchmark":
        critical_roles = ["planner", "searcher", "verifier", "reflector"]
    elif dataset_id == "browsecomp-plus":
        critical_roles = ["planner", "searcher", "verifier"]
    else:
        critical_roles = ["searcher"]

    run_cfg = RunConfig(**{
        "agent": {
            "name": agent_name,
            "prompts": {
                "base_agent": {"local_path": "prompts/multi-agent/independent_worker.yaml"},
                "aggregator": {"local_path": "prompts/multi-agent/independent_worker.yaml"},
            },
            "agent_specific_config": {
                "n_base_agents": n_base_agents,
                "min_iterations_per_agent": min_iterations_per_agent,
                "max_iterations_per_agent": max_iterations_per_agent,
                "aggregation_method": "llm_vote",
            },
        },
        "dataset": {
            "dataset_id": dataset_id,
            "from_langfuse": False,
            "local_path": dataset_local_path,
            "templates_local_path": dataset_templates_path,
            "env": {
                "name": dataset_env_name,
                "tools": dataset_tools,
            },
        },
        "llm": agent_llm_config.model_dump(),
        "run_name": "evolution_gemini_dynamic_mas",
        "save_dir": "./outputs/evolution_gemini",
        "log_langfuse": False,
        "num_workers": 1,
    })
    dataset = run_cfg.dataset.dataset

    logger.info("=" * 60)
    logger.info("Evolution experiment started")
    logger.info(
        "Runtime cfg: agent=%s template_n_base_agents=%s min_iter=%s max_iter=%s relax_constraints=%s",
        agent_name,
        n_base_agents,
        min_iterations_per_agent,
        max_iterations_per_agent,
        relax_constraints,
    )
    logger.info("Meta init agent range: [%s, %s]", init_n_min, init_n_max)


    '''
    if USE_DMX_META_NONSTREAM:
        logger.info(f"Meta LLM : DMX non-stream ({DMX_MODEL})")
        logger.info(f"Agent LLM: DMX non-stream ({DMX_MODEL})")
    else:
        logger.info("Meta LLM : %s", meta_model)
        logger.info("Agent LLM: %s", agent_model)
    '''

    
    logger.info("Meta LLM : %s", meta_model)
    logger.info("Agent LLM: %s", agent_model)
    logger.info(f"Dataset range: [{start_idx}, {end_idx})")
    logger.info("=" * 60)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path("outputs")
    dataset_slug = re.sub(r"[^A-Za-z0-9]+", "-", dataset_id).strip("-") or "dataset"
    run_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", os.environ.get("RUN_TAG", "").strip()).strip("-")
    if run_tag:
        output_dir = output_root / f"evolution_trace_{run_ts}_{run_tag}_{dataset_slug}_{start_idx}_{end_idx}"
    else:
        output_dir = output_root / f"evolution_trace_{run_ts}_{dataset_slug}_{start_idx}_{end_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []

    run_hparams = {
        "timestamp": run_ts,
        "run_tag": run_tag,
        "dataset_range": {"start_idx": start_idx, "end_idx": end_idx},
        "per_instance_reset": True,
        "runtime": {
            "agent_name": agent_name,
            "n_base_agents": n_base_agents,
            "min_iterations_per_agent": min_iterations_per_agent,
            "max_iterations_per_agent": max_iterations_per_agent,
            "relax_constraints": relax_constraints,
            "max_fast_rounds_per_instance": int(os.environ.get("MAX_FAST_ROUNDS_PER_INSTANCE", "6")),
            "target_answer_quality": float(os.environ.get("TARGET_ANSWER_QUALITY", "0.82")),
        },
        "agent_llm": {
            "model": agent_model,
            "api_base": agent_api_base,
            "temperature": agent_temperature,
        },
        "meta_llm": {
            "model": meta_model,
            "api_base": meta_api_base,
            "temperature": meta_temperature,
        },
        "meta_init_population_range": {"n_min": init_n_min, "n_max": init_n_max},
        "evolution_population_range": {"n_min": eff_pop_n_min, "n_max": eff_pop_n_max},
        "evolution_constraints": {
            "max_birth_death_pairs": max_birth_death_pairs,
            "max_edge_edits": max_edge_edits,
        },
        "fixed_topology_file": fixed_topology_file,
    }
    _write_json(output_dir / "run_hparams.json", run_hparams)

    all_scores = []
    all_raw_scores = []
    failed_instances = 0
    stopped_by_quota = False

    for instance_idx in range(start_idx, end_idx):
        logger.info(f"\n{'─'*50}")
        logger.info(f"Instance {instance_idx}  |  building fresh controller/runtime state")
        if instance_idx >= len(dataset.instances):
            logger.warning(f"Instance index {instance_idx} out of range, stop.")
            break

        instance = dataset.instances[instance_idx]

        max_instance_retries = int(os.environ.get("INSTANCE_RETRIES", "2"))
        metrics = {}
        runtime_result = {}
        for attempt in range(max_instance_retries):
            try:
                controller = build_evolution_controller_with_config(
                    meta_model=meta_model,
                    meta_api_key=final_meta_api_key,
                    meta_api_base=meta_api_base,
                    meta_temperature=meta_temperature,
                    n_min=eff_pop_n_min,
                    n_max=eff_pop_n_max,
                    max_birth_death_pairs=max_birth_death_pairs,
                    max_edge_edits=max_edge_edits,
                    fast_steps_per_window=fast_steps_per_window,
                    bd_check_interval=bd_check_interval,
                    graph_rewire_interval=graph_rewire_interval,
                    critical_roles=critical_roles,
                )
                controller.set_logging_context(
                    run_id=output_dir.name,
                    dataset_id=dataset_id,
                    instance_idx=instance_idx,
                )
                initialize_population_with_meta_llm(
                    controller,
                    task_description=_instance_task_text(instance),
                    dataset_id=dataset_id,
                    skip_llm_init=skip_meta_init,
                    fixed_topology_file=fixed_topology_file,
                    init_n_min=init_n_min,
                    init_n_max=init_n_max,
                )
                logger.info(
                    "Instance %s fresh controller state: %s",
                    instance_idx,
                    controller.get_system_state(),
                )
                template_agent = run_cfg.get_agent()
                runtime = MetaEvolutionMASRuntime(
                    controller=controller,
                    template_agent=template_agent,
                    task_instance=instance,
                    task_description=_instance_task_text(instance),
                    min_iterations_per_agent=min_iterations_per_agent,
                    max_iterations_per_agent=max_iterations_per_agent,
                    max_fast_rounds=int(os.environ.get("MAX_FAST_ROUNDS_PER_INSTANCE", "6")),
                    target_answer_quality=float(os.environ.get("TARGET_ANSWER_QUALITY", "0.82")),
                    role_permissions=runtime_role_permissions,
                )
                runtime_result = runtime.run_until_stable()

                final_answer = str(runtime_result.get("final_answer", "") or "")
                logger.info(
                    "Instance %s final answer for eval (head): %s",
                    instance_idx,
                    final_answer[:300].replace("\n", " "),
                )
                output = DatasetInstanceOutputWithTrajectory(
                    data_instance=instance,
                    agent_output=final_answer,
                    trajectory=[],
                    final_env_output=DatasetEnvStatus(
                        success=bool(final_answer.strip()),
                        num_steps=int(runtime_result.get("rounds", 0)),
                    ),
                )
                metrics = dataset.get_instance_eval_metrics(output)
                logger.info(f"Instance {instance_idx} metrics: {metrics}")
                logger.info(
                    "Instance %s runtime: rounds=%s quality=%.3f stop=%s",
                    instance_idx,
                    runtime_result.get("rounds", 0),
                    float(runtime_result.get("answer_quality", 0.0)),
                    runtime_result.get("stop_reason", ""),
                )

                instance_trace = {
                    "instance_idx": instance_idx,
                    "question": _instance_task_text(instance),
                    "metrics": metrics,
                    "runtime_result": {
                        "rounds": runtime_result.get("rounds", 0),
                        "answer_quality": float(runtime_result.get("answer_quality", 0.0)),
                        "stop_reason": runtime_result.get("stop_reason", ""),
                        "final_answer": final_answer,
                        "ground_truth": _instance_ground_truth(instance),
                        "judge_similarity_score": metrics.get("score", metrics.get("exact_match", metrics.get("avg_score", None))),
                        "final_answer_head": final_answer[:500],
                        "agent_outputs": runtime_result.get("agent_outputs", {}),
                        "fast_round_logs": runtime_result.get("fast_round_logs", []),
                        "slow_update_logs": runtime_result.get("slow_update_logs", []),
                        "final_graph": runtime_result.get("final_graph", {}),
                        "coverage_schema": runtime_result.get("coverage_schema", {}),
                        "coverage_memory": runtime_result.get("coverage_memory", {}),
                        "task_profile": runtime_result.get("task_profile", {}),
                        "task_schema": runtime_result.get("task_schema", {}),
                        "final_notes": runtime_result.get("final_notes", ""),
                    },
                }
                instance_records.append(instance_trace)
                evaluation_records.append(
                    _build_evaluation_record(
                        instance_idx=instance_idx,
                        instance=instance,
                        final_answer=final_answer,
                        metrics=metrics,
                        runtime_result=runtime_result,
                    )
                )
                _write_json(output_dir / f"instance_{instance_idx}.json", instance_trace)
                _export_readable_instance_views(
                    output_dir=output_dir,
                    instance_idx=instance_idx,
                    runtime_result=runtime_result,
                )
                break
            except Exception as e:
                msg = str(e)
                is_network_error = (
                    "ConnectionResetError" in msg
                    or "Connection aborted" in msg
                    or "requests.exceptions.ConnectionError" in msg
                    or "WinError 10054" in msg
                )
                last_try = attempt >= max_instance_retries - 1
                if is_network_error and not last_try:
                    backoff = 2 ** attempt
                    logger.warning(
                        f"Instance {instance_idx} network error, retrying in {backoff}s "
                        f"({attempt + 1}/{max_instance_retries}): {msg[:200]}"
                    )
                    time.sleep(backoff)
                    continue

                logger.exception(f"Instance {instance_idx} failed with exception")
                failed_instances += 1
                if "insufficient_user_quota" in msg or "额度不足" in msg:
                    logger.error("Quota insufficient detected, stopping remaining instances early.")
                    stopped_by_quota = True
                metrics = {}
                instance_trace = {
                    "instance_idx": instance_idx,
                    "question": _instance_task_text(instance),
                    "metrics": {},
                    "runtime_result": {},
                    "error": msg,
                }
                instance_records.append(instance_trace)
                evaluation_records.append(
                    {
                        "instance_idx": instance_idx,
                        "question": _instance_task_text(instance),
                        "prediction": "",
                        "ground_truth": _instance_ground_truth(instance),
                        "rubric": _instance_rubric(instance),
                        "judge_similarity_score": 0.0,
                        "judge_success": 0.0,
                        "judge_reasoning": f"runtime_error: {msg}",
                        "runtime_answer_quality": 0.0,
                        "rounds": 0,
                        "stop_reason": "runtime_error",
                        "error": msg,
                    }
                )
                _write_json(output_dir / f"instance_{instance_idx}.json", instance_trace)
                break

        if stopped_by_quota and not metrics:
            break

        raw_score = metrics.get(
            "score",
            metrics.get("exact_match", metrics.get("avg_score", None)),
        )
        raw_score = float(raw_score) if raw_score is not None else None
        if raw_score is not None:
            all_raw_scores.append(raw_score)

        answer_quality = float(runtime_result.get("answer_quality", raw_score or 0.0))
        all_scores.append(answer_quality)

        # 默认按免费额度做轻微限速，可通过 PER_INSTANCE_SLEEP_SECONDS 调整
        per_instance_sleep = float(os.environ.get("PER_INSTANCE_SLEEP_SECONDS", "2"))
        if per_instance_sleep > 0:
            time.sleep(per_instance_sleep)

    # ── 最终统计 ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Experiment finished")
    logger.info(f"Instances run   : {end_idx - start_idx}")
    logger.info(f"Instances failed: {failed_instances}")
    total_fast_steps = sum(
        int((record.get("runtime_result") or {}).get("rounds", 0) or 0)
        for record in instance_records
    )
    total_slow_updates = sum(
        len((record.get("runtime_result") or {}).get("slow_update_logs", []) or [])
        for record in instance_records
    )
    logger.info(f"Evolution steps : {total_fast_steps}")
    logger.info(f"Slow updates    : {total_slow_updates}")
    if all_raw_scores:
        raw_avg = sum(all_raw_scores) / len(all_raw_scores)
        logger.info(f"Avg runner raw score      : {raw_avg:.3f} | valid={len(all_raw_scores)}")
    else:
        logger.warning("Avg runner raw score      : N/A (no valid metric score)")

    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        logger.info(f"Avg evolution quality score: {avg:.3f} | valid={len(all_scores)}")
    else:
        logger.warning("Avg evolution quality score: N/A (no successful instances)")
    logger.info("=" * 60)

    summary_payload = {
        "timestamp": run_ts,
        "output_dir": str(output_dir),
        "instances_run": end_idx - start_idx,
        "instances_failed": failed_instances,
        "stopped_by_quota": stopped_by_quota,
        "evolution_steps": total_fast_steps,
        "slow_updates": total_slow_updates,
        "avg_runner_raw_score": (sum(all_raw_scores) / len(all_raw_scores)) if all_raw_scores else None,
        "avg_evolution_quality": (sum(all_scores) / len(all_scores)) if all_scores else None,
        "per_instance_reset": True,
        "instance_files": [f"instance_{r.get('instance_idx')}.json" for r in instance_records],
    }
    _write_json(output_dir / "summary.json", summary_payload)
    instances_index = [
        {
            "instance_idx": r.get("instance_idx"),
            "question": r.get("question"),
            "metrics": r.get("metrics", {}),
            "instance_file": f"instance_{r.get('instance_idx')}.json",
        }
        for r in instance_records
    ]
    _write_json(output_dir / "instances.json", instances_index)
    _write_json(output_dir / "evaluation.json", evaluation_records)
    _write_json(
        output_dir / "evaluation_summary.json",
        {
            "num_instances": len(evaluation_records),
            "avg_judge_similarity_score": (
                sum(
                    float(record.get("judge_similarity_score", 0.0) or 0.0)
                    for record in evaluation_records
                ) / len(evaluation_records)
                if evaluation_records
                else None
            ),
            "records_file": "evaluation.json",
        },
    )
    logger.info("Saved experiment trace to %s", output_dir)


def run_evolution_in_batches(
    start_total: int,
    end_total: int,
    batch_size: int,
    skip_meta_init: bool = False,
    agent_name: str = "multi-agent-independent",
    n_base_agents: int = 1,
    min_iterations_per_agent: int = 1,
    max_iterations_per_agent: int = 2,
    agent_model_override: str = "",
    agent_api_base: str = "",
    agent_api_key: str = "",
    agent_temperature: float = 0.0,
    meta_model: str = "gemini/gemini-2.0-flash",
    meta_api_base: str = "",
    meta_api_key: str = "",
    meta_temperature: float = 0.3,
    relax_constraints: bool = False,
    max_birth_death_pairs: int = 2,
    max_edge_edits: int = 4,
    init_n_min: int = 3,
    init_n_max: int = 8,
    pop_n_min: int = 0,
    pop_n_max: int = 0,
    fast_steps_per_window: int = 1,
    bd_check_interval: int = 1,
    graph_rewire_interval: int = 20,
    fixed_topology_file: str = "",
):
    """按批次切分区间并重复调用主实验。"""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if end_total <= start_total:
        logger.warning(
            "Empty total range: start_total=%s, end_total=%s. Nothing to run.",
            start_total,
            end_total,
        )
        return

    sleep_between_batches = float(os.environ.get("BATCH_SLEEP_SECONDS", "0"))

    i = start_total
    while i < end_total:
        current_end = min(i + batch_size, end_total)
        logger.info(
            "Running batch: start_idx=%s, end_idx=%s, batch_size=%s",
            i,
            current_end,
            batch_size,
        )
        run_evolution_experiment(
            start_idx=i,
            end_idx=current_end,
            skip_meta_init=skip_meta_init,
            agent_name=agent_name,
            n_base_agents=n_base_agents,
            min_iterations_per_agent=min_iterations_per_agent,
            max_iterations_per_agent=max_iterations_per_agent,
            agent_model_override=agent_model_override,
            agent_api_base=agent_api_base,
            agent_api_key=agent_api_key,
            agent_temperature=agent_temperature,
            meta_model=meta_model,
            meta_api_base=meta_api_base,
            meta_api_key=meta_api_key,
            meta_temperature=meta_temperature,
            relax_constraints=relax_constraints,
            max_birth_death_pairs=max_birth_death_pairs,
            max_edge_edits=max_edge_edits,
            init_n_min=init_n_min,
            init_n_max=init_n_max,
            pop_n_min=pop_n_min,
            pop_n_max=pop_n_max,
            fast_steps_per_window=fast_steps_per_window,
            bd_check_interval=bd_check_interval,
            graph_rewire_interval=graph_rewire_interval,
            fixed_topology_file=fixed_topology_file,
        )
        i = current_end

        if i < end_total and sleep_between_batches > 0:
            logger.info("Sleep %.2fs before next batch", sleep_between_batches)
            time.sleep(sleep_between_batches)


# ════════════════════════════════════════════════════
# 5. 入口
# ════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    if not os.getenv("SLURM_JOB_ID") and os.getenv("ALLOW_DIRECT_RUN", "0") != "1":
        print(
            "ERROR: scripts/run_evolution.py must be launched from Slurm. "
            "Use `sbatch scripts/run_demo.slurm`, or set ALLOW_DIRECT_RUN=1 "
            "for local debugging only."
        )
        raise SystemExit(2)

    parser = argparse.ArgumentParser(description="Run meta-evolution experiment with Gemini API")
    parser.add_argument("--start",      type=int, default=38,  help="Dataset start index")
    parser.add_argument("--end",        type=int, default=40,  help="Dataset end index (exclusive)")
    parser.add_argument("--start_total", type=int, default=None, help="Total range start for batch mode")
    parser.add_argument("--end_total",   type=int, default=None, help="Total range end (exclusive) for batch mode")
    parser.add_argument("--batch_size",  type=int, default=0, help="Batch size for batch mode; 0 means single run mode")
    parser.add_argument("--batch_sleep", type=float, default=0.0, help="Sleep seconds between batches")
    parser.add_argument("--agent_name", type=str, default="multi-agent-independent", help="Agent system name")
    parser.add_argument("--n_base_agents", type=int, default=1, help="Template agent-system worker count (not meta init population size)")
    parser.add_argument("--min_iterations_per_agent", type=int, default=1, help="Minimum iterations per worker per round")
    parser.add_argument("--max_iterations_per_agent", type=int, default=2, help="Maximum iterations per worker per round")
    parser.add_argument("--agent_model", type=str, default="", help="Agent LLM model, e.g. openai/qwen-local")
    parser.add_argument("--agent_api_base", type=str, default="", help="Agent LLM API base, e.g. http://localhost:10907/v1")
    parser.add_argument("--agent_api_key", type=str, default="", help="Agent LLM API key")
    parser.add_argument("--agent_temperature", type=float, default=0.0, help="Agent LLM temperature")
    parser.add_argument("--meta_model", type=str, default="gemini/gemini-2.5-pro", help="Meta LLM model")
    parser.add_argument("--meta_api_base", type=str, default="", help="Meta LLM API base")
    parser.add_argument("--meta_api_key", type=str, default="", help="Meta LLM API key")
    parser.add_argument("--meta_temperature", type=float, default=0.3, help="Meta LLM temperature")
    parser.add_argument("--relax_constraints", action="store_true", help="Relax BD/rewire constraints, keep per-round upper bounds")
    parser.add_argument("--max_birth_death_pairs", type=int, default=2, help="Upper bound of BD pairs per slow update")
    parser.add_argument("--max_edge_edits", type=int, default=8, help="Upper bound of add/remove edges per slow update")
    parser.add_argument("--fast_steps_per_window", type=int, default=1, help="Fast-step window length used for slow-update summary statistics")
    parser.add_argument("--bd_check_interval", type=int, default=1, help="Initial/minimum interval (in fast steps) before triggering birth-death checks")
    parser.add_argument("--graph_rewire_interval", type=int, default=20, help="Minimum interval (in fast steps) before allowing graph rewire checks")
    parser.add_argument("--init_n_min", type=int, default=3, help="Initial topology minimum population size (how many agents to spawn at start)")
    parser.add_argument("--init_n_max", type=int, default=8, help="Initial topology maximum population size")
    parser.add_argument("--pop_n_min", type=int, default=0, help="Evolution population lower bound (0 = same as init_n_min)")
    parser.add_argument("--pop_n_max", type=int, default=0, help="Evolution population upper bound (0 = same as init_n_max)")
    parser.add_argument("--api_key",    type=str, default="",  help="Gemini API key (overrides env var)")
    parser.add_argument("--use_dmx_meta", action="store_true", help="Use DMX non-stream API for both meta and agent LLM")
    parser.add_argument("--dmx_api_key", type=str, default="", help="DMX API key")
    parser.add_argument("--dmx_model",   type=str, default="gemini-2.5-flash", help="DMX Gemini model")
    parser.add_argument("--skip_meta_init", action="store_true", help="Skip initial meta-LLM call and use default topology")
    parser.add_argument("--fixed_topology_file", type=str, default="", help="Load a fixed initial topology from JSON")
    parser.add_argument("--instance_retries", type=int, default=None, help="Retry count per instance")
    parser.add_argument("--max_fast_rounds", type=int, default=None, help="Max fast rounds per instance")
    parser.add_argument("--target_answer_quality", type=float, default=None, help="Target answer quality for runtime stop")
    parser.add_argument("--per_instance_sleep", type=float, default=None, help="Sleep seconds between instances")
    args = parser.parse_args()

    if args.api_key:
        GEMINI_API_KEY = args.api_key

    if args.use_dmx_meta:
        USE_DMX_META_NONSTREAM = True
        DMX_MODEL = args.dmx_model
        if args.dmx_api_key:
            DMX_API_KEY = args.dmx_api_key
        if not DMX_API_KEY:
            print("ERROR: --use_dmx_meta requires --dmx_api_key or DMX_API_KEY env var")
            exit(1)

    '''
    if (not USE_DMX_META_NONSTREAM) and GEMINI_API_KEY == "AIza-YOUR-KEY-HERE":
        print("ERROR: Please set your Gemini API key via --api_key or GEMINI_API_KEY env var")
        print("  export GEMINI_API_KEY=AIza...")
        print("  python run_scripts/run_evolution_gemini.py --api_key AIza...")
        exit(1)
    '''

    if args.instance_retries is not None:
        os.environ["INSTANCE_RETRIES"] = str(args.instance_retries)
    if args.max_fast_rounds is not None:
        os.environ["MAX_FAST_ROUNDS_PER_INSTANCE"] = str(args.max_fast_rounds)
    if args.target_answer_quality is not None:
        os.environ["TARGET_ANSWER_QUALITY"] = str(args.target_answer_quality)
    if args.per_instance_sleep is not None:
        os.environ["PER_INSTANCE_SLEEP_SECONDS"] = str(args.per_instance_sleep)
    if args.batch_sleep > 0:
        os.environ["BATCH_SLEEP_SECONDS"] = str(args.batch_sleep)

    use_batch_mode = args.batch_size > 0 and args.start_total is not None and args.end_total is not None
    if use_batch_mode:
        run_evolution_in_batches(
            start_total=args.start_total,
            end_total=args.end_total,
            batch_size=args.batch_size,
            skip_meta_init=args.skip_meta_init,
            agent_name=args.agent_name,
            n_base_agents=args.n_base_agents,
            min_iterations_per_agent=args.min_iterations_per_agent,
            max_iterations_per_agent=args.max_iterations_per_agent,
            agent_model_override=args.agent_model,
            agent_api_base=args.agent_api_base,
            agent_api_key=args.agent_api_key,
            agent_temperature=args.agent_temperature,
            meta_model=args.meta_model,
            meta_api_base=args.meta_api_base,
            meta_api_key=args.meta_api_key,
            meta_temperature=args.meta_temperature,
            relax_constraints=args.relax_constraints,
            max_birth_death_pairs=args.max_birth_death_pairs,
            max_edge_edits=args.max_edge_edits,
            init_n_min=args.init_n_min,
            init_n_max=args.init_n_max,
            pop_n_min=args.pop_n_min,
            pop_n_max=args.pop_n_max,
            fast_steps_per_window=args.fast_steps_per_window,
            bd_check_interval=args.bd_check_interval,
            graph_rewire_interval=args.graph_rewire_interval,
            fixed_topology_file=args.fixed_topology_file,
        )
    else:
        run_evolution_experiment(
            start_idx=args.start,
            end_idx=args.end,
            skip_meta_init=args.skip_meta_init,
            agent_name=args.agent_name,
            n_base_agents=args.n_base_agents,
            min_iterations_per_agent=args.min_iterations_per_agent,
            max_iterations_per_agent=args.max_iterations_per_agent,
            agent_model_override=args.agent_model,
            agent_api_base=args.agent_api_base,
            agent_api_key=args.agent_api_key,
            agent_temperature=args.agent_temperature,
            meta_model=args.meta_model,
            meta_api_base=args.meta_api_base,
            meta_api_key=args.meta_api_key,
            meta_temperature=args.meta_temperature,
            relax_constraints=args.relax_constraints,
            max_birth_death_pairs=args.max_birth_death_pairs,
            max_edge_edits=args.max_edge_edits,
            init_n_min=args.init_n_min,
            init_n_max=args.init_n_max,
            pop_n_min=args.pop_n_min,
            pop_n_max=args.pop_n_max,
            fast_steps_per_window=args.fast_steps_per_window,
            bd_check_interval=args.bd_check_interval,
            graph_rewire_interval=args.graph_rewire_interval,
            fixed_topology_file=args.fixed_topology_file,
        )
