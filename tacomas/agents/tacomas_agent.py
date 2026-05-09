"""TacoMAS: A role-specialized multi-agent system for financial research.

Six fixed roles with hard tool-permission boundaries:
  planner    — task decomposition, no tools
  searcher   — evidence collection (web_search / edgar_search / parse_html_page / retrieve_information)
  calculator — numerical derivation (retrieve_information only)
  verifier   — cross-validation (web_search / edgar_search / parse_html_page / retrieve_information)
  summary    — final synthesis + submit_final_result (retrieve_information + submit_final_result)
  reflector  — post-mortem analysis, no tools

Usage:
    from tacomas.agents.tacomas import TacoMASSystem

    # Programmatic (e.g. in tests or notebooks)
    from tacomas.config.llm import LLMConfig
    from tacomas.config.dataset import DatasetConfig
    from tacomas.config.prompts import Prompt

    agent = TacoMASSystem.from_config(
        llm_config=LLMConfig(model="openai/qwen-local", ...),
        dataset_config=dataset_config,
        prompts={...},  # load from run_conf/agent/tacomas.yaml
    )
    output = agent.run_agent(instance)

    # Via Hydra (recommended for experiments)
    # Set `- agent: tacomas` in run_conf/run_exp.yaml
    # and `- dataset: finance-benchmark` then run:
    #   python -m run_scripts.run_experiment
"""

import asyncio
import os.path as osp
import time
from typing import Dict, List, Optional

import yaml

from tacomas.agents.base import AgentSystemWithTools
from tacomas.config.llm import LLMParams
from tacomas.datasets import DatasetInstance, DatasetInstanceOutputWithTrajectory
from tacomas.logger import logger
from tacomas.utils import write_yaml

from .multiagent_components.conversation import OrchestrationResult
from .multiagent_components.mas_lead_agent import LeadAgent
from .multiagent_components.memory import EnhancedMemory
from .registry import register_agent

# ---------------------------------------------------------------------------
# Default permission tables (override via YAML or constructor kwargs)
# ---------------------------------------------------------------------------
_DEFAULT_ROLE_SEQUENCE: List[str] = [
    "planner",
    "searcher",
    "calculator",
    "verifier",
    "summary",
    "reflector",
]

_DEFAULT_ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "planner":    [],
    "searcher":   ["prepare_primary_filing", "web_search", "edgar_search", "parse_html_page", "retrieve_information"],
    "calculator": ["retrieve_information"],
    "verifier":   ["retrieve_information"],
    "summary":    ["retrieve_information", "submit_final_result"],
    "reflector":  [],
}

_DEFAULT_ROLE_TOOL_BUDGET: Dict[str, int] = {
    "planner":    0,
    "searcher":   12,
    "calculator": 6,
    "verifier":   8,
    "summary":    2,
    "reflector":  0,
}


def _load_permissions_yaml(path: str) -> Dict:
    """Load role_permissions and role_tool_budget from a YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except FileNotFoundError:
        logger.warning(f"TacoMAS permissions file not found: {path}. Using defaults.")
        return {}


@register_agent("tacomas")
class TacoMASSystem(AgentSystemWithTools):
    """TacoMAS: role-specialized 6-agent pipeline for financial Q&A.

    Each agent role has a fixed tool whitelist and per-run tool budget.
    Only the `summary` agent may call `submit_final_result`.

    Constructor parameters
    ----------------------
    n_base_agents : int
        Number of worker agents (default 6, one per role).
    min_iterations_per_agent / max_iterations_per_agent : int
        Inner LLM loop bounds per agent.
    min_orchestrator_turns / max_orchestrator_turns : int
        Outer orchestration round bounds.
    role_sequence : list[str]
        Ordered role names assigned round-robin to agents.
    role_permissions : dict[str, list[str]]
        Tool whitelist per role.  Pass {} to use defaults.
    role_tool_budget : dict[str, int]
        Max tool calls per role per run. Pass {} to use defaults.
    permissions_yaml : str or None
        Optional path to a YAML file with role_permissions and
        role_tool_budget keys (overrides constructor dicts).
    """

    required_prompts = [
        "lead_agent",
        "subagent",
        "planner",
        "searcher",
        "calculator",
        "verifier",
        "summary",
        "reflector",
    ]

    def __init__(
        self,
        *args,
        n_base_agents: int = 6,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        min_orchestrator_turns: int = 2,
        max_orchestrator_turns: int = 6,
        role_sequence: Optional[List[str]] = None,
        role_permissions: Optional[Dict[str, List[str]]] = None,
        role_tool_budget: Optional[Dict[str, int]] = None,
        permissions_yaml: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # ----------------------------------------------------------------
        # Resolve permissions: YAML > constructor kwargs > defaults
        # ----------------------------------------------------------------
        yaml_data: Dict = {}
        if permissions_yaml:
            yaml_data = _load_permissions_yaml(permissions_yaml)

        resolved_permissions = (
            yaml_data.get("role_permissions")
            or role_permissions
            or _DEFAULT_ROLE_PERMISSIONS
        )
        resolved_budget = (
            yaml_data.get("role_tool_budget")
            or role_tool_budget
            or _DEFAULT_ROLE_TOOL_BUDGET
        )
        resolved_sequence = (
            yaml_data.get("role_sequence")
            or role_sequence
            or _DEFAULT_ROLE_SEQUENCE
        )

        logger.info(
            f"TacoMASSystem initializing:\n"
            f"  roles       : {resolved_sequence}\n"
            f"  permissions : {resolved_permissions}\n"
            f"  budgets     : {resolved_budget}"
        )

        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.min_orchestrator_turns = min_orchestrator_turns
        self.max_orchestrator_turns = max_orchestrator_turns
        self.role_sequence = resolved_sequence
        self.role_permissions = resolved_permissions
        self.role_tool_budget = resolved_budget

        self.memory = EnhancedMemory()
        self.lead_agent = LeadAgent(
            *args,
            memory=self.memory,
            min_iterations_per_agent=min_iterations_per_agent,
            max_iterations_per_agent=max_iterations_per_agent,
            min_orchestrator_turns=min_orchestrator_turns,
            max_orchestrator_turns=max_orchestrator_turns,
            num_base_agents=n_base_agents,
            role_sequence=resolved_sequence,
            role_permissions=resolved_permissions,
            role_tool_budget=resolved_budget,
            domain_config={"task_blurb": kwargs.get("task_blurb", "tacomas pipeline coordinator")},
            **kwargs,
        )
        self.last_processing_result: Optional[OrchestrationResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        """Synchronous entry point (wraps async)."""
        return asyncio.run(
            self.run_agent_async(instance, instance_dir, llm_params, instance_idx)
        )

    async def run_agent_async(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        """Async pipeline execution."""
        start_time = time.time()
        logger.info(
            f"TacoMASSystem starting instance {instance_idx} "
            f"with {self.n_base_agents} agents, roles={self.role_sequence}"
        )

        llm_params_dict = llm_params.model_dump() if llm_params else {}

        processing_result: OrchestrationResult = await self.lead_agent.orchestrate_work(
            task_instance=instance,
            llm_params_dict=llm_params_dict,
        )
        self.last_processing_result = processing_result

        final_answer = processing_result.synthesized_answer
        execution_time = time.time() - start_time
        logger.info(
            f"TacoMASSystem finished in {execution_time:.1f}s | "
            f"iterations={processing_result.total_agent_iterations}"
        )

        if instance_dir is not None:
            write_yaml(
                processing_result.model_dump(),
                osp.join(instance_dir, "multi_agent_output.yaml"),
                use_long_str_representer=True,
                truncate_floats=False,
            )

        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=[],
            final_env_output=processing_result.combined_env_status,
        )
