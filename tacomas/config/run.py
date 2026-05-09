from typing import Any, Dict, Optional, Self

from litellm import _logging as litellm_logging
from pydantic import BaseModel, Field, model_validator

from tacomas.agents import AgentSystem, get_agent_cls
from tacomas.langfuse_client import get_lf_client
from tacomas.utils import (
    disable_local_cache,
    enable_local_cache,
    enable_local_logging,
)

from .dataset import DatasetConfig
from .llm import LLMConfig
from .prompts import Prompt


class MultiAgentResearchConfig(BaseModel):
    # hyc-revised

    n_base_agents: int = 3
    min_orchestrator_turns: int = 1
    max_orchestrator_turns: int = 3
    
    # 修正变量名并添加缺失的字段
    min_iterations_per_agent: int = 3  # 从 min_searches... 改过来，或者都保留
    max_iterations_per_agent: int = 10 # 默认值可以改

    # for independent agent, these parameters are not used but we keep them for compatibility
    aggregation_method: str = "llm_vote"

    # for decentralized debate, these parameters are not used but we keep them for compatibility
    max_debate_rounds: int = 3 
    max_exploration_steps: int = 5

    
    
    # 添加缺失的字段（如果不加，Hydra 传进来也会报错说 unexpected arguments）
    # max_rounds: int = 20
    
    # 为了省事，允许任意额外参数传入 (Recommended)
    class Config:
        extra = "allow" 


class AgentConfig(BaseModel):
    name: str
    prompts: Dict[str, Prompt] = Field(default_factory=dict)
    agent_specific_config: Optional[MultiAgentResearchConfig] = None

    @model_validator(mode="after")
    def check_prompts(self) -> Self:
        agent_cls = get_agent_cls(self.name)
        agent_cls.check_required_prompts(self.prompts)
        return self

    @model_validator(mode="before")
    @classmethod
    def add_prompt_names(cls, data: Any) -> Dict[str, Any]:
        data = dict(data)
        prompts = dict(data.get("prompts", {}))
        for k, prompt in prompts.items():
            prompt = dict(prompt)
            if prompt.get("name") is None:
                prompt["name"] = k
            prompts[k] = prompt
        data["prompts"] = prompts
        return data

    def get_run_metadata(self) -> Dict[str, Any]:
        prompts = {}
        assert self.prompts is not None, "Prompts must be defined in AgentConfig"
        for k, prompt in self.prompts.items():
            prompts[k] = {
                "name": prompt.name,
            }
        return {
            "name": self.name,
            "prompts": prompts,
        }


class RunConfig(BaseModel):
    agent: AgentConfig
    dataset: DatasetConfig
    llm: LLMConfig
    run_name: str
    save_dir: Optional[str] = None
    log_langfuse: bool = True
    use_disk_cache: bool = False
    debug: bool = False
    max_instances: Optional[int] = None
    num_workers: int = 1

    # === [新增] 添加这两个字段 ===
    start_idx: int = 0
    end_idx: Optional[int] = None
    # ==========================

    @property
    def run_parallel(self) -> bool:
        return self.num_workers > 1

    def model_post_init(self, context: Any) -> None:
        client = get_lf_client()
        self.log_langfuse = (
            self.log_langfuse and client is not None
            # and self.dataset.langfuse_dataset is not None
        )
        if not self.log_langfuse and not self.run_parallel:
            enable_local_logging(prompt_only=True)
        litellm_logging._disable_debugging()  # type: ignore
        # Disable litellm debugging logs
        if self.use_disk_cache:
            enable_local_cache()
        else:
            disable_local_cache()

    def get_agent(self) -> AgentSystem:
        return get_agent_cls(self.agent.name).from_config(
            llm_config=self.llm,
            dataset_config=self.dataset,
            prompts=self.agent.prompts,
            **(
                self.agent.agent_specific_config.model_dump(exclude_none=True)
                if self.agent.agent_specific_config is not None
                else {}
            ),
        )

    def get_run_metadata(self) -> Dict[str, Any]:
        ret: Dict[str, Any] = {
            "agent": self.agent.get_run_metadata(),
            "llm": self.llm.model_dump(exclude_none=True),
            "dataset": self.dataset.model_dump(exclude_none=True),
        }
        if self.save_dir is not None:
            ret["save_dir"] = self.save_dir
        ret["run_name"] = self.run_name
        ret["num_workers"] = self.num_workers
        return ret
