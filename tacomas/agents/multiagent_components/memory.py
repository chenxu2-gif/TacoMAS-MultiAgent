import threading
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .plan import OrchestrationPlan


class EnhancedMemory(BaseModel):
    """Per-agent memory built from rolling self-summaries."""

    all_findings: List[str] = Field(default_factory=list)
    agent_findings: Dict[str, List[str]] = Field(default_factory=dict)
    execution_plan: Optional[OrchestrationPlan] = None
    original_task: str = ""

    def model_post_init(self, __context: Any):
        self._lock = threading.RLock()

    def add_findings(self, agent_id: str, findings: str):
        """Store the latest self-summary for an agent."""
        with self._lock:
            if agent_id not in self.agent_findings:
                self.agent_findings[agent_id] = []

            self.agent_findings[agent_id].append(findings)
            self.all_findings.append(findings)

    def get_findings_for_agent(self, agent_id: str) -> List[str]:
        with self._lock:
            return self.agent_findings.get(agent_id, []).copy()

    def get_latest_summary(self, agent_id: str) -> str:
        with self._lock:
            findings = self.agent_findings.get(agent_id, [])
            return findings[-1] if findings else ""

    def get_all_latest_summaries(self) -> Dict[str, str]:
        with self._lock:
            return {
                agent_id: findings[-1]
                for agent_id, findings in self.agent_findings.items()
                if findings
            }

    def get_all_findings(self) -> List[str]:
        with self._lock:
            return self.all_findings.copy()

    def get_memory_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "original_task": self.original_task,
                "execution_plan": self.execution_plan.model_dump()
                if self.execution_plan is not None
                else None,
                "agent_latest_summaries": self.get_all_latest_summaries(),
                "agent_summary_history": {
                    agent_id: findings.copy()
                    for agent_id, findings in self.agent_findings.items()
                },
                "total_summaries": len(self.all_findings),
            }
