from typing import List, Optional

from pydantic import BaseModel


class Subtask(BaseModel):
    agent_id: str
    objective: str
    focus: str
    role: Optional[str] = None


class OrchestrationPlan(BaseModel):
    subtasks: List[Subtask]
    reasoning: str
