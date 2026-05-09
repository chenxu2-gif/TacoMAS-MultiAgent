import json
import threading
import traceback
import re
import uuid
import os
from typing import cast, Dict, Any, List

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages

from tacomas.agents.base import BaseAgentWithTools
from tacomas.datasets import DatasetInstance
from tacomas.datasets.base import DatasetEnvStatus
from tacomas.logger import logger

# Reuse SubAgent data structures
from .conversation import SubAgentConversationHistory, SubAgentRoundResult


ROLE_PSEUDO_TOOLS = {
    "planner": ["decompose_task", "set_evidence_requirements"],
    "searcher": ["collect_sources", "extract_facts"],
    "researcher": ["collect_sources", "triage_candidates"],
    "analyst": ["collect_sources", "extract_facts"],
    "curator": ["collect_sources", "rank_evidence"],
    "calculator": ["derive_metrics", "check_units"],
    "forecaster": ["derive_metrics", "compare_trends"],
    "verifier": ["cross_validate", "find_conflicts"],
    "schema_verifier": ["validate_bundle", "check_field_semantics"],
    "auditor": ["cross_validate", "stress_test_claims"],
    "critic": ["cross_validate", "challenge_assumptions"],
    "summary": ["compose_answer", "format_sources"],
    "synthesizer": ["compose_answer", "merge_evidence"],
    "reflector": ["postmortem", "prompt_patch"],
}

ROLE_PROMPT_ALIASES = {
    "orchestrator": "planner",
    "lead_agent": "planner",
    "researcher": "searcher",
    "analyst": "searcher",
    "curator": "searcher",
    "worker": "searcher",
    "subagent": "searcher",
    "auditor": "verifier",
    "schema_verifier": "verifier",
    "critic": "verifier",
    "forecaster": "calculator",
    "synthesizer": "reflector",
}


class BaseWorkerAgent(BaseAgentWithTools):
    """
    Independent worker agent that solves tasks directly without an orchestrator loop.
    Strictly mirrors WorkerSubagent execution logic for tool handling and error resilience.
    """

    required_prompts = ["base_agent"]

    def __init__(
        self,
        agent_id: str,
        task_instance: DatasetInstance,
        role: str = "worker",
        allowed_tools: list[str] | None = None,
        tool_budget: int | None = None,
        policy: str = "",
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        **kwargs,
    ):
        # BaseAgent keeps only required prompts by default; keep a copy so role prompts are not lost.
        all_prompts = kwargs.get("prompts")
        super().__init__(**kwargs)
        if isinstance(all_prompts, dict) and all_prompts:
            self.prompts = all_prompts
        self.agent_id = agent_id
        self.role = role
        self.allowed_tools = list(allowed_tools) if allowed_tools is not None else list(self.tools)
        self.tool_budget = tool_budget
        self.policy = policy
        self.objective = ""
        self.strategy = policy
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.task_instance = task_instance
        
        # Initialize environment and tools
        self.env, self.llm_w_tools = self.init_environment(
            task_instance,
            agent_id,
            tools_override=self.allowed_tools,
            tool_budget=self.tool_budget,
            role=self.role,
        )
        
        # Shared templates (e.g. valid_tools, task_rule)
        self.shared_prompt_templates = self.get_dataset_prompt_templates(self.env)
        
        self.conv_history = SubAgentConversationHistory(agent_id=agent_id)
        self._execution_lock = threading.Lock()

        logger.info(f"BaseWorkerAgent {agent_id} initialized.")

    @classmethod
    def init_from_agent(cls, agent: BaseAgentWithTools, agent_id: str, task_instance: DatasetInstance, **kwargs):
        prompts = kwargs.pop("prompts", agent.prompts)
        if isinstance(prompts, dict) and "base_agent" not in prompts:
            fallback = prompts.get("subagent") or prompts.get("searcher")
            if fallback is not None:
                prompts = {**prompts, "base_agent": fallback}
        return cls(
            agent_id=agent_id,
            task_instance=task_instance,
            llm=agent.llm,
            dataset=agent.dataset,
            prompts=prompts,
            tools=agent.tools,
            env=agent.env_name,
            env_prompts=agent.env_prompts,
            **kwargs,
        )

    def _get_role_prompt(self):
        """Return a role-specific prompt if available, otherwise fallback safely."""
        if self.role in self.prompts:
            return self.prompts[self.role]
        alias = ROLE_PROMPT_ALIASES.get(self.role)
        if alias and alias in self.prompts:
            return self.prompts[alias]
        if "subagent" in self.prompts:
            return self.prompts["subagent"]
        return self.prompts["base_agent"]

    def _build_role_note(self, objective: str) -> str:
        pseudo_tools = ROLE_PSEUDO_TOOLS.get(self.role, [])
        forbidden_tools = [
            tool
            for tool in [
                "web_search",
                "edgar_search",
                "prepare_primary_filing",
                "parse_html_page",
                "retrieve_information",
                "submit_final_result",
            ]
            if tool not in self.allowed_tools
        ]
        return (
            "ROLE_NOTE\n"
            f"- role: {self.role}\n"
            f"- policy: {self.policy}\n"
            f"- objective: {objective}\n"
            f"- strategy: {self.policy}\n"
            f"- allowed_tools: {self.allowed_tools}\n"
            f"- forbidden_tools: {forbidden_tools}\n"
            f"- tool_budget: {self.tool_budget}\n"
            f"- pseudo_tools: {pseudo_tools}\n"
            "- required_output_schema: facts/sources/confidence(or PASS/FAIL for verifier)"
        )

    def _get_start_messages(self, message: str):
        prompt_obj = self._get_role_prompt()
        role_note = self._build_role_note(message)
        tools_description = (self.env.tools_description or "").strip() or "none"
        prompt_vars = {
            **self.shared_prompt_templates,
            "tools_description": tools_description,
        }
        for template_name in ["planning", "coordination", "base", "start_with_orchestrator_guidance"]:
            try:
                template = prompt_obj.get_template(template_name, with_base=False)
                compiled_messages = template.compile(
                    orchestrator_objective=message,
                    orchestrator_guidance=message,
                    role_note=role_note,
                    **prompt_vars,
                )
                # Ensure runtime round context (task + incoming graph messages) is always visible
                # even if the role template does not explicitly render orchestrator_* fields.
                compiled_messages.append(
                    {
                        "role": "user",
                        "content": f"ROUND_CONTEXT:\n{message}",
                    }
                )
                return compiled_messages
            except Exception:
                continue
        # Last-resort fallback for malformed prompt packs.
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{message}\n\n{role_note}"},
        ]

    def _extract_tool_result_text(self, tool_payload: Any) -> str:
        text = str(tool_payload or "").strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ["result", "output", "final_result", "answer"]:
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            return text
        except Exception:
            return text

    def _looks_like_non_final_answer(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        lower = t.lower()
        if lower.startswith("failed to generate a valid answer"):
            return True
        if "\"type\": \"function\"" in lower:
            return True
        if lower.startswith("[") and ("\"raw_content\"" in lower or "\"filedat\"" in lower):
            return True
        # DSML / textual function-calling outputs are not final answers.
        if "<function_calls>" in lower or "</function_calls>" in lower:
            return True
        if "<invoke " in lower or "</invoke>" in lower:
            return True
        if "|dsml|function_calls" in lower or "|dsml|invoke" in lower:
            return True
        return False

    def _extract_env_final_answer(self) -> str:
        try:
            if hasattr(self.env, "get_env_output"):
                env_out = self.env.get_env_output()
                if hasattr(env_out, "output") and isinstance(env_out.output, str) and env_out.output.strip():
                    return env_out.output.strip()
        except Exception:
            pass
        return ""

    def _has_tool_access(self) -> bool:
        return bool(self.allowed_tools)

    def _looks_like_completed_work(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        lower = t.lower()
        placeholder_markers = [
            "url_from_task",
            "document_from_task",
            "tool_code:",
            "tool_input",
            "placeholder",
        ]
        if any(marker in lower for marker in placeholder_markers):
            return False
        weak_markers = [
            "here's the plan",
            "i will",
            "i'll start by",
            "use `web_search`",
            "use `edgar_search`",
            "use `parse_html_page`",
            "use `retrieve_information`",
            "search for",
            "plan:",
        ]
        if any(marker in lower for marker in weak_markers):
            return False
        strong_markers = [
            "source",
            "verified",
            "evidence",
            "confidence",
            "verdict",
            "pass",
            "fail",
            "adjustments",
            "reconciliation",
        ]
        return any(marker in lower for marker in strong_markers)

    def _is_repeated_tool_call(
        self,
        tool_calls_made: List[Dict[str, Any]],
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> bool:
        if len(tool_calls_made) < 2:
            return False
        recent = tool_calls_made[-2:]
        return all(
            call.get("name") == tool_name and call.get("arguments", {}) == tool_args
            for call in recent
        )

    def _extract_task_text_from_query(self, query: str) -> str:
        text = str(query or "").strip()
        if not text:
            return ""
        block = re.search(r"\[Task\]\s*(.*?)\s*(?:\n\[|$)", text, flags=re.DOTALL)
        if block:
            candidate = block.group(1).strip()
            if candidate:
                return candidate
        objective = re.search(r"Objective:\s*(.*)", text)
        if objective:
            candidate = objective.group(1).strip()
            if candidate:
                return candidate
        return text.splitlines()[0].strip()

    def _is_generic_search_query(self, search_query: str) -> bool:
        q = str(search_query or "").strip().lower()
        if not q:
            return True
        generic_patterns = [
            "10-k",
            "10-q",
            "filings",
            "sec filings",
            "investor relations",
            "latest filing",
            "annual report",
            "quarterly report",
        ]
        # Queries that mainly ask for filings rather than the semantic task answer
        # should be replaced by the task-level query.
        hits = sum(1 for pattern in generic_patterns if pattern in q)
        return hits >= 2 or len(q.split()) <= 4

    def _build_semantic_search_query(self, query: str) -> str:
        task_text = self._extract_task_text_from_query(query)
        task_text = re.sub(r"\s+", " ", task_text).strip()
        return task_text[:300]

    def _build_task_search_anchors(self, task_text: str, tool_name: str) -> str:
        anchors: List[str] = []

        def _add(value: str) -> None:
            cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
            if cleaned and cleaned.lower() not in {item.lower() for item in anchors}:
                anchors.append(cleaned)

        years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", str(task_text or "").lower())))
        quarter_refs = re.findall(r"\bq[1-4]\s*(?:fy\s*)?(?:19|20)\d{2}\b", str(task_text or "").lower())

        for year in years[-4:]:
            _add(year)
        for quarter in quarter_refs[:4]:
            _add(quarter.upper())

        return " ".join(anchors[:14]).strip()

    def _extract_guided_search_query_candidates(self, query: str) -> List[str]:
        text = str(query or "")
        candidates: List[str] = []

        def _add(candidate: str) -> None:
            cleaned = re.sub(r"\s+", " ", str(candidate or "")).strip(" \"'\n\t")
            if len(cleaned) < 8:
                return
            lower = cleaned.lower()
            if lower in {item.lower() for item in candidates}:
                return
            candidates.append(cleaned[:420])

        # Strongest signal: explicit runtime hint block.
        for block in re.findall(r"Next search hints:\s*(.+)", text):
            for part in re.split(r"\s*\|\s*", block):
                _add(part)

        # Searcher guidance can include an explicit rewritten query suggestion.
        for block in re.findall(r"Suggested rewritten query.*?:\s*\"([^\"]+)\"", text, flags=re.IGNORECASE):
            _add(block)

        # Meta guidance can be made tool-executable by using an explicit SEARCH_QUERY label.
        for block in re.findall(r"SEARCH_QUERY\s*:\s*(.+)", text, flags=re.IGNORECASE):
            _add(block)

        # Fall back to quoted candidate queries mentioned in workflow guidance.
        for section_name in ["Meta next-round workflow", "Meta workflow correction"]:
            pattern = rf"\[{re.escape(section_name)}\]\s*(.*?)(?:\n\[|$)"
            match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if not match:
                continue
            section = match.group(1).strip()
            for quoted in re.findall(r"\"([^\"]{8,240})\"", section):
                _add(quoted)
            if "query" in section.lower() or "search" in section.lower():
                lines = [ln.strip("- ").strip() for ln in section.splitlines() if ln.strip()]
                for line in lines:
                    if any(token in line.lower() for token in ["search query", "rewrite", "re-search", "search for", "look for"]):
                        _add(line)

        return candidates

    def _build_guided_search_query(self, query: str, tool_name: str) -> str:
        candidates = self._extract_guided_search_query_candidates(query)
        if candidates:
            return candidates[0]
        return self._build_combined_search_query(query, tool_name)

    def _extract_search_constraints_payload(self, query: str) -> Dict[str, Any]:
        text = str(query or "")
        match = re.search(
            r"\[Search constraints JSON\]\s*(\{.*?\})(?:\n\[|$)",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return {}
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _build_combined_search_query(self, query: str, tool_name: str) -> str:
        task_text = self._build_semantic_search_query(query)
        if not task_text:
            return ""
        anchor_text = self._build_task_search_anchors(task_text, tool_name)
        combined = f"{task_text} {anchor_text}".strip()
        return combined[:420]

    def _rewrite_tool_args_for_semantic_query(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        query: str,
    ) -> Dict[str, Any]:
        rewritten = dict(tool_args or {})
        effective_query = self._build_guided_search_query(query, tool_name)
        constraint_payload = self._extract_search_constraints_payload(query)
        if not effective_query:
            if tool_name == "prepare_primary_filing" and constraint_payload:
                for field in [
                    "required_constraints",
                    "avoid_constraints",
                    "prefer_source_patterns",
                    "avoid_source_patterns",
                    "missing_targets",
                ]:
                    value = constraint_payload.get(field)
                    if value:
                        rewritten[field] = value
            return rewritten

        if tool_name == "prepare_primary_filing":
            rewritten["search_query"] = effective_query
            rewritten["top_n_results"] = 10
            rewritten["max_valid_docs"] = 5
            for field in [
                "required_constraints",
                "avoid_constraints",
                "prefer_source_patterns",
                "avoid_source_patterns",
                "missing_targets",
            ]:
                value = constraint_payload.get(field)
                if value:
                    rewritten[field] = value
            return rewritten

        if tool_name in {"edgar_search", "web_search"}:
            arg_name = "search_query"
            current = str(rewritten.get(arg_name, "")).strip()
            if self._is_generic_search_query(current):
                rewritten[arg_name] = effective_query
        return rewritten

        
    def solve_task(self, query: str) -> SubAgentRoundResult:
        """
        Entry point for the worker to solve a task.
        """
        # Initialize a new conversation round before starting the task.
        # This appends an empty list [] to internal_comms, preventing index-out-of-range errors downstream.
        self.conv_history.start_new_round()

        # [Fix 1] convert "user" to "lead_agent" to satisfy Pydantic Literal constraint
        self.conv_history.add_external_message("lead_agent", query)

        # Track submission on a per-round basis so runtime synthesis can identify
        # whether this round actually produced a submitted final answer.
        self._submitted_this_round = False

        # Run the execution loop (ReAct / Chain of Thought)
        final_result = self._run_execution_loop(query)

        # [Fix 2] Ensure env_status is not None
        current_env_status = None
        if hasattr(self, "env") and self.env is not None:
            if hasattr(self.env, "env_status"):
                current_env_status = self.env.env_status()
            else:
                current_env_status = getattr(self.env, "get_status", lambda: {})()

        if current_env_status is None:
            current_env_status = DatasetEnvStatus(success=False, num_steps=0)

        if isinstance(current_env_status, dict):
            current_env_status = DatasetEnvStatus(
                success=bool(current_env_status.get("success", False)),
                num_steps=int(current_env_status.get("num_steps", 0)),
            )
        elif not isinstance(current_env_status, DatasetEnvStatus):
            current_env_status = DatasetEnvStatus(success=False, num_steps=0)

        # Success here means "submitted final answer in this round".
        current_env_status = DatasetEnvStatus(
            success=bool(getattr(self, "_submitted_this_round", False)),
            num_steps=int(current_env_status.num_steps),
        )

        # [Change 1] final_result.response -> final_result.findings
        self.conv_history.add_external_message("subagent", final_result.findings)

        # Return final_result directly (already a SubAgentRoundResult), just update env_status.
        final_result.env_status = current_env_status
        return final_result


        '''
        # (optional) mirror subagent: also log the final result to external history for completeness
        self.conv_history.add_external_message("subagent", final_result.response)

        return SubAgentRoundResult(
            response=final_result.response,
            conversation=self.conv_history.get_history(),
            env_status=current_env_status, 
            status="success",
            is_final=True
        )
        '''
    

    def _run_execution_loop(self, query: str) -> SubAgentRoundResult:
        """Core ReAct loop. Mirrors WorkerSubagent logic."""
        
        # 1. Compile Initial Prompt
        messages: List[Dict[str, Any]] = []
        try:
            messages = self._get_start_messages(query)
        except Exception as e:
            logger.warning(f"Error compiling structured prompt: {e}")
            logger.warning("Falling back to simple prompt construction.")
            # Fallback
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]


        # 2. Iteration Loop
        final_answer = ""
        submitted_this_round = False
        tool_calls_made: list[Dict[str, Any]] = []
        tool_names_made: list[str] = []
        curr_iteration = 0
        max_tool_calls_per_round = int(
            os.getenv(
                "MAX_TOOL_CALLS_PER_SUBAGENT_ROUND",
                str(max(self.max_iterations_per_agent, 4)),
            )
        )
        no_progress_turns = 0
        tool_failures = 0
        no_more_tools = False
        
        for iteration in range(self.max_iterations_per_agent):
            curr_iteration += 1
            logger.info(f"Agent {self.agent_id} iteration {iteration+1}/{self.max_iterations_per_agent}")
            
            # Invoke LLM
            response: AIMessage = cast(
                AIMessage,
                self.llm_w_tools.invoke(
                    messages, 
                    num_retries=0,
                    **self._get_llm_params_dict(),
                ),
            )
            # Recover tool calls from textual formats when provider does not return structured tool_calls.
            if not response.tool_calls and isinstance(response.content, str):
                recovered_calls = self._recover_tool_calls_from_text(response.content)
                if recovered_calls:
                    response.tool_calls = recovered_calls
                    logger.info(
                        "Agent %s recovered %s textual tool call(s).",
                        self.agent_id,
                        len(recovered_calls),
                    )

            # Tool Handling Logic (Matched with WorkerSubagent)
            if response.tool_calls and not no_more_tools:
                # Enforce single tool call policy
                tool_call = response.tool_calls[0]
                response.tool_calls = [tool_call] 
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {}) or {}
                tool_args = self._rewrite_tool_args_for_semantic_query(tool_name, tool_args, query)
                tool_call["args"] = tool_args
                if len(tool_calls_made) >= max_tool_calls_per_round:
                    logger.warning(
                        "Agent %s reached max tool calls per round (%s); stopping to avoid loops.",
                        self.agent_id,
                        max_tool_calls_per_round,
                    )
                    break
                if self._is_repeated_tool_call(tool_calls_made, tool_name, tool_args):
                    logger.warning(
                        "Agent %s repeated tool %s with same arguments; stopping to avoid loops.",
                        self.agent_id,
                        tool_name,
                    )
                    break
                tool_calls_made.append(
                    {
                        "iteration": iteration + 1,
                        "name": tool_name,
                        "arguments": tool_args,
                    }
                )
                tool_names_made.append(tool_name)
                no_progress_turns = 0

                # Add AIMessage to history
                response_msg = convert_to_openai_messages(response)
                messages.append(response_msg)
                self.conv_history.add_internal_message(response_msg, curr_iteration)
                
                try:
                    # Execute tool
                    tool_resp = self.env.execute_tool(tool_call)
                    
                    # Add ToolMessage to history
                    tool_msg = convert_to_openai_messages(tool_resp)
                    messages.append(tool_msg)
                    self.conv_history.add_internal_message(tool_msg, curr_iteration)
                    
                    # Termination check
                    if tool_name.lower() == "done":
                        final_answer = self._extract_tool_result_text(tool_msg.get("content", ""))
                        logger.info(f"Agent {self.agent_id} finished via 'done' tool")
                        break
                    if tool_name.lower() == "submit_final_result":
                        final_answer = self._extract_env_final_answer() or self._extract_tool_result_text(tool_msg.get("content", ""))
                        submitted_this_round = True
                        self._submitted_this_round = True
                        logger.info(
                            "Agent %s submitted final result in iteration %s; continuing this round.",
                            self.agent_id,
                            iteration + 1,
                        )
                        
                except Exception as e:
                    # Error Handling (Exact match with Subagent format)
                    error_content = f"ERROR: Tool **{tool_name}** failed with error: {str(e)}. Please check the tool call."
                    error_msg = {"role": "user", "content": error_content}
                    
                    messages.append(error_msg)
                    self.conv_history.add_internal_message(error_msg, curr_iteration)
                    logger.warning(f"Tool execution failed: {e}")
                    no_progress_turns += 1
                    tool_failures += 1
                    if tool_name in ("parse_html_page", "retrieve_information"):
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "The last tool call failed. Re-run web_search/edgar_search to get a valid URL "
                                    "or a correct document key, then retry parse_html_page and retrieve_information."
                                ),
                            }
                        )
                    if tool_failures >= 3:
                        logger.warning(
                            "Agent %s hit repeated tool failures; skipping further tool calls this round.",
                            self.agent_id,
                        )
                        no_more_tools = True
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You have hit the tool failure limit. Do NOT describe what you were trying to do "
                                    "or output planning statements. Instead, immediately summarize the concrete facts "
                                    "you have already retrieved (specific numbers, figures, sources). "
                                    "If you have retrieved nothing useful, state the best-effort answer given the "
                                    "task description and mark your confidence as low."
                                ),
                            }
                        )
            else:
                # No tool calls usually means we have an answer or the model wants to talk
                # If content is substantial, we treat it as the final answer
                response_msg = convert_to_openai_messages(response)
                messages.append(response_msg)
                self.conv_history.add_internal_message(response_msg, curr_iteration)
                
                if response.content:
                    candidate_text = str(response.content).strip()
                    if not self._looks_like_non_final_answer(candidate_text):
                        if self._has_tool_access() and not submitted_this_round and not no_more_tools:
                            if not tool_calls_made:
                                no_progress_turns += 1
                                if iteration + 1 < self.max_iterations_per_agent:
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": (
                                                "You have not used any tools yet. Do not stop at a plan. "
                                                "Continue working with the allowed tools, gather evidence, "
                                                "and only stop once you have extracted concrete facts or reached the safety limit."
                                            ),
                                        }
                                    )
                                    continue
                            elif not self._looks_like_completed_work(candidate_text):
                                no_progress_turns += 1
                                if iteration + 1 < self.max_iterations_per_agent:
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": (
                                                "Your current response is still incomplete or contains placeholders. "
                                                "Re-run search to get real URLs/keys, then extract evidence."
                                            ),
                                        }
                                    )
                                    continue

                        # Keep submitted answer as authoritative for this round.
                        if not submitted_this_round:
                            final_answer = candidate_text
                        logger.info(f"Agent {self.agent_id} provided text response.")
                        break
                else:
                    # Logic for "No tool calls found" warning if response is empty
                    error_msg = {"role": "user", "content": "ERROR: No tool calls found. Please use tools or output an answer."}
                    messages.append(error_msg)
                    self.conv_history.add_internal_message(error_msg, curr_iteration)
                    no_progress_turns += 1
                    if no_progress_turns >= 2:
                        logger.warning(
                            "Agent %s made no progress for multiple iterations; stopping this round.",
                            self.agent_id,
                        )
                        break

        # 3. Finalize
        # Extract answer from last message/tool if not set
        if not final_answer:
            final_answer = self._extract_env_final_answer()
            
        # Re-scan history for the last actual text generated by assistant or 'done' payload
        # This is a simplification; Subagent relies on a summarizing step or the 'done' tool payload.
        # Since we removed the "Summarize Findings" step (as it's orchestrator specific),
        # we try to grab the last Assistant content.
        if not final_answer:
            for m in reversed(messages):
                if m.get("role") == "assistant" and m.get("content"):
                    candidate_text = str(m["content"]).strip()
                    if not self._looks_like_non_final_answer(candidate_text):
                        final_answer = candidate_text
                        break
        
        if not final_answer:
            final_answer = "Failed to generate a valid answer."

        return SubAgentRoundResult(
            agent_id=self.agent_id,
            findings=str(final_answer),
            env_status=self.env.env_status(),
            tool_calls=tool_calls_made,
            tool_names=tool_names_made,
            total_iterations=curr_iteration,
        )

    def _recover_tool_calls_from_text(self, content: str) -> List[Dict[str, Any]]:
        """Best-effort recovery for XML/DSML style tool calls in plain text."""
        recovered: List[Dict[str, Any]] = []
        if not content:
            return recovered

        txt = content.strip()
        # Normalize escaped tags and full-width separators seen in DSML outputs.
        txt = txt.replace("\\<", "<").replace("\\>", ">")
        txt = txt.replace("｜", "|")

        # Format A: <tool_call>{"name":"...", "arguments": {...}}</tool_call>
        for block in re.findall(r"<tool_call>(.*?)</tool_call>", txt, flags=re.DOTALL):
            call = self._parse_json_tool_block(block)
            if call:
                recovered.append(call)

        # Format B: <function_calls><invoke name="..."><parameter ...>...</parameter></invoke></function_calls>
        # Also supports DSML variants: <|DSML|function_calls> ... <|DSML|invoke ...>
        invoke_pattern = r"<(?:\|DSML\|)?invoke\s+name=['\"]([^'\"]+)['\"]\s*>(.*?)</(?:\|DSML\|)?invoke>"
        for name, body in re.findall(invoke_pattern, txt, flags=re.DOTALL):
            args: Dict[str, Any] = {}
            param_pattern = r"<parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</parameter>"
            for p_name, p_val in re.findall(param_pattern, body, flags=re.DOTALL):
                raw_val = p_val.strip()
                args[p_name] = self._coerce_param_value(raw_val)
            recovered.append(
                {
                    "name": name.strip(),
                    "args": args,
                    "id": f"call_{uuid.uuid4().hex[:20]}",
                    "type": "tool_call",
                }
            )
        if not recovered and ("<function_calls" in txt.lower() or "<invoke " in txt.lower() or "|dsml|" in txt.lower()):
            logger.warning(
                "Agent %s produced textual function-calls but parser recovered none. Snippet: %s",
                self.agent_id,
                txt[:300],
            )
        return recovered

    def _parse_json_tool_block(self, block: str) -> Dict[str, Any] | None:
        clean = block.strip()
        if clean.startswith("```"):
            clean = clean.strip("`").replace("json", "").strip()
        try:
            data = json.loads(clean)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        args = data.get("arguments", data.get("args", {}))
        if args is None:
            args = {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        name = str(data.get("name", "")).strip()
        if not name:
            return None
        return {
            "name": name,
            "args": args,
            "id": f"call_{uuid.uuid4().hex[:20]}",
            "type": "tool_call",
        }

    def _coerce_param_value(self, raw_val: str) -> Any:
        # Try JSON first (for arrays/objects/numbers/bool), then fallback to string.
        v = raw_val.strip()
        if not v:
            return ""
        try:
            return json.loads(v)
        except Exception:
            return v

    def _get_llm_params_dict(self) -> Dict[str, Any]:
        """Retrieve param overrides if available, else empty."""
        # Typically managed by Hydra config passed to AgentSystem
        return {}
