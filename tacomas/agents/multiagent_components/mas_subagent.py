import threading
import traceback
from typing import List, Optional, cast

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages

from tacomas.agents.base import BaseAgentWithTools
from tacomas.datasets import DatasetInstance
from tacomas.logger import logger

from .conversation import SubAgentConversationHistory, SubAgentRoundResult
from .memory import EnhancedMemory


ROLE_PSEUDO_TOOLS = {
    "planner": ["decompose_task", "set_evidence_requirements"],
    "searcher": ["collect_sources", "extract_facts"],
    "calculator": ["derive_metrics", "check_units"],
    "verifier": ["cross_validate", "find_conflicts"],
    "summary": ["compose_answer", "format_sources"],
    "reflector": ["postmortem", "prompt_patch"],
}


class WorkerSubagent(BaseAgentWithTools):
    """Generic worker subagent that works with proper environment access"""

    required_prompts = ["subagent"]

    def __init__(
        self,
        agent_id: str,
        objective: str,
        original_query: str,
        strategy: str,
        role: str,
        allowed_tools: List[str],
        tool_budget: Optional[int],
        task_instance: DatasetInstance,
        memory: Optional[EnhancedMemory] = None,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        # Keep full prompt map (tacomas role templates) when available.
        incoming_prompts = kwargs.get("prompts")
        if isinstance(incoming_prompts, dict):
            self.prompts = incoming_prompts
        self.agent_id = agent_id
        # Core agent attributes
        self.objective = objective
        self.original_query = original_query
        self.strategy = strategy
        self.role = role
        self.allowed_tools = allowed_tools
        self.tool_budget = tool_budget
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.task_instance = task_instance
        self.memory = memory
        self.env, self.llm_w_tools = self.init_environment(
            task_instance,
            agent_id,
            tools_override=self.allowed_tools,
            tool_budget=self.tool_budget,
            role=self.role,
        )
        self.shared_prompt_templates = self.get_dataset_prompt_templates(self.env)
        # Conversation management
        self.conv_history = SubAgentConversationHistory(agent_id=agent_id)

        self._execution_lock = threading.Lock()

        logger.info(
            f"WorkerSubagent {agent_id} role={role} initialized with objective: {objective[:100]}..."
        )

    @classmethod
    def init_from_agent(
        cls,
        agent: BaseAgentWithTools,
        agent_id: str,
        objective: str,
        original_query: str,
        strategy: str,
        role: str,
        allowed_tools: List[str],
        tool_budget: Optional[int],
        memory: Optional[EnhancedMemory] = None,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        **kwargs,
    ):
        return cls(
            agent_id=agent_id,
            objective=objective,
            original_query=original_query,
            strategy=strategy,
            role=role,
            allowed_tools=allowed_tools,
            tool_budget=tool_budget,
            memory=memory,
            min_iterations_per_agent=min_iterations_per_agent,
            max_iterations_per_agent=max_iterations_per_agent,
            llm=agent.llm,
            dataset=agent.dataset,
            prompts=agent.prompts,
            tools=agent.tools,
            env=agent.env_name,
            env_prompts=agent.env_prompts,
            **kwargs,
        )

    def _get_role_prompt(self):
        """Return role-specific prompt if available, otherwise fallback to subagent."""
        if self.role in self.prompts:
            return self.prompts[self.role]
        return self.prompts["subagent"]

    def _build_role_note(self) -> str:
        pseudo_tools = ROLE_PSEUDO_TOOLS.get(self.role, [])
        return (
            "ROLE_NOTE\n"
            f"- role: {self.role}\n"
            f"- objective: {self.objective}\n"
            f"- strategy: {self.strategy}\n"
            f"- allowed_tools: {self.allowed_tools}\n"
            f"- forbidden_tools: {[t for t in ['prepare_primary_filing','web_search','edgar_search','parse_html_page','retrieve_information','submit_final_result'] if t not in self.allowed_tools]}\n"
            f"- tool_budget: {self.tool_budget}\n"
            f"- pseudo_tools: {pseudo_tools}\n"
            "- required_output_schema: facts/sources/confidence(or PASS/FAIL for verifier)"
        )

    def _get_start_messages(self, message: str):
        summary_context = self._build_summary_context(message)
        try:
            prompt_obj = self._get_role_prompt()
            template = prompt_obj.get_template("start_with_orchestrator_guidance", with_base=False)
            return template.compile(
                orchestrator_objective=self.objective,
                orchestrator_guidance=message,
                role_note=self._build_role_note(),
                summary_context=summary_context,
                **self.shared_prompt_templates,
            )
        except Exception:
            # fallback to legacy subagent flow
            template = self.prompts["subagent"].get_template("start_with_orchestrator_guidance")
            return template.compile(
                orchestrator_objective=self.objective,
                orchestrator_guidance=message,
                role_note=self._build_role_note(),
                summary_context=summary_context,
                **self.shared_prompt_templates,
            )

    def _get_summary_prompt(self):
        try:
            prompt_obj = self._get_role_prompt()
            return prompt_obj.get_template("summarize_findings", with_base=False)
        except Exception:
            return self.prompts["subagent"].get_template("summarize_findings", with_base=False)

    def _build_summary_context(self, workflow: str) -> str:
        previous_summary = self.memory.get_latest_summary(self.agent_id) if self.memory else ""
        current_workflow = workflow.strip()

        if previous_summary:
            return (
                "PREVIOUS_SUMMARY\n"
                f"{previous_summary}\n\n"
                "CURRENT_WORKFLOW\n"
                f"{current_workflow}"
            )

        return (
            "PREVIOUS_SUMMARY\n"
            "None. This is the first activation for this agent.\n\n"
            "CURRENT_WORKFLOW\n"
            f"{current_workflow}\n\n"
            "FIRST_SUMMARY_RULE\n"
            "The first summary must record what this agent did in the current activation."
        )

    def _validate_findings_schema(self, findings_text: str) -> str:
        """Light schema guard for verifier/summary outputs."""
        upper_txt = findings_text.upper()
        if self.role == "verifier":
            if "PASS" not in upper_txt and "FAIL" not in upper_txt:
                return (
                    "VERDICT: FAIL\n"
                    "Conflicts found: Missing explicit PASS/FAIL verdict from verifier output.\n"
                    "Needs re-check: add explicit pass/fail and evidence list.\n"
                    f"Raw findings:\n{findings_text}"
                )
        if self.role == "summary":
            if "sources" not in findings_text.lower():
                return findings_text + "\n\n{\"sources\": []}"
        return findings_text

    def _run_one_round(self, message: str) -> SubAgentRoundResult:
        """Process the task by calling tools with proper conversation state management"""
        import re
        import json
        import uuid
        
        logger.info(f"Agent {self.agent_id} starting process")
        logger.info(f"Agent {self.agent_id} objective: {self.objective}")

        # Compile role-specific prompt (with fallback to legacy subagent template)
        messages = self._get_start_messages(message)
        # Continue from previous conversation state if exists and valid
        if len(self.conv_history.internal_comms) > 0:
            logger.info(
                f"Agent {self.agent_id} continuing from previous conversation state with {len(self.conv_history.internal_comms)} messages"
            )
            last_n_messages = self.conv_history.last_n_iterations_messages(n=20)
            messages.extend(last_n_messages)
        
        # === [INSERT FIX HERE] 历史记录自愈逻辑 ===
            # 检查加载的历史记录最后一条是否是未完成的 Assistant tool_call
            if messages and isinstance(messages[-1], dict):
                last_msg = messages[-1]
                # 如果最后一条是 Assistant 且包含 tool_calls
                if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
                    logger.warning(f"Agent {self.agent_id}: Found dangling tool call in history load. Repairing...")
                    
                    # 策略：追加一个虚拟的 System/Tool 错误消息来闭环
                    # 这样 API 就不会报错 "tool_calls must be followed by tool messages"
                    for tc in last_msg["tool_calls"]:
                        dummy_tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc.get("name", "unknown"),
                            "content": "System Error: Tool execution result was lost in conversation history transition. Please ignore this tool call and proceed."
                        }
                        messages.append(dummy_tool_msg)
                        # 注意：这里我们只修复了当前上下文的 messages 用于本次 invoke
                        # 不必强求写回 conv_history，因为只要通过了这次 invoke，后续的新消息会覆盖这个断点
            # ===========================================

        # Simple iteration loop with conversation persistence
        curr_iteration = 0
        for iteration in range(self.max_iterations_per_agent):
            curr_iteration += 1
            logger.info(
                f"Agent {self.agent_id} iteration {iteration}/{self.max_iterations_per_agent}"
            )

                
            # Invoke LLM with tools using retry logic
            try:
                response: AIMessage = cast(
                    AIMessage,
                    self.llm_w_tools.invoke(
                        messages,  # type: ignore
                        num_retries=2,
                        **self._get_llm_params_dict(),
                    ),
                )
            except Exception as e:
                logger.error(f"LLM invoke failed: {e}")
                break

            # === FIX: Qwen/DeepSeek XML Output Patch ===
            # If no structured tool_calls found but XML tags exist in content
            if not response.tool_calls and response.content and isinstance(response.content, str):
                xml_matches = re.findall(r"<tool_call>(.*?)</tool_call>", response.content, re.DOTALL)
                if xml_matches:
                    fixed_calls = []
                    for xml_content in xml_matches:
                        try:
                            clean_json = xml_content.strip()
                            if clean_json.startswith("```"):
                                clean_json = clean_json.strip("`").replace("json", "").strip()
                            data = json.loads(clean_json)
                            args_val = data.get("arguments", data.get("args", {}))
                            if args_val is None:
                                args_val = {}
                            if isinstance(args_val, str):
                                try:
                                    args_val = json.loads(args_val)
                                except Exception:
                                    args_val = {}
                            fixed_calls.append(
                                {
                                    "name": data.get("name"),
                                    "args": args_val,
                                    "id": f"call_{uuid.uuid4().hex[:20]}",
                                    "type": "tool_call",
                                }
                            )
                        except Exception:
                            pass
                    if fixed_calls:
                        response.tool_calls = fixed_calls
                        logger.info(f"Patched {len(fixed_calls)} XML tool calls")

            # Add Assistant Message (with tool_calls if any) to history
            response_msg = convert_to_openai_messages(response)
            messages.append(response_msg)  # type: ignore
            self.conv_history.add_internal_message(
                message=response_msg,  # type: ignore
                iteration_num=curr_iteration,
            )

            # === FIX: Handle ALL Tool Calls (Parallel & Robust) ===
            if response.tool_calls:
                
                tool_results_to_add = []
                
                for tool_call in response.tool_calls:
                    tool_id = tool_call["id"]
                    tool_name = tool_call.get("name", "unknown")
                    
                    try:
                        # Execute tool
                        tool_resp = self.env.execute_tool(tool_call)
                        
                        # Convert to message (LangChain handles tool_call_id automatically for ToolMessage)
                        tool_msg = convert_to_openai_messages(tool_resp)
                        
                        # Safety check: Ensure it's a dict or ToolMessage with correct ID
                        if isinstance(tool_msg, dict) and "tool_call_id" not in tool_msg:
                             tool_msg["tool_call_id"] = tool_id
                        
                        tool_results_to_add.append(tool_msg)

                    except Exception as e:
                        # CRITICAL FIX: Even on error, MUST return a tool message with the same ID
                        # to prevent "hanging" tool calls which cause API 400 errors
                        error_content = f"ERROR: Tool **{tool_name}** failed: {str(e)}"
                        error_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],  # <--- MUST MATCH
                            "name": tool_name,
                            "content": error_content,
                        }
                        tool_results_to_add.append(error_msg)
                        logger.warning(f"Tool {tool_name} failed: {e}")

                # Batch append results to history
                for msg in tool_results_to_add:
                    messages.append(msg) # type: ignore
                    self.conv_history.add_internal_message(msg, iteration_num=curr_iteration)
                
                # Check termination conditions
                if any(tc.get("name") == "done" for tc in response.tool_calls):
                    logger.info(f"Agent {self.agent_id} decided to finish with 'done' tool")
                    break
                elif self.env.env_done():
                    logger.info(f"Environment {self.env_name} is done for agent {self.agent_id}")
                    break
            
            else:
                # No tool calls found
                error_msg = {
                    "role": "user",
                    "content": "ERROR: No tool calls found. Please use the tools to solve the task.",
                }
                messages.append(error_msg)  # type: ignore
                self.conv_history.add_internal_message(
                    message=error_msg,  # type: ignore
                    iteration_num=curr_iteration,
                )
                logger.warning(
                    f"Agent {self.agent_id}: No tool calls found in iteration {iteration}"
                )

        # === Summarize Findings ===
        curr_iteration += 1
                
        # === [CRITICAL FIX] Sanitize history before Summarize ===
        # OpenAI forbids: Assistant(tool_calls) -> User(...)
        # We must ensure the last message is NOT an assistant message with pending tool_calls.
        if messages:
            last_msg = messages[-1]
            pending_ids = []
            
            # 识别悬挂调用
            if isinstance(last_msg, dict):
                if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
                    pending_ids = [tc["id"] for tc in last_msg["tool_calls"]]
            elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls: # type: ignore
                pending_ids = [tc.id for tc in last_msg.tool_calls] # type: ignore

            # 强制闭环
            if pending_ids:
                logger.warning(f"Agent {self.agent_id}: Closing {len(pending_ids)} pending tool calls before summary.")
                for tid in pending_ids:
                    dummy_msg = {
                        "role": "tool",
                        "tool_call_id": tid,
                        "name": "system_interrupt",
                        "content": "System: Tool execution interrupted because the agent is moving to the summary phase."
                    }
                    # 同时更新局部 messages 和 全局 history
                    messages.append(dummy_msg) # type: ignore
                    self.conv_history.add_internal_message(dummy_msg, iteration_num=curr_iteration) # type: ignore


        
        # Also check strict object types if your messages list contains objects
        elif messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
                logger.warning(f"Agent {self.agent_id}: Found pending tool calls (obj) before summary. Stripping.")
                messages[-1].tool_calls = []

        # ========================================================

        
        # Robust prompt compilation
        findings_template = self._get_summary_prompt()
        findings_prompt_res = findings_template.compile(
             orchestrator_objective=self.objective,
             summary_context=self._build_summary_context(message),
             **self.shared_prompt_templates
        )
        
        # Handle list vs single message return type
        findings_message = findings_prompt_res[0] if isinstance(findings_prompt_res, list) else findings_prompt_res

        messages.append(findings_message)  # type: ignore
        self.conv_history.add_internal_message(
            message=findings_message,  # type: ignore
            iteration_num=curr_iteration,
        )

        llm_response = self.llm.invoke(messages)

        # Update agent's conversation
        self.conv_history.add_internal_message(
            message=convert_to_openai_messages(llm_response),  # type: ignore
            iteration_num=curr_iteration,
        )
        
        # Use .content for reliability + role schema guard
        findings_text = self._validate_findings_schema(str(llm_response.content))
        
        logger.info(
            f"Agent {self.agent_id} completed round {self.conv_history.current_round}. Findings saved."
        )

        return SubAgentRoundResult(
            agent_id=self.agent_id,
            findings=findings_text,
            env_status=self.env.env_status(),
        )

    '''
    def _run_one_round(self, message: str) -> SubAgentRoundResult:
        """Process the task by calling tools with proper conversation state management"""
        logger.info(f"Agent {self.agent_id} starting process")
        logger.info(f"Agent {self.agent_id} objective: {self.objective}")

        # Compile role-specific prompt (with fallback to legacy subagent template)
        messages = self._get_start_messages(message)
        # Continue from previous conversation state if exists and valid
        if len(self.conv_history.internal_comms) > 0:
            logger.info(
                f"Agent {self.agent_id} continuing from previous conversation state with {len(self.conv_history.internal_comms)} messages"
            )
            last_n_messages = self.conv_history.last_n_iterations_messages(n=20)
            messages.extend(last_n_messages)

        # Simple iteration loop with conversation persistence
        curr_iteration = 0
        for iteration in range(self.max_iterations_per_agent):
            curr_iteration += 1
            logger.info(
                f"Agent {self.agent_id} iteration {iteration}/{self.max_iterations_per_agent}"
            )
            # Invoke LLM with tools using retry logic
            response: AIMessage = cast(
                AIMessage,
                self.llm_w_tools.invoke(
                    messages,  # type: ignore
                    num_retries=2,
                    **self._get_llm_params_dict(),
                ),
            )


                        

                        尝试修改
            for iteration in range(self.max_iterations_per_agent):
                # ...
                response = self.llm_w_tools.invoke(...)
                
                # === [Step 1] 补救：检查 Raw XML Tool Calls ===
                # 如果 tool_calls 为空，但内容里有 XML，手动提取
                if not response.tool_calls and isinstance(response.content, str):
                    xml_matches = re.findall(r"<tool_call>(.*?)</tool_call>", response.content, re.DOTALL)
                    if xml_matches:
                        fixed_tool_calls = []
                        for xml_content in xml_matches:
                            try:
                                # 清理 markdown
                                clean_json = xml_content.strip()
                                if clean_json.startswith("```"):
                                    clean_json = clean_json.strip("`").replace("json", "").strip()
                                
                                data = json.loads(clean_json)
                                args_val = data.get("arguments", data.get("args", {}))
                                if args_val is None:
                                    args_val = {}
                                if isinstance(args_val, str):
                                    try:
                                        args_val = json.loads(args_val)
                                    except Exception:
                                        args_val = {}
                                # 构造伪造 ID
                                fixed_tool_calls.append({
                                    "name": data.get("name"),
                                    "args": args_val,
                                    "id": f"call_{uuid.uuid4().hex[:20]}",
                                    "type": "tool_call"
                                })
                            except Exception as e:
                                logger.warning(f"Failed to parse XML tool call: {e}")
                        
                        if fixed_tool_calls:
                            response.tool_calls = fixed_tool_calls
                            logger.info(f"Patched {len(fixed_tool_calls)} XML tool calls")

                # 将 Assistant 消息加入历史
                response_msg = convert_to_openai_messages(response)
                messages.append(response_msg)
                # ... update conv_history ...

                # === [Step 2] 安全执行 Tool Calls ===
                if response.tool_calls:
                    # 遍历处理每一个 tool call，绝不能漏
                    for tc in response.tool_calls:
                        tool_id = tc["id"]
                        tool_name = tc.get("name", "unknown")
                        
                        try:
                            # 执行
                            raw_output = self.env.execute_tool(tc)
                            
                            # 构造 Tool Message
                            # 确保一定包含 tool_call_id
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": str(raw_output)
                            }
                            
                        except Exception as e:
                            # 每一处异常都捕捉，并生成 Error Tool Message
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": tool_id, # 关键：即使报错也要回传 ID
                                "name": tool_name,
                                "content": f"Error: {str(e)}"
                            }
                        
                        # 放入 messages
                        messages.append(tool_msg)
                else:
                    # Here we CAN use role="user" because there was no pending tool_call
                    error_msg = {
                        "role": "user",
                        "content": "ERROR: No tool calls found. Please use the tools to solve the task.",
                    }
                    messages.append(error_msg)  # type: ignore
                    self.conv_history.add_internal_message(
                        message=error_msg,  # type: ignore
                        iteration_num=curr_iteration,
                    )
                    logger.warning(
                        f"Agent {self.agent_id}: No tool calls found in iteration {iteration}"
                    )
                    修改结束




    '''

    '''
            # 原版
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                response.tool_calls = [response.tool_calls[0]]
                tool_name = ""

                response_msg = convert_to_openai_messages(response)
                messages.append(response_msg)  # type: ignore
                self.conv_history.add_internal_message(
                    message=response_msg,  # type: ignore
                    iteration_num=curr_iteration,
                )
                
                
                try:
                    # Execute tool with retry logic
                    tool_name = tool_call["name"]
                    tool_resp = self.env.execute_tool(tool_call)
                except Exception as e:
                    # Add error message to conversation state (consistent with single_agent.py)
                    error_msg = {
                        "role": "user",
                        "content": f"ERROR: Tool **{tool_name}** failed with error: {str(e)}. Please check the tool call.",
                    }
                    messages.append(error_msg)  # type: ignore
                    self.conv_history.add_internal_message(
                        message=error_msg,
                        iteration_num=curr_iteration,  # type: ignore
                    )
                    logger.warning(
                        f"Tool **{tool_name}** failed with error: {str(e)}\n{traceback.format_exc()}"
                    )
                
                else:
                    # Add tool response to messages and state
                    tool_msg = convert_to_openai_messages(tool_resp)
                    messages.append(tool_msg)  # type: ignore
                    self.conv_history.add_internal_message(
                        message=tool_msg,  # type: ignore
                        iteration_num=curr_iteration,
                    )
                    # Check if done
                    if tool_name == "done":
                        logger.info(
                            f"Agent {self.agent_id} decided to finish with 'done' tool"
                        )
                        break
                    elif self.env.env_done():
                        logger.info(
                            f"Environment {self.env_name} is done for agent {self.agent_id}"
                        )
                        break
            else:
                # No tool calls - add error message to conversation state
                error_msg = {
                    "role": "user",
                    "content": "ERROR: No tool calls found. Please use the tools to solve the task.",
                }
                messages.append(error_msg)  # type: ignore
                self.conv_history.add_internal_message(
                    message=error_msg,  # type: ignore
                    iteration_num=curr_iteration,
                )
                logger.warning(
                    f"Agent {self.agent_id}: No tool calls found in iteration {iteration}"
                )
    '''
    '''
        curr_iteration += 1
        findings_message = (
            self.prompts["subagent"]
            .get_template("summarize_findings", with_base=False)
            .compile()
        )[0]
        messages.append(findings_message)  # type: ignore
        self.conv_history.add_internal_message(
            message=findings_message,  # type: ignore
            iteration_num=curr_iteration,
        )

        llm_response = self.llm.invoke(messages)

        # Update agent's conversation
        self.conv_history.add_internal_message(
            message=convert_to_openai_messages(llm_response),  # type: ignore
            iteration_num=curr_iteration,
        )
        logger.info(
            f"Agent {self.agent_id} completed round {self.conv_history.current_round} (n_iterations={self.conv_history.curr_iteration}). Findings:\n{llm_response.text()} "
        )

        return SubAgentRoundResult(
            agent_id=self.agent_id,
            findings=llm_response.text(),
            env_status=self.env.env_status(),
        )
    '''

    def process_orchestrator_message(self, message: str) -> SubAgentRoundResult:
        """Process a message from orchestrator with conversation persistence and proper error handling"""
        # Increment round counter for new orchestrator message
        self.conv_history.start_new_round()
        logger.info(
            f"Agent {self.agent_id} ({self.strategy}) processing message for round {self.conv_history.current_round}"
        )
        self.conv_history.add_external_message("lead_agent", message)
        round_result = self._run_one_round(message)
        self.conv_history.add_external_message("subagent", round_result.findings)

        if self.env.env_done():
            self.conv_history.status = "completed"
        elif self.should_stop_due_to_rate_limiting():
            self.conv_history.status = "rate_limited"

        return round_result

    def should_stop_due_to_rate_limiting(self) -> bool:
        """Check if agent should stop due to excessive rate limiting"""
        if hasattr(self.env, "should_stop_due_to_rate_limiting"):
            return self.env.should_stop_due_to_rate_limiting()
        return False
