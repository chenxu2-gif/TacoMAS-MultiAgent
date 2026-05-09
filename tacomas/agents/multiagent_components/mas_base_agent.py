import threading
import traceback
from typing import cast, Dict, Any, List

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages

from tacomas.agents.base import BaseAgentWithTools
from tacomas.datasets import DatasetInstance
from tacomas.logger import logger

# 复用 SubAgent 的数据结构
from .conversation import SubAgentConversationHistory, SubAgentRoundResult


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
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.task_instance = task_instance
        
        # Initialize environment and tools
        self.env, self.llm_w_tools = self.init_environment(task_instance, agent_id)
        
        # Shared templates (e.g. valid_tools, task_rule)
        self.shared_prompt_templates = self.get_dataset_prompt_templates(self.env)
        
        self.conv_history = SubAgentConversationHistory(agent_id=agent_id)
        self._execution_lock = threading.Lock()

        logger.info(f"BaseWorkerAgent {agent_id} initialized.")

    @classmethod
    def init_from_agent(cls, agent: BaseAgentWithTools, agent_id: str, task_instance: DatasetInstance, **kwargs):
        return cls(
            agent_id=agent_id,
            task_instance=task_instance,
            llm=agent.llm,
            dataset=agent.dataset,
            prompts=agent.prompts,
            tools=agent.tools,
            env=agent.env_name,
            env_prompts=agent.env_prompts,
            **kwargs,
        )

        
    def solve_task(self, query: str) -> SubAgentRoundResult:
        """
        Entry point for the worker to solve a task.
        """
        # === [新增关键行] 在开始任务前，必须初始化新的一轮对话 ===
        # 这会在 internal_comms 里 append 一个空列表 []，避免后续 index out of range
        self.conv_history.start_new_round()

        # [Fix 1] convert "user" to "lead_agent" to satisfy Pydantic Literal constraint
        self.conv_history.add_external_message("lead_agent", query)

        # Run the execution loop (ReAct / Chain of Thought)
        final_result = self._run_execution_loop(query)

        # [Fix 2] Ensure env_status is not None
        current_env_status = None
        if hasattr(self, "env") and self.env is not None:
                current_env_status = getattr(self.env, "get_status", lambda: {})()
        
        if current_env_status is None:
            current_env_status = {}

        # [Change 1] final_result.response -> final_result.findings
        self.conv_history.add_external_message("subagent", final_result.findings)

        # [Change 2] 构造返回值时，使用正确的字段名 (findings)
        # 注意：这里我们直接利用 final_result 的数据来构建，或者直接返回 final_result 并更新其 env_status
        
        # 简单且安全的方法：直接返回 final_result (因为它已经是 SubAgentRoundResult 类型)
        # 但我们需要确保 env_status 是更新过的
        final_result.env_status = current_env_status
        return final_result


        '''
        # (可选优化) 模仿 subagent，将最终结果也记录到 external 历史中，保持历史完整性
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
            # === [Fix] Reliable Prompt Loading ===
            # 从 Log 可知 p_obj 是 Prompt 类，拥有 named_templates 属性
            # 且 named_templates 包含 'base', 'aggregator' 等键
            prompt_container = self.prompts["base_agent"]
            
            # 直接从 named_templates 获取 'base' 对应的 NamedPrompt 对象
            if hasattr(prompt_container, "named_templates") and "base" in prompt_container.named_templates:
                base_named_prompt = prompt_container.named_templates["base"]
            else:
                # 理论上不会发生，因为我们看到了 debug log
                raise ValueError(f"Template 'base' not found in fields: {prompt_container.named_templates.keys()}")

            # 准备模版变量
            template_kwargs = {
                "task_instance": query,
                **self.shared_prompt_templates 
            }
            
            # 编译逻辑
            # 根据 Log，Prompt 类本身有 compile 方法，可能 NamedPrompt 也有
            # 为了保险，我们手动检查 prompt_template 属性并渲染
            if hasattr(base_named_prompt, "prompt_template") and base_named_prompt.prompt_template:
                raw_list = base_named_prompt.prompt_template
                compiled_messages = []
                for item in raw_list:
                    content = item.get("content", "")
                    role = item.get("role", "user")
                    # 手动执行替换
                    for k, v in template_kwargs.items():
                        if v and isinstance(v, str):
                            content = content.replace(f"{{{{{k}}}}}", v)
                    compiled_messages.append({"role": role, "content": content})
            elif hasattr(base_named_prompt, "compile"):
                # 如果有 compile 方法，尝试使用它
                compiled_obj = base_named_prompt.compile(**template_kwargs)
                compiled_messages = [convert_to_openai_messages(m) if not isinstance(m, dict) else m for m in compiled_obj]
            else:
                    raise ValueError(f"NamedPrompt 'base' has no format/compile/template: {dir(base_named_prompt)}")

            # 最终确认格式
            messages = compiled_messages

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
        curr_iteration = 0
        
        for iteration in range(self.max_iterations_per_agent):
            curr_iteration += 1
            logger.info(f"Agent {self.agent_id} iteration {iteration+1}/{self.max_iterations_per_agent}")
            
            # Invoke LLM
            response: AIMessage = cast(
                AIMessage,
                self.llm_w_tools.invoke(
                    messages, 
                    num_retries=2,
                    **self._get_llm_params_dict(),
                ),
            )

            # Tool Handling Logic (Matched with WorkerSubagent)
            if response.tool_calls:
                # Enforce single tool call policy
                tool_call = response.tool_calls[0]
                response.tool_calls = [tool_call] 
                tool_name = tool_call["name"]

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
                        logger.info(f"Agent {self.agent_id} finished via 'done' tool")
                        break
                    elif self.env.env_done():
                        logger.info(f"Environment done for agent {self.agent_id}")
                        break
                        
                except Exception as e:
                    # Error Handling (Exact match with Subagent format)
                    error_content = f"ERROR: Tool **{tool_name}** failed with error: {str(e)}. Please check the tool call."
                    error_msg = {"role": "user", "content": error_content}
                    
                    messages.append(error_msg)
                    self.conv_history.add_internal_message(error_msg, curr_iteration)
                    logger.warning(f"Tool execution failed: {e}")
            else:
                # No tool calls usually means we have an answer or the model wants to talk
                # If content is substantial, we treat it as the final answer
                response_msg = convert_to_openai_messages(response)
                messages.append(response_msg)
                self.conv_history.add_internal_message(response_msg, curr_iteration)
                
                if response.content:
                    final_answer = response.content
                    logger.info(f"Agent {self.agent_id} provided text response.")
                    break
                else:
                    # Logic for "No tool calls found" warning if response is empty
                    error_msg = {"role": "user", "content": "ERROR: No tool calls found. Please use tools or output an answer."}
                    messages.append(error_msg)
                    self.conv_history.add_internal_message(error_msg, curr_iteration)

        # 3. Finalize
        # Extract answer from last message/tool if not set
        if not final_answer and messages:
            # Try to find the last meaningful assistant message or tool output
            # For simplicity, if we broke out of loop, we assume the history contains the answer logic
            # Many "done" tools return the answer in the tool output too.
            pass 
            
        # Re-scan history for the last actual text generated by assistant or 'done' payload
        # This is a simplification; Subagent relies on a summarizing step or the 'done' tool payload.
        # Since we removed the "Summarize Findings" step (as it's orchestrator specific),
        # we try to grab the last Assistant content.
        if not final_answer:
            for m in reversed(messages):
                if m.get("role") == "tool" and "done" in str(m.get("content", "")).lower():
                     final_answer = m["content"]
                     break
                if m.get("role") == "assistant" and m.get("content"):
                    final_answer = m["content"]
                    break
        
        if not final_answer:
            final_answer = "Failed to generate a valid answer."

        return SubAgentRoundResult(
            agent_id=self.agent_id,
            findings=str(final_answer),
            env_status=self.env.env_status(),
        )

    def _get_llm_params_dict(self) -> Dict[str, Any]:
        """Retrieve param overrides if available, else empty."""
        # Typically managed by Hydra config passed to AgentSystem
        return {}