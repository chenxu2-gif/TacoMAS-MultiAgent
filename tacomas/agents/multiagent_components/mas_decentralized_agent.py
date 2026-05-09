from typing import Dict, Any, List, Optional
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from tacomas.agents.multiagent_components.mas_base_agent import BaseWorkerAgent, SubAgentRoundResult
from tacomas.logger import logger
from langchain_core.messages.utils import convert_to_openai_messages
from tacomas.agents.multiagent_components.conversation import SubAgentConversationHistory


class DecentralizedWorkerAgent(BaseWorkerAgent):
    def __init__(self, agent_id: str, shared_memory_ref: Dict[str, List[str]], task_instance, **kwargs):
        # 显式初始化 Base 属性
        super().__init__(agent_id=agent_id, task_instance=task_instance,**kwargs)
        
        self.agent_id = agent_id
        self.shared_memory = shared_memory_ref
        self.task_instance = task_instance
        
        # [一致性修改 1] 使用标准的 init_environment
        self.env, self.llm_w_tools = self.init_environment(task_instance, agent_id)
        self.shared_prompt_templates = self.get_dataset_prompt_templates(self.env)
        
        # [一致性修改 2] 使用 SubAgentConversationHistory 替代 self.messages
        self.conv_history = SubAgentConversationHistory(agent_id=agent_id)
        
        self.is_finished = False
        self.final_response = None
        self.env_status = {}

    @classmethod
    def init_from_agent(cls, agent, agent_id, shared_memory_ref, task_instance, **kwargs):
        # 保持与 WorkerSubagent 类似的 Factory Pattern
        return cls(
            agent_id=agent_id,
            shared_memory_ref=shared_memory_ref,
            task_instance=task_instance,
            llm=agent.llm,
            dataset=agent.dataset,
            prompts=agent.prompts,
            tools=agent.tools,
            env=agent.env_name,
            env_prompts=agent.env_prompts,
            **kwargs
        )

    def _initialize_prompt(self, query: str):
        """
        初始化初始 Prompt。
        复用之前的 Prompt Loading 修复逻辑，但这次我们将结果存入 self.messages 而不是直接发送。
        """
        try:
            # 1. 加载 Template (使用之前的健壮逻辑)
            prompt_container = self.prompts["base_agent"]
            base_template = None
            
            if hasattr(prompt_container, "named_templates") and "base" in prompt_container.named_templates:
                base_template = prompt_container.named_templates["base"]
            elif isinstance(prompt_container, dict) and "base" in prompt_container:
                base_template = prompt_container["base"]
            
            if not base_template:
                # Fallback: 假设 container 本身就是 template
                base_template = prompt_container

            # 2. 准备变量
            template_kwargs = {
                "task_instance": query,
                "agent_name": self.agent_id,
                **self.shared_prompt_templates 
            }
            
            # 3. 渲染
            compiled_messages = []
            if hasattr(base_template, "format_messages"):
                compiled_messages = base_template.format_messages(**template_kwargs)
            else:
                # 手动渲染逻辑 (Fallback)
                raw_list = getattr(base_template, "prompt_template", [])
                for item in raw_list:
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    for k, v in template_kwargs.items():
                        if v and isinstance(v, str):
                            content = content.replace(f"{{{{{k}}}}}", v)
                    
                    if role == "system":
                        compiled_messages.append(SystemMessage(content=content))
                    elif role == "user":
                        compiled_messages.append(HumanMessage(content=content))
            
            self.messages = compiled_messages
            
        except Exception as e:
            logger.error(f"Error initializing decentralized prompt: {e}")
            self.messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=query)
            ]


    
    def step(self, query: str, round_idx: int) -> SubAgentRoundResult:
        import re, uuid # 确保引入
        
        if self.is_finished:
            return SubAgentRoundResult(
                response=self.final_response,
                env_status=self.env_status,
                status="success",
                is_final=True
            )

        # 1. Start / System Prompt
        # 修改逻辑：如果是第0轮且历史为空，调用 _initialize_prompt，它会负责填充 self.conv_history
        if round_idx == 0 and not self.conv_history.internal_comms:
             self._initialize_prompt(query)
             
        # 初始化 messages 列表
        messages = []
            
        # 2. 从 History 加载上下文 & [缺失补全] History Repair
        if len(self.conv_history.internal_comms) > 0:
            last_n = self.conv_history.last_n_iterations_messages(n=15)
            
            # [Fix] History Repair (Copy from Centralized)
            if last_n and isinstance(last_n[-1], dict) or hasattr(last_n[-1], "tool_calls"):
                last_msg = last_n[-1]
                has_tc = False
                if isinstance(last_msg, dict): has_tc = (last_msg.get("role") == "assistant" and last_msg.get("tool_calls"))
                else: has_tc = (getattr(last_msg, "tool_calls", None))
                
                if has_tc:
                    # Append dummy to close broken chain
                    dummy = {"role": "tool", "tool_call_id": "repair", "name": "system", "content": "Error: Missing history."}
                    if hasattr(last_msg, "tool_calls"): # Object
                        ids = [t.id for t in last_msg.tool_calls]
                        for i in ids: 
                            last_n.append(ToolMessage(tool_call_id=i, content="Error: History repair.", name="system"))
                    else: # Dict
                         for t in last_msg["tool_calls"]:
                             last_n.append({"role": "tool", "tool_call_id": t["id"], "name": "system", "content": "Repair"})

            messages.extend(last_n)

        # 3. [修复] 注入 Peer Info (作为临时 Ephemeral Context)
        peer_updates = []
        for aid, history in self.shared_memory.items():
            if aid != self.agent_id and history: # 不包含自己
                last_msg = history[-1]
                # 过滤掉 Debate 阶段的消息干扰 Exploration
                if not last_msg.startswith("[Debate"):
                    peer_updates.append(f"- Agent {aid}: {last_msg}")
        
        if peer_updates and round_idx > 0:
            note_content = (f"[System Notification - Round {round_idx}]\n"
                            f"Updates from your teammates:\n"
                            + "\n".join(peer_updates) + 
                            "\n\nUse this information to adjust your plan if necessary, but focus on your own sub-tasks.")
            
            # 关键修改：只添加到发送给 LLM 的 messages 列表中
            # 不要调用 self.conv_history.add_internal_message(...) 
            messages.append(HumanMessage(content=note_content))

        # 4. LLM Invoke
        try:
            response = self.llm_w_tools.invoke(messages)
        except Exception as e:
            logger.error(f"LLM invoke failed: {e}")
            return SubAgentRoundResult(
                response=f"Error: {str(e)}",
                agent_id=self.agent_id,
                status="error",
                is_final=False,
                findings="",                  # 修复：改为字符串
                env_status={"success": False, "num_steps": round_idx}  # 修复：补全必须字段
            )

        # [缺失补全] XML Patch Logic
        if not response.tool_calls and isinstance(response.content, str):
            xml_matches = re.findall(r"<tool_call>(.*?)</tool_call>", response.content, re.DOTALL)
            if xml_matches:
                fixed_calls = []
                for xml_c in xml_matches:
                    try:
                        clean = xml_c.strip().strip("`").replace("json", "")
                        data = json.loads(clean)
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
                                "id": f"call_{uuid.uuid4().hex[:10]}",
                                "type": "tool_call",
                            }
                        )
                    except: pass
                if fixed_calls: response.tool_calls = fixed_calls

        self.conv_history.add_internal_message(response, iteration_num=round_idx)
        
        step_summary = "Thinking..."

        # 5. [重写] Tool Execution Logic (必须实际执行)
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc.get("name")
                tool_id = tc.get("id")
                
                step_summary = f"Used tool {tool_name}"
                
                # Special Handle
                if tool_name in ["submit_final_result", "done"]:
                    self.is_finished = True
                    args = tc.get("args", {})
                    self.final_response = args.get("final_result", str(args))
                    self.env_status = {"success": True}
                    tool_output = "Submitted final result."
                    self.is_finished = True 
                else:
                    # [缺失补全] 实际执行工具
                    try:
                        tool_output = self.env.execute_tool(tc)
                    except Exception as e:
                        tool_output = f"Error executing tool {tool_name}: {e}"
                
                # Construct Tool Message
                tool_msg = ToolMessage(tool_call_id=tool_id, content=str(tool_output), name=tool_name)
                self.conv_history.add_internal_message(tool_msg, iteration_num=round_idx)

        else:
            step_summary = f"Reasoning: {response.content[:50]}..."

        self.shared_memory[self.agent_id].append(step_summary)

        return SubAgentRoundResult(
            response=str(self.final_response) if self.final_response else response.content,
            env_status=self.env_status,
            status="success" if self.is_finished else "running",
            is_final=self.is_finished
        )


    def debate(self, all_peer_solutions: List[str], current_round: int) -> str:
        """
        Phase 2: Debate Step (纯 LLM 推理)
        此方法不再硬编码 Prompt，而是从 YAML 调用 'debate' 模板
        """
        # 如果之前没结束，先假设当前的最后一条消息是自己的结论
        if not self.final_response:
                last_content = "No clear conclusion reached yet."
                # 尝试从 History 寻找最近的 AIMessage
                if self.conv_history.internal_comms:
                     last_valid = [m for m in self.conv_history.internal_comms if isinstance(m, AIMessage)]
                     if last_valid: last_content = last_valid[-1].content
                self.final_response = last_content

        current_solution = self.final_response

        # [修改] 使用我们刚才定义的 YAML 模板
        debate_template = self.prompts["base_agent"].get_template("debate")
        
        # 准备变量
        debate_kwargs = {
            "current_round_display": current_round + 1,
            "current_solution": current_solution,
            "peer_solutions": self._format_peer_solutions(all_peer_solutions),
            **self.shared_prompt_templates
        }
        
        # 编译并生成 HumanMessage
        debate_prompts = debate_template.compile(**debate_kwargs)
        
        # 将生成的 Debate Prompt 加入历史
        # 注意: compile 返回列表，通常只有一条 HumanMessage，如果有 system 也会包含
        for msg in debate_prompts:
             self.conv_history.add_internal_message(msg, iteration_num=f"debate_{current_round}")
        
        # 构造用于 Invoke 的完整历史
        chat_history = []
        # 可选: 添加之前的 Exploration 上下文，或者只取最近的
        if len(self.conv_history.internal_comms) > 0:
            chat_history = self.conv_history.last_n_iterations_messages(n=10) # 辩论阶段可能不需要太早的 tool calls
        
        # 确保 debate prompt 在最后
        # 注意：add_internal_message 已经存入 history，如果在 chat_history 里包含了就不用重复 append
        # 为了保险起见，直接用 history 里的
        
        try:
            # 调用 LLM
            response_msg = self.llm_w_tools.invoke(chat_history)
            
            # 更新历史
            self.conv_history.add_internal_message(response_msg, iteration_num=f"debate_{current_round}")
            
            # 更新自己的结论
            new_solution = response_msg.content
            self.final_response = new_solution
            
            # 更新 Shared Memory 供记录
            self.shared_memory[self.agent_id].append(f"[Debate R{current_round+1}] Updated solution.")
            
            return new_solution
        except Exception as e:
            logger.error(f"Agent {self.agent_id} Debate invoke failed: {e}")
            return str(current_solution)

    # ... existing code ...

    def _format_peer_solutions(self, solutions: List[str]) -> str:
        formatted = []
        for i, sol in enumerate(solutions):
            # 简单去重，不显示自己的（虽然外面传进来可能已经过滤了）
            formatted.append(f"[Peer {i+1}]: {sol[:500]}..." if len(sol) > 500 else f"[Peer {i+1}]: {sol}")
        return "\n".join(formatted)

    def get_current_solution(self) -> str:
        if self.final_response:
            return str(self.final_response)
        # Fallback
        for m in reversed(self.messages):
            if isinstance(m, AIMessage) and m.content:
                return m.content
        return "No solution yet."