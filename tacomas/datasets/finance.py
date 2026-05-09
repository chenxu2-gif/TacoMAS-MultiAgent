import json
import logging
import os
from typing import Any, Dict, List, Optional
from tacomas.datasets.registry import register_dataset, register_dataset_instance
from tacomas.datasets.base import Dataset, DatasetInstance, DatasetInstanceOutputWithTrajectory

# 假设你有一个可以调用的简单 LLM 接口，如果没有，需要接入 litellm 等
from litellm import completion 
logger = logging.getLogger(__name__)



# 定义单个 Finance 题目
@register_dataset_instance(["finance-benchmark"])
class FinanceInstance(DatasetInstance):
    id: str
    question: str
    expected_answer: str
    question_type: str
    rubric: str  # JSON string containing evaluation criteria
    expert_time_mins: str

    def get_prompt_info(self) -> Dict[str, Any]:
        # 只把问题暴露给 Prompt
        return {
            "question": self.question
        }

# 定义 Dataset 容器
@register_dataset(["finance-benchmark"])
class FinanceDataset(Dataset):
    dataset_id: str = "finance-benchmark"
    instances: List[FinanceInstance]


    # 1. 获取模型产生的答案
    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutputWithTrajectory[FinanceInstance]
    ) -> Dict[str, Any]:
        
        # --- 修复核心：直接使用基类定义的 data_instance 字段 ---
        # base.py 中定义: class DatasetInstanceOutput(BaseModel, Generic[T]): data_instance: T
        dataset_instance = instance_output.data_instance
        # ----------------------------------------------------

        # 从 Environment 输出中提取模型提交的 answer
        model_answer = ""
        # 注意：这里要做好判空，防止 env 没跑或者没输出
        if hasattr(instance_output, "agent_output") and instance_output.agent_output:
            # 这里的 agent_output 就是 "US Steel has addressed..." 那段话
            model_answer = instance_output.agent_output
        elif hasattr(instance_output, "prediction") and instance_output.prediction:
            # 有些 AgentSystem 可能把结果存在 prediction 字段
            model_answer = instance_output.prediction
        elif instance_output.final_env_output and hasattr(instance_output.final_env_output, "output"):
            # 最后的备选：从 Environment 状态里拿（仅当使用 submit_final_result 工具时有效）
            model_answer = instance_output.final_env_output.output
        
        return {
            "model_answer": model_answer,
            "expected_answer": dataset_instance.expected_answer,
            "rubric": dataset_instance.rubric
        }

    # 2. 核心评分逻辑 (LLM-as-a-Judge)
    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutputWithTrajectory[FinanceInstance]
    ) -> Dict[str, Any]:
        
        import ast
        
        eval_data = self.get_instance_eval_output(instance_output)
        model_answer = eval_data.get("model_answer", "")
        rubric_str = eval_data.get("rubric", "[]")
        
        if not model_answer:
            return {"score": 0.0, "reasoning": "No answer provided"}

        rubrics = []
        try:
            # 优先尝试标准 JSON 解析
            rubrics = json.loads(rubric_str)
        except json.JSONDecodeError:
            try:
                # 如果失败，尝试作为 Python 字面量解析 (处理单引号问题)
                rubrics = ast.literal_eval(rubric_str)
            except Exception as e:
                logger.error(f"Rubric parsing failed: {e} | Raw rubric: {rubric_str}")
                return {"score": 0.0, "reasoning": "Rubric parsing failed"}

        if not rubrics:
             return {"score": 0.0, "reasoning": "Empty rubric"}

        # 调用 LLM 进行打分 (同步调用)
        # 这里为了演示方便直接调用，实际工程中可能需要封装
        score = self._evaluate_with_llm(model_answer, rubrics)
        
        return {
            "success": 1.0 if score == 1.0 else 0.0, #或者直接返回 score
            "score": score
        }

    # 辅助函数：调用 LLM 检查 Rubric
    def _evaluate_with_llm(self, answer: str, rubrics: List[Dict]) -> float:
        passed_count = 0
        total_count = len(rubrics)
        
        # 构建 Prompt，一次性让 LLM 判断所有 criteria
        rubric_text = json.dumps(rubrics, indent=2)
        prompt = f"""
You are an expert grader. Your task is to evaluate a candidate's answer based on a specific rubric.

Candidate Answer:
"{answer}"

Rubric (JSON checklist):
{rubric_text}

For each item in the rubric:
- If operator is "correctness": Does the answer support/contain this fact? (True/False)
- If operator is "contradiction": Does the answer contradict this fact? (True/False) NOTE: If answer contradicts, it is BAD.

Output a JSON object with a list of boolean results, one for each rubric item in order.
Example: {{ "results": [true, false, true] }}
"""
        try:
            # [新增] 打印送给 Judge 的 Prompt
            logger.warning(f"--- Eval Prompt ---\n{prompt}\n-------------------")

            raw_eval_model = os.getenv("EVAL_MODEL", os.getenv("DMX_MODEL", "openai/gpt-4o-mini"))
            eval_model = raw_eval_model if "/" in raw_eval_model else f"openai/{raw_eval_model}"
            # Default judge backend to DMX OpenAI-compatible endpoint.
            eval_api_base = os.getenv("EVAL_API_BASE", "https://www.dmxapi.cn/v1")
            eval_api_key = os.getenv(
                "EVAL_API_KEY",
                os.getenv(
                    "DMX_API_KEY",
                    os.getenv(
                        "OPENAI_API_KEY",
                        os.getenv("GEMINI_API_KEY_ARG", os.getenv("GEMINI_API_KEY", "")),
                    ),
                ),
            )

            if not eval_api_key:
                logger.error("Eval LLM skipped: no key found in EVAL_API_KEY/DMX_API_KEY/GEMINI_API_KEY")
                return 0.0

            logger.info(f"Eval LLM backend: model={eval_model}, api_base={eval_api_base}")

            response = completion(
                model=eval_model,
                num_retries=5,
                messages=[{"role": "user", "content": prompt}],
                api_base=eval_api_base or None,
                api_key=eval_api_key,
            )
            content = response.choices[0].message.content
            # [新增] 打印 Judge 的原始回复
            logger.warning(f"--- Judge Response ---\n{content}\n--------------------")
            # 简单的 JSON 提取逻辑
            import re
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
                results = result_json.get("results", [])

                # [新增] 打印解析后的结果列表
                logger.warning(f"Parsed Results: {results}")
                
                # 计算分数
                passed = 0
                for i, res in enumerate(results):
                    if i >= len(rubrics): break
                    op = rubrics[i].get("operator")
                    
                    if op == "correctness" and res is True:
                        passed += 1
                    elif op == "contradiction" and res is False: # 没矛盾就是好事? 或者 contradiction operator 的意思是“如果包含这个矛盾点就扣分”？
                        # 通常 contradiction 的意思是：如果 LLM 说“是的，有矛盾”，那这一项就挂了。
                        # 如果 LLM 说“没有矛盾”，那这一项通过。
                        # 需根据 rubric 定义微调逻辑。假设 False = 没有矛盾 = Pass
                        passed += 1
                    # [新增] 打印每一项的判定详情 (可选，太长可以不用)
                    logger.warning(f"Item {i}: Op={op}, Judge={res} -> Passed? { (op=='correctness' and res) or (op=='contradiction' and not res) }")
                
                final_score = passed / total_count if total_count > 0 else 0.0
                
                # [新增] 打印最终得分
                logger.info(f"Final Score: {final_score} ({passed}/{total_count})")
                
                return final_score
                
        except Exception as e:
            logger.error(f"Eval LLM failed: {e}")
            return 0.0
        
        return 0.0
    
    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not eval_outputs:
            return {"avg_score": 0.0, "num_instances": 0}
            
        # 提取每道题的得分 (0.0 ~ 1.0)
        scores = [e.get("score", 0.0) for e in eval_outputs]
        
        # 计算完美率 (Perfect Match Rate)
        perfect_scores = [1.0 if s >= 0.99 else 0.0 for s in scores]

        return {
            "avg_score": sum(scores) / len(scores),      # 这是你要的平均分 (0.25分这种会拉低这里)
            "perfect_rate": sum(perfect_scores) / len(perfect_scores), # 满分率
            "num_instances": len(eval_outputs),
        }