import ast
import json
import logging
import os
import re
from typing import Any, Dict, List

from litellm import completion

from tacomas.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
)
from tacomas.datasets.registry import register_dataset, register_dataset_instance

logger = logging.getLogger(__name__)


DATASET_IDS = ["workbench", "plancraft", "browsecomp-plus"]


@register_dataset_instance(DATASET_IDS)
class ConvertedRubricInstance(DatasetInstance):
    id: str
    question: str
    expected_answer: str
    question_type: str
    rubric: str
    expert_time_mins: str

    def get_prompt_info(self) -> Dict[str, Any]:
        return {"question": self.question}


@register_dataset(DATASET_IDS)
class ConvertedRubricDataset(Dataset):
    dataset_id: str
    instances: List[ConvertedRubricInstance]

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutputWithTrajectory[ConvertedRubricInstance]
    ) -> Dict[str, Any]:
        dataset_instance = instance_output.data_instance
        model_answer = ""
        if hasattr(instance_output, "agent_output") and instance_output.agent_output:
            model_answer = instance_output.agent_output
        elif hasattr(instance_output, "prediction") and instance_output.prediction:
            model_answer = instance_output.prediction
        elif instance_output.final_env_output and hasattr(instance_output.final_env_output, "output"):
            model_answer = instance_output.final_env_output.output

        return {
            "model_answer": model_answer,
            "expected_answer": dataset_instance.expected_answer,
            "rubric": dataset_instance.rubric,
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutputWithTrajectory[ConvertedRubricInstance]
    ) -> Dict[str, Any]:
        eval_data = self.get_instance_eval_output(instance_output)
        model_answer = eval_data.get("model_answer", "")
        rubric_str = eval_data.get("rubric", "[]")

        if not model_answer:
            return {"success": 0.0, "score": 0.0, "reasoning": "No answer provided"}

        rubrics: List[Dict[str, Any]] = []
        try:
            rubrics = json.loads(rubric_str)
        except json.JSONDecodeError:
            try:
                rubrics = ast.literal_eval(rubric_str)
            except Exception as exc:
                logger.error("Rubric parsing failed: %s | Raw rubric: %s", exc, rubric_str)
                return {"success": 0.0, "score": 0.0, "reasoning": "Rubric parsing failed"}

        if not rubrics:
            return {"success": 0.0, "score": 0.0, "reasoning": "Empty rubric"}

        score = self._evaluate_with_llm(model_answer, rubrics)
        return {"success": 1.0 if score >= 0.999 else 0.0, "score": score}

    def _evaluate_with_llm(self, answer: str, rubrics: List[Dict[str, Any]]) -> float:
        total_count = len(rubrics)
        rubric_text = json.dumps(rubrics, indent=2)
        prompt = f"""
You are an expert grader. Evaluate a candidate answer using the rubric below.

Candidate Answer:
"{answer}"

Rubric (JSON checklist):
{rubric_text}

For each rubric item:
- If operator is "correctness": does the answer satisfy/support the criterion? (true/false)
- If operator is "contradiction": does the answer contradict the criterion? (true/false)

Return JSON only:
{{"results":[true,false,...]}}
"""
        try:
            raw_eval_model = os.getenv("EVAL_MODEL", os.getenv("DMX_MODEL", "openai/gpt-4o-mini"))
            eval_model = raw_eval_model if "/" in raw_eval_model else f"openai/{raw_eval_model}"
            # Route by model provider: gemini/* uses GEMINI_API_KEY and
            # needs no api_base (litellm hits Google natively); openai/*
            # uses OPENAI_API_KEY and any explicit EVAL_API_BASE. The old
            # default of https://www.dmxapi.cn/v1 broke eval whenever
            # EVAL_API_BASE wasn't explicitly set, because DMX only
            # accepts DMX-issued tokens (returns rix_api_error).
            model_provider = eval_model.split("/", 1)[0].lower()
            if model_provider == "gemini":
                eval_api_base = os.getenv("EVAL_API_BASE", "") or None
                eval_api_key = (
                    os.getenv("EVAL_API_KEY")
                    or os.getenv("GEMINI_API_KEY", "")
                    or os.getenv("GOOGLE_API_KEY", "")
                )
            else:
                eval_api_base = os.getenv("EVAL_API_BASE", "") or None
                eval_api_key = (
                    os.getenv("EVAL_API_KEY")
                    or os.getenv("DMX_API_KEY")
                    or os.getenv("OPENAI_API_KEY", "")
                )
            if not eval_api_key:
                logger.error("Eval LLM skipped: no key found.")
                return 0.0

            # num_retries=0: a 429 (billing cap / quota exhausted) does not
            # clear in our retry window, so retrying just burns SLURM time.
            # Transient errors are propagated to the caller, which logs and
            # returns 0.0 for that instance; the rescore script can catch up
            # later once quota is back.
            response = completion(
                model=eval_model,
                num_retries=0,
                messages=[{"role": "user", "content": prompt}],
                api_base=eval_api_base or None,
                api_key=eval_api_key,
            )
            content = response.choices[0].message.content
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                return 0.0
            result_json = json.loads(json_match.group())
            results = result_json.get("results", [])
            passed = 0
            for i, res in enumerate(results):
                if i >= len(rubrics):
                    break
                op = rubrics[i].get("operator")
                if op == "correctness" and res is True:
                    passed += 1
                elif op == "contradiction" and res is False:
                    passed += 1
            return passed / total_count if total_count > 0 else 0.0
        except Exception as exc:
            logger.error("Eval LLM failed: %s", exc)
            return 0.0

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not eval_outputs:
            return {"avg_score": 0.0, "num_instances": 0}
        scores = [e.get("score", 0.0) for e in eval_outputs]
        perfect_scores = [1.0 if s >= 0.999 else 0.0 for s in scores]
        return {
            "avg_score": sum(scores) / len(scores),
            "perfect_rate": sum(perfect_scores) / len(perfect_scores),
            "num_instances": len(eval_outputs),
        }
