import json
import os
import time
import uuid
from typing import Any, Dict, List, Mapping, Optional, cast

import langfuse
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_litellm.chat_models.litellm import ChatLiteLLM
from litellm.types.utils import ModelResponse


class ChatLiteLLMLC(ChatLiteLLM):
    log_langfuse: bool = False

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        res: ChatResult = super()._create_chat_result(response)
        if res.llm_output is None:
            res.llm_output = {}
        res.llm_output["litellm_response"] = response
        return res

    def invoke(self, *args, **kwargs) -> AIMessage:
        return cast(AIMessage, super().invoke(*args, **kwargs))

    def _log_langfuse(
        self,
        message_dicts: List[Dict[str, Any]],
        params: Dict[str, Any],
        response: ModelResponse,
    ) -> None:
        client = langfuse.Langfuse()  # type: ignore
        model_params = {
            k: v
            for k, v in params.items()
            if k not in ["model", "stream"] and v is not None
        }

        gen_context = client.start_generation(
            name=f"call {params.get('model')}"
            + (" (from cache)" if response._hidden_params.get("cache_hit", "") else ""),
            input=message_dicts,
            model=params.get("model"),
            model_parameters=model_params,
        )

        gen_context.update(
            output=response.choices[0].message,  # type: ignore
            metadata=response.model_dump(),
            usage_details=response.usage if hasattr(response, "usage") else None,  # type: ignore
            cost_details={"total": response._hidden_params.get("response_cost", 0)},
        )
        gen_context.end()
        client.flush()

    def _dmx_extract_text(self, result_json: Dict[str, Any]) -> str:
        try:
            return result_json["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return ""

    def _dmx_extract_message(self, result_json: Dict[str, Any]) -> Dict[str, Any]:
        text_chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        try:
            parts = result_json["candidates"][0]["content"].get("parts", [])
        except Exception:
            parts = []

        for part in parts:
            if not isinstance(part, dict):
                continue

            text = part.get("text")
            if text:
                text_chunks.append(str(text))

            fn_call = part.get("functionCall")
            if isinstance(fn_call, dict) and fn_call.get("name"):
                args = fn_call.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:20]}",
                        "type": "tool_call",
                        "name": str(fn_call["name"]),
                        "args": args,
                    }
                )

        return {"text": "\n".join(text_chunks).strip(), "tool_calls": tool_calls}

    def _dmx_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    chunks.append(str(item["text"]))
                else:
                    chunks.append(str(item))
            return "\n".join(chunks)
        return str(content)

    def _sanitize_dmx_schema(self, node: Any) -> Any:
        if isinstance(node, list):
            return [self._sanitize_dmx_schema(x) for x in node if x is not None]
        if not isinstance(node, dict):
            return node

        banned_keys = {
            "additionalProperties",
            "additional_properties",
            "anyOf",
            "any_of",
            "oneOf",
            "one_of",
            "allOf",
            "all_of",
            "$schema",
            "examples",
            "example",
            "default",
            "title",
        }

        cleaned: Dict[str, Any] = {}
        for k, v in node.items():
            if k in banned_keys:
                continue
            cleaned[k] = self._sanitize_dmx_schema(v)

        # Gemini function declaration parameters requires object-ish schema;
        # when upstream schema is too complex, degrade to permissive string.
        if "type" not in cleaned and "properties" in cleaned:
            cleaned["type"] = "object"

        return cleaned

    def _dmx_tools_from_params(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_tools = params.get("tools") or []
        declarations: List[Dict[str, Any]] = []

        for t in raw_tools:
            if not isinstance(t, dict):
                continue
            if t.get("type") != "function":
                continue
            fn = t.get("function", {})
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            raw_parameters = fn.get("parameters", {"type": "object", "properties": {}})
            parameters = self._sanitize_dmx_schema(raw_parameters)
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            if parameters.get("type") not in {"object", "string", "number", "integer", "boolean", "array"}:
                parameters["type"] = "object"

            declarations.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": parameters,
                }
            )

        if not declarations:
            return []
        return [{"functionDeclarations": declarations}]

    def _call_dmx_nonstream(self, message_dicts: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        dmx_api_key = os.environ.get("DMX_API_KEY", "")
        if not dmx_api_key:
            raise RuntimeError("USE_DMX_ALL_NONSTREAM=1 but DMX_API_KEY is empty")

        configured_model = os.environ.get("DMX_MODEL", "gemini-2.5-flash")
        model = configured_model

        timeout = int(os.environ.get("DMX_TIMEOUT", "60"))
        url = f"https://www.dmxapi.cn/v1beta/models/{model}:generateContent?key={dmx_api_key}"

        system_texts: List[str] = []
        contents: List[Dict[str, Any]] = []
        for m in message_dicts:
            role = str(m.get("role", "user"))

            if role == "system":
                system_texts.append(self._dmx_content_to_text(m.get("content", "")))
                continue

            if role == "assistant" and m.get("tool_calls"):
                parts: List[Dict[str, Any]] = []
                for tc in m.get("tool_calls", []):
                    if not isinstance(tc, dict):
                        continue
                    name = tc.get("function", {}).get("name") or tc.get("name")
                    arguments = tc.get("function", {}).get("arguments") or tc.get("args", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except Exception:
                            arguments = {}
                    if name:
                        parts.append({"functionCall": {"name": name, "args": arguments or {}}})
                if parts:
                    contents.append({"role": "model", "parts": parts})
                    continue

            if role == "tool":
                tool_name = m.get("name", "tool")
                tool_content = self._dmx_content_to_text(m.get("content", ""))
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {"content": tool_content},
                                }
                            }
                        ],
                    }
                )
                continue

            text = self._dmx_content_to_text(m.get("content", ""))
            dmx_role = "model" if role == "assistant" else "user"
            contents.append({"role": dmx_role, "parts": [{"text": text}]})

        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]

        payload: Dict[str, Any] = {"contents": contents}

        if system_texts:
            payload["system_instruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}

        dmx_tools = self._dmx_tools_from_params(params)
        if dmx_tools:
            payload["tools"] = dmx_tools

        max_retries = int(os.environ.get("DMX_HTTP_RETRIES", "3"))
        base_backoff = float(os.environ.get("DMX_HTTP_BACKOFF", "1.5"))
        last_err: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"DMX API failed: {resp.status_code}, {resp.text[:500]}")

                data = resp.json()
                parsed = self._dmx_extract_message(data)
                if not parsed["text"] and not parsed["tool_calls"]:
                    parsed["text"] = self._dmx_extract_text(data) or json.dumps(data, ensure_ascii=False)
                return parsed
            except Exception as e:
                last_err = e
                if attempt >= max_retries - 1:
                    break
                time.sleep(base_backoff * (2 ** attempt))

        raise RuntimeError(f"DMX request failed after {max_retries} attempts: {last_err}")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        log_langfuse = kwargs.pop("log_langfuse", None)
        if log_langfuse is None:
            log_langfuse = self.log_langfuse

        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}

        use_dmx_all_nonstream = os.environ.get("USE_DMX_ALL_NONSTREAM", "0") == "1"
        if use_dmx_all_nonstream and not should_stream:
            parsed = self._call_dmx_nonstream(message_dicts, params)
            if parsed.get("tool_calls"):
                message = AIMessage(content=parsed.get("text", ""), tool_calls=parsed["tool_calls"])
            else:
                message = AIMessage(content=parsed.get("text", ""))
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=message,
                    )
                ],
                llm_output={"provider": "dmx_nonstream"},
            )

        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )

        if log_langfuse:
            self._log_langfuse(message_dicts, params, response)
        return self._create_chat_result(response)
