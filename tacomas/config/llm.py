from typing import Dict, Literal, Optional

import litellm
from pydantic import BaseModel, model_validator
from typing_extensions import Self

from tacomas.llm.litellm_lc import ChatLiteLLMLC


class LLMParams(BaseModel):
    """
    see litellm.completion() for parameter details
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    # cache controls: https://docs.litellm.ai/docs/proxy/caching#dynamic-cache-controls
    cache: Optional[Dict[Literal["no-cache", "no-store"], bool]] = None

    @model_validator(mode="after")
    def check_cache(self) -> Self:
        if not self.cache:
            self.cache = {}
        return self


class LLMConfig(BaseModel):
    """
    use litellm.get_valid_models() to see model names
    """

    #params: LLMParams
    params: LLMParams = LLMParams()
    model: str = "gemini/gemini-2.0-flash"

    # hyc-added
    temperature: Optional[float] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None

    def get_llm(self) -> ChatLiteLLMLC:

        # hyc-added获取基础参数
        llm_params = self.params.model_dump(exclude={"cache"})
        
        # hyc-added逻辑修复: 如果顶层设置了 temperature，覆盖 params 中的设置
        if self.temperature is not None:
            llm_params["temperature"] = self.temperature
        # --- 关键修改开始 ---
        # 直接设置 litellm 的参数，这比传参给构造函数更稳妥
        extra_kwargs = {}
        
        if self.api_base:
            # 1. 设置给构造函数用的参数
            extra_kwargs["api_base"] = self.api_base
            # 2. 也是给 litellm 用的通用参数
            extra_kwargs["openai_api_base"] = self.api_base
            
        if self.api_key:
            # 1. 设置给构造函数用的参数
            extra_kwargs["api_key"] = self.api_key
            # 2. 也是给 litellm 用的通用参数 (OpenAI provider 需要这个)
            extra_kwargs["openai_api_key"] = self.api_key

        # 3. 终极保险：直接设置 litellm 的模块属性 (防止某些内部调用丢失 context)
        if self.api_key:
            litellm.api_key = self.api_key
        if self.api_base:
            litellm.api_base = self.api_base
        # --- 关键修改结束 ---

        return ChatLiteLLMLC(
            model=self.model, 
            **extra_kwargs,
            **llm_params
        )


        #return ChatLiteLLMLC(
        #    model=self.model, **self.params.model_dump(exclude={"cache"})
        #)
