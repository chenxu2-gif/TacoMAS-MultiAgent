from copy import deepcopy
from typing import Callable, Dict

from langchain_core.tools import BaseTool
from tacomas.langfuse_client import get_lf_client
from .registry import register_tool, list_registered_tools, get_tool

# Optional tool: python_repl depends on langfun -> libmagic (often missing on Windows)
try:
    from .code import python_repl
except Exception:
    python_repl = None  # type: ignore
from .done import done_with_confidence
from .multiply import multiply, multiply_by_max
from .search import tavily_search_tool, tavily_search_simple_tool
from .utils import enhance_tool, cls_tool
