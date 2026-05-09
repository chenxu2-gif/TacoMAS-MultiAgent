from .base import AgentEnvironment
from .basic import BasicEnvironment

# Optional envs with heavy/extra dependencies
try:
    from .browsecomp import BrowseCompPlusEnvironment
except Exception:
    BrowseCompPlusEnvironment = None  # type: ignore

try:
    from .finance import FinanceEnvironment
except Exception:
    FinanceEnvironment = None  # type: ignore

try:
    from .plancraft import PlancraftEnvironment
except Exception:
    PlancraftEnvironment = None  # type: ignore
try:
    from .plancraft_converted import PlancraftConvertedEnvironment
except Exception:
    PlancraftConvertedEnvironment = None  # type: ignore
from .registry import (
    T,
    get_env,
    get_env_cls,
    is_env_registered,
    list_envs,
    register_env,
)
from .web_search import WebSearchEnvironment
