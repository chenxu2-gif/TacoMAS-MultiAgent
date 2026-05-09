import json
import logging
import os
import re
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from pydantic import BaseModel

from tacomas.env.base import AgentEnvironmentTools
from tacomas.env.registry import register_env
from tacomas.env.tools import cls_tool

# 引入第三方库
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import requests
except ImportError:
    requests = None

try:
    from litellm import completion
except ImportError:
    completion = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

logger = logging.getLogger(__name__)

PRIMARY_FILING_KEY_PATTERN = re.compile(r"^primary_filing(?:_(?:manifest|[1-5]))?$")


def _infer_provider_from_model(model_name: str) -> str:
    model = str(model_name or "").strip().lower()
    if not model:
        return ""
    if "/" in model:
        return model.split("/", 1)[0]
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return "openai"
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("deepseek"):
        return "deepseek"
    return ""


def _default_api_key_for_model(model_name: str) -> str:
    provider = _infer_provider_from_model(model_name)
    provider_key_map = {
        "openai": os.getenv("OPENAI_API_KEY", "").strip(),
        "gemini": os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip(),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", os.getenv("CLAUDE_API_KEY", "")).strip(),
        "deepseek": os.getenv("DEEPSEEK_API_KEY", "").strip(),
    }
    return provider_key_map.get(provider, "")

class FinanceEnvOutput(BaseModel):
    output: str
    success: bool = False
    details: Dict[str, Any] = {}

MAX_END_DATE = "2025-04-07"
GENERIC_EVIDENCE_HINTS = [
    "reconciliation",
    "adjustment",
    "summary",
    "discussion",
    "table",
    "note",
    "breakdown",
    "details",
]

PREFERRED_FORM_PREFIXES = [
    "10-k",
    "10-q",
    "8-k",
    "20-f",
    "6-k",
]

DEPRIORITIZED_FORM_PREFIXES = [
    "s-1",
    "s-3",
    "s-4",
    "f-1",
    "f-3",
    "424b",
    "prospectus",
]

BAD_ATTACHMENT_FORM_PREFIXES = [
    "ex-",
]

PREFERRED_OFFICIAL_DOMAINS = [
    "sec.gov",
    "www.sec.gov",
]

ENTITY_STOPWORDS = {
    "what", "which", "where", "when", "does", "make", "made", "into", "derive",
    "adjustments", "adjustment", "adjusted", "ebitda", "net", "income", "loss",
    "official", "source", "document", "evidence", "report", "reports", "filing",
    "filings", "from", "before", "after", "with", "without", "quarter", "annual",
    "financial", "results", "company", "inc", "corp", "ltd", "plc",
}

@register_env("finance")
class FinanceEnvironment(AgentEnvironmentTools):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_answer = ""
        self.is_done = False
        
        # 模拟 data_storage
        self.data_storage: Dict[str, str] = {}
        
        # 初始化 API Clients
        self.serper_api_key = os.getenv("SERPER_API_KEY", os.getenv("Serper_API_KEY", ""))
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self.sec_api_key = os.getenv("SEC_EDGAR_API_KEY")

        # Prefer Serper; fall back to Tavily
        if self.tavily_api_key:
            try:
                self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
            except Exception:
                self.tavily_client = None
        else:
            self.tavily_client = None
        
        self.llm_api_base = os.getenv("LLM_API_BASE", "")
        raw_model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
        # Keep the gemini/ prefix so litellm routes to AI Studio instead of Vertex AI ADC.
        if not raw_model.startswith("gemini/") and "/" not in raw_model and raw_model.startswith("gemini"):
            raw_model = f"gemini/{raw_model}"
        self.llm_model = raw_model
        explicit_llm_api_key = os.getenv("LLM_API_KEY", "").strip()
        self.llm_api_key = explicit_llm_api_key or _default_api_key_for_model(self.llm_model)

    def _run_web_search(
        self,
        search_query: str,
        start_date: str = None,
        end_date: str = None,
        number_of_results: int = 10,
    ) -> str:
        if not self.tavily_client and not self.serper_api_key:
            return json.dumps({"success": False, "result": "ERROR: No search API key found (TAVILY_API_KEY or SERPER_API_KEY)."})

        DATE_REGEX = r"^\d{4}-\d{2}-\d{2}$"
        if end_date is None:
            end_date = MAX_END_DATE
        if end_date and end_date > MAX_END_DATE:
            end_date = MAX_END_DATE
        if start_date and start_date > MAX_END_DATE:
            start_date = MAX_END_DATE

        if self.serper_api_key:
            try:
                payload = {"q": search_query, "num": number_of_results}
                if start_date:
                    payload["tbs"] = f"cdr:1,cd_min:{start_date},cd_max:{end_date}"
                resp = requests.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                results = [
                    {"url": r.get("link", ""), "title": r.get("title", ""), "content": r.get("snippet", "")}
                    for r in data.get("organic", [])
                ]
                return json.dumps(results, indent=2)
            except Exception as e:
                return json.dumps({"success": False, "result": f"Serper search failed: {str(e)}"})

        try:
            kwargs = {}
            if start_date:
                if not re.match(DATE_REGEX, start_date):
                    return json.dumps({"success": False, "result": f"Invalid start_date format: '{start_date}'."})
                if start_date > end_date:
                    return json.dumps({"success": False, "result": f"start_date '{start_date}' is later than end_date '{end_date}'"})
                kwargs["start_date"] = start_date
            response = self.tavily_client.search(
                query=search_query,
                search_depth="fast",
                max_results=number_of_results,
                include_answer=False,
                **kwargs,
            )
            return json.dumps(response.get("results", []), indent=2)
        except Exception as e:
            return json.dumps({"success": False, "result": str(e)})

    def _run_edgar_search(
        self,
        search_query: str,
        start_date: str = "1900-01-01",
        end_date: str = MAX_END_DATE,
        top_n_results: int = 100,
        page: int = 1,
        form_types: Optional[List[str]] = None,
        ciks: Optional[List[str]] = None,
    ) -> str:
        if not self.sec_api_key:
            return "ERROR: SEC API Key not found."

        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        if not re.match(date_pattern, start_date):
            return json.dumps({"success": False, "result": f"start_date '{start_date}' is not in yyyy-mm-dd format"})
        if not re.match(date_pattern, end_date):
            return json.dumps({"success": False, "result": f"end_date '{end_date}' is not in yyyy-mm-dd format"})

        if start_date > MAX_END_DATE:
            start_date = MAX_END_DATE
        if end_date > MAX_END_DATE:
            end_date = MAX_END_DATE
        if start_date > end_date:
            return json.dumps({"success": False, "result": f"start_date '{start_date}' > end_date '{end_date}'"})

        payload = {
            "query": search_query,
            "startDate": start_date,
            "endDate": end_date,
            "page": page,
        }
        if form_types:
            payload["formTypes"] = form_types
        if ciks:
            payload["ciks"] = ciks

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.sec_api_key,
        }

        try:
            response = requests.post("https://api.sec-api.io/full-text-search", json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            filings = self._rank_sec_results(data.get("filings", []))[:top_n_results]
            return json.dumps(filings, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "result": f"SEC API error: {str(e)}"})

    def _normalize_form_type(self, item: Dict[str, Any]) -> str:
        return str(
            item.get("formType")
            or item.get("type")
            or item.get("description")
            or ""
        ).strip()

    def _score_form_type(self, form_type: str, title: str = "", url: str = "") -> float:
        joined = f"{form_type} {title} {url}".lower().strip()
        if not joined:
            return 0.0
        if any(joined.startswith(prefix) or f" {prefix}" in joined for prefix in BAD_ATTACHMENT_FORM_PREFIXES):
            return -8.0
        if any(joined.startswith(prefix) or f" {prefix}" in joined for prefix in DEPRIORITIZED_FORM_PREFIXES):
            return -5.0
        if any(joined.startswith(prefix) or f" {prefix}" in joined for prefix in PREFERRED_FORM_PREFIXES):
            return 4.0
        if "annual report" in joined or "quarterly report" in joined:
            return 3.0
        if "shareholder letter" in joined or "earnings release" in joined:
            return 2.5
        return 0.0

    def _should_drop_sec_result(self, item: Dict[str, Any]) -> bool:
        form_type = self._normalize_form_type(item).lower()
        url = str(
            item.get("filingUrl")
            or item.get("linkToHtml")
            or item.get("linkToFilingDetails")
            or item.get("linkToTxt")
            or ""
        ).lower()
        title = str(item.get("description") or item.get("title") or "").lower()
        joined = f"{form_type} {title} {url}"
        return any(joined.startswith(prefix) or f" {prefix}" in joined for prefix in BAD_ATTACHMENT_FORM_PREFIXES)

    def _rank_sec_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked: List[tuple[float, Dict[str, Any]]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            if self._should_drop_sec_result(item):
                continue
            form_type = self._normalize_form_type(item)
            title = str(item.get("description") or item.get("title") or "")
            url = str(
                item.get("filingUrl")
                or item.get("linkToHtml")
                or item.get("linkToFilingDetails")
                or item.get("linkToTxt")
                or ""
            )
            form_score = self._score_form_type(form_type, title, url)
            # Favor newer filings when available.
            filed_at = str(item.get("filedAt") or "")
            recency_bonus = 0.0
            if filed_at:
                recency_bonus = min(len(filed_at), 10) * 0.01
            ranked.append((form_score + recency_bonus, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked]

    def _extract_candidate_urls(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            url = (
                item.get("linkToHtml")
                or item.get("linkToFilingDetails")
                or item.get("linkToTxt")
                or item.get("filingUrl")
                or item.get("url")
                or item.get("link")
                or ""
            )
            url = str(url or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            form_type = self._normalize_form_type(item)
            title = str(
                item.get("title")
                or item.get("description")
                or form_type
                or item.get("filedAt")
                or ""
            ).strip()
            candidates.append(
                {
                    "url": url,
                    "title": title,
                    "form_type": form_type,
                    "source_type": "sec" if item.get("accessionNo") or item.get("cik") else "web",
                    "form_score": self._score_form_type(form_type, title, url),
                }
            )
        return candidates

    def _clean_plain_text(self, text: str) -> str:
        lines = (line.strip() for line in str(text or "").splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)

    def _score_source_quality(self, url: str, title: str = "", source_type: str = "", form_type: str = "") -> float:
        parsed = urlparse(url or "")
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        title_low = str(title or "").lower()
        score = 0.0

        if any(domain.endswith(official) for official in PREFERRED_OFFICIAL_DOMAINS):
            score += 4.0
        if source_type == "sec":
            score += 1.5
        if any(token in domain for token in ["investor", "ir.", "q4cdn", "news."]):
            score += 2.0
        if path.endswith(".pdf"):
            score -= 0.4
        if path.endswith(".htm") or path.endswith(".html"):
            score += 0.6
        if any(token in domain for token in [
            "cnbc.com",
            "finance.yahoo.com",
            "stocklight.com",
            "stock-analysis-on.net",
            "finbox.com",
            "investing.com",
            "demandsage.com",
            "tridenstechnology.com",
            "businessofapps.com",
            "tastylive.com",
            "henryfund.tippie.uiowa.edu",
            "statista.com",
            "macrotrends.net",
            "companiesmarketcap.com",
            "reddit.com",
            "platformonomics.com",
            "deriv.com",
            "seekingalpha.com",
            "fool.com",
        ]):
            score -= 3.0
        if "shareholder letter" in title_low or "earnings release" in title_low:
            score += 1.0
        if str(form_type or "").lower() in {"10-k", "10-q", "8-k", "20-f", "6-k"}:
            score += 0.8
        return score

    def _is_low_trust_secondary_source(self, url: str, title: str = "", source_type: str = "", form_type: str = "") -> bool:
        parsed = urlparse(url or "")
        domain = parsed.netloc.lower()
        title_low = str(title or "").lower()
        form_low = str(form_type or "").lower()

        if source_type == "sec" or self._is_official_preferred_source(url, source_type):
            return False
        if form_low in {"10-k", "10-q", "8-k", "20-f", "6-k"}:
            return False

        low_trust_domains = [
            "cnbc.com",
            "finance.yahoo.com",
            "stocklight.com",
            "stock-analysis-on.net",
            "finbox.com",
            "investing.com",
            "demandsage.com",
            "tridenstechnology.com",
            "businessofapps.com",
            "tastylive.com",
            "henryfund.tippie.uiowa.edu",
            "statista.com",
            "macrotrends.net",
            "companiesmarketcap.com",
            "reddit.com",
            "platformonomics.com",
            "deriv.com",
            "seekingalpha.com",
            "fool.com",
            "daloopa.com",
            "soax.com",
        ]
        if any(token in domain for token in low_trust_domains):
            return True
        if any(token in title_low for token in ["statistics", "spreadsheet", "download", "market data", "company news"]):
            return True
        return False

    def _is_official_preferred_source(self, url: str, source_type: str = "") -> bool:
        parsed = urlparse(url or "")
        domain = parsed.netloc.lower()
        if source_type == "sec":
            return True
        if any(domain.endswith(official) for official in PREFERRED_OFFICIAL_DOMAINS):
            return True
        return any(token in domain for token in ["investor", "ir.", "q4cdn"])

    def _normalize_constraint_map(self, raw: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}
        if not isinstance(raw, dict):
            return normalized
        for key, values in raw.items():
            key_text = str(key or "").strip()
            if not key_text:
                continue
            bucket: List[str] = []
            if isinstance(values, list):
                iterable = values
            else:
                iterable = [values]
            for value in iterable:
                text = str(value or "").strip().lower()
                if text and text not in bucket:
                    bucket.append(text)
            if bucket:
                normalized[key_text] = bucket[:8]
        return normalized

    def _merge_constraint_maps(
        self,
        base: Optional[Dict[str, Any]],
        extra: Optional[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        merged = self._normalize_constraint_map(base)
        for key, values in self._normalize_constraint_map(extra).items():
            bucket = list(merged.get(key, []))
            for value in values:
                if value not in bucket:
                    bucket.append(value)
            if bucket:
                merged[key] = bucket[:8]
        return merged

    def _merge_pattern_lists(
        self,
        base: Optional[List[str]],
        extra: Optional[List[str]],
    ) -> List[str]:
        merged: List[str] = []
        for value in list(base or []) + list(extra or []):
            cleaned = str(value or "").strip().lower()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged[:12]

    def _derive_prepare_primary_filing_constraints(self, search_query: str) -> Dict[str, Any]:
        query = str(search_query or "")
        low = query.lower()
        years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", low)))
        entities = self._extract_query_entities(search_query)
        phrases = self._extract_query_phrases(search_query)
        keywords = self._extract_query_keywords(search_query)

        derived_required: Dict[str, List[str]] = {}
        derived_avoid: Dict[str, List[str]] = {}
        prefer_patterns: List[str] = []
        avoid_patterns: List[str] = []
        missing_targets: List[str] = []

        def _add_pattern(bucket: List[str], value: str) -> None:
            cleaned = str(value or "").strip().lower()
            if cleaned and cleaned not in bucket:
                bucket.append(cleaned)

        def _add_constraint(bucket: Dict[str, List[str]], key: str, value: str) -> None:
            cleaned = str(value or "").strip().lower()
            if not cleaned:
                return
            values = bucket.setdefault(key, [])
            if cleaned not in values:
                values.append(cleaned)

        if years:
            for year in years[:8]:
                if year not in missing_targets:
                    missing_targets.append(year)
            if len(years) == 1:
                _add_constraint(derived_required, "time", years[0])

        for name in entities.get("names", [])[:3]:
            _add_constraint(derived_required, "entity", name)
            _add_pattern(prefer_patterns, name)
        for ticker in entities.get("tickers", [])[:2]:
            _add_constraint(derived_required, "entity", ticker)
            _add_pattern(prefer_patterns, ticker)
        for alias in entities.get("aliases", [])[:5]:
            _add_pattern(prefer_patterns, alias)

        explicit_global = any(
            token in low
            for token in [" global ", " worldwide ", " company-wide ", " overall "]
        ) or low.startswith("global ") or low.endswith(" global")
        explicit_regional = any(
            token in low
            for token in [
                "north america",
                "regional",
                "emea",
                "apac",
                "asia pacific",
                "latin america",
                "us and canada",
                "u.s. and canada",
            ]
        )
        if explicit_global:
            _add_constraint(derived_required, "scope", "global")
            _add_constraint(derived_avoid, "scope", "north america")
            _add_constraint(derived_avoid, "scope", "regional")
            _add_pattern(prefer_patterns, "global")
            _add_pattern(avoid_patterns, "north america")
            _add_pattern(avoid_patterns, "regional")
        elif explicit_regional:
            for scope_term in [
                "north america",
                "regional",
                "emea",
                "apac",
                "asia pacific",
                "latin america",
                "us and canada",
                "u.s. and canada",
            ]:
                if scope_term in low:
                    _add_constraint(derived_required, "scope", scope_term)
                    _add_pattern(prefer_patterns, scope_term)

        metric_patterns = [
            "arpu",
            "arppu",
            "average revenue per paying user",
            "average revenue per paying membership",
            "average revenue per membership",
            "take rate",
            "pre-tax margin",
            "pretax margin",
            "adjusted ebitda",
            "reconciliation",
        ]
        for metric in metric_patterns:
            if metric in low:
                _add_constraint(derived_required, "metric", metric)
                _add_pattern(prefer_patterns, metric)

        for phrase in phrases[:5]:
            if any(year in phrase for year in years):
                continue
            if sum(1 for token in phrase.split() if token not in ENTITY_STOPWORDS) >= 2:
                _add_pattern(prefer_patterns, phrase)
        for keyword in keywords[:8]:
            if keyword not in ENTITY_STOPWORDS and len(keyword) >= 4:
                _add_pattern(prefer_patterns, keyword)

        return {
            "required_constraints": derived_required,
            "avoid_constraints": derived_avoid,
            "prefer_source_patterns": prefer_patterns,
            "avoid_source_patterns": avoid_patterns,
            "missing_targets": missing_targets,
        }

    def _score_constraint_alignment(
        self,
        *,
        title: str,
        url: str,
        text: str,
        required_constraints: Optional[Dict[str, Any]] = None,
        avoid_constraints: Optional[Dict[str, Any]] = None,
        prefer_source_patterns: Optional[List[str]] = None,
        avoid_source_patterns: Optional[List[str]] = None,
        missing_targets: Optional[List[str]] = None,
    ) -> tuple[float, bool, Dict[str, Any]]:
        required = self._normalize_constraint_map(required_constraints)
        avoid = self._normalize_constraint_map(avoid_constraints)
        prefer_patterns = [str(v).strip().lower() for v in (prefer_source_patterns or []) if str(v).strip()]
        avoid_patterns = [str(v).strip().lower() for v in (avoid_source_patterns or []) if str(v).strip()]
        targets = [str(v).strip().lower() for v in (missing_targets or []) if str(v).strip()]

        head_text = str(text or "")[:12000].lower()
        haystack = f"{str(title or '').lower()}\n{str(url or '').lower()}\n{head_text}"
        score = 0.0
        rejected = False
        info: Dict[str, Any] = {"required_hits": {}, "avoid_hits": {}, "target_hits": []}

        for key, values in required.items():
            hits = [value for value in values if value in haystack]
            info["required_hits"][key] = hits
            if hits:
                score += min(len(hits), 2) * 1.2
            elif values:
                score -= 1.0

        for key, values in avoid.items():
            hits = [value for value in values if value in haystack]
            info["avoid_hits"][key] = hits
            if hits:
                score -= min(len(hits), 3) * 2.0
                if key in {"scope", "entity", "time"}:
                    rejected = True

        for pattern in prefer_patterns:
            if pattern in haystack:
                score += 1.0

        for pattern in avoid_patterns:
            if pattern in haystack:
                score -= 1.5
                if "north america" in pattern or "regional" in pattern:
                    rejected = True

        for target in targets[:8]:
            if target in haystack:
                info["target_hits"].append(target)
                score += 0.5

        return score, rejected, info

    def _extract_query_phrases(self, search_query: str) -> List[str]:
        keywords = self._extract_query_keywords(search_query)
        phrases: List[str] = []
        for n in (2, 3):
            for idx in range(0, max(0, len(keywords) - n + 1)):
                phrase = " ".join(keywords[idx: idx + n]).strip()
                if len(phrase) >= 8 and phrase not in phrases:
                    phrases.append(phrase)
        return phrases[:8]

    def _extract_query_entities(self, search_query: str) -> Dict[str, List[str]]:
        query = str(search_query or "")
        lower = query.lower()

        tickers: List[str] = []
        names: List[str] = []
        aliases: List[str] = []

        for match in re.findall(r"\(([A-Z]{1,6}):\s*([A-Z]{1,6})\)", query):
            exchange, ticker = match
            ticker = ticker.strip().lower()
            if ticker and ticker not in tickers:
                tickers.append(ticker)
            if exchange and exchange.strip().lower() not in aliases:
                aliases.append(exchange.strip().lower())

        for match in re.findall(r"\(([A-Z]{1,6})\)", query):
            ticker = match.strip().lower()
            if ticker and ticker not in tickers and ticker not in {"usd", "eur", "gaap"}:
                tickers.append(ticker)

        company_patterns = [
            r"what adjustments does\s+([A-Z][A-Za-z0-9&\.\- ]{1,80}?)\s+\(",
            r"what does\s+([A-Z][A-Za-z0-9&\.\- ]{1,80}?)\s+make",
            r"for\s+([A-Z][A-Za-z0-9&\.\- ]{1,80}?)\s+\(",
        ]
        for pattern in company_patterns:
            match = re.search(pattern, query)
            if match:
                candidate = re.sub(r"\s+", " ", match.group(1)).strip(" ,.-")
                if candidate and candidate.lower() not in names:
                    names.append(candidate.lower())

        capitalized_chunks = re.findall(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b", query)
        for chunk in capitalized_chunks:
            normalized = re.sub(r"\s+", " ", chunk).strip().lower()
            if normalized in ENTITY_STOPWORDS or len(normalized) < 3:
                continue
            if normalized not in names:
                names.append(normalized)

        for name in list(names):
            parts = [part for part in re.split(r"[\s&\.\-]+", name) if part and part not in ENTITY_STOPWORDS]
            for part in parts:
                if len(part) >= 3 and part not in aliases:
                    aliases.append(part)

        return {
            "tickers": tickers[:4],
            "names": names[:4],
            "aliases": aliases[:8],
        }

    def _score_entity_alignment(self, search_query: str, title: str, url: str, text: str) -> tuple[float, bool, Dict[str, Any]]:
        entities = self._extract_query_entities(search_query)
        haystack_title = f"{title}\n{url}".lower()
        haystack_body = str(text or "")[:12000].lower()
        haystack = f"{haystack_title}\n{haystack_body}"
        haystack_front = f"{haystack_title}\n{str(text or '')[:2500].lower()}"

        score = 0.0
        matched_name = False
        matched_ticker = False
        matched_name_title = False
        matched_ticker_title = False

        for ticker in entities["tickers"]:
            if ticker and re.search(rf"\b{re.escape(ticker)}\b", haystack):
                matched_ticker = True
                score += 3.5
            if ticker and re.search(rf"\b{re.escape(ticker)}\b", haystack_title):
                matched_ticker_title = True

        for name in entities["names"]:
            if not name:
                continue
            compact = name.replace(" ", "")
            if name in haystack or compact in haystack.replace(" ", ""):
                matched_name = True
                score += 4.0
            if name in haystack_title or compact in haystack_title.replace(" ", ""):
                matched_name_title = True

        alias_hits = 0
        front_alias_hits = 0
        for alias in entities["aliases"]:
            if alias and re.search(rf"\b{re.escape(alias)}\b", haystack):
                alias_hits += 1
            if alias and re.search(rf"\b{re.escape(alias)}\b", haystack_front):
                front_alias_hits += 1
        score += min(alias_hits, 3) * 0.8
        if matched_name_title or matched_ticker_title:
            score += 1.8
        elif front_alias_hits > 0:
            score += 0.8

        strong_subject = bool(entities["tickers"] or entities["names"])
        reject = False
        if strong_subject and not matched_name and not matched_ticker and alias_hits == 0:
            reject = True
            score -= 8.0
        elif strong_subject and not matched_name and not matched_ticker:
            score -= 3.0
        elif strong_subject and not matched_name_title and not matched_ticker_title and front_alias_hits == 0:
            score -= 2.0

        info = {
            "entities": entities,
            "matched_name": matched_name,
            "matched_ticker": matched_ticker,
            "matched_name_title": matched_name_title,
            "matched_ticker_title": matched_ticker_title,
            "alias_hits": alias_hits,
            "front_alias_hits": front_alias_hits,
        }
        return score, reject, info

    def _score_text_block(self, text: str, search_query: str = "") -> float:
        block = str(text or "")
        low = block.lower()
        query_keywords = self._extract_query_keywords(search_query)
        query_phrases = self._extract_query_phrases(search_query)
        score = 0.0
        for phrase in query_phrases:
            if phrase in low:
                score += 2.0
        for keyword in query_keywords:
            if keyword in low:
                score += 1.2
        for keyword in GENERIC_EVIDENCE_HINTS:
            if keyword in low:
                score += 0.35
        if re.search(r"adjusted\s+ebitda", low):
            score += 2.5
        if re.search(r"net income|net loss", low):
            score += 1.8
        if low.count("\n") >= 6:
            score += 0.3
        if len(re.findall(r"\b\d[\d,\.]*\b", block)) >= 3:
            score += 0.5
        return score

    def _extract_relevant_passages(
        self,
        text: str,
        search_query: str,
        max_passages: int = 5,
        passage_chars: int = 6000,
    ) -> List[str]:
        source = str(text or "")
        if not source:
            return []
        if len(source) <= passage_chars:
            return [source]

        step = max(1500, passage_chars // 2)
        windows: List[tuple[float, int, str]] = []
        for start in range(0, len(source), step):
            chunk = source[start: start + passage_chars]
            if len(chunk.strip()) < 400:
                continue
            score = self._score_text_block(chunk, search_query)
            if start == 0:
                score += 0.2
            windows.append((score, start, chunk))

        windows.sort(key=lambda item: item[0], reverse=True)
        picked: List[tuple[int, str]] = []
        seen_ranges: List[tuple[int, int]] = []
        for score, start, chunk in windows:
            if score <= 0:
                continue
            end = start + len(chunk)
            if any(abs(start - existing_start) < passage_chars // 2 for existing_start, _ in seen_ranges):
                continue
            seen_ranges.append((start, end))
            picked.append((start, chunk))
            if len(picked) >= max_passages:
                break

        if not picked:
            return [source[:passage_chars]]
        picked.sort(key=lambda item: item[0])
        return [chunk for _, chunk in picked]

    def _build_storage_document_view(self, text: str, search_query: str) -> str:
        max_chars = int(os.getenv("PRIMARY_FILING_DOC_MAX_CHARS", "0"))
        cleaned = str(text or "")
        if max_chars > 0 and len(cleaned) > max_chars:
            return cleaned[:max_chars]
        return cleaned

    def _primary_filing_keys(self) -> List[str]:
        keys = [
            key
            for key in self.data_storage.keys()
            if PRIMARY_FILING_KEY_PATTERN.match(str(key or ""))
        ]
        return sorted(
            keys,
            key=lambda item: (
                0 if item == "primary_filing" else 1,
                0 if item == "primary_filing_manifest" else 1,
                item,
            ),
        )

    def _build_handoff_document_view(self, text: str, query_hint: str = "") -> str:
        cleaned = str(text or "")
        if not cleaned:
            return ""
        max_chars = int(os.getenv("PRIMARY_FILING_HANDOFF_MAX_CHARS", "120000"))
        passage_chars = int(os.getenv("PRIMARY_FILING_HANDOFF_PASSAGE_CHARS", "12000"))
        max_passages = int(os.getenv("PRIMARY_FILING_HANDOFF_MAX_PASSAGES", "10"))
        if not query_hint.strip():
            return cleaned[:max_chars] if max_chars > 0 else cleaned
        passages = self._extract_relevant_passages(
            cleaned,
            query_hint,
            max_passages=max_passages,
            passage_chars=passage_chars,
        )
        if not passages:
            return cleaned[:max_chars] if max_chars > 0 else cleaned
        rendered = []
        for idx, passage in enumerate(passages, start=1):
            rendered.append(f"[Handoff Passage {idx}]\n{passage}")
        combined = "\n\n".join(rendered).strip()
        if max_chars > 0:
            return combined[:max_chars]
        return combined

    def export_storage_artifacts(self, query_hint: str = "", include_full_docs: bool = False) -> Dict[str, Any]:
        exported_keys = self._primary_filing_keys()
        exported_storage: Dict[str, str] = {}
        for key in exported_keys:
            if key not in self.data_storage:
                continue
            value = self.data_storage[key]
            if re.match(r"^primary_filing_[1-5]$", key):
                exported_storage[key] = (
                    str(value)
                    if include_full_docs
                    else self._build_handoff_document_view(str(value), query_hint=query_hint)
                )
            else:
                exported_storage[key] = str(value)
        return {
            "storage_namespace": "finance_primary_filing",
            "artifact_mode": "full_docs" if include_full_docs else "query_topk_passages",
            "query_hint": str(query_hint or "")[:1000],
            "data_storage": exported_storage,
            "available_keys": exported_keys,
        }

    def clear_storage_artifacts(self) -> Dict[str, Any]:
        removed_keys: List[str] = []
        for key in list(self.data_storage.keys()):
            if PRIMARY_FILING_KEY_PATTERN.match(str(key or "")):
                removed_keys.append(str(key))
                self.data_storage.pop(key, None)
        return {"removed_keys": sorted(removed_keys)}

    def _load_primary_filing_manifest(self, raw_manifest: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_manifest, list):
            return [item for item in raw_manifest if isinstance(item, dict)]
        if not raw_manifest:
            return []
        try:
            loaded = json.loads(str(raw_manifest))
        except Exception:
            return []
        return [item for item in loaded if isinstance(item, dict)] if isinstance(loaded, list) else []

    def _merge_primary_filing_bundle_text(self, existing_text: str, incoming_text: str) -> str:
        existing = str(existing_text or "").strip()
        incoming = str(incoming_text or "").strip()
        if not incoming:
            return existing
        if not existing:
            return incoming
        if incoming in existing:
            return existing
        if existing in incoming:
            return incoming
        return f"{existing}\n\n[Imported bundle]\n{incoming}".strip()

    def _manifest_identity(self, item: Dict[str, Any]) -> tuple[str, str]:
        if not isinstance(item, dict):
            return ("", "")
        return (
            str(item.get("url", "") or "").strip(),
            str(item.get("title", "") or "").strip(),
        )

    def _passage_token_set(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{3,}", str(text or "").lower()))

    def _passage_similarity(self, left: str, right: str) -> float:
        left_text = str(left or "").strip()
        right_text = str(right or "").strip()
        if not left_text or not right_text:
            return 0.0
        if left_text == right_text:
            return 1.0
        if left_text in right_text or right_text in left_text:
            shorter = min(len(left_text), len(right_text))
            longer = max(len(left_text), len(right_text))
            if longer > 0:
                return shorter / longer
        left_tokens = self._passage_token_set(left_text)
        right_tokens = self._passage_token_set(right_text)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return overlap / union if union else 0.0

    def _passage_coverage_score(self, text: str, query_hint: str = "") -> float:
        content = str(text or "")
        if not content.strip():
            return 0.0
        score = float(len(content))
        score += self._score_text_block(content[:20000], query_hint or "")
        digit_hits = len(re.findall(r"\b\d[\d,\.]*\b", content))
        score += min(digit_hits, 12) * 80.0
        heading_hits = content.count("[Handoff Passage")
        score += heading_hits * 120.0
        return score

    def _pick_preferred_passage(self, existing_text: str, incoming_text: str, query_hint: str = "") -> str:
        existing = str(existing_text or "").strip()
        incoming = str(incoming_text or "").strip()
        if not existing:
            return incoming
        if not incoming:
            return existing
        existing_score = self._passage_coverage_score(existing, query_hint=query_hint)
        incoming_score = self._passage_coverage_score(incoming, query_hint=query_hint)
        return incoming if incoming_score > existing_score else existing

    def _find_matching_primary_slot(
        self,
        incoming_text: str,
        incoming_identity: tuple[str, str],
        query_hint: str = "",
        existing_identities: Optional[Dict[str, tuple[str, str]]] = None,
    ) -> str:
        existing_identities = existing_identities or {}
        if incoming_identity != ("", ""):
            for key, identity in existing_identities.items():
                if identity == incoming_identity and str(self.data_storage.get(key, "") or "").strip():
                    return key

        best_key = ""
        best_score = 0.0
        for idx in range(1, 6):
            key = f"primary_filing_{idx}"
            existing_text = str(self.data_storage.get(key, "") or "").strip()
            if not existing_text:
                continue
            similarity = self._passage_similarity(existing_text, incoming_text)
            if similarity < 0.72:
                continue
            coverage_bonus = min(
                self._passage_coverage_score(incoming_text, query_hint=query_hint),
                self._passage_coverage_score(existing_text, query_hint=query_hint),
            ) / 100000.0
            score = similarity + coverage_bonus
            if score > best_score:
                best_score = score
                best_key = key
        return best_key

    def import_storage_artifacts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {"imported_keys": []}
        storage = payload.get("data_storage")
        if not isinstance(storage, dict):
            return {"imported_keys": []}

        incoming_primary = {
            str(key): str(value)
            for key, value in storage.items()
            if PRIMARY_FILING_KEY_PATTERN.match(str(key or ""))
        }
        if not incoming_primary:
            return {"imported_keys": []}

        query_hint = str(payload.get("query_hint", "") or "")
        imported_keys: List[str] = []
        existing_manifest = self._load_primary_filing_manifest(self.data_storage.get("primary_filing_manifest"))
        incoming_manifest = self._load_primary_filing_manifest(incoming_primary.get("primary_filing_manifest"))
        existing_identities: Dict[str, tuple[str, str]] = {}
        for item in existing_manifest:
            key = str(item.get("key", "") or "").strip()
            if re.match(r"^primary_filing_[1-5]$", key):
                existing_identities[key] = self._manifest_identity(item)
        incoming_identity_by_key: Dict[str, tuple[str, str]] = {}
        for item in incoming_manifest:
            key = str(item.get("key", "") or "").strip()
            if re.match(r"^primary_filing_[1-5]$", key):
                incoming_identity_by_key[key] = self._manifest_identity(item)

        incoming_doc_map: Dict[str, str] = {}
        for key in sorted(incoming_primary.keys()):
            if not re.match(r"^primary_filing_[1-5]$", key):
                continue
            value = str(incoming_primary.get(key, "") or "").strip()
            if not value:
                continue
            incoming_identity = incoming_identity_by_key.get(key, ("", ""))
            assigned_key = self._find_matching_primary_slot(
                value,
                incoming_identity,
                query_hint=query_hint,
                existing_identities=existing_identities,
            )
            if not assigned_key:
                assigned_key = next(
                    (f"primary_filing_{idx}" for idx in range(1, 6) if not str(self.data_storage.get(f"primary_filing_{idx}", "") or "").strip()),
                    "",
                )
                if not assigned_key:
                    continue
                self.data_storage[assigned_key] = value
            else:
                self.data_storage[assigned_key] = self._pick_preferred_passage(
                    str(self.data_storage.get(assigned_key, "") or ""),
                    value,
                    query_hint=query_hint,
                )
            if incoming_identity != ("", ""):
                existing_identities[assigned_key] = incoming_identity
            incoming_doc_map[key] = assigned_key
            if assigned_key not in imported_keys:
                imported_keys.append(assigned_key)

        if "primary_filing" in incoming_primary:
            merged_bundle = self._merge_primary_filing_bundle_text(
                str(self.data_storage.get("primary_filing", "") or ""),
                incoming_primary.get("primary_filing", ""),
            )
            if merged_bundle:
                self.data_storage["primary_filing"] = merged_bundle
                imported_keys.append("primary_filing")

        merged_manifest: List[Dict[str, Any]] = list(existing_manifest)
        seen_manifest_ids = {
            (
                str(item.get("url", "") or "").strip(),
                str(item.get("title", "") or "").strip(),
                str(item.get("key", "") or "").strip(),
            )
            for item in merged_manifest
        }
        for item in incoming_manifest:
            normalized = dict(item)
            original_key = str(normalized.get("key", "") or "").strip()
            if original_key in incoming_doc_map:
                normalized["key"] = incoming_doc_map[original_key]
            identity = (
                str(normalized.get("url", "") or "").strip(),
                str(normalized.get("title", "") or "").strip(),
                str(normalized.get("key", "") or "").strip(),
            )
            if identity in seen_manifest_ids:
                continue
            seen_manifest_ids.add(identity)
            merged_manifest.append(normalized)
        if merged_manifest:
            self.data_storage["primary_filing_manifest"] = json.dumps(merged_manifest[:5], ensure_ascii=False, indent=2)
            imported_keys.append("primary_filing_manifest")

        return {"imported_keys": sorted(set(imported_keys))}

    def _fetch_and_clean_page(self, url: str) -> Dict[str, Any]:
        response = requests.get(
            url,
            timeout=60,
            headers={"User-Agent": "ValsAI/antoine@vals.ai"},
        )
        response.raise_for_status()
        content_type = str(response.headers.get("Content-Type", ""))
        raw_bytes = response.content or b""

        if "pdf" in content_type.lower() or str(url or "").lower().endswith(".pdf"):
            reader = PdfReader(BytesIO(raw_bytes))
            pdf_text_parts: List[str] = []
            for page in reader.pages:
                try:
                    extracted = page.extract_text() or ""
                except Exception:
                    extracted = ""
                if extracted.strip():
                    pdf_text_parts.append(extracted)
            cleaned_text = self._clean_plain_text("\n".join(pdf_text_parts))
        elif any(token in content_type.lower() for token in ["html", "xml"]):
            raw_text = response.text or raw_bytes.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(raw_text, "html.parser")
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
            cleaned_text = self._clean_plain_text(soup.get_text())
        else:
            cleaned_text = self._clean_plain_text(raw_bytes.decode("utf-8", errors="ignore"))

        return {
            "content_type": content_type,
            "text": cleaned_text,
            "status_code": int(response.status_code),
        }

    def _looks_like_bad_filing(self, url: str, title: str, text: str) -> bool:
        joined = f"{url}\n{title}\n{text[:2000]}".lower()
        if "ex-32" in joined or "exhibit 32" in joined or "sarbanes-oxley" in joined:
            return True
        xbrl_markers = [
            "xbrl",
            "contextref=",
            "us-gaap:",
            "dei:",
            "link:schemaref",
            "<xml",
        ]
        marker_hits = sum(1 for marker in xbrl_markers if marker in joined)
        if "sec.gov" in str(url or "").lower() and str(url or "").lower().endswith((".htm", ".html")):
            return False
        if marker_hits >= 3 and len(text) < 25000:
            return True
        return False

    def _extract_query_keywords(self, search_query: str) -> List[str]:
        stopwords = {
            "the", "a", "an", "of", "to", "in", "is", "are", "was", "were",
            "for", "and", "or", "from", "this", "that", "with", "on", "by",
            "what", "which", "who", "when", "where", "how", "why", "does",
            "do", "did", "make", "made", "derive", "into", "about", "latest",
            "most", "recent", "report", "filing", "document", "page", "company",
            "inc", "corp", "ltd", "plc", "nasdaq", "nyse",
        }
        words = re.findall(r"[a-z][a-z0-9\-]{2,}", (search_query or "").lower())
        deduped: List[str] = []
        seen: set[str] = set()
        for word in words:
            if word in stopwords or word in seen:
                continue
            seen.add(word)
            deduped.append(word)
        return deduped[:12]

    def _score_relevance(self, text: str, search_query: str = "") -> float:
        passages = self._extract_relevant_passages(text, search_query, max_passages=4, passage_chars=5000)
        if not passages:
            return 0.0
        return max(self._score_text_block(passage, search_query) for passage in passages)

    def _summarize_document(self, text: str, search_query: str, url: str, title: str) -> str:
        passages = self._extract_relevant_passages(text, search_query, max_passages=3, passage_chars=4500)
        excerpt = "\n\n---\n\n".join(passages)[:15000]
        prompt = (
            "You are summarizing one financial filing candidate for downstream retrieval.\n"
            f"Task: {search_query}\n"
            f"Source title: {title}\n"
            f"Source URL: {url}\n\n"
            "Read the document excerpt and produce a concise summary with:\n"
            "1. Whether it contains a reconciliation or adjustment items relevant to the task.\n"
            "2. Any concrete line items or figures found.\n"
            "3. A short verdict: RELEVANT or NOT_RELEVANT.\n"
            "Return plain text only.\n\n"
            f"Document excerpt:\n{excerpt}"
        )
        try:
            response = completion(
                model=self.llm_model,
                num_retries=5,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.llm_api_base or None,
                api_key=self.llm_api_key,
            )
            content = response.choices[0].message.content or ""
            return str(content).strip()
        except Exception:
            snippet_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            picked: List[str] = []
            query_keywords = self._extract_query_keywords(search_query)
            for line in snippet_lines:
                low = line.lower()
                if any(keyword in low for keyword in query_keywords) or any(keyword in low for keyword in GENERIC_EVIDENCE_HINTS):
                    picked.append(line)
                if len(picked) >= 5:
                    break
            if not picked:
                picked = snippet_lines[:3]
            verdict = "RELEVANT" if self._score_relevance(text, search_query) >= 2.5 else "NOT_RELEVANT"
            return (
                f"{verdict}\n"
                f"Source: {title or url}\n"
                + "\n".join(picked[:5])
            ).strip()

    def _extract_relevant_excerpts(self, text: str, search_query: str, max_excerpts: int = 5) -> List[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []
        scored: List[tuple[float, str]] = []
        for line in lines:
            score = self._score_text_block(line, search_query)
            if any(ch.isdigit() for ch in line):
                score += 0.4
            if len(line) > 30:
                score += 0.1
            if score > 0:
                scored.append((score, line))
        scored.sort(key=lambda item: item[0], reverse=True)
        excerpts: List[str] = []
        seen: set[str] = set()
        for _, line in scored:
            if line in seen:
                continue
            seen.add(line)
            excerpts.append(line)
            if len(excerpts) >= max_excerpts:
                break
        return excerpts

    def _store_primary_filing_bundle(
        self,
        docs: List[Dict[str, Any]],
        search_query: str,
    ) -> None:
        bundle_parts = [
            f"PRIMARY_FILING_BUNDLE\nTask: {search_query}\nRelevant document count: {len(docs)}\n"
            "This bundle contains only source metadata, summaries, and high-relevance excerpts.\n"
            "For full details, read the underlying document keys primary_filing_1 ... primary_filing_5.\n"
        ]
        manifest: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            key = f"primary_filing_{idx}"
            self.data_storage[key] = self._build_storage_document_view(doc["text"], search_query)
            summary = doc["summary"]
            excerpts = self._extract_relevant_excerpts(doc["text"], search_query, max_excerpts=5)
            bundle_parts.append(
                f"[Document {idx}]\n"
                f"key: {key}\n"
                f"title: {doc['title']}\n"
                f"form_type: {doc.get('form_type', '')}\n"
                f"url: {doc['url']}\n"
                f"relevance_score: {doc['relevance_score']:.2f}\n"
                f"summary:\n{summary}\n\n"
                "high_relevance_excerpts:\n"
                + "\n".join(f"- {excerpt}" for excerpt in excerpts)
                + "\n"
            )
            manifest.append(
                {
                    "key": key,
                    "title": doc["title"],
                    "form_type": doc.get("form_type", ""),
                    "url": doc["url"],
                    "relevance_score": round(float(doc["relevance_score"]), 3),
                    "summary": summary,
                    "high_relevance_excerpts": excerpts,
                }
            )
        self.data_storage["primary_filing"] = "\n\n".join(bundle_parts)
        self.data_storage["primary_filing_manifest"] = json.dumps(manifest, ensure_ascii=False, indent=2)

    def env_done(self) -> bool:
        return self.is_done

    def get_env_output(self) -> FinanceEnvOutput:
        return FinanceEnvOutput(output=self.current_answer, success=self.is_done)

    # --- 工具 1: 提交答案 ---
    @cls_tool
    def submit_final_result(self, final_result: str) -> str:
        """
        Submits the final answer to the user. You should include your final answer, as well as any necessary 
        reasoning, justification, calculations, and explanation. Finally, you should provide any sources used to answer the question.

        This tool records the current best final answer in the environment state.
        You may continue working after calling this tool in the same round.
        If called multiple times, the latest submitted answer overwrites earlier submissions.
        """
        self.current_answer = final_result
        self.is_done = True
        return json.dumps({"success": True, "result": final_result}, indent=2)

    # --- 工具 2: 网络搜索 ---
    @cls_tool
    def web_search(self, search_query: str, start_date: str = None, end_date: str = None, number_of_results: int = 10) -> str:
        """
        Search the public internet for information. Each result will contain a url, a title, and one excerpt taken directly from the page.
        """
        return self._run_web_search(
            search_query=search_query,
            start_date=start_date,
            end_date=end_date,
            number_of_results=number_of_results,
        )

    # --- 工具 3: SEC 搜索 ---
    
    @cls_tool
    def edgar_search(
        self, 
        search_query: str, 
        start_date: str = "1900-01-01", 
        end_date: str = MAX_END_DATE, 
        top_n_results: int = 100,
        page: int = 1,
        form_types: Optional[List[str]] = None,
        ciks: Optional[List[str]] = None
    ) -> str:
        """
        Search the EDGAR Database through the SEC API.
        You should provide a search query. You can also optionally provide a start date, an end date, a page number, top N results, a list of form types, and/or a list of CIKs.
        The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing.
        """
        return self._run_edgar_search(
            search_query=search_query,
            start_date=start_date,
            end_date=end_date,
            top_n_results=top_n_results,
            page=page,
            form_types=form_types,
            ciks=ciks,
        )
    
    
    # --- 工具 4: 网页解析 ---
    @cls_tool
    def parse_html_page(self, url: str, key: str) -> str:
        """
        This tool is used to parse the contents of an HTML page and save it to the agent's data storage system. 

        The tool will retrieve the HTML page from the URL provided, then parse it from HTML to plain text. 
        Finally, it will save it to the agent's data storage system under the key provided.
        
        You can use the retrieve_information tool to later retrieve information about the stored page.
        """
        if not url or not key:
            return json.dumps({"success": False, "result": "URL and key are required"})

        try:
            parsed = self._fetch_and_clean_page(url)
            cleaned_text = parsed["text"]
            msg = ""
            if key in self.data_storage:
                msg += (
                    f"WARNING: The key '{key}' already existed. "
                    "Overwriting to keep a consistent retrieval key.\n"
                )

            self.data_storage[key] = cleaned_text

            msg += f"SUCCESS: The result has been saved to the data storage under the key: '{key}'.\n"
            keys_list = "\n".join(self.data_storage.keys())
            msg += f"The data_storage currently contains the following keys:\n{keys_list}"
            
            return json.dumps({"success": True, "result": msg}) # 原版 success=True 包装

        except Exception as e:
            return json.dumps({"success": False, "result": f"Failed to parse page: {str(e)}"})

    @cls_tool
    def prepare_primary_filing(
        self,
        search_query: str,
        top_n_results: int = 10,
        max_valid_docs: int = 5,
        include_web_results: bool = True,
        required_constraints: Optional[Dict[str, Any]] = None,
        avoid_constraints: Optional[Dict[str, Any]] = None,
        prefer_source_patterns: Optional[List[str]] = None,
        avoid_source_patterns: Optional[List[str]] = None,
        missing_targets: Optional[List[str]] = None,
    ) -> str:
        """
        Deterministically build `primary_filing` for downstream agents.

        The tool searches SEC filings (and optionally web results), iterates through the top candidate URLs,
        parses them programmatically, keeps up to `max_valid_docs` relevant documents, stores them as
        `primary_filing_1..N`, and stores a combined multi-document bundle in `primary_filing`.
        """
        max_valid_docs = max(1, min(int(max_valid_docs), 5))
        top_n_results = max(1, min(int(top_n_results), 20))

        derived_constraints = self._derive_prepare_primary_filing_constraints(search_query)
        required_constraints = self._merge_constraint_maps(
            derived_constraints.get("required_constraints"),
            required_constraints,
        )
        avoid_constraints = self._merge_constraint_maps(
            derived_constraints.get("avoid_constraints"),
            avoid_constraints,
        )
        prefer_source_patterns = self._merge_pattern_lists(
            derived_constraints.get("prefer_source_patterns"),
            prefer_source_patterns,
        )
        avoid_source_patterns = self._merge_pattern_lists(
            derived_constraints.get("avoid_source_patterns"),
            avoid_source_patterns,
        )
        merged_missing_targets: List[str] = []
        for value in list(derived_constraints.get("missing_targets", []) or []) + list(missing_targets or []):
            cleaned = str(value or "").strip().lower()
            if cleaned and cleaned not in merged_missing_targets:
                merged_missing_targets.append(cleaned)
        missing_targets = merged_missing_targets[:8]

        sec_results: List[Dict[str, Any]] = []
        web_results: List[Dict[str, Any]] = []
        attempts: List[Dict[str, Any]] = []

        try:
            sec_payload = self._run_edgar_search(
                search_query=search_query,
                top_n_results=top_n_results,
            )
            sec_loaded = json.loads(sec_payload)
            if isinstance(sec_loaded, list):
                sec_results = sec_loaded
        except Exception:
            sec_results = []

        if include_web_results:
            try:
                web_payload = self._run_web_search(
                    search_query=search_query,
                    number_of_results=top_n_results,
                )
                web_loaded = json.loads(web_payload)
                if isinstance(web_loaded, list):
                    web_results = web_loaded
            except Exception:
                web_results = []

        candidates = self._extract_candidate_urls(sec_results) + self._extract_candidate_urls(web_results)
        candidates.sort(
            key=lambda item: (
                float(item.get("form_score", 0.0)) + self._score_source_quality(
                    item.get("url", ""),
                    item.get("title", ""),
                    item.get("source_type", ""),
                    item.get("form_type", ""),
                ) + self._score_constraint_alignment(
                    title=str(item.get("title", "") or ""),
                    url=str(item.get("url", "") or ""),
                    text="",
                    required_constraints=required_constraints,
                    avoid_constraints=avoid_constraints,
                    prefer_source_patterns=prefer_source_patterns,
                    avoid_source_patterns=avoid_source_patterns,
                    missing_targets=missing_targets,
                )[0] + self._score_entity_alignment(
                    search_query,
                    str(item.get("title", "") or ""),
                    str(item.get("url", "") or ""),
                    "",
                )[0],
                1.0 if item.get("source_type") == "sec" else 0.0,
            ),
            reverse=True,
        )
        valid_docs: List[Dict[str, Any]] = []

        for candidate in candidates[: top_n_results * 2]:
            if len(valid_docs) >= max_valid_docs:
                break
            url = candidate["url"]
            title = candidate["title"]
            form_type = str(candidate.get("form_type", "") or "")
            source_type = str(candidate.get("source_type", "") or "")
            try:
                parsed = self._fetch_and_clean_page(url)
                text = parsed["text"]
                if len(text) < 1000:
                    attempts.append({"url": url, "title": title, "form_type": form_type, "status": "skip_short"})
                    continue
                if self._looks_like_bad_filing(url, title, text):
                    attempts.append({"url": url, "title": title, "form_type": form_type, "status": "skip_bad_filing"})
                    continue
                entity_score, reject_for_entity_mismatch, entity_info = self._score_entity_alignment(
                    search_query,
                    title,
                    url,
                    text,
                )
                if reject_for_entity_mismatch:
                    attempts.append(
                        {
                            "url": url,
                            "title": title,
                            "form_type": form_type,
                            "status": "skip_entity_mismatch",
                            "entity_info": entity_info,
                        }
                    )
                    continue
                source_quality = self._score_source_quality(
                    url,
                    title,
                    source_type,
                    form_type,
                )
                constraint_score, reject_for_constraints, constraint_info = self._score_constraint_alignment(
                    title=title,
                    url=url,
                    text=text,
                    required_constraints=required_constraints,
                    avoid_constraints=avoid_constraints,
                    prefer_source_patterns=prefer_source_patterns,
                    avoid_source_patterns=avoid_source_patterns,
                    missing_targets=missing_targets,
                )
                if reject_for_constraints:
                    attempts.append(
                        {
                            "url": url,
                            "title": title,
                            "form_type": form_type,
                            "status": "skip_constraint_mismatch",
                            "constraint_info": constraint_info,
                        }
                    )
                    continue
                relevance_score = (
                    self._score_relevance(text, search_query)
                    + float(candidate.get("form_score", 0.0))
                    + source_quality
                    + entity_score
                    + constraint_score
                )
                if relevance_score < 2.5:
                    attempts.append({
                        "url": url,
                        "title": title,
                        "form_type": form_type,
                        "status": "skip_low_relevance",
                        "relevance_score": relevance_score,
                        "constraint_info": constraint_info,
                    })
                    continue
                summary = self._summarize_document(text, search_query, url, title)
                if "NOT_RELEVANT" in summary.upper():
                    attempts.append(
                        {
                            "url": url,
                            "title": title,
                            "form_type": form_type,
                            "status": "skip_summary_not_relevant",
                            "relevance_score": relevance_score,
                            "constraint_info": constraint_info,
                        }
                    )
                    continue
                excerpts = self._extract_relevant_excerpts(text, search_query, max_excerpts=5)
                valid_docs.append(
                    {
                        "url": url,
                        "title": title or urlparse(url).path.rsplit("/", 1)[-1],
                        "form_type": form_type,
                        "source_type": str(candidate.get("source_type", "") or ""),
                        "text": text,
                        "summary": summary[:3000],
                        "excerpts": excerpts,
                        "relevance_score": relevance_score,
                        "constraint_info": constraint_info,
                    }
                )
                attempts.append({
                    "url": url,
                    "title": title,
                    "form_type": form_type,
                    "status": "accepted",
                    "relevance_score": relevance_score,
                    "constraint_info": constraint_info,
                })
            except Exception as exc:
                attempts.append({"url": url, "title": title, "form_type": form_type, "status": f"parse_failed: {exc}"})

        valid_docs.sort(key=lambda item: item["relevance_score"], reverse=True)
        valid_docs = valid_docs[:max_valid_docs]

        if not valid_docs:
            return json.dumps(
                {
                    "success": False,
                    "result": "No valid relevant filings were prepared for primary_filing.",
                    "attempts": attempts[: top_n_results * 2],
                },
                ensure_ascii=False,
                indent=2,
            )

        self._store_primary_filing_bundle(valid_docs, search_query)
        return json.dumps(
            {
                "success": True,
                "result": {
                    "prepared_key": "primary_filing",
                    "valid_doc_count": len(valid_docs),
                    "doc_keys": [f"primary_filing_{i}" for i in range(1, len(valid_docs) + 1)],
                    "manifest_key": "primary_filing_manifest",
                    "top_sources": [
                        {
                            "key": f"primary_filing_{idx}",
                            "title": doc["title"],
                            "url": doc["url"],
                            "relevance_score": round(float(doc["relevance_score"]), 3),
                            "summary": doc["summary"][:500],
                        }
                        for idx, doc in enumerate(valid_docs, start=1)
                    ],
                    "attempts": attempts[: top_n_results * 2],
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    # --- 工具 5: 信息检索 ---
    @cls_tool
    def retrieve_information(self, prompt: str, input_character_ranges: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        This tool allows you to retrieve data from previously saved documents from the agent's data storage system, by applying an LLM prompt to the stored document.

        To use the tool, you will need to provide a prompt. This prompt will include both the query to be sent to the LLM, 
        as well as the keys of files you have previously saved to the data storage system.
        
        For example, if you want to analyze data stored under the key "financial_report", your prompt should look like the following:
        "Analyze the following financial report and extract the revenue figures: {{financial_report}}"
        
        The {{key_name}} will be replaced with the full text of the document stored under that key before the query is sent.

        IMPORTANT: Your prompt MUST include at least one key from the data storage using this exact format: {{key_name}}.
        If you don't use this exact format with double braces, the tool will fail to retrieve the information.
        
        You can also optionally only pass *a portion* of each document to the LLM, rather than the entire document. This can be used to avoid token limit errors or improve efficiency.
        To do so, use the input_character_ranges parameter to specify which portions of documents to extract.
        For example, if "financial_report" contains "Annual Report 2023" and you specify:  [{"key": "financial_report", "start": 1, "end": 6}], then only "nnual" will be inserted into the prompt (characters 1 through 5, as end is exclusive).

        NOTE ON SYNTAX: 
        The double braces {{ }} syntax is ONLY used inside the 'prompt' string for this specific tool to reference data keys.
        DO NOT use double braces for the JSON arguments of this tool or any other tool.
        """
        
        if input_character_ranges is None:
            input_character_ranges = []

        # Special mode: allow agents to inspect available storage keys safely.
        # This avoids repeated invalid calls with fake placeholders like {{placeholder}}.
        prompt_lower = (prompt or "").strip().lower()
        if (
            "list all available keys" in prompt_lower
            or "list available document keys" in prompt_lower
            or "what documents are available" in prompt_lower
        ):
            return json.dumps(
                {
                    "success": True,
                    "result": {
                        "available_keys": list(self.data_storage.keys()),
                        "num_keys": len(self.data_storage),
                    },
                },
                indent=2,
            )
            
        # 1. 提取 Prompt 中的 Keys
        # 优先找双花括号 {{key}}，兼容单花括号 {key}（agent 常见格式错误）
        keys_in_prompt = re.findall(r"\{\{([\w\.\-\_]+)\}\}", prompt)
        if not keys_in_prompt:
            keys_in_prompt = re.findall(r"\{([\w\.\-\_]+)\}", prompt)
            # 将单花括号统一替换为双花括号，让后续替换逻辑一致
            if keys_in_prompt:
                for k in keys_in_prompt:
                    prompt = prompt.replace(f"{{{k}}}", f"{{{{{k}}}}}")

        # 兼容 agent 在自然语言里直接写 primary_filing / primary_filing_1 但没有花括号。
        if not keys_in_prompt:
            bare_keys = re.findall(r"\b(primary_filing(?:_[1-5])?)\b", prompt)
            if bare_keys:
                for k in bare_keys:
                    if k in self.data_storage:
                        keys_in_prompt.append(k)
                        prompt += f"\n\n{{{{{k}}}}}"

        # 若 prompt 里完全没有 key 占位符，从 input_character_ranges 里补 key
        if not keys_in_prompt and input_character_ranges:
            range_keys = [r["key"] for r in input_character_ranges if "key" in r]
            for k in range_keys:
                if k in self.data_storage:
                    keys_in_prompt.append(k)
                    prompt = prompt + f"\n\n{{{{{k}}}}}"  # 追加到 prompt 末尾

        # 最后兜底：若存在主 bundle，则默认补 primary_filing，避免 agent 因为忘写 key 直接失败。
        if not keys_in_prompt and "primary_filing" in self.data_storage:
            keys_in_prompt.append("primary_filing")
            prompt = prompt + "\n\n{{primary_filing}}"

        if not keys_in_prompt:
            return json.dumps({"success": False, "result": "ERROR: Prompt must include at least one {{key}} to refer to stored data."})

        keys_set = set(keys_in_prompt)

        # 2. 校验 Keys 是否存在（带别名兼容与单一 key 回退）
        missing_keys = [k for k in keys_set if k not in self.data_storage]
        if missing_keys:
            # 常见别名统一到 primary_filing
            alias_map = {
                "airbnb_10k_filing": "primary_filing",
                "airbnb_10q_filing": "primary_filing",
                "shareholder_letter_q1_2024": "primary_filing",
                "shareholder_letter": "primary_filing",
                "filing": "primary_filing",
                "document": "primary_filing",
            }
            replaced = False
            for missing in list(missing_keys):
                alias = alias_map.get(missing)
                if alias and alias in self.data_storage:
                    prompt = prompt.replace(f"{{{{{missing}}}}}", f"{{{{{alias}}}}}")
                    keys_set.discard(missing)
                    keys_set.add(alias)
                    missing_keys.remove(missing)
                    replaced = True

            # 若仅有一个可用 key，自动回退到该 key
            available = list(self.data_storage.keys())
            if missing_keys and len(available) == 1:
                fallback = available[0]
                for missing in list(missing_keys):
                    prompt = prompt.replace(f"{{{{{missing}}}}}", f"{{{{{fallback}}}}}")
                    keys_set.discard(missing)
                    keys_set.add(fallback)
                    missing_keys.remove(missing)
                replaced = True

            if missing_keys:
                return json.dumps({
                    "success": False,
                    "result": f"ERROR: The following keys were not found in storage: {missing_keys}. Available keys: {available}"
                })

            if replaced:
                pass

        # 3-5. 分块 + 关键词打分 → top-k chunk → LLM 提取 → 汇总
        CHUNK_SIZE = 80_000
        CHUNK_OVERLAP = 5_000
        TOP_K_CHUNKS = int(os.getenv("RETRIEVE_TOP_K_CHUNKS", "3"))

        def _llm_call(prompt_text: str) -> str:
            system_prompt = (
                "You are extracting information from document text that has already been inserted directly into the prompt. "
                "Do not say that you cannot access external websites, PDFs, URLs, or documents. "
                "Do not mention browsing limitations. "
                "If the answer is not present in the provided text, say exactly that it is not found in the provided text."
            )
            response = completion(
                model=self.llm_model,
                num_retries=5,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                api_base=self.llm_api_base or None,
                api_key=self.llm_api_key,
            )
            content = response.choices[0].message.content
            content_text = str(content or "").strip()
            if re.search(
                r"cannot access external (websites?|documents?|urls?|pdfs?)|cannot access.*pdf|i do not have access to the document|you provided",
                content_text,
                flags=re.IGNORECASE,
            ):
                retry_prompt = (
                    "The document text is already embedded below. "
                    "Answer only from the provided text. "
                    "Never mention inability to access websites or PDFs.\n\n"
                    f"{prompt_text}"
                )
                retry_resp = completion(
                    model=self.llm_model,
                    num_retries=5,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": retry_prompt},
                    ],
                    api_base=self.llm_api_base or None,
                    api_key=self.llm_api_key,
                )
                content_text = str(retry_resp.choices[0].message.content or "").strip()
            return content_text

        def _keyword_score(text: str, keywords: list) -> float:
            """Count normalised keyword hits in a text chunk."""
            if not keywords or not text:
                return 0.0
            text_lower = text.lower()
            hits = sum(text_lower.count(kw) for kw in keywords)
            return hits / max(len(text), 1) * 10_000  # hits per 10k chars

        def _extract_keywords(query_text: str) -> list:
            """Pull meaningful words from the query for relevance scoring."""
            stopwords = {"the", "a", "an", "of", "to", "in", "is", "are", "was",
                         "for", "and", "or", "from", "this", "that", "with", "on",
                         "please", "extract", "find", "provide", "document", "chunk",
                         "following", "information", "below", "based", "answer"}
            words = re.findall(r"[a-z][a-z\-]{2,}", query_text.lower())
            return [w for w in words if w not in stopwords]

        try:
            detail_keys = sorted(
                key for key in self.data_storage.keys()
                if re.match(r"^primary_filing_[1-5]$", str(key or ""))
            )
            auto_expand_bundle = keys_set == {"primary_filing"} and bool(detail_keys)

            # 取最长的 key 作为主文档，其余 key 直接全文替换
            if auto_expand_bundle:
                search_keys = detail_keys
                query_only = prompt.replace("{{primary_filing}}", "[DOCUMENT_CHUNK]")
                bundle_context = str(self.data_storage.get("primary_filing", "") or "").strip()
                if bundle_context:
                    query_only = (
                        "Bundle context (metadata / summaries / excerpts):\n"
                        f"{bundle_context[:20000]}\n\n"
                        f"{query_only}"
                    )
            else:
                search_keys = [max(keys_set, key=lambda k: len(self.data_storage[k]))]
                longest_key = search_keys[0]
                query_only = prompt.replace(f"{{{{{longest_key}}}}}", "[DOCUMENT_CHUNK]")
                for k in keys_set - {longest_key}:
                    query_only = query_only.replace(f"{{{{{k}}}}}", self.data_storage[k])

            per_doc_answers: List[str] = []
            keywords = _extract_keywords(query_only.replace("[DOCUMENT_CHUNK]", ""))

            for search_key in search_keys:
                doc_text = self.data_storage[search_key]
                if len(doc_text) <= CHUNK_SIZE:
                    content = _llm_call(query_only.replace("[DOCUMENT_CHUNK]", doc_text))
                    if content and len(content.strip()) > 30:
                        per_doc_answers.append(f"[Source key={search_key}]\n{content.strip()}")
                    continue

                chunks = []
                pos = 0
                while pos < len(doc_text):
                    chunk = doc_text[pos: pos + CHUNK_SIZE]
                    score = _keyword_score(chunk, keywords)
                    chunks.append((score, pos, chunk))
                    pos += CHUNK_SIZE - CHUNK_OVERLAP

                chunks.sort(key=lambda x: x[0], reverse=True)
                top_chunks = chunks[:TOP_K_CHUNKS]
                top_chunks.sort(key=lambda x: x[1])

                chunk_answers = []
                for rank, (score, pos, chunk) in enumerate(top_chunks):
                    try:
                        ans = _llm_call(query_only.replace("[DOCUMENT_CHUNK]", chunk))
                        ans = (ans or "").strip()
                        if ans and len(ans) > 30:
                            chunk_answers.append(
                                f"[Chunk rank={rank+1}, chars {pos}-{pos+len(chunk)}, relevance={score:.1f}]\n{ans}"
                            )
                    except Exception:
                        pass

                if chunk_answers:
                    if len(chunk_answers) == 1:
                        per_doc_answers.append(f"[Source key={search_key}]\n{chunk_answers[0]}")
                    else:
                        combined_doc = "\n\n".join(chunk_answers)
                        summary_prompt = (
                            f"Below are answers extracted from the most relevant chunks of document key {search_key}.\n"
                            f"Original question: {query_only.replace('[DOCUMENT_CHUNK]', '').strip()}\n\n"
                            f"Partial answers:\n{combined_doc}\n\n"
                            "Please merge and deduplicate these into one concise document-level answer."
                        )
                        per_doc_answers.append(
                            f"[Source key={search_key}]\n{_llm_call(summary_prompt).strip()}"
                        )

            if not per_doc_answers:
                return json.dumps({"success": False, "result": f"No relevant information found in searched document chunks (keywords: {keywords[:10]})."})

            if len(per_doc_answers) == 1:
                return json.dumps({"success": True, "result": per_doc_answers[0]}, indent=2)

            combined = "\n\n".join(per_doc_answers)
            final_answer = _llm_call(
                "Below are answers extracted from one or more stored documents.\n"
                f"Original question: {query_only.replace('[DOCUMENT_CHUNK]', '').strip()}\n\n"
                f"Document-level answers:\n{combined}\n\n"
                "Please merge, deduplicate, and keep explicit source-key attribution in the final answer."
            )
            return json.dumps({"success": True, "result": final_answer}, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "result": f"Chunked LLM query failed: {str(e)}"})
