import ast
import re
from dataclasses import dataclass
from typing import Dict, Optional

from .plancraft import PlancraftEnvironment
from .registry import register_env


@dataclass
class _ConvertedPlancraftInstance:
    id: str
    target: str
    inventory: Dict[str, int]
    slotted_inventory: Dict[int, Dict[str, int | str]]
    impossible: bool
    expected_output: str | list[str] | None = None


def _extract_target(question: str) -> str:
    match = re.search(r"Target item:\s*([^\n]+)", question, flags=re.IGNORECASE)
    return (match.group(1).strip() if match else "").lower()


def _extract_inventory_block(question: str) -> str:
    match = re.search(
        r"Inventory:\s*(.*?)(?:\n\s*\n|$)",
        question,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _parse_inventory(question: str) -> Dict[str, int]:
    block = _extract_inventory_block(question)
    inventory: Dict[str, int] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        match = re.match(r"-\s*([^:]+):\s*(\d+)", line)
        if not match:
            continue
        item = match.group(1).strip().lower()
        qty = int(match.group(2))
        if item:
            inventory[item] = qty
    return inventory


def _parse_expected_output(raw: object) -> tuple[bool, str | list[str] | None]:
    text = str(raw or "").strip()
    if not text:
        return False, None
    if text.upper() == "IMPOSSIBLE":
        return True, "IMPOSSIBLE"
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return False, parsed
    except Exception:
        pass
    return False, text


def _convert_instance(dataset_instance) -> _ConvertedPlancraftInstance:
    question = str(getattr(dataset_instance, "question", "") or "")
    target = _extract_target(question)
    inventory = _parse_inventory(question)
    slotted_inventory = {
        idx: {"type": item, "quantity": qty}
        for idx, (item, qty) in enumerate(inventory.items(), start=1)
    }
    impossible, expected = _parse_expected_output(
        getattr(dataset_instance, "expected_answer", None)
    )
    return _ConvertedPlancraftInstance(
        id=str(getattr(dataset_instance, "id", "") or ""),
        target=target,
        inventory=inventory,
        slotted_inventory=slotted_inventory,
        impossible=impossible,
        expected_output=expected,
    )


@register_env("plancraft-converted")
class PlancraftConvertedEnvironment(PlancraftEnvironment):
    def __init__(self, *args, dataset_instance=None, **kwargs):
        converted = dataset_instance
        if dataset_instance is not None and hasattr(dataset_instance, "question"):
            converted = _convert_instance(dataset_instance)
        super().__init__(*args, dataset_instance=converted, **kwargs)

