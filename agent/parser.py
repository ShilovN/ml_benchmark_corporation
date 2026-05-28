"""Parse LLM tool commands into validated Python objects."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedCommand:
    name: str
    args: dict[str, Any]


POSITIONAL_ARGS = {
    "list_files": ["path"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "edit_file": ["path", "diff"],
    "load_dataset": ["path"],
    "show_sample_rows": ["n"],
    "run_python": ["code_or_file"],
    "submit": ["file"],
}


NO_ARG_COMMANDS = {
    "show_dataset_info",
    "get_budget_status",
    "get_remaining_time",
    "get_trajectory",
    "get_hints",
}

COMMAND_NAMES = set(POSITIONAL_ARGS) | NO_ARG_COMMANDS
FENCED_BLOCK_RE = re.compile(r"```(?:command|tool|json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_command(text: str) -> ParsedCommand:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Command is empty")

    if stripped.startswith("{"):
        return _parse_json_command(stripped)

    return _parse_call_command(stripped)


def extract_command_text(response_text: str) -> str:
    """Extract a single executable command from a free-form LLM response."""
    stripped = response_text.strip()
    if not stripped:
        raise ValueError("Response is empty")

    if stripped.startswith("{"):
        parse_command(stripped)
        return stripped

    fenced_candidates = [
        block.strip()
        for block in FENCED_BLOCK_RE.findall(response_text)
        if block.strip()
    ]
    if fenced_candidates:
        valid_fenced = _valid_command_candidates(fenced_candidates)
        return _select_single_candidate(valid_fenced, source="fenced command block")

    line_candidates = [
        line.strip()
        for line in response_text.splitlines()
        if _looks_like_command_line(line.strip())
    ]
    if line_candidates:
        valid_lines = _valid_command_candidates(line_candidates)
        return _select_single_candidate(valid_lines, source="command line")

    inline_candidates = _find_inline_call_candidates(response_text)
    if inline_candidates:
        valid_inline = _valid_command_candidates(inline_candidates)
        return _select_single_candidate(valid_inline, source="inline command")

    raise ValueError("No executable command found in response")


def parse_model_response(response_text: str) -> ParsedCommand:
    """Extract and parse one command from a free-form model response."""
    return parse_command(extract_command_text(response_text))


def _parse_json_command(text: str) -> ParsedCommand:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("JSON command must be an object")

    name = payload.get("command") or payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("JSON command must contain a command name")

    args = payload.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("JSON command args must be an object")

    return ParsedCommand(name=name, args=args)


def _parse_call_command(text: str) -> ParsedCommand:
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid command syntax: {exc.msg}") from exc

    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("Command must look like command_name(...)")

    name = expression.func.id
    positional_names = POSITIONAL_ARGS.get(name)
    if positional_names is None and name not in NO_ARG_COMMANDS:
        raise ValueError(f"Unknown command: {name}")

    if expression.keywords:
        args = _parse_keyword_args(expression)
    else:
        args = _parse_positional_args(name, positional_names or [], expression.args)

    return ParsedCommand(name=name, args=args)


def _parse_keyword_args(expression: ast.Call) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("Expanded keyword arguments are not supported")
        args[keyword.arg] = ast.literal_eval(keyword.value)
    if expression.args:
        raise ValueError("Do not mix positional and keyword arguments")
    return args


def _parse_positional_args(name: str, positional_names: list[str], nodes: list[ast.expr]) -> dict[str, Any]:
    if name in NO_ARG_COMMANDS:
        if nodes:
            raise ValueError(f"{name} does not accept arguments")
        return {}

    if len(nodes) != len(positional_names):
        raise ValueError(f"{name} expects {len(positional_names)} argument(s), got {len(nodes)}")

    return {
        arg_name: ast.literal_eval(node)
        for arg_name, node in zip(positional_names, nodes)
    }


def _valid_command_candidates(candidates: list[str]) -> list[str]:
    valid: list[str] = []
    for candidate in candidates:
        try:
            parse_command(candidate)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        valid.append(candidate)
    return valid


def _select_single_candidate(candidates: list[str], source: str) -> str:
    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    if not unique_candidates:
        raise ValueError(f"No valid command found in {source}")
    if len(unique_candidates) > 1:
        raise ValueError(f"Multiple commands found in {source}: {len(unique_candidates)}")
    return unique_candidates[0]


def _looks_like_command_line(line: str) -> bool:
    if not line or line.startswith("#"):
        return False
    if line.startswith("{"):
        return True
    return any(line.startswith(f"{name}(") and line.endswith(")") for name in COMMAND_NAMES)


def _find_inline_call_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for name in sorted(COMMAND_NAMES):
        start = 0
        needle = f"{name}("
        while True:
            index = text.find(needle, start)
            if index == -1:
                break
            candidate = _extract_balanced_call(text, index)
            if candidate:
                candidates.append(candidate)
                start = index + len(candidate)
            else:
                start = index + len(needle)
    return candidates


def _extract_balanced_call(text: str, start: int) -> str | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {'"', "'"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
