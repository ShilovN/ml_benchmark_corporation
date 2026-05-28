"""Execute parsed LLM commands in a controlled workspace."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from checker.metric_checker import compute_metric_details

from .parser import ParsedCommand, parse_command, parse_model_response


MAX_RESULT_CHARS = 8000
MAX_FILE_READ_CHARS = 20000
DEFAULT_MAX_STEPS = 100
DEFAULT_TIME_LIMIT_SECONDS = 3600


@dataclass
class DatasetState:
    path: Path
    columns: list[str]
    rows: list[dict[str, str]]


@dataclass
class AgentContext:
    workspace: Path
    task_id: str = "salary_prediction"
    tasks_dir: Path = Path("checker/tasks")
    max_steps: int = DEFAULT_MAX_STEPS
    time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS
    start_time: float = field(default_factory=time.monotonic)
    used_steps: int = 0
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    dataset: DatasetState | None = None


class CommandExecutor:
    def __init__(self, context: AgentContext) -> None:
        self.context = context
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "load_dataset": self._load_dataset,
            "show_dataset_info": self._show_dataset_info,
            "show_sample_rows": self._show_sample_rows,
            "run_python": self._run_python,
            "get_budget_status": self._get_budget_status,
            "get_remaining_time": self._get_remaining_time,
            "get_trajectory": self._get_trajectory,
            "submit": self._submit,
        }

    def execute_text(self, text: str) -> dict[str, Any]:
        return self.execute(parse_command(text))

    def execute_model_response(self, text: str) -> dict[str, Any]:
        return self.execute(parse_model_response(text))

    def execute(self, command: ParsedCommand) -> dict[str, Any]:
        started_at = time.perf_counter()
        self.context.used_steps += 1
        if self.context.used_steps > self.context.max_steps:
            result = self._error(command, "Step budget exceeded", started_at)
            self.context.trajectory.append(result)
            return result

        handler = self.handlers.get(command.name)
        if handler is None:
            result = self._error(command, f"Unknown command: {command.name}", started_at)
            self.context.trajectory.append(result)
            return result

        try:
            payload = handler(command.args)
            result = {
                "status": "ok",
                "command": command.name,
                "result": payload,
                "elapsed_ms": self._elapsed_ms(started_at),
            }
        except Exception as exc:
            result = self._error(command, str(exc), started_at)

        self.context.trajectory.append(_trajectory_record(command, result))
        return result

    def _list_files(self, args: dict[str, Any]) -> list[str]:
        directory = self._resolve_path(_required_str(args, "path"))
        if not directory.exists():
            raise ValueError(f"Path does not exist: {directory}")
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        return sorted(item.name + ("/" if item.is_dir() else "") for item in directory.iterdir())

    def _read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(_required_str(args, "path"))
        if not path.exists() or not path.is_file():
            raise ValueError(f"File does not exist: {path}")
        content = path.read_text(encoding="utf-8")
        truncated = len(content) > MAX_FILE_READ_CHARS
        if truncated:
            content = content[:MAX_FILE_READ_CHARS]
        return {"path": self._display_path(path), "content": content, "truncated": truncated}

    def _write_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(_required_str(args, "path"))
        content = _required_str(args, "content")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": self._display_path(path), "bytes_written": len(content.encode("utf-8"))}

    def _edit_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(_required_str(args, "path"))
        diff = args.get("diff")
        if not path.exists() or not path.is_file():
            raise ValueError(f"File does not exist: {path}")

        content = path.read_text(encoding="utf-8")
        old, new = _parse_edit_diff(diff)
        if old not in content:
            raise ValueError("Old text was not found in file")
        path.write_text(content.replace(old, new, 1), encoding="utf-8")
        return {"path": self._display_path(path), "replacements": 1}

    def _load_dataset(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(_required_str(args, "path"))
        if path.suffix.lower() not in {".csv", ".tsv"}:
            raise ValueError("Only CSV and TSV datasets are supported")
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            if not reader.fieldnames:
                raise ValueError("Dataset must contain a header row")
            rows = list(reader)
        self.context.dataset = DatasetState(path=path, columns=list(reader.fieldnames), rows=rows)
        return {"path": self._display_path(path), "rows": len(rows), "columns": reader.fieldnames}

    def _show_dataset_info(self, args: dict[str, Any]) -> dict[str, Any]:
        _ensure_no_args(args)
        dataset = self._require_dataset()
        missing_by_column = {
            column: sum(1 for row in dataset.rows if row.get(column, "") == "")
            for column in dataset.columns
        }
        return {
            "path": self._display_path(dataset.path),
            "rows": len(dataset.rows),
            "columns": dataset.columns,
            "missing_by_column": missing_by_column,
        }

    def _show_sample_rows(self, args: dict[str, Any]) -> list[dict[str, str]]:
        n = _required_int(args, "n")
        if n < 1:
            raise ValueError("n must be positive")
        dataset = self._require_dataset()
        return dataset.rows[:n]

    def _run_python(self, args: dict[str, Any]) -> dict[str, Any]:
        code_or_file = _required_str(args, "code_or_file")
        candidate = self._maybe_workspace_path(code_or_file)
        if candidate and candidate.exists() and candidate.is_file():
            command = ["python3", str(candidate)]
        else:
            command = ["python3", "-c", code_or_file]

        completed = subprocess.run(
            command,
            cwd=self.context.workspace,
            text=True,
            capture_output=True,
            timeout=20,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return {
            "returncode": completed.returncode,
            "stdout": _truncate(completed.stdout),
            "stderr": _truncate(completed.stderr),
        }

    def _get_budget_status(self, args: dict[str, Any]) -> dict[str, int]:
        _ensure_no_args(args)
        return {
            "max_steps": self.context.max_steps,
            "used_steps": self.context.used_steps,
            "remaining_steps": max(0, self.context.max_steps - self.context.used_steps),
        }

    def _get_remaining_time(self, args: dict[str, Any]) -> dict[str, float]:
        _ensure_no_args(args)
        elapsed = time.monotonic() - self.context.start_time
        return {
            "time_limit_seconds": self.context.time_limit_seconds,
            "elapsed_seconds": round(elapsed, 3),
            "remaining_seconds": round(max(0.0, self.context.time_limit_seconds - elapsed), 3),
        }

    def _get_trajectory(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        _ensure_no_args(args)
        return list(self.context.trajectory)

    def _submit(self, args: dict[str, Any]) -> dict[str, Any]:
        submission_path = self._resolve_path(_required_str(args, "file"))
        task_config = self._load_task_config(self.context.task_id)
        answer_path = task_config["task_dir"] / task_config["answer_file"]
        return compute_metric_details(
            answer_path,
            submission_path,
            task_config["metric"],
            column=task_config.get("column"),
            true_column=task_config.get("true_column"),
            pred_column=task_config.get("pred_column"),
            id_column=task_config.get("id_column"),
        )

    def _load_task_config(self, task_id: str) -> dict[str, Any]:
        tasks_dir = self._resolve_path(str(self.context.tasks_dir))
        task_dir = tasks_dir / task_id
        config_path = task_dir / "task.json"
        if not config_path.exists():
            raise ValueError(f"Task config not found: {task_id}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data["task_dir"] = task_dir
        return data

    def _resolve_path(self, path: str) -> Path:
        raw_path = Path(path)
        candidate = raw_path if raw_path.is_absolute() else self.context.workspace / raw_path
        resolved = candidate.resolve()
        workspace = self.context.workspace.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"Path is outside workspace: {path}")
        return resolved

    def _display_path(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.context.workspace.resolve()))

    def _maybe_workspace_path(self, value: str) -> Path | None:
        if "\n" in value or len(value) > 240:
            return None
        try:
            return self._resolve_path(value)
        except ValueError:
            return None

    def _require_dataset(self) -> DatasetState:
        if self.context.dataset is None:
            raise ValueError("Dataset is not loaded")
        return self.context.dataset

    def _error(self, command: ParsedCommand, error: str, started_at: float) -> dict[str, Any]:
        return {
            "status": "error",
            "command": command.name,
            "error": error,
            "elapsed_ms": self._elapsed_ms(started_at),
        }

    def _elapsed_ms(self, started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000, 3)


def _required_str(args: dict[str, Any], name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str):
        raise ValueError(f"Argument '{name}' must be a string")
    return value


def _required_int(args: dict[str, Any], name: str) -> int:
    value = args.get(name)
    if not isinstance(value, int):
        raise ValueError(f"Argument '{name}' must be an integer")
    return value


def _ensure_no_args(args: dict[str, Any]) -> None:
    if args:
        raise ValueError("This command does not accept arguments")


def _parse_edit_diff(diff: Any) -> tuple[str, str]:
    if isinstance(diff, dict):
        old = diff.get("old")
        new = diff.get("new")
    elif isinstance(diff, str):
        try:
            parsed = json.loads(diff)
        except json.JSONDecodeError as exc:
            raise ValueError('edit_file diff must be {"old": "...", "new": "..."}') from exc
        old = parsed.get("old") if isinstance(parsed, dict) else None
        new = parsed.get("new") if isinstance(parsed, dict) else None
    else:
        old = new = None

    if not isinstance(old, str) or not isinstance(new, str):
        raise ValueError('edit_file diff must contain string fields "old" and "new"')
    return old, new


def _trajectory_record(command: ParsedCommand, result: dict[str, Any]) -> dict[str, Any]:
    preview_source = result.get("result") if result["status"] == "ok" else result.get("error", "")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command.name,
        "args": command.args,
        "status": result["status"],
        "result_preview": _truncate(json.dumps(preview_source, ensure_ascii=False, default=str), 500),
    }


def _truncate(value: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"
