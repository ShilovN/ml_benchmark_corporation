#!/usr/bin/env python3
"""Web UI for running the LLM benchmark agent."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.executor import AgentContext, CommandExecutor
from agent.llm_client import (
    AVAILABLE_MODELS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    MODEL,
    MODEL_GROUPS,
    SYSTEM_MESSAGE,
    auth_headers,
    chat_completion,
    open_url,
    resolve_model_url,
)
from agent.parser import COMMAND_NAMES, FENCED_BLOCK_RE, ParsedCommand, parse_command, parse_model_response


DEFAULT_TASKS_DIR = Path("checker/tasks")
DEFAULT_RUN_ROOT = Path("benchmark_runs")
TEXT_EXTENSIONS = {".csv", ".json", ".md", ".py", ".txt", ".tsv", ".yaml", ".yml"}
MAX_CONTEXT_CHARS = 28000
MAX_FILE_PREVIEW_CHARS = 2500
ALLOWED_MODES = {"single-shot", "repeated", "fixed-transitions", "flexible"}
DEFAULT_MODE = "flexible"
REPEATED_MAX_ATTEMPTS = 5
FIXED_TRANSITION_STAGES = ["EDA", "FEATURES", "TRAIN"]
FIXED_STAGE_ALLOWED_COMMANDS = {
    "EDA": {
        "list_files",
        "read_file",
        "load_dataset",
        "show_dataset_info",
        "show_sample_rows",
        "run_python",
        "get_budget_status",
        "get_remaining_time",
        "get_trajectory",
        "get_hints",
    },
    "FEATURES": {
        "list_files",
        "read_file",
        "write_file",
        "edit_file",
        "load_dataset",
        "show_dataset_info",
        "show_sample_rows",
        "run_python",
        "get_budget_status",
        "get_remaining_time",
        "get_trajectory",
        "get_hints",
    },
    "TRAIN": set(COMMAND_NAMES),
}


@dataclass
class RunConfig:
    task_id: str
    model: str = MODEL
    llm_url: str | None = None
    mode: str = DEFAULT_MODE
    max_steps: int = 40
    token_limit: int = 120000
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    request_timeout: int = DEFAULT_TIMEOUT


@dataclass
class RunState:
    run_id: str
    config: RunConfig
    workspace: Path
    executor: CommandExecutor
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "running"
    stop_reason: str = "not_finished"
    events: list[dict[str, Any]] = field(default_factory=list)
    requests: int = 0
    total_tokens: int = 0
    submitted: bool = False
    submission_result: dict[str, Any] | None = None
    error: str | None = None
    thread: threading.Thread | None = None
    stop_requested: bool = False
    fixed_stage_index: int = 0

    def public_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.config.task_id,
            "created_at": self.created_at,
            "mode": self.config.mode,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "requests": self.requests,
            "total_tokens": self.total_tokens,
            "used_steps": self.executor.context.used_steps,
            "max_steps": self.config.max_steps,
            "submitted": self.submitted,
            "submission_result": self.submission_result,
            "error": self.error,
            "events": self.events[-200:],
        }


class AgentRunManager:
    def __init__(self, project_root: Path, tasks_dir: Path, run_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.tasks_dir = resolve_path(project_root, tasks_dir)
        self.run_root = resolve_path(project_root, run_root)
        self.runs: dict[str, RunState] = {}
        self.lock = threading.Lock()

    def list_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for config_path in sorted(self.tasks_dir.glob("*/task.json")):
            data = json.loads(config_path.read_text(encoding="utf-8"))
            tasks.append(
                {
                    "id": data.get("id") or config_path.parent.name,
                    "name": data.get("name") or data.get("id") or config_path.parent.name,
                    "metric": data.get("metric"),
                    "column": data.get("column") or data.get("pred_column"),
                    "id_column": data.get("id_column"),
                    "public_files": data.get("public_files") or [],
                }
            )
        return tasks

    def start_run(self, config: RunConfig) -> RunState:
        run_id = uuid.uuid4().hex[:12]
        workspace = prepare_run_workspace(self.project_root, self.tasks_dir, self.run_root, config.task_id, run_id)
        executor = CommandExecutor(
            AgentContext(
                workspace=workspace,
                task_id=config.task_id,
                tasks_dir=self.tasks_dir,
                max_steps=config.max_steps,
                history_file=Path("agent_history.txt"),
            )
        )
        state = RunState(run_id=run_id, config=config, workspace=workspace, executor=executor)
        thread = threading.Thread(target=self._run_loop, args=(state,), daemon=True)
        state.thread = thread
        with self.lock:
            self.runs[run_id] = state
        thread.start()
        return state

    def get_run(self, run_id: str) -> RunState | None:
        with self.lock:
            return self.runs.get(run_id)

    def latest_run(self) -> RunState | None:
        with self.lock:
            if not self.runs:
                return None
            return list(self.runs.values())[-1]

    def stop_run(self, run_id: str) -> bool:
        run = self.get_run(run_id)
        if run is None:
            return False
        run.stop_requested = True
        run.stop_reason = "stopped_by_user"
        return True

    def _run_loop(self, state: RunState) -> None:
        user_message = apply_mode_instruction(state, build_initial_prompt(state.workspace, state.config.task_id))
        history: list[dict[str, str]] = []
        add_event(state, "system", "Run started", {"task_id": state.config.task_id, "mode": state.config.mode})

        try:
            while not state.stop_requested:
                if state.executor.context.used_steps >= state.config.max_steps:
                    state.stop_reason = "step_limit"
                    break
                if state.config.token_limit and state.total_tokens >= state.config.token_limit:
                    state.stop_reason = "token_limit"
                    break

                add_event(state, "prompt", "Prompt sent to LLM", {"content": user_message})
                response = chat_completion(
                    user_message,
                    system_message=SYSTEM_MESSAGE,
                    history=history,
                    url=build_chat_completions_url(
                        state.config.llm_url or resolve_model_url(state.config.model)
                    ),
                    model=state.config.model,
                    max_tokens=state.config.max_tokens,
                    temperature=state.config.temperature,
                    timeout=state.config.request_timeout,
                )
                state.requests += 1
                usage = response.get("usage", {})
                if isinstance(usage, dict):
                    state.total_tokens += int(usage.get("total_tokens") or 0)

                assistant_text = extract_assistant_text(response)
                add_event(state, "llm", "LLM response", {"content": assistant_text, "usage": usage})
                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": assistant_text})

                try:
                    commands = extract_commands(assistant_text)
                except ValueError as exc:
                    add_event(state, "parse_error", "Could not parse command", {"error": str(exc)})
                    if state.config.mode == "single-shot":
                        state.stop_reason = "single_shot_parse_error"
                        break
                    if state.config.mode == "repeated" and state.requests >= REPEATED_MAX_ATTEMPTS:
                        state.stop_reason = "repeated_attempt_limit"
                        break
                    user_message = apply_mode_instruction(
                        state,
                        "Твой ответ не удалось распарсить как команду.\n"
                        f"Ошибка: {exc}\n"
                        "Верни только валидную команду или несколько команд, по одной на строку.",
                    )
                    continue

                results: list[dict[str, Any]] = []
                for command in commands:
                    add_event(state, "command", command.name, {"args": command.args})
                    validation_error = validate_mode_command(state, command)
                    if validation_error:
                        result = validation_error
                    else:
                        result = state.executor.execute(command)
                    results.append(result)
                    add_event(state, "result", f"{command.name}: {result['status']}", result)
                    if command.name == "submit" and result["status"] == "ok":
                        state.submitted = True
                        state.submission_result = result
                        state.stop_reason = "submitted"
                        break

                if state.submitted:
                    break

                mode_stop_reason = apply_mode_after_results(state, results)
                if mode_stop_reason:
                    state.stop_reason = mode_stop_reason
                    break

                feedback = state.executor.build_feedback()
                add_event(state, "feedback", "Feedback hints", feedback)
                user_message = apply_mode_instruction(state, build_followup_prompt(results, feedback, state))

            if state.stop_requested:
                state.status = "stopped"
            elif state.submitted:
                state.status = "completed"
            else:
                self._force_final_submit(state)
                state.status = "stopped"
            if state.stop_reason == "not_finished":
                state.stop_reason = "finished"
        except Exception as exc:
            state.status = "error"
            state.stop_reason = "error"
            state.error = str(exc)
            add_event(state, "error", "Run failed", {"error": str(exc)})

    def _force_final_submit(self, state: RunState) -> None:
        if state.submitted:
            return

        reason = state.stop_reason
        try:
            submission_path = state.workspace / "submission.csv"
            if submission_path.exists():
                source = "existing submission.csv"
            else:
                create_baseline_submission(state.workspace)
                source = "auto-generated baseline submission.csv"

            command = ParsedCommand("submit", {"file": "submission.csv"})
            add_event(
                state,
                "command",
                "Forced final submit",
                {"reason": reason, "source": source, "args": command.args},
            )
            state.executor.context.max_steps = max(
                state.executor.context.max_steps,
                state.executor.context.used_steps + 1,
            )
            result = state.executor.execute(command)
            state.submitted = True
            state.submission_result = result
            state.stop_reason = f"{reason}_forced_submit" if reason else "forced_submit"
            add_event(state, "result", f"forced submit: {result['status']}", result)
        except Exception as exc:
            state.submitted = True
            state.submission_result = {
                "status": "error",
                "command": "submit",
                "error": f"Forced final submit failed: {exc}",
            }
            state.stop_reason = f"{reason}_forced_submit_failed" if reason else "forced_submit_failed"
            add_event(state, "error", "Forced final submit failed", {"error": str(exc)})


def make_handler(manager: AgentRunManager) -> type[BaseHTTPRequestHandler]:
    class AgentWebHandler(BaseHTTPRequestHandler):
        server_version = "AgentRunnerHTTP/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_agent_page())
                return
            if parsed.path == "/api/tasks":
                self._send_json({"status": "ok", "tasks": manager.list_tasks()})
                return
            if parsed.path == "/api/models":
                model = _first_query_value(parsed.query, "model") or MODEL
                llm_url = resolve_model_url(model)
                try:
                    self._send_json({"status": "ok", "models": fetch_model_options(llm_url)})
                except Exception as exc:
                    self._send_json(
                        {
                            "status": "error",
                            "error": str(exc),
                            "models": AVAILABLE_MODELS,
                        },
                        HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path == "/api/runs/latest":
                run = manager.latest_run()
                self._send_json({"status": "ok", "run": run.public_dict() if run else None})
                return
            if parsed.path.startswith("/api/runs/"):
                run_id = parsed.path.rsplit("/", 1)[-1]
                run = manager.get_run(run_id)
                if not run:
                    self._send_json({"status": "error", "error": "Run not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_json({"status": "ok", "run": run.public_dict()})
                return
            self._send_json({"status": "error", "error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/runs":
                payload = self._read_json()
                mode = str(payload.get("mode") or DEFAULT_MODE)
                if mode not in ALLOWED_MODES:
                    self._send_json(
                        {
                            "status": "error",
                            "error": f"Unknown mode: {mode}. Allowed modes: {sorted(ALLOWED_MODES)}",
                        },
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                config = RunConfig(
                    task_id=str(payload.get("task_id") or "salary_prediction"),
                    model=str(payload.get("model") or MODEL),
                    llm_url=(
                        str(payload["llm_url"])
                        if payload.get("llm_url")
                        else None
                    ),
                    mode=mode,
                    max_steps=int(payload["max_steps"]) if "max_steps" in payload else 40,
                    token_limit=int(payload["token_limit"]) if "token_limit" in payload else 120000,
                    temperature=float(payload["temperature"]) if "temperature" in payload else DEFAULT_TEMPERATURE,
                    max_tokens=int(payload["max_tokens"]) if "max_tokens" in payload else DEFAULT_MAX_TOKENS,
                    request_timeout=int(payload["request_timeout"]) if "request_timeout" in payload else DEFAULT_TIMEOUT,
                )
                run = manager.start_run(config)
                self._send_json({"status": "ok", "run": run.public_dict()})
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/stop"):
                run_id = parsed.path.split("/")[-2]
                if not manager.stop_run(run_id):
                    self._send_json({"status": "error", "error": "Run not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_json({"status": "ok"})
                return
            self._send_json({"status": "error", "error": "Not found"}, HTTPStatus.NOT_FOUND)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return AgentWebHandler


def prepare_run_workspace(
    project_root: Path,
    tasks_dir: Path,
    run_root: Path,
    task_id: str,
    run_id: str,
) -> Path:
    task_dir = tasks_dir / task_id
    config_path = task_dir / "task.json"
    if not config_path.exists():
        raise ValueError(f"Task config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workspace = run_root / f"{task_id}-{run_id}"
    workspace.mkdir(parents=True, exist_ok=False)
    for filename in config.get("public_files") or ["train.csv", "test.csv"]:
        source = task_dir / str(filename)
        if source.exists():
            shutil.copy2(source, workspace / source.name)
    public_config = {key: value for key, value in config.items() if key not in {"answer_file", "private_files"}}
    (workspace / "task.json").write_text(json.dumps(public_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return workspace


def create_baseline_submission(workspace: Path) -> Path:
    task_config_path = workspace / "task.json"
    train_path = workspace / "train.csv"
    test_path = workspace / "test.csv"
    if not task_config_path.exists():
        raise ValueError("task.json is missing in workspace")
    if not train_path.exists() or not test_path.exists():
        raise ValueError("train.csv and test.csv are required for baseline submission")

    task_config = json.loads(task_config_path.read_text(encoding="utf-8"))
    id_column = str(task_config.get("id_column") or "id")
    target_column = str(task_config.get("column") or task_config.get("pred_column") or "target")
    metric = str(task_config.get("metric") or "")

    with train_path.open("r", encoding="utf-8", newline="") as file:
        train_rows = list(csv.DictReader(file))
    with test_path.open("r", encoding="utf-8", newline="") as file:
        test_rows = list(csv.DictReader(file))

    if not train_rows:
        raise ValueError("train.csv is empty")
    if not test_rows:
        raise ValueError("test.csv is empty")
    if id_column not in test_rows[0]:
        raise ValueError(f"test.csv is missing id column: {id_column}")
    if target_column not in train_rows[0]:
        raise ValueError(f"train.csv is missing target column: {target_column}")

    prediction = baseline_prediction(train_rows, target_column, metric)
    submission_path = workspace / "submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[id_column, target_column])
        writer.writeheader()
        for row in test_rows:
            writer.writerow({id_column: row[id_column], target_column: prediction})
    return submission_path


def baseline_prediction(rows: list[dict[str, str]], target_column: str, metric: str) -> str:
    values = [row.get(target_column, "") for row in rows if row.get(target_column, "") != ""]
    if not values:
        raise ValueError("target column has no values")

    regression_metrics = {"mae", "mse", "rmse", "r2", "r2_score"}
    if metric in regression_metrics:
        numeric = sorted(float(value) for value in values)
        middle = len(numeric) // 2
        if len(numeric) % 2:
            prediction = numeric[middle]
        else:
            prediction = (numeric[middle - 1] + numeric[middle]) / 2
        return f"{prediction:g}"

    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def build_initial_prompt(workspace: Path, task_id: str) -> str:
    return (
        "Ты решаешь ML benchmark task через агентские команды.\n"
        "В одном ответе можно вернуть одну или несколько команд, по одной на строку.\n"
        "Когда подготовишь файл submission.csv, вызови submit(\"submission.csv\").\n\n"
        f"task_id: {task_id}\n\n"
        "Файлы workspace:\n"
        f"{collect_file_previews(workspace)}"
    )


def apply_mode_instruction(state: RunState, prompt: str) -> str:
    instruction = mode_instruction(state)
    if not instruction:
        return prompt
    return f"{prompt}\n\n{instruction}"


def mode_instruction(state: RunState) -> str:
    mode = state.config.mode
    if mode == "single-shot":
        return (
            "Режим single-shot: это единственный ответ модели. "
            "Сразу создай/проверь submission.csv и вызови submit(\"submission.csv\")."
        )
    if mode == "repeated":
        attempt = min(state.requests + 1, REPEATED_MAX_ATTEMPTS)
        return (
            f"Режим repeated: попытка {attempt}/{REPEATED_MAX_ATTEMPTS}. "
            "Каждая попытка должна улучшать или проверять решение; при готовности вызывай submit."
        )
    if mode == "fixed-transitions":
        stage = current_fixed_stage(state)
        if stage == "EDA":
            detail = "изучи файлы и данные; не вызывай submit на этом этапе"
        elif stage == "FEATURES":
            detail = "подготовь признаки, скрипты или промежуточные файлы; submit пока запрещен"
        else:
            detail = "обучи модель, создай submission.csv, проверь формат и вызови submit"
        return f"Режим fixed-transitions: текущий обязательный этап {stage}. Задача этапа: {detail}."
    if mode == "flexible":
        return "Режим flexible: можно свободно выбирать следующие агентские команды до submit или лимитов."
    return ""


def current_fixed_stage(state: RunState) -> str:
    index = min(state.fixed_stage_index, len(FIXED_TRANSITION_STAGES) - 1)
    return FIXED_TRANSITION_STAGES[index]


def validate_mode_command(state: RunState, command: ParsedCommand) -> dict[str, Any] | None:
    if state.config.mode != "fixed-transitions":
        return None
    stage = current_fixed_stage(state)
    allowed = FIXED_STAGE_ALLOWED_COMMANDS[stage]
    if command.name in allowed:
        return None
    return {
        "status": "error",
        "command": command.name,
        "error": (
            f"Pipeline violation: command {command.name} is not allowed during {stage}. "
            f"Allowed commands: {', '.join(sorted(allowed))}."
        ),
    }


def apply_mode_after_results(state: RunState, results: list[dict[str, Any]]) -> str | None:
    if state.config.mode == "single-shot":
        return "single_shot_finished"

    if state.config.mode == "repeated" and state.requests >= REPEATED_MAX_ATTEMPTS:
        return "repeated_attempt_limit"

    if state.config.mode != "fixed-transitions":
        return None

    if any(result.get("status") == "error" for result in results):
        return None

    stage = current_fixed_stage(state)
    add_event(state, "stage", f"{stage} completed", {"stage": stage})
    if stage == "TRAIN":
        return "fixed_transitions_finished"
    state.fixed_stage_index += 1
    next_stage = current_fixed_stage(state)
    add_event(state, "stage", f"Next stage: {next_stage}", {"stage": next_stage})
    return None


def build_followup_prompt(results: list[dict[str, Any]], feedback: dict[str, Any], state: RunState) -> str:
    used_steps = state.executor.context.used_steps
    max_steps = state.config.max_steps
    remaining_steps = max(0, max_steps - used_steps)
    lines = [
        "Продолжай решение. Верни только следующую команду или команды.",
        "",
        "Статус benchmark:",
        (
            f"requests={state.requests}; tokens={state.total_tokens}/{state.config.token_limit or 'без лимита'}; "
            f"steps={used_steps}/{max_steps}; remaining_steps={remaining_steps}"
        ),
        "",
        "Результаты команд:",
    ]
    if results:
        for index, result in enumerate(results, start=1):
            lines.extend(format_web_command_result(index, result))
    else:
        lines.append("- команд не было")
    if feedback:
        lines.extend(["", "Подсказки:"])
        lines.extend(format_web_feedback(feedback))
    return "\n".join(lines)


def format_web_command_result(index: int, result: dict[str, Any]) -> list[str]:
    command = str(result.get("command", "unknown"))
    status = str(result.get("status", "unknown"))
    if status != "ok":
        return [f"{index}. {command}: error", f"   {short_web_text(str(result.get('error', 'unknown error')), 1200)}"]
    lines = [f"{index}. {command}: ok"]
    lines.extend(f"   {line}" for line in format_web_result_payload(command, result.get("result")))
    return lines


def format_web_result_payload(command: str, payload: Any) -> list[str]:
    if command == "read_file" and isinstance(payload, dict):
        lines = [f"path={payload.get('path', '?')}; truncated={payload.get('truncated', False)}"]
        content = str(payload.get("content", ""))
        if content:
            lines.append("content:")
            lines.extend(indent_web_block(short_web_text(content, 2500), "  "))
        return lines
    if command == "run_python" and isinstance(payload, dict):
        lines = [f"returncode={payload.get('returncode', '?')}"]
        stdout = str(payload.get("stdout", ""))
        stderr = str(payload.get("stderr", ""))
        if stdout:
            lines.append("stdout:")
            lines.extend(indent_web_block(short_web_text(stdout, 1800), "  "))
        if stderr:
            lines.append("stderr:")
            lines.extend(indent_web_block(short_web_text(stderr, 1800), "  "))
        return lines
    if command == "list_files" and isinstance(payload, list):
        shown = [str(item) for item in payload[:30]]
        suffix = f" ... (+{len(payload) - len(shown)} more)" if len(payload) > len(shown) else ""
        return [", ".join(shown) + suffix if shown else "(empty)"]
    if isinstance(payload, (dict, list)):
        return indent_web_block(short_web_json(payload, 2500), "")
    return [short_web_text(str(payload), 2500)]


def format_web_feedback(feedback: dict[str, Any]) -> list[str]:
    hints = feedback.get("hints")
    if not isinstance(hints, list) or not hints:
        return [f"- {short_web_json(feedback, 1500)}"]
    lines: list[str] = []
    for hint in hints[:8]:
        if isinstance(hint, dict):
            lines.append(f"- {hint.get('stage', 'hint')}: {hint.get('message', '')}")
        else:
            lines.append(f"- {hint}")
    if len(hints) > 8:
        lines.append(f"- ... (+{len(hints) - 8} more hints)")
    return lines


def short_web_json(value: Any, limit: int) -> str:
    return short_web_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str), limit)


def short_web_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit < 80:
        return value[:limit]
    marker = f"...[truncated {len(value) - limit} chars]..."
    keep = max(0, limit - len(marker))
    head = keep // 2
    tail = keep - head
    return value[:head] + marker + value[-tail:]


def indent_web_block(text: str, prefix: str) -> list[str]:
    lines = text.splitlines()
    return [prefix + line for line in lines] if lines else [prefix]


def collect_file_previews(workspace: Path) -> str:
    chunks: list[str] = []
    used = 0
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.name.startswith("answers"):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        preview = content[:MAX_FILE_PREVIEW_CHARS]
        chunk = f"### {path.relative_to(workspace)}\n```text\n{preview}\n```\n"
        if used + len(chunk) > MAX_CONTEXT_CHARS:
            chunks.append("...[context truncated]")
            break
        chunks.append(chunk)
        used += len(chunk)
    return "\n".join(chunks) if chunks else "(no readable files)"


def extract_assistant_text(response: dict[str, Any]) -> str:
    return str(response["choices"][0]["message"]["content"])


def extract_commands(text: str) -> list[ParsedCommand]:
    try:
        return [parse_model_response(text)]
    except ValueError as first_error:
        commands: list[ParsedCommand] = []
        for candidate in command_candidates(text):
            try:
                commands.append(parse_command(candidate))
            except ValueError:
                continue
        if commands:
            return commands
        raise first_error


def command_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for block in FENCED_BLOCK_RE.findall(text):
        if block.strip():
            candidates.append(block.strip())
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") or any(
            stripped.startswith(f"{name}(") and stripped.endswith(")") for name in COMMAND_NAMES
        ):
            candidates.append(stripped)
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def add_event(state: RunState, kind: str, title: str, data: dict[str, Any]) -> None:
    state.events.append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "title": title,
            "data": data,
        }
    )


def build_chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped == "https://api.openai.com":
        return f"{stripped}/v1/chat/completions"
    if stripped.endswith("/openai"):
        return f"{stripped}/chat/completions"
    if stripped.endswith("/v1") or stripped.endswith("/compatible-mode/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/compatible-mode/v1/chat/completions"


def build_models_url(base_url: str) -> str:
    chat_url = build_chat_completions_url(base_url).rstrip("/")
    suffix = "/chat/completions"
    if chat_url.endswith(suffix):
        return chat_url[: -len(suffix)] + "/models"
    return chat_url + "/models"


def fetch_model_options(base_url: str, timeout: int = 10) -> list[str]:
    models_url = build_models_url(base_url)
    request = urllib.request.Request(models_url, headers=auth_headers(url=models_url), method="GET")
    try:
        with open_url(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM models HTTP {exc.code}: {body}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise RuntimeError("LLM /models response does not contain a data list")
    models = [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]
    if not models:
        raise RuntimeError("LLM /models returned no model ids")
    return models


def resolve_path(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _first_query_value(query: str, name: str) -> str | None:
    values = parse_qs(query).get(name)
    return values[0] if values else None


def render_agent_page() -> str:
    model_options = "\n".join(
        "            <optgroup label=\""
        + html.escape(group_name, quote=True)
        + "\">\n"
        + "\n".join(
            f'              <option value="{html.escape(model, quote=True)}">'
            f"{html.escape(model)}</option>"
            for model in models
        )
        + "\n            </optgroup>"
        for group_name, models in MODEL_GROUPS
    )
    page = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Benchmark Runner</title>
  <style>
    :root { --bg:#f5f7fa; --surface:#fff; --border:#d8e0ea; --text:#152033; --muted:#667085; --accent:#155eef; --ok:#067647; --bad:#b42318; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }
    main { max-width:1240px; margin:0 auto; padding:28px 20px 44px; overflow-x:hidden; }
    header { display:flex; justify-content:space-between; gap:18px; align-items:flex-start; margin-bottom:20px; }
    h1 { margin:0 0 8px; font-size:30px; line-height:1.15; }
    h2 { margin:0 0 14px; font-size:18px; }
    p { margin:0; color:var(--muted); }
    section { min-width:0; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:18px; margin-bottom:16px; }
    .layout { display:grid; grid-template-columns:360px minmax(0,1fr); gap:18px; align-items:start; }
    .layout > * { min-width:0; }
    label { display:block; margin:0 0 7px; font-weight:700; }
    input, select, button { box-sizing:border-box; width:100%; min-height:40px; font:inherit; }
    input, select { border:1px solid var(--border); border-radius:6px; padding:8px; background:white; }
    .field { margin-bottom:14px; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    button { border:0; border-radius:6px; background:var(--accent); color:white; font-weight:800; cursor:pointer; padding:0 14px; }
    button.secondary { background:#eef3f8; color:var(--text); border:1px solid var(--border); }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .status { display:inline-flex; min-height:32px; align-items:center; padding:0 12px; border-radius:999px; border:1px solid var(--border); background:white; color:var(--muted); }
    .cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
    .card { border:1px solid var(--border); border-radius:8px; padding:12px; background:#f8fafc; }
    .card strong { display:block; font-size:22px; margin-top:4px; }
    .events { display:grid; gap:12px; min-width:0; max-height:72vh; overflow:auto; padding-right:4px; }
    .event { min-width:0; max-width:100%; overflow:hidden; }
    .chat-event { display:grid; gap:6px; }
    .chat-row { display:flex; gap:10px; align-items:flex-start; }
    .chat-row.user { flex-direction:row-reverse; }
    .chat-avatar { flex:0 0 auto; width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12px; color:#fff; background:var(--accent); }
    .chat-row.user .chat-avatar { background:#0f766e; }
    .chat-meta { display:flex; justify-content:space-between; gap:12px; color:var(--muted); font-size:12px; margin:0 4px; }
    .chat-row.user .chat-meta { flex-direction:row-reverse; }
    .bubble { min-width:0; max-width:min(100%, 920px); border:1px solid var(--border); border-radius:14px; padding:12px 14px; background:#f8fafc; }
    .chat-row.user .bubble { background:#effdf7; border-color:#c8e7da; }
    .chat-title { font-weight:800; margin-bottom:8px; color:var(--text); }
    .chat-body { width:100%; min-width:0; white-space:pre-wrap; line-height:1.5; background:transparent; border:0; padding:0; margin:0; max-height:none; overflow:visible; }
    .chat-body.structured { white-space:normal; }
    .chat-body .code-block { max-height:calc(1.45em * 5 + 24px); overflow:hidden; white-space:pre-wrap; word-break:break-word; }
    .chat-body .code-block pre { line-height:1.45; }
    .chat-body .block { margin:10px 0; }
    .chat-body pre { max-height:none; white-space:pre-wrap; word-break:break-word; }
    .chat-tail { display:flex; justify-content:space-between; gap:10px; color:var(--muted); font-size:12px; margin:0 44px; }
    .event.command, .event.result, .event.feedback, .event.system, .event.error, .event.parse_error { border:1px solid var(--border); border-radius:10px; padding:10px 12px; background:white; }
    .event.command { border-left:4px solid #c2410c; }
    .event.result { border-left:4px solid #067647; }
    .event.feedback { border-left:4px solid #7c3aed; }
    .event.system { border-left:4px solid #64748b; }
    .event.error, .event.parse_error { border-left:4px solid var(--bad); }
    .event-head { display:flex; justify-content:space-between; gap:10px; color:var(--muted); font-size:13px; margin-bottom:6px; }
    .kind { font-weight:800; color:var(--accent); text-transform:uppercase; }
    .event-title { font-weight:800; margin-bottom:6px; }
    .event-preview { color:var(--muted); font-size:14px; line-height:1.4; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .event-body { box-sizing:border-box; width:100%; min-width:0; white-space:pre-wrap; line-height:1.45; max-height:220px; overflow:auto; background:#f8fafc; border:1px solid var(--border); border-radius:6px; padding:12px; margin-top:10px; }
    .event-body.structured { white-space:normal; }
    .event.expanded .event-body { max-height:none; }
    .event.collapsed .event-body, .event.collapsed details { display:none; }
    .block { min-width:0; overflow:hidden; border:1px solid var(--border); border-radius:6px; background:white; padding:10px; margin:8px 0; }
    .block-title { font-weight:800; margin-bottom:6px; color:var(--text); }
    .block pre { background:#f8fafc; color:var(--text); border:1px solid var(--border); max-height:180px; }
    .code-block { box-sizing:border-box; width:100%; max-width:100%; background:#111827; color:#f9fafb; border-radius:6px; padding:12px; overflow:auto; max-height:340px; font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; font-size:13px; white-space:pre; position:relative; }
    .chat-body .code-block.collapsible { max-height:calc(1.45em * 5 + 24px); overflow:hidden; white-space:pre-wrap; word-break:break-word; }
    .chat-body .code-block.collapsible.is-expanded { max-height:none; overflow:auto; }
    .code-toggle { position:absolute; top:6px; right:6px; width:18px; height:18px; border:1px solid rgba(148,163,184,0.5); border-radius:4px; background:rgba(15,23,42,0.6); padding:0; cursor:pointer; }
    .code-toggle::before { content:""; position:absolute; right:4px; top:4px; width:8px; height:8px; border-top:2px solid #9ca3af; border-right:2px solid #9ca3af; transform:rotate(0deg); }
    .code-block.is-expanded .code-toggle::before { transform:rotate(180deg); }
    .code-toggle:hover { border-color:rgba(148,163,184,0.9); }
    .code-toggle:focus-visible { outline:2px solid #93c5fd; outline-offset:2px; }
    .message { border-radius:8px; padding:12px; background:#eef3f8; color:var(--muted); margin-top:12px; }
    .error { color:var(--bad); background:#fef3f2; border:1px solid #fecdca; }
    @media (max-width:900px){ header,.layout{display:block}.row,.cards{grid-template-columns:1fr} }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>LLM Benchmark Runner</h1>
      <p>Запускает модель, выполняет ее команды, показывает trajectory, подсказки и финальный submit.</p>
    </div>
    <div class="status" id="status">Idle</div>
  </header>
  <div class="layout">
    <aside>
      <section>
        <h2>Run Settings</h2>
        <div class="field"><label for="task">Task</label><select id="task"></select></div>
        <div class="field">
          <label for="model">Model</label>
          <select id="model">
__MODEL_OPTIONS__
          </select>
        </div>
        <div class="field">
          <label for="mode">Mode</label>
          <select id="mode">
            <option value="flexible" selected>flexible</option>
            <option value="single-shot">single-shot</option>
            <option value="repeated">repeated</option>
            <option value="fixed-transitions">fixed-transitions</option>
          </select>
        </div>
        <div class="row">
          <div class="field"><label for="steps">Max steps</label><input id="steps" type="number" value="40" min="1"></div>
          <div class="field"><label for="tokens">Token limit</label><input id="tokens" type="number" value="120000" min="1"></div>
        </div>
        <div class="row">
          <button id="start">Start LLM Run</button>
          <button class="secondary" id="stop" disabled>Stop</button>
        </div>
        <div class="message" id="message">Готово к запуску. Нужен доступ к OpenAI-compatible LLM endpoint.</div>
      </section>
    </aside>
    <div>
      <section>
        <h2>Run Summary</h2>
        <div class="cards">
          <div class="card">Status<strong id="s-status">-</strong></div>
          <div class="card">Requests<strong id="s-requests">0</strong></div>
          <div class="card">Steps<strong id="s-steps">0</strong></div>
          <div class="card">Tokens<strong id="s-tokens">0</strong></div>
        </div>
        <div class="message" id="submission">No submission yet.</div>
      </section>
      <section>
        <h2>Live Trajectory</h2>
        <div class="events" id="events"><div class="message">No events yet.</div></div>
      </section>
    </div>
  </div>
</main>
<script>
let currentRun = null;
let timer = null;
const task = document.getElementById('task');
const model = document.getElementById('model');
const mode = document.getElementById('mode');
const steps = document.getElementById('steps');
const tokens = document.getElementById('tokens');
const start = document.getElementById('start');
const stop = document.getElementById('stop');
const message = document.getElementById('message');
const statusEl = document.getElementById('status');
const eventsEl = document.getElementById('events');
function truncateText(value, limit=180){
  const text = cleanText(value).replace(/\\s+/g, ' ');
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}
function esc(v){return String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');}
function cleanText(value){return String(value ?? '').trim();}
function isLikelyJsonChunk(text){
  const trimmed = cleanText(text);
  if(!trimmed) return false;
  if(!((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']')))) return false;
  try {
    JSON.parse(trimmed);
    return true;
  } catch {
    return false;
  }
}
function isLikelyCommandLine(text){
  const trimmed = cleanText(text);
  return /^(?:list_files|read_file|write_file|edit_file|load_dataset|show_dataset_info|show_sample_rows|run_python|get_budget_status|get_remaining_time|get_trajectory|get_hints|submit)\s*\(/.test(trimmed);
}
function stripContentFromText(text){
  text = cleanText(text);
  if(!text) return '';
  const hasCommandLine = (value) => value.split('\n').some(line => isLikelyCommandLine(line.trim()));
  const hasJsonLine = (value) => value.split('\n').some(line => isLikelyJsonChunk(line.trim()));
  const countParenDelta = (value) => {
    let delta = 0;
    let inSingle = false;
    let inDouble = false;
    let escaped = false;
    for(const ch of value){
      if(escaped){
        escaped = false;
        continue;
      }
      if(ch === '\\'){
        escaped = true;
        continue;
      }
      if(inSingle){
        if(ch === "'") inSingle = false;
        continue;
      }
      if(inDouble){
        if(ch === '"') inDouble = false;
        continue;
      }
      if(ch === "'"){
        inSingle = true;
        continue;
      }
      if(ch === '"'){
        inDouble = true;
        continue;
      }
      if(ch === '(') delta += 1;
      if(ch === ')') delta -= 1;
    }
    return delta;
  };
  text = text.replace(/```(?:json|tool|command)\s*[\s\S]*?```/gi, '');
  text = text.replace(/```[\s\S]*?```/g, block => {
    const inner = block.replace(/^```[^\n]*\n?/, '').replace(/```$/, '');
    return isLikelyJsonChunk(inner) || hasJsonLine(inner) || hasCommandLine(inner) ? '' : block;
  });

  const lines = text.split('\n');
  const kept = [];
  for(let i = 0; i < lines.length; i++){
    const line = lines[i];
    const trimmed = line.trim();
    if(!trimmed) {
      kept.push('');
      continue;
    }
    if(isLikelyCommandLine(trimmed)) {
      let balance = countParenDelta(line);
      while(i + 1 < lines.length && balance > 0) {
        i++;
        balance += countParenDelta(lines[i]);
      }
      continue;
    }
    if(trimmed.startsWith('{') || trimmed.startsWith('[')) {
      let buffer = trimmed;
      let j = i;
      let parsed = false;
      while(true) {
        if(isLikelyJsonChunk(buffer)) {
          parsed = true;
          break;
        }
        if(j + 1 >= lines.length) {
          break;
        }
        if(lines[j + 1].trim() === '' && buffer.length > 0) {
          break;
        }
        if(j - i > 40) {
          break;
        }
        j++;
        buffer += `\n${lines[j].trim()}`;
      }
      if(parsed) {
        i = j;
        continue;
      }
    }
    kept.push(line);
  }
  const cleaned = kept.join('\n').replace(/\n{3,}/g, '\n\n').trim();
  return cleaned;
}
function renderInlineMarkup(text){
  const parts = String(text).split(/(`[^`]*`)/g);
  return parts.map(part => {
    if(part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
      return `<span class="inline-code">${esc(part.slice(1, -1))}</span>`;
    }
    return esc(part);
  }).join('');
}
function renderMarkdownLine(line){
  const trimmed = line.trim();
  if(!trimmed) return '';
  if(trimmed === '***' || trimmed === '---') return '<hr class="md-rule">';
  if(trimmed.startsWith('### ')) return `<div class="md-heading">${renderInlineMarkup(trimmed.slice(4))}</div>`;
  if(trimmed.startsWith('## ')) return `<div class="md-heading">${renderInlineMarkup(trimmed.slice(3))}</div>`;
  if(trimmed.startsWith('# ')) return `<div class="md-heading">${renderInlineMarkup(trimmed.slice(2))}</div>`;
  if(/^(?:- |\* |• )/.test(trimmed)) {
    return `<div class="md-list-item">• ${renderInlineMarkup(trimmed.replace(/^(?:- |\* |• )/, ''))}</div>`;
  }
  if(/^\d+\.\s+/.test(trimmed)) {
    return `<div class="md-list-item">${renderInlineMarkup(trimmed)}</div>`;
  }
  return `<div class="md-paragraph">${renderInlineMarkup(trimmed)}</div>`;
}
function renderMarkdownish(text){
  text = stripContentFromText(text);
  if(!text) return '<div class="md-paragraph">Empty message.</div>';
  const parts = [];
  const fence = /```(?:([a-zA-Z0-9_-]+)\n)?([\s\S]*?)```/g;
  let last = 0;
  for(const match of text.matchAll(fence)){
    const before = text.slice(last, match.index);
    parts.push(before.split('\n').map(renderMarkdownLine).join(''));
    const lang = match[1] ? `<div class="code-lang">${esc(match[1])}</div>` : '';
    parts.push(`<div class="code-block collapsible">${lang}<button type="button" class="code-toggle" aria-label="Expand code"></button><pre>${esc(match[2].replace(/\n$/, ''))}</pre></div>`);
    last = match.index + match[0].length;
  }
  parts.push(text.slice(last).split('\n').map(renderMarkdownLine).join(''));
  return parts.join('');
}
function setupCodeToggles(){
  document.addEventListener('click', (event) => {
    const button = event.target.closest('.code-toggle');
    if(!button) return;
    const block = button.closest('.code-block');
    if(!block) return;
    block.classList.toggle('is-expanded');
    button.setAttribute('aria-label', block.classList.contains('is-expanded') ? 'Collapse code' : 'Expand code');
  });
}
function renderJsonScalar(value){
  if(value === null || value === undefined) return '<span class="json-muted">-</span>';
  if(typeof value === 'boolean' || typeof value === 'number') return `<span class="json-pill">${esc(String(value))}</span>`;
  if(typeof value === 'string') {
    const text = cleanText(value);
    if(!text) return '<span class="json-muted">-</span>';
    if(isProbablyCode(text) || text.includes('\\n')) return `<div class="code-block">${esc(text)}</div>`;
    return `<span class="json-pill">${esc(shortText(text, 240))}</span>`;
  }
  return `<span class="json-pill">${esc(shortText(JSON.stringify(value), 240))}</span>`;
}
function renderJsonList(value, depth=0){
  const items = value.slice(0, depth === 0 ? 3 : 2);
  return `
    <div class="json-array">
      <div class="json-array-head">${esc(value.length)} item${value.length === 1 ? '' : 's'}</div>
      ${items.length ? items.map(item => `<div class="json-item">${renderJsonSummary(item, depth + 1)}</div>`).join('') : '<div class="json-muted">Empty list.</div>'}
      ${value.length > items.length ? `<div class="json-more">+${value.length - items.length} more</div>` : ''}
    </div>`;
}
function renderJsonObject(value, depth=0){
  const preferredOrder = [
    'status', 'command', 'error', 'message', 'metric', 'value', 'rows_checked',
    'returncode', 'used_steps', 'max_steps', 'remaining_steps', 'requests',
    'total_tokens', 'token_limit', 'stage', 'title', 'task_id', 'model', 'file',
    'path', 'stdout', 'stderr'
  ];
  const entries = Object.entries(value);
  const selected = preferredOrder
    .filter(key => Object.prototype.hasOwnProperty.call(value, key))
    .map(key => [key, value[key]]);
  const extras = entries.filter(([key]) => !preferredOrder.includes(key));
  const shownExtras = depth === 0 ? extras.slice(0, 3) : extras.slice(0, 2);
  const rows = [...selected, ...shownExtras];
  const hidden = Math.max(0, entries.length - rows.length);
  return `
    <div class="json-view">
      ${rows.length ? `<div class="json-grid">${rows.map(([key, item]) => `
        <div class="kv">
          <span>${esc(key)}</span>
          <span>${renderJsonSummary(item, depth + 1)}</span>
        </div>
      `).join('')}</div>` : '<div class="json-muted">Empty object.</div>'}
      ${hidden > 0 ? `<div class="json-more">+${hidden} more field${hidden === 1 ? '' : 's'}</div>` : ''}
    </div>`;
}
function renderJsonSummary(value, depth=0){
  if(value === null || value === undefined) return '<span class="json-muted">-</span>';
  if(Array.isArray(value)) return renderJsonList(value, depth);
  if(isPlainObject(value)) return renderJsonObject(value, depth);
  return renderJsonScalar(value);
}
function renderValue(value){
  if(isProbablyCode(value)) return `<div class="code-block">${esc(cleanText(value))}</div>`;
  return renderJsonSummary(value);
}
function renderCommandCall(name, args){
  const argValues = Object.entries(args || {}).map(([key, value]) => {
    if(typeof value === 'string') {
      const cleaned = cleanText(value);
      if(isProbablyCode(cleaned)) return `${key}=<code block>`;
      return JSON.stringify(cleaned);
    }
    return `${key}=${JSON.stringify(value)}`;
  });
  return `${name}(${argValues.join(', ')})`;
}
function tryParseFollowupPrompt(text){
  const marker = '\\n\\n';
  const index = text.indexOf(marker);
  if(index === -1) return null;
  const intro = text.slice(0, index).trim();
  const rest = text.slice(index + marker.length).trim();
  if(!rest.startsWith('{')) return null;
  try {
    return {intro, payload: JSON.parse(rest)};
  } catch {
    return null;
  }
}
function commandResultSummary(result){
  const payload = result.result || {};
  if(result.command === 'run_python') {
    return `
      <div class="block">
        <div class="block-title">run_python finished with code ${esc(payload.returncode ?? 'unknown')}</div>
        ${payload.stdout ? `<div class="block-title">stdout</div><pre>${esc(payload.stdout)}</pre>` : ''}
        ${payload.stderr ? `<div class="block-title">stderr</div><pre>${esc(payload.stderr)}</pre>` : ''}
      </div>`;
  }
  if(result.command === 'submit') {
    return `
      <div class="block">
        <div class="block-title">submit result</div>
        <div class="kv">
          <span>Status</span><span>${esc(result.status)}</span>
          <span>Metric</span><span>${esc(payload.metric || '-')}</span>
          <span>Value</span><span>${esc(payload.value ?? '-')}</span>
          <span>Rows checked</span><span>${esc(payload.rows_checked ?? '-')}</span>
        </div>
      </div>`;
  }
  return `
    <div class="block">
      <div class="block-title">${esc(result.command || 'command')} result</div>
      ${renderJsonSummary(payload || result)}
    </div>`;
}
function renderCommandHtml(event){
  const args = event.data?.args || {};
  const entries = Object.entries(args);
  return `
    <div class="event-body structured">
      <div class="command-name">${esc(cleanText(event.title))}</div>
      <div class="command-call"><span class="inline-code">${esc(renderCommandCall(cleanText(event.title), args))}</span></div>
      <div class="arg-list">
        ${entries.length ? entries.map(([name, value]) => `
          <div class="arg-row">
            <span class="arg-name">${esc(name)}</span>
            ${renderValue(value)}
          </div>
        `).join('') : '<div>No arguments.</div>'}
      </div>
    </div>`;
}
function renderFeedbackHtml(data){
  const hints = data.hints || [];
  return `
    <div class="event-body structured">
      <div class="block">
        <div class="block-title">Feedback hints</div>
        ${hints.length ? `<ul class="hint-list">${hints.map(h => `<li><strong>${esc(h.stage)}</strong>: ${esc(h.message)}</li>`).join('')}</ul>` : '<div>No hints.</div>'}
      </div>
    </div>`;
}
function renderSubmissionHtml(submission){
  const payload = submission.result || {};
  return `
    <strong>${submission.status === 'ok' ? 'Submitted.' : 'Submission attempted.'}</strong>
    <div class="kv" style="margin-top:10px">
      <span>Status</span><span>${esc(submission.status || '-')}</span>
      <span>Metric</span><span>${esc(payload.metric || '-')}</span>
      <span>Value</span><span>${esc(payload.value ?? '-')}</span>
      <span>Rows checked</span><span>${esc(payload.rows_checked ?? '-')}</span>
    </div>
    <details>
      <summary>Details</summary>
      <div class="kv">
        <span>Command</span><span>${esc(submission.command || 'submit')}</span>
        <span>Elapsed ms</span><span>${esc(submission.elapsed_ms ?? '-')}</span>
        <span>Result status</span><span>${esc(payload.status || submission.status || '-')}</span>
        ${payload.error ? `<span>Error</span><span>${esc(payload.error)}</span>` : ''}
      </div>
    </details>`;
}
function renderPromptHtml(text){
  const parsed = tryParseFollowupPrompt(text);
  return `<div class="chat-body structured">${renderMarkdownish(parsed ? parsed.intro : text)}</div>`;
}
function renderCommandNote(event){
  const name = cleanText(event.title || event.data?.command || '');
  const args = event.data?.args || {};
  const code = cleanText(args.code_or_file || '');
  const path = cleanText(args.path || args.file || '');
  const content = cleanText(args.content || args.diff || '');
  const isCode = name === 'run_python';
  const fileOps = new Set(['read_file', 'write_file', 'edit_file', 'load_dataset', 'list_files', 'show_dataset_info', 'show_sample_rows']);
  if(!isCode && !fileOps.has(name)) return '';
  const summary = isCode ? 'Выполняю код' : 'Открываю файл';
  const details = [];
  if(isCode) {
    if(code) details.push(`<div class="note-meta">Код</div><pre>${esc(code)}</pre>`);
    else details.push('<div class="note-meta">Код не передан.</div>');
  } else {
    if(path) details.push(`<div class="note-meta">Путь: <span class="inline-code">${esc(path)}</span></div>`);
    if(content) details.push(`<div class="note-meta">Содержимое</div><pre>${esc(content)}</pre>`);
    if(!path && !content) details.push('<div class="note-meta">Без дополнительных данных.</div>');
  }
  return `
    <div class="command-note">
      <details>
        <summary>${esc(summary)}</summary>
        <div class="note-body">${details.join('')}</div>
      </details>
    </div>`;
}
function eventMainText(event){
  const data = event.data || {};
  if(event.kind === 'prompt') return cleanText(data.content || '');
  if(event.kind === 'llm') return cleanText(data.content || '');
  if(event.kind === 'command') return renderCommandCall(cleanText(event.title), data.args || {});
  if(event.kind === 'result') {
    const status = data.status || 'unknown';
    const result = data.result ? JSON.stringify(data.result, null, 2) : (data.error || '');
    return `${event.title}\nstatus: ${status}\n${result}`;
  }
  if(event.kind === 'feedback') {
    const hints = (data.hints || []).map(h => `- ${h.stage}: ${h.message}`).join(String.fromCharCode(10));
    return hints || JSON.stringify(data, null, 2);
  }
  if(event.kind === 'error' || event.kind === 'parse_error') return data.error || JSON.stringify(data, null, 2);
  return JSON.stringify(data, null, 2);
}
function eventBodyHtml(event){
  const data = event.data || {};
  if(event.kind === 'prompt') return renderPromptHtml(data.content || '');
  if(event.kind === 'llm') return `<div class="chat-body structured">${renderMarkdownish(data.content || '')}</div>`;
  if(event.kind === 'result') {
    if(data.command === 'submit') return renderSubmissionHtml(data);
    return '';
  }
  return '';
}
function renderChatEvent(kind, title, bodyHtml, time, side){
  const avatar = kind === 'prompt' ? 'You' : 'LLM';
  return `
    <div class="event chat-event ${esc(kind)} ${esc(side)}">
      <div class="chat-row ${esc(side)}">
        <div class="chat-avatar">${esc(avatar)}</div>
        <div class="bubble">
          <div class="chat-title">${esc(title)}</div>
          ${bodyHtml}
        </div>
      </div>
      <div class="chat-tail">
        <span>${esc(kind === 'prompt' ? 'Your message' : 'LLM response')}</span>
        <span>${esc(time)}</span>
      </div>
    </div>`;
}
function renderTechEvent(event, index){
  const data = event.data || {};
  const summary = cleanText(data.error || data.message || data.status || event.title || event.kind);
  return `
    <div class="event ${esc(event.kind)}" data-event-index="${index}">
      <div class="event-head"><span class="kind">${esc(eventLabel(event.kind))}</span><span>${esc(new Date(event.time).toLocaleTimeString())}</span></div>
      <div class="event-title">${esc(event.title)}</div>
      ${summary ? `<div class="event-preview">${esc(summary)}</div>` : ''}
    </div>`;
}
function eventPreviewHtml(event){
  const text = truncateText(eventMainText(event));
  return text ? `<div class="event-preview">${esc(text)}</div>` : '';
}
function eventLabel(kind){
  const labels = {
    prompt: 'Prompt to LLM',
    llm: 'LLM answer',
    command: 'Command',
    result: 'Result',
    feedback: 'Hints',
    system: 'System',
    error: 'Error',
    parse_error: 'Parse error'
  };
  return labels[kind] || kind;
}
async function loadTasks(){
  const res = await fetch('/api/tasks');
  const data = await res.json();
  task.innerHTML = data.tasks.map(t => `<option value="${esc(t.id)}">${esc(t.name)} (${esc(t.metric)})</option>`).join('');
}
async function startRun(){
  start.disabled = true; stop.disabled = false; message.textContent = 'Starting run...';
  const res = await fetch('/api/runs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({
    task_id: task.value,
    model: model.value,
    mode: mode.value,
    max_steps: Number(steps.value),
    token_limit: Number(tokens.value)
  })});
  const data = await res.json();
  if(data.status !== 'ok'){ message.textContent = data.error || 'Failed to start'; start.disabled=false; stop.disabled=true; return; }
  currentRun = data.run.run_id;
  poll();
  timer = setInterval(poll, 1500);
}
async function stopRun(){
  if(!currentRun) return;
  await fetch(`/api/runs/${currentRun}/stop`, {method:'POST'});
  poll();
}
async function poll(){
  if(!currentRun) return;
  const res = await fetch(`/api/runs/${currentRun}`);
  const data = await res.json();
  if(data.status !== 'ok') return;
  render(data.run);
  if(['completed','stopped','error'].includes(data.run.status)){
    clearInterval(timer); start.disabled=false; stop.disabled=true;
  }
}
function render(run){
  statusEl.textContent = run.status + ' / ' + run.stop_reason;
  document.getElementById('s-status').textContent = run.status;
  document.getElementById('s-requests').textContent = run.requests;
  document.getElementById('s-steps').textContent = `${run.used_steps}/${run.max_steps}`;
  document.getElementById('s-tokens').textContent = run.total_tokens;
  message.textContent = `Mode: ${run.mode}. Task: ${run.task_id}.`;
  if(run.submission_result){
    document.getElementById('submission').className = 'message';
    document.getElementById('submission').innerHTML = renderSubmissionHtml(run.submission_result);
  } else if(run.error){
    document.getElementById('submission').className = 'message error';
    document.getElementById('submission').textContent = run.error;
  } else if(['completed','stopped','error'].includes(run.status)){
    document.getElementById('submission').className = 'message error';
    const reason = run.stop_reason === 'token_limit'
      ? 'Token limit ended before a final submission could be attempted.'
      : run.stop_reason === 'step_limit'
        ? 'Step limit ended before a final submission could be attempted.'
        : `Run finished without submit(file). Stop reason: ${run.stop_reason}.`;
    document.getElementById('submission').textContent = reason;
  } else {
    document.getElementById('submission').className = 'message';
    document.getElementById('submission').textContent = 'No submission yet.';
  }
  const turns = [];
  let currentTurn = null;
  for(const event of run.events) {
    if(event.kind === 'prompt') {
      currentTurn = { prompt: event, llm: null, hints: [], notes: [] };
      turns.push(currentTurn);
      continue;
    }
    if(event.kind === 'llm') {
      if(!currentTurn) {
        currentTurn = { prompt: null, llm: event, hints: [], notes: [] };
        turns.push(currentTurn);
      } else {
        currentTurn.llm = event;
      }
      continue;
    }
    if(event.kind === 'feedback') {
      if(currentTurn) currentTurn.hints = Array.isArray(event.data?.hints) ? event.data.hints : [];
      continue;
    }
    if(event.kind === 'command') {
      if(currentTurn) {
        const note = renderCommandNote(event);
        if(note) currentTurn.notes.push(note);
      }
      continue;
    }
    if(event.kind === 'result' && event.data?.command === 'submit') {
      continue;
    }
    if(event.kind === 'error' || event.kind === 'parse_error' || event.kind === 'system') {
      if(currentTurn) currentTurn.notes.push(renderTechEvent(event, turns.length));
    }
  }
  eventsEl.innerHTML = turns.length ? turns.map((turn, index) => {
    const prompt = turn.prompt ? renderChatEvent(
      'prompt',
      'You',
      `${renderPromptHtml(turn.prompt.data?.content || '')}${turn.hints.length ? `<div class="turn-hint-label">Подсказки к вашему сообщению</div><ul class="turn-hints">${turn.hints.map(h => `<li><strong>${esc(h.stage)}</strong>: ${esc(h.message)}</li>`).join('')}</ul>` : ''}`,
      new Date(turn.prompt.time).toLocaleTimeString(),
      'user'
    ) : '';
    const llm = turn.llm ? `
      <div class="event chat-event llm assistant">
        <div class="chat-row assistant">
          <div class="chat-avatar">LLM</div>
          <div class="bubble">
            <div class="chat-title">LLM</div>
            <div class="chat-body structured">${renderMarkdownish(turn.llm.data?.content || '')}</div>
            ${turn.notes.length ? `<div class="command-notes">${turn.notes.join('')}</div>` : ''}
          </div>
        </div>
        <div class="chat-tail">
          <span>LLM response</span>
          <span>${esc(new Date(turn.llm.time).toLocaleTimeString())}</span>
        </div>
      </div>` : '';
    return `<div class="turn" data-turn-index="${index}">${prompt}${llm}</div>`;
  }).join('') : '<div class="message">No events yet.</div>';
}
start.addEventListener('click', startRun);
stop.addEventListener('click', stopRun);
setupCodeToggles();
loadTasks();
</script>
</body>
</html>"""
    return page.replace("__MODEL_OPTIONS__", model_options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM benchmark web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = AgentRunManager(args.project_root, args.tasks_dir, args.run_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(manager))
    print(f"Agent web UI is running on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping agent web UI")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
