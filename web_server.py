#!/usr/bin/env python3
"""Web UI for running the LLM benchmark agent."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import shutil
import subprocess
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

from agent.executor import AgentContext, CommandExecutor, DEFAULT_TIME_LIMIT_SECONDS
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
DEFAULT_DOCKER_IMAGE = "ml-benchmark-runner:latest"
TEXT_EXTENSIONS = {".csv", ".json", ".md", ".py", ".txt", ".tsv", ".yaml", ".yml"}
MAX_CONTEXT_CHARS = 28000
MAX_FILE_PREVIEW_CHARS = 2500
FINALIZE_REMAINING_STEPS = 4
FINALIZE_REMAINING_TOKEN_RATIO = 0.12
LLM_REQUEST_TOKEN_RESERVE = 1000
TOKEN_ESTIMATE_CHARS_PER_TOKEN = 3
MAX_HISTORY_MESSAGES = 6
FINAL_SUBMIT_MAX_TOKENS = 1200
FINAL_SUBMIT_TOKEN_RESERVE = 300
SINGLE_SHOT_MIN_MAX_TOKENS = 4096
ALLOWED_MODES = {"single-shot", "repeated", "fixed-transitions", "flexible"}
DEFAULT_MODE = "flexible"
REPEATED_MAX_ATTEMPTS = 5
SINGLE_SHOT_ALLOWED_COMMANDS = {"write_file", "run_python", "submit"}
PRODUCTIVE_SOLUTION_COMMANDS = {"write_file", "edit_file", "run_python"}
FIXED_TRANSITION_STAGES = ["EDA", "FEATURES", "TRAIN"]
FIXED_STAGE_MIN_ATTEMPTS = {"EDA": 3, "FEATURES": 5, "TRAIN": 1}
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
    time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS
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
    final_submit_requested: bool = False
    fixed_stage_index: int = 0
    fixed_stage_attempts: dict[str, int] = field(default_factory=dict)
    docker_container: str | None = None
    created_container: bool = False

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
            "code_time_limit_seconds": self.config.time_limit_seconds,
            "elapsed_seconds": round(time.monotonic() - self.executor.context.start_time, 1),
            "submitted": self.submitted,
            "submission_result": self.submission_result,
            "error": self.error,
            "docker_container": self.docker_container,
            "events": self.events[-200:],
        }


class AgentRunManager:
    def __init__(
        self,
        project_root: Path,
        tasks_dir: Path,
        run_root: Path,
        *,
        use_docker: bool = True,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        keep_containers: bool = False,
    ) -> None:
        self.project_root = project_root.resolve()
        self.tasks_dir = resolve_path(project_root, tasks_dir)
        self.run_root = resolve_path(project_root, run_root)
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.keep_containers = keep_containers
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
        docker_container = None
        created_container = False
        if self.use_docker:
            docker_container = default_container_name(config.task_id, run_id)
            create_container(name=docker_container, image=self.docker_image, workspace=workspace)
            created_container = True
        executor = CommandExecutor(
            AgentContext(
                workspace=workspace,
                task_id=config.task_id,
                tasks_dir=self.tasks_dir,
                max_steps=config.max_steps,
                time_limit_seconds=config.time_limit_seconds,
                history_file=Path("agent_history.txt"),
                docker_container=docker_container,
            )
        )
        state = RunState(
            run_id=run_id,
            config=config,
            workspace=workspace,
            executor=executor,
            docker_container=docker_container,
            created_container=created_container,
        )
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
        add_event(
            state,
            "system",
            "Run started",
            {
                "task_id": state.config.task_id,
                "mode": state.config.mode,
                "docker_container": state.docker_container,
            },
        )

        try:
            while not state.stop_requested:
                if state.executor.context.used_steps >= state.config.max_steps:
                    state.stop_reason = "step_limit"
                    break
                if state.config.token_limit and state.total_tokens >= state.config.token_limit:
                    state.stop_reason = "token_limit"
                    break
                history = compact_history(history)
                request_max_tokens = state.config.max_tokens
                if state.config.mode == "single-shot":
                    request_max_tokens = max(request_max_tokens, SINGLE_SHOT_MIN_MAX_TOKENS)
                if should_stop_before_next_llm_request(state, user_message, history, request_max_tokens):
                    final_message = build_emergency_submit_prompt(state)
                    if state.final_submit_requested or not can_send_llm_request(
                        state,
                        final_message,
                        [],
                        FINAL_SUBMIT_MAX_TOKENS,
                        FINAL_SUBMIT_TOKEN_RESERVE,
                    ):
                        state.stop_reason = "token_margin"
                        break
                    user_message = final_message
                    history = []
                    request_max_tokens = FINAL_SUBMIT_MAX_TOKENS
                    state.final_submit_requested = True
                    add_event(
                        state,
                        "system",
                        "Emergency final prompt",
                        {"reason": "next normal LLM request would exceed token budget"},
                    )

                add_event(state, "prompt", "Prompt sent to LLM", {"content": user_message})
                response = chat_completion(
                    user_message,
                    system_message=SYSTEM_MESSAGE,
                    history=history,
                    url=build_chat_completions_url(
                        state.config.llm_url or resolve_model_url(state.config.model)
                    ),
                    model=state.config.model,
                    max_tokens=request_max_tokens,
                    temperature=state.config.temperature,
                    timeout=state.config.request_timeout,
                )
                state.requests += 1
                if state.config.mode == "fixed-transitions":
                    record_fixed_stage_attempt(state)
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
                    if state.config.mode == "single-shot" or state.final_submit_requested:
                        state.stop_reason = "single_shot_parse_error"
                        break
                    if state.config.mode == "repeated" and state.requests >= REPEATED_MAX_ATTEMPTS:
                        state.stop_reason = "repeated_attempt_limit"
                        break
                    user_message = apply_mode_instruction(
                        state,
                        "Твой ответ не удалось распарсить как команду.\n"
                        f"Ошибка: {exc}\n"
                        "Верни короткую строку `Мысль: ...`, затем валидную команду или несколько команд, по одной на строку.",
                    )
                    continue

                batch_error = validate_mode_command_batch(state, commands)
                if batch_error:
                    add_event(state, "parse_error", "Invalid command batch", batch_error)
                    if should_retry_after_batch_error(state):
                        user_message = apply_mode_instruction(state, build_batch_error_prompt(batch_error))
                        continue
                    state.stop_reason = batch_error["reason"]
                    break

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

                if state.config.mode == "single-shot":
                    state.stop_reason = "single_shot_missing_successful_submit"
                    break

                mode_stop_reason = apply_mode_after_results(state, results)
                if mode_stop_reason:
                    state.stop_reason = mode_stop_reason
                    break

                feedback = state.executor.build_feedback()
                if state.requests > 1:
                    add_event(state, "feedback", "Feedback hints", feedback)
                user_message = apply_mode_instruction(state, build_followup_prompt(results, feedback, state))

            if state.stop_requested:
                state.status = "stopped"
            elif state.submitted:
                state.status = "completed"
            elif state.config.mode == "single-shot":
                self._force_final_submit(state)
                state.status = "completed"
            elif not should_force_final_submit(state):
                state.status = "error"
                if not state.error:
                    state.error = mode_failure_message(state)
                add_event(state, "error", "Run failed before valid submit", {"error": state.error})
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
        finally:
            if state.created_container and state.docker_container and not self.keep_containers:
                remove_container(state.docker_container)

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
            if state.config.mode == "single-shot":
                title = "Single-shot final submit"
                result_title = "single-shot submit"
                stop_suffix = "single_shot_submit"
                error_prefix = "Single-shot final submit failed"
            else:
                title = "Forced final submit"
                result_title = "forced submit"
                stop_suffix = "forced_submit"
                error_prefix = "Forced final submit failed"
            add_event(
                state,
                "command",
                title,
                {"reason": reason, "source": source, "args": command.args},
            )
            state.executor.context.max_steps = max(
                state.executor.context.max_steps,
                state.executor.context.used_steps + 1,
            )
            result = state.executor.execute(command)
            state.submitted = True
            state.submission_result = result
            if state.config.mode == "single-shot":
                state.stop_reason = stop_suffix
            else:
                state.stop_reason = f"{reason}_{stop_suffix}" if reason else stop_suffix
            add_event(state, "result", f"{result_title}: {result['status']}", result)
        except Exception as exc:
            state.submitted = True
            state.submission_result = {
                "status": "error",
                "command": "submit",
                "error": f"{error_prefix}: {exc}",
            }
            if state.config.mode == "single-shot":
                state.stop_reason = f"{stop_suffix}_failed"
            else:
                state.stop_reason = f"{reason}_{stop_suffix}_failed" if reason else f"{stop_suffix}_failed"
            add_event(state, "error", error_prefix, {"error": str(exc)})


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
                parts = parsed.path.strip("/").split("/")
                run_id = parts[2] if len(parts) >= 3 else ""
                run = manager.get_run(run_id)
                if not run:
                    self._send_json({"status": "error", "error": "Run not found"}, HTTPStatus.NOT_FOUND)
                    return
                if len(parts) == 4 and parts[3] == "files":
                    self._send_json({"status": "ok", "files": list_workspace_files(run.workspace)})
                    return
                if len(parts) == 5 and parts[3] == "files" and parts[4] == "view":
                    file_path = _first_query_value(parsed.query, "path") or ""
                    try:
                        payload = workspace_file_preview(run.workspace, file_path)
                    except ValueError as exc:
                        self._send_json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                        return
                    self._send_json({"status": "ok", "file": payload})
                    return
                if len(parts) == 5 and parts[3] == "files" and parts[4] == "download":
                    file_path = _first_query_value(parsed.query, "path") or ""
                    try:
                        path = resolve_workspace_file(run.workspace, file_path)
                    except ValueError as exc:
                        self._send_json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                        return
                    self._send_file_download(path)
                    return
                if len(parts) != 3:
                    self._send_json({"status": "error", "error": "Not found"}, HTTPStatus.NOT_FOUND)
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
                    time_limit_seconds=(
                        int(payload["time_limit_seconds"])
                        if "time_limit_seconds" in payload
                        else DEFAULT_TIME_LIMIT_SECONDS
                    ),
                    temperature=float(payload["temperature"]) if "temperature" in payload else DEFAULT_TEMPERATURE,
                    max_tokens=int(payload["max_tokens"]) if "max_tokens" in payload else DEFAULT_MAX_TOKENS,
                    request_timeout=int(payload["request_timeout"]) if "request_timeout" in payload else DEFAULT_TIMEOUT,
                )
                try:
                    run = manager.start_run(config)
                except subprocess.CalledProcessError as exc:
                    self._send_json(
                        {
                            "status": "error",
                            "error": "Could not start benchmark Docker container",
                            "details": (exc.stderr or exc.stdout or str(exc)).strip(),
                        },
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
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

        def _send_file_download(self, path: Path) -> None:
            body = path.read_bytes()
            filename = path.name.replace('"', "")
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
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


def create_container(*, name: str, image: str, workspace: Path) -> None:
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-v",
            f"{workspace.resolve()}:/workspace",
            "-w",
            "/workspace",
            image,
            "sleep",
            "infinity",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def remove_container(container: str) -> None:
    subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True, text=True)


def default_container_name(task_id: str, run_id: str) -> str:
    safe_task_id = "".join(char if char.isalnum() or char in "-_" else "-" for char in task_id)
    return f"ml-benchmark-web-{safe_task_id}-{run_id}"


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
        "В одном ответе сначала напиши короткую строку `Мысль: ...`, затем одну или несколько команд, по одной на строку.\n"
        "Работай как в обычной ML-задаче: осмотри данные, сделай признаки, обучи и проверь модель. "
        "Перед submit проверь, что submission.csv существует, имеет нужные колонки, правильное число строк "
        "и не содержит пустых предсказаний. Когда всё готово, вызови submit(\"submission.csv\").\n\n"
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
            "Режим single-shot: у тебя ровно один ответ модели на всю задачу. "
            "В этом ответе можно вернуть несколько команд подряд, но после их выполнения новых попыток не будет. "
            "Файлы train.csv, test.csv и task.json уже показаны выше, поэтому НЕ трать единственный ответ на list_files/read_file/load_dataset/show_sample_rows. "
            "В single-shot разрешены только продуктивные команды: write_file, run_python, submit. "
            "Ответ без submit(\"submission.csv\") будет отклонен.\n"
            "Лучший формат для длинного кода:\n"
            "Мысль: делаю полный быстрый пайплайн и отправляю решение\n"
            "```command\n"
            "run_python(\"\"\"\n"
            "<полный python-код: читает train.csv/test.csv, обучает или строит baseline, создает submission.csv и печатает проверку формата>\n"
            "\"\"\")\n"
            "```\n"
            "```command\n"
            "submit(\"submission.csv\")\n"
            "```\n"
            "Если сложная модель рискованна, сделай простой валидный baseline, но обязательно создай submission.csv и вызови submit."
        )
    if mode == "repeated":
        attempt = min(state.requests + 1, REPEATED_MAX_ATTEMPTS)
        if attempt == 1:
            stage_note = (
                "Это единственная попытка, где можно в основном читать и изучать файлы. "
                "К концу ответа сформулируй следующий продуктивный шаг."
            )
        elif attempt < REPEATED_MAX_ATTEMPTS:
            stage_note = (
                "На этой попытке одних read_file/list_files уже недостаточно. "
                "Обязательно создай или измени решение продуктивной командой: write_file, edit_file или run_python. "
                "Сделай baseline, признаки, обучение, валидацию или генерацию submission.csv."
            )
        else:
            stage_note = (
                "Это последняя попытка. Используй лучшее уже подготовленное решение, "
                "создай/проверь submission.csv и обязательно вызови submit(\"submission.csv\")."
            )
        return (
            f"Режим repeated: попытка {attempt}/{REPEATED_MAX_ATTEMPTS}. "
            "После каждой попытки ты получаешь результаты команд и можешь улучшить решение в следующей попытке. "
            "Не трать несколько попыток подряд только на открытие файлов. "
            f"{stage_note}"
        )
    if mode == "fixed-transitions":
        stage = current_fixed_stage(state)
        attempt = current_fixed_stage_attempt(state) + 1
        minimum = FIXED_STAGE_MIN_ATTEMPTS.get(stage, 1)
        if stage == "EDA":
            detail = (
                "это этап 1/3. Изучи файлы, целевую колонку, размер данных, пропуски, распределения и первые идеи признаков. "
                "На этом этапе submit запрещен."
            )
        elif stage == "FEATURES":
            detail = (
                "это этап 2/3. Подготовь и сравни признаки, предобработку и скрипты для обучения/инференса. "
                "На этом этапе submit еще запрещен."
            )
        else:
            detail = (
                "это этап 3/3. Обучи или запусти лучшую модель, создай и проверь submission.csv, "
                "затем обязательно вызови submit(\"submission.csv\")."
            )
        return (
            f"Режим fixed-transitions: текущий обязательный этап {stage}. "
            f"Итерация этапа {attempt}/{minimum}. "
            "Этапы идут строго по порядку: EDA -> FEATURES -> TRAIN; команды вне текущего этапа блокируются. "
            "Не пытайся завершить этап за один короткий ответ: используй несколько итераций на глубину, как в flexible. "
            f"Задача этапа: {detail}"
        )
    if mode == "flexible":
        return flexible_mode_instruction(state)
    return ""


def flexible_mode_instruction(state: RunState) -> str:
    budget = build_budget_payload(state)
    remaining_steps = int(budget["remaining_steps"])
    remaining_percent = budget["remaining_token_percent"]
    if remaining_steps <= FINALIZE_REMAINING_STEPS or (
        remaining_percent is not None and remaining_percent <= FINALIZE_REMAINING_TOKEN_RATIO * 100
    ):
        return (
            "Режим flexible: финальная стадия. Используй лучшее уже подготовленное решение, "
            "проверь submission.csv и вызывай submit(\"submission.csv\")."
        )
    if state.requests <= 1:
        return (
            "Режим flexible: это ранняя стадия решения, не отправляй submit сразу. "
            "Сначала используй доступные итерации на EDA, проверку формата, baseline, валидацию и хотя бы одно улучшение модели. "
            "Submit уместен только после созданного submission.csv и понятной проверки качества/формата."
        )
    if remaining_percent is not None and remaining_percent >= 55:
        return (
            "Режим flexible: бюджет еще большой, продолжай улучшать решение вместо раннего submit. "
            "Сравни baseline с более сильной моделью, проверь признаки, валидацию и формат submission.csv."
        )
    return (
        "Режим flexible: можно свободно выбирать следующие агентские команды. "
        "Доведи решение до лучшего найденного качества и отправляй submit только когда файл проверен или началась финальная стадия."
    )


def current_fixed_stage(state: RunState) -> str:
    index = min(state.fixed_stage_index, len(FIXED_TRANSITION_STAGES) - 1)
    return FIXED_TRANSITION_STAGES[index]


def current_fixed_stage_attempt(state: RunState) -> int:
    return int(state.fixed_stage_attempts.get(current_fixed_stage(state), 0))


def record_fixed_stage_attempt(state: RunState) -> None:
    stage = current_fixed_stage(state)
    state.fixed_stage_attempts[stage] = current_fixed_stage_attempt(state) + 1


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


def validate_mode_command_batch(state: RunState, commands: list[ParsedCommand]) -> dict[str, Any] | None:
    command_names = [command.name for command in commands]
    if state.config.mode == "repeated":
        return validate_repeated_command_batch(state, command_names)
    if state.config.mode == "fixed-transitions":
        return validate_fixed_transition_command_batch(state, command_names)
    if state.config.mode != "single-shot":
        return None
    disallowed = [name for name in command_names if name not in SINGLE_SHOT_ALLOWED_COMMANDS]
    if disallowed:
        return {
            "reason": "single_shot_disallowed_commands",
            "error": (
                "Single-shot must not spend its only response on exploration commands. "
                f"Disallowed commands: {', '.join(disallowed)}. "
                "Use run_python/write_file to create submission.csv, then submit(\"submission.csv\")."
            ),
            "commands": command_names,
            "allowed_commands": sorted(SINGLE_SHOT_ALLOWED_COMMANDS),
        }
    if "submit" not in command_names:
        return {
            "reason": "single_shot_missing_submit",
            "error": "Single-shot response must include submit(\"submission.csv\") in the same answer.",
            "commands": command_names,
            "allowed_commands": sorted(SINGLE_SHOT_ALLOWED_COMMANDS),
        }
    return None


def validate_fixed_transition_command_batch(state: RunState, command_names: list[str]) -> dict[str, Any] | None:
    stage = current_fixed_stage(state)
    has_productive = any(name in PRODUCTIVE_SOLUTION_COMMANDS for name in command_names)
    has_submit = "submit" in command_names
    if stage == "FEATURES" and not has_productive:
        return {
            "reason": "fixed_features_no_productive_progress",
            "error": (
                "FEATURES stage must create or change the solution. "
                "Use write_file, edit_file, or run_python for preprocessing, features, or reusable scripts."
            ),
            "stage": stage,
            "commands": command_names,
            "productive_commands": sorted(PRODUCTIVE_SOLUTION_COMMANDS),
        }
    if stage == "TRAIN":
        if not has_productive and not (state.workspace / "submission.csv").exists():
            return {
                "reason": "fixed_train_no_solution",
                "error": (
                    "TRAIN stage must train/run a solution or create submission.csv before submit. "
                    "Use run_python or write_file, then submit(\"submission.csv\")."
                ),
                "stage": stage,
                "commands": command_names,
                "productive_commands": sorted(PRODUCTIVE_SOLUTION_COMMANDS),
            }
        if not has_submit:
            return {
                "reason": "fixed_train_missing_submit",
                "error": "TRAIN stage must include the model's own submit(\"submission.csv\") command.",
                "stage": stage,
                "commands": command_names,
            }
    return None


def validate_repeated_command_batch(state: RunState, command_names: list[str]) -> dict[str, Any] | None:
    attempt = state.requests
    has_productive = any(name in PRODUCTIVE_SOLUTION_COMMANDS for name in command_names)
    has_submit = "submit" in command_names
    if attempt >= REPEATED_MAX_ATTEMPTS and not has_submit:
        return {
            "reason": "repeated_final_missing_submit",
            "error": "Final repeated attempt must include submit(\"submission.csv\").",
            "commands": command_names,
        }
    if attempt >= 2 and not has_productive and not has_submit:
        return {
            "reason": "repeated_no_productive_progress",
            "error": (
                "Repeated mode allows exploration at the beginning, but this attempt must make productive progress. "
                "Use write_file, edit_file, or run_python to build, validate, or improve the solution."
            ),
            "commands": command_names,
            "productive_commands": sorted(PRODUCTIVE_SOLUTION_COMMANDS),
        }
    return None


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
    stage_attempts = current_fixed_stage_attempt(state)
    minimum_attempts = FIXED_STAGE_MIN_ATTEMPTS.get(stage, 1)
    if stage_attempts < minimum_attempts:
        add_event(
            state,
            "stage",
            f"{stage} continues",
            {
                "stage": stage,
                "attempts": stage_attempts,
                "minimum_attempts": minimum_attempts,
            },
        )
        return None

    add_event(state, "stage", f"{stage} completed", {"stage": stage})
    if stage == "TRAIN":
        return "fixed_transitions_finished"
    state.fixed_stage_index += 1
    next_stage = current_fixed_stage(state)
    add_event(state, "stage", f"Next stage: {next_stage}", {"stage": next_stage})
    return None


def should_force_final_submit(state: RunState) -> bool:
    if state.config.mode == "single-shot":
        return False
    if state.config.mode == "fixed-transitions":
        return False
    if state.config.mode == "repeated" and not (state.workspace / "submission.csv").exists():
        return has_productive_history(state)
    return True


def has_productive_history(state: RunState) -> bool:
    return any(
        record.get("command") in PRODUCTIVE_SOLUTION_COMMANDS
        for record in state.executor.context.trajectory
    )


def mode_failure_message(state: RunState) -> str:
    if state.config.mode == "single-shot":
        return "Single-shot response did not produce executable commands or submission.csv."
    if state.config.mode == "repeated":
        return "Repeated mode ended without productive solution commands or submission.csv."
    if state.config.mode == "fixed-transitions":
        return "Fixed-transitions mode ended before the model produced its own submit(\"submission.csv\")."
    return "Run ended before a valid submission was produced."


def should_retry_after_batch_error(state: RunState) -> bool:
    if state.config.mode == "repeated":
        return state.requests < REPEATED_MAX_ATTEMPTS
    return state.config.mode == "fixed-transitions"


def build_batch_error_prompt(batch_error: dict[str, Any]) -> str:
    return (
        "Предыдущий ответ не подходит для текущего режима.\n"
        f"Ошибка: {batch_error.get('error', 'invalid command batch')}\n"
        "Верни короткую строку `Мысль: ...`, затем команды, которые реально продвигают решение. "
        "Если это последняя попытка, создай/проверь submission.csv и вызови submit(\"submission.csv\")."
    )


def build_followup_prompt(results: list[dict[str, Any]], feedback: dict[str, Any], state: RunState) -> str:
    used_steps = state.executor.context.used_steps
    max_steps = state.config.max_steps
    remaining_steps = max(0, max_steps - used_steps)
    if state.config.mode == "fixed-transitions":
        opener = fixed_transition_followup_opener(state)
    elif state.config.mode == "repeated":
        opener = repeated_followup_opener(state)
    elif state.config.mode == "flexible" and not is_finalization_phase(state):
        opener = (
            "Продолжай улучшать решение. Верни короткую строку `Мысль: ...`, затем следующую команду или команды. "
            "Не вызывай submit рано: сначала используй итерации на проверку данных, признаки, валидацию и улучшение модели."
        )
    else:
        opener = "Продолжай решение. Верни короткую строку `Мысль: ...`, затем следующую команду или команды."
    lines = [
        opener,
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
    feedback_lines = format_web_feedback(feedback) if feedback and state.requests > 1 else []
    if feedback_lines:
        lines.extend(["", "Подсказки:"])
        lines.extend(feedback_lines)
    return "\n".join(lines)


def repeated_followup_opener(state: RunState) -> str:
    next_attempt = min(state.requests + 1, REPEATED_MAX_ATTEMPTS)
    if next_attempt >= REPEATED_MAX_ATTEMPTS:
        return (
            "Это последняя repeated-попытка. Верни короткую строку `Мысль: ...`, затем команды: "
            "создай или проверь submission.csv и обязательно вызови submit(\"submission.csv\")."
        )
    if next_attempt >= 2:
        return (
            "Продолжай repeated-решение. Верни короткую строку `Мысль: ...`, затем команды с продуктивным шагом: "
            "write_file, edit_file или run_python для baseline, признаков, обучения, валидации или submission.csv. "
            "Не трать попытку только на открытие файлов."
        )
    return "Продолжай решение. Верни короткую строку `Мысль: ...`, затем следующую команду или команды."


def fixed_transition_followup_opener(state: RunState) -> str:
    stage = current_fixed_stage(state)
    attempts = current_fixed_stage_attempt(state)
    minimum = FIXED_STAGE_MIN_ATTEMPTS.get(stage, 1)
    progress = f"Итераций этапа: {attempts}/{minimum}."
    if stage == "EDA":
        return (
            f"Продолжай fixed-transitions этап EDA. {progress} "
            "Верни короткую строку `Мысль: ...`, затем команды для более глубокого анализа данных. "
            "Переход к FEATURES произойдет только после минимального числа EDA-итераций."
        )
    if stage == "FEATURES":
        return (
            f"Продолжай fixed-transitions этап FEATURES. {progress} "
            "Верни короткую строку `Мысль: ...`, затем продуктивные команды "
            "write_file, edit_file или run_python для признаков, preprocessing или скриптов решения."
        )
    return (
        "Продолжай fixed-transitions этап TRAIN. Верни короткую строку `Мысль: ...`, затем команды, которые создают/проверяют "
        "submission.csv, и обязательно вызови submit(\"submission.csv\"). Без собственного submit этап TRAIN не завершится."
    )


def is_finalization_phase(state: RunState) -> bool:
    budget = build_budget_payload(state)
    if budget["remaining_steps"] <= FINALIZE_REMAINING_STEPS:
        return True
    percent = budget["remaining_token_percent"]
    return percent is not None and percent <= FINALIZE_REMAINING_TOKEN_RATIO * 100


def format_web_command_result(index: int, result: dict[str, Any]) -> list[str]:
    command = str(result.get("command", "unknown"))
    status = str(result.get("status", "unknown"))
    if status != "ok":
        lines = [f"{index}. {command}: error"]
        lines.extend(f"   {line}" for line in fenced_web_text_block(str(result.get("error", "unknown error")), 1200))
        return lines
    lines = [f"{index}. {command}: ok"]
    detail = "\n".join(format_web_result_payload(command, result.get("result")))
    lines.extend(f"   {line}" for line in fenced_web_text_block(detail, 4000))
    return lines


def format_web_result_payload(command: str, payload: Any) -> list[str]:
    if command == "submit" and isinstance(payload, dict):
        lines = [
            f"metric={payload.get('metric', '?')}; value={payload.get('value', '?')}; "
            f"rows_checked={payload.get('rows_checked', '?')}"
        ]
        source = payload.get("submission_source")
        if isinstance(source, dict):
            path = source.get("path") or source.get("kind") or "inline"
            lines.append(f"submission_source={path}; truncated={source.get('truncated', False)}")
        return lines
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
        return []
    lines: list[str] = []
    for hint in hints[:8]:
        if isinstance(hint, dict):
            message = str(hint.get("message") or "").strip()
            if not message:
                continue
            stage = str(hint.get("stage") or "hint").strip()
            lines.append(f"- {stage}: {message}")
        else:
            text = str(hint).strip()
            if text:
                lines.append(f"- {text}")
    if len(hints) > 8:
        lines.append(f"- ... (+{len(hints) - 8} more hints)")
    return lines


def short_web_json(value: Any, limit: int) -> str:
    return short_web_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str), limit)


def fenced_web_text_block(value: str, limit: int) -> list[str]:
    return ["```text", *short_web_text(value, limit).splitlines(), "```"]


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


def build_emergency_submit_prompt(state: RunState) -> str:
    files = sorted(
        str(path.relative_to(state.workspace))
        for path in state.workspace.rglob("*")
        if path.is_file() and not path.name.startswith("answers")
    )
    budget = build_budget_payload(state)
    task_config = read_workspace_task_config(state.workspace)
    if (state.workspace / "submission.csv").exists():
        instruction = (
            "FINAL SUBMIT NOW.\n"
            "В workspace уже есть submission.csv. Не улучшай решение и не запускай EDA.\n"
            "Верни короткую строку `Мысль: отправляю готовый файл`, затем одну команду:\n"
            "submit(\"submission.csv\")\n\n"
        )
    else:
        instruction = (
            "FINAL SUBMIT NOW.\n"
            "Нужно дать последнее лучшее решение. Не делай EDA и долгие улучшения.\n"
            "Создай submission.csv самым надежным быстрым способом и в этом же ответе вызови submit(\"submission.csv\").\n"
            "Верни короткую строку `Мысль: завершаю решение`, затем команды.\n\n"
        )
    payload = {
        "task": task_config,
        "workspace_files": files[-80:],
        "budget": budget,
    }
    return instruction + json.dumps(payload, ensure_ascii=False, default=str)


def read_workspace_task_config(workspace: Path) -> dict[str, Any]:
    path = workspace / "task.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_budget_payload(state: RunState) -> dict[str, Any]:
    remaining_steps = max(0, state.config.max_steps - state.executor.context.used_steps)
    if state.config.token_limit:
        remaining_tokens = max(0, state.config.token_limit - state.total_tokens)
        remaining_token_ratio = remaining_tokens / state.config.token_limit
    else:
        remaining_tokens = None
        remaining_token_ratio = None
    return {
        "used_steps": state.executor.context.used_steps,
        "max_steps": state.config.max_steps,
        "remaining_steps": remaining_steps,
        "requests": state.requests,
        "total_tokens": state.total_tokens,
        "token_limit": state.config.token_limit,
        "remaining_tokens": remaining_tokens,
        "remaining_token_percent": (
            round(remaining_token_ratio * 100, 1) if remaining_token_ratio is not None else None
        ),
    }


def should_request_final_submit(state: RunState) -> bool:
    budget = build_budget_payload(state)
    if budget["remaining_steps"] <= FINALIZE_REMAINING_STEPS:
        return True
    percent = budget["remaining_token_percent"]
    return percent is not None and percent <= FINALIZE_REMAINING_TOKEN_RATIO * 100


def compact_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history
    return history[-MAX_HISTORY_MESSAGES:]


def estimate_message_tokens(system_message: str, user_message: str, history: list[dict[str, str]]) -> int:
    total_chars = len(system_message) + len(user_message)
    total_chars += sum(len(message.get("content", "")) for message in history)
    return max(1, total_chars // TOKEN_ESTIMATE_CHARS_PER_TOKEN)


def can_send_llm_request(
    state: RunState,
    user_message: str,
    history: list[dict[str, str]],
    max_tokens: int,
    reserve: int = LLM_REQUEST_TOKEN_RESERVE,
) -> bool:
    remaining_tokens = state.config.token_limit - state.total_tokens
    estimated_request_tokens = (
        estimate_message_tokens(SYSTEM_MESSAGE, user_message, history)
        + max_tokens
        + reserve
    )
    return remaining_tokens > estimated_request_tokens


def should_stop_before_next_llm_request(
    state: RunState,
    user_message: str,
    history: list[dict[str, str]],
    max_tokens: int,
) -> bool:
    if not state.config.token_limit:
        return False
    return not can_send_llm_request(state, user_message, history, max_tokens)


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


def list_workspace_files(workspace: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    workspace = workspace.resolve()
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        relative = path.resolve().relative_to(workspace)
        stat = path.stat()
        files.append(
            {
                "path": str(relative),
                "name": path.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "previewable": path.suffix.lower() in TEXT_EXTENSIONS,
            }
        )
    return files


def workspace_file_preview(workspace: Path, file_path: str) -> dict[str, Any]:
    path = resolve_workspace_file(workspace, file_path)
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        raise ValueError("Preview is available only for text-like files")
    content = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "path": str(path.resolve().relative_to(workspace.resolve())),
        "content": content[:MAX_FILE_PREVIEW_CHARS],
        "truncated": len(content) > MAX_FILE_PREVIEW_CHARS,
        "size": path.stat().st_size,
    }


def resolve_workspace_file(workspace: Path, file_path: str) -> Path:
    if not file_path:
        raise ValueError("File path is required")
    raw_path = Path(file_path)
    if raw_path.is_absolute():
        raise ValueError("Absolute paths are not allowed")
    workspace = workspace.resolve()
    path = (workspace / raw_path).resolve()
    if path != workspace and workspace not in path.parents:
        raise ValueError("File path is outside workspace")
    if not path.exists() or not path.is_file():
        raise ValueError("File not found")
    return path


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
            block_commands = command_block_candidates(block)
            candidates.extend(block_commands if block_commands else [block.strip()])
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


def command_block_candidates(block: str) -> list[str]:
    stripped = block.strip()
    if not stripped:
        return []
    try:
        module = ast.parse(stripped, mode="exec")
    except SyntaxError:
        return []

    candidates: list[str] = []
    for node in module.body:
        if not isinstance(node, ast.Expr):
            continue
        if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name):
            continue
        if node.value.func.id not in COMMAND_NAMES:
            continue
        source = ast.get_source_segment(stripped, node)
        if source:
            candidates.append(source.strip())
    return candidates


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
    main { max-width:1480px; margin:0 auto; padding:28px 20px 44px; overflow-x:hidden; }
    header { display:flex; justify-content:space-between; gap:18px; align-items:flex-start; margin-bottom:20px; }
    h1 { margin:0 0 8px; font-size:30px; line-height:1.15; }
    h2 { margin:0 0 14px; font-size:18px; }
    p { margin:0; color:var(--muted); }
    section { min-width:0; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:18px; margin-bottom:16px; }
    .layout { display:grid; grid-template-columns:320px minmax(0,1fr); gap:18px; align-items:start; }
    .layout > * { min-width:0; }
    label { display:block; margin:0 0 7px; font-weight:700; }
    input, select, button { box-sizing:border-box; width:100%; min-height:40px; font:inherit; }
    input, select { border:1px solid var(--border); border-radius:6px; padding:8px; background:white; }
    .field { margin-bottom:14px; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .limit-row { grid-template-columns:repeat(3,minmax(0,1fr)); }
    button { border:0; border-radius:6px; background:var(--accent); color:white; font-weight:800; cursor:pointer; padding:0 14px; }
    button.secondary { background:#eef3f8; color:var(--text); border:1px solid var(--border); }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .status { display:inline-flex; min-height:32px; align-items:center; padding:0 12px; border-radius:999px; border:1px solid var(--border); background:white; color:var(--muted); }
    .cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
    .card { border:1px solid var(--border); border-radius:8px; padding:12px; background:#f8fafc; }
    .card strong { display:block; font-size:22px; margin-top:4px; }
    .events { display:grid; gap:12px; min-width:0; max-width:100%; max-height:72vh; overflow-y:auto; overflow-x:hidden; padding-right:4px; }
    .event { min-width:0; max-width:100%; overflow:hidden; }
    .turn { min-width:0; max-width:100%; overflow:hidden; }
    .chat-event { display:grid; gap:6px; min-width:0; max-width:100%; overflow:hidden; }
    .chat-row { display:flex; gap:10px; align-items:flex-start; min-width:0; max-width:100%; }
    .chat-row.user { flex-direction:row-reverse; }
    .chat-avatar { flex:0 0 auto; width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12px; color:#fff; background:var(--accent); }
    .chat-row.user .chat-avatar { background:#0f766e; }
    .chat-meta { display:flex; justify-content:space-between; gap:12px; color:var(--muted); font-size:12px; margin:0 4px; }
    .chat-row.user .chat-meta { flex-direction:row-reverse; }
    .bubble { box-sizing:border-box; flex:0 1 auto; width:fit-content; min-width:0; max-width:min(100%, 980px); overflow:hidden; border:1px solid var(--border); border-radius:14px; padding:12px 14px; background:#f8fafc; }
    .chat-row.user .bubble { background:#effdf7; border-color:#c8e7da; }
    .chat-title { font-weight:800; margin-bottom:8px; color:var(--text); }
    .chat-body { box-sizing:border-box; width:100%; max-width:100%; min-width:0; white-space:pre-wrap; line-height:1.5; background:transparent; border:0; padding:0; margin:0; max-height:none; overflow:visible; }
    .chat-body.structured { white-space:normal; }
    .chat-body .code-block { max-height:calc(1.45em * 5 + 24px); overflow:hidden; white-space:pre-wrap; word-break:break-word; }
    .chat-body .code-block pre { line-height:1.45; }
    .chat-body .block { margin:10px 0; }
    .chat-body pre { max-height:none; white-space:pre-wrap; word-break:break-word; }
    .chat-tail { box-sizing:border-box; display:flex; justify-content:space-between; gap:10px; min-width:0; max-width:100%; color:var(--muted); font-size:12px; padding:0 44px; }
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
    .submit-card { display:grid; gap:12px; min-width:0; max-width:100%; }
    .csv-table-wrap { box-sizing:border-box; width:100%; max-width:100%; max-height:360px; overflow:auto; border:1px solid var(--border); border-radius:8px; background:white; margin:8px 0; }
    .csv-table-inner { width:max-content; min-width:100%; }
    .csv-table { width:max-content; min-width:100%; border-collapse:collapse; font-size:13px; }
    .csv-table th, .csv-table td { max-width:240px; padding:7px 9px; border-bottom:1px solid var(--border); border-right:1px solid var(--border); text-align:left; vertical-align:top; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .csv-table th { position:sticky; top:0; background:#eef3f8; color:var(--text); font-weight:900; z-index:1; }
    .csv-table td { color:#344054; }
    .csv-table tr:last-child td { border-bottom:0; }
    .csv-caption { display:flex; justify-content:space-between; gap:12px; padding:8px 10px; color:var(--muted); font-size:12px; font-weight:700; background:#f8fafc; border-top:1px solid var(--border); }
    .submit-head { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; flex-wrap:wrap; }
    .submit-title { font-weight:900; color:var(--text); }
    .submit-subtitle { margin-top:3px; color:var(--muted); font-size:13px; }
    .submit-badge { display:inline-flex; align-items:center; min-height:26px; padding:0 10px; border-radius:999px; font-size:12px; font-weight:900; text-transform:uppercase; }
    .submit-badge.ok { color:#067647; background:#ecfdf3; border:1px solid #abefc6; }
    .submit-badge.error { color:#b42318; background:#fef3f2; border:1px solid #fecdca; }
    .submit-metrics { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; }
    .submit-metric { min-width:0; border:1px solid var(--border); border-radius:8px; background:#f8fafc; padding:10px; }
    .submit-metric span { display:block; color:var(--muted); font-size:12px; font-weight:800; margin-bottom:4px; }
    .submit-metric strong { display:block; color:var(--text); font-size:18px; overflow-wrap:anywhere; }
    .submit-error { border:1px solid #fecdca; border-radius:8px; background:#fef3f2; color:#b42318; padding:10px; line-height:1.45; overflow-wrap:anywhere; }
    .submission-source { box-sizing:border-box; min-width:0; max-width:100%; overflow:hidden; border:1px solid var(--border); border-radius:8px; background:#fff; padding:10px 12px; }
    .submission-source summary { cursor:pointer; font-weight:800; color:var(--text); overflow-wrap:anywhere; }
    .submission-source pre { box-sizing:border-box; width:100%; max-width:100%; min-width:0; margin:10px 0 0; background:#111827; color:#f9fafb; border-radius:6px; padding:12px; max-height:360px; overflow:auto; font-size:13px; line-height:1.45; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; }
    .message { border-radius:8px; padding:12px; background:#eef3f8; color:var(--muted); margin-top:12px; }
    .file-panel { display:grid; gap:10px; }
    .file-list { display:grid; gap:6px; max-height:240px; overflow:auto; padding-right:2px; }
    .file-row { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:8px; align-items:center; border:1px solid var(--border); border-radius:6px; background:#f8fafc; padding:8px; }
    .file-main { min-width:0; }
    .file-name { color:var(--text); font-weight:800; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .file-meta { margin-top:2px; color:var(--muted); font-size:12px; }
    .file-actions { display:flex; gap:6px; }
    .file-actions button, .file-actions a { display:inline-flex; align-items:center; justify-content:center; min-height:28px; width:auto; padding:0 8px; border-radius:6px; font-size:12px; font-weight:800; text-decoration:none; border:1px solid var(--border); }
    .file-actions button { color:var(--accent); background:#fff; }
    .file-actions button:disabled { color:var(--muted); cursor:not-allowed; background:#eef3f8; }
    .file-actions a { color:white; background:var(--accent); border-color:var(--accent); }
    .file-preview { min-width:0; border:1px solid var(--border); border-radius:6px; background:#fff; overflow:hidden; }
    .file-preview-head { display:flex; justify-content:space-between; gap:8px; padding:8px 10px; color:var(--muted); font-size:12px; font-weight:800; border-bottom:1px solid var(--border); }
    .file-preview pre { box-sizing:border-box; width:100%; max-height:260px; margin:0; padding:10px; overflow:auto; background:#111827; color:#f9fafb; font-size:12px; line-height:1.45; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; }
    .error { color:var(--bad); background:#fef3f2; border:1px solid #fecdca; }
    @media (max-width:900px){ header,.layout{display:block}.row,.cards,.submit-metrics{grid-template-columns:1fr} }
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
        <div class="field"><label for="timeLimit">Code timeout, sec</label><input id="timeLimit" type="number" value="__DEFAULT_TIME_LIMIT_SECONDS__" min="1"></div>
        <div class="row">
          <button id="start">Start LLM Run</button>
          <button class="secondary" id="stop" disabled>Stop</button>
        </div>
        <div class="message" id="message">Готово к запуску. Нужен доступ к OpenAI-compatible LLM endpoint.</div>
      </section>
      <section>
        <h2>Workspace Files</h2>
        <div class="file-panel">
          <div class="file-list" id="file-list"><div class="message">Start a run to browse files.</div></div>
          <div class="file-preview" id="file-preview">
            <div class="file-preview-head"><span>Preview</span><span>-</span></div>
            <pre>Select a text file to preview it.</pre>
          </div>
        </div>
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
const csvScrollPositions = new Map();
const expandedCodeBlocks = new Set();
const openDetails = new Set();
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
const fileListEl = document.getElementById('file-list');
const filePreviewEl = document.getElementById('file-preview');
function truncateText(value, limit=180){
  const text = cleanText(value).replace(/\\s+/g, ' ');
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}
function esc(v){return String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');}
function escAttr(v){return esc(v).replaceAll('"','&quot;').replaceAll("'","&#39;");}
function cleanText(value){return String(value ?? '').trim();}
function isProbablyCode(value){
  const text = cleanText(value);
  if(!text) return false;
  if(text.includes('\n')) return true;
  return /\b(?:import|from|def|class|for|while|if|try|except|with|return|print)\b|[=;{}]/.test(text);
}
function isPlainObject(value){return value !== null && typeof value === 'object' && !Array.isArray(value);}
function safeTextBlock(value){
  return `<div class="chat-body structured"><pre>${esc(cleanText(value) || 'Empty message.')}</pre></div>`;
}
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
function renderMarkdownish(text, options = {}){
  const shouldStrip = options.strip !== false;
  const allowCsv = options.csv === true;
  text = shouldStrip ? stripContentFromText(text) : cleanText(text);
  if(!text) return '<div class="md-paragraph">Empty message.</div>';
  const parts = [];
  const fence = /```(?:([a-zA-Z0-9_-]+)\n)?([\s\S]*?)```/g;
  let last = 0;
  for(const match of text.matchAll(fence)){
    const before = text.slice(last, match.index);
    parts.push(before.split('\n').map(renderMarkdownLine).join(''));
    const langName = cleanText(match[1] || '');
    const blockText = match[2].replace(/\n$/, '');
    if(allowCsv && (langName.toLowerCase() === 'csv' || isLikelyCsv(blockText))) {
      parts.push(renderCsvTable(blockText));
    } else {
      const lang = langName ? `<div class="code-lang">${esc(langName)}</div>` : '';
      parts.push(`<div class="code-block collapsible">${lang}<button type="button" class="code-toggle" aria-label="Expand code"></button><pre>${esc(blockText)}</pre></div>`);
    }
    last = match.index + match[0].length;
  }
  parts.push(text.slice(last).split('\n').map(renderMarkdownLine).join(''));
  return parts.join('');
}
function parseCsvLine(line){
  const cells = [];
  let current = '';
  let quoted = false;
  for(let i = 0; i < line.length; i++){
    const ch = line[i];
    const next = line[i + 1];
    if(ch === '"'){
      if(quoted && next === '"'){
        current += '"';
        i++;
      } else {
        quoted = !quoted;
      }
      continue;
    }
    if(ch === ',' && !quoted){
      cells.push(current);
      current = '';
      continue;
    }
    current += ch;
  }
  cells.push(current);
  return cells.map(cell => cell.trim());
}
function isLikelyCsv(text){
  text = cleanText(text);
  if(!text || !text.includes(',')) return false;
  const lines = text.split('\n').map(line => line.trim()).filter(Boolean);
  if(lines.length < 2 || lines.length > 300) return false;
  const first = parseCsvLine(lines[0]);
  if(first.length < 2 || first.length > 40) return false;
  const comparable = lines.slice(1, Math.min(lines.length, 6)).map(parseCsvLine);
  const matching = comparable.filter(row => Math.abs(row.length - first.length) <= 1).length;
  return matching >= Math.min(2, comparable.length);
}
function renderCsvTable(text){
  const lines = cleanText(text).split('\n').filter(line => line.trim());
  if(lines.length < 2) return `<div class="code-block">${esc(cleanText(text))}</div>`;
  const header = parseCsvLine(lines[0]);
  const rows = lines.slice(1, 13).map(parseCsvLine);
  const totalRows = Math.max(0, lines.length - 1);
  const shownCols = header.slice(0, 12);
  const hiddenCols = Math.max(0, header.length - shownCols.length);
  return `
    <div class="csv-table-wrap">
      <div class="csv-table-inner">
        <table class="csv-table">
          <thead><tr>${shownCols.map(cell => `<th>${esc(cell || '-')}</th>`).join('')}${hiddenCols ? '<th>...</th>' : ''}</tr></thead>
          <tbody>
            ${rows.map(row => `<tr>${shownCols.map((_, index) => `<td title="${escAttr(row[index] ?? '')}">${esc(row[index] ?? '')}</td>`).join('')}${hiddenCols ? `<td>+${hiddenCols} cols</td>` : ''}</tr>`).join('')}
          </tbody>
        </table>
        <div class="csv-caption">
          <span>${esc(totalRows)} row${totalRows === 1 ? '' : 's'}</span>
          <span>${esc(header.length)} column${header.length === 1 ? '' : 's'}${totalRows > rows.length ? `, showing ${rows.length}` : ''}</span>
        </div>
      </div>
    </div>`;
}
function setupCodeToggles(){
  document.addEventListener('click', (event) => {
    const button = event.target.closest('.code-toggle');
    if(!button) return;
    const block = button.closest('.code-block');
    if(!block) return;
    block.classList.toggle('is-expanded');
    if(block.dataset.uiKey) {
      if(block.classList.contains('is-expanded')) expandedCodeBlocks.add(block.dataset.uiKey);
      else expandedCodeBlocks.delete(block.dataset.uiKey);
    }
    button.setAttribute('aria-label', block.classList.contains('is-expanded') ? 'Collapse code' : 'Expand code');
  });
  document.addEventListener('toggle', (event) => {
    const details = event.target.closest('details');
    if(!details || !details.dataset.uiKey) return;
    if(details.open) openDetails.add(details.dataset.uiKey);
    else openDetails.delete(details.dataset.uiKey);
  }, true);
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
function renderTextPreview(value){
  const text = cleanText(value);
  if(!text) return '<span class="json-muted">-</span>';
  return `<pre>${esc(text)}</pre>`;
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
        ${payload.stdout ? `<div class="block-title">stdout</div>${renderTextPreview(payload.stdout)}` : ''}
        ${payload.stderr ? `<div class="block-title">stderr</div>${renderTextPreview(payload.stderr)}` : ''}
      </div>`;
  }
  if(result.command === 'submit') {
    return `
      <div class="block">
        ${renderSubmissionHtml(result)}
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
  const status = submission.status || 'unknown';
  const ok = status === 'ok';
  const error = submission.error || payload.error || '';
  const metric = payload.metric || '-';
  const value = payload.value ?? '-';
  const rows = payload.rows_checked ?? '-';
  const elapsed = submission.elapsed_ms ?? '-';
  const source = payload.submission_source || null;
  const sourcePath = source && (source.path || source.kind || 'inline');
  const sourceCode = source && source.content ? cleanText(source.content) : '';
  return `
    <div class="submit-card">
      <div class="submit-head">
        <div>
          <div class="submit-title">${ok ? 'Submission accepted' : 'Submission failed'}</div>
          <div class="submit-subtitle">${esc(submission.command || 'submit')} ${elapsed !== '-' ? `completed in ${esc(elapsed)} ms` : ''}</div>
        </div>
        <span class="submit-badge ${ok ? 'ok' : 'error'}">${esc(status)}</span>
      </div>
      ${error ? `<div class="submit-error">${esc(error)}</div>` : ''}
      <div class="submit-metrics">
        <div class="submit-metric"><span>Metric</span><strong>${esc(metric)}</strong></div>
        <div class="submit-metric"><span>Value</span><strong>${esc(value)}</strong></div>
        <div class="submit-metric"><span>Rows checked</span><strong>${esc(rows)}</strong></div>
      </div>
      ${sourceCode ? `
        <details class="submission-source">
          <summary>Код, который сформировал submission.csv</summary>
          <div class="note-meta">Источник: <span class="inline-code">${esc(sourcePath)}</span>${source.truncated ? ' · truncated' : ''}</div>
          <pre>${esc(sourceCode)}</pre>
        </details>` : ''}
    </div>`;
}
function renderPromptHtml(text, allowCsv=false){
  const fullText = cleanText(text);
  try {
    return `<div class="chat-body structured">${renderMarkdownish(fullText, {strip:false, csv:allowCsv})}</div>`;
  } catch {
    return safeTextBlock(fullText);
  }
}
function validHints(hints){
  if(!Array.isArray(hints)) return [];
  return hints.filter(h => h && typeof h === 'object' && cleanText(h.message));
}
function renderCommandNote(event){
  const name = cleanText(event.title || event.data?.command || '');
  const args = event.data?.args || {};
  const code = cleanText(args.code_or_file || '');
  const path = cleanText(args.path || args.file || '');
  const content = cleanText(args.content || args.diff || '');
  const labels = {
    list_files: 'Показываю список файлов',
    read_file: 'Открываю файл',
    write_file: 'Создаю файл',
    edit_file: 'Редактирую файл',
    load_dataset: 'Загружаю датасет',
    show_dataset_info: 'Показываю информацию о датасете',
    show_sample_rows: 'Показываю sample строки',
    run_python: 'Выполняю код',
    get_budget_status: 'Проверяю бюджет шагов',
    get_remaining_time: 'Проверяю timeout кода',
    get_trajectory: 'Показываю trajectory',
    get_hints: 'Запрашиваю подсказки',
    submit: 'Отправляю submission'
  };
  const summary = labels[name] || `Выполняю команду ${name || 'unknown'}`;
  const details = [];
  details.push(`<div class="note-meta">Команда: <span class="inline-code">${esc(renderCommandCall(name, args))}</span></div>`);
  if(name === 'run_python') {
    if(code) details.push(`<div class="note-meta">Код</div><pre>${esc(code)}</pre>`);
    else details.push('<div class="note-meta">Код не передан.</div>');
  } else {
    if(path) details.push(`<div class="note-meta">Путь: <span class="inline-code">${esc(path)}</span></div>`);
    if(args.n !== undefined) details.push(`<div class="note-meta">Строк: <span class="inline-code">${esc(args.n)}</span></div>`);
    if(content) details.push(`<div class="note-meta">Содержимое</div>${renderTextPreview(content)}`);
    if(!path && !content && args.n === undefined) {
      const argEntries = Object.keys(args || {});
      if(argEntries.length) details.push(renderJsonSummary(args));
      else details.push('<div class="note-meta">Без аргументов.</div>');
    }
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
  if(event.kind === 'prompt') return renderPromptHtml(data.content || '', false);
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
function renderTurn(turn, index){
  try {
    const promptText = turn.prompt?.data?.content || '';
    const llmText = turn.llm?.data?.content || '';
    const prompt = turn.prompt ? renderChatEvent(
      'prompt',
      'You',
      `${renderPromptHtml(promptText, index === 0)}${validHints(turn.hints).length ? `<div class="turn-hint-label">Подсказки к вашему сообщению</div><ul class="turn-hints">${validHints(turn.hints).map(h => `<li><strong>${esc(h.stage || 'hint')}</strong>: ${esc(h.message)}</li>`).join('')}</ul>` : ''}`,
      new Date(turn.prompt.time).toLocaleTimeString(),
      'user'
    ) : '';
    const llm = turn.llm ? `
      <div class="event chat-event llm assistant">
        <div class="chat-row assistant">
          <div class="chat-avatar">LLM</div>
          <div class="bubble">
            <div class="chat-title">LLM</div>
            <div class="chat-body structured">${renderMarkdownish(llmText)}</div>
            ${turn.notes.length ? `<div class="command-notes">${turn.notes.join('')}</div>` : ''}
          </div>
        </div>
        <div class="chat-tail">
          <span>LLM response</span>
          <span>${esc(new Date(turn.llm.time).toLocaleTimeString())}</span>
        </div>
      </div>` : '';
    return `<div class="turn" data-turn-index="${index}">${prompt}${llm}</div>`;
  } catch (error) {
    const promptText = turn.prompt?.data?.content || '';
    const llmText = turn.llm?.data?.content || '';
    return `
      <div class="turn" data-turn-index="${index}">
        ${turn.prompt ? renderChatEvent('prompt', 'You', safeTextBlock(promptText), new Date(turn.prompt.time).toLocaleTimeString(), 'user') : ''}
        ${turn.llm ? renderChatEvent('llm', 'LLM', safeTextBlock(llmText), new Date(turn.llm.time).toLocaleTimeString(), 'assistant') : ''}
        <div class="event error"><div class="event-title">Render fallback</div><div class="event-preview">${esc(error.message || String(error))}</div></div>
      </div>`;
  }
}
function saveCsvScrollPositions(){
  eventsEl.querySelectorAll('.turn').forEach((turn, turnIndex) => {
    turn.querySelectorAll('.csv-table-wrap').forEach((table, index) => {
      const key = table.dataset.uiKey || `turn-${turnIndex}-csv-${index}`;
      csvScrollPositions.set(key, {left: table.scrollLeft, top: table.scrollTop});
    });
  });
}
function applyTrajectoryUiState(){
  eventsEl.querySelectorAll('.turn').forEach((turn, turnIndex) => {
    turn.querySelectorAll('.code-block').forEach((block, index) => {
      const key = `turn-${turnIndex}-code-${index}`;
      block.dataset.uiKey = key;
      if(expandedCodeBlocks.has(key)) {
        block.classList.add('is-expanded');
        const button = block.querySelector('.code-toggle');
        if(button) button.setAttribute('aria-label', 'Collapse code');
      }
    });
    turn.querySelectorAll('.csv-table-wrap').forEach((table, index) => {
      const key = `turn-${turnIndex}-csv-${index}`;
      table.dataset.uiKey = key;
      const position = csvScrollPositions.get(key);
      if(position) {
        table.scrollLeft = position.left;
        table.scrollTop = position.top;
      }
    });
    turn.querySelectorAll('details').forEach((details, index) => {
      const key = `turn-${turnIndex}-details-${index}`;
      details.dataset.uiKey = key;
      if(openDetails.has(key)) details.open = true;
    });
  });
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
  renderFileList([]);
  renderFilePreview(null);
  const res = await fetch('/api/runs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({
    task_id: task.value,
    model: model.value,
    mode: mode.value,
    max_steps: Number(steps.value),
    token_limit: Number(tokens.value),
    time_limit_seconds: Number(timeLimit.value)
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
  loadRunFiles();
  if(['completed','stopped','error'].includes(data.run.status)){
    clearInterval(timer); start.disabled=false; stop.disabled=true;
  }
}
async function loadRunFiles(){
  if(!currentRun) {
    renderFileList([]);
    return;
  }
  const res = await fetch(`/api/runs/${currentRun}/files`);
  const data = await res.json();
  if(data.status !== 'ok') {
    fileListEl.innerHTML = `<div class="message error">${esc(data.error || 'Failed to load files')}</div>`;
    return;
  }
  renderFileList(data.files || []);
}
function renderFileList(files){
  if(!currentRun) {
    fileListEl.innerHTML = '<div class="message">Start a run to browse files.</div>';
    return;
  }
  if(!files.length) {
    fileListEl.innerHTML = '<div class="message">No files in workspace yet.</div>';
    return;
  }
  fileListEl.innerHTML = files.map(file => {
    const path = cleanText(file.path);
    const viewDisabled = file.previewable ? '' : 'disabled';
    const downloadHref = `/api/runs/${encodeURIComponent(currentRun)}/files/download?path=${encodeURIComponent(path)}`;
    return `
      <div class="file-row">
        <div class="file-main">
          <div class="file-name" title="${escAttr(path)}">${esc(path)}</div>
          <div class="file-meta">${esc(formatBytes(file.size || 0))}</div>
        </div>
        <div class="file-actions">
          <button type="button" data-file-view="${escAttr(path)}" ${viewDisabled}>View</button>
          <a href="${escAttr(downloadHref)}">Download</a>
        </div>
      </div>`;
  }).join('');
}
function renderFilePreview(file){
  if(!file) {
    filePreviewEl.innerHTML = '<div class="file-preview-head"><span>Preview</span><span>-</span></div><pre>Select a text file to preview it.</pre>';
    return;
  }
  filePreviewEl.innerHTML = `
    <div class="file-preview-head">
      <span>${esc(file.path)}</span>
      <span>${file.truncated ? 'truncated' : formatBytes(file.size || 0)}</span>
    </div>
    <pre>${esc(file.content || '')}</pre>`;
}
async function viewRunFile(path){
  if(!currentRun || !path) return;
  const res = await fetch(`/api/runs/${encodeURIComponent(currentRun)}/files/view?path=${encodeURIComponent(path)}`);
  const data = await res.json();
  if(data.status !== 'ok') {
    filePreviewEl.innerHTML = `<div class="file-preview-head"><span>Preview error</span><span>-</span></div><pre>${esc(data.error || 'Failed to preview file')}</pre>`;
    return;
  }
  renderFilePreview(data.file);
}
function formatBytes(size){
  const value = Number(size) || 0;
  if(value < 1024) return `${value} B`;
  if(value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}
function render(run){
  saveCsvScrollPositions();
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
      if(currentTurn) currentTurn.hints = validHints(event.data?.hints);
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
  eventsEl.innerHTML = turns.length ? turns.map(renderTurn).join('') : '<div class="message">No events yet.</div>';
applyTrajectoryUiState();
}
start.addEventListener('click', startRun);
stop.addEventListener('click', stopRun);
fileListEl.addEventListener('click', (event) => {
  const button = event.target.closest('[data-file-view]');
  if(!button) return;
  viewRunFile(button.dataset.fileView || '');
});
setupCodeToggles();
loadTasks();
</script>
</body>
</html>"""
    return page.replace("__MODEL_OPTIONS__", model_options).replace(
        "__DEFAULT_TIME_LIMIT_SECONDS__",
        str(DEFAULT_TIME_LIMIT_SECONDS),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM benchmark web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
        help="Prebuilt Docker image used for agent run_python commands.",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run agent Python commands on the host instead of a benchmark Docker container.",
    )
    parser.add_argument(
        "--keep-containers",
        action="store_true",
        help="Do not remove per-run benchmark containers after runs finish.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = AgentRunManager(
        args.project_root,
        args.tasks_dir,
        args.run_root,
        use_docker=not args.no_docker,
        docker_image=args.docker_image,
        keep_containers=args.keep_containers,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(manager))
    print(f"Agent web UI is running on http://{args.host}:{args.port}")
    if args.no_docker:
        print("Agent Python execution: host python3")
    else:
        print(f"Agent Python execution: Docker image {args.docker_image}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping agent web UI")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
