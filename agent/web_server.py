#!/usr/bin/env python3
"""Web UI for running the LLM benchmark agent."""

from __future__ import annotations

import argparse
import csv
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
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    MODEL,
    SYSTEM_MESSAGE,
    URL,
    auth_headers,
    chat_completion,
)
from agent.parser import COMMAND_NAMES, FENCED_BLOCK_RE, ParsedCommand, parse_command, parse_model_response


DEFAULT_TASKS_DIR = Path("checker/tasks")
DEFAULT_RUN_ROOT = Path("benchmark_runs")
TEXT_EXTENSIONS = {".csv", ".json", ".md", ".py", ".txt", ".tsv", ".yaml", ".yml"}
MAX_CONTEXT_CHARS = 28000
MAX_FILE_PREVIEW_CHARS = 2500


@dataclass
class RunConfig:
    task_id: str
    model: str = MODEL
    llm_url: str = URL
    mode: str = "multi-shot"
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
        user_message = build_initial_prompt(state.workspace, state.config.task_id)
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
                    url=build_chat_completions_url(state.config.llm_url),
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
                    user_message = (
                        "Твой ответ не удалось распарсить как команду.\n"
                        f"Ошибка: {exc}\n"
                        "Верни только валидную команду или несколько команд, по одной на строку."
                    )
                    continue

                results: list[dict[str, Any]] = []
                for command in commands:
                    add_event(state, "command", command.name, {"args": command.args})
                    result = state.executor.execute(command)
                    results.append(result)
                    add_event(state, "result", f"{command.name}: {result['status']}", result)
                    if command.name == "submit":
                        state.submitted = True
                        state.submission_result = result
                        state.stop_reason = "submitted"
                        break

                if state.submitted:
                    break

                if state.config.mode == "single-shot":
                    state.stop_reason = "single_shot_finished"
                    break

                feedback = state.executor.build_feedback()
                add_event(state, "feedback", "Feedback hints", feedback)
                user_message = build_followup_prompt(results, feedback, state)

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
                llm_url = _first_query_value(parsed.query, "url") or URL
                try:
                    self._send_json({"status": "ok", "models": fetch_model_options(llm_url)})
                except Exception as exc:
                    self._send_json(
                        {
                            "status": "error",
                            "error": str(exc),
                            "models": [MODEL],
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
                config = RunConfig(
                    task_id=str(payload.get("task_id") or "salary_prediction"),
                    model=str(payload.get("model") or MODEL),
                    llm_url=str(payload.get("llm_url") or URL),
                    mode=str(payload.get("mode") or "multi-shot"),
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


def build_followup_prompt(results: list[dict[str, Any]], feedback: dict[str, Any], state: RunState) -> str:
    payload = {
        "command_results": results,
        "feedback": feedback,
        "budget": {
            "used_steps": state.executor.context.used_steps,
            "max_steps": state.config.max_steps,
            "remaining_steps": max(0, state.config.max_steps - state.executor.context.used_steps),
            "requests": state.requests,
            "total_tokens": state.total_tokens,
            "token_limit": state.config.token_limit,
        },
    }
    return "Продолжай решение. Верни только следующую команду или команды.\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
    )


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
    request = urllib.request.Request(build_models_url(base_url), headers=auth_headers(), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
    return """<!doctype html>
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
    .events { display:grid; gap:10px; min-width:0; }
    .event { min-width:0; max-width:100%; overflow:hidden; border:1px solid var(--border); border-radius:8px; padding:12px; background:white; }
    .event.prompt { border-left:4px solid #155eef; }
    .event.llm { border-left:4px solid #7c3aed; }
    .event.command { border-left:4px solid #c2410c; }
    .event.result { border-left:4px solid #067647; }
    .event.error, .event.parse_error { border-left:4px solid var(--bad); }
    .event-head { display:flex; justify-content:space-between; gap:10px; color:var(--muted); font-size:13px; margin-bottom:8px; }
    .kind { font-weight:800; color:var(--accent); text-transform:uppercase; }
    .event-body { box-sizing:border-box; width:100%; min-width:0; white-space:pre-wrap; line-height:1.45; max-height:220px; overflow:auto; background:#f8fafc; border:1px solid var(--border); border-radius:6px; padding:12px; margin-top:10px; }
    .event-body.structured { white-space:normal; }
    .event.expanded .event-body { max-height:none; }
    .block { min-width:0; overflow:hidden; border:1px solid var(--border); border-radius:6px; background:white; padding:10px; margin:8px 0; }
    .block-title { font-weight:800; margin-bottom:6px; color:var(--text); }
    .block pre { background:#f8fafc; color:var(--text); border:1px solid var(--border); max-height:180px; }
    .code-block { box-sizing:border-box; width:100%; max-width:100%; background:#111827; color:#f9fafb; border-radius:6px; padding:12px; overflow:auto; max-height:340px; font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; font-size:13px; white-space:pre; }
    .inline-code { display:inline-block; box-sizing:border-box; max-width:100%; overflow:hidden; text-overflow:ellipsis; vertical-align:bottom; font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; background:#eef3f8; border:1px solid var(--border); border-radius:4px; padding:1px 4px; }
    .md-heading { font-weight:800; margin:10px 0 6px; }
    .md-paragraph { margin:4px 0; }
    .md-rule { border:0; border-top:1px solid var(--border); margin:10px 0; }
    .command-name { font-size:16px; font-weight:900; color:var(--text); }
    .command-call { max-width:100%; margin-top:8px; overflow:hidden; }
    .arg-list { display:grid; gap:8px; margin-top:8px; }
    .arg-row { min-width:0; overflow:hidden; border:1px solid var(--border); border-radius:6px; padding:10px; background:#fff; }
    .arg-name { display:block; color:var(--muted); font-weight:800; margin-bottom:6px; }
    .hint-list { margin:6px 0 0; padding-left:20px; }
    .hint-list li { margin-bottom:6px; }
    .kv { display:grid; grid-template-columns:180px minmax(0,1fr); gap:6px 12px; font-size:14px; }
    .kv span { min-width:0; overflow-wrap:anywhere; }
    .kv span:nth-child(odd) { color:var(--muted); font-weight:700; }
    .event-actions { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; }
    .small-button { width:auto; min-height:32px; padding:0 10px; font-size:13px; background:#eef3f8; color:var(--text); border:1px solid var(--border); }
    details { margin-top:10px; }
    summary { cursor:pointer; color:var(--muted); font-weight:700; }
    pre { box-sizing:border-box; width:100%; max-width:100%; margin:0; white-space:pre-wrap; word-break:break-word; max-height:280px; overflow:auto; background:#111827; color:#f9fafb; border-radius:6px; padding:12px; font-size:13px; }
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
        <div class="field"><label for="url">LLM URL</label><input id="url" value="http://llm.letovo.site:8809/openai"></div>
        <div class="field">
          <label for="model">Model</label>
          <select id="model">
            <option value="deepseek-v4-flash">deepseek-v4-flash</option>
            <option value="gemma-4-26b">gemma-4-26b</option>
          </select>
        </div>
        <div class="field">
          <label for="mode">Mode</label>
          <select id="mode">
            <option value="multi-shot">multi-shot</option>
            <option value="single-shot">single-shot</option>
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
const url = document.getElementById('url');
const model = document.getElementById('model');
const mode = document.getElementById('mode');
const steps = document.getElementById('steps');
const tokens = document.getElementById('tokens');
const start = document.getElementById('start');
const stop = document.getElementById('stop');
const message = document.getElementById('message');
const statusEl = document.getElementById('status');
const eventsEl = document.getElementById('events');
function esc(v){return String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');}
function cleanText(value){return String(value ?? '').trim();}
function renderMarkdownish(text){
  text = cleanText(text);
  if(!text) return '<div class="md-paragraph">Empty message.</div>';
  const parts = [];
  const fence = /```(\w+)?\\n([\s\S]*?)```/g;
  let last = 0;
  for(const match of text.matchAll(fence)){
    parts.push(renderPlainText(text.slice(last, match.index)));
    parts.push(`<div class="code-block">${esc(match[2].trim())}</div>`);
    last = match.index + match[0].length;
  }
  parts.push(renderPlainText(text.slice(last)));
  return parts.join('');
}
function renderPlainText(text){
  text = cleanText(text);
  if(!text) return '';
  return text
    .split('\\n')
    .map(line => {
      const trimmed = line.trim();
      if(!trimmed) return '';
      if(trimmed === '***' || trimmed === '---') return '<hr class="md-rule">';
      if(trimmed.startsWith('### ')) return `<div class="md-heading">${esc(trimmed.slice(4))}</div>`;
      if(trimmed.startsWith('## ')) return `<div class="md-heading">${esc(trimmed.slice(3))}</div>`;
      if(trimmed.startsWith('# ')) return `<div class="md-heading">${esc(trimmed.slice(2))}</div>`;
      return `<div class="md-paragraph">${esc(trimmed)}</div>`;
    })
    .join('');
}
function isProbablyCode(value){
  return typeof value === 'string' && (
    value.includes('\\n') ||
    value.includes('import ') ||
    value.includes('def ') ||
    value.includes('pd.read_csv') ||
    value.includes('print(')
  );
}
function renderValue(value){
  if(isProbablyCode(value)) return `<div class="code-block">${esc(cleanText(value))}</div>`;
  if(typeof value === 'string') return `<span class="inline-code">${esc(cleanText(value))}</span>`;
  return `<pre>${esc(JSON.stringify(value, null, 2))}</pre>`;
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
      <pre>${esc(JSON.stringify(payload || result, null, 2))}</pre>
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
      <summary>Secondary data</summary>
      <pre>${esc(JSON.stringify(submission, null, 2))}</pre>
    </details>`;
}
function renderPromptHtml(text){
  const parsed = tryParseFollowupPrompt(text);
  if(!parsed) return `<div class="event-body structured">${renderMarkdownish(text)}</div>`;
  const payload = parsed.payload;
  const results = payload.command_results || [];
  const hints = payload.feedback?.hints || [];
  const budget = payload.budget || {};
  return `
    <div class="event-body structured">
      <div class="block">
        <div class="block-title">Instruction to LLM</div>
        <div>${esc(parsed.intro)}</div>
      </div>
      <div class="block">
        <div class="block-title">Command results sent back</div>
        ${results.length ? results.map(commandResultSummary).join('') : '<div>No command results.</div>'}
      </div>
      <div class="block">
        <div class="block-title">Feedback hints</div>
        ${hints.length ? `<ul class="hint-list">${hints.map(h => `<li><strong>${esc(h.stage)}</strong>: ${esc(h.message)}</li>`).join('')}</ul>` : '<div>No hints.</div>'}
      </div>
      <div class="block">
        <div class="block-title">Budget</div>
        <div class="kv">
          <span>Steps</span><span>${esc(budget.used_steps ?? 0)} / ${esc(budget.max_steps ?? '-')}</span>
          <span>Remaining steps</span><span>${esc(budget.remaining_steps ?? '-')}</span>
          <span>Requests</span><span>${esc(budget.requests ?? '-')}</span>
          <span>Tokens</span><span>${esc(budget.total_tokens ?? 0)} / ${esc(budget.token_limit ?? '-')}</span>
        </div>
      </div>
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
  if(event.kind === 'llm') return `<div class="event-body structured">${renderMarkdownish(data.content || '')}</div>`;
  if(event.kind === 'command') return renderCommandHtml(event);
  if(event.kind === 'result') return `<div class="event-body structured">${commandResultSummary(data)}</div>`;
  if(event.kind === 'feedback') return renderFeedbackHtml(data);
  return `<div class="event-body">${esc(eventMainText(event))}</div>`;
}
function eventLabel(kind){
  const labels = {
    prompt: 'Prompt to LLM',
    llm: 'LLM answer',
    command: 'Command',
    result: 'Command result',
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
    llm_url: url.value,
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
  eventsEl.innerHTML = run.events.length ? run.events.slice().reverse().map((e, index) => {
    const details = JSON.stringify(e.data, null, 2);
    return `
      <div class="event ${esc(e.kind)}" data-event-index="${index}">
        <div class="event-head"><span class="kind">${esc(eventLabel(e.kind))}</span><span>${esc(new Date(e.time).toLocaleTimeString())}</span></div>
        <strong>${esc(e.title)}</strong>
        ${eventBodyHtml(e)}
        <div class="event-actions">
          <button class="small-button" type="button" data-action="expand">Expand</button>
        </div>
        <details>
          <summary>Secondary data</summary>
          <pre>${esc(details)}</pre>
        </details>
      </div>`;
  }).join('') : '<div class="message">No events yet.</div>';
  eventsEl.querySelectorAll('[data-action="expand"]').forEach(button => {
    button.addEventListener('click', () => {
      const event = button.closest('.event');
      event.classList.toggle('expanded');
      button.textContent = event.classList.contains('expanded') ? 'Collapse' : 'Expand';
    });
  });
}
start.addEventListener('click', startRun);
stop.addEventListener('click', stopRun);
loadTasks();
</script>
</body>
</html>"""


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
