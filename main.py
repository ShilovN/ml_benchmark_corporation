#!/usr/bin/env python3
"""Run the full ML benchmark loop for an OpenAI-compatible model server."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.executor import AgentContext, CommandExecutor
from agent.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    MODEL,
    SYSTEM_MESSAGE,
    URL,
    Message,
    auth_headers,
    chat_completion,
)
from agent.parser import (
    COMMAND_NAMES,
    FENCED_BLOCK_RE,
    ParsedCommand,
    parse_command,
    parse_model_response,
)


TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".tsv",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "checker/submissions",
}
SKIP_FILENAMES = {
    "agent_history.txt",
}
DEFAULT_CONTEXT_CHARS_PER_FILE = 900
DEFAULT_CONTEXT_TOTAL_CHARS = 10000
DEFAULT_DOCKER_IMAGE = "ml-benchmark-runner:latest"
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_MESSAGE_CHARS = 2500
MAX_FOLLOWUP_PROMPT_CHARS = 30000
MIN_COMPACT_STRING_CHARS = 1000


@dataclass
class BenchmarkStats:
    started_at: float = field(default_factory=time.monotonic)
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    executed_commands: int = 0
    parse_errors: int = 0
    command_errors: int = 0
    submitted: bool = False
    submission_result: dict[str, Any] | None = None
    stop_reason: str = "not_started"

    @property
    def elapsed_seconds(self) -> float:
        return round(time.monotonic() - self.started_at, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full ML task benchmark loop.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--task-id", default="salary_prediction")
    parser.add_argument("--tasks-dir", type=Path, default=Path("checker/tasks"))
    parser.add_argument("--url", default=URL, help="OpenAI-compatible chat completions endpoint.")
    parser.add_argument(
        "--base-url",
        help=(
            "OpenAI-compatible base URL. Examples: http://host:8000, "
            "http://host:8000/v1, http://host:8000/compatible-mode/v1."
        ),
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--token-limit", type=int, default=120000)
    parser.add_argument("--time-limit-seconds", type=int, default=3600)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--context-chars-per-file", type=int, default=DEFAULT_CONTEXT_CHARS_PER_FILE)
    parser.add_argument("--context-total-chars", type=int, default=DEFAULT_CONTEXT_TOTAL_CHARS)
    parser.add_argument(
        "--docker-container",
        help="Existing Docker container name/id to reuse instead of creating a new one.",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run Python commands on the host instead of a benchmark Docker container.",
    )
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
        help="Prebuilt Docker image used for the benchmark workspace container.",
    )
    parser.add_argument(
        "--docker-name",
        help="Name for the created Docker container. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("benchmark_runs"),
        help="Directory where per-run mounted workspaces are created.",
    )
    parser.add_argument(
        "--keep-container",
        action="store_true",
        help="Do not remove the created benchmark container after the run.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=Path("agent_history.txt"),
        help="JSONL command trajectory file inside the workspace.",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip the initial /models connectivity check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_workspace = args.workspace.resolve()
    tasks_dir = resolve_tasks_dir(source_workspace, args.tasks_dir)
    llm_url = resolve_llm_url(args)
    docker_container: str | None = None
    created_container = False

    if args.docker_container:
        docker_container = args.docker_container
        start_container(docker_container)
    elif not args.no_docker:
        workspace = prepare_docker_workspace(
            source_workspace,
            tasks_dir=tasks_dir,
            task_id=args.task_id,
            run_root=args.run_root,
        )
        docker_container = args.docker_name or default_container_name(args.task_id)
        create_container(
            name=docker_container,
            image=args.docker_image,
            workspace=workspace,
        )
        created_container = True
    else:
        workspace = source_workspace

    stats = BenchmarkStats()
    executor = CommandExecutor(
        AgentContext(
            workspace=workspace,
            task_id=args.task_id,
            tasks_dir=tasks_dir,
            max_steps=args.max_steps,
            time_limit_seconds=args.time_limit_seconds,
            history_file=args.history_file,
            docker_container=docker_container,
        )
    )
    history: list[Message] = []
    user_message = build_initial_prompt(
        workspace,
        task_id=args.task_id,
        per_file_limit=args.context_chars_per_file,
        total_limit=args.context_total_chars,
        stats=stats,
        args=args,
    )

    print("Benchmark started")
    print(f"workspace: {workspace}")
    print(f"task_id: {args.task_id}")
    print(f"model: {args.model}")
    print(f"llm_url: {llm_url}")
    if docker_container:
        print(f"docker_container: {docker_container}")

    try:
        if not args.no_preflight:
            try:
                preflight_llm(llm_url, args.request_timeout)
            except RuntimeError as exc:
                stats.stop_reason = "llm_preflight_error"
                print(str(exc))
                print(json.dumps(build_report(stats, executor), ensure_ascii=False, indent=2, default=str))
                return 1

        while True:
            if time_exhausted(stats, args.time_limit_seconds):
                stats.stop_reason = "time_limit"
                break
            if args.token_limit and stats.total_tokens >= args.token_limit:
                stats.stop_reason = "token_limit"
                break
            if executor.context.used_steps >= args.max_steps:
                stats.stop_reason = "step_limit"
                break

            print_debug_state("before_model_request", stats, args, history, user_message)
            try:
                response = chat_completion(
                    user_message,
                    system_message=SYSTEM_MESSAGE,
                    history=compact_history_for_model(history),
                    url=llm_url,
                    model=args.model,
                    max_tokens=response_max_tokens(stats, args),
                    temperature=args.temperature,
                    timeout=args.request_timeout,
                )
            except Exception as exc:
                stats.stop_reason = "model_request_error"
                user_message = f"Model request failed: {exc}"
                print(user_message)
                break

            stats.requests += 1
            update_token_stats(stats, response.get("usage", {}))
            assistant_text = extract_assistant_text(response)
            print_debug_state(
                "after_model_response",
                stats,
                args,
                history,
                user_message,
                assistant_text=assistant_text,
                usage=response.get("usage", {}),
            )
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_text})

            try:
                commands = extract_commands(assistant_text)
            except ValueError as exc:
                stats.parse_errors += 1
                user_message = build_parse_error_prompt(assistant_text, exc, stats, args)
                continue

            command_results: list[dict[str, Any]] = []
            for command in commands:
                if stats.submitted:
                    command_results.append(
                        {
                            "status": "error",
                            "command": command.name,
                            "error": "submit() has already been executed; no more commands are allowed.",
                        }
                    )
                    continue

                result = executor.execute(command)
                stats.executed_commands += 1
                if result["status"] == "error":
                    stats.command_errors += 1
                command_results.append(result)

                if command.name == "submit":
                    stats.submitted = True
                    stats.submission_result = result
                    stats.stop_reason = "submitted"
                    break

            if stats.submitted:
                break

            feedback = executor.build_feedback() if command_results else None
            user_message = build_followup_prompt(command_results, stats, args, feedback)
    finally:
        if created_container and not args.keep_container and docker_container:
            remove_container(docker_container)

    print(json.dumps(build_report(stats, executor), ensure_ascii=False, indent=2, default=str))
    return 0 if stats.stop_reason == "submitted" else 1


def start_container(container: str) -> None:
    subprocess.run(["docker", "start", container], check=True, capture_output=True, text=True)


def create_container(*, name: str, image: str, workspace: Path) -> None:
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-v",
            f"{workspace}:/workspace",
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


def default_container_name(task_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_task_id = "".join(char if char.isalnum() or char in "-_" else "-" for char in task_id)
    return f"ml-benchmark-{safe_task_id}-{timestamp}"


def resolve_tasks_dir(source_workspace: Path, tasks_dir: Path) -> Path:
    if tasks_dir.is_absolute():
        return tasks_dir.resolve()
    return (source_workspace / tasks_dir).resolve()


def prepare_docker_workspace(
    source_workspace: Path,
    *,
    tasks_dir: Path,
    task_id: str,
    run_root: Path,
) -> Path:
    task_dir = tasks_dir / task_id
    config_path = task_dir / "task.json"
    if not config_path.exists():
        raise ValueError(f"Task config not found: {config_path}")

    task_config = json.loads(config_path.read_text(encoding="utf-8"))
    run_workspace = create_run_workspace(source_workspace, run_root, task_id)
    public_files = task_config.get("public_files") or ["train.csv", "test.csv"]
    for filename in public_files:
        source = task_dir / str(filename)
        if not source.exists():
            raise ValueError(f"Public task file not found: {source}")
        destination = run_workspace / Path(str(filename)).name
        shutil.copy2(source, destination)

    public_config = {
        key: value
        for key, value in task_config.items()
        if key not in {"answer_file", "private_files"}
    }
    (run_workspace / "task.json").write_text(
        json.dumps(public_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_workspace


def create_run_workspace(source_workspace: Path, run_root: Path, task_id: str) -> Path:
    root = run_root if run_root.is_absolute() else source_workspace / run_root
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    workspace = root / f"{task_id}-{timestamp}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def resolve_llm_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return build_chat_completions_url(args.base_url)
    return build_chat_completions_url(args.url)


def build_chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/openai"):
        return f"{stripped}/chat/completions"
    if stripped.endswith("/v1") or stripped.endswith("/compatible-mode/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/compatible-mode/v1/chat/completions"


def preflight_llm(chat_url: str, timeout: int) -> None:
    models_url = build_models_url(chat_url)
    request = urllib.request.Request(models_url, headers=auth_headers(), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, 10)) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM preflight failed: HTTP {exc.code} from {models_url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "LLM preflight failed: cannot connect to "
            f"{models_url}. Chat endpoint would be {chat_url}. Reason: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            "LLM preflight failed: timed out while connecting to "
            f"{models_url}. Chat endpoint would be {chat_url}."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "LLM preflight failed: cannot connect to "
            f"{models_url}. Chat endpoint would be {chat_url}. Reason: {exc}"
        ) from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM preflight failed: {models_url} returned non-JSON data.") from exc
    if not isinstance(data, dict) or "data" not in data:
        raise RuntimeError(f"LLM preflight failed: {models_url} is not an OpenAI-compatible /models endpoint.")


def build_models_url(chat_url: str) -> str:
    suffix = "/chat/completions"
    stripped = chat_url.rstrip("/")
    if stripped.endswith(suffix):
        return stripped[: -len(suffix)] + "/models"
    return stripped + "/models"


def build_initial_prompt(
    workspace: Path,
    *,
    task_id: str,
    per_file_limit: int,
    total_limit: int,
    stats: BenchmarkStats,
    args: argparse.Namespace,
) -> str:
    file_previews = collect_file_previews(
        workspace,
        per_file_limit=per_file_limit,
        total_limit=total_limit,
    )
    budget_line = format_budget_line(stats, args)
    return (
        f"{budget_line}\n\n"
        "ML benchmark. Отвечай только агентскими командами, по одной на строку. "
        "Сначала проверь данные, затем быстро готовь submission.csv и вызови submit(\"submission.csv\"). "
        "Не трать токены на объяснения.\n\n"
        f"task_id: {task_id}\n"
        "Компактный обзор workspace. Для деталей используй read_file/load_dataset/show_sample_rows.\n\n"
        f"{file_previews}"
    )


def collect_file_previews(workspace: Path, *, per_file_limit: int, total_limit: int) -> str:
    chunks: list[str] = []
    used_chars = 0
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or should_skip_path(workspace, path):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        relative = path.relative_to(workspace)
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        hidden_answer_file = path.name.startswith("answers.") or path.name.startswith("answer.")
        if hidden_answer_file:
            continue

        preview = compact_file_preview(relative, content, per_file_limit)
        chunk = f"{relative} ({len(content)} chars)\n{preview}\n"
        if used_chars + len(chunk) > total_limit:
            chunks.append("...[initial context truncated]")
            break
        chunks.append(chunk)
        used_chars += len(chunk)
    return "\n".join(chunks) if chunks else "(no readable text files found)"


def compact_file_preview(relative: Path, content: str, limit: int) -> str:
    suffix = relative.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        lines = content.splitlines()
        delimiter = "\\t" if suffix == ".tsv" else ","
        preview_lines = lines[:4]
        preview = "\n".join(preview_lines)
        if len(lines) > len(preview_lines):
            preview += f"\n...[{len(lines) - len(preview_lines)} more rows; delimiter={delimiter}]"
    else:
        preview = content
    return truncate_middle(preview, limit)


def should_skip_path(workspace: Path, path: Path) -> bool:
    relative = path.relative_to(workspace)
    parts = set(relative.parts)
    if parts & SKIP_DIRS:
        return True
    relative_text = relative.as_posix()
    if any(relative_text == skip or relative_text.startswith(f"{skip}/") for skip in SKIP_DIRS):
        return True
    if path.name in SKIP_FILENAMES:
        return True
    if path.name.startswith("submission") and path.suffix.lower() in {".csv", ".txt", ".json"}:
        return True
    return False


def extract_assistant_text(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected model response format: {response}") from exc


def extract_commands(response_text: str) -> list[ParsedCommand]:
    try:
        return [parse_model_response(response_text)]
    except ValueError as first_error:
        commands: list[ParsedCommand] = []
        candidates = command_candidates(response_text)
        for candidate in candidates:
            try:
                commands.append(parse_command(candidate))
            except ValueError:
                continue
        if commands:
            return commands
        raise first_error


def command_candidates(response_text: str) -> list[str]:
    candidates: list[str] = []
    for block in FENCED_BLOCK_RE.findall(response_text):
        text = block.strip()
        if text:
            candidates.append(text)

    for line in response_text.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("{"):
            candidates.append(text)
        elif any(text.startswith(f"{name}(") and text.endswith(")") for name in COMMAND_NAMES):
            candidates.append(text)

    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def build_parse_error_prompt(
    assistant_text: str,
    exc: Exception,
    stats: BenchmarkStats,
    args: argparse.Namespace,
) -> str:
    return (
        f"{format_budget_line(stats, args)}\n\n"
        "Твой прошлый ответ не распарсился как команда.\n"
        f"Ошибка парсинга: {exc}\n\n"
        "Верни только валидные команды, по одной на строку. Без объяснений.\n\n"
        f"Прошлый ответ:\n{truncate_middle(assistant_text, 1200)}"
    )


def build_followup_prompt(
    command_results: list[dict[str, Any]],
    stats: BenchmarkStats,
    args: argparse.Namespace,
    feedback: dict[str, Any] | None = None,
) -> str:
    budget_status = build_budget_status(stats, args)
    payload = {
        "command_results": command_results,
        "benchmark_status": {
            "elapsed_seconds": stats.elapsed_seconds,
            "requests": stats.requests,
            "total_tokens": stats.total_tokens,
            "token_limit": args.token_limit,
            "remaining_tokens": budget_status["remaining_tokens"],
            "remaining_token_percent": budget_status["remaining_token_percent"],
            "used_steps": stats.executed_commands,
            "max_steps": args.max_steps,
            "remaining_iterations": budget_status["remaining_iterations"],
            "remaining_seconds": max(0.0, args.time_limit_seconds - stats.elapsed_seconds),
        },
    }
    if feedback is not None:
        payload["feedback"] = feedback
    prefix = (
        f"{format_budget_line(stats, args)} "
        "Дальше верни только команды.\n\n"
    )
    return prefix + compact_json_for_prompt(payload, MAX_FOLLOWUP_PROMPT_CHARS - len(prefix))


def compact_json_for_prompt(payload: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) <= max_chars:
        return text

    compacted = copy.deepcopy(payload)
    low = MIN_COMPACT_STRING_CHARS
    high = longest_string_length(compacted)
    best_text: str | None = None
    while low <= high:
        limit = (low + high) // 2
        candidate = truncate_long_strings(compacted, limit)
        candidate_text = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(candidate_text) <= max_chars:
            best_text = candidate_text
            low = limit + 1
        else:
            high = limit - 1

    if best_text is not None:
        return best_text

    compacted = truncate_long_strings(compacted, MIN_COMPACT_STRING_CHARS)
    return json.dumps(compacted, ensure_ascii=False, separators=(",", ":"), default=str)


def longest_string_length(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return max((longest_string_length(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return max((longest_string_length(item) for item in value), default=0)
    return 0


def truncate_long_strings(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return truncate_middle(value, limit)
    if isinstance(value, dict):
        return {key: truncate_long_strings(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [truncate_long_strings(item, limit) for item in value]
    return value


def truncate_middle(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit < 80:
        return value[:limit]
    marker = f"...[truncated {len(value) - limit} chars]..."
    keep = max(0, limit - len(marker))
    head = keep // 2
    tail = keep - head
    return value[:head] + marker + value[-tail:]


def build_budget_status(stats: BenchmarkStats, args: argparse.Namespace) -> dict[str, Any]:
    token_limit = getattr(args, "token_limit", None)
    max_steps = getattr(args, "max_steps", 0)
    remaining_iterations = max(0, max_steps - stats.executed_commands)
    remaining_tokens: int | None = None
    remaining_token_percent: float | None = None
    if token_limit:
        remaining_tokens = max(0, token_limit - stats.total_tokens)
        remaining_token_percent = round((remaining_tokens / token_limit) * 100, 2)
    return {
        "remaining_tokens": remaining_tokens,
        "remaining_token_percent": remaining_token_percent,
        "remaining_iterations": remaining_iterations,
    }


def format_budget_line(stats: BenchmarkStats, args: argparse.Namespace) -> str:
    status = build_budget_status(stats, args)
    percent = status["remaining_token_percent"]
    if percent is None:
        token_text = "лимит токенов не задан"
    else:
        token_text = f"осталось {percent}% токенов"
    line = f"Бюджет: {token_text}; осталось итераций: {status['remaining_iterations']}."
    directive = budget_directive(percent)
    if directive:
        line += f" {directive}"
    return line


def budget_directive(percent: float | None) -> str:
    if percent is None:
        return ""
    if percent < 15:
        return "КРИТИЧНО: вызывай submit с лучшим текущим файлом сейчас, иначе проиграешь."
    if percent < 30:
        return "Экономь токены: следующий шаг должен готовить/проверять submission или делать submit, иначе проиграешь."
    if percent < 50:
        return "Экономь токены: прекращай исследование, делай простой baseline и двигайся к submit."
    return ""


def response_max_tokens(stats: BenchmarkStats, args: argparse.Namespace) -> int:
    configured = getattr(args, "max_tokens", DEFAULT_MAX_TOKENS)
    percent = build_budget_status(stats, args)["remaining_token_percent"]
    if percent is None:
        return configured
    if percent < 15:
        return min(configured, 512)
    if percent < 30:
        return min(configured, 768)
    if percent < 50:
        return min(configured, 1024)
    return configured


def compact_history_for_model(history: list[Message]) -> list[Message]:
    if len(history) <= MAX_HISTORY_MESSAGES:
        return [compact_history_message(message) for message in history]
    compacted = [
        {
            "role": "user",
            "content": (
                f"[history compacted: hidden {len(history) - MAX_HISTORY_MESSAGES} old messages. "
                "Use get_trajectory() if old command results are needed.]"
            ),
        }
    ]
    compacted.extend(compact_history_message(message) for message in history[-MAX_HISTORY_MESSAGES:])
    return compacted


def compact_history_message(message: Message) -> Message:
    return {
        "role": message.get("role", "user"),
        "content": truncate_middle(message.get("content", ""), MAX_HISTORY_MESSAGE_CHARS),
    }


def print_debug_state(
    event: str,
    stats: BenchmarkStats,
    args: argparse.Namespace,
    history: list[Message],
    user_message: str,
    *,
    assistant_text: str | None = None,
    usage: Any | None = None,
) -> None:
    budget_status = build_budget_status(stats, args)
    token_limit = getattr(args, "token_limit", None)
    max_steps = getattr(args, "max_steps", None)
    title = event.replace("_", " ").title()
    print(f"\n=== DEBUG: {title} ===")
    print(f"Elapsed: {stats.elapsed_seconds}s | Requests: {stats.requests}")
    print(
        "Tokens: "
        f"{stats.total_tokens}/{format_optional_int(token_limit)} total "
        f"(prompt {stats.prompt_tokens}, completion {stats.completion_tokens}) | "
        f"remaining {format_optional_int(budget_status['remaining_tokens'])} "
        f"({format_optional_percent(budget_status['remaining_token_percent'])})"
    )
    print(f"Next max_tokens: {response_max_tokens(stats, args)}")
    print(
        "Iterations: "
        f"{stats.executed_commands}/{format_optional_int(max_steps)} used | "
        f"remaining {budget_status['remaining_iterations']}"
    )
    if usage is not None:
        print(f"Last usage: {format_usage(usage)}")
    print(f"History: {len(history)} messages")
    for index, message in enumerate(history[-6:], start=max(1, len(history) - 5)):
        role = str(message.get("role", "unknown")).upper()
        content = message.get("content", "")
        print(f"  [{index:02d}] {role} ({len(content)} chars)")
        print(indent_preview(preview_text(content, limit=700), prefix="       "))
    print("Next user message:")
    print(indent_preview(preview_text(user_message, limit=1000), prefix="  "))
    if assistant_text is not None:
        print("Assistant response:")
        print(indent_preview(preview_text(assistant_text, limit=1000), prefix="  "))
    print("=" * 32)


def format_optional_int(value: Any) -> str:
    return "unlimited" if value is None else str(value)


def format_optional_percent(value: Any) -> str:
    return "n/a" if value is None else f"{value}%"


def format_usage(usage: Any) -> str:
    if not isinstance(usage, dict):
        return str(usage)
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", 0)
    return f"prompt {prompt}, completion {completion}, total {total}"


def indent_preview(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def preview_text(text: str, limit: int = 1200) -> str:
    compact = text.replace("\r\n", "\n")
    if len(compact) <= limit:
        return compact
    return compact[:limit] + f"... <truncated {len(compact) - limit} chars>"


def update_token_stats(stats: BenchmarkStats, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    stats.prompt_tokens += prompt_tokens
    stats.completion_tokens += completion_tokens
    stats.total_tokens += total_tokens


def time_exhausted(stats: BenchmarkStats, time_limit_seconds: int) -> bool:
    return stats.elapsed_seconds >= time_limit_seconds


def build_report(stats: BenchmarkStats, executor: CommandExecutor) -> dict[str, Any]:
    report = {
        "stop_reason": stats.stop_reason,
        "elapsed_seconds": stats.elapsed_seconds,
        "requests": stats.requests,
        "tokens": {
            "prompt": stats.prompt_tokens,
            "completion": stats.completion_tokens,
            "total": stats.total_tokens,
        },
        "commands": {
            "executed": stats.executed_commands,
            "parse_errors": stats.parse_errors,
            "command_errors": stats.command_errors,
        },
        "submission": stats.submission_result,
        "history_file": str(executor.history_path),
    }
    if stats.submission_result and stats.submission_result.get("status") == "ok":
        result = stats.submission_result.get("result", {})
        report["metric"] = {
            "name": result.get("metric"),
            "value": result.get("value"),
            "rows_checked": result.get("rows_checked"),
        }
    return report


if __name__ == "__main__":
    raise SystemExit(main())
