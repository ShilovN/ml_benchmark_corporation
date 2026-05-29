#!/usr/bin/env python3
"""Run the full ML benchmark loop for an OpenAI-compatible model server."""

from __future__ import annotations

import argparse
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

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from agent.executor import AgentContext, CommandExecutor
from agent.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    MODEL,
    SYSTEM_MESSAGE,
    URL,
    Message,
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
DEFAULT_CONTEXT_CHARS_PER_FILE = 2500
DEFAULT_CONTEXT_TOTAL_CHARS = 30000
DEFAULT_DOCKER_IMAGE = "ml-benchmark-runner:latest"


class AutoGym:
    def __init__(self, dataset_path: str | Path, target_column: str):
        df = pd.read_csv(dataset_path)
        target = target_column
        self.mode = "flexible"
        self.logs = []
        self.current_stage = "EDA"
        self.max_attempts = 5
        self.attempts_made = 0
        self.final_evaluated = False

        trainval, test = train_test_split(df, test_size=0.2, random_state=42)
        train, val = train_test_split(trainval, test_size=0.2, random_state=42)

        self.train_data = train
        self.val_x = val.drop(columns=[target])
        self.val_y = val[target]
        self.private_x = test.drop(columns=[target])
        self.private_y = test[target]
        self.candidates = {}

        self.logs.append(
            {
                "action": "init_env",
                "train_size": len(self.train_data),
                "val_size": len(self.val_x),
                "test_size": len(self.private_x),
            }
        )

    def _trigger_fallback(self):
        if self.candidates:
            return max(self.candidates, key=lambda k: self.candidates[k]["val_score"])
        return None

    def run_replay_and_evaluate(self, chosen_id=None):
        if self.final_evaluated:
            return {
                "status": "rejected",
                "message": "Private final evaluation уже была выполнена. Повторный запуск запрещен.",
            }

        if chosen_id is None or chosen_id not in self.candidates:
            chosen_id = self._trigger_fallback()

        if chosen_id is None:
            return {
                "status": "rejected",
                "message": "Нет валидированных кандидатов для final evaluation.",
            }

        best_candidate = self.candidates[chosen_id]
        agent_code_used = best_candidate["agent_code"]
        expected_val_score = best_candidate["val_score"]

        self.logs.append({"action": "replay_started", "candidate_id": chosen_id})

        try:
            replay_vars = {"train_df": self.train_data}
            exec(agent_code_used, {}, replay_vars)

            reproduced_model = replay_vars.get("model")
            if reproduced_model is None:
                raise KeyError("В коде агента после повторного запуска не найден объект 'model'")

            replay_predictions = reproduced_model.predict(self.val_x)
            replay_val_score = accuracy_score(self.val_y, replay_predictions)

            if abs(replay_val_score - expected_val_score) > 1e-5:
                self.logs.append(
                    {
                        "action": "replay_warning",
                        "candidate_id": chosen_id,
                        "message": "Скор изменился при повторном запуске",
                    }
                )
            else:
                self.logs.append({"action": "replay_success", "candidate_id": chosen_id})

        except Exception:
            import traceback

            error_msg = traceback.format_exc()
            self.logs.append({"action": "replay_failed", "candidate_id": chosen_id, "error": error_msg})
            return {
                "status": "rejected",
                "message": "Финальное решение отклонено: код не воспроизводится (падает при перезапуске).",
                "error": error_msg,
            }

        final_predictions = reproduced_model.predict(self.private_x)
        final_private_score = accuracy_score(self.private_y, final_predictions)
        self.final_evaluated = True

        self.logs.append(
            {
                "action": "final_evaluation_success",
                "candidate_id": chosen_id,
                "private_score": final_private_score,
            }
        )

        return {
            "status": "success",
            "chosen_candidate_id": chosen_id,
            "validation_score": replay_val_score,
            "final_private_score": final_private_score,
        }

    def set_mode(self, mode: str):
        allowed_modes = ["single-shot", "repeated", "fixed-transitions", "flexible"]
        if mode not in allowed_modes:
            raise ValueError(f"Неизвестный режим. Выберите из: {allowed_modes}")

        self.mode = mode
        self.attempts_made = 0
        self.current_stage = "EDA"
        self.logs.append({"action": "set_mode", "mode": mode})

    def _execute_code(self, agent_code: str) -> dict:
        try:
            local_vars = {"train_df": self.train_data}
            exec(agent_code, {}, local_vars)

            model = local_vars.get("model")
            if model is None:
                error_msg = "Объект 'model' не найден в вашем коде."
                self.logs.append({"action": "execute_failed", "error": error_msg})
                return {"success": False, "error": error_msg}

            preds = model.predict(self.val_x)
            val_score = accuracy_score(self.val_y, preds)

            attempt_id = f"attempt_{len(self.candidates) + 1}"
            self.candidates[attempt_id] = {
                "agent_code": agent_code,
                "val_score": val_score,
            }

            self.logs.append(
                {
                    "action": "execute_success",
                    "candidate_id": attempt_id,
                    "val_score": val_score,
                }
            )
            return {"success": True, "val_score": val_score, "candidate_id": attempt_id, "error": None}

        except Exception:
            import traceback

            err_trace = traceback.format_exc()
            self.logs.append({"action": "execute_runtime_error", "error": err_trace})
            return {"success": False, "error": err_trace}

    def _execute_stage_code(self, agent_code: str) -> dict:
        try:
            local_vars = {"train_df": self.train_data}
            exec(agent_code, {}, local_vars)
            self.logs.append({"action": "stage_execute_success"})
            return {"success": True, "error": None}

        except Exception:
            import traceback

            err_trace = traceback.format_exc()
            self.logs.append({"action": "stage_execute_runtime_error", "error": err_trace})
            return {"success": False, "error": err_trace}

    def step(self, agent_code: str, stage_action: str = None) -> dict:
        if stage_action:
            stage_action = stage_action.upper()

        self.attempts_made += 1

        if self.mode == "single-shot":
            if self.attempts_made > 1:
                self.logs.append({"action": "rule_violation", "error": "Превышение попыток в single-shot"})
                return {"status": "Rejected", "error": "В режиме Single-shot разрешена только 1 попытка."}

            self._execute_code(agent_code)
            return {"status": "Accepted"}

        if self.mode == "repeated":
            if self.attempts_made > self.max_attempts:
                self.logs.append({"action": "rule_violation", "error": "Превышение лимита попыток в repeated"})
                return {"status": "Rejected", "error": f"Превышен лимит попыток ({self.max_attempts})."}

            res = self._execute_code(agent_code)
            if not res["success"]:
                return {
                    "status": "Execution Error",
                    "validation_score": None,
                    "message": "В коде произошла ошибка. Логи компилятора скрыты настройками режима.",
                }

            return {
                "status": "Success",
                "validation_score": res["val_score"],
            }

        if self.mode == "fixed-transitions":
            pipeline = ["EDA", "FEATURES", "TRAIN"]

            if stage_action not in pipeline:
                self.logs.append({"action": "rule_violation", "error": f"Неизвестный шаг пайплайна: {stage_action}"})
                return {"status": "Rejected", "error": f"Неизвестный шаг. Возможные шаги: {pipeline}"}

            if stage_action != self.current_stage:
                self.logs.append(
                    {
                        "action": "rule_violation",
                        "error": f"Нарушен порядок шагов. Ожидался: {self.current_stage}, вызван: {stage_action}",
                    }
                )
                return {
                    "status": "Pipeline Violation",
                    "error": (
                        f"Нарушена последовательность! Сейчас вы должны выполнять этап: {self.current_stage}. "
                        f"Вызов этапа {stage_action} заблокирован."
                    ),
                }

            if stage_action == "TRAIN":
                res = self._execute_code(agent_code)
            else:
                res = self._execute_stage_code(agent_code)

            if not res["success"]:
                return {"status": "Runtime Error", "traceback": res["error"]}

            current_idx = pipeline.index(self.current_stage)
            if current_idx < len(pipeline) - 1:
                self.current_stage = pipeline[current_idx + 1]

            response = {
                "status": "Success",
                "message": f"Этап {stage_action} пройден успешно. Следующий обязательный этап: {self.current_stage}",
            }

            if "val_score" in res:
                response["validation_score"] = res["val_score"]
                response["candidate_id"] = res["candidate_id"]

            return response

        if self.mode == "flexible":
            res = self._execute_code(agent_code)
            if not res["success"]:
                return {
                    "status": "Runtime Error",
                    "traceback": res["error"],
                    "hint": "Проверьте совместимость типов данных или размерность матриц.",
                }

            return {
                "status": "Success",
                "validation_score": res["val_score"],
                "candidate_id": res["candidate_id"],
            }

        return {"status": "Rejected", "error": f"Неизвестный режим: {self.mode}"}


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

            try:
                response = chat_completion(
                    user_message,
                    system_message=SYSTEM_MESSAGE,
                    history=history,
                    url=llm_url,
                    model=args.model,
                    max_tokens=args.max_tokens,
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
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_text})

            try:
                commands = extract_commands(assistant_text)
            except ValueError as exc:
                stats.parse_errors += 1
                user_message = build_parse_error_prompt(assistant_text, exc)
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
    return args.url


def build_chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/v1") or stripped.endswith("/compatible-mode/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/compatible-mode/v1/chat/completions"


def preflight_llm(chat_url: str, timeout: int) -> None:
    models_url = build_models_url(chat_url)
    request = urllib.request.Request(models_url, method="GET")
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
) -> str:
    file_previews = collect_file_previews(
        workspace,
        per_file_limit=per_file_limit,
        total_limit=total_limit,
    )
    return (
        "Ты решаешь ML benchmark task. Работай только через доступные агентские команды. "
        "В одном ответе можно вернуть одну или несколько команд, но каждая команда должна быть "
        "отдельной строкой или отдельным fenced-блоком. Вызови submit(file) ровно один раз, "
        "когда файл submission готов.\n\n"
        f"task_id: {task_id}\n"
        "Ниже частичное содержимое файлов workspace. Секретные answer/submission файлы не включены.\n\n"
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

        preview = content[:per_file_limit]
        if len(content) > per_file_limit:
            preview += "\n...[truncated]"
        chunk = f"### {relative}\n```text\n{preview}\n```\n"
        if used_chars + len(chunk) > total_limit:
            chunks.append("...[initial context truncated]")
            break
        chunks.append(chunk)
        used_chars += len(chunk)
    return "\n".join(chunks) if chunks else "(no readable text files found)"


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


def build_parse_error_prompt(assistant_text: str, exc: Exception) -> str:
    return (
        "Твой прошлый ответ не удалось распарсить как агентскую команду.\n"
        f"Ошибка парсинга: {exc}\n\n"
        "Верни только валидные команды, по одной на строку. "
        "Не объясняй ход мыслей вне команд.\n\n"
        f"Прошлый ответ:\n{assistant_text[:4000]}"
    )


def build_followup_prompt(
    command_results: list[dict[str, Any]],
    stats: BenchmarkStats,
    args: argparse.Namespace,
    feedback: dict[str, Any] | None = None,
) -> str:
    payload = {
        "command_results": command_results,
        "feedback": feedback,
        "benchmark_status": {
            "elapsed_seconds": stats.elapsed_seconds,
            "requests": stats.requests,
            "total_tokens": stats.total_tokens,
            "token_limit": args.token_limit,
            "used_steps": stats.executed_commands,
            "max_steps": args.max_steps,
            "remaining_seconds": max(0.0, args.time_limit_seconds - stats.elapsed_seconds),
        },
    }
    return (
        "Результаты выполнения твоих команд и подсказки hidden checklist ниже. "
        "Продолжай решать задачу только агентскими командами. "
        "Когда submission.csv готов, вызови submit(\"submission.csv\").\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


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
