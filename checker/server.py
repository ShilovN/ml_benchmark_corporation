#!/usr/bin/env python3
"""HTTP server for checking submitted solution files against configured tasks."""

from __future__ import annotations

import argparse
import cgi
import csv
import json
import shutil
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from metric_checker import compute_metric, metric_registry, normalize_metric_name


DEFAULT_TASKS_DIR = Path(__file__).resolve().parent / "tasks"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class TaskConfig:
    def __init__(
        self,
        task_id: str,
        name: str,
        metric: str,
        answer_file: Path,
        column: str | None = None,
        true_column: str | None = None,
        pred_column: str | None = None,
    ) -> None:
        self.task_id = task_id
        self.name = name
        self.metric = normalize_metric_name(metric)
        self.answer_file = answer_file
        self.column = column
        self.true_column = true_column
        self.pred_column = pred_column

    @classmethod
    def from_json(cls, path: Path) -> "TaskConfig":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        task_id = str(data.get("id") or path.parent.name)
        metric = str(data["metric"])
        answer_file = path.parent / str(data["answer_file"])

        if metric not in metric_registry():
            raise ValueError(f"Unsupported metric '{metric}' in {path}")
        if not answer_file.exists():
            raise ValueError(f"Answer file does not exist for task '{task_id}': {answer_file}")

        return cls(
            task_id=task_id,
            name=str(data.get("name") or task_id),
            metric=metric,
            answer_file=answer_file,
            column=data.get("column"),
            true_column=data.get("true_column"),
            pred_column=data.get("pred_column"),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "name": self.name,
            "metric": self.metric,
            "column": self.column,
            "pred_column": self.pred_column,
        }


def load_tasks(tasks_dir: Path) -> dict[str, TaskConfig]:
    tasks: dict[str, TaskConfig] = {}
    if not tasks_dir.exists():
        return tasks

    for config_path in sorted(tasks_dir.glob("*/task.json")):
        task = TaskConfig.from_json(config_path)
        if task.task_id in tasks:
            raise ValueError(f"Duplicate task id: {task.task_id}")
        tasks[task.task_id] = task
    return tasks


def make_handler(tasks: dict[str, TaskConfig]) -> type[BaseHTTPRequestHandler]:
    class CheckerRequestHandler(BaseHTTPRequestHandler):
        server_version = "MetricCheckerHTTP/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json({"status": "ok"})
                return

            if parsed.path == "/tasks":
                self._send_json({"tasks": [task.public_dict() for task in tasks.values()]})
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/check":
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return

            try:
                self._handle_check(parsed.query)
            except RequestError as exc:
                self._send_json({"error": str(exc)}, status=exc.status)
            except (KeyError, ValueError, OSError, csv.Error, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def _handle_check(self, query: str) -> None:
            form = self._read_multipart_form()
            task_id = _first_value(form, "task_id") or _first_query_value(query, "task_id")
            if not task_id:
                raise RequestError("Missing task_id", HTTPStatus.BAD_REQUEST)
            if task_id not in tasks:
                raise RequestError(f"Unknown task_id: {task_id}", HTTPStatus.NOT_FOUND)

            submission = form["file"] if "file" in form else None
            if submission is None or not getattr(submission, "file", None):
                raise RequestError("Missing uploaded file field named 'file'", HTTPStatus.BAD_REQUEST)

            task = tasks[task_id]
            suffix = Path(getattr(submission, "filename", "") or "submission.txt").suffix
            with tempfile.TemporaryDirectory(prefix="checker_submission_") as tmp_dir:
                submission_path = Path(tmp_dir) / f"submission{suffix}"
                with submission_path.open("wb") as output:
                    shutil.copyfileobj(submission.file, output)

                score = compute_metric(
                    task.answer_file,
                    submission_path,
                    task.metric,
                    column=task.column,
                    true_column=task.true_column,
                    pred_column=task.pred_column,
                )

            self._send_json(
                {
                    "task_id": task.task_id,
                    "metric": task.metric,
                    "value": score,
                }
            )

        def _read_multipart_form(self) -> cgi.FieldStorage:
            content_length = int(self.headers.get("Content-Length") or "0")
            if content_length <= 0:
                raise RequestError("Empty request body", HTTPStatus.BAD_REQUEST)
            if content_length > MAX_UPLOAD_BYTES:
                raise RequestError("Uploaded file is too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise RequestError("Content-Type must be multipart/form-data", HTTPStatus.BAD_REQUEST)

            return cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": str(content_length),
                },
            )

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return CheckerRequestHandler


class RequestError(Exception):
    def __init__(self, message: str, status: HTTPStatus) -> None:
        super().__init__(message)
        self.status = status


def _first_value(form: cgi.FieldStorage, name: str) -> str | None:
    if name not in form:
        return None

    item = form[name]
    if isinstance(item, list):
        item = item[0]

    value = getattr(item, "value", None)
    return str(value) if value is not None else None


def _first_query_value(query: str, name: str) -> str | None:
    values = parse_qs(query).get(name)
    return values[0] if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the solution checking server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="Directory with task configs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        tasks = load_tasks(args.tasks_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load tasks: {exc}")
        return 1

    if not tasks:
        print(f"No tasks found in {args.tasks_dir}")
        return 1

    handler = make_handler(tasks)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Checker server is running on http://{args.host}:{args.port}")
    print(f"Loaded tasks: {', '.join(sorted(tasks))}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping checker server")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
