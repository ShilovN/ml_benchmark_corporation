#!/usr/bin/env python3
"""HTTP server for checking submitted solution files against configured tasks."""

from __future__ import annotations

import argparse
import cgi
import csv
import json
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from metric_checker import compute_metric_details, metric_registry, normalize_metric_name


DEFAULT_TASKS_DIR = Path(__file__).resolve().parent / "tasks"
DEFAULT_SUBMISSIONS_DIR = Path(__file__).resolve().parent / "submissions"
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
        id_column: str | None = None,
    ) -> None:
        self.task_id = task_id
        self.name = name
        self.metric = normalize_metric_name(metric)
        self.answer_file = answer_file
        self.column = column
        self.true_column = true_column
        self.pred_column = pred_column
        self.id_column = id_column

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
            id_column=data.get("id_column"),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "name": self.name,
            "metric": self.metric,
            "column": self.column,
            "id_column": self.id_column,
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


def make_handler(
    tasks: dict[str, TaskConfig],
    submissions_dir: Path = DEFAULT_SUBMISSIONS_DIR,
) -> type[BaseHTTPRequestHandler]:
    class CheckerRequestHandler(BaseHTTPRequestHandler):
        server_version = "MetricCheckerHTTP/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_index_page(tasks))
                return

            if parsed.path == "/health":
                self._send_json({"status": "ok", "tasks_count": len(tasks)})
                return

            if parsed.path == "/tasks":
                self._send_json({"status": "ok", "tasks": [task.public_dict() for task in tasks.values()]})
                return

            if parsed.path == "/submissions":
                task_id = _first_query_value(parsed.query, "task_id")
                self._send_json(
                    {
                        "status": "ok",
                        "submissions": load_submission_history(submissions_dir, task_id=task_id),
                    }
                )
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
                self._send_json({"status": "error", "error": str(exc)}, status=exc.status)
            except (KeyError, ValueError, OSError, csv.Error, json.JSONDecodeError) as exc:
                self._send_json({"status": "error", "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

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
            original_filename = Path(getattr(submission, "filename", "") or "submission.txt").name
            suffix = Path(original_filename).suffix or ".txt"
            submission_id = uuid.uuid4().hex
            task_submission_dir = submissions_dir / task.task_id
            task_submission_dir.mkdir(parents=True, exist_ok=True)
            saved_submission_path = task_submission_dir / f"{submission_id}{suffix}"

            started_at = time.perf_counter()
            with saved_submission_path.open("wb") as output:
                shutil.copyfileobj(submission.file, output)

            try:
                result = compute_metric_details(
                    task.answer_file,
                    saved_submission_path,
                    task.metric,
                    column=task.column,
                    true_column=task.true_column,
                    pred_column=task.pred_column,
                    id_column=task.id_column,
                )
            except Exception:
                saved_submission_path.unlink(missing_ok=True)
                raise

            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            record = {
                "submission_id": submission_id,
                "task_id": task.task_id,
                "filename": original_filename,
                "metric": result["metric"],
                "value": result["value"],
                "rows_checked": result["rows_checked"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": elapsed_ms,
            }
            append_submission_history(submissions_dir, record)
            self._send_json(
                {
                    "status": "ok",
                    "submission_id": submission_id,
                    "task_id": task.task_id,
                    "metric": result["metric"],
                    "value": result["value"],
                    "rows_checked": result["rows_checked"],
                    "elapsed_ms": elapsed_ms,
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

        def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = html.encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return CheckerRequestHandler


def append_submission_history(submissions_dir: Path, record: dict[str, Any]) -> None:
    submissions_dir.mkdir(parents=True, exist_ok=True)
    history_path = submissions_dir / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_submission_history(
    submissions_dir: Path,
    task_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    history_path = submissions_dir / "history.jsonl"
    if not history_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if task_id and record.get("task_id") != task_id:
                continue
            records.append(record)

    return list(reversed(records[-limit:]))


def render_index_page(tasks: dict[str, TaskConfig]) -> str:
    options = "\n".join(
        f'<option value="{task.task_id}">{task.name} ({task.metric})</option>'
        for task in tasks.values()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Solution Checker</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1d2430;
    }}
    main {{
      max-width: 760px;
      margin: 48px auto;
      padding: 0 20px;
    }}
    h1 {{
      font-size: 32px;
      margin: 0 0 20px;
    }}
    form, section {{
      background: #ffffff;
      border: 1px solid #dfe3e8;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 16px;
    }}
    label {{
      display: block;
      font-weight: 600;
      margin: 0 0 8px;
    }}
    select, input, button {{
      box-sizing: border-box;
      width: 100%;
      min-height: 40px;
      margin-bottom: 16px;
      font: inherit;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      background: #1864ab;
      color: white;
      font-weight: 700;
      cursor: pointer;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #111827;
      color: #f9fafb;
      border-radius: 6px;
      padding: 14px;
      min-height: 48px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Solution Checker</h1>
    <form id="check-form">
      <label for="task_id">Task</label>
      <select id="task_id" name="task_id" required>
        {options}
      </select>
      <label for="file">Submission file</label>
      <input id="file" name="file" type="file" required>
      <button type="submit">Check solution</button>
    </form>
    <section>
      <label>Result</label>
      <pre id="result">Waiting for submission...</pre>
    </section>
  </main>
  <script>
    const form = document.getElementById('check-form');
    const result = document.getElementById('result');
    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      result.textContent = 'Checking...';
      const response = await fetch('/check', {{
        method: 'POST',
        body: new FormData(form)
      }});
      const payload = await response.json();
      result.textContent = JSON.stringify(payload, null, 2);
    }});
  </script>
</body>
</html>"""


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
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=DEFAULT_SUBMISSIONS_DIR,
        help="Directory for uploaded submissions and history",
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

    handler = make_handler(tasks, args.submissions_dir)
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
