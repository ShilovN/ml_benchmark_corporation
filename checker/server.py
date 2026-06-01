#!/usr/bin/env python3
"""HTTP server for checking submitted solution files against configured tasks."""

from __future__ import annotations

import argparse
import cgi
import csv
import html
import json
import mimetypes
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
        public_files: list[str] | None = None,
    ) -> None:
        self.task_id = task_id
        self.name = name
        self.metric = normalize_metric_name(metric)
        self.answer_file = answer_file
        self.column = column
        self.true_column = true_column
        self.pred_column = pred_column
        self.id_column = id_column
        self.public_files = public_files or []

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
            public_files=list(data.get("public_files") or []),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "name": self.name,
            "metric": self.metric,
            "column": self.column,
            "id_column": self.id_column,
            "pred_column": self.pred_column,
            "public_files": self.public_files,
        }

    def public_file_path(self, filename: str) -> Path:
        if filename not in self.public_files:
            raise RequestError(f"File is not public for task '{self.task_id}': {filename}", HTTPStatus.NOT_FOUND)

        candidate = (self.answer_file.parent / filename).resolve()
        task_dir = self.answer_file.parent.resolve()
        if task_dir not in candidate.parents and candidate != task_dir:
            raise RequestError("Invalid file path", HTTPStatus.BAD_REQUEST)
        if not candidate.exists() or not candidate.is_file():
            raise RequestError(f"Public file not found: {filename}", HTTPStatus.NOT_FOUND)
        return candidate


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

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = render_index_page(tasks).encode("utf-8")
                self.send_response(HTTPStatus.OK.value)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return
            if parsed.path == "/health":
                body = json.dumps({"status": "ok", "tasks_count": len(tasks)}).encode("utf-8")
                self.send_response(HTTPStatus.OK.value)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return
            self.send_response(HTTPStatus.NOT_FOUND.value)
            self.end_headers()

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

            if parsed.path.startswith("/tasks/") and "/files/" in parsed.path:
                self._handle_public_file(parsed.path)
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

        def _handle_public_file(self, path: str) -> None:
            parts = path.strip("/").split("/")
            if len(parts) != 4 or parts[0] != "tasks" or parts[2] != "files":
                self._send_json({"status": "error", "error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return

            task_id = parts[1]
            filename = parts[3]
            if task_id not in tasks:
                self._send_json({"status": "error", "error": f"Unknown task_id: {task_id}"}, status=HTTPStatus.NOT_FOUND)
                return

            try:
                file_path = tasks[task_id].public_file_path(filename)
            except RequestError as exc:
                self._send_json({"status": "error", "error": str(exc)}, status=exc.status)
                return

            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_path.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{file_path.name}"')
            self.end_headers()
            with file_path.open("rb") as file:
                try:
                    shutil.copyfileobj(file, self.wfile)
                except BrokenPipeError:
                    pass

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
    task_list = list(tasks.values())
    options = "\n".join(
        f'<option value="{html.escape(task.task_id)}">{html.escape(task.name)} ({html.escape(task.metric)})</option>'
        for task in task_list
    )
    tasks_json = json.dumps([task.public_dict() for task in task_list], ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ML Benchmark Checker</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --surface: #ffffff;
      --surface-2: #eef3f8;
      --text: #17202c;
      --muted: #5d6b7a;
      --border: #d7dee8;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --success: #067647;
      --focus: #2563eb;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 22px;
    }}
    h1 {{
      font-size: 30px;
      line-height: 1.15;
      margin: 0 0 8px;
    }}
    h2 {{
      font-size: 18px;
      margin: 0 0 14px;
    }}
    p {{
      margin: 0;
      color: var(--muted);
    }}
    .status {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 32px;
      padding: 0 12px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--surface);
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }}
    .status-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}
    section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    .task-list {{
      display: grid;
      gap: 10px;
    }}
    .task-button {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      padding: 12px;
      text-align: left;
      cursor: pointer;
      min-height: 74px;
    }}
    .task-button:hover,
    .task-button[aria-selected="true"] {{
      border-color: var(--accent);
      background: #ecfdf5;
    }}
    .task-name {{
      display: block;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .task-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      background: var(--surface-2);
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .field {{
      margin-bottom: 14px;
    }}
    label {{
      display: block;
      font-weight: 600;
      margin: 0 0 7px;
    }}
    select, input, button {{
      box-sizing: border-box;
      width: 100%;
      min-height: 40px;
      font: inherit;
    }}
    select, input[type="file"] {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface);
      padding: 8px;
    }}
    select:focus, input:focus, button:focus-visible, a:focus-visible {{
      outline: 3px solid color-mix(in srgb, var(--focus) 26%, transparent);
      outline-offset: 2px;
    }}
    .drop-zone {{
      border: 1px dashed #9aa8b8;
      border-radius: 8px;
      padding: 18px;
      background: #f8fafc;
    }}
    .drop-zone.dragging {{
      border-color: var(--accent);
      background: #ecfdf5;
    }}
    .button-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .primary-button, .secondary-button {{
      width: auto;
      min-width: 140px;
      padding: 0 16px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }}
    .primary-button:hover {{
      background: var(--accent-dark);
    }}
    .primary-button:disabled {{
      opacity: 0.58;
      cursor: not-allowed;
    }}
    .secondary-button {{
      background: var(--surface-2);
      color: var(--text);
      border: 1px solid var(--border);
    }}
    .file-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .file-links a {{
      display: inline-flex;
      align-items: center;
      min-height: 32px;
      padding: 0 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface);
      color: var(--accent-dark);
      font-weight: 600;
      text-decoration: none;
    }}
    .metric-panel {{
      display: grid;
      grid-template-columns: minmax(120px, 160px) minmax(0, 1fr);
      gap: 14px;
      align-items: center;
      min-height: 96px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #f8fafc;
      padding: 14px;
    }}
    .metric-value {{
      font-size: 34px;
      line-height: 1;
      font-weight: 800;
      color: var(--accent-dark);
      word-break: break-word;
    }}
    .result-list {{
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    .result-list strong {{
      color: var(--text);
    }}
    .message {{
      border-radius: 8px;
      padding: 12px;
      margin-top: 12px;
      background: var(--surface-2);
      color: var(--muted);
    }}
    .message.error {{
      background: #fef3f2;
      color: var(--danger);
      border: 1px solid #fecdca;
    }}
    .message.success {{
      background: #ecfdf3;
      color: var(--success);
      border: 1px solid #abefc6;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: #f8fafc;
    }}
    .empty {{
      color: var(--muted);
      padding: 18px 0 4px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #111827;
      color: #f9fafb;
      border-radius: 6px;
      padding: 14px;
      min-height: 48px;
      max-height: 260px;
      overflow: auto;
    }}
    .sr-only {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}
    @media (max-width: 820px) {{
      header, .layout {{
        display: block;
      }}
      .status {{
        margin-top: 14px;
      }}
      .grid, .metric-panel {{
        grid-template-columns: 1fr;
      }}
      .primary-button, .secondary-button {{
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>ML Benchmark Checker</h1>
        <p>Upload submissions, inspect task files, and review recent scores.</p>
      </div>
      <div class="status" aria-live="polite"><span class="status-dot"></span><span id="server-status">Server online</span></div>
    </header>
    <div class="layout">
      <aside>
        <section aria-labelledby="tasks-title">
          <h2 id="tasks-title">Tasks</h2>
          <div id="task-list" class="task-list" role="listbox" aria-label="Available tasks"></div>
        </section>
      </aside>
      <div>
        <section aria-labelledby="details-title">
          <h2 id="details-title">Task Details</h2>
          <div class="grid">
            <div>
              <div class="field">
                <label for="task_id">Selected task</label>
                <select id="task_id" name="task_id" form="check-form" required>
                  {options}
                </select>
              </div>
              <div class="result-list" id="task-summary"></div>
            </div>
            <div>
              <label>Public files</label>
              <div class="file-links" id="file-links"></div>
            </div>
          </div>
        </section>
        <section aria-labelledby="submit-title">
          <h2 id="submit-title">Submit Solution</h2>
          <form id="check-form">
            <div class="drop-zone" id="drop-zone">
              <label for="file">Submission file</label>
              <input id="file" name="file" type="file" accept=".csv,.txt,.tsv,.json" required>
              <p id="file-help">Expected format depends on the selected task. For salary prediction: id,salary.</p>
            </div>
            <div class="button-row" style="margin-top: 14px;">
              <button class="primary-button" id="submit-button" type="submit">Check Submission</button>
              <button class="secondary-button" id="refresh-history" type="button">Refresh History</button>
            </div>
          </form>
          <div id="form-message" class="message" role="status" aria-live="polite">Choose a task and upload a submission file.</div>
        </section>
        <section aria-labelledby="result-title">
          <h2 id="result-title">Latest Result</h2>
          <div id="result-panel" class="metric-panel">
            <div class="metric-value" id="metric-value">-</div>
            <div class="result-list" id="result-details">
              <div>No submission checked yet.</div>
            </div>
          </div>
          <details style="margin-top: 14px;">
            <summary>Raw response</summary>
            <pre id="raw-result">{{}}</pre>
          </details>
        </section>
        <section aria-labelledby="history-title">
          <h2 id="history-title">Submission History</h2>
          <div id="history"></div>
        </section>
      </div>
    </div>
  </main>
  <script>
    const tasks = {tasks_json};
    const form = document.getElementById('check-form');
    const taskSelect = document.getElementById('task_id');
    const taskList = document.getElementById('task-list');
    const taskSummary = document.getElementById('task-summary');
    const fileLinks = document.getElementById('file-links');
    const fileInput = document.getElementById('file');
    const formMessage = document.getElementById('form-message');
    const submitButton = document.getElementById('submit-button');
    const refreshHistory = document.getElementById('refresh-history');
    const metricValue = document.getElementById('metric-value');
    const resultDetails = document.getElementById('result-details');
    const rawResult = document.getElementById('raw-result');
    const history = document.getElementById('history');
    const dropZone = document.getElementById('drop-zone');

    function currentTask() {{
      return tasks.find((task) => task.id === taskSelect.value) || tasks[0];
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }}

    function renderTaskButtons() {{
      taskList.innerHTML = tasks.map((task) => `
        <button class="task-button" type="button" role="option" data-task-id="${{escapeHtml(task.id)}}" aria-selected="${{task.id === taskSelect.value}}">
          <span class="task-name">${{escapeHtml(task.name)}}</span>
          <span class="task-meta">
            <span class="badge">${{escapeHtml(task.metric)}}</span>
            ${{task.id_column ? `<span class="badge">id: ${{escapeHtml(task.id_column)}}</span>` : ''}}
            ${{task.column ? `<span class="badge">target: ${{escapeHtml(task.column)}}</span>` : ''}}
          </span>
        </button>
      `).join('');
      taskList.querySelectorAll('button').forEach((button) => {{
        button.addEventListener('click', () => {{
          taskSelect.value = button.dataset.taskId;
          renderSelectedTask();
          loadHistory();
        }});
      }});
    }}

    function renderSelectedTask() {{
      const task = currentTask();
      if (!task) return;
      taskSummary.innerHTML = `
        <div><strong>ID:</strong> ${{escapeHtml(task.id)}}</div>
        <div><strong>Metric:</strong> ${{escapeHtml(task.metric)}}</div>
        <div><strong>Target column:</strong> ${{escapeHtml(task.column || task.pred_column || 'not specified')}}</div>
        <div><strong>ID column:</strong> ${{escapeHtml(task.id_column || 'row order')}}</div>
      `;
      if (task.public_files.length) {{
        fileLinks.innerHTML = task.public_files.map((filename) =>
          `<a href="/tasks/${{encodeURIComponent(task.id)}}/files/${{encodeURIComponent(filename)}}">${{escapeHtml(filename)}}</a>`
        ).join('');
      }} else {{
        fileLinks.innerHTML = '<span class="empty">No public files for this task.</span>';
      }}
      renderTaskButtons();
    }}

    function setMessage(text, kind = '') {{
      formMessage.className = `message ${{kind}}`;
      formMessage.textContent = text;
    }}

    function renderResult(payload) {{
      rawResult.textContent = JSON.stringify(payload, null, 2);
      if (payload.status !== 'ok') {{
        metricValue.textContent = '-';
        resultDetails.innerHTML = `<div><strong>Error:</strong> ${{escapeHtml(payload.error || 'Unknown error')}}</div>`;
        setMessage(payload.error || 'Submission failed.', 'error');
        return;
      }}
      metricValue.textContent = Number(payload.value).toLocaleString(undefined, {{ maximumFractionDigits: 8 }});
      resultDetails.innerHTML = `
        <div><strong>Task:</strong> ${{escapeHtml(payload.task_id)}}</div>
        <div><strong>Metric:</strong> ${{escapeHtml(payload.metric)}}</div>
        <div><strong>Rows checked:</strong> ${{escapeHtml(payload.rows_checked)}}</div>
        <div><strong>Submission:</strong> ${{escapeHtml(payload.submission_id)}}</div>
        <div><strong>Elapsed:</strong> ${{escapeHtml(payload.elapsed_ms)}} ms</div>
      `;
      setMessage('Submission checked successfully.', 'success');
    }}

    async function loadHistory() {{
      const task = currentTask();
      history.innerHTML = '<div class="empty">Loading history...</div>';
      const response = await fetch(`/submissions?task_id=${{encodeURIComponent(task.id)}}`);
      const payload = await response.json();
      const rows = payload.submissions || [];
      if (!rows.length) {{
        history.innerHTML = '<div class="empty">No submissions for this task yet.</div>';
        return;
      }}
      history.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>File</th>
              <th>Metric</th>
              <th>Value</th>
              <th>Rows</th>
            </tr>
          </thead>
          <tbody>
            ${{rows.map((row) => `
              <tr>
                <td>${{escapeHtml(new Date(row.created_at).toLocaleString())}}</td>
                <td>${{escapeHtml(row.filename)}}</td>
                <td>${{escapeHtml(row.metric)}}</td>
                <td><strong>${{escapeHtml(Number(row.value).toLocaleString(undefined, {{ maximumFractionDigits: 8 }}))}}</strong></td>
                <td>${{escapeHtml(row.rows_checked)}}</td>
              </tr>
            `).join('')}}
          </tbody>
        </table>
      `;
    }}

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      if (!fileInput.files.length) {{
        setMessage('Choose a submission file first.', 'error');
        return;
      }}
      submitButton.disabled = true;
      setMessage('Checking submission...');
      try {{
        const response = await fetch('/check', {{
          method: 'POST',
          body: new FormData(form)
        }});
        const payload = await response.json();
        renderResult(payload);
        await loadHistory();
      }} catch (error) {{
        renderResult({{ status: 'error', error: error.message }});
      }} finally {{
        submitButton.disabled = false;
      }}
    }});

    taskSelect.addEventListener('change', () => {{
      renderSelectedTask();
      loadHistory();
    }});
    refreshHistory.addEventListener('click', loadHistory);
    ['dragenter', 'dragover'].forEach((eventName) => {{
      dropZone.addEventListener(eventName, (event) => {{
        event.preventDefault();
        dropZone.classList.add('dragging');
      }});
    }});
    ['dragleave', 'drop'].forEach((eventName) => {{
      dropZone.addEventListener(eventName, (event) => {{
        event.preventDefault();
        dropZone.classList.remove('dragging');
      }});
    }});
    dropZone.addEventListener('drop', (event) => {{
      if (event.dataTransfer.files.length) {{
        fileInput.files = event.dataTransfer.files;
        setMessage(`Selected file: ${{event.dataTransfer.files[0].name}}`);
      }}
    }});
    fileInput.addEventListener('change', () => {{
      if (fileInput.files.length) {{
        setMessage(`Selected file: ${{fileInput.files[0].name}}`);
      }}
    }});

    renderSelectedTask();
    loadHistory();
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
