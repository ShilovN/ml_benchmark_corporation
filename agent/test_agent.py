import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.executor import AgentContext, CommandExecutor
from agent.hints import CsvTable, HintEngine, _contains_metric_value, _perfect_numeric_correlations
from agent.llm_client import ask_llm, build_messages, chat_completion
from agent.parser import extract_command_text, parse_command, parse_model_response
from main import (
    BenchmarkStats,
    build_followup_prompt,
    compact_history_for_model,
    compact_json_for_prompt,
    response_max_tokens,
)
from web_server import (
    AgentRunManager,
    REPEATED_MAX_ATTEMPTS,
    RunConfig,
    RunState,
    apply_mode_after_results,
    apply_mode_instruction,
    best_repeated_submission,
    current_fixed_stage,
    detect_fixed_stage_violation,
    extract_commands,
    extract_reported_stage,
    handle_successful_submit,
    list_workspace_files,
    prepare_run_workspace,
    record_fixed_stage_attempt,
    repeated_attempt_workspace_id,
    should_isolate_llm_history,
    validate_reported_stage,
    validate_mode_command_batch,
    list_workspace_files,
    resolve_workspace_file,
    should_isolate_llm_history,
    validate_mode_command,
    validate_mode_command_batch,
    validate_reported_stage,
    workspace_file_preview,
    zero_fixed_transition_result_if_needed,
)


class ParserTest(unittest.TestCase):
    def test_parse_call_command(self) -> None:
        command = parse_command('read_file("notes.txt")')

        self.assertEqual(command.name, "read_file")
        self.assertEqual(command.args, {"path": "notes.txt"})

    def test_parse_json_command(self) -> None:
        command = parse_command('{"command": "show_sample_rows", "args": {"n": 2}}')

        self.assertEqual(command.name, "show_sample_rows")
        self.assertEqual(command.args, {"n": 2})

    def test_parse_get_hints(self) -> None:
        command = parse_command("get_hints()")

        self.assertEqual(command.name, "get_hints")
        self.assertEqual(command.args, {})

    def test_extract_command_from_fenced_block(self) -> None:
        text = """Сначала посмотрю файлы.

```command
list_files("checker")
```
"""

        self.assertEqual(extract_command_text(text), 'list_files("checker")')

    def test_extract_command_from_plain_text_line(self) -> None:
        text = """Нужно загрузить датасет.
load_dataset("train.csv")
Потом посмотрю строки."""

        command = parse_model_response(text)

        self.assertEqual(command.name, "load_dataset")
        self.assertEqual(command.args, {"path": "train.csv"})

    def test_extract_inline_command(self) -> None:
        text = 'Дальше выполню read_file("checker/README.md") и изучу описание.'

        self.assertEqual(extract_command_text(text), 'read_file("checker/README.md")')

    def test_extract_command_after_stage_and_short_comment(self) -> None:
        text = """EDA
Посмотрю, какие файлы есть в проекте.
list_files(".")"""

        command = parse_model_response(text)

        self.assertEqual(command.name, "list_files")
        self.assertEqual(command.args, {"path": "."})

    def test_multiple_commands_raise_error(self) -> None:
        text = """read_file("a.txt")
read_file("b.txt")"""

        with self.assertRaisesRegex(ValueError, "Multiple commands"):
            extract_command_text(text)

    def test_no_command_raises_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "No executable command"):
            extract_command_text("Я пока думаю, какую команду вызвать.")

    def test_rejects_mixed_positional_and_keyword_args(self) -> None:
        with self.assertRaisesRegex(ValueError, "Do not mix"):
            parse_command('read_file("a.txt", path="b.txt")')

    def test_rejects_args_for_no_arg_command(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not accept arguments"):
            parse_command('get_budget_status("extra")')

    def test_extract_ignores_invalid_fenced_block_and_uses_valid_one(self) -> None:
        text = """```command
not a command
```

```command
get_remaining_time()
```"""

        self.assertEqual(extract_command_text(text), "get_remaining_time()")


class ExecutorTest(unittest.TestCase):
    def test_file_commands_and_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            executor = CommandExecutor(AgentContext(workspace=workspace))

            write_result = executor.execute_text('write_file("a.txt", "hello")')
            read_result = executor.execute_text('read_file("a.txt")')
            trajectory_result = executor.execute_text("get_trajectory()")
            history_path = workspace / "agent_history.txt"
            history_lines = history_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(write_result["status"], "ok")
        self.assertNotIn("feedback", write_result)
        self.assertEqual(read_result["result"]["content"], "hello")
        self.assertEqual(len(trajectory_result["result"]), 2)
        self.assertEqual(len(history_lines), 3)
        self.assertEqual(json.loads(history_lines[0])["command"], "write_file")

    def test_followup_prompt_contains_one_batch_feedback(self) -> None:
        args = argparse.Namespace(token_limit=None, max_steps=100, time_limit_seconds=3600)
        command_results = [
            {"status": "ok", "command": "write_file", "result": {"path": "a.txt"}},
            {"status": "ok", "command": "read_file", "result": {"content": "hello"}},
        ]
        feedback = {"hints": [{"stage": "EDA", "message": "hint"}]}
        stats = BenchmarkStats()
        stats.requests = 2

        prompt = build_followup_prompt(command_results, stats, args, feedback)

        self.assertIn("Результаты команд:", prompt)
        self.assertIn("1. write_file: ok", prompt)
        self.assertIn("2. read_file: ok", prompt)
        self.assertIn("Подсказки:", prompt)
        self.assertIn("- EDA: hint", prompt)
        self.assertNotIn('"command_results"', prompt)
        self.assertNotIn('"feedback"', prompt)

    def test_followup_prompt_skips_feedback_in_first_message(self) -> None:
        args = argparse.Namespace(token_limit=None, max_steps=100, time_limit_seconds=3600)
        command_results = [{"status": "ok", "command": "read_file", "result": {"content": "hello"}}]
        feedback = {"hints": [{"stage": "EDA", "message": "hint"}]}
        stats = BenchmarkStats()
        stats.requests = 1

        prompt = build_followup_prompt(command_results, stats, args, feedback)

        self.assertIn("1. read_file: ok", prompt)
        self.assertNotIn("Подсказки:", prompt)
        self.assertNotIn("- EDA: hint", prompt)

    def test_followup_prompt_contains_remaining_budget(self) -> None:
        args = argparse.Namespace(token_limit=1000, max_steps=10, time_limit_seconds=3600)
        stats = BenchmarkStats()
        stats.total_tokens = 250
        stats.executed_commands = 4

        prompt = build_followup_prompt([], stats, args)

        self.assertIn("осталось 75.0% токенов", prompt)
        self.assertIn("осталось итераций: 6", prompt)
        self.assertIn("Статус benchmark:", prompt)
        self.assertIn("tokens=250/1000", prompt)
        self.assertIn("steps=4/10", prompt)
        self.assertIn("remaining_steps=6", prompt)
        self.assertIn("- команд не было", prompt)
        self.assertNotIn('"benchmark_status"', prompt)

    def test_followup_prompt_wraps_python_output_in_code_blocks(self) -> None:
        args = argparse.Namespace(token_limit=None, max_steps=10, time_limit_seconds=3600)
        command_results = [
            {
                "status": "ok",
                "command": "run_python",
                "result": {
                    "returncode": 1,
                    "stdout": "printed value\nsecond line",
                    "stderr": "Traceback line",
                },
            }
        ]

        prompt = build_followup_prompt(command_results, BenchmarkStats(), args)

        self.assertIn("1. run_python: ok", prompt)
        self.assertIn("   ```text\n   returncode=1\n   stdout:\n     printed value\n     second line\n   stderr:\n     Traceback line\n   ```", prompt)

    def test_followup_prompt_wraps_all_command_results_in_code_blocks(self) -> None:
        args = argparse.Namespace(token_limit=None, max_steps=20, time_limit_seconds=3600)
        command_results = [
            {"status": "ok", "command": "list_files", "result": ["a.py", "data/"]},
            {"status": "ok", "command": "read_file", "result": {"path": "a.py", "content": "print(1)", "truncated": False}},
            {"status": "ok", "command": "write_file", "result": {"path": "a.py", "bytes_written": 8}},
            {"status": "ok", "command": "edit_file", "result": {"path": "a.py", "replacements": 1}},
            {"status": "ok", "command": "load_dataset", "result": {"path": "train.csv", "rows": 2, "columns": ["id", "target"]}},
            {"status": "ok", "command": "show_dataset_info", "result": {"path": "train.csv", "rows": 2, "columns": ["id"], "missing_by_column": {"id": 0}}},
            {"status": "ok", "command": "show_sample_rows", "result": [{"id": "1"}]},
            {"status": "ok", "command": "run_python", "result": {"returncode": 0, "stdout": "ok", "stderr": ""}},
            {"status": "ok", "command": "get_budget_status", "result": {"remaining_steps": 5}},
            {"status": "ok", "command": "get_remaining_time", "result": {"remaining_seconds": 10.0}},
            {"status": "ok", "command": "get_hints", "result": {"hints": []}},
            {"status": "ok", "command": "get_trajectory", "result": [{"command": "read_file"}]},
            {"status": "ok", "command": "submit", "result": {"metric": "mae", "value": 0.0}},
        ]

        prompt = build_followup_prompt(command_results, BenchmarkStats(), args)

        for index, result in enumerate(command_results, start=1):
            self.assertIn(f"{index}. {result['command']}: ok\n   ```text", prompt)
        self.assertGreaterEqual(prompt.count("   ```"), len(command_results) * 2)

    def test_followup_prompt_wraps_command_errors_in_code_blocks(self) -> None:
        args = argparse.Namespace(token_limit=None, max_steps=10, time_limit_seconds=3600)
        command_results = [
            {
                "status": "error",
                "command": "read_file",
                "error": "File does not exist:\nmissing.csv",
            }
        ]

        prompt = build_followup_prompt(command_results, BenchmarkStats(), args)

        self.assertIn("1. read_file: error", prompt)
        self.assertIn("   ```text\n   File does not exist:\n   missing.csv\n   ```", prompt)

    def test_followup_prompt_warns_when_token_budget_is_low(self) -> None:
        args = argparse.Namespace(token_limit=1000, max_steps=10, time_limit_seconds=3600)
        stats = BenchmarkStats()
        stats.total_tokens = 701

        prompt = build_followup_prompt([], stats, args)

        self.assertIn("осталось 29.9% токенов", prompt)
        self.assertIn("Экономь токены", prompt)
        self.assertIn("делать submit", prompt)
        self.assertIn("иначе проиграешь", prompt)

    def test_followup_prompt_uses_soft_warning_at_exactly_30_percent(self) -> None:
        args = argparse.Namespace(token_limit=1000, max_steps=10, time_limit_seconds=3600)
        stats = BenchmarkStats()
        stats.total_tokens = 700

        prompt = build_followup_prompt([], stats, args)

        self.assertIn("осталось 30.0% токенов", prompt)
        self.assertIn("прекращай исследование", prompt)
        self.assertNotIn("иначе проиграешь", prompt)

    def test_followup_prompt_pushes_submit_when_token_budget_is_critical(self) -> None:
        args = argparse.Namespace(token_limit=1000, max_steps=10, time_limit_seconds=3600)
        stats = BenchmarkStats()
        stats.total_tokens = 851

        prompt = build_followup_prompt([], stats, args)

        self.assertIn("осталось 14.9% токенов", prompt)
        self.assertIn("КРИТИЧНО", prompt)
        self.assertIn("вызывай submit", prompt)

    def test_response_max_tokens_shrinks_when_budget_is_low(self) -> None:
        args = argparse.Namespace(token_limit=1000, max_steps=10, max_tokens=4096)
        stats = BenchmarkStats()

        stats.total_tokens = 200
        self.assertEqual(response_max_tokens(stats, args), 4096)
        stats.total_tokens = 501
        self.assertEqual(response_max_tokens(stats, args), 1024)
        stats.total_tokens = 701
        self.assertEqual(response_max_tokens(stats, args), 768)
        stats.total_tokens = 851
        self.assertEqual(response_max_tokens(stats, args), 512)

    def test_compact_history_limits_old_messages_and_long_content(self) -> None:
        history = [
            {"role": "user", "content": f"message-{index}-" + ("x" * 4000)}
            for index in range(12)
        ]

        compacted = compact_history_for_model(history)

        self.assertEqual(len(compacted), 9)
        self.assertIn("history compacted", compacted[0]["content"])
        self.assertTrue(all(len(message["content"]) <= 2600 for message in compacted[1:]))
        self.assertIn("message-4", compacted[1]["content"])
        self.assertIn("[truncated", compacted[1]["content"])

    def test_compact_json_truncates_long_strings_from_middle_only_when_needed(self) -> None:
        payload = {"result": {"content": "a" * 3000 + "middle" + "z" * 3000}}

        full = compact_json_for_prompt(payload, 10000)
        compact = compact_json_for_prompt(payload, 1800)

        self.assertIn("middle", full)
        self.assertLess(len(compact), len(full))
        self.assertIn("[truncated", compact)
        self.assertIn("aaa", compact)
        self.assertIn("zzz", compact)

    def test_get_hints_returns_abstract_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            task_dir = workspace / "checker" / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "toy",
                        "metric": "mae",
                        "answer_file": "answers.csv",
                        "id_column": "id",
                        "column": "salary",
                        "public_files": ["train.csv", "test.csv"],
                    }
                ),
                encoding="utf-8",
            )
            (task_dir / "train.csv").write_text(
                "id,feature,salary\n1,10,100\n1,10,100\n2,,10000\n",
                encoding="utf-8",
            )
            (task_dir / "test.csv").write_text("id,feature\n3,\n", encoding="utf-8")
            executor = CommandExecutor(
                AgentContext(workspace=workspace, task_id="toy", tasks_dir=Path("checker/tasks"))
            )

            result = executor.execute_text("get_hints()")

        messages = [hint["message"] for hint in result["result"]["hints"]]
        self.assertTrue(any("повтор" in message for message in messages))
        self.assertFalse(any("пропус" in message for message in messages))
        self.assertFalse(any("распределение целевой" in message for message in messages))
        self.assertEqual({hint["stage"] for hint in result["result"]["hints"]}, {"EDA"})

    def test_get_hints_returns_only_first_incomplete_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            task_dir = workspace / "checker" / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "toy",
                        "metric": "mae",
                        "answer_file": "answers.csv",
                        "column": "salary",
                        "public_files": ["train.csv", "test.csv"],
                    }
                ),
                encoding="utf-8",
            )
            (task_dir / "train.csv").write_text(
                "id,feature,salary\n1,10,100\n2,10,120\n",
                encoding="utf-8",
            )
            (task_dir / "test.csv").write_text("id,feature,salary\n3,12,130\n", encoding="utf-8")
            engine = HintEngine(workspace=workspace, task_id="toy", tasks_dir=Path("checker/tasks"))

            feedback = engine.build_feedback()

        self.assertEqual({hint["stage"] for hint in feedback["hints"]}, {"EDA"})
        self.assertTrue(any("целевая переменная" in hint["message"] for hint in feedback["hints"]))

    def test_get_hints_falls_through_to_next_stage_when_previous_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            task_dir = workspace / "checker" / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "toy",
                        "metric": "mae",
                        "answer_file": "answers.csv",
                        "column": "salary",
                        "public_files": ["train.csv", "test.csv"],
                    }
                ),
                encoding="utf-8",
            )
            (task_dir / "train.csv").write_text(
                "id,feature_a,feature_b,salary\n1,10,10,100\n2,20,20,120\n",
                encoding="utf-8",
            )
            (task_dir / "test.csv").write_text("id,feature_a,feature_b\n3,12,12\n", encoding="utf-8")
            (workspace / "agent_history.txt").write_text(
                "\n".join(
                    [
                        json.dumps({"command": "read_file", "args": {"path": "train.csv"}, "status": "ok"}),
                        json.dumps({"command": "read_file", "args": {"path": "test.csv"}, "status": "ok"}),
                        json.dumps({"command": "show_dataset_info", "args": {}, "status": "ok"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            engine = HintEngine(workspace=workspace, task_id="toy", tasks_dir=Path("checker/tasks"))

            feedback = engine.build_feedback()

        self.assertEqual({hint["stage"] for hint in feedback["hints"]}, {"Feature engineering"})
        self.assertTrue(any("дублирующих" in hint["message"] for hint in feedback["hints"]))

    def test_get_trajectory_reads_history_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            history_path = workspace / "agent_history.txt"
            history_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-05-28T00:00:00+00:00",
                        "command": "read_file",
                        "args": {"path": "a.txt"},
                        "status": "ok",
                        "result_preview": "preview",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            executor = CommandExecutor(AgentContext(workspace=workspace))

            result = executor.execute_text("get_trajectory()")

        self.assertEqual(result["result"][0]["command"], "read_file")
        self.assertEqual(result["result"][0]["result_preview"], "preview")

    def test_execute_model_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "notes.txt").write_text("hello", encoding="utf-8")
            executor = CommandExecutor(AgentContext(workspace=workspace))

            result = executor.execute_model_response('Посмотрю файл: read_file("notes.txt")')

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["result"]["content"], "hello")

    def test_rejects_path_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = CommandExecutor(AgentContext(workspace=Path(tmp_dir)))

            result = executor.execute_text('read_file("../secret.txt")')

        self.assertEqual(result["status"], "error")
        self.assertIn("outside workspace", result["error"])

    def test_dataset_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "data.csv").write_text("id,value\n1,10\n2,\n", encoding="utf-8")
            executor = CommandExecutor(AgentContext(workspace=workspace))

            load_result = executor.execute_text('load_dataset("data.csv")')
            info_result = executor.execute_text("show_dataset_info()")
            sample_result = executor.execute_text("show_sample_rows(1)")

        self.assertEqual(load_result["result"]["rows"], 2)
        self.assertEqual(info_result["result"]["missing_by_column"]["value"], 1)
        self.assertEqual(sample_result["result"], [{"id": "1", "value": "10"}])

    def test_submit_salary_task(self) -> None:
        project_root = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            tasks_dir = workspace / "checker" / "tasks" / "salary_prediction"
            tasks_dir.mkdir(parents=True)
            (tasks_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "salary_prediction",
                        "metric": "mae",
                        "answer_file": "answers.csv",
                        "column": "salary",
                    }
                ),
                encoding="utf-8",
            )
            (tasks_dir / "answers.csv").write_text(
                (project_root / "checker" / "tasks" / "salary_prediction" / "answers.csv").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            executor = CommandExecutor(AgentContext(workspace=workspace, task_id="salary_prediction"))

            result = executor.execute_text('submit("checker/tasks/salary_prediction/answers.csv")')

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["result"]["metric"], "mae")
        self.assertEqual(result["result"]["value"], 0.0)

    def test_submit_includes_submission_source_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            tasks_dir = workspace / "checker" / "tasks" / "toy"
            tasks_dir.mkdir(parents=True)
            (tasks_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "toy",
                        "metric": "mae",
                        "answer_file": "answers.csv",
                        "id_column": "id",
                        "column": "target",
                    }
                ),
                encoding="utf-8",
            )
            (tasks_dir / "answers.csv").write_text("id,target\n1,10\n2,20\n", encoding="utf-8")
            executor = CommandExecutor(
                AgentContext(workspace=workspace, task_id="toy", tasks_dir=workspace / "checker" / "tasks")
            )
            code = (
                "import csv\n"
                "with open('submission.csv', 'w', newline='') as file:\n"
                "    writer = csv.writer(file)\n"
                "    writer.writerow(['id', 'target'])\n"
                "    writer.writerow([1, 10])\n"
                "    writer.writerow([2, 20])\n"
            )

            write_result = executor.execute_text(f'write_file("solution.py", {code!r})')
            run_result = executor.execute_text('run_python("solution.py")')
            submit_result = executor.execute_text('submit("submission.csv")')

        source = submit_result["result"]["submission_source"]
        self.assertEqual(write_result["status"], "ok")
        self.assertEqual(run_result["status"], "ok")
        self.assertEqual(submit_result["status"], "ok")
        self.assertEqual(source["path"], "solution.py")
        self.assertEqual(source["kind"], "python_file")
        self.assertIn("csv.writer", source["content"])
        self.assertIn("submission.csv", source["content"])

    def test_submit_source_requires_submission_file_and_csv_writer_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            tasks_dir = workspace / "checker" / "tasks" / "toy"
            tasks_dir.mkdir(parents=True)
            (tasks_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "toy",
                        "metric": "mae",
                        "answer_file": "answers.csv",
                        "id_column": "id",
                        "column": "target",
                    }
                ),
                encoding="utf-8",
            )
            (tasks_dir / "answers.csv").write_text("id,target\n1,10\n2,20\n", encoding="utf-8")
            executor = CommandExecutor(
                AgentContext(workspace=workspace, task_id="toy", tasks_dir=workspace / "checker" / "tasks")
            )
            unrelated_code = "print('submission.csv exists elsewhere')\n"

            executor.execute_text(f'write_file("notes.py", {unrelated_code!r})')
            executor.execute_text('run_python("notes.py")')
            (workspace / "submission.csv").write_text("id,target\n1,10\n2,20\n", encoding="utf-8")
            submit_result = executor.execute_text('submit("submission.csv")')

        self.assertEqual(submit_result["status"], "ok")
        self.assertNotIn("submission_source", submit_result["result"])

    def test_edit_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            path = workspace / "a.txt"
            path.write_text("hello old", encoding="utf-8")
            diff = json.dumps({"old": "old", "new": "new"})
            executor = CommandExecutor(AgentContext(workspace=workspace))

            result = executor.execute_text(f'edit_file("a.txt", {diff!r})')
            content = path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(content, "hello new")

    def test_run_python_inline_code_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "script.py").write_text('print("from file")', encoding="utf-8")
            executor = CommandExecutor(AgentContext(workspace=workspace))

            inline_result = executor.execute_text('run_python("print(2 + 3)")')
            file_result = executor.execute_text('run_python("script.py")')

        self.assertEqual(inline_result["result"]["returncode"], 0)
        self.assertIn("5", inline_result["result"]["stdout"])
        self.assertEqual(file_result["result"]["returncode"], 0)
        self.assertIn("from file", file_result["result"]["stdout"])

    @patch("agent.executor.subprocess.run")
    def test_run_python_uses_configured_code_timeout(self, run_mock: Mock) -> None:
        run_mock.return_value = Mock(returncode=0, stdout="ok", stderr="")
        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = CommandExecutor(
                AgentContext(workspace=Path(tmp_dir), time_limit_seconds=123)
            )

            result = executor.execute_text('run_python("print(1)")')

        self.assertEqual(result["status"], "ok")
        timeout = run_mock.call_args.kwargs["timeout"]
        self.assertEqual(timeout, 123)

    def test_step_budget_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = CommandExecutor(AgentContext(workspace=Path(tmp_dir), max_steps=1))

            first = executor.execute_text("get_budget_status()")
            second = executor.execute_text("get_budget_status()")

        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "error")
        self.assertIn("Step budget exceeded", second["error"])


class WebServerModeTest(unittest.TestCase):
    def _state(self, mode: str) -> RunState:
        workspace = Path(tempfile.mkdtemp())
        executor = CommandExecutor(AgentContext(workspace=workspace))
        return RunState(
            run_id="test",
            config=RunConfig(task_id="toy", mode=mode),
            workspace=workspace,
            executor=executor,
        )

    def test_fixed_transitions_blocks_submit_before_train(self) -> None:
        state = self._state("fixed-transitions")

        error = validate_mode_command(state, parse_command('submit("submission.csv")'))
        allowed = validate_mode_command(state, parse_command('read_file("train.csv")'))

        self.assertIsNotNone(error)
        self.assertEqual(error["status"], "error")
        self.assertIn("not allowed during EDA", error["error"])
        self.assertIsNone(allowed)

    def test_extract_commands_accepts_single_shot_fenced_batch(self) -> None:
        text = '''Делаю решение
```command
run_python("""
print("create submission")
""")
submit("submission.csv")
```'''

        commands = extract_commands(text)

        self.assertEqual([command.name for command in commands], ["run_python", "submit"])
        self.assertIn("create submission", commands[0].args["code_or_file"])

    def test_extract_commands_ignores_inline_command_mentions(self) -> None:
        text = 'Подготовлю решение и потом вызову submit("submission.csv").'

        with self.assertRaisesRegex(ValueError, "No executable command"):
            extract_commands(text)

    def test_extract_commands_accepts_command_on_own_line(self) -> None:
        text = '''Отправляю готовый файл
submit("submission.csv")'''

        commands = extract_commands(text)

        self.assertEqual([command.name for command in commands], ["submit"])

    def test_single_shot_submit_only_is_not_batch_error(self) -> None:
        state = self._state("single-shot")

        error = validate_mode_command_batch(state, [parse_command('submit("submission.csv")')])

        self.assertIsNone(error)

    def test_fixed_transitions_advances_stages_and_finishes_after_train(self) -> None:
        state = self._state("fixed-transitions")

        self.assertEqual(current_fixed_stage(state), "EDA")
        for _ in range(2):
            record_fixed_stage_attempt(state)
            self.assertIsNone(apply_mode_after_results(state, [{"status": "ok", "command": "read_file"}]))
            self.assertEqual(current_fixed_stage(state), "EDA")
        record_fixed_stage_attempt(state)
        self.assertIsNone(apply_mode_after_results(state, [{"status": "ok", "command": "read_file"}]))
        self.assertEqual(current_fixed_stage(state), "FEATURES")
        self.assertIsNone(apply_mode_after_results(state, [{"status": "error", "command": "write_file"}]))
        self.assertEqual(current_fixed_stage(state), "FEATURES")
        for _ in range(4):
            record_fixed_stage_attempt(state)
            self.assertIsNone(apply_mode_after_results(state, [{"status": "ok", "command": "write_file"}]))
            self.assertEqual(current_fixed_stage(state), "FEATURES")
        record_fixed_stage_attempt(state)
        self.assertIsNone(apply_mode_after_results(state, [{"status": "ok", "command": "write_file"}]))
        self.assertEqual(current_fixed_stage(state), "TRAIN")
        record_fixed_stage_attempt(state)
        self.assertEqual(
            apply_mode_after_results(state, [{"status": "ok", "command": "run_python"}]),
            "fixed_transitions_finished",
        )

    def test_repeated_mode_stops_after_attempt_limit(self) -> None:
        state = self._state("repeated")
        state.requests = REPEATED_MAX_ATTEMPTS

        self.assertEqual(
            apply_mode_after_results(state, [{"status": "ok", "command": "read_file"}]),
            "repeated_attempt_limit",
        )

    def test_repeated_requires_submit_on_every_attempt(self) -> None:
        state = self._state("repeated")
        state.requests = 1

        error = validate_mode_command_batch(state, [parse_command('run_python("print(1)")')])

        self.assertIsNotNone(error)
        self.assertEqual(error["reason"], "repeated_missing_submit")

    def test_repeated_rejects_submit_only(self) -> None:
        state = self._state("repeated")

        error = validate_mode_command_batch(state, [parse_command('submit("submission.csv")')])

        self.assertIsNotNone(error)
        self.assertEqual(error["reason"], "repeated_submit_without_solution")

    def test_repeated_accepts_solution_command_before_submit(self) -> None:
        state = self._state("repeated")

        error = validate_mode_command_batch(
            state,
            [parse_command('run_python("print(1)")'), parse_command('submit("submission.csv")')],
        )

        self.assertIsNone(error)

    def test_repeated_submit_does_not_finish_before_attempt_limit(self) -> None:
        state = self._state("repeated")
        state.requests = 1
        result = {"status": "ok", "command": "submit", "result": {"metric": "mae", "value": 10.0}}

        handle_successful_submit(state, result)

        self.assertFalse(state.submitted)
        self.assertEqual(state.submission_result, result)

    def test_repeated_best_submission_uses_lower_error_metric(self) -> None:
        first = {"status": "ok", "command": "submit", "result": {"metric": "mae", "value": 10.0}}
        second = {"status": "ok", "command": "submit", "result": {"metric": "mae", "value": 7.0}}

        self.assertEqual(best_repeated_submission([first, second]), second)

    def test_repeated_best_submission_uses_higher_score_metric(self) -> None:
        first = {"status": "ok", "command": "submit", "result": {"metric": "f1_score", "value": 0.72}}
        second = {"status": "ok", "command": "submit", "result": {"metric": "f1_score", "value": 0.81}}

        self.assertEqual(best_repeated_submission([first, second]), second)

    def test_mode_instruction_does_not_reveal_fixed_stage(self) -> None:
        fixed_state = self._state("fixed-transitions")
        flexible_state = self._state("flexible")
        repeated_state = self._state("repeated")

        fixed_prompt = apply_mode_instruction(fixed_state, "base")
        flexible_prompt = apply_mode_instruction(flexible_state, "base")

        self.assertIn("Самостоятельно определи текущий этап", fixed_prompt)
        self.assertNotIn("текущий обязательный этап EDA", fixed_prompt)
        self.assertNotIn("должна быть `EDA`", fixed_prompt)
        self.assertIn("Режим flexible", flexible_prompt)
        self.assertIn("Самостоятельно выбери актуальный этап", flexible_prompt)
        self.assertIn(f"попытка 1/{REPEATED_MAX_ATTEMPTS}", apply_mode_instruction(repeated_state, "base"))

    def test_fixed_transitions_accepts_correct_reported_stage(self) -> None:
        state = self._state("fixed-transitions")

        self.assertEqual(extract_reported_stage('EDA\nИзучу данные.\nlist_files(".")'), "EDA")
        self.assertIsNone(validate_reported_stage(state, 'EDA\nИзучу данные.\nlist_files(".")'))

    def test_stage_check_rejects_missing_stage(self) -> None:
        state = self._state("flexible")

        error = validate_reported_stage(state, 'Посмотрю данные.\nlist_files(".")')

        self.assertIsNotNone(error)
        self.assertIn("Первая непустая строка", error["error"])

    def test_fixed_transition_early_stage_zeroes_submit_result(self) -> None:
        state = self._state("fixed-transitions")
        violation = detect_fixed_stage_violation(state, 'FEATURES\nrun_python("print(1)")')

        self.assertIsNotNone(violation)
        state.fixed_stage_violations.append(violation)
        result = {"status": "ok", "command": "submit", "result": {"metric": "f1_score", "value": 0.71}}
        zeroed = zero_fixed_transition_result_if_needed(state, result)

        self.assertEqual(zeroed["result"]["raw_value"], 0.71)
        self.assertEqual(zeroed["result"]["value"], 0.0)
        self.assertTrue(zeroed["result"]["score_zeroed"])
        self.assertEqual(zeroed["result"]["stage_violations"][0]["declared_stage"], "FEATURES")
        self.assertEqual(zeroed["result"]["stage_violations"][0]["expected_stage"], "EDA")

    def test_fixed_transition_current_stage_does_not_zero_result(self) -> None:
        state = self._state("fixed-transitions")

        self.assertIsNone(detect_fixed_stage_violation(state, 'EDA\nread_file("train.csv")'))
        self.assertIsNone(detect_fixed_stage_violation(state, 'read_file("train.csv")'))

    def test_repeated_prompt_requires_current_attempt_submit(self) -> None:
        state = self._state("repeated")
        state.requests = 1

        prompt = apply_mode_instruction(state, "base")

        self.assertIn("Ответ только с коротким комментарием будет отклонён", prompt)
        self.assertIn("Обязательный формат ответа", prompt)
        self.assertIn("run_python", prompt)
        self.assertIn("не обещай улучшить модель позже", prompt)
        self.assertIn("Submit-only будет отклонён", prompt)
        self.assertIn("submit(\"submission.csv\")", prompt)
        self.assertNotIn("сможешь в следующей попытке", prompt)

    def test_repeated_mode_isolates_llm_history_between_attempts(self) -> None:
        self.assertTrue(should_isolate_llm_history(self._state("repeated")))
        self.assertFalse(should_isolate_llm_history(self._state("flexible")))

    def test_repeated_attempts_use_clean_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            task_dir = project_root / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "toy",
                        "metric": "mae",
                        "column": "target",
                        "public_files": ["train.csv", "test.csv"],
                    }
                ),
                encoding="utf-8",
            )
            (task_dir / "train.csv").write_text("id,x,target\n1,10,100\n", encoding="utf-8")
            (task_dir / "test.csv").write_text("id,x\n2,20\n", encoding="utf-8")
            manager = AgentRunManager(project_root, Path("tasks"), Path("runs"), use_docker=False)
            config = RunConfig(task_id="toy", mode="repeated")
            first_workspace = prepare_run_workspace(
                project_root,
                manager.tasks_dir,
                manager.run_root,
                "toy",
                repeated_attempt_workspace_id("run", 1),
            )
            (first_workspace / "solution.py").write_text("old attempt", encoding="utf-8")
            executor, docker_container, created_container = manager._make_executor(
                config, first_workspace, repeated_attempt_workspace_id("run", 1)
            )
            state = RunState(
                run_id="run",
                config=config,
                workspace=first_workspace,
                executor=manager._make_executor(config, first_workspace, repeated_attempt_workspace_id("run", 1))[0],
            )
            state.requests = 1

            manager._prepare_next_repeated_attempt(state)

            self.assertTrue(state.workspace.name.endswith("attempt-2"))
            self.assertTrue((state.workspace / "train.csv").exists())
            self.assertFalse((state.workspace / "solution.py").exists())


class WebServerFileBrowserTest(unittest.TestCase):
    def test_workspace_file_listing_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "solution.py").write_text("print('ok')\n", encoding="utf-8")
            (workspace / "model.bin").write_bytes(b"\x00\x01")

            files = list_workspace_files(workspace)
            preview = workspace_file_preview(workspace, "solution.py")

        by_path = {item["path"]: item for item in files}
        self.assertTrue(by_path["solution.py"]["previewable"])
        self.assertFalse(by_path["model.bin"]["previewable"])
        self.assertEqual(preview["path"], "solution.py")
        self.assertEqual(preview["content"], "print('ok')\n")

    def test_workspace_file_resolution_rejects_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "inside.txt").write_text("ok", encoding="utf-8")

            resolved = resolve_workspace_file(workspace, "inside.txt")

            with self.assertRaisesRegex(ValueError, "outside workspace"):
                resolve_workspace_file(workspace, "../outside.txt")

        self.assertEqual(resolved.name, "inside.txt")


class HintEngineTest(unittest.TestCase):
    def test_perfect_numeric_correlations_detects_positive_and_negative_pairs(self) -> None:
        table = CsvTable(
            path=Path("train.csv"),
            columns=["x", "double_x", "negative_x", "target"],
            rows=[
                {"x": "1", "double_x": "2", "negative_x": "-1", "target": "10"},
                {"x": "2", "double_x": "4", "negative_x": "-2", "target": "20"},
                {"x": "3", "double_x": "6", "negative_x": "-3", "target": "30"},
            ],
        )

        pairs = _perfect_numeric_correlations(table, exclude={"target"})

        self.assertIn(("x", "double_x"), pairs)
        self.assertIn(("x", "negative_x"), pairs)

    def test_contains_metric_value_nested_data(self) -> None:
        self.assertTrue(_contains_metric_value({"folds": [{"validation_rmse": 12.5}]}))
        self.assertFalse(_contains_metric_value({"folds": [{"validation_rmse": "bad"}]}))

    def test_build_feedback_detects_leakage_and_validation_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            task_dir = workspace / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps({"id": "toy", "metric": "mae", "column": "target"}),
                encoding="utf-8",
            )
            (workspace / "train.csv").write_text(
                "id,x,target\n1,10,100\n2,20,200\n3,30,300\n",
                encoding="utf-8",
            )
            (workspace / "test.csv").write_text("id,x\n4,40\n", encoding="utf-8")
            (workspace / "leaky.csv").write_text("id,target\n1,100\n", encoding="utf-8")
            (workspace / "metrics.json").write_text('{"mae": 0.1}', encoding="utf-8")
            (workspace / "agent_history.txt").write_text(
                "\n".join(
                    [
                        json.dumps({"command": "read_file", "args": {"path": "train.csv"}, "status": "ok"}),
                        json.dumps({"command": "read_file", "args": {"path": "test.csv"}, "status": "ok"}),
                        json.dumps({"command": "show_dataset_info", "args": {}, "status": "ok"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            feedback = HintEngine(workspace, "toy", workspace / "tasks").build_feedback()

        messages = [hint["message"] for hint in feedback["hints"]]
        self.assertTrue(any("утеч" in message for message in messages))
        self.assertFalse(any("валидац" in message for message in messages))

    def test_build_feedback_requires_dataset_inspection_before_feature_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            task_dir = workspace / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps({"id": "toy", "metric": "mae", "column": "target"}),
                encoding="utf-8",
            )
            (workspace / "train.csv").write_text("id,city,target\n1,msk,10\n2,spb,20\n", encoding="utf-8")
            (workspace / "test.csv").write_text("id,city\n3,kzn\n", encoding="utf-8")

            feedback = HintEngine(workspace, "toy", workspace / "tasks").build_feedback()

        self.assertEqual({hint["stage"] for hint in feedback["hints"]}, {"EDA"})
        self.assertTrue(any("сводк" in hint["message"] for hint in feedback["hints"]))

    def test_build_feedback_detects_shared_preprocessing_issue_after_eda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            task_dir = workspace / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps({"id": "toy", "metric": "mae", "column": "target"}),
                encoding="utf-8",
            )
            (workspace / "train.csv").write_text(
                "sample_id,city,target\n1,msk,10\n2,spb,20\n3,kzn,30\n",
                encoding="utf-8",
            )
            (workspace / "test.csv").write_text("sample_id,city\n4,ekb\n", encoding="utf-8")
            (workspace / "agent_history.txt").write_text(
                "\n".join(
                    [
                        json.dumps({"command": "read_file", "args": {"path": "train.csv"}, "status": "ok"}),
                        json.dumps({"command": "read_file", "args": {"path": "test.csv"}, "status": "ok"}),
                        json.dumps({"command": "show_dataset_info", "args": {}, "status": "ok"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            feedback = HintEngine(workspace, "toy", workspace / "tasks").build_feedback()

        self.assertEqual({hint["stage"] for hint in feedback["hints"]}, {"Feature engineering"})
        self.assertTrue(any("предобработка" in hint["message"] for hint in feedback["hints"]))

    def test_build_feedback_adds_next_stage_after_three_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            task_dir = workspace / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps({"id": "toy", "metric": "mae", "column": "target"}),
                encoding="utf-8",
            )
            (workspace / "train.csv").write_text(
                "id,feature,target\n1,foo,100\n1,foo,100\n2,bar,200\n",
                encoding="utf-8",
            )
            (workspace / "test.csv").write_text("id,feature\n3,baz\n", encoding="utf-8")
            (workspace / "agent_history.txt").write_text(
                "\n".join(
                    [
                        json.dumps({"command": "read_file", "args": {"path": "train.csv"}, "status": "ok"}),
                        json.dumps({"command": "read_file", "args": {"path": "test.csv"}, "status": "ok"}),
                        json.dumps({"command": "show_dataset_info", "args": {}, "status": "ok"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            engine = HintEngine(workspace, "toy", workspace / "tasks")

            first = engine.build_feedback()
            second = engine.build_feedback()
            third = engine.build_feedback()

        self.assertEqual({hint["stage"] for hint in first["hints"]}, {"EDA"})
        self.assertEqual({hint["stage"] for hint in second["hints"]}, {"EDA"})
        self.assertEqual({hint["stage"] for hint in third["hints"]}, {"EDA", "Feature engineering"})
        self.assertTrue(any("повторяющихся" in hint["message"] for hint in third["hints"]))
        self.assertTrue(any("утечки" in hint["message"] or "дублирующих" in hint["message"] for hint in third["hints"]))


class LlmClientTest(unittest.TestCase):
    def test_build_messages_orders_system_history_and_user_message(self) -> None:
        history = [{"role": "assistant", "content": "old"}]

        messages = build_messages("new", system_message="system", history=history)

        self.assertEqual(messages, [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "new"},
        ])

    @patch("agent.llm_client.urllib.request.urlopen")
    def test_chat_completion_posts_payload_and_decodes_json(self, urlopen_mock: Mock) -> None:
        response = Mock()
        response.read.return_value = b'{"choices":[{"message":{"content":"ok"}}]}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        urlopen_mock.return_value = response

        result = chat_completion(
            "hello",
            system_message="system",
            history=[{"role": "assistant", "content": "old"}],
            url="http://example.test/chat",
            model="toy-model",
            max_tokens=7,
            temperature=0.2,
            timeout=3,
            extra_payload={"top_p": 0.9},
        )

        request = urlopen_mock.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertEqual(payload["model"], "toy-model")
        self.assertEqual(payload["messages"][-1]["content"], "hello")
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(urlopen_mock.call_args.kwargs["timeout"], 3)

    @patch("agent.llm_client.chat_completion")
    def test_ask_llm_returns_first_message_content(self, chat_completion_mock: Mock) -> None:
        chat_completion_mock.return_value = {"choices": [{"message": {"content": "answer"}}]}

        self.assertEqual(ask_llm("question"), "answer")


if __name__ == "__main__":
    unittest.main()
