import argparse
import json
import tempfile
import unittest
from pathlib import Path

from agent.executor import AgentContext, CommandExecutor
from agent.parser import extract_command_text, parse_command, parse_model_response
from main import BenchmarkStats, build_followup_prompt, compact_json_for_prompt


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

    def test_multiple_commands_raise_error(self) -> None:
        text = """read_file("a.txt")
read_file("b.txt")"""

        with self.assertRaisesRegex(ValueError, "Multiple commands"):
            extract_command_text(text)

    def test_no_command_raises_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "No executable command"):
            extract_command_text("Я пока думаю, какую команду вызвать.")


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

        prompt = build_followup_prompt(command_results, BenchmarkStats(), args, feedback)
        payload = json.loads(prompt.split("\n\n", 1)[1])

        self.assertEqual(payload["feedback"], feedback)
        self.assertNotIn("feedback", payload["command_results"][0])
        self.assertNotIn("feedback", payload["command_results"][1])

    def test_followup_prompt_contains_remaining_budget(self) -> None:
        args = argparse.Namespace(token_limit=1000, max_steps=10, time_limit_seconds=3600)
        stats = BenchmarkStats()
        stats.total_tokens = 250
        stats.executed_commands = 4

        prompt = build_followup_prompt([], stats, args)
        payload = json.loads(prompt.split("\n\n", 1)[1])

        self.assertIn("осталось 75.0% токенов", prompt)
        self.assertIn("осталось итераций: 6", prompt)
        self.assertNotIn("\n  ", prompt)
        self.assertEqual(payload["benchmark_status"]["remaining_tokens"], 750)
        self.assertEqual(payload["benchmark_status"]["remaining_token_percent"], 75.0)
        self.assertEqual(payload["benchmark_status"]["remaining_iterations"], 6)

    def test_followup_prompt_warns_when_token_budget_is_low(self) -> None:
        args = argparse.Namespace(token_limit=1000, max_steps=10, time_limit_seconds=3600)
        stats = BenchmarkStats()
        stats.total_tokens = 701

        prompt = build_followup_prompt([], stats, args)

        self.assertIn("осталось 29.9% токенов", prompt)
        self.assertIn("Экономь токены", prompt)
        self.assertIn("сделай submit", prompt)
        self.assertIn("иначе проиграешь", prompt)

    def test_followup_prompt_does_not_warn_at_exactly_30_percent(self) -> None:
        args = argparse.Namespace(token_limit=1000, max_steps=10, time_limit_seconds=3600)
        stats = BenchmarkStats()
        stats.total_tokens = 700

        prompt = build_followup_prompt([], stats, args)

        self.assertIn("осталось 30.0% токенов", prompt)
        self.assertNotIn("Экономь токены", prompt)

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
        self.assertTrue(any("пропус" in message for message in messages))
        self.assertTrue(any("повтор" in message for message in messages))
        self.assertTrue(any("модель" in message for message in messages))
        self.assertTrue(any("отправ" in message for message in messages))

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
        workspace = Path.cwd()
        executor = CommandExecutor(AgentContext(workspace=workspace, task_id="salary_prediction"))

        result = executor.execute_text('submit("checker/tasks/salary_prediction/answers.csv")')

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["result"]["metric"], "mae")
        self.assertEqual(result["result"]["value"], 0.0)

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


if __name__ == "__main__":
    unittest.main()
