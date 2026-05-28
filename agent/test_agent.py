import json
import tempfile
import unittest
from pathlib import Path

from agent.executor import AgentContext, CommandExecutor
from agent.parser import extract_command_text, parse_command, parse_model_response


class ParserTest(unittest.TestCase):
    def test_parse_call_command(self) -> None:
        command = parse_command('read_file("notes.txt")')

        self.assertEqual(command.name, "read_file")
        self.assertEqual(command.args, {"path": "notes.txt"})

    def test_parse_json_command(self) -> None:
        command = parse_command('{"command": "show_sample_rows", "args": {"n": 2}}')

        self.assertEqual(command.name, "show_sample_rows")
        self.assertEqual(command.args, {"n": 2})

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

        self.assertEqual(write_result["status"], "ok")
        self.assertEqual(read_result["result"]["content"], "hello")
        self.assertEqual(len(trajectory_result["result"]), 2)

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
