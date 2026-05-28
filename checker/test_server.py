import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path

from server import load_tasks, make_handler


class FakeRequest:
    def __init__(self, headers, body: bytes) -> None:
        self.headers = headers
        self.rfile = io.BytesIO(body)


class ServerTest(unittest.TestCase):
    def test_load_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir) / "task"
            task_dir.mkdir()
            (task_dir / "answers.txt").write_text("1\n2\n", encoding="utf-8")
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "numbers",
                        "name": "Numbers",
                        "metric": "mae",
                        "answer_file": "answers.txt",
                    }
                ),
                encoding="utf-8",
            )

            tasks = load_tasks(Path(tmp_dir))

        self.assertEqual(list(tasks), ["numbers"])
        self.assertEqual(tasks["numbers"].metric, "mae")

    def test_multipart_form_parsing(self) -> None:
        handler_class = make_handler({})
        boundary = "test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="task_id"\r\n\r\n'
            "sample\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="submission.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
            "cat\ncat\ncat\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        headers = Message()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        headers["Content-Length"] = str(len(body))
        request = FakeRequest(headers, body)

        form = handler_class._read_multipart_form(request)

        self.assertEqual(form["task_id"].value, "sample")
        self.assertEqual(form["file"].file.read().decode("utf-8"), "cat\ncat\ncat")


if __name__ == "__main__":
    unittest.main()
