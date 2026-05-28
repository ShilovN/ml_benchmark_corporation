"""HTTP client for OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import urllib.request
from typing import Any


URL = "http://10.129.143.63:8000/compatible-mode/v1/chat/completions"
MODEL = "qwen3-4b-local"
SYSTEM_MESSAGE = """Твоя задача решить ML задачу, используя ТОЛКЬО агентские команды. Запросы пользователя - результат действий твоих команд. Вот список доступных тебе команд:
list_files(path) - посмотреть файлы
read_file(path) - получить содержание файла
write_file(path, content) - записать содержимое в файл
edit_file(path, diff) - отредактировать файл

# Data
load_dataset(path) - загрузить датасет
show_dataset_info() - показать информацию о датасете
show_sample_rows(n) - показать первые n строк

# Execution
run_python(code) - выполнить питоновский код
run_python(file) - выполнить питоновский файл

# Environment
get_budget_status() - получить информацию о количестве оставшихся токенах
get_remaining_time() - получить информацию об оставшемся времени

# Logging
get_trajectory() - получить предыдущие команды, которые выполнялись

# Submission
submit(file) - загрузить файл на проверку (можно выполнить только 1 раз)"""
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TOKENS = 10240
DEFAULT_TEMPERATURE = 0.7


Message = dict[str, str]
JsonDict = dict[str, Any]


def build_messages(
    user_message: str,
    *,
    system_message: str = SYSTEM_MESSAGE,
    history: list[Message] | None = None,
) -> list[Message]:
    """Build messages for a single user request with optional prior history."""
    messages = [{"role": "system", "content": system_message}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def chat_completion(
    user_message: str,
    *,
    system_message: str = SYSTEM_MESSAGE,
    history: list[Message] | None = None,
    url: str = URL,
    model: str = MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: int = DEFAULT_TIMEOUT,
    extra_payload: JsonDict | None = None,
) -> JsonDict:
    """Send a chat completion request and return the decoded JSON response."""
    payload: JsonDict = {
        "model": model,
        "messages": build_messages(
            user_message,
            system_message=system_message,
            history=history,
        ),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_payload:
        payload.update(extra_payload)

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask_llm(user_message: str, **kwargs: Any) -> str:
    """Send a prompt and return the first assistant message content."""
    data = chat_completion(user_message, **kwargs)
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    response = chat_completion("Выведи список всех доступных команд:")
    print(json.dumps(response, ensure_ascii=False, indent=2))
