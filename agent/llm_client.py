"""HTTP client for OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


URL = os.environ.get("LLM_URL", "http://llm.letovo.site:8809/openai")
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")


def _dotenv_value(name: str) -> str | None:
    env_path = os.getcwd() + "/.env"
    try:
        with open(env_path, encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() == name:
                    return value.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


API_KEY = (
    os.environ.get("LLM_API_KEY")
    or _dotenv_value("LLM_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or _dotenv_value("OPENAI_API_KEY")
)
SYSTEM_MESSAGE = """Решай ML benchmark только агентскими командами. Любой ответ без команд считается ошибкой и тратой бюджета.

ЖЕСТКИЙ БЮДЖЕТ:
- В каждом user prompt есть процент оставшихся токенов и итераций. Это главный сигнал управления.
- Если осталось <50% токенов: прекращай долгие исследования, делай простой надежный baseline и проверяй файл.
- Если осталось <30% токенов: следующий шаг должен быть подготовкой/проверкой submission или submit.
- Если осталось <15% токенов: вызывай submit с лучшим текущим файлом.
- Не пиши объяснения, планы, Markdown или рассуждения. Возвращай только команды.

Команды:
list_files(path), read_file(path), write_file(path, content), edit_file(path, diff)
load_dataset(path), show_dataset_info(), show_sample_rows(n)
run_python(code) или run_python(file)
get_budget_status(), get_remaining_time(), get_trajectory(), submit(file)

Правила:
- Каждая команда отдельной строкой: command_name(args).
- run_python запускается заново каждый раз; все импорты и чтение файлов пиши внутри каждого вызова.
- Сначала быстро осмотри данные, затем создай submission.csv, проверь формат и вызови submit("submission.csv").
- submit можно сделать только один раз; после него команды больше не выполняются."""

DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7


Message = dict[str, str]
JsonDict = dict[str, Any]


def auth_headers(api_key: str | None = API_KEY) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


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
        headers=auth_headers(),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot connect to LLM endpoint {url}: {exc.reason}") from exc


def ask_llm(user_message: str, **kwargs: Any) -> str:
    """Send a prompt and return the first assistant message content."""
    data = chat_completion(user_message, **kwargs)
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    response = chat_completion("Выведи список всех доступных команд:")
    print(json.dumps(response, ensure_ascii=False, indent=2))
