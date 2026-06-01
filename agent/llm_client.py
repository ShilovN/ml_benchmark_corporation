"""HTTP client for OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


LETOVO_URL = "http://llm.letovo.site:8809/openai"
OPENAI_URL = "https://api.openai.com/v1"
LETOVO_MODELS = [
    "deepseek-v4-flash",
    "gemma-4-26b",
]
CHATGPT_MODELS = [
    "gpt-5-mini",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-4.1-nano",
]
MODEL_GROUPS = [
    ("Letovo", LETOVO_MODELS),
    ("ChatGPT", CHATGPT_MODELS),
]
AVAILABLE_MODELS = [model for _name, models in MODEL_GROUPS for model in models]

MODEL = os.environ.get("LLM_MODEL", CHATGPT_MODELS[0])
URL = os.environ.get("LLM_URL", "")


def resolve_model_url(model: str) -> str:
    if model in LETOVO_MODELS:
        return LETOVO_URL
    if model in CHATGPT_MODELS:
        return OPENAI_URL
    return URL or LETOVO_URL


if not URL:
    URL = resolve_model_url(MODEL)


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


def _llm_api_key() -> str | None:
    return os.environ.get("LLM_API_KEY") or _dotenv_value("LLM_API_KEY")


def _letovo_api_key() -> str | None:
    return (
        os.environ.get("LETOVO_API_KEY")
        or _dotenv_value("LETOVO_API_KEY")
        or _llm_api_key()
    )


def _openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or _dotenv_value("OPENAI_API_KEY")


def api_key(url: str | None = None) -> str | None:
    resolved_url = url or URL
    if _is_letovo_url(resolved_url):
        return _letovo_api_key()
    if _is_openai_url(resolved_url):
        return _openai_api_key()
    llm_key = _llm_api_key()
    if llm_key:
        return llm_key
    return None


def _is_openai_url(url: str) -> bool:
    return url.rstrip("/").lower().startswith("https://api.openai.com")


def _is_letovo_url(url: str) -> bool:
    return url.rstrip("/").lower().startswith(LETOVO_URL.lower())


SYSTEM_MESSAGE = """Решай ML benchmark только агентскими командами. Любой ответ без команд считается ошибкой.

Команды:
list_files(path), read_file(path), write_file(path, content), edit_file(path, diff)
load_dataset(path), show_dataset_info(), show_sample_rows(n)
run_python(code) или run_python(file)
get_budget_status(), get_remaining_time(), get_trajectory(), submit(file)

Правила:
- Каждая команда отдельной строкой: command_name(args).
- run_python запускается заново каждый раз; все импорты и чтение файлов пиши внутри каждого вызова.
- Работай как в обычной ML-задаче: осмотри данные, сделай признаки, обучи и проверь модель.
- Перед submit проверь, что submission.csv существует, имеет правильный формат и содержит нужные предсказания.
- Если система явно сообщает FINALIZATION MODE, прекращай улучшения и вызывай submit("submission.csv") с лучшим подготовленным файлом."""

DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7


Message = dict[str, str]
JsonDict = dict[str, Any]


def https_context(url: str) -> ssl.SSLContext | None:
    if not url.lower().startswith("https://"):
        return None

    certifi_path = _certifi_path()
    cafile = certifi_path or _system_ca_path()
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def _certifi_path() -> str | None:
    try:
        import certifi  # type: ignore[import-not-found]
    except ImportError:
        return None
    return str(certifi.where())


def _system_ca_path() -> str | None:
    env_cafile = os.environ.get("SSL_CERT_FILE")
    candidates = [
        env_cafile,
        "/opt/homebrew/etc/openssl@3/cert.pem",
        "/opt/homebrew/etc/ca-certificates/cert.pem",
        "/etc/ssl/cert.pem",
        "/usr/local/etc/openssl@3/cert.pem",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def open_url(request: urllib.request.Request, *, timeout: int) -> Any:
    return urllib.request.urlopen(
        request,
        timeout=timeout,
        context=https_context(request.full_url),
    )


def auth_headers(api_key_value: str | None = None, url: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = api_key_value if api_key_value is not None else api_key(url)
    if key:
        headers["Authorization"] = f"Bearer {key}"
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
    }
    if not _uses_default_temperature(model):
        payload["temperature"] = temperature
    if _uses_max_completion_tokens(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    if extra_payload:
        payload.update(extra_payload)

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=auth_headers(url=url),
        method="POST",
    )

    try:
        with open_url(request, timeout=timeout) as response:
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


def _uses_max_completion_tokens(model: str) -> bool:
    return model.startswith("gpt-5")


def _uses_default_temperature(model: str) -> bool:
    return model.startswith("gpt-5")


if __name__ == "__main__":
    response = chat_completion("Выведи список всех доступных команд:")
    print(json.dumps(response, ensure_ascii=False, indent=2))
