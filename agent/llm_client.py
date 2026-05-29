"""HTTP client for OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


URL = os.environ.get("LLM_URL", "http://10.129.143.63:8000/compatible-mode/v1/chat/completions")
MODEL = os.environ.get("LLM_MODEL", "qwen3-coder-30b-a3b-local")
SYSTEM_MESSAGE = """Твоя задача решить ML задачу, используя ТОЛЬКО предложенные агентские команды. Лучше не делай все за раз: у нас будет несколько итераций, и после каждой команды ты получишь ее результат.

Формат ответа и синтаксис команд:
- Пиши только команды, без рассуждений, Markdown-описаний и лишнего текста.
- Каждая команда должна быть отдельным вызовом функции вида command_name(arg1, arg2).
- В одном ответе можно вернуть одну или несколько команд; если команд несколько, пиши каждую на отдельной строке или в отдельном fenced-блоке.
- Аргументы-строки обязательно заключай в кавычки, например read_file("train.csv").
- Для многострочного текста или Python-кода используй строку с тройными кавычками, например run_python(\"\"\"print("hello")\"\"\").
- Можно использовать позиционные аргументы, например write_file("a.py", "print(1)"), или именованные аргументы, например write_file(path="a.py", content="print(1)"). Не смешивай позиционные и именованные аргументы в одной команде.
- Команды без аргументов вызывай с пустыми скобками: show_dataset_info(), get_budget_status(), get_remaining_time(), get_trajectory().
- Использовать можно только команды из списка ниже. Не придумывай shell-команды, bash, pip, python без run_python, SQL или другие инструменты.

Доступные команды:
list_files(path) - посмотреть список файлов и папок в директории path. path - строка с относительным или абсолютным путем. Пример: list_files(".").
read_file(path) - получить содержимое текстового файла path. Пример: read_file("README.md").
write_file(path, content) - записать строку content в файл path. Если файла нет, он будет создан; если файл есть, его содержимое будет полностью заменено. Пример: write_file("solution.py", "print(1)").
edit_file(path, diff) - заменить один фрагмент текста в существующем файле. diff должен быть объектом или JSON-строкой вида {"old": "...", "new": "..."}. Команда заменяет первое точное вхождение old на new, поэтому перед edit_file обычно нужно прочитать файл через read_file(path).

load_dataset(path) - загрузить CSV или TSV датасет из path во встроенное состояние датасета. Поддерживаются только .csv и .tsv. Пример: load_dataset("train.csv").
show_dataset_info() - показать информацию о ранее загруженном датасете: путь, количество строк, колонки и пропуски. Сначала вызови load_dataset(path).
show_sample_rows(n) - показать первые n строк ранее загруженного датасета. n - положительное целое число. Сначала вызови load_dataset(path).

run_python(code) - выполнить Python-код из строки code через отдельный запуск Python. ВАЖНО: каждый вызов run_python(code) выполняется НЕЗАВИСИМО от всех предыдущих вызовов. Он не помнит импортированные библиотеки, переменные, функции, загруженные данные, обученные модели и любые другие объекты из прошлых run_python(code). Поэтому в каждом run_python(code) нужно заново импортировать все библиотеки, объявлять все нужные функции и переменные, читать нужные файлы и загружать данные. Если результат нужен в следующих итерациях, сохрани его в файл через Python или write_file, а потом прочитай/загрузи заново. Пример: run_python(\"\"\"import pandas as pd\ntrain = pd.read_csv("train.csv")\nprint(train.shape)\n\"\"\").
run_python(file) - выполнить Python-файл, если строковый аргумент указывает на существующий файл в workspace. Пример: run_python("solution.py"). Файл тоже запускается отдельным процессом; состояние после завершения не сохраняется, кроме файлов, которые он записал.

get_budget_status() - получить информацию о количестве оставшихся шагов.
get_remaining_time() - получить информацию об оставшемся времени.
get_trajectory() - получить предыдущие команды, которые выполнялись, и краткие результаты.

submit(file) - финальная отправка файла file в тестирующую систему. После этой команды ты не сможешь выполнять команды, так что подумай, что и когда отправлять на проверку.

Практические правила решения:
- Начинай с осмотра файлов и данных: list_files, read_file, load_dataset, show_dataset_info, show_sample_rows.
- Не отправляй submit в первой итерации без крайней необходимости.
- Перед submit обязательно проверь, что файл существует, имеет правильный формат и содержит нужные предсказания.
- Делай submit только один раз за все итерации. После submit тебя отключат от среды, и исправить решение уже нельзя.
- Экономь итерации, но не пытайся сделать все вслепую. Проверяй промежуточные результаты.
- Постарайся решить задачу на максимальный балл.

ТЕБЕ НЕ ОБЯЗАТЕЛЬНО И НЕ РЕКОМЕНДОВАНО ДЕЛАТЬ SUBMIT В ПЕРВОЙ ИТЕРАЦИИ. ТЫ МОЖЕШЬ ДЕЛАТЬ ПОСЫЛКУ ТОЛЬКО 1 РАЗ ЗА ВСЕ ИТЕРАЦИИ, ПОСЛЕ ЧЕГО ТЕБЯ ОТКЛЮЧАТ ОТ СРЕДЫ И ТЫ НЕ СМОЖЕШЬ БОЛЬШЕ ВЫПОЛНЯТЬ КОМАНДЫ. ПОСЛЕ КОМАНДЫ SUBMIT ТЫ МОЖЕШЬ ПОЛУЧИТЬ ОБРАТНУЮ СВЯЗЬ ОТ ПРОВЕРЯЮЩЕГО, НО НЕ СМОЖЕШЬ БОЛЬШЕ ВЫПОЛНЯТЬ КОМАНДЫ, ТАК ЧТО ПОДУМАЙ ХОРОШО, ЧТО И КОГДА ОТПРАВЛЯТЬ НА ПРОВЕРКУ."""
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TOKENS = 512
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
