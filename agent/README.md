# Agent commands

This module parses, validates, executes, and logs commands produced by an LLM.

Supported command syntax:

```text
read_file("checker/README.md")
show_sample_rows(5)
submit("submission.csv")
```

JSON syntax is also supported:

```json
{"command": "read_file", "args": {"path": "checker/README.md"}}
```

Free-form model responses are supported through `parse_model_response` or the CLI
flag `--model-response`. The extractor accepts one command in a fenced block, on
its own line, or inline in text:

```text
Сначала посмотрю файлы проекта.

```command
list_files("checker")
```
```

If several commands are found in one response, execution is rejected as
ambiguous.

## Commands

Files:

- `list_files(path)`
- `read_file(path)`
- `write_file(path, content)`
- `edit_file(path, diff)`

Data:

- `load_dataset(path)`
- `show_dataset_info()`
- `show_sample_rows(n)`

Execution:

- `run_python(code)`
- `run_python(file)`

Environment:

- `get_budget_status()`
- `get_remaining_time()`
- `get_hints()`

Logging:

- `get_trajectory()`

Submission:

- `submit(file)`

## Manual usage

From the repository root:

```bash
python3 -m agent.cli 'list_files("checker")'
python3 -m agent.cli 'load_dataset("checker/tasks/salary_prediction/train.csv")'
python3 -m agent.cli 'get_hints()'
python3 -m agent.cli 'submit("checker/tasks/salary_prediction/answers.csv")'
python3 -m agent.cli --model-response 'Сейчас прочитаю README: read_file("checker/README.md")'
```

`submit(file)` uses the `salary_prediction` task by default. Pass another task:

```bash
python3 -m agent.cli --task-id sample_with_id 'submit("checker/tasks/sample_with_id/submission_example.csv")'
```

## Validation

The executor rejects paths outside the workspace, unknown commands, invalid
argument types, missing datasets, invalid submission files, and exhausted step
budgets.

Every executed command is appended to `agent_history.txt` in the workspace as one
JSON object per line. `get_trajectory()` returns the parsed contents of that
file, so the LLM can inspect the persistent command history.

## Feedback hints

The full benchmark loop adds feedback once after all commands from the latest
model response have executed. The feedback contains abstract suggestions for
improving the current ML solution. The hidden checklist covers:

- EDA: target, duplicates, missing values, outliers;
- feature engineering: target leakage and duplicated numeric signal;
- training: task type, saved model, validation signal, and submission file.

The LLM can request the same feedback explicitly:

```bash
python3 -m agent.cli 'get_hints()'
```

Hints intentionally stay high-level, for example: "Подумай над пропусками и
стратегией их обработки."

`edit_file(path, diff)` expects `diff` to be a JSON string or object:

```json
{"old": "text to replace", "new": "replacement text"}
```
