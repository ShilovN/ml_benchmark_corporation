# ML Benchmark Corporation

This repository contains a prototype of an AutoML Gym style benchmark
environment: an LLM agent solves ML tasks through a controlled command API, and
the checker evaluates the produced `submission.csv`.

The project has three main parts:

- `web_server.py` - browser UI and benchmark loop for running the LLM agent;
- `agent/` - command parser, executor, trajectory logging, and feedback hints;
- `checker/` - task files, metric checker, and upload/scoring server.

Bundled tasks live in `checker/tasks/`, including salary prediction, map
prediction, and wifi prediction.

## Project Loop

The intended benchmark loop is:

1. The web runner copies public task files into an isolated workspace under
   `benchmark_runs/`.
2. The LLM receives the task prompt and answers with agent commands.
3. `agent.parser` extracts valid commands even if the model includes short
   surrounding text.
4. `agent.executor` executes commands inside the run workspace.
5. Every executed command is appended to `agent_history.txt`.
6. The LLM receives command outputs plus abstract feedback hints.
7. The loop continues according to the selected run mode.
8. The run ends when the model calls `submit(file)` or when the runner performs
   a forced final submit.
9. `checker` validates the submitted file and returns the metric.

The LLM never receives hidden answer files. They are only used by `submit(file)`
for scoring.

## Agent Web UI

The main interface for the benchmark is the agent runner. It starts the LLM,
shows the model responses, parsed commands, command outputs, feedback hints,
budget counters, and the final submission result.

Run:

```bash
export OPENAI_API_KEY="your_openai_key"
export LETOVO_API_KEY="your_letovo_key"
python3 web_server.py --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010/
```

From the page you can:

- choose a task;
- choose a model;
- choose a run mode: `single-shot`, `repeated`, `fixed-transitions`, or `flexible`;
- start or stop a benchmark run;
- watch the live trajectory of prompts, model responses, commands, results, and hints;
- see the final `submit(file)` result.

Supported model choices:

- Letovo: `deepseek-v4-flash`, `gemma-4-26b`;
- OpenAI: `gpt-5-mini`, `gpt-4.1-mini`, `gpt-4o-mini`, `gpt-4.1-nano`.

Letovo models use `LETOVO_API_KEY`; OpenAI models use `OPENAI_API_KEY`. A custom
OpenAI-compatible endpoint can be set with `LLM_URL`.

Each run gets an isolated workspace under `benchmark_runs/`. Public task files
such as `train.csv`, `test.csv`, and public `task.json` are copied there. Hidden
answer files stay in `checker/tasks/...` and are only used by `submit(file)`.

## Run Modes

Run modes define how many model responses the agent gets and how the benchmark
loop transitions between attempts.

- `single-shot`: the model gets exactly one response for the whole task. It may
  return multiple commands in that response, but there is no second model
  attempt. The prompt tells the model to complete the solution end-to-end and
  finish with `submit("submission.csv")`.
- `repeated`: the model gets up to five attempts. After each attempt, the runner
  sends back command results and feedback hints. The prompt tells the model to
  improve or verify the solution on each attempt, avoid premature submission
  when useful checks remain, and submit on the final attempt.
- `fixed-transitions`: the runner moves through three required stages:
  `EDA -> FEATURES -> TRAIN`. The model gets stage-specific instructions.
  `submit(file)` is blocked during `EDA` and `FEATURES`; the `TRAIN` stage must
  create, verify, and submit `submission.csv`.
- `flexible`: the model can choose commands freely until it submits or reaches
  configured limits.

If limits are reached before submit, the runner still performs a forced final
submit: existing `submission.csv` is used when present, otherwise a simple
baseline `submission.csv` is generated.

## Checker

The checker is the lower-level scoring and upload server. It computes metrics
from answer files and submission files. It supports:

- text, CSV, TSV, and JSON inputs;
- row alignment by `id`;
- validation for missing, extra, and duplicated ids;
- classification metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`;
- regression metrics: `mae`, `mse`, `rmse`, `r2`;
- an HTTP server for uploads and task files.

Run the checker server:

```bash
python3 checker/server.py --host 127.0.0.1 --port 8000
```

Open the browser UI:

```text
http://127.0.0.1:8000/
```

The UI provides:

- task selection and metric metadata;
- links for public task files such as `train.csv` and `test.csv`;
- drag-and-drop or file-picker submission upload;
- latest score summary;
- raw JSON response for debugging;
- per-task submission history.

Useful endpoints:

```text
GET  /health
GET  /tasks
GET  /tasks/<task_id>/files/<filename>
GET  /submissions
POST /check
```

Salary task example:

```bash
curl http://127.0.0.1:8000/tasks/salary_prediction/files/train.csv -o train.csv
curl http://127.0.0.1:8000/tasks/salary_prediction/files/test.csv -o test.csv
```

Expected submission format:

```csv
id,salary
5000,32513
5001,31157
```

Submit:

```bash
curl -X POST http://127.0.0.1:8000/check \
  -F task_id=salary_prediction \
  -F file=@submission.csv
```

## Agent Commands

The LLM can operate through this command set:

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

Logging:

- `get_trajectory()`
- `get_hints()`

Submission:

- `submit(file)`

The parser accepts both clean commands:

```text
read_file("checker/README.md")
```

and free-form model responses:

```text
Сначала посмотрю README: read_file("checker/README.md")
```

It also supports JSON commands:

```json
{"command": "read_file", "args": {"path": "checker/README.md"}}
```

Manual agent usage:

```bash
python3 -m agent.cli 'list_files("checker")'
python3 -m agent.cli --model-response 'Сначала посмотрю README: read_file("checker/README.md")'
python3 -m agent.cli 'get_hints()'
python3 -m agent.cli 'submit("checker/tasks/salary_prediction/answers.csv")'
```

## Docker ML Environment

Build the benchmark image once:

```bash
docker build -f Dockerfile.benchmark -t ml-benchmark-runner:latest .
```

`main.py` can run a lower-level benchmark loop with or without Docker. By
default, it creates a per-run container from this prebuilt image and mounts a
fresh workspace containing public task files. Dependencies are not installed
during each benchmark run. The image includes the ML stack from
`requirements-ml.txt`: pandas, NumPy, SciPy, scikit-learn, plotting libraries,
gradient boosting libraries, and Optuna.

Use a different prebuilt image:

```bash
python3 main.py --docker-image my-benchmark-image:latest
```

Run without Docker:

```bash
python3 main.py --no-docker
```

## Trajectory

Every executed command is stored in:

```text
agent_history.txt
```

The file uses JSONL format: one command record per line. It is ignored by git.

The command:

```text
get_trajectory()
```

returns the parsed contents of this file to the LLM.

## Feedback Hints

The agent includes an abstract hint system that helps the LLM improve its
solution without exposing a direct checklist.

Hidden checks are grouped into:

- EDA: target, duplicates, missing values, outliers;
- feature engineering: target leakage and fully duplicated numeric signal;
- model training: task type, saved model file, validation signal, submission file.

Example hint:

```json
{
  "stage": "EDA",
  "message": "Подумай над пропусками и стратегией их обработки."
}
```

During the benchmark loop, feedback is added after the full batch of commands
from the latest model response has executed. The hints are intentionally
abstract: they point the model toward issues such as missing values, target
definition, leakage, validation, or submission format without revealing hidden
answers. The LLM can explicitly request the same hints with:

```text
get_hints()
```

## Tests

Run all tests:

```bash
python3 -m unittest discover agent
python3 -m unittest discover checker
```

Current expected result:

```text
agent:   OK
checker: OK
```

## Repository Structure

```text
web_server.py      browser UI and LLM benchmark runner
main.py            lower-level benchmark loop

agent/
  parser.py        command parsing and command extraction from model responses
  executor.py      command execution and trajectory logging
  hints.py         abstract feedback system
  cli.py           manual command runner

checker/
  metric_checker.py
  server.py
  tasks/
    salary_prediction/
    map/
    wifi/
```
