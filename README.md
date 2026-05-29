# ML Benchmark Corporation

This repository contains a prototype of an AutoML-style benchmark environment.
The project has two main parts:

- `checker/` - task server and metric checker for submitted solutions;
- `agent/` - command parser, executor, trajectory logging, and feedback hints for
  an LLM agent.

The current demo task is salary prediction: predict `salary` for vacancies.
The metric is `mae`.

## Project Mechanics

The intended loop is:

1. An LLM produces a command or a short response containing one command.
2. `agent` extracts, parses, validates, and executes the command.
3. The executed command is appended to `agent_history.txt`.
4. The agent receives command output plus abstract feedback hints.
5. The agent improves the solution and eventually calls `submit(file)`.
6. `checker` validates the submission and returns the metric.

## Checker

The checker computes metrics from answer files and submission files. It supports:

- text, CSV, TSV, and JSON inputs;
- row alignment by `id`;
- validation for missing, extra, and duplicated ids;
- classification metrics: `accuracy`, `precision`, `recall`, `f1`;
- regression metrics: `mae`, `mse`, `rmse`, `r2`;
- an HTTP server for uploads and task files.

Run the server:

```bash
python3 checker/server.py --host 127.0.0.1 --port 8000
```

Open the browser UI:

```text
http://127.0.0.1:8000/
```

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
- `get_hints()`

Logging:

- `get_trajectory()`

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

By default, `main.py` creates a per-run container from this prebuilt image and
mounts a fresh workspace containing the public task files. Dependencies are not
installed during each benchmark run. The image includes OpenMP runtime plus the
ML stack from `requirements-ml.txt`: pandas, NumPy, SciPy, scikit-learn,
plotting libraries, gradient boosting libraries, and Optuna.

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

During the benchmark loop, feedback is added once after the full batch of
commands from the latest model response has executed. The LLM can explicitly
request the same hints with:

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
```
