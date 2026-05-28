# Metric checker

CLI script that accepts two files and a metric name, then prints the metric value.
It also includes a small HTTP server for checking submitted solution files against
task configs stored on the server.

## Usage

```bash
python3 checker/metric_checker.py TRUE_FILE PRED_FILE METRIC
```

Examples:

```bash
python3 checker/metric_checker.py y_true.txt y_pred.txt accuracy
python3 checker/metric_checker.py y_true.csv y_pred.csv f1 --column label
python3 checker/metric_checker.py y_true.json y_pred.json rmse
```

Supported input formats:

- plain text: one value per line, or whitespace/comma separated values;
- CSV/TSV: one column, or a named column via `--column`;
- JSON: an array, an object with a single array field, or an array of objects with `--column`.

Supported metrics:

- classification: `accuracy`, `precision`, `recall`, `f1`;
- regression: `mae`, `mse`, `rmse`, `r2`.

`precision`, `recall`, and `f1` are macro-averaged for multiclass labels.

## Server

Run:

```bash
python3 checker/server.py --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health` - server health check;
- `GET /tasks` - list public task metadata;
- `POST /check` - upload a solution file and get the metric value.

Example submission:

```bash
curl -X POST http://127.0.0.1:8000/check \
  -F task_id=sample_accuracy \
  -F file=@submission.txt
```

Successful response:

```json
{"task_id": "sample_accuracy", "metric": "accuracy", "value": 0.6666666666666666}
```

Task configs live in `checker/tasks/<task_id>/task.json`. The answer file stays
on the server and is not exposed by the API.

Minimal task config:

```json
{
  "id": "sample_accuracy",
  "name": "Sample classification task",
  "metric": "accuracy",
  "answer_file": "answers.txt"
}
```
