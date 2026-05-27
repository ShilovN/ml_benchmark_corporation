# Metric checker

CLI script that accepts two files and a metric name, then prints the metric value.

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
