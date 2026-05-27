#!/usr/bin/env python3
"""Compute a metric for two files with expected and predicted values."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Sequence, Union


Number = float
Value = Union[str, Number]


CLASSIFICATION_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_score",
}

REGRESSION_METRICS = {
    "mae",
    "mean_absolute_error",
    "mse",
    "mean_squared_error",
    "rmse",
    "root_mean_squared_error",
    "r2",
    "r2_score",
}


def load_values(path: Path, column: str | None = None) -> list[Value]:
    """Load a one-dimensional sequence from json, csv, tsv, or plain text."""
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        values = _load_json(path, column)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        values = _load_delimited(path, delimiter, column)
    else:
        values = _load_text(path)

    if not values:
        raise ValueError(f"No values found in {path}")
    return values


def _load_json(path: Path, column: str | None) -> list[Value]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        if column:
            data = data[column]
        elif len(data) == 1:
            data = next(iter(data.values()))
        else:
            raise ValueError(
                f"{path} contains a JSON object. Pass --column with one of: "
                f"{', '.join(map(str, data.keys()))}"
            )

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array or an object with an array field")

    if data and isinstance(data[0], dict):
        if not column:
            raise ValueError(f"{path} contains objects. Pass --column")
        return [_parse_value(row[column]) for row in data]

    return [_parse_value(item) for item in data]


def _load_delimited(path: Path, delimiter: str, column: str | None) -> list[Value]:
    with path.open("r", encoding="utf-8", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        except csv.Error:
            has_header = False

        if column:
            reader = csv.DictReader(file, delimiter=delimiter)
            if not reader.fieldnames:
                return []
            if column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{column}' not found in {path}. "
                    f"Available columns: {', '.join(reader.fieldnames)}"
                )
            return [_parse_value(row[column]) for row in reader]

        if has_header:
            reader = csv.DictReader(file, delimiter=delimiter)
            if not reader.fieldnames:
                return []
            selected_column = reader.fieldnames[-1]
            return [_parse_value(row[selected_column]) for row in reader]

        reader = csv.reader(file, delimiter=delimiter)
        values: list[Value] = []
        for row in reader:
            if not row:
                continue
            if len(row) != 1:
                raise ValueError(f"{path} has multiple columns. Pass --column or add a header")
            values.append(_parse_value(row[0]))
        return values


def _load_text(path: Path) -> list[Value]:
    values: list[Value] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            values.extend(_parse_value(part) for part in line.replace(",", " ").split())
    return values


def _parse_value(value: object) -> Value:
    if isinstance(value, (int, float)):
        return float(value)

    if value is None:
        raise ValueError("Missing value found")

    text = str(value).strip()
    if not text:
        raise ValueError("Empty value found")

    try:
        return float(text)
    except ValueError:
        return text


def ensure_same_length(y_true: Sequence[Value], y_pred: Sequence[Value]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Files contain different numbers of values: {len(y_true)} and {len(y_pred)}"
        )


def numeric_values(values: Sequence[Value], name: str) -> list[Number]:
    converted: list[Number] = []
    for value in values:
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as exc:
                raise ValueError(f"{name} contains a non-numeric value: {value!r}") from exc
        converted.append(float(value))
    return converted


def accuracy(y_true: Sequence[Value], y_pred: Sequence[Value]) -> float:
    return sum(true == pred for true, pred in zip(y_true, y_pred)) / len(y_true)


def precision(y_true: Sequence[Value], y_pred: Sequence[Value]) -> float:
    return _macro_classification_metric(y_true, y_pred, "precision")


def recall(y_true: Sequence[Value], y_pred: Sequence[Value]) -> float:
    return _macro_classification_metric(y_true, y_pred, "recall")


def f1(y_true: Sequence[Value], y_pred: Sequence[Value]) -> float:
    return _macro_classification_metric(y_true, y_pred, "f1")


def _macro_classification_metric(
    y_true: Sequence[Value], y_pred: Sequence[Value], metric: str
) -> float:
    labels = sorted(set(y_true) | set(y_pred), key=str)
    scores: list[float] = []

    for label in labels:
        tp = sum(true == label and pred == label for true, pred in zip(y_true, y_pred))
        fp = sum(true != label and pred == label for true, pred in zip(y_true, y_pred))
        fn = sum(true == label and pred != label for true, pred in zip(y_true, y_pred))

        label_precision = _safe_divide(tp, tp + fp)
        label_recall = _safe_divide(tp, tp + fn)

        if metric == "precision":
            scores.append(label_precision)
        elif metric == "recall":
            scores.append(label_recall)
        else:
            scores.append(_safe_divide(2 * label_precision * label_recall, label_precision + label_recall))

    return sum(scores) / len(scores)


def mae(y_true: Sequence[Value], y_pred: Sequence[Value]) -> float:
    true = numeric_values(y_true, "First file")
    pred = numeric_values(y_pred, "Second file")
    return sum(abs(a - b) for a, b in zip(true, pred)) / len(true)


def mse(y_true: Sequence[Value], y_pred: Sequence[Value]) -> float:
    true = numeric_values(y_true, "First file")
    pred = numeric_values(y_pred, "Second file")
    return sum((a - b) ** 2 for a, b in zip(true, pred)) / len(true)


def rmse(y_true: Sequence[Value], y_pred: Sequence[Value]) -> float:
    return math.sqrt(mse(y_true, y_pred))


def r2(y_true: Sequence[Value], y_pred: Sequence[Value]) -> float:
    true = numeric_values(y_true, "First file")
    pred = numeric_values(y_pred, "Second file")
    mean_true = sum(true) / len(true)
    total_sum_squares = sum((value - mean_true) ** 2 for value in true)
    if total_sum_squares == 0:
        raise ValueError("R2 is undefined when all true values are equal")
    residual_sum_squares = sum((a - b) ** 2 for a, b in zip(true, pred))
    return 1 - residual_sum_squares / total_sum_squares


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def metric_registry() -> dict[str, Callable[[Sequence[Value], Sequence[Value]], float]]:
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_score": f1,
        "mae": mae,
        "mean_absolute_error": mae,
        "mse": mse,
        "mean_squared_error": mse,
        "rmse": rmse,
        "root_mean_squared_error": rmse,
        "r2": r2,
        "r2_score": r2,
    }


def normalize_metric_name(metric: str) -> str:
    return metric.strip().lower().replace("-", "_")


def compute_metric(
    true_path: Path,
    pred_path: Path,
    metric_name: str,
    column: str | None = None,
    true_column: str | None = None,
    pred_column: str | None = None,
) -> float:
    registry = metric_registry()
    normalized_metric = normalize_metric_name(metric_name)
    if normalized_metric not in registry:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Available metrics: "
            f"{', '.join(sorted(registry))}"
        )

    y_true = load_values(true_path, true_column or column)
    y_pred = load_values(pred_path, pred_column or column)
    ensure_same_length(y_true, y_pred)
    return registry[normalized_metric](y_true, y_pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a metric from two files: true values and predicted values."
    )
    parser.add_argument("true_path", type=Path, help="Path to the file with true values")
    parser.add_argument("pred_path", type=Path, help="Path to the file with predicted values")
    parser.add_argument("metric", help="Metric name")
    parser.add_argument(
        "--column",
        help="Column name for both files when reading CSV/TSV/JSON objects",
    )
    parser.add_argument("--true-column", help="Column name for the true-values file")
    parser.add_argument("--pred-column", help="Column name for the predicted-values file")
    parser.add_argument(
        "--digits",
        type=int,
        default=10,
        help="Number of significant digits to print",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = compute_metric(
            args.true_path,
            args.pred_path,
            args.metric,
            column=args.column,
            true_column=args.true_column,
            pred_column=args.pred_column,
        )
    except (KeyError, ValueError, OSError, csv.Error, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"{result:.{args.digits}g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
