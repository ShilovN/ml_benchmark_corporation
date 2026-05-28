"""Abstract feedback hints for improving an ML solution."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_ROWS_FOR_ANALYSIS = 5000


@dataclass(frozen=True)
class Hint:
    stage: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"stage": self.stage, "message": self.message}


class HintEngine:
    """Run hidden checks and expose only high-level feedback."""

    def __init__(self, workspace: Path, task_id: str, tasks_dir: Path) -> None:
        self.workspace = workspace.resolve()
        self.task_id = task_id
        self.tasks_dir = self._resolve_path(tasks_dir)

    def build_feedback(self) -> dict[str, Any]:
        task_config = self._load_task_config()
        train_path = self._find_data_file("train.csv", task_config)
        test_path = self._find_data_file("test.csv", task_config)
        target = str(task_config.get("column") or task_config.get("true_column") or "target")
        metric = str(task_config.get("metric") or "")

        train = _read_csv(train_path) if train_path else None
        test = _read_csv(test_path) if test_path else None
        hints: list[Hint] = []

        hints.extend(self._eda_hints(train, test, target))
        hints.extend(self._feature_engineering_hints(train, test, target))
        hints.extend(self._training_hints(train, target, metric))

        return {
            "task_id": self.task_id,
            "hints": [hint.to_dict() for hint in hints],
        }

    def _eda_hints(
        self,
        train: CsvTable | None,
        test: CsvTable | None,
        target: str,
    ) -> list[Hint]:
        hints: list[Hint] = []
        if train is None:
            return [Hint("EDA", "Начни с внимательного просмотра обучающего датасета.")]

        if target not in train.columns:
            hints.append(Hint("EDA", "Подумай, какая колонка является целевой переменной."))

        if test is not None and target in test.columns:
            hints.append(Hint("EDA", "Проверь, не попала ли целевая переменная туда, где ее быть не должно."))

        if _has_duplicate_rows(train):
            hints.append(Hint("EDA", "Подумай, нет ли повторяющихся объектов и как они влияют на обучение."))

        if _has_missing_values(train) or (test is not None and _has_missing_values(test)):
            hints.append(Hint("EDA", "Подумай над пропусками и стратегией их обработки."))

        if target in train.columns and _has_numeric_outliers(train, target):
            hints.append(Hint("EDA", "Посмотри на распределение целевой переменной и возможные крайние значения."))

        return hints

    def _feature_engineering_hints(
        self,
        train: CsvTable | None,
        test: CsvTable | None,
        target: str,
    ) -> list[Hint]:
        hints: list[Hint] = []
        if train is None:
            return hints

        if _has_suspicious_target_feature(self.workspace, target):
            hints.append(Hint("Feature engineering", "Проверь признаки на возможную утечку информации о таргете."))

        correlations = _perfect_numeric_correlations(train, exclude={target})
        if correlations:
            hints.append(Hint("Feature engineering", "Посмотри, нет ли полностью дублирующих друг друга признаков."))

        if test is not None and set(train.columns) == set(test.columns) and target in train.columns:
            hints.append(Hint("Feature engineering", "Сверь, какие признаки доступны на обучении и на тесте."))

        return hints

    def _training_hints(self, train: CsvTable | None, target: str, metric: str) -> list[Hint]:
        hints: list[Hint] = []
        if metric in {"mae", "mse", "rmse", "r2", "r2_score"}:
            expected_task_type = "регрессию"
        elif metric in {"accuracy", "precision", "recall", "f1", "f1_score"}:
            expected_task_type = "классификацию"
        else:
            expected_task_type = "тип ML-задачи"

        if train is not None and target in train.columns:
            numeric_target = _numeric_values(train, target)
            if expected_task_type == "регрессию" and len(numeric_target) < max(3, len(train.rows) // 2):
                hints.append(Hint("Обучение модели", "Проверь, правильно ли выбран тип ML-задачи."))
        else:
            hints.append(Hint("Обучение модели", f"Перед обучением явно определи {expected_task_type}."))

        if not (self.workspace / "model.pt").exists():
            hints.append(Hint("Обучение модели", "Убедись, что обученная модель сохраняется в файл."))

        if not self._has_validation_signal():
            hints.append(Hint("Обучение модели", "Добавь честную валидацию и сравни результат с простым бейзлайном."))

        if not (self.workspace / "submission.csv").exists():
            hints.append(Hint("Обучение модели", "Не забудь сформировать файл отправки с предсказаниями."))

        return hints

    def _has_validation_signal(self) -> bool:
        metric_files = [
            self.workspace / "metrics.json",
            self.workspace / "validation_metrics.json",
            self.workspace / "cv_results.json",
        ]
        for path in metric_files:
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if _contains_metric_value(data):
                return True

        text_markers = ["val_", "valid", "validation", "cross_val", "train_test_split"]
        for path in self.workspace.glob("*.py"):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if any(marker in content for marker in text_markers):
                return True
        return False

    def _load_task_config(self) -> dict[str, Any]:
        config_path = self.tasks_dir / self.task_id / "task.json"
        if not config_path.exists():
            return {}
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _find_data_file(self, filename: str, task_config: dict[str, Any]) -> Path | None:
        workspace_candidate = self.workspace / filename
        if workspace_candidate.exists():
            return workspace_candidate

        task_candidate = self.tasks_dir / self.task_id / filename
        if task_candidate.exists():
            return task_candidate

        for public_file in task_config.get("public_files", []):
            candidate = self.tasks_dir / self.task_id / str(public_file)
            if candidate.name == filename and candidate.exists():
                return candidate
        return None

    def _resolve_path(self, path: Path) -> Path:
        candidate = path if path.is_absolute() else self.workspace / path
        return candidate.resolve()


@dataclass
class CsvTable:
    path: Path
    columns: list[str]
    rows: list[dict[str, str]]


def _read_csv(path: Path) -> CsvTable:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])
        rows = []
        for index, row in enumerate(reader):
            if index >= MAX_ROWS_FOR_ANALYSIS:
                break
            rows.append(row)
    return CsvTable(path=path, columns=columns, rows=rows)


def _has_duplicate_rows(table: CsvTable) -> bool:
    seen: set[tuple[tuple[str, str], ...]] = set()
    for row in table.rows:
        signature = tuple(sorted((key, value) for key, value in row.items()))
        if signature in seen:
            return True
        seen.add(signature)
    return False


def _has_missing_values(table: CsvTable) -> bool:
    return any(value is None or value == "" for row in table.rows for value in row.values())


def _has_numeric_outliers(table: CsvTable, column: str) -> bool:
    values = sorted(_numeric_values(table, column))
    if len(values) < 8:
        return False
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return False
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    return any(value < lower or value > upper for value in values)


def _numeric_values(table: CsvTable, column: str) -> list[float]:
    values: list[float] = []
    for row in table.rows:
        try:
            values.append(float(row.get(column, "")))
        except (TypeError, ValueError):
            continue
    return values


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[int(index)]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def _perfect_numeric_correlations(table: CsvTable, exclude: set[str]) -> list[tuple[str, str]]:
    numeric_columns = [
        column
        for column in table.columns
        if column not in exclude and len(_numeric_values(table, column)) == len(table.rows)
    ]
    pairs: list[tuple[str, str]] = []
    for left_index, left in enumerate(numeric_columns):
        for right in numeric_columns[left_index + 1 :]:
            if _is_perfectly_correlated(_numeric_values(table, left), _numeric_values(table, right)):
                pairs.append((left, right))
    return pairs


def _is_perfectly_correlated(left: list[float], right: list[float]) -> bool:
    if len(left) < 2 or len(left) != len(right):
        return False
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denom_left = math.sqrt(sum(value * value for value in centered_left))
    denom_right = math.sqrt(sum(value * value for value in centered_right))
    if denom_left == 0 or denom_right == 0:
        return False
    corr = sum(a * b for a, b in zip(centered_left, centered_right)) / (denom_left * denom_right)
    return abs(abs(corr) - 1.0) < 1e-12


def _has_suspicious_target_feature(workspace: Path, target: str) -> bool:
    allowed_names = {"train.csv", "answers.csv", "submission.csv"}
    for path in workspace.glob("*.csv"):
        if path.name in allowed_names:
            continue
        try:
            table = _read_csv(path)
        except OSError:
            continue
        if target in table.columns:
            return True
    return False


def _contains_metric_value(data: Any) -> bool:
    if isinstance(data, dict):
        for key, value in data.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("mae", "rmse", "mse", "accuracy", "f1", "score", "metric")):
                if isinstance(value, (int, float)):
                    return True
            if _contains_metric_value(value):
                return True
    if isinstance(data, list):
        return any(_contains_metric_value(item) for item in data)
    return False
