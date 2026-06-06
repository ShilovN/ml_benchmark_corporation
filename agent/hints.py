"""Abstract feedback hints for improving an ML solution."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_ROWS_FOR_ANALYSIS = 5000
HINT_STAGE_REPEAT_THRESHOLD = 3


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
        target = str(
            task_config.get("column")
            or task_config.get("true_column")
            or task_config.get("pred_column")
            or "target"
        )
        metric = str(task_config.get("metric") or "")
        history = self._load_history()
        python_sources = self._python_sources()

        train = _read_csv(train_path) if train_path else None
        test = _read_csv(test_path) if test_path else None
        stage_hints = [
            self._eda_hints(train, test, target, history, python_sources),
            self._feature_engineering_hints(train, test, target, python_sources),
            self._training_hints(train, target, metric),
        ]
        hints = self._select_hints(stage_hints)

        return {
            "task_id": self.task_id,
            "hints": [hint.to_dict() for hint in hints],
        }

    def _eda_hints(
        self,
        train: CsvTable | None,
        test: CsvTable | None,
        target: str,
        history: list[dict[str, Any]],
        python_sources: list[str],
    ) -> list[Hint]:
        hints: list[Hint] = []
        if train is None:
            return [Hint("EDA", "Сначала стоит внимательно посмотреть на обучающий датасет.")]

        if not _has_dataset_summary_signal(history, python_sources, self.workspace):
            hints.append(Hint("EDA", "Подумай над базовой сводкой по колонкам и типам признаков."))

        if target not in train.columns:
            hints.append(Hint("EDA", "Подумай, какая колонка является целевой переменной."))

        if test is not None and target in test.columns:
            hints.append(Hint("EDA", "Подумай, не попала ли целевая переменная туда, где ее быть не должно."))

        if _has_duplicate_rows(train):
            hints.append(Hint("EDA", "Подумай, нет ли повторяющихся объектов и как они влияют на обучение."))

        return hints

    def _feature_engineering_hints(
        self,
        train: CsvTable | None,
        test: CsvTable | None,
        target: str,
        python_sources: list[str],
    ) -> list[Hint]:
        hints: list[Hint] = []
        if train is None:
            return hints

        if _has_suspicious_target_feature(self.workspace, target):
            hints.append(Hint("Feature engineering", "Подумай, нет ли среди признаков возможной утечки информации о таргете."))

        correlations = _perfect_numeric_correlations(train, exclude={target})
        if correlations:
            hints.append(Hint("Feature engineering", "Подумай, нет ли полностью дублирующих друг друга признаков."))

        if _has_datetime_like_features(train, exclude={target}) and not _has_datetime_processing_signal(python_sources):
            hints.append(Hint("Feature engineering", "Подумай, можно ли извлечь полезные признаки из даты и времени."))

        if test is not None and _needs_shared_preprocessing(train, test, target):
            if not _has_shared_preprocessing_signal(python_sources):
                hints.append(Hint("Feature engineering", "Подумай, одинаково ли применяется предобработка к train и test и нет ли здесь утечки."))

        return hints

    def _training_hints(self, train: CsvTable | None, target: str, metric: str) -> list[Hint]:
        hints: list[Hint] = []
        if metric in {"mae", "mse", "rmse", "r2", "r2_score"}:
            expected_task_type = "регрессию"
        elif metric in {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f1_score",
            "roc_auc",
            "roc_auc_score",
        }:
            expected_task_type = "классификацию"
        else:
            expected_task_type = "тип ML-задачи"

        if train is not None and target in train.columns:
            numeric_target = _numeric_values(train, target)
            if expected_task_type == "регрессию" and len(numeric_target) < max(3, len(train.rows) // 2):
                hints.append(Hint("Обучение модели", "Подумай, правильно ли выбран тип ML-задачи."))
        else:
            hints.append(Hint("Обучение модели", f"Перед обучением стоит явно определить {expected_task_type}."))

        if not (self.workspace / "submission.csv").exists():
            hints.append(Hint("Обучение модели", "Подумай над формированием файла отправки с предсказаниями."))

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

    def _load_history(self) -> list[dict[str, Any]]:
        history_path = self.workspace / "agent_history.txt"
        if not history_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with history_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    records.append(data)
        return records

    def _python_sources(self) -> list[str]:
        sources: list[str] = []
        for path in self.workspace.glob("*.py"):
            try:
                sources.append(path.read_text(encoding="utf-8", errors="ignore").lower())
            except OSError:
                continue
        return sources

    def _select_hints(self, stage_hints: list[list[Hint]]) -> list[Hint]:
        primary_index = next((index for index, items in enumerate(stage_hints) if items), None)
        state = self._load_hint_state()
        if primary_index is None:
            self._save_hint_state({"task_id": self.task_id, "stage": None, "repeats": 0})
            return []

        primary_stage = stage_hints[primary_index][0].stage
        repeats = 1
        if state.get("task_id") == self.task_id and state.get("stage") == primary_stage:
            repeats = int(state.get("repeats") or 0) + 1

        self._save_hint_state({"task_id": self.task_id, "stage": primary_stage, "repeats": repeats})

        hints = list(stage_hints[primary_index])
        if repeats < HINT_STAGE_REPEAT_THRESHOLD:
            return hints

        next_index = next(
            (index for index in range(primary_index + 1, len(stage_hints)) if stage_hints[index]),
            None,
        )
        if next_index is not None:
            hints.extend(stage_hints[next_index])
        return hints

    def _load_hint_state(self) -> dict[str, Any]:
        path = self.workspace / ".hint_state.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _save_hint_state(self, state: dict[str, Any]) -> None:
        path = self.workspace / ".hint_state.json"
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


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


def _has_dataset_inspection_signal(
    history: list[dict[str, Any]],
    train_name: str,
    test_name: str | None,
) -> bool:
    required = {train_name}
    if test_name:
        required.add(test_name)
    seen: set[str] = set()
    for record in history:
        if record.get("status") != "ok":
            continue
        command = str(record.get("command") or "")
        if command not in {"read_file", "load_dataset"}:
            continue
        args = record.get("args")
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if not isinstance(path, str):
            continue
        name = Path(path).name
        if name in required:
            seen.add(name)
    return seen >= required


def _has_dataset_summary_signal(
    history: list[dict[str, Any]],
    python_sources: list[str],
    workspace: Path,
) -> bool:
    for record in history:
        if record.get("status") == "ok" and str(record.get("command") or "") in {"show_dataset_info", "show_sample_rows"}:
            return True

    keywords = [
        "describe(",
        "info(",
        "isnull(",
        "missing_by_column",
        "value_counts(",
        "nunique(",
        "dtypes",
    ]
    if any(any(keyword in source for keyword in keywords) for source in python_sources):
        return True

    for name in ("eda.json", "eda_report.json", "feature_report.json"):
        path = workspace / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and any(key in data for key in ("columns", "missing", "dtypes", "nunique")):
            return True
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


def _has_categorical_features(table: CsvTable, exclude: set[str]) -> bool:
    for column in table.columns:
        if column in exclude:
            continue
        values = [row.get(column, "").strip() for row in table.rows if row.get(column, "").strip()]
        if not values:
            continue
        if any(not _is_float_like(value) for value in values):
            return True
    return False


def _has_categorical_processing_signal(python_sources: list[str]) -> bool:
    keywords = [
        "get_dummies",
        "onehotencoder",
        "ordinalencoder",
        "labelencoder",
        "targetencoder",
        "catboost",
        "xgboost",
        "lightgbm",
        "category_encoders",
        "columntransformer",
        "dictvectorizer",
        "countvectorizer",
        "tfidfvectorizer",
    ]
    return any(any(keyword in source for keyword in keywords) for source in python_sources)


def _has_datetime_like_features(table: CsvTable, exclude: set[str]) -> bool:
    tokens = ("date", "time", "timestamp", "datetime")
    return any(column not in exclude and any(token in column.lower() for token in tokens) for column in table.columns)


def _has_datetime_processing_signal(python_sources: list[str]) -> bool:
    keywords = ["to_datetime", ".dt.", ".dt[", "dayofweek", ".year", ".month", ".hour", "strftime("]
    return any(any(keyword in source for keyword in keywords) for source in python_sources)


def _id_like_columns(table: CsvTable, exclude: set[str]) -> list[str]:
    candidates: list[str] = []
    row_count = len(table.rows)
    if row_count < 2:
        return candidates
    for column in table.columns:
        if column in exclude:
            continue
        values = [row.get(column, "").strip() for row in table.rows if row.get(column, "").strip()]
        if len(values) < max(2, row_count // 2):
            continue
        unique_ratio = len(set(values)) / len(values)
        if "id" in column.lower() or (row_count >= 20 and unique_ratio > 0.95):
            candidates.append(column)
    return candidates


def _has_id_handling_signal(python_sources: list[str], columns: list[str]) -> bool:
    for source in python_sources:
        if "drop(" not in source and "dropna(" not in source:
            continue
        if any(column.lower() in source for column in columns):
            return True
    return False


def _needs_shared_preprocessing(train: CsvTable, test: CsvTable, target: str) -> bool:
    feature_columns = [column for column in train.columns if column != target]
    if set(feature_columns) != set(test.columns):
        return False
    return _has_categorical_features(train, exclude={target}) or _has_datetime_like_features(train, exclude={target})


def _has_shared_preprocessing_signal(python_sources: list[str]) -> bool:
    keywords = [
        "pipeline(",
        "make_pipeline(",
        "columntransformer",
        "preprocessor",
        "fit_transform(",
        ".transform(",
    ]
    return any(any(keyword in source for keyword in keywords) for source in python_sources)


def _is_float_like(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


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
