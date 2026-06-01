import tempfile
import unittest
from pathlib import Path

from metric_checker import compute_metric


class MetricCheckerTest(unittest.TestCase):
    def test_accuracy_from_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            true_path = Path(tmp_dir) / "true.txt"
            pred_path = Path(tmp_dir) / "pred.txt"
            true_path.write_text("cat\ndog\ncat\n", encoding="utf-8")
            pred_path.write_text("cat\ncat\ncat\n", encoding="utf-8")

            result = compute_metric(true_path, pred_path, "accuracy")

        self.assertAlmostEqual(result, 2 / 3)

    def test_rmse_from_csv_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            true_path = Path(tmp_dir) / "true.csv"
            pred_path = Path(tmp_dir) / "pred.csv"
            true_path.write_text("value\n1\n2\n3\n", encoding="utf-8")
            pred_path.write_text("value\n1\n2\n5\n", encoding="utf-8")

            result = compute_metric(true_path, pred_path, "rmse", column="value")

        self.assertAlmostEqual(result, (4 / 3) ** 0.5)

    def test_accuracy_from_csv_aligned_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            true_path = Path(tmp_dir) / "true.csv"
            pred_path = Path(tmp_dir) / "pred.csv"
            true_path.write_text("id,label\n1,cat\n2,dog\n3,cat\n", encoding="utf-8")
            pred_path.write_text("id,label\n3,cat\n1,cat\n2,cat\n", encoding="utf-8")

            result = compute_metric(
                true_path,
                pred_path,
                "accuracy",
                column="label",
                id_column="id",
            )

        self.assertAlmostEqual(result, 2 / 3)

    def test_roc_auc_from_csv_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            true_path = Path(tmp_dir) / "true.csv"
            pred_path = Path(tmp_dir) / "pred.csv"
            true_path.write_text("label\n0\n0\n1\n1\n", encoding="utf-8")
            pred_path.write_text("score\n0.1\n0.4\n0.35\n0.8\n", encoding="utf-8")

            result = compute_metric(
                true_path,
                pred_path,
                "roc_auc",
                true_column="label",
                pred_column="score",
            )

        self.assertAlmostEqual(result, 0.75)

    def test_roc_auc_handles_tied_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            true_path = Path(tmp_dir) / "true.txt"
            pred_path = Path(tmp_dir) / "pred.txt"
            true_path.write_text("0\n1\n", encoding="utf-8")
            pred_path.write_text("0.5\n0.5\n", encoding="utf-8")

            result = compute_metric(true_path, pred_path, "roc_auc")

        self.assertAlmostEqual(result, 0.5)

    def test_roc_auc_requires_binary_true_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            true_path = Path(tmp_dir) / "true.txt"
            pred_path = Path(tmp_dir) / "pred.txt"
            true_path.write_text("0\n1\n2\n", encoding="utf-8")
            pred_path.write_text("0.1\n0.5\n0.9\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exactly two classes"):
                compute_metric(true_path, pred_path, "roc_auc")

    def test_missing_submission_id_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            true_path = Path(tmp_dir) / "true.csv"
            pred_path = Path(tmp_dir) / "pred.csv"
            true_path.write_text("id,label\n1,cat\n2,dog\n", encoding="utf-8")
            pred_path.write_text("id,label\n1,cat\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing ids"):
                compute_metric(
                    true_path,
                    pred_path,
                    "accuracy",
                    column="label",
                    id_column="id",
                )

    def test_different_lengths_raise_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            true_path = Path(tmp_dir) / "true.txt"
            pred_path = Path(tmp_dir) / "pred.txt"
            true_path.write_text("1\n2\n", encoding="utf-8")
            pred_path.write_text("1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "different numbers"):
                compute_metric(true_path, pred_path, "mae")


if __name__ == "__main__":
    unittest.main()
