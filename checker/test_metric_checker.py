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
