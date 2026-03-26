"""Aggregate results across experiments and produce summary tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cidl.training.metrics import compute_statistical_tests


class ExperimentEvaluator:
    """Load and aggregate experiment results from JSON files.

    Reads all result JSONs from the results directory and produces
    summary DataFrames for comparison across models, assets, lookbacks,
    and horizons.
    """

    def __init__(self, results_dir: str = "outputs/results"):
        self.results_dir = Path(results_dir)
        self.results: list[dict] = []

    def load_results(self) -> list[dict]:
        """Load all experiment result JSON files."""
        self.results = []
        for path in sorted(self.results_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                if "error" not in data:
                    # Extract asset info from filename
                    parts = path.stem.split("_")
                    # Format: modelname_symbol_tf_lbN_hN
                    if len(parts) >= 5:
                        data["_asset"] = f"{parts[-4]}_{parts[-3]}"
                    self.results.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
        return self.results

    def summary_table(self) -> pd.DataFrame:
        """Create summary DataFrame of all experiments.

        Returns:
            DataFrame with columns: model, asset, lookback, horizon,
            and all metric columns.
        """
        if not self.results:
            self.load_results()

        rows = []
        for r in self.results:
            row = {
                "model": r.get("model_name", "unknown"),
                "asset": r.get("_asset", "unknown"),
                "lookback": r.get("lookback", 0),
                "horizon": r.get("horizon", 0),
                "param_count": r.get("param_count", 0),
            }
            if "metrics" in r:
                row.update(r["metrics"])
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def best_models(self) -> pd.DataFrame:
        """Find best model for each asset by Sharpe ratio."""
        df = self.summary_table()
        if df.empty:
            return df
        idx = df.groupby("asset")["sharpe_ratio"].idxmax()
        return df.loc[idx].reset_index(drop=True)

    def model_comparison(self) -> pd.DataFrame:
        """Average metrics across all assets/configs per model."""
        df = self.summary_table()
        if df.empty:
            return df
        metric_cols = [
            c for c in df.columns
            if c not in ["model", "asset", "lookback", "horizon", "param_count"]
        ]
        return df.groupby("model")[metric_cols].mean().reset_index()

    def run_statistical_tests(self) -> dict:
        """Run cross-model statistical tests.

        Groups results by (asset, lookback, horizon) and runs
        Diebold-Mariano and Pesaran-Timmermann tests.
        """
        if not self.results:
            self.load_results()

        # Group by experiment setting
        groups = {}
        for r in self.results:
            key = (r.get("_asset", ""), r.get("lookback", 0), r.get("horizon", 0))
            if key not in groups:
                groups[key] = {}
            model_name = r.get("model_name", "unknown")
            if "actuals" in r and "predictions" in r:
                groups[key][model_name] = {
                    "actuals": np.array(r["actuals"]),
                    "predictions": np.array(r["predictions"]),
                }

        all_tests = {}
        for key, models in groups.items():
            if len(models) < 2:
                continue
            asset, lb, h = key
            test_key = f"{asset}_lb{lb}_h{h}"

            # Get common actuals (should be same across models for same setting)
            first_model = list(models.values())[0]
            y_true = first_model["actuals"]
            predictions_dict = {
                name: data["predictions"] for name, data in models.items()
            }

            # Only compare if lengths match
            lengths = [len(p) for p in predictions_dict.values()]
            if len(set(lengths)) == 1 and lengths[0] == len(y_true):
                all_tests[test_key] = compute_statistical_tests(
                    y_true, predictions_dict
                )

        return all_tests
