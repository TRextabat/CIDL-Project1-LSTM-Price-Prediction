#!/usr/bin/env python3
"""Run full experiment matrix: all models x all assets x all lookbacks x all horizons."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cidl.utils.seed import set_seed
from cidl.utils.device import get_device
from cidl.data.loader import load_asset
from cidl.data.features import compute_features
from cidl.models.registry import create_model
from cidl.training.trainer import Trainer
from cidl.training.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_financial_metrics,
    compute_statistical_tests,
)
from cidl.visualization.plots import (
    plot_price_and_returns,
    plot_feature_correlation,
    plot_walk_forward_splits,
    plot_training_curves,
    plot_predictions_overlay,
    plot_prediction_scatter,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_lookback_comparison,
    plot_horizon_comparison,
    plot_residual_analysis,
    plot_equity_curve,
    plot_attention_weights,
)
from cidl.evaluation.evaluator import ExperimentEvaluator

import pandas as pd


def main():
    # Load config
    config_path = Path(__file__).resolve().parent.parent / "configs" / "experiment.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")
    assert device.type == "cuda", "CUDA is required for training!"

    output_dir = Path(config["output_dir"])
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Pre-load and pre-process all assets
    asset_data = {}
    for asset in config["data"]["assets"]:
        symbol, tf = asset["symbol"], asset["timeframe"]
        asset_key = f"{symbol}_{tf}"
        print(f"\n{'='*60}")
        print(f"Loading {asset_key}...")

        df = load_asset(config["data"]["finagent_data_dir"], symbol, tf)
        print(f"  Raw data: {df.shape[0]} bars, {df.index[0]} to {df.index[-1]}")

        features_df = compute_features(df)
        print(f"  Features: {features_df.shape[0]} rows, {features_df.shape[1]} cols")

        asset_data[asset_key] = {"df": df, "features": features_df}

        # Generate data exploration plots
        plot_price_and_returns(df, symbol, str(figures_dir))
        plot_feature_correlation(features_df, str(figures_dir), symbol)
        print(f"  Data exploration figures saved.")

    # Walk-forward split visualization
    sample_n = list(asset_data.values())[0]["features"].shape[0]
    plot_walk_forward_splits(
        sample_n,
        config["experiments"]["training"]["walk_forward_folds"],
        str(figures_dir),
        purge=config["experiments"]["training"]["purge_bars"],
    )

    # Run experiment matrix
    total_start = time.time()
    experiment_count = 0
    total_experiments = (
        len(config["data"]["assets"])
        * len(config["models"])
        * len(config["experiments"]["lookback_windows"])
        * len(config["experiments"]["prediction_horizons"])
    )

    for asset in config["data"]["assets"]:
        symbol, tf = asset["symbol"], asset["timeframe"]
        asset_key = f"{symbol}_{tf}"
        features_df = asset_data[asset_key]["features"]

        # Collect results per asset for visualization
        asset_results = {}

        for model_cfg in config["models"]:
            for lookback in config["experiments"]["lookback_windows"]:
                for horizon in config["experiments"]["prediction_horizons"]:
                    experiment_count += 1
                    experiment_name = (
                        f"{model_cfg['name']}_{symbol}_{tf}_lb{lookback}_h{horizon}"
                    )
                    print(f"\n--- [{experiment_count}/{total_experiments}] {experiment_name} ---")

                    exp_start = time.time()

                    # Run walk-forward training
                    trainer = Trainer(config["experiments"]["training"], device)
                    result = trainer.run_walk_forward(
                        features_df, model_cfg, lookback, horizon
                    )

                    # Add metadata
                    result["experiment_name"] = experiment_name
                    result["asset"] = asset_key
                    result["elapsed_seconds"] = round(time.time() - exp_start, 1)

                    # Save individual result
                    result_path = results_dir / f"{experiment_name}.json"
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2, default=str)

                    all_results.append(result)
                    asset_results[experiment_name] = result

                    metrics = result.get("metrics", {})
                    print(f"  RMSE: {metrics.get('rmse', 'N/A'):.6f}")
                    print(f"  Dir Acc: {metrics.get('direction_accuracy', 'N/A'):.4f}")
                    print(f"  Sharpe: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
                    print(f"  Time: {result['elapsed_seconds']}s")

        # Per-asset visualizations
        if asset_results:
            # Training curves
            plot_training_curves(asset_results, str(figures_dir))

            # ROC curves
            plot_roc_curves(asset_results, str(figures_dir), asset_key)

            # Confusion matrices
            plot_confusion_matrices(asset_results, str(figures_dir), asset_key)

            # Per-model predictions overlay and scatter
            for exp_name, result in asset_results.items():
                preds = np.array(result.get("predictions", []))
                actuals = np.array(result.get("actuals", []))
                if len(preds) > 0:
                    model_name = result.get("model_name", "unknown")
                    lb = result.get("lookback", 0)
                    h = result.get("horizon", 0)
                    label = f"{model_name}_lb{lb}_h{h}"

                    plot_predictions_overlay(
                        actuals, preds, None, label, asset_key, str(figures_dir)
                    )
                    plot_prediction_scatter(actuals, preds, f"{label}_{asset_key}", str(figures_dir))
                    plot_residual_analysis(actuals, preds, str(figures_dir), f"{label}_{asset_key}")

                    # Equity curve
                    strategy_rets = np.where(preds > 0, actuals, -actuals)
                    plot_equity_curve(strategy_rets, actuals, str(figures_dir), f"{label}_{asset_key}")

    # Cross-experiment visualizations
    if all_results:
        # Build metrics DataFrame for comparison plots
        rows = []
        for r in all_results:
            row = {
                "model": r.get("model_name", "?"),
                "asset": r.get("asset", "?"),
                "lookback": r.get("lookback", 0),
                "horizon": r.get("horizon", 0),
                "param_count": r.get("param_count", 0),
            }
            row.update(r.get("metrics", {}))
            rows.append(row)
        metrics_df = pd.DataFrame(rows)

        plot_model_comparison(metrics_df, str(figures_dir))
        plot_lookback_comparison(metrics_df, str(figures_dir))
        plot_horizon_comparison(metrics_df, str(figures_dir))

    # Save summary
    summary_path = results_dir / "summary.json"
    # Strip large arrays from summary for readability
    summary = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k not in ["predictions", "actuals", "directions_pred", "directions_actual"]}
        summary.append(s)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All {len(all_results)} experiments complete in {total_elapsed:.0f}s!")
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
