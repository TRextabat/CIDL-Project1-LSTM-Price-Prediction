#!/usr/bin/env python3
"""Collect experiment results and generate PDF report."""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cidl.visualization.report import ReportGenerator
from cidl.evaluation.evaluator import ExperimentEvaluator


def main():
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "experiment.yaml"
    results_dir = project_root / "outputs" / "results"
    figures_dir = project_root / "outputs" / "figures"
    report_dir = project_root / "outputs" / "report"

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load results
    evaluator = ExperimentEvaluator(str(results_dir))
    results = evaluator.load_results()
    print(f"Loaded {len(results)} experiment results")

    if not results:
        print("No results found. Run experiments first.")
        return

    # Print summary table
    summary_df = evaluator.summary_table()
    print("\nResults Summary:")
    print(summary_df.to_string(index=False))

    # Best models
    best = evaluator.best_models()
    print("\nBest Models by Asset (Sharpe):")
    print(best.to_string(index=False))

    # Model comparison
    comparison = evaluator.model_comparison()
    print("\nModel Comparison (averages):")
    print(comparison.to_string(index=False))

    # Statistical tests
    stat_tests = evaluator.run_statistical_tests()
    if stat_tests:
        print("\nStatistical Tests:")
        for key, tests in stat_tests.items():
            print(f"  {key}:")
            if "adf_test" in tests:
                adf = tests["adf_test"]
                print(f"    ADF: stat={adf.get('statistic', 'N/A')}, p={adf.get('p_value', 'N/A')}, stationary={adf.get('stationary')}")
            if "diebold_mariano" in tests:
                for pair, dm in tests["diebold_mariano"].items():
                    print(f"    DM {pair}: stat={dm['statistic']:.3f}, p={dm['p_value']:.4f}, better={dm['better_model']}")

    # Generate PDF report
    print("\nGenerating PDF report...")
    generator = ReportGenerator(str(report_dir))
    pdf_path = generator.generate(results, str(figures_dir), config)
    print(f"Report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
