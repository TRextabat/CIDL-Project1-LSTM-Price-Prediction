"""Metrics computation for regression, classification, financial, and statistical tests."""

import warnings

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics.

    Returns:
        Dict with mae, rmse, mape, r2.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE: handle zeros by adding small epsilon
    nonzero_mask = np.abs(y_true) > 1e-10
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
    else:
        mape = float("inf")

    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


def compute_classification_metrics(
    y_true_dir: np.ndarray,
    y_pred_dir: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> dict:
    """Compute directional classification metrics.

    Args:
        y_true_dir: True direction labels (0 or 1).
        y_pred_dir: Predicted direction labels (0 or 1).
        y_pred_proba: Predicted probabilities for ROC-AUC (optional).

    Returns:
        Dict with direction_accuracy, precision, recall, f1, roc_auc.
    """
    y_true_dir = np.asarray(y_true_dir).flatten().astype(int)
    y_pred_dir = np.asarray(y_pred_dir).flatten().astype(int)

    result = {
        "direction_accuracy": float(accuracy_score(y_true_dir, y_pred_dir)),
        "precision": float(precision_score(y_true_dir, y_pred_dir, zero_division=0)),
        "recall": float(recall_score(y_true_dir, y_pred_dir, zero_division=0)),
        "f1": float(f1_score(y_true_dir, y_pred_dir, zero_division=0)),
    }

    if y_pred_proba is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true_dir, y_pred_proba))
        except ValueError:
            result["roc_auc"] = None
    else:
        result["roc_auc"] = None

    return result


def compute_financial_metrics(
    returns_pred_direction: np.ndarray,
    actual_returns: np.ndarray,
) -> dict:
    """Compute financial metrics from a directional trading strategy.

    Strategy: go long when predicted direction is positive, short otherwise.

    Args:
        returns_pred_direction: Predicted direction signs (+1 or -1, or probabilities).
        actual_returns: Actual log-returns.

    Returns:
        Dict with sharpe_ratio, sortino_ratio, max_drawdown, profit_factor,
        hit_ratio, cumulative_return.
    """
    returns_pred_direction = np.asarray(returns_pred_direction).flatten()
    actual_returns = np.asarray(actual_returns).flatten()

    # Strategy returns: go long if pred > 0, short otherwise
    strategy_returns = np.where(returns_pred_direction > 0, actual_returns, -actual_returns)

    # Sharpe ratio (annualized with sqrt(252))
    if np.std(strategy_returns) > 1e-10:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino ratio (downside deviation)
    downside = strategy_returns[strategy_returns < 0]
    if len(downside) > 0 and np.std(downside) > 1e-10:
        sortino = np.mean(strategy_returns) / np.std(downside) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    cumulative = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Profit factor
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = float(gains / (losses + 1e-8))

    # Hit ratio
    hit_ratio = float(np.mean(strategy_returns > 0))

    # Cumulative return
    cum_return = float(cumulative[-1]) if len(cumulative) > 0 else 0.0

    return {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "profit_factor": float(profit_factor),
        "hit_ratio": float(hit_ratio),
        "cumulative_return": float(cum_return),
    }


def compute_statistical_tests(
    y_true: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
) -> dict:
    """Run statistical tests on predictions.

    Tests:
    - ADF (Augmented Dickey-Fuller) on returns: test stationarity
    - Ljung-Box on residuals: test autocorrelation
    - Pesaran-Timmermann: test directional accuracy significance (z-test vs 0.5)
    - Diebold-Mariano: pairwise forecast comparison (if multiple models)

    Args:
        y_true: True values.
        predictions_dict: Dict of {model_name: predictions}.

    Returns:
        Dict with test results.
    """
    y_true = np.asarray(y_true).flatten()
    results = {}

    # ADF test on actual returns
    try:
        adf_stat, adf_pvalue, *_ = adfuller(y_true, maxlag=20)
        results["adf_test"] = {
            "statistic": float(adf_stat),
            "p_value": float(adf_pvalue),
            "stationary": bool(adf_pvalue < 0.05),
        }
    except Exception:
        results["adf_test"] = {"statistic": None, "p_value": None, "stationary": None}

    # Per-model tests
    model_tests = {}
    residuals_dict = {}

    for model_name, y_pred in predictions_dict.items():
        y_pred = np.asarray(y_pred).flatten()
        residuals = y_true - y_pred
        residuals_dict[model_name] = residuals

        model_result = {}

        # Ljung-Box on residuals
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
                lb_stat = float(lb_result["lb_stat"].values[0])
                lb_pvalue = float(lb_result["lb_pvalue"].values[0])
            model_result["ljung_box"] = {
                "statistic": lb_stat,
                "p_value": lb_pvalue,
                "no_autocorrelation": bool(lb_pvalue > 0.05),
            }
        except Exception:
            model_result["ljung_box"] = {
                "statistic": None,
                "p_value": None,
                "no_autocorrelation": None,
            }

        # Pesaran-Timmermann (z-test on directional accuracy)
        true_dir = (y_true > 0).astype(int)
        pred_dir = (y_pred > 0).astype(int)
        n = len(true_dir)
        correct = (true_dir == pred_dir).sum()
        accuracy = correct / n
        # Z-test: H0: accuracy = 0.5
        z_stat = (accuracy - 0.5) / np.sqrt(0.25 / n)
        z_pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        model_result["pesaran_timmermann"] = {
            "accuracy": float(accuracy),
            "z_statistic": float(z_stat),
            "p_value": float(z_pvalue),
            "significant": bool(z_pvalue < 0.05),
        }

        model_tests[model_name] = model_result

    results["model_tests"] = model_tests

    # Diebold-Mariano test: pairwise comparison
    model_names = list(predictions_dict.keys())
    if len(model_names) >= 2:
        dm_results = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_i, name_j = model_names[i], model_names[j]
                e_i = residuals_dict[name_i]
                e_j = residuals_dict[name_j]

                # DM statistic using squared errors
                d = e_i**2 - e_j**2
                d_mean = np.mean(d)
                d_var = np.var(d, ddof=1) / len(d)
                if d_var > 1e-15:
                    dm_stat = d_mean / np.sqrt(d_var)
                    dm_pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
                else:
                    dm_stat = 0.0
                    dm_pvalue = 1.0

                dm_results[f"{name_i}_vs_{name_j}"] = {
                    "statistic": float(dm_stat),
                    "p_value": float(dm_pvalue),
                    "significant": bool(dm_pvalue < 0.05),
                    "better_model": name_i if dm_stat < 0 else name_j,
                }
        results["diebold_mariano"] = dm_results

    return results
