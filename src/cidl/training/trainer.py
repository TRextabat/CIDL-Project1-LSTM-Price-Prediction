"""Full training loop with walk-forward cross-validation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cidl.data.dataset import TimeSeriesDataset
from cidl.data.scaler import WalkForwardScaler
from cidl.models.registry import create_model
from cidl.training.walk_forward import walk_forward_split
from cidl.training.metrics import (
    compute_classification_metrics,
    compute_financial_metrics,
    compute_regression_metrics,
)


@dataclass
class TrainResult:
    """Container for training results from a single walk-forward run."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    val_direction_acc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    test_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    test_actuals: np.ndarray = field(default_factory=lambda: np.array([]))
    test_directions_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    test_directions_actual: np.ndarray = field(default_factory=lambda: np.array([]))


class Trainer:
    """Manages training, validation, and walk-forward cross-validation.

    Args:
        config: Training configuration dict with keys:
            max_epochs, patience, batch_size, learning_rate, grad_clip,
            walk_forward_folds, purge_bars.
        device: Torch device (cuda or cpu).
    """

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

    def _train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        mse_loss: nn.Module,
        bce_loss: nn.Module,
    ) -> float:
        """Train for one epoch and return average loss."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_batch, y_reg, y_cls in train_loader:
            x_batch = x_batch.to(self.device)
            y_reg = y_reg.to(self.device)
            y_cls = y_cls.to(self.device)

            optimizer.zero_grad()
            pred = model(x_batch)  # (batch, 1)
            pred = pred.squeeze(-1)

            # Combined loss: MSE + 0.1 * BCE
            loss_mse = mse_loss(pred, y_reg)
            pred_prob = torch.sigmoid(pred)
            loss_bce = bce_loss(pred_prob, y_cls)
            loss = loss_mse + 0.1 * loss_bce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config["grad_clip"]
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        mse_loss: nn.Module,
        bce_loss: nn.Module,
    ) -> tuple[float, float, float, float]:
        """Validate and return (loss, mae, rmse, direction_accuracy)."""
        model.eval()
        all_pred = []
        all_true = []
        total_loss = 0.0
        n_batches = 0

        for x_batch, y_reg, y_cls in val_loader:
            x_batch = x_batch.to(self.device)
            y_reg = y_reg.to(self.device)
            y_cls = y_cls.to(self.device)

            pred = model(x_batch).squeeze(-1)
            loss_mse = mse_loss(pred, y_reg)
            pred_prob = torch.sigmoid(pred)
            loss_bce = bce_loss(pred_prob, y_cls)
            loss = loss_mse + 0.1 * loss_bce

            total_loss += loss.item()
            n_batches += 1

            all_pred.extend(pred.cpu().numpy())
            all_true.extend(y_reg.cpu().numpy())

        all_pred = np.array(all_pred)
        all_true = np.array(all_true)

        avg_loss = total_loss / max(n_batches, 1)
        mae = float(np.mean(np.abs(all_true - all_pred)))
        rmse = float(np.sqrt(np.mean((all_true - all_pred) ** 2)))
        dir_acc = float(np.mean((all_true > 0) == (all_pred > 0)))

        return avg_loss, mae, rmse, dir_acc

    def _train_fold(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        train_idx: list[int],
        test_idx: list[int],
        model_cfg: dict,
        lookback: int,
        horizon: int,
        fold_num: int,
    ) -> TrainResult:
        """Train a single fold and return results."""
        # Split data
        train_features = features[train_idx]
        train_targets = targets[train_idx]
        test_features = features[test_idx]
        test_targets = targets[test_idx]

        # Fit scaler on train only (no look-ahead)
        scaler = WalkForwardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)

        # Create datasets
        train_ds = TimeSeriesDataset(
            train_features_scaled, train_targets, lookback, horizon
        )
        test_ds = TimeSeriesDataset(
            test_features_scaled, test_targets, lookback, horizon
        )

        if len(train_ds) == 0 or len(test_ds) == 0:
            print(f"  Fold {fold_num}: insufficient data, skipping")
            return TrainResult()

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
        )

        # Create model
        input_size = features.shape[1]
        model = create_model(
            model_cfg["name"],
            input_size=input_size,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        result = TrainResult()
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.config["max_epochs"]):
            train_loss = self._train_one_epoch(
                model, train_loader, optimizer, mse_loss, bce_loss
            )
            val_loss, val_mae, val_rmse, val_dir_acc = self._validate(
                model, test_loader, mse_loss, bce_loss
            )

            scheduler.step(val_loss)

            result.train_losses.append(train_loss)
            result.val_losses.append(val_loss)
            result.val_mae.append(val_mae)
            result.val_rmse.append(val_rmse)
            result.val_direction_acc.append(val_dir_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                result.best_epoch = epoch
                result.best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.config["patience"]:
                print(
                    f"  Fold {fold_num}: early stop at epoch {epoch + 1}, "
                    f"best={result.best_epoch + 1}"
                )
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        # Generate test predictions
        model.eval()
        all_pred = []
        all_true_reg = []
        all_true_cls = []

        with torch.no_grad():
            for x_batch, y_reg, y_cls in test_loader:
                x_batch = x_batch.to(self.device)
                pred = model(x_batch).squeeze(-1)
                all_pred.extend(pred.cpu().numpy())
                all_true_reg.extend(y_reg.numpy())
                all_true_cls.extend(y_cls.numpy())

        result.test_predictions = np.array(all_pred)
        result.test_actuals = np.array(all_true_reg)
        result.test_directions_pred = (result.test_predictions > 0).astype(float)
        result.test_directions_actual = np.array(all_true_cls)

        return result

    def run_walk_forward(
        self,
        features_df,
        model_cfg: dict,
        lookback: int,
        horizon: int,
        save_dir: str | None = None,
    ) -> dict:
        """Run full walk-forward training and return aggregated results.

        Args:
            features_df: DataFrame with feature columns (from compute_features).
            model_cfg: Model config dict with name, hidden_size, num_layers, dropout.
            lookback: Number of lookback steps.
            horizon: Prediction horizon.
            save_dir: Optional directory to save models/scalers.

        Returns:
            Dict with all metrics, predictions, and metadata.
        """
        feature_cols = [c for c in features_df.columns if c != "log_returns"]
        features = features_df[feature_cols].values
        targets = features_df["log_returns"].values

        n_folds = self.config["walk_forward_folds"]
        purge = self.config["purge_bars"]

        all_predictions = []
        all_actuals = []
        all_dirs_pred = []
        all_dirs_actual = []
        fold_results = []

        print(f"  Walk-forward: {n_folds} folds, lookback={lookback}, horizon={horizon}")

        for fold_num, (train_idx, test_idx) in enumerate(
            walk_forward_split(len(features), n_folds, purge)
        ):
            print(
                f"  Fold {fold_num}: train={len(train_idx)}, test={len(test_idx)}"
            )
            t0 = time.time()

            fold_result = self._train_fold(
                features, targets, train_idx, test_idx,
                model_cfg, lookback, horizon, fold_num,
            )
            elapsed = time.time() - t0
            print(
                f"  Fold {fold_num}: done in {elapsed:.1f}s, "
                f"best_epoch={fold_result.best_epoch + 1}, "
                f"val_loss={fold_result.best_val_loss:.6f}"
            )

            fold_results.append(fold_result)

            if len(fold_result.test_predictions) > 0:
                all_predictions.extend(fold_result.test_predictions)
                all_actuals.extend(fold_result.test_actuals)
                all_dirs_pred.extend(fold_result.test_directions_pred)
                all_dirs_actual.extend(fold_result.test_directions_actual)

        # Aggregate metrics across folds
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        all_dirs_pred = np.array(all_dirs_pred)
        all_dirs_actual = np.array(all_dirs_actual)

        if len(all_predictions) == 0:
            return {"error": "No predictions generated"}

        reg_metrics = compute_regression_metrics(all_actuals, all_predictions)
        cls_metrics = compute_classification_metrics(
            all_dirs_actual, all_dirs_pred,
            y_pred_proba=torch.sigmoid(torch.FloatTensor(all_predictions)).numpy(),
        )
        fin_metrics = compute_financial_metrics(all_predictions, all_actuals)

        # Model parameter count
        input_size = features.shape[1]
        temp_model = create_model(
            model_cfg["name"],
            input_size=input_size,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
        )
        param_count = temp_model.count_parameters()

        # Per-fold training curves (average across folds for plotting)
        max_epochs = max(len(fr.train_losses) for fr in fold_results) if fold_results else 0
        avg_train_losses = []
        avg_val_losses = []
        for e in range(max_epochs):
            tl = [fr.train_losses[e] for fr in fold_results if e < len(fr.train_losses)]
            vl = [fr.val_losses[e] for fr in fold_results if e < len(fr.val_losses)]
            avg_train_losses.append(float(np.mean(tl)) if tl else 0.0)
            avg_val_losses.append(float(np.mean(vl)) if vl else 0.0)

        result = {
            "model_name": model_cfg["name"],
            "lookback": lookback,
            "horizon": horizon,
            "n_folds": n_folds,
            "param_count": param_count,
            "metrics": {**reg_metrics, **cls_metrics, **fin_metrics},
            "training_curves": {
                "train_losses": avg_train_losses,
                "val_losses": avg_val_losses,
            },
            "predictions": all_predictions.tolist(),
            "actuals": all_actuals.tolist(),
            "directions_pred": all_dirs_pred.tolist(),
            "directions_actual": all_dirs_actual.tolist(),
            "fold_details": [
                {
                    "best_epoch": fr.best_epoch,
                    "best_val_loss": float(fr.best_val_loss),
                    "n_test_samples": len(fr.test_predictions),
                }
                for fr in fold_results
            ],
        }

        return result
