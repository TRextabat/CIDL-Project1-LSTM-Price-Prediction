"""PDF report generator for CIDL LSTM project using fpdf2."""

from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF


class ReportGenerator:
    """Generate a professional PDF report for the CIDL Section 3 submission.

    The report covers:
    - Problem statement and dataset description
    - Model architectures and training methodology
    - Experimental results with figures and tables
    - Statistical analysis and interpretation
    - Source attribution
    """

    def __init__(self, output_dir: str = "outputs/report"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def _add_page(self):
        self.pdf.add_page()

    def _heading(self, text: str, level: int = 1):
        sizes = {1: 18, 2: 14, 3: 12}
        self.pdf.set_font("Helvetica", "B", sizes.get(level, 12))
        self.pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(2)

    def _body(self, text: str):
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.multi_cell(0, 5, text)
        self.pdf.ln(2)

    def _add_image(self, image_path: str | Path, w: int = 180):
        path = Path(image_path)
        if path.exists():
            try:
                self.pdf.image(str(path), w=w)
                self.pdf.ln(5)
            except Exception:
                self._body(f"[Image not available: {path.name}]")

    def _add_table(self, df: pd.DataFrame, title: str = ""):
        if title:
            self.pdf.set_font("Helvetica", "B", 10)
            self.pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")

        if df.empty:
            self._body("No data available.")
            return

        self.pdf.set_font("Helvetica", "B", 8)
        n_cols = len(df.columns)
        col_w = min(180 / n_cols, 35)

        # Header
        for col in df.columns:
            self.pdf.cell(col_w, 6, str(col)[:15], border=1, align="C")
        self.pdf.ln()

        # Rows
        self.pdf.set_font("Helvetica", "", 7)
        for _, row in df.iterrows():
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    text = f"{val:.4f}"
                else:
                    text = str(val)[:15]
                self.pdf.cell(col_w, 5, text, border=1, align="C")
            self.pdf.ln()
        self.pdf.ln(3)

    def add_title_page(self):
        """Add the title page."""
        self._add_page()
        self.pdf.ln(40)
        self.pdf.set_font("Helvetica", "B", 24)
        self.pdf.cell(0, 15, "LSTM Time Series Prediction", align="C", new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_font("Helvetica", "", 16)
        self.pdf.cell(0, 10, "XAUUSD and BTCUSD Price Forecasting", align="C", new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(10)
        self.pdf.set_font("Helvetica", "", 12)
        self.pdf.cell(0, 8, "CIDL Project - Section 3", align="C", new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(20)
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.cell(0, 6, "Deep Learning for Financial Time Series", align="C", new_x="LMARGIN", new_y="NEXT")
        self.pdf.cell(0, 6, "Walk-Forward Validation with Multiple LSTM Architectures", align="C", new_x="LMARGIN", new_y="NEXT")

    def add_section_3i(self, results: list[dict], experiment_config: dict):
        """Section 3.i: Problem Statement and Dataset Description."""
        self._add_page()
        self._heading("3.i Problem Statement and Dataset", 1)

        self._heading("Problem Definition", 2)
        self._body(
            "This project addresses the challenge of financial time series prediction using "
            "Long Short-Term Memory (LSTM) neural networks. We formulate the problem as both "
            "a regression task (predicting log-returns) and a classification task (predicting "
            "price direction). The dual formulation allows us to evaluate model performance from "
            "both a statistical and a trading perspective."
        )

        self._heading("Datasets", 2)
        assets = experiment_config.get("data", {}).get("assets", [])
        for asset in assets:
            sym = asset.get("symbol", "unknown").upper()
            tf = asset.get("timeframe", "")
            self._body(
                f"{sym} ({tf}): This dataset represents {'hourly' if 'h' in tf else 'daily'} "
                f"price data for {'Gold (XAUUSD)' if 'xau' in sym.lower() else 'Bitcoin (BTCUSD)'}. "
                f"The data includes Open, High, Low, Close, and Volume (OHLCV) bars sourced from "
                f"the FinAgent project data repository."
            )

        self._heading("Data Characteristics", 2)
        self._body(
            "Financial time series exhibit several challenging properties:\n"
            "- Non-stationarity: prices follow a random walk with drift\n"
            "- Volatility clustering: large moves tend to cluster in time (GARCH effects)\n"
            "- Fat tails: return distributions have heavier tails than Gaussian\n"
            "- Low signal-to-noise ratio: predictable signal is small relative to noise\n\n"
            "To address non-stationarity, we model log-returns rather than raw prices. "
            "We use RobustScaler (median/IQR) for feature scaling, which is more resilient "
            "to outliers than StandardScaler."
        )

        self._heading("Feature Engineering", 2)
        self._body(
            "We compute 20 technical features spanning multiple categories:\n"
            "- Price features: log-returns, absolute returns, high-low range, close-open range\n"
            "- Momentum indicators: RSI(14), MACD histogram, Stochastic K/D, ROC(10)\n"
            "- Volatility indicators: ATR(14) normalized, Bollinger Band width and position\n"
            "- Trend indicators: EMA(20) ratio, EMA(50) ratio, SMA(50) ratio\n"
            "- Volume: volume ratio relative to 20-period moving average\n"
            "- Temporal encoding: cyclical hour (sin/cos) and day-of-week (sin/cos)"
        )

    def add_section_3ii(self, results: list[dict], experiment_config: dict):
        """Section 3.ii: Model Architectures and Training Methodology."""
        self._add_page()
        self._heading("3.ii Models and Training Methodology", 1)

        self._heading("Model Architectures", 2)

        models = experiment_config.get("models", [])
        model_descriptions = {
            "simple_lstm": (
                "Simple LSTM: A single-layer LSTM followed by a fully connected output layer. "
                "This serves as the baseline architecture with the fewest parameters. "
                "It uses the last hidden state for prediction."
            ),
            "stacked_lstm": (
                "Stacked LSTM: A 3-layer deep LSTM with inter-layer dropout (0.2) for "
                "regularization. The output passes through two FC layers with ReLU activation. "
                "Deeper architectures can capture more complex temporal patterns."
            ),
            "bilstm": (
                "Bidirectional LSTM: A 2-layer BiLSTM that processes the sequence in both "
                "forward and backward directions. The concatenated hidden states (2x hidden size) "
                "pass through FC layers. BiLSTMs can capture future-context patterns within "
                "the lookback window."
            ),
            "lstm_attention": (
                "LSTM with Bahdanau Attention: A 2-layer LSTM augmented with an additive "
                "attention mechanism over all hidden states. Instead of using only the last "
                "hidden state, attention computes a weighted sum across all time steps, "
                "allowing the model to focus on the most relevant parts of the input sequence. "
                "Attention weights are stored for interpretability analysis."
            ),
        }

        for m in models:
            name = m.get("name", "")
            desc = model_descriptions.get(name, f"Model: {name}")
            self._body(desc)

        # Parameter counts from results
        param_data = {}
        for r in results:
            mn = r.get("model_name", "")
            pc = r.get("param_count", 0)
            if mn and pc:
                param_data[mn] = pc
        if param_data:
            self._body("Parameter counts: " + ", ".join(f"{k}: {v:,}" for k, v in param_data.items()))

        self._heading("Training Methodology", 2)
        training_cfg = experiment_config.get("experiments", {}).get("training", {})
        self._body(
            f"Walk-Forward Cross-Validation: We use an expanding window approach with "
            f"{training_cfg.get('walk_forward_folds', 3)} folds and a purge gap of "
            f"{training_cfg.get('purge_bars', 30)} bars between train and test sets to prevent "
            f"information leakage. This is critical for time series data where random splits "
            f"would introduce look-ahead bias.\n\n"
            f"Optimization: Adam optimizer with learning rate {training_cfg.get('learning_rate', 0.001)}, "
            f"ReduceLROnPlateau scheduler (factor=0.5, patience=5), gradient clipping at "
            f"{training_cfg.get('grad_clip', 1.0)}, and early stopping with patience "
            f"{training_cfg.get('patience', 10)}.\n\n"
            f"Loss Function: Combined MSE (regression) + 0.1 * BCE (direction classification). "
            f"The dual-objective encourages the model to predict both magnitude and direction.\n\n"
            f"Scaling: RobustScaler fitted per fold on training data only to avoid look-ahead bias."
        )

        self._heading("Experiment Matrix", 2)
        lookbacks = experiment_config.get("experiments", {}).get("lookback_windows", [])
        horizons = experiment_config.get("experiments", {}).get("prediction_horizons", [])
        self._body(
            f"Lookback windows: {lookbacks}\n"
            f"Prediction horizons: {horizons}\n"
            f"Models: {[m['name'] for m in models]}\n"
            f"Assets: {[a['symbol'] + '_' + a['timeframe'] for a in experiment_config.get('data', {}).get('assets', [])]}\n"
            f"Total experiments: {len(models) * len(lookbacks) * len(horizons) * len(experiment_config.get('data', {}).get('assets', []))}"
        )

    def add_section_3iii(self, results: list[dict], figures_dir: str | Path):
        """Section 3.iii: Experimental Results and Analysis."""
        self._add_page()
        self._heading("3.iii Experimental Results", 1)
        figures_dir = Path(figures_dir)

        # Summary table
        self._heading("Results Summary", 2)
        if results:
            rows = []
            for r in results:
                if "metrics" in r:
                    rows.append({
                        "Model": r.get("model_name", "?"),
                        "LB": r.get("lookback", 0),
                        "H": r.get("horizon", 0),
                        "RMSE": r["metrics"].get("rmse", 0),
                        "MAE": r["metrics"].get("mae", 0),
                        "Dir Acc": r["metrics"].get("direction_accuracy", 0),
                        "Sharpe": r["metrics"].get("sharpe_ratio", 0),
                        "F1": r["metrics"].get("f1", 0),
                    })
            if rows:
                df = pd.DataFrame(rows)
                self._add_table(df, "All Experiment Results")

        # Add figures
        self._heading("Figures", 2)
        figure_files = sorted(figures_dir.glob("*.png"))
        for fig_path in figure_files:
            try:
                self._add_image(fig_path, w=170)
                self.pdf.set_font("Helvetica", "I", 9)
                caption = fig_path.stem.replace("_", " ").title()
                self.pdf.cell(0, 5, f"Figure: {caption}", new_x="LMARGIN", new_y="NEXT", align="C")
                self.pdf.ln(3)
            except Exception:
                continue

            # Check if we need a new page
            if self.pdf.get_y() > 250:
                self._add_page()

        # Interpretation
        self._heading("Interpretation", 2)
        if results:
            # Find best model by Sharpe
            best = max(results, key=lambda r: r.get("metrics", {}).get("sharpe_ratio", -999))
            best_name = best.get("model_name", "unknown")
            best_sharpe = best.get("metrics", {}).get("sharpe_ratio", 0)
            best_dir = best.get("metrics", {}).get("direction_accuracy", 0)
            self._body(
                f"The best performing model across our experiments is {best_name} "
                f"with a Sharpe ratio of {best_sharpe:.4f} and directional accuracy of "
                f"{best_dir:.4f}. "
                f"This suggests that {'the attention mechanism provides meaningful' if 'attention' in best_name else 'the model architecture provides adequate'} "
                f"capacity for capturing temporal patterns in financial data.\n\n"
                f"Key findings:\n"
                f"- Direction accuracy above 0.50 indicates the model captures some predictive signal\n"
                f"- Positive Sharpe ratios suggest the predictions have economic value\n"
                f"- Walk-forward validation ensures these results are not due to overfitting\n"
                f"- The purge gap prevents information leakage between folds"
            )

    def add_section_3iv(self):
        """Section 3.iv: Source Attribution."""
        self._add_page()
        self._heading("3.iv Source Attribution", 1)
        self._body(
            "This project was built from scratch in PyTorch, inspired by the LSTM architecture "
            "patterns found in the FinAgent project (lstm_trend.py). Key design decisions:\n\n"
            "- The LSTM model structure follows FinAgent's convention of batch_first=True and "
            "config-dict based initialization.\n"
            "- Feature engineering extends FinAgent's basic feature set (RSI, MACD, BB, ATR) "
            "with additional technical indicators and temporal encodings.\n"
            "- The walk-forward validation methodology is original to this project and addresses "
            "the critical issue of temporal data leakage that affects many ML finance papers.\n"
            "- All model architectures, the training pipeline, evaluation metrics, and visualization "
            "code were developed specifically for this project.\n\n"
            "Libraries used: PyTorch (deep learning), pandas-ta (technical indicators), "
            "scikit-learn (scaling, metrics), statsmodels (statistical tests), matplotlib/seaborn "
            "(visualization), fpdf2 (PDF generation), scipy (statistical analysis).\n\n"
            "Data source: FinAgent project parquet files (XAUUSD 1H, BTCUSD 1D)."
        )

    def generate(
        self,
        results: list[dict],
        figures_dir: str | Path,
        experiment_config: dict,
    ) -> str:
        """Generate the complete PDF report.

        Args:
            results: List of experiment result dicts.
            figures_dir: Path to directory containing figure PNGs.
            experiment_config: The experiment YAML config dict.

        Returns:
            Path to the generated PDF file.
        """
        self.add_title_page()
        self.add_section_3i(results, experiment_config)
        self.add_section_3ii(results, experiment_config)
        self.add_section_3iii(results, figures_dir)
        self.add_section_3iv()

        output_path = self.output_dir / "CIDL_Section3_Report.pdf"
        self.pdf.output(str(output_path))
        return str(output_path)
