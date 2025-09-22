"""
This module implements backtesting and walk-forward validation utilities
for evaluating trading strategies.

- Backtester:
  Runs a trading simulation on a price series and signals, producing
  performance metrics such as CAGR, Sharpe, and Max Drawdown.

- WalkForward:
  Performs rolling window training and testing (walk-forward analysis)
  to assess model robustness over time.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


class Backtester:
    """
    I'm a backtester that simulates strategy performance over time.

    - Takes in prices and trading signals.
    - Computes equity curve, buy-and-hold benchmark.
    - Provides performance metrics: CAGR, Sharpe ratio, Max Drawdown, etc.
    """

    def __init__(
        self,
        prices: pd.Series,
        signals: pd.Series,
        initial_capital: float = 10000.0,
        rf: float = 0.0,
    ):
        self.prices = prices.astype(float)
        self.signals = signals.astype(float)
        self.initial_capital = float(initial_capital)
        self.rf = float(rf)
        self.results = None

    def run(self):
        """I run the backtest and return aligned price, signals, equity, and buy-hold results."""
        aligned = pd.concat([self.prices, self.signals], axis=1).dropna()
        aligned.columns = ["price", "signal"]

        if aligned.empty:
            self.results = pd.DataFrame()
            return self.results

        # i'm calculating daily returns
        aligned["returns"] = aligned["price"].pct_change().fillna(0.0)
        # I apply strategy returns when signal=1
        aligned["strategy"] = aligned["signal"].shift(1).fillna(0.0) * aligned["returns"]

        # i'm compounding strategy equity over time
        aligned["equity"] = (1.0 + aligned["strategy"]).cumprod() * self.initial_capital
        # I compare it with simple buy-and-hold performance
        aligned["buyhold"] = (aligned["price"] / aligned["price"].iloc[0]) * self.initial_capital

        self.results = aligned
        return aligned

    def metrics(self):
        """I calculate key performance metrics: CAGR, Sharpe, MaxDD, and out/underperformance."""
        if self.results is None or self.results.empty:
            return self._default_metrics()

        strat = self.results["strategy"].dropna().astype(float)
        equity = self.results["equity"].astype(float)
        buyhold = self.results["buyhold"].astype(float)

        if equity.empty or len(equity) < 2:
            return self._default_metrics()

        # i'm calculating CAGR (annualized growth)
        total_return = float(equity.iloc[-1]) / float(equity.iloc[0]) - 1.0
        n_years = float(len(equity)) / 252.0
        cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

        # I compute sharpe ratio using daily excess returns
        if strat.std(ddof=0) > 0:
            excess = float(strat.mean()) - self.rf / 252.0
            sharpe = float(np.sqrt(252.0) * excess / strat.std(ddof=0))
        else:
            sharpe = 0.0

        # i'm finding maximum drawdown
        roll_max = equity.cummax()
        dd = equity / roll_max - 1.0
        max_dd = float(dd.min()) if not dd.empty else 0.0

        final_val = float(equity.iloc[-1])
        bh_final = float(buyhold.iloc[-1])
        outperf = final_val - bh_final

        return {
            "CAGR": round(cagr, 3),
            "Sharpe": round(sharpe, 3),
            "MaxDD": round(max_dd, 3),
            "Final Value": round(final_val, 2),
            "BuyHold Final": round(bh_final, 2),
            "Outperformance": round(outperf, 2),
        }

    def _default_metrics(self):
        """I return zero/neutral metrics if no results are available."""
        # i return zero metrics if no results available
        return {
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "MaxDD": 0.0,
            "Final Value": float(self.initial_capital),
            "BuyHold Final": float(self.initial_capital),
            "Outperformance": 0.0,
        }


class WalkForward:
    """
    I’m a walk-forward validator for time series strategies.

    - Splits dataset into rolling train/test windows.
    - Trains model on each fold and tests on out-of-sample data.
    - Aggregates performance metrics and backtest equity.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        price_series: pd.Series,
        feature_cols: list,
        train_window: int = 60,
        test_window: int = 30,
        step: int = 63,
        proba_threshold: float = 0.5,
        initial_capital: float = 10000.0,
        rf: float = 0.0,
        model_factory=None,
    ):
        subset_cols = list(dict.fromkeys(feature_cols + ["target"]))
        self.df = df.copy().dropna(subset=subset_cols)

        if "Date" in self.df.columns:
            self.df = self.df.sort_values("Date").set_index("Date")

        self.prices = price_series.copy().astype(float).sort_index()
        self.feature_cols = feature_cols
        self.train_window = int(train_window)
        self.test_window = int(test_window)
        self.step = int(step)
        self.proba_threshold = float(proba_threshold)
        self.initial_capital = float(initial_capital)
        self.rf = float(rf)

        # i'm defaulting to logistic regression if no custom model is provided
        self.model_factory = model_factory or (lambda: LogisticRegression(max_iter=1000))

        self.signals = None
        self.equity = None
        self.fold_metrics = []

    def _iter_folds(self):
        """I yield train/test index ranges for each walk-forward fold."""
        n = len(self.df)
        start = 0
        while True:
            train_end = start + self.train_window
            test_end = min(train_end + self.test_window, n)
            if test_end - start < self.train_window + 5:
                break
            yield start, train_end, train_end, test_end
            start += self.step
            if start + self.train_window + 5 > n:
                break

    def run(self):
        """
        I execute the walk-forward process:
        - Train model on each fold.
        - Predict signals on test window.
        - Backtest performance and collect metrics.
        """
        all_signals = pd.Series(dtype=float)
        self.fold_metrics = []

        for (tr_s, tr_e, te_s, te_e) in self._iter_folds():
            train_df = self.df.iloc[tr_s:tr_e]
            test_df = self.df.iloc[te_s:te_e]

            if train_df.empty or test_df.empty:
                continue

            # i'm splitting into features and target
            X_tr = train_df[self.feature_cols].values
            y_tr = train_df["target"].values.astype(int)
            X_te = test_df[self.feature_cols].values

            model = self.model_factory()
            model.fit(X_tr, y_tr)

            # I calculate probability of upward move
            proba_up = model.predict_proba(X_te)[:, 1]
            sig = (proba_up >= self.proba_threshold).astype(int)
            sig = pd.Series(sig, index=test_df.index, name="signal")

            prices_slice = self.prices.reindex(test_df.index).dropna()
            sig = sig.reindex(prices_slice.index).fillna(0.0)

            if prices_slice.empty:
                continue

            # i'm backtesting signals for this fold
            bt = Backtester(prices_slice, sig, initial_capital=self.initial_capital, rf=self.rf)
            bt.run()
            m = bt.metrics()
            m["fold_start"] = str(prices_slice.index[0])
            m["fold_end"] = str(prices_slice.index[-1])
            self.fold_metrics.append(m)

            all_signals = pd.concat([all_signals, sig])

        all_signals = all_signals.sort_index()
        aligned_prices = self.prices.reindex(all_signals.index).dropna()
        all_signals = all_signals.reindex(aligned_prices.index).fillna(0.0)

        if len(all_signals) == 0:
            self.signals = pd.Series(dtype=float)
            self.equity = pd.DataFrame()
            return {
                "metrics": Backtester(aligned_prices, all_signals, self.initial_capital, self.rf)._default_metrics(),
                "fold_metrics": self.fold_metrics,
                "equity": self.equity,
                "signals": self.signals,
            }

        self.signals = all_signals.astype(float)
        full_bt = Backtester(aligned_prices, self.signals, initial_capital=self.initial_capital, rf=self.rf)
        self.equity = full_bt.run()
        overall = full_bt.metrics()

        # I return metrics for overall strategy plus fold breakdown
        return {
            "metrics": overall,
            "fold_metrics": self.fold_metrics,
            "equity": self.equity,
            "signals": self.signals,
        }
