# strategies.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd

from advisor import StockAdvisor  # i'm making sure we avoid circular import

class Strategy(ABC):
    """Abstract strategy -> returns a dated signal series (1=long, 0=flat)."""
    name: str

    @abstractmethod
    def predict_signals(self) -> pd.Series:
        """Return a pd.Series of {0,1} indexed by trading dates (no NaNs)."""
        ...  # I know this is just a placeholder for subclasses

class LRStrategy(Strategy):
    def __init__(
        self,
        advisor: StockAdvisor,
        features: Optional[List[str]] = None,
        proba_threshold: float = 0.5,
        name: Optional[str] = None,
    ):
        # i'm checking if advisor has a trained model
        if not hasattr(advisor, "model"):
            raise ValueError("Advisor model not trained. Call advisor.train_model() first.")
        # I also make sure training_df exists before continuing
        if not hasattr(advisor, "training_df"):
            raise ValueError("Advisor has no training_df. Build data before training.")

        self.advisor = advisor
        # i'm setting default features if none are provided
        self.features = features or ["return", "volatility", "sentiment"]
        self.proba_threshold = proba_threshold
        # I create a default name if one isn't passed in
        self.name = name or f"LR-{advisor.symbol}"

    def predict_signals(self) -> pd.Series:
        df = self.advisor.training_df
        X = df[self.features]
        # i'm predicting the probability of class 1 (stock going up)
        proba_up = self.advisor.model.predict_proba(X)[:, 1]
        # I turn probabilities into signals (1 or 0) based on threshold
        signals = (proba_up >= self.proba_threshold).astype(int)
        # i'm aligning signals with trading dates for clarity
        return pd.Series(signals, index=df["Date"], name="signal")
