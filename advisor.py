"""
This module provides two main advisor classes for stock prediction:

1. StockAdvisor:
   - Uses Logistic Regression with financial + sentiment features.
   - Fetches stock prices from Yahoo Finance and headlines from Finviz.
   - Builds training dataset, trains, evaluates, and makes predictions.

2. StockLSTMAdvisor:
   - Inherits from StockAdvisor but extends with LSTM deep learning.
   - Prepares sequential data, trains LSTM, evaluates, and generates predictions.

Both advisors produce recommendations (BUY, HOLD) with confidence levels
and provide summaries interpretable for end-users.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import StandardScaler
import numpy as np


class StockAdvisor:
    """I'm a stock advisor using Logistic Regression with features like return, volatility, and sentiment."""

    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.headlines_df = None
        self.stock_data = None
        self.features = {}
        self.sentiment_score = None
        self.recommendation = None

    def fetch_stock_data(self):
        """I fetch historical stock data from Yahoo Finance."""
        self.stock_data = yf.download(
            self.symbol, start=self.start_date, end=self.end_date, auto_adjust=True
        )

    def fetch_headlines(self, n=200, years_back=3):
        """I'm scraping recent news headlines from Finviz for sentiment analysis."""
        url = f"https://finviz.com/quote.ashx?t={self.symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        news_table = soup.find("table", class_="fullview-news-outer")

        rows_data = []
        if news_table:
            rows = news_table.find_all("tr")
            last_date = None
            for row in rows:
                a = row.find("a")
                if not a:
                    continue
                text = a.get_text().strip()
                date_cell = row.find("td").get_text().strip()

                # i'm parsing date and time from Finviz format
                if " " in date_cell:
                    date_str, time_str = date_cell.split(" ")
                    last_date = pd.to_datetime(date_str, errors="coerce")
                else:
                    time_str = date_cell
                if last_date is None:
                    last_date = datetime.now()

                if pd.isnull(last_date):
                    date_val = None
                else:
                    date_val = last_date.strftime("%Y-%m-%d")
                rows_data.append({"date": date_val, "headline": text})
                if len(rows_data) >= n:
                    break

        df = pd.DataFrame(rows_data)

        if not df.empty:
            cutoff = pd.Timestamp(datetime.now()) - pd.DateOffset(years=years_back)
            df = df[df["date"].notnull()]
            df = df[df["date"] >= cutoff.strftime("%Y-%m-%d")]

        self.headlines_df = df.reset_index(drop=True)

    def calculate_features(self):
        """I compute simple return and volatility features from stock data."""
        if self.stock_data is not None and not self.stock_data.empty:
            close = self.stock_data["Close"]
            self.features["return"] = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
            self.features["volatility"] = close.pct_change().std()
        else:
            self.features["return"] = None
            self.features["volatility"] = None

    def analyze_sentiment(self):
        """I calculate sentiment score of headlines using Vader."""
        analyzer = SentimentIntensityAnalyzer()
        if self.headlines_df is not None and not self.headlines_df.empty:
            sentiments = [analyzer.polarity_scores(h)["compound"] for h in self.headlines_df["headline"]]
            self.sentiment_score = sum(sentiments) / len(sentiments) if sentiments else 0
        else:
            self.sentiment_score = 0

    def build_training_data(self):
        """I'm building dataset with features (return, volatility, sentiment) and next-day up/down target."""
        df = self.stock_data.copy().reset_index()
        df["return"] = df["Close"].pct_change()
        df["volatility"] = df["Close"].rolling(window=5).std()

        # i create daily sentiment averages
        analyzer = SentimentIntensityAnalyzer()
        if self.headlines_df is not None and not self.headlines_df.empty:
            daily_sent = (
                self.headlines_df.groupby("date")["headline"]
                .apply(lambda x: np.mean([analyzer.polarity_scores(h)["compound"] for h in x]))
            )
            daily_sent = daily_sent.rolling(window=7, min_periods=1).mean()
            sent_dict = daily_sent.to_dict()
        else:
            sent_dict = {}

        df["sentiment"] = df["Date"].astype(str).map(lambda d: sent_dict.get(d, 0))

        # i'm setting up next-day up/down as target variable
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna()

        df.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in df.columns]
        self.training_df = df
        return df

    def train_model(self):
        """I train a Logistic Regression model with balanced class weights."""
        features = ["return", "volatility", "sentiment"]
        X = self.training_df[features]
        y = self.training_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        self.model = LogisticRegression(class_weight="balanced", max_iter=500)
        self.model.fit(X_train, y_train)
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_model(self):
        """I'm returning test accuracy of the Logistic Regression model."""
        acc = self.model.score(self.X_test, self.y_test)
        return acc

    def predict(self):
        """I generate a recommendation (BUY or HOLD) for the latest data point."""
        if hasattr(self, "model") and self.model:
            last_row = self.training_df.iloc[[-1]][["return", "volatility", "sentiment"]]
            prediction = self.model.predict(last_row)[0]
            proba = self.model.predict_proba(last_row)[0].max()
            self.recommendation = "BUY" if prediction == 1 else "HOLD"
            self.ml_confidence = proba
        else:
            self.recommendation = "UNKNOWN"
            self.ml_confidence = 0

    def generate_advice(self):
        """I'm summarizing features, sentiment, model performance, and recommendation in plain text."""
        ret = self.features.get("return", 0)
        vol = self.features.get("volatility", 0)
        if hasattr(ret, "item"):
            ret = ret.item()
        if hasattr(vol, "item"):
            vol = vol.item()
        acc = None
        if hasattr(self, "model") and hasattr(self, "X_test"):
            acc = self.model.score(self.X_test, self.y_test)
        coefs = None
        features = ["return", "volatility", "sentiment"]
        if hasattr(self, "model"):
            coefs = dict(zip(features, [round(float(v), 4) for v in self.model.coef_[0]]))

        summary = (
            f"Recommendation: {self.recommendation}\n"
            f"Confidence: {round(self.ml_confidence*100, 1)}%\n"
            f"\nLatest stats used:\n"
            f" - Recent return: {round(ret*100, 2)}%\n"
            f" - Volatility: {round(vol*100, 2)}%\n"
            f" - News sentiment: {round(self.sentiment_score, 2)}\n"
            f"\nModel accuracy on test data: {round(acc, 2) if acc else 'N/A'}\n"
            f"\nWhat matters most to the model:\n"
            + ("\n".join([f" - {k.capitalize()}: {v}" for k, v in coefs.items()]) if coefs else "")
        )
        return summary


class StockLSTMAdvisor(StockAdvisor):
    """I extend StockAdvisor with an LSTM model for sequence learning on financial features."""

    def __init__(self, symbol, start_date, end_date):
        super().__init__(symbol, start_date, end_date)
        self.scaler = StandardScaler()
        self.lstm_model = None

    def prepare_lstm_data(self, sequence_length=30):
        """I'm preparing sequential training/testing datasets for the LSTM model."""
        df = self.training_df.copy()
        features = ["return", "volatility", "sentiment"]
        X = df[features].values
        y = df["target"].values

        X_scaled = self.scaler.fit_transform(X)

        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i: i + sequence_length])
            y_seq.append(y[i + sequence_length])
        self.X_lstm = np.array(X_seq)
        self.y_lstm = np.array(y_seq)

        split = int(0.7 * len(self.X_lstm))
        self.X_lstm_train = self.X_lstm[:split]
        self.y_lstm_train = self.y_lstm[:split]
        self.X_lstm_test = self.X_lstm[split:]
        self.y_lstm_test = self.y_lstm[split:]

    def train_lstm(self, epochs=10):
        """I define and train a simple LSTM for binary classification (up/down)."""
        n_features = self.X_lstm_train.shape[2]
        self.lstm_model = Sequential(
            [
                Input(shape=(self.X_lstm_train.shape[1], n_features)),
                LSTM(32),
                Dense(1, activation="sigmoid"),
            ]
        )
        self.lstm_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.lstm_model.fit(
            self.X_lstm_train, self.y_lstm_train, epochs=epochs, verbose=1
        )

    def evaluate_lstm(self):
        """I'm evaluating the trained LSTM on the test set and returning accuracy."""
        loss, acc = self.lstm_model.evaluate(
            self.X_lstm_test, self.y_lstm_test, verbose=0
        )
        print("LSTM Test Accuracy:", round(acc, 3))
        return acc

    def predict_lstm(self):
        """I predict a recommendation (BUY or HOLD) using the last sequence in training data."""
        last_seq = self.X_lstm[-1].reshape(
            1, self.X_lstm.shape[1], self.X_lstm.shape[2]
        )
        proba = self.lstm_model.predict(last_seq, verbose=0)[0][0]

        pred = int(proba > 0.5)
        self.lstm_recommendation = "BUY" if pred == 1 else "HOLD"
        self.lstm_confidence = proba

    def generate_lstm_summary(self):
        """I'm summarizing LSTM test performance and the final recommendation."""
        acc = None
        if hasattr(self, "lstm_model") and hasattr(self, "X_lstm_test"):
            loss, acc = self.lstm_model.evaluate(
                self.X_lstm_test, self.y_lstm_test, verbose=0
            )
        summary = (
            f"LSTM Recommendation: {self.lstm_recommendation}\n"
            f"Confidence: {round(self.lstm_confidence*100, 1)}%\n"
            f"\nModel test accuracy: {round(acc, 2) if acc else 'N/A'}\n"
        )
        return summary
