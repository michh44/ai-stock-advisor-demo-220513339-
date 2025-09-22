import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # i'm disabling OneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # I suppress TensorFlow warnings

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from advisor import StockAdvisor, StockLSTMAdvisor
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score

from strategies import LRStrategy
from backtest import Backtester, WalkForward

st.set_page_config(page_title="AI Stock Advisor", layout="wide")

"""
This is my **Streamlit UI** for the AI Stock Advisor project.  
It lets users choose a stock, fetch data + news, run Logistic Regression or LSTM,  
and then shows recommendations, backtests, benchmarking, and walk-forward validation.
"""

# ---------- Session state ----------
if "advisor_obj" not in st.session_state:
    st.session_state["advisor_obj"] = None   # i'm keeping advisor object across runs
if "advisor_inputs" not in st.session_state:
    st.session_state["advisor_inputs"] = {}  # I track last inputs to detect changes

st.title("AI Stock Advisor (CM3020 – Project Idea 2 - 220513339)")
st.caption("Logistic Regression and LSTM with price + news sentiment (free data only)")

with st.sidebar:
    st.header("Inputs")
    symbol = st.text_input("Ticker", value="MSFT").upper()
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start", pd.to_datetime("2024-01-01")).strftime("%Y-%m-%d")
    with col2:
        end = st.date_input("End", pd.to_datetime("2024-12-31")).strftime("%Y-%m-%d")
    model_choice = st.selectbox("Model", ["Logistic Regression", "LSTM (deep learning)"])
    epochs = st.slider("LSTM epochs", 5, 30, 10)
    investment = st.number_input(
        "Initial Investment (€)", min_value=1000, max_value=1000000,
        value=10000, step=1000, help="Default is €10,000"
    )
    run_btn = st.button("Run Analysis", use_container_width=True)

# ---------- Helper functions ----------
def plot_confmat(y_true, y_pred, title):
    """i'm plotting confusion matrix for classification results."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    st.pyplot(fig)

def explain_metrics(m: dict, initial_investment: float) -> str:
    """Interpretation for Backtest results.
    I'm turning raw metrics into human-readable insights for users.
    """
    cagr = m.get("CAGR", np.nan)
    sharpe = m.get("Sharpe", np.nan)
    maxdd = m.get("MaxDD", np.nan)
    finalv = m.get("Final Value", np.nan)
    bh_final = m.get("BuyHold Final", np.nan)
    outperf = m.get("Outperformance", np.nan)

    explanation = ["### Backtest Interpretation"]

    if np.isfinite(finalv) and np.isfinite(bh_final):
        gain_loss = finalv - initial_investment
        sign = "profit" if gain_loss >= 0 else "loss"
        explanation.append(
            f"- Starting with **€{initial_investment:,.0f}**, strategy ends at "
            f"**€{finalv:,.0f}** ({sign} of **€{gain_loss:,.0f}**)."
        )
        explanation.append(f"- Buy & Hold would have ended at **€{bh_final:,.0f}**.")
        if outperf > 0:
            explanation.append(f"- ✅ Strategy outperformed Buy & Hold by **€{outperf:,.0f}**.")
        elif outperf < 0:
            explanation.append(f"- ❌ Strategy lagged Buy & Hold by **€{abs(outperf):,.0f}**.")
        else:
            explanation.append("- Strategy matched Buy & Hold.")

    if np.isfinite(cagr):
        explanation.append(f"- CAGR (annual growth): **{cagr*100:.1f}%** → shows long-term growth potential.")

    if np.isfinite(sharpe):
        if sharpe >= 1.5:
            explanation.append("- ✅ Excellent Sharpe ratio → strong risk-adjusted returns.")
        elif sharpe >= 1.0:
            explanation.append("- 👍 Good Sharpe ratio → balanced growth vs volatility.")
        elif sharpe >= 0.5:
            explanation.append("- ↔ Mediocre Sharpe ratio → returns came with high swings.")
        else:
            explanation.append("- ⚠️ Weak Sharpe → risk not well rewarded.")

    if np.isfinite(maxdd):
        if maxdd <= -0.3:
            explanation.append(f"- 🚨 Max drawdown **{maxdd*100:.1f}%** → very high risk.")
        elif maxdd <= -0.15:
            explanation.append(f"- ⚠️ Max drawdown **{maxdd*100:.1f}%** → noticeable stress for investors.")
        else:
            explanation.append(f"- ✅ Max drawdown only **{maxdd*100:.1f}%** → relatively safe.")

    return "\n".join(explanation)

def explain_walkforward(m: dict, initial_investment: float) -> str:
    """Interpretation for Walk-Forward results.
    I explain rolling retraining performance (robustness test).
    """
    cagr = m.get("CAGR", np.nan)
    sharpe = m.get("Sharpe", np.nan)
    maxdd = m.get("MaxDD", np.nan)
    finalv = m.get("Final Value", np.nan)
    bh_final = m.get("BuyHold Final", np.nan)
    outperf = m.get("Outperformance", np.nan)

    explanation = ["### Walk-Forward Interpretation"]

    if np.isfinite(finalv) and np.isfinite(bh_final):
        gain_loss = finalv - initial_investment
        sign = "profit" if gain_loss >= 0 else "loss"
        explanation.append(
            f"- Rolling re-training ends at **€{finalv:,.0f}** "
            f"({sign} of **€{gain_loss:,.0f}**) vs Buy & Hold **€{bh_final:,.0f}**."
        )
        if outperf > 0:
            explanation.append(f"- ✅ Strategy beat Buy & Hold by **€{outperf:,.0f}**.")
        elif outperf < 0:
            explanation.append(f"- ❌ Strategy lagged Buy & Hold by **€{abs(outperf):,.0f}**.")
        else:
            explanation.append("- Strategy matched Buy & Hold.")

    if np.isfinite(cagr):
        if cagr > 0.15:
            explanation.append(f"- 🚀 CAGR **{cagr*100:.1f}%** → strong consistent growth.")
        elif cagr > 0.05:
            explanation.append(f"- 👍 CAGR **{cagr*100:.1f}%** → steady growth.")
        elif cagr > 0:
            explanation.append(f"- ↔ CAGR **{cagr*100:.1f}%** → tiny gains.")
        else:
            explanation.append(f"- ⚠️ CAGR **{cagr*100:.1f}%** → strategy lost value.")

    if np.isfinite(sharpe):
        if sharpe >= 1.5:
            explanation.append("- ✅ Excellent Sharpe ratio → good reward vs volatility.")
        elif sharpe >= 1.0:
            explanation.append("- 👍 Solid Sharpe ratio → reasonable risk-adjusted performance.")
        elif sharpe >= 0.5:
            explanation.append("- ↔ Mixed Sharpe → results uneven.")
        else:
            explanation.append("- ⚠️ Weak Sharpe → high risk for little reward.")

    if np.isfinite(maxdd):
        if maxdd <= -0.3:
            explanation.append(f"- 🚨 Max drawdown **{maxdd*100:.1f}%** → severe losses possible.")
        elif maxdd <= -0.15:
            explanation.append(f"- ⚠️ Max drawdown **{maxdd*100:.1f}%** → significant dips.")
        else:
            explanation.append(f"- ✅ Max drawdown only **{maxdd*100:.1f}%** → relatively safe.")

    return "\n".join(explanation)


# ---------- Main Logic ----------
have_advisor = isinstance(st.session_state.get("advisor_obj"), StockAdvisor)  # i'm checking if we already have advisor in state

# i decide whether to run analysis now or wait for button/input change
if "ran_wf_lstm" in st.session_state and st.session_state["ran_wf_lstm"]:
    show_analysis = True
else:
    show_analysis = run_btn or (model_choice == "Logistic Regression" and have_advisor) or \
                    (model_choice == "LSTM (deep learning)" and isinstance(st.session_state.get("advisor_obj"), StockLSTMAdvisor))


if show_analysis:
    try:
        # Logistic Regression
        if model_choice == "Logistic Regression":
            # I'm checking if inputs changed or model not trained yet
            inputs_now = {"symbol": symbol, "start": start, "end": end}
            inputs_changed = inputs_now != st.session_state.get("advisor_inputs")

            if run_btn or (not have_advisor) or inputs_changed:
                advisor = StockAdvisor(symbol, start, end)  # i build fresh advisor
                advisor.fetch_stock_data()
                advisor.fetch_headlines()
                advisor.calculate_features()
                advisor.analyze_sentiment()
                advisor.build_training_data()
                advisor.train_model()
                advisor.predict()
                st.session_state["advisor_obj"] = advisor
                st.session_state["advisor_inputs"] = inputs_now
            else:
                advisor = st.session_state["advisor_obj"]

            # -------- Visual Decision Card --------
            st.header("📌 Final Recommendation")
            rec = advisor.recommendation
            conf = round(advisor.ml_confidence * 100, 1)
            color = {"BUY": "green", "HOLD": "orange", "SELL": "red"}.get(rec, "gray")
            st.markdown(
                f"""
                <div style="padding:15px;border-radius:10px;background-color:{color};color:white;text-align:center;font-size:22px;">
                    <b>{rec}</b> &nbsp;&nbsp; (Confidence: {conf}%)
                </div>
                """,
                unsafe_allow_html=True,
            )

            # -------- Backtest --------
            st.header("📊 Backtest Results (Past Performance)")
            try:
                strategy = LRStrategy(advisor)   # I'm using LR strategy class
                signals = strategy.predict_signals()
                prices = advisor.stock_data["Close"]
                bt = Backtester(prices, signals, initial_capital=investment)
                bt.run()
                metrics = bt.metrics()

                st.caption("📈 Equity = AI strategy performance, Buy & Hold = baseline stock performance")
                st.line_chart(bt.results[["equity", "buyhold"]])
                st.markdown(explain_metrics(metrics, investment))
                with st.expander("See raw backtest metrics"):
                    st.json(metrics)

            except Exception as e:
                st.warning(f"Backtest failed: {e}")

            # -------- Walk-Forward --------
            st.header("🔄 Walk-Forward Validation (Robustness)")
            if st.button("Run Walk-Forward", key="wf_btn") or st.session_state.get("ran_wf"):
                st.session_state["ran_wf"] = True
                try:
                    wf = WalkForward(
                        df=advisor.training_df,
                        price_series=advisor.stock_data["Close"],
                        feature_cols=["return", "volatility", "sentiment"],
                        train_window=60,
                        test_window=30,
                        step=20,
                        proba_threshold=0.5,
                        initial_capital=investment,
                        rf=0.0,
                    )
                    out = wf.run()
                    st.caption("📈 Equity = AI strategy performance, Buy & Hold = baseline stock performance")

                    st.line_chart(out["equity"][["equity", "buyhold"]])
                    st.markdown(explain_walkforward(out["metrics"], investment))
                    with st.expander("See raw walk-forward metrics"):
                        st.json(out["metrics"])
                        if out["fold_metrics"]:
                            st.dataframe(pd.DataFrame(out["fold_metrics"]))
                except Exception as e:
                    st.warning(f"Walk-forward failed: {e}")

            # -------- Benchmarking Dashboard --------
            st.header("📊 Model Benchmarking (AI vs Baselines)")
            try:
                y_true = advisor.y_test
                y_pred_lr = advisor.model.predict(advisor.X_test)
                random_preds = np.random.randint(0, 2, size=len(y_true))
                buyhold_preds = np.ones_like(y_true)  # I'm simulating buy-hold baseline

                benchmarks = {
                    "Logistic Regression": {
                        "Precision": precision_score(y_true, y_pred_lr, zero_division=0),
                        "Recall": recall_score(y_true, y_pred_lr, zero_division=0),
                        "ROC-AUC": roc_auc_score(y_true, y_pred_lr)
                    },
                    "Buy & Hold baseline": {
                        "Precision": precision_score(y_true, buyhold_preds, zero_division=0),
                        "Recall": recall_score(y_true, buyhold_preds, zero_division=0),
                        "ROC-AUC": roc_auc_score(y_true, buyhold_preds)
                    },
                    "Random baseline": {
                        "Precision": precision_score(y_true, random_preds, zero_division=0),
                        "Recall": recall_score(y_true, random_preds, zero_division=0),
                        "ROC-AUC": roc_auc_score(y_true, random_preds)
                    }
                }

                bench_df = pd.DataFrame(benchmarks).T.round(3)
                st.dataframe(bench_df, use_container_width=True)

                best_model = bench_df["ROC-AUC"].idxmax()
                st.markdown(f"""
                ### Benchmarking Interpretation
                - ✅ Best performing model (ROC-AUC): **{best_model}**  
                - Logistic Regression vs Random: shows if AI beats pure chance.  
                - Logistic Regression vs Buy & Hold: shows if AI adds value beyond holding stock.  
                - Use this to judge whether AI is really offering an **edge** or not.
                """)

            except Exception as e:
                st.warning(f"Benchmarking failed: {e}")

            # -------- Transparency --------
            st.header("🔍 Transparency & Details")
            colA, colB = st.columns([1, 1])
            with colA:
                st.subheader("Latest Headlines (live)")
                if advisor.headlines_df is not None and not advisor.headlines_df.empty:
                    st.dataframe(advisor.headlines_df, use_container_width=True, height=240)
                else:
                    st.info("No headlines available.")
            with colB:
                st.subheader("Model Evaluation")
                preds = advisor.model.predict(advisor.X_test)
                plot_confmat(advisor.y_test, preds, f"{symbol} — Logistic Regression")

            st.subheader("Training Data Preview")
            st.dataframe(advisor.training_df.head(20), use_container_width=True, height=300)

        # -------- LSTM --------
        else:
            # I'm checking inputs for LSTM run
            inputs_now = {"symbol": symbol, "start": start, "end": end, "epochs": epochs}
            inputs_changed = inputs_now != st.session_state.get("advisor_inputs")

            if run_btn or (not isinstance(st.session_state.get("advisor_obj"), StockLSTMAdvisor)) or inputs_changed:
                advisor = StockLSTMAdvisor(symbol, start, end)
                advisor.fetch_stock_data()
                advisor.fetch_headlines()
                advisor.calculate_features()
                advisor.analyze_sentiment()
                advisor.build_training_data()
                advisor.prepare_lstm_data()
                advisor.train_lstm(epochs=epochs)
                advisor.evaluate_lstm()
                advisor.predict_lstm()
                st.session_state["advisor_obj"] = advisor
                st.session_state["advisor_inputs"] = inputs_now
            else:
                advisor = st.session_state["advisor_obj"]

            # -------- Visual Decision Card --------
            st.header("📌 Final Recommendation (LSTM)")
            rec = advisor.lstm_recommendation
            conf = round(advisor.lstm_confidence * 100, 1)
            color = {"BUY": "green", "HOLD": "orange", "SELL": "red"}.get(rec, "gray")
            st.markdown(
                f"""
                <div style="padding:15px;border-radius:10px;background-color:{color};color:white;text-align:center;font-size:22px;">
                    <b>{rec}</b> &nbsp;&nbsp; (Confidence: {conf}%)
                </div>
                """,
                unsafe_allow_html=True,
            )

            # -------- Backtest --------
            st.header("📊 Backtest Results (Past Performance)")
            try:
                if advisor.lstm_model is not None and advisor.X_lstm is not None:
                    seq_len = advisor.X_lstm.shape[1]
                    preds = advisor.lstm_model.predict(advisor.X_lstm, verbose=0).ravel()
                    signals = pd.Series(
                        (preds > 0.5).astype(int),
                        index=advisor.training_df["Date"].iloc[seq_len:],  # I align sequences with dates
                        name="signal"
                    )
                    prices = advisor.training_df.set_index("Date")["Close"].iloc[seq_len:]
                    bt = Backtester(prices, signals, initial_capital=investment)
                    bt.run()
                    metrics = bt.metrics()

                    st.caption("📈 Equity = AI strategy performance, Buy & Hold = baseline stock performance")
                    if bt.results is not None and all(col in bt.results for col in ["equity", "buyhold"]):
                        st.line_chart(bt.results[["equity", "buyhold"]])
                    else:
                        st.warning("Backtest results incomplete (missing equity/buyhold).")

                    st.markdown(explain_metrics(metrics, investment))
                    with st.expander("See raw backtest metrics"):
                        st.json(metrics)
                else:
                    st.info("LSTM model or input data not available for backtest.")
            except Exception as e:
                st.warning(f"Backtest failed: {e}")

            # -------- Walk-Forward --------
            st.header("🔄 Walk-Forward Validation (Robustness)")
            if st.button("Run Walk-Forward (LSTM)", key="wf_btn_lstm") or st.session_state.get("ran_wf_lstm"):
                st.session_state["ran_wf_lstm"] = True
                try:
                    if advisor.training_df is not None and not advisor.training_df.empty:
                        wf = WalkForward(
                            df=advisor.training_df,
                            price_series=advisor.training_df.set_index("Date")["Close"],
                            feature_cols=["return", "volatility", "sentiment"],
                            train_window=60,
                            test_window=30,
                            step=20,
                            proba_threshold=0.5,
                            initial_capital=investment,
                            rf=0.0,
                        )
                        out = wf.run()
                        st.caption("📈 Equity = AI strategy performance, Buy & Hold = baseline stock performance")
                        if out["equity"] is not None and not out["equity"].empty and all(col in out["equity"] for col in ["equity", "buyhold"]):
                            st.line_chart(out["equity"][ ["equity", "buyhold"] ])
                        else:
                            st.warning("Walk-forward results incomplete (missing equity/buyhold).")

                        st.markdown(explain_walkforward(out["metrics"], investment))
                        with st.expander("See raw walk-forward metrics"):
                            st.json(out["metrics"])
                            if out["fold_metrics"]:
                                st.dataframe(pd.DataFrame(out["fold_metrics"]))
                    else:
                        st.info("No training data available for walk-forward.")
                except Exception as e:
                    st.warning(f"Walk-forward failed: {e}")

            # -------- Benchmarking Dashboard --------
            st.header("📊 Model Benchmarking (AI vs Baselines)")
            try:
                if advisor.lstm_model is not None and advisor.X_lstm_test is not None and advisor.y_lstm_test is not None:
                    y_true = advisor.y_lstm_test
                    y_pred_lstm = (advisor.lstm_model.predict(advisor.X_lstm_test, verbose=0).ravel() > 0.5).astype(int)
                    random_preds = np.random.randint(0, 2, size=len(y_true))
                    buyhold_preds = np.ones_like(y_true)

                    benchmarks = {
                        "LSTM": {
                            "Precision": precision_score(y_true, y_pred_lstm, zero_division=0),
                            "Recall": recall_score(y_true, y_pred_lstm, zero_division=0),
                            "ROC-AUC": roc_auc_score(y_true, y_pred_lstm)
                        },
                        "Buy & Hold baseline": {
                            "Precision": precision_score(y_true, buyhold_preds, zero_division=0),
                            "Recall": recall_score(y_true, buyhold_preds, zero_division=0),
                            "ROC-AUC": roc_auc_score(y_true, buyhold_preds)
                        },
                        "Random baseline": {
                            "Precision": precision_score(y_true, random_preds, zero_division=0),
                            "Recall": recall_score(y_true, random_preds, zero_division=0),
                            "ROC-AUC": roc_auc_score(y_true, random_preds)
                        }
                    }

                    bench_df = pd.DataFrame(benchmarks).T.round(3)
                    st.dataframe(bench_df, use_container_width=True)

                    best_model = bench_df["ROC-AUC"].idxmax()
                    st.markdown(f"""
                    ### Benchmarking Interpretation
                    - ✅ Best performing model (ROC-AUC): **{best_model}**  
                    - LSTM vs Random: shows if AI beats pure chance.  
                    - LSTM vs Buy & Hold: shows if AI adds value beyond holding stock.  
                    - Use this to judge whether AI is really offering an **edge** or not.
                    """)
                else:
                    st.info("LSTM model or test data not available for benchmarking.")
            except Exception as e:
                st.warning(f"Benchmarking failed: {e}")

            # -------- Transparency --------
            st.header("🔍 Transparency & Details")
            colA, colB = st.columns([1, 1])
            with colA:
                st.subheader("Latest Headlines (live)")
                if advisor.headlines_df is not None and not advisor.headlines_df.empty:
                    st.dataframe(advisor.headlines_df, use_container_width=True, height=240)
                else:
                    st.info("No headlines available.")
            with colB:
                st.subheader("Model Evaluation")
                if advisor.lstm_model is not None and advisor.X_lstm_test is not None and advisor.y_lstm_test is not None:
                    yhat = (advisor.lstm_model.predict(advisor.X_lstm_test, verbose=0).ravel() > 0.5).astype(int)
                    plot_confmat(advisor.y_lstm_test, yhat, f"{symbol} — LSTM")
                else:
                    st.info("LSTM model or test data not available for evaluation.")

            st.subheader("Training Data Preview")
            st.dataframe(advisor.training_df.head(20), use_container_width=True, height=300)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Set your inputs on the left and click **Run Analysis**.")
