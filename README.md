# AI Stock Advisor (CM3020 – Project Idea 2) - 220513339

Hi 👋  

This is my final project for CM3020, where I built an **AI-powered financial advisor bot**.  
The goal is to create a tool that can analyze stock prices and news sentiment, then provide investment recommendations (Buy / Hold / Sell) in a way that’s easy for non-technical users.

---

## What the project does
- Lets me input a stock ticker and time range.  
- Runs two different models:  
  - **Logistic Regression** (classical machine learning).  
  - **LSTM** (deep learning, sequence-based).  
- Uses both **price data** and **news sentiment analysis**.  
- Produces:  
  - A clear recommendation (Buy, Hold, or Sell).  
  - Confidence score.  
  - Backtest results vs Buy & Hold.  
  - Walk-forward validation (robustness).  
  - Benchmarking against random and baseline strategies.  

---

## Why I built it
I wanted to test how well machine learning and deep learning can adapt to real financial markets.  
By combining classical ML (Logistic Regression) and modern DL (LSTM), I could compare performance and understand trade-offs in risk, accuracy, and robustness.

---

## How to use
### 1. Online demo
You can try the app directly here:  
👉 [Streamlit App](https://project-final-mich.streamlit.app/)

