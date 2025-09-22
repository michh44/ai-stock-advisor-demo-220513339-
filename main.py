import argparse
from advisor import StockAdvisor, StockLSTMAdvisor

def run_logistic(symbol, start, end):
    advisor = StockAdvisor(symbol, start, end)
    advisor.fetch_stock_data()
    advisor.fetch_headlines()
    advisor.calculate_features()
    advisor.analyze_sentiment()
    advisor.build_training_data()
    advisor.train_model()
    advisor.predict()
    print(advisor.generate_advice())
    advisor.plot_confusion_matrix()
    # advisor.plot_predicted_vs_actual()

def run_lstm(symbol, start, end):
    advisor = StockLSTMAdvisor(symbol, start, end)
    advisor.fetch_stock_data()
    advisor.fetch_headlines()
    advisor.calculate_features()
    advisor.analyze_sentiment()
    advisor.build_training_data()
    advisor.prepare_lstm_data()
    advisor.train_lstm(epochs=10)
    advisor.evaluate_lstm()
    advisor.predict_lstm()
    print(advisor.generate_lstm_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["logistic", "lstm"], default="logistic")
    args = parser.parse_args()

    if args.model == "logistic":
        run_logistic(args.symbol, args.start, args.end)
    else:
        run_lstm(args.symbol, args.start, args.end)
