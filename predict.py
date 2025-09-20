
"""
PREDICT

"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression

# 1) Prédiction

def predict_single(data, lag, G, stock_index, train_frac=0.8):

    predicted = []


    causes_name = list(G.predecessors(str(stock_index)))
    causes_index = [int(i) for i in causes_name]


    causes = data[:, causes_index] if len(causes_index) > 0 else np.empty((data.shape[0], 0))
    target = data[:, stock_index]


    Y = target[lag:]


    X = np.empty((0, lag * max(1, len(causes_index))))
    for t in range(lag, data.shape[0]):
        if causes.size > 0:

            lagged = causes[(t - lag):t, :]
            features = np.concatenate(lagged)
        else:

            features = np.zeros(lag)
        X = np.vstack([X, features])


    train_length = int(len(Y) * train_frac)


    for t in range(train_length, len(Y)):
        model = LinearRegression()
        model.fit(X[:t, :], Y[:t])
        y_hat = model.predict(X[t, :].reshape(1, -1))
        predicted.append(float(y_hat[0]))

    return predicted



# 2) Prédictions pour TOUTES les colonnes (j=0..N-1)

def predict_batch(data, lag, G, train_frac=0.8):

    predictions = np.empty((0, 0))
    for j in range(data.shape[1]):
        print(f"[PRED] actif {j}")
        col_pred = predict_single(data, lag, G, j, train_frac)
        col_pred = np.asarray(col_pred).reshape(-1, 1)
        predictions = np.hstack((predictions, col_pred)) if predictions.size else col_pred
    return predictions



# 3) train_frac

def _parse_train_frac(argv, default=0.8):

    if "--train_frac" in argv:
        i = argv.index("--train_frac")
        if i + 1 < len(argv):
            try:
                return float(argv[i + 1])
            except ValueError:
                pass
    return default



# 4) Main

if __name__ == "__main__":

    if len(sys.argv) < 6:
        print("Usage: python predict.py data.csv graph.txt L market algo [--train_frac 0.8]")
        sys.exit(1)


    output_directory = "./predictions"
    os.makedirs(output_directory, exist_ok=True)


    data_filename        = sys.argv[1]
    causal_graph_filename= sys.argv[2]
    num_lags             = int(sys.argv[3])
    market_name          = sys.argv[4]
    algorithm            = sys.argv[5]


    train_frac = _parse_train_frac(sys.argv, default=0.8)
    print(f"[CONFIG] market={market_name} | lag={num_lags} | algo={algorithm} | train_frac={train_frac:.2f}")

    df = pd.read_csv(data_filename, delimiter=",", header=0)
    tickers = df.columns.tolist()
    data = df.to_numpy()


    G = nx.read_adjlist(causal_graph_filename, create_using=nx.DiGraph)


    # Prédictions "causal"

    print("[RUN] Making Predictions (causal)")
    preds = predict_batch(data, num_lags, G, train_frac=train_frac)
    out_path = os.path.join(output_directory, f"{market_name}_predictions_{algorithm}_lag_{num_lags}.csv")
    np.savetxt(out_path, preds, delimiter=",", header=",".join(tickers), comments="")
    print("[SAVE]", out_path)


    # Baseline "self-cause only"
    summary_matrix = np.eye(data.shape[1])
    G_self = nx.from_numpy_array(summary_matrix.T, create_using=nx.DiGraph)

    for _, _, d in G_self.edges(data=True):
        d.pop("weight", None)
    G_self = nx.relabel_nodes(G_self, lambda x: str(x))

    print("[RUN] Making Predictions (self-cause)")
    preds_self = predict_batch(data, num_lags, G_self, train_frac=train_frac)
    out_self = os.path.join(output_directory, f"{market_name}_predictions_self_{algorithm}_lag_{num_lags}.csv")
    np.savetxt(out_self, preds_self, delimiter=",", header=",".join(tickers), comments="")
    print("[SAVE]", out_self)
