
"""
Découverte causale

"""
from __future__ import annotations
import os
import sys
import numpy as np
import networkx as nx
import lingam

# 1)

def load_prices_matrix(path: str) -> np.ndarray:

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim != 2:
        raise ValueError(f"CSV mal formé (pas 2D) : {path}")
    return data



# 2) VARLiNGAM

def fit_varlingam(prices: np.ndarray, lag: int) -> np.ndarray:

    model = lingam.VARLiNGAM(lags=lag)
    model.fit(prices)
    A_all = np.asarray(model.adjacency_matrices_)  # (K, N, N)
    # Diagnostic rapide :
    nnz_per_slice = [int((np.abs(A_all[k]) > 0).sum()) for k in range(A_all.shape[0])]
    print(f"[INFO] Slices K={A_all.shape[0]} (attendu {lag} ou {lag}+1) | nonzeros/lag: {nnz_per_slice}")
    return A_all


def build_summary_matrix(A_all: np.ndarray, lag: int) -> np.ndarray:

    K, N, _ = A_all.shape

    if K == lag + 1:
        A_lags = A_all[1:]           # k = 1..L
    else:
        A_lags = A_all               # déjà uniquement les lags retardés

    summary = np.sum(np.abs(A_lags), axis=0).copy()  # (N, N)
    np.fill_diagonal(summary, 0.0)                   # no self-loop
    return summary



# 3) Graphe orienté

def summary_graph_from_matrix(summary: np.ndarray) -> nx.DiGraph:

    G = nx.from_numpy_array(summary.T, create_using=nx.DiGraph)

    for _, _, d in G.edges(data=True):
        d.clear()
    return G


def print_graph_diagnostics(G: nx.DiGraph) -> None:
    N = G.number_of_nodes()
    E = G.number_of_edges()
    density = E / (N * (N - 1)) if N > 1 else 0.0
    indeg = [d for _, d in G.in_degree()]
    outdeg = [d for _, d in G.out_degree()]
    q = lambda a, p: sorted(a)[int(p * (len(a) - 1))] if a else 0
    print(f"[DIAG] nodes={N} | edges={E} | densité≈{density:.3f}")
    print(f"       in-degree  : médiane={q(indeg,0.5)} | P90={q(indeg,0.9)} | max={max(indeg) if indeg else 0}")
    print(f"       out-degree : médiane={q(outdeg,0.5)} | P90={q(outdeg,0.9)} | max={max(outdeg) if outdeg else 0}")



# 4) Main

def main():

    if len(sys.argv) < 5:
        print("Usage: python causal_discovery_varlingam.py <data_file> <num_lags> <market_name> <algorithm>")
        sys.exit(1)

    data_file   = sys.argv[1]
    num_lags    = int(sys.argv[2])
    market_name = sys.argv[3]
    algorithm   = sys.argv[4]

    out_dir = "./causal_graphs/graphs"
    os.makedirs(out_dir, exist_ok=True)


    prices = load_prices_matrix(data_file)
    print(f"[RUN] market={market_name} | lag={num_lags} | data={prices.shape} | algo={algorithm}")


    A_all   = fit_varlingam(prices, num_lags)
    summary = build_summary_matrix(A_all, num_lags)


    G = summary_graph_from_matrix(summary)
    print_graph_diagnostics(G)


    out_path = os.path.join(out_dir, f"{market_name}_graph_{algorithm}_lag_{num_lags}.txt")
    nx.write_edgelist(G, out_path, data=False, delimiter=" ")
    print(f"[SAVE] {out_path}")


    out_csv = os.path.join(out_dir, f"{market_name}_graph_{algorithm}_lag_{num_lags}.csv")
    edges = np.array(list(G.edges()), dtype=int)
    np.savetxt(out_csv, edges, delimiter=",", fmt="%d", header="src,dst", comments="")
    print(f"[SAVE] {out_csv}")

if __name__ == "__main__":
    main()