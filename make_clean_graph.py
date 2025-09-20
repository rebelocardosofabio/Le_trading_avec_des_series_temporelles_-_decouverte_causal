# make_clean_graph.py (version clean, sans résumé CSV)

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def find_grid(root, market, lag):
    path = os.path.join(root, f"{market}_backtest_returns_VARLiNGAM_lag_{lag}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Introuvable : {path}")
    return path

def load_grid(path):
    df = pd.read_csv(path)

    cols = {c.lower(): c for c in df.columns}
    req = ["winner_num", "ar", "ar_self"]
    missing = [c for c in req if c not in cols]
    if missing:

        rename = {}
        for c in df.columns:
            lc = c.lower()
            if lc in req:
                rename[c] = lc
        df = df.rename(columns=rename)

    df = df[["winner_num", "ar", "ar_self"]].copy()
    df = df[df["winner_num"] >= 1].sort_values("winner_num")
    return df

def plot_clean(df, market, lag, out_png):
    k = df["winner_num"].values
    ar = df["ar"].values
    ar_self = df["ar_self"].values


    k_star_idx = int(ar.argmax())
    k_star = int(k[k_star_idx])

    plt.figure(figsize=(7,5))
    plt.plot(k, ar,      label="Causal discovery", linewidth=2, color="C0")
    plt.plot(k, ar_self, label="Self-cause only",  linewidth=2, color="C1", linestyle="--")
    plt.axvline(k_star, color="gray", linestyle=":", linewidth=1)
    plt.title(f"AR / k — {market} — VARLiNGAM (lag={lag})")
    plt.xlabel("Number of winners/losers (k)")
    plt.ylabel("Annualized return")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out_png}  |  k*={k_star}")

def main():
    ap = argparse.ArgumentParser(description="Figures 'AR / k' propres (sans résumé).")
    ap.add_argument("--root",   default="backtesting", help="Dossier des grilles backtest (CSV).")
    ap.add_argument("--outdir", default="clean_graphs", help="Dossier de sortie des figures.")
    ap.add_argument("--markets", nargs="*", default=["eurostoxx50","nasdaq100"],
                    help="Marchés à traiter.")
    ap.add_argument("--lags", nargs="*", type=int, default=[1,2],
                    help="Lags à traiter.")
    args = ap.parse_args()

    for market in args.markets:
        for lag in args.lags:
            try:
                csv_path = find_grid(args.root, market, lag)
                df = load_grid(csv_path)
                out_png = os.path.join(args.outdir, f"AR_vs_k_{market}_VARLiNGAM_lag_{lag}.png")
                plot_clean(df, market, lag, out_png)
            except Exception as e:
                print(f"[SKIP] {market}, lag={lag}: {e}")

if __name__ == "__main__":
    main()
