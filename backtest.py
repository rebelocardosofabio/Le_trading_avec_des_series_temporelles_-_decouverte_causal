
"""
BACKTEST — Long/Short

"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# 1) Cœur du backtest

def calculate_annualized_portfolio_returns(data: np.ndarray,
                                           predictions: np.ndarray,
                                           num_winners: int,
                                           cost_per_day: float = 0.001):


    data_backtest = data[-(predictions.shape[0] + 1):]
    predicted_returns = (predictions - data_backtest[:-1]) / data_backtest[:-1]
    real_returns      = (data_backtest[1:] - data_backtest[:-1]) / data_backtest[:-1]
    winners = np.argpartition(predicted_returns, -num_winners, axis=1)[:, -num_winners:]
    losers  = np.argpartition(predicted_returns,  num_winners,  axis=1)[:, :num_winners]
    portfolio_returns = []
    for t in range(predictions.shape[0]):
        winner_return = np.mean(real_returns[t, winners[t]])
        loser_return  = np.mean(real_returns[t, losers[t]])
        daily_pnl     = winner_return - loser_return - cost_per_day
        portfolio_returns.append(daily_pnl)

    cumulative_portfolio_return = np.exp(np.sum(np.log(np.array(portfolio_returns) + 1.0))) - 1.0
    annualized_return = (1.0 + cumulative_portfolio_return) ** (252.0 / predictions.shape[0]) - 1.0

    return annualized_return, winners, losers, portfolio_returns


# 2) Main

if __name__ == "__main__":

    if len(sys.argv) < 7:
        print("Usage: data_file prediction_file num_lags market_name algorithm test_winner_num")
        sys.exit(1)
    output_directory = "./backtesting"
    os.makedirs(output_directory, exist_ok=True)

    data_filename        = sys.argv[1]
    predictions_filename = sys.argv[2]
    num_lags             = int(sys.argv[3])
    market_name          = sys.argv[4]
    algorithm            = sys.argv[5]
    test_winner_num      = int(sys.argv[6])

    data = np.genfromtxt(data_filename, delimiter=",", skip_header=1)
    predictions = np.genfromtxt(predictions_filename, delimiter=",", skip_header=1)
    predictions_self_filename = predictions_filename.replace("_predictions_", "_predictions_self_")
    predictions_self = np.genfromtxt(predictions_self_filename, delimiter=",", skip_header=1)

    print(f"[RUN] Backtesting market={market_name} | lag={num_lags} | algo={algorithm}")


    # 2.1) Balayage de k

    n_assets = data.shape[1]
    n_winners_range = np.arange(1, max(int(0.2 * n_assets), 5))

    ar      = [calculate_annualized_portfolio_returns(data, predictions,      k)[0]
               for k in n_winners_range]
    ar_self = [calculate_annualized_portfolio_returns(data, predictions_self, k)[0]
               for k in n_winners_range]


    backtest_output_filename = os.path.join(
        output_directory,
        f"{market_name}_backtest_returns_{algorithm}_lag_{num_lags}.csv"
    )
    backtest_returns = np.column_stack((n_winners_range, ar, ar_self))
    with open(backtest_output_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["winner_num", "ar", "ar_self"])
        writer.writerows(backtest_returns)
    print("[SAVE]", backtest_output_filename)

    plt.figure(figsize=(7, 5))
    plt.plot(n_winners_range, ar,      label="Causal discovery", linewidth=2, color="C0")
    plt.plot(n_winners_range, ar_self, label="Self-cause only",  linewidth=2, color="C1", linestyle="--")
    plt.xlabel("Number of Winners/Losers (k)")
    plt.ylabel("Annualized return")
    plt.legend()
    plt.tight_layout()
    backtest_plot_filename = os.path.join(
        output_directory,
        f"{market_name}_backtest_returns_plot_{algorithm}_lag_{num_lags}.png"
    )
    plt.savefig(backtest_plot_filename, dpi=160, bbox_inches="tight")
    plt.close()
    print("[SAVE]", backtest_plot_filename)


    # 2.2) Sauvegardes

    winner_filename = os.path.join(
        output_directory,
        f"{market_name}_{test_winner_num}_winners_{algorithm}_lag_{num_lags}.csv"
    )
    loser_filename  = os.path.join(
        output_directory,
        f"{market_name}_{test_winner_num}_losers_{algorithm}_lag_{num_lags}.csv"
    )
    daily_port_return_filename = os.path.join(
        output_directory,
        f"{market_name}_daily_portfolio_returns_{test_winner_num}_{algorithm}_lag_{num_lags}.csv"
    )

    _, winners, losers, daily_port_returns = calculate_annualized_portfolio_returns(
        data, predictions, test_winner_num
    )

    with open(winner_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(winners)

    with open(loser_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(losers)


    with open(daily_port_return_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["daily_portfolio_return"])
        for x in daily_port_returns:
            writer.writerow([x])

    print("[SAVE] winners/losers & daily returns for k =", test_winner_num)
