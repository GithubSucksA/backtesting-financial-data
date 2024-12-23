# main_backtest.py
import argparse
import numpy as np
from base_trading_backtester import BaseTradingBacktester
import pandas as pd
from aostrat import AoBacktester
from apostrat import ApoBacktester
from biasstrat import BiasBacktester
from bosstrat import BosBacktester
from brarstrat import BrarBacktester
from ccistrat import CciBacktester
from cfostrat import CfoBacktester
from gcstrat import GcBacktester
from mappings import symbol_mapping, interval_mapping

def run_backtester(strategy: str, symbol: str, interval: str, initial_balance: float, risk_per_trade: float, data_source: str = 'kraken', buy_signal=None, sell_signal=None):
    strategies = {
        "ao": AoBacktester,
        "apo": ApoBacktester,
        "bias": BiasBacktester,
        "bos": BosBacktester,
        "brar": BrarBacktester,
        "cci": CciBacktester,
        "cfo": CfoBacktester,
        "gc": GcBacktester
    }
    
    if strategy in strategies:
        backtester = strategies[strategy](symbol, interval, initial_balance, risk_per_trade, data_source, buy_signal, sell_signal)
    else:
        raise ValueError("Invalid strategy name. Choose a valid strategy.")

    return backtester

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a backtesting strategy.")
    parser.add_argument("--strategy", type=str, nargs='*', help="Strategy to use (leave empty to run all strategies).")
    parser.add_argument("--source", type=str, default='kraken', help="Source to retrieve data from")  # Set default value here
    parser.add_argument("--symbol", type=str, nargs='+', help="Symbol for trading pair.")
    parser.add_argument("--interval", type=str, nargs='+', help="Time interval for trading data.")
    parser.add_argument("--initial_balance", type=float, default=10000, help="Initial balance for the backtest.")
    parser.add_argument("--risk_per_trade", type=float, default=0.05, help="Risk per trade as a decimal.")

    args = parser.parse_args()

    # Set default strategies if none are provided
    if not args.strategy:
        args.strategy = ["ao", "apo", "bias", "bos", "brar", "cci", "cfo", "gc"]

    intervals = []
    # Set intervals based on data source
    if not args.interval:
        if args.source == 'yahoo':
            intervals = ['1d', '1wk']
        elif args.source == 'kraken':
            intervals = [15, 30, 60, 240, 1440]
        elif args.source == 'htx':
            intervals = ['15min', '30min', '60min', '4hour', '1day']
    else:
        # Normalize intervals based on the source
        if args.source in interval_mapping:  # Check if the source is valid
            intervals = [interval_mapping[args.source].get(i, None) for i in args.interval]
            intervals = [i for i in intervals if i is not None]  # Filter out None values

    # Normalize symbols based on the source
    if args.source in symbol_mapping:
        if args.symbol:  # Check if symbols are provided
            symbols = [symbol_mapping[args.source].get(sym, sym) for sym in args.symbol]
        else:
            symbols = ['BTC/USD', 'ETH/USD']  # Default symbols if none provided
    elif args.symbol:
        symbols = args.symbol

    all_results = []

    # Define different buy and sell signals for testing
    buy_sell_configs = [
        {'buy_signal': 'signal1', 'sell_signal': 'signal1'},
        {'buy_signal': 'signal2', 'sell_signal': 'signal2'},
        {'buy_signal': 'signal3', 'sell_signal': 'signal3'},
    ]

    for interval in intervals:
        print(f"\n{interval}")
        interval_results = []
        date_range_printed = False
        interval_total_trades = 0
        interval_winning_trades = 0
        interval_trading_volume = 0

        for symbol in symbols:
            for strat in args.strategy:  # Loop through each specified strategy
                for config in buy_sell_configs:
                    try:
                        backtester = run_backtester(strat, symbol, interval, 10000, 0.05, args.source, config['buy_signal'], config['sell_signal'])
                        # Assuming backtester.run_backtest() returns results
                        results = backtester.run_backtest()  # Call the method to get results
                        #results['strategy'] = f"{strat} ({config['buy_signal']}, {config['sell_signal']})"  # Add the strategy and signals to the results
                        results['strategy'] = f"{strat}"
                        interval_results.append(results)  # Collect results for this interval
                    except Exception as e:
                        print(e)
                        print(f"Error running backtest for {symbol} with interval {interval}: {e}")
        
        # Sort interval results by total return
        if interval_results:  # Check if there are results to process
            sorted_results = sorted(interval_results, key=lambda x: x['total_return'], reverse=True)
            
            # Display top performers for this interval
            print(f"\nTop {len(sorted_results)} performers for interval {interval}:")
            print(f"{'Rank':<5} {'Symbol':<10} {'Strategy':<10} {'Total Return':<20} {'Win Rate':<15} {'Trades Executed':<15} {'Total Fees Paid':<20} {'NET Profit/Loss':<20} {'Trading Volume':<20} {'Sharpe Ratio':<15}")
            print("=" * 160)  # Separator line
            for i, result in enumerate(sorted_results, 1):
                print(f"{i:<5} {result['symbol']:<10} {result['strategy']:<10} {result['total_return']:<20.2%} {result['win_rate']:<15.2%} {result['trades_executed']:<15} {result['total_fees_paid']:<20.2f} {result['net_profit_loss']:<20.2f} {result['trading_volume']:<20.2f} {result['sharpe_ratio']:<15.2f}")
            # Calculate and display average metrics for this interval
            avg_return = np.mean([r['total_return'] for r in interval_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in interval_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in interval_results])
            avg_winrate = np.mean([r['win_rate'] for r in interval_results])
            
            print(f"\nAverage metrics:")
            print(f"Avg Return: {avg_return:.2%}, Avg Sharpe: {avg_sharpe:.2f}, Avg Drawdown: {avg_drawdown:.2%}, Avg Win Rate: {avg_winrate:.2%}")
            print(f"Total trades: {interval_total_trades}, Total winning trades: {interval_winning_trades}")
            print(f"Overall win rate: {interval_winning_trades / interval_total_trades:.2%}" if interval_total_trades > 0 else "No trades executed")
            print(f"Total trading volume: {interval_trading_volume:.2f}")
        else:
            print("No results to display for this interval.")

        print('\n\n')
        all_results.extend(interval_results)
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('backtest_results.csv', index=False)
    print("\nDetailed results saved to 'backtest_results.csv'")
    
    #end_time = time.time()
    #print(f"\nEND TIME {end_time}")
    #total_duration = end_time - start_time
    #print(f"TOTAL TIME {total_duration:.2f} seconds")
