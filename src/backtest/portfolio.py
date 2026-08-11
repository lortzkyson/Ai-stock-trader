"""Monthly-rebalance portfolio backtester for cross-sectional strategies.

Distinct from `backtest/engine.py` (event-driven, per-symbol, intraday
stop/target exits) because the strategy shape is genuinely different: hold a
basket of N names, rebalance on a schedule, no per-trade barriers. Shared
assumptions with the intraday engine are reused, not re-derived — Alpaca's
real fee schedule comes from `backtest/costs.py`.

Execution realism:
- The signal is computed from the *close* of the rebalance date; the trade
  fills at the *next* trading day's open. Never same-bar.
- Slippage is applied adversely in both directions (buys fill higher, sells
  lower).
- Sells pay SEC + FINRA TAF fees per Alpaca's actual schedule.
- Positions are whole shares; leftover value stays in cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.costs import sell_regulatory_fees


@dataclass
class PortfolioBacktestConfig:
    starting_equity: float = 10_000.0
    slippage_bps: float = 5.0


@dataclass
class PortfolioBacktestResult:
    daily_equity: pd.Series
    trades: pd.DataFrame
    turnover: pd.Series
    holdings_history: list[dict] = field(default_factory=list)


def run_portfolio_backtest(
    targets: dict[pd.Timestamp, list[str]],
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    config: PortfolioBacktestConfig | None = None,
) -> PortfolioBacktestResult:
    config = config or PortfolioBacktestConfig()
    slip = config.slippage_bps / 10_000

    dates = list(close_prices.index)
    date_position = {d: i for i, d in enumerate(dates)}

    # Map each signal date to the next trading day, where the trade actually fills.
    execution_plan: dict[pd.Timestamp, list[str]] = {}
    for signal_date, symbols in targets.items():
        i = date_position.get(signal_date)
        if i is None or i + 1 >= len(dates):
            continue  # no next session to fill on (end of data)
        execution_plan[dates[i + 1]] = symbols

    cash = config.starting_equity
    holdings: dict[str, int] = {}
    equity_by_date: dict[pd.Timestamp, float] = {}
    turnover_by_date: dict[pd.Timestamp, float] = {}
    trades: list[dict] = []
    holdings_history: list[dict] = []

    for current_date in dates:
        if current_date in execution_plan:
            target_symbols = execution_plan[current_date]
            cash, holdings, executed, traded_value = _rebalance(
                cash, holdings, target_symbols, open_prices.loc[current_date], slip, current_date
            )
            trades.extend(executed)
            pre_trade_equity = _mark_to_market(cash, holdings, open_prices.loc[current_date])
            turnover_by_date[current_date] = (
                traded_value / pre_trade_equity if pre_trade_equity > 0 else 0.0
            )
            holdings_history.append(
                {"date": current_date, "n_positions": len(holdings), "symbols": sorted(holdings)}
            )

        equity_by_date[current_date] = _mark_to_market(
            cash, holdings, close_prices.loc[current_date]
        )

    return PortfolioBacktestResult(
        daily_equity=pd.Series(equity_by_date).sort_index(),
        trades=pd.DataFrame(trades),
        turnover=pd.Series(turnover_by_date).sort_index(),
        holdings_history=holdings_history,
    )


def _mark_to_market(cash: float, holdings: dict[str, int], prices: pd.Series) -> float:
    value = cash
    for symbol, shares in holdings.items():
        price = prices.get(symbol, np.nan)
        if not np.isnan(price):
            value += shares * price
    return value


def _rebalance(
    cash: float,
    holdings: dict[str, int],
    target_symbols: list[str],
    execution_prices: pd.Series,
    slip: float,
    current_date: pd.Timestamp,
) -> tuple[float, dict[str, int], list[dict], float]:
    executed: list[dict] = []
    traded_value = 0.0

    # 1. Sell everything not in the new target (and anything with no price today).
    for symbol in list(holdings):
        price = execution_prices.get(symbol, np.nan)
        if symbol in target_symbols and not np.isnan(price):
            continue
        if np.isnan(price):
            # Delisted / halted with no print: drop the position without booking
            # a fictional fill. Flagged as a trade with price NaN so it's visible
            # in the log rather than silently vanishing.
            executed.append(
                {"date": current_date, "symbol": symbol, "side": "drop_no_price",
                 "shares": holdings[symbol], "price": np.nan, "fees": 0.0}
            )
            del holdings[symbol]
            continue

        shares = holdings.pop(symbol)
        fill = price * (1 - slip)
        proceeds = shares * fill
        fees = sell_regulatory_fees(shares, fill)
        cash += proceeds - fees
        traded_value += proceeds
        executed.append(
            {"date": current_date, "symbol": symbol, "side": "sell",
             "shares": shares, "price": fill, "fees": fees}
        )

    # 2. Equal-weight the target names against total post-sale equity.
    investable = [s for s in target_symbols if not np.isnan(execution_prices.get(s, np.nan))]
    if not investable:
        return cash, holdings, executed, traded_value

    equity = _mark_to_market(cash, holdings, execution_prices)
    target_value_each = equity / len(investable)

    # 2a. Trim positions that are now overweight, freeing cash before buying.
    for symbol in list(holdings):
        if symbol not in investable:
            continue
        price = execution_prices[symbol]
        target_shares = int(target_value_each // price)
        if target_shares < holdings[symbol]:
            shares = holdings[symbol] - target_shares
            fill = price * (1 - slip)
            proceeds = shares * fill
            fees = sell_regulatory_fees(shares, fill)
            cash += proceeds - fees
            traded_value += proceeds
            holdings[symbol] = target_shares
            if holdings[symbol] == 0:
                del holdings[symbol]
            executed.append(
                {"date": current_date, "symbol": symbol, "side": "sell",
                 "shares": shares, "price": fill, "fees": fees}
            )

    # 2b. Buy up to target weight.
    for symbol in investable:
        price = execution_prices[symbol]
        fill = price * (1 + slip)
        current_shares = holdings.get(symbol, 0)
        target_shares = int(target_value_each // fill)
        to_buy = target_shares - current_shares
        if to_buy <= 0:
            continue
        cost = to_buy * fill
        if cost > cash:
            to_buy = int(cash // fill)
            cost = to_buy * fill
        if to_buy <= 0:
            continue
        cash -= cost
        traded_value += cost
        holdings[symbol] = current_shares + to_buy
        executed.append(
            {"date": current_date, "symbol": symbol, "side": "buy",
             "shares": to_buy, "price": fill, "fees": 0.0}
        )

    return cash, holdings, executed, traded_value
