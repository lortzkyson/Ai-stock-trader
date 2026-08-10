"""Live monitoring: equity, open positions, and today's day-trade count vs.
the PDT limit — the things docs/runbook.md says to check on a normal day.
"""

from __future__ import annotations

from datetime import date

from execution.client import AlpacaExecutionClient
from risk.pdt import MAX_DAY_TRADES_PER_5_DAYS, DayTradeTracker


def render_dashboard(
    client: AlpacaExecutionClient,
    day_trade_tracker: DayTradeTracker,
    trading_calendar: list[date],
    as_of: date | None = None,
) -> str:
    as_of = as_of or date.today()
    equity = client.get_account_equity()
    positions = client.get_positions()
    day_trade_count = day_trade_tracker.day_trade_count_in_window(as_of, trading_calendar)

    lines = [
        f"Account equity: ${equity:,.2f}",
        f"Open positions: {len(positions)}",
    ]
    for p in positions:
        pnl = float(p.unrealized_pl)
        lines.append(f"  {p.symbol}: {p.qty} shares, unrealized P&L ${pnl:,.2f}")
    lines.append(
        f"Day trades (rolling 5-day window): {day_trade_count}/{MAX_DAY_TRADES_PER_5_DAYS}"
    )
    if day_trade_count >= MAX_DAY_TRADES_PER_5_DAYS:
        lines.append(
            "  -> PDT budget spent: new entries will be blocked until the window rolls forward."
        )
    return "\n".join(lines)
