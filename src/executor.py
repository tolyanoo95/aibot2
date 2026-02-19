"""
Trade executor: paper trading and live order execution.
Controlled by TRADING_MODE in config: "paper" or "live".
"""

import logging
import json
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

PAPER_LOG = "paper_trades.json"


class TradeExecutor:
    """Execute trades — either paper (log only) or live (Binance orders)."""

    def __init__(self, config, exchange=None):
        self.config = config
        self.mode = config.TRADING_MODE  # "paper" or "live"
        self.exchange = exchange
        self._open_positions: dict = {}  # symbol → position info
        self._paper_trades: list = []
        self._daily_pnl: float = 0.0
        self._last_direction: dict = {}  # symbol → last closed direction

        if self.mode == "live" and not self.exchange:
            logger.error("Live mode requires exchange instance!")
            self.mode = "paper"

        self._load_paper_log()

    # ── public API ───────────────────────────────────────────

    def open_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        leverage: int,
        confidence: float,
    ) -> bool:
        """Open a trade. Returns True if successful."""

        # safety checks
        if symbol in self._open_positions:
            logger.warning("Already have open position on %s", symbol)
            return False

        if len(self._open_positions) >= self.config.MAX_OPEN_TRADES:
            logger.warning("Max open trades (%d) reached", self.config.MAX_OPEN_TRADES)
            return False

        # don't re-enter same direction on same pair (wait for opposite)
        no_reentry = os.getenv("NO_SAME_DIRECTION_REENTRY", "false").lower() == "true"
        if no_reentry and self._last_direction.get(symbol) == direction:
            logger.info("Skip %s %s — same direction as last trade, waiting for opposite", direction, symbol)
            return False

        max_same = getattr(self.config, "MAX_SAME_DIRECTION", 10)
        same_dir_count = sum(1 for p in self._open_positions.values() if p.get("direction") == direction)
        if same_dir_count >= max_same:
            logger.warning("Max same direction (%d %s) reached", max_same, direction)
            return False

        # position size
        balance = self.config.TRADE_BALANCE_USDT
        risk_amount = balance * self.config.RISK_PER_TRADE
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance == 0:
            return False
        qty = risk_amount / sl_distance
        qty_usdt = qty * entry_price

        # round prices to match exchange tick size
        decimals = self._price_decimals(entry_price)
        trade_info = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": round(entry_price, decimals),
            "stop_loss": round(stop_loss, decimals),
            "take_profit": round(take_profit, decimals),
            "leverage": leverage,
            "confidence": confidence,
            "quantity": round(qty, 6),
            "quantity_usdt": round(qty_usdt, 2),
            "open_time": datetime.now().isoformat(),
            "status": "OPEN",
        }

        if self.mode == "paper":
            return self._paper_open(trade_info)
        else:
            return self._live_open(trade_info)

    def close_trade(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
    ) -> Optional[dict]:
        """Close a trade. Returns trade result or None."""
        if symbol not in self._open_positions:
            return None

        pos = self._open_positions[symbol]

        if pos["direction"] == "LONG":
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        else:
            pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"] * 100

        pnl_pct -= 0.08  # commission

        result = {
            **pos,
            "exit_price": exit_price,
            "exit_time": datetime.now().isoformat(),
            "exit_reason": reason,
            "pnl_pct": round(pnl_pct, 4),
        }
        result["status"] = "CLOSED"

        self._daily_pnl += pnl_pct

        if self.mode == "paper":
            self._paper_close(result)
        else:
            self._live_close(result)

        self._last_direction[symbol] = pos["direction"]
        del self._open_positions[symbol]
        return result

    def get_open_positions(self) -> dict:
        return self._open_positions.copy()

    def has_position(self, symbol: str) -> bool:
        return symbol in self._open_positions

    def check_daily_limit(self) -> bool:
        """Return True if daily loss limit NOT reached."""
        max_loss = self.config.MAX_DAILY_LOSS * 100
        return self._daily_pnl > -max_loss

    # ── paper trading ────────────────────────────────────────

    def _paper_open(self, trade_info: dict) -> bool:
        self._open_positions[trade_info["symbol"]] = trade_info
        self._paper_trades.append(trade_info)
        self._save_paper_log()

        logger.info(
            "[PAPER] Opened %s %s @ %s (qty: %s USDT, SL: %s, TP: %s)",
            trade_info["direction"],
            trade_info["symbol"],
            trade_info["entry_price"],
            trade_info["quantity_usdt"],
            trade_info["stop_loss"],
            trade_info["take_profit"],
        )
        return True

    def _paper_close(self, result: dict):
        # update in paper log
        for i, t in enumerate(self._paper_trades):
            if (t["symbol"] == result["symbol"]
                    and t["status"] == "OPEN"
                    and t["open_time"] == result["open_time"]):
                self._paper_trades[i] = result
                break
        self._save_paper_log()

        pnl_c = "+" if result["pnl_pct"] > 0 else ""
        logger.info(
            "[PAPER] Closed %s %s @ %s (%s) PnL: %s%.2f%%",
            result["direction"],
            result["symbol"],
            result["exit_price"],
            result["exit_reason"],
            pnl_c,
            result["pnl_pct"],
        )

    def _load_paper_log(self):
        if os.path.exists(PAPER_LOG):
            try:
                with open(PAPER_LOG) as f:
                    self._paper_trades = json.load(f)
                # restore open positions
                for t in self._paper_trades:
                    if t.get("status") == "OPEN":
                        self._open_positions[t["symbol"]] = t
            except Exception:
                self._paper_trades = []

    def _save_paper_log(self):
        with open(PAPER_LOG, "w") as f:
            json.dump(self._paper_trades, f, indent=2, default=str)

    # ── live trading ─────────────────────────────────────────

    @staticmethod
    def _price_decimals(price: float) -> int:
        """Guess appropriate decimal places from price magnitude."""
        if price >= 10000:    return 2   # BTC: 67141.00
        if price >= 100:      return 2   # BNB: 615.92
        if price >= 1:        return 4   # ETH: 1965.25, XRP: 1.4567
        if price >= 0.01:     return 5   # DOGE: 0.09979
        return 6                         # very small prices

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange precision for the symbol."""
        try:
            return float(self.exchange.price_to_precision(symbol, price))
        except Exception:
            return price

    def _round_qty(self, symbol: str, qty: float) -> float:
        """Round quantity to exchange precision for the symbol."""
        try:
            return float(self.exchange.amount_to_precision(symbol, qty))
        except Exception:
            return qty

    def _live_open(self, trade_info: dict) -> bool:
        try:
            symbol = trade_info["symbol"]
            side = "buy" if trade_info["direction"] == "LONG" else "sell"
            qty = self._round_qty(symbol, trade_info["quantity"])

            # set leverage
            self.exchange.set_leverage(trade_info["leverage"], symbol)

            # market order
            order = self.exchange.create_market_order(
                symbol, side, qty,
            )
            logger.info("[LIVE] Market order: %s", order)

            # recalculate SL/TP based on actual fill price
            fill_price = float(order.get("average", 0) or order.get("price", 0))
            if fill_price > 0 and fill_price != trade_info["entry_price"]:
                old_entry = trade_info["entry_price"]
                sl_dist = abs(old_entry - trade_info["stop_loss"])
                tp_dist = abs(old_entry - trade_info["take_profit"])

                if trade_info["direction"] == "LONG":
                    trade_info["stop_loss"] = fill_price - sl_dist
                    trade_info["take_profit"] = fill_price + tp_dist
                else:
                    trade_info["stop_loss"] = fill_price + sl_dist
                    trade_info["take_profit"] = fill_price - tp_dist

                logger.info(
                    "[LIVE] Recalculated SL/TP: entry %.6g→%.6g, SL=%.6g, TP=%.6g",
                    old_entry, fill_price,
                    trade_info["stop_loss"], trade_info["take_profit"],
                )
                trade_info["entry_price"] = fill_price

            # round prices to exchange precision
            sl_price = self._round_price(symbol, trade_info["stop_loss"])
            tp_price = self._round_price(symbol, trade_info["take_profit"])
            trade_info["stop_loss"] = sl_price
            trade_info["take_profit"] = tp_price

            # set SL
            sl_side = "sell" if trade_info["direction"] == "LONG" else "buy"
            self.exchange.create_order(
                symbol, "STOP_MARKET", sl_side, qty,
                params={"stopPrice": sl_price, "closePosition": True},
            )

            # set TP
            tp_side = sl_side
            self.exchange.create_order(
                symbol, "TAKE_PROFIT_MARKET", tp_side, qty,
                params={"stopPrice": tp_price, "closePosition": True},
            )

            trade_info["order_id"] = order.get("id")
            self._open_positions[symbol] = trade_info

            logger.info(
                "[LIVE] Opened %s %s @ %.6g (qty: %s, SL: %.6g, TP: %.6g)",
                trade_info["direction"], symbol, trade_info["entry_price"],
                qty, trade_info["stop_loss"], trade_info["take_profit"],
            )
            return True

        except Exception as exc:
            logger.error("[LIVE] Order failed: %s", exc)
            return False

    def _live_close(self, result: dict):
        try:
            symbol = result["symbol"]
            side = "sell" if result["direction"] == "LONG" else "buy"

            # cancel existing SL/TP orders
            self.exchange.cancel_all_orders(symbol)

            # close position at market
            pos = self.exchange.fetch_position(symbol)
            if pos and float(pos.get("contracts", 0)) > 0:
                qty = float(pos["contracts"])
                self.exchange.create_market_order(symbol, side, qty, params={"reduceOnly": True})

            logger.info(
                "[LIVE] Closed %s %s (%s) PnL: %.2f%%",
                result["direction"], symbol, result["exit_reason"], result["pnl_pct"],
            )
        except Exception as exc:
            logger.error("[LIVE] Close failed: %s", exc)

    # ── stats ────────────────────────────────────────────────

    def get_paper_stats(self) -> dict:
        """Summary of paper trading results."""
        closed = [t for t in self._paper_trades if t.get("status") == "CLOSED"]
        if not closed:
            return {"trades": 0}

        wins = sum(1 for t in closed if t.get("pnl_pct", 0) > 0)
        total_pnl = sum(t.get("pnl_pct", 0) for t in closed)

        return {
            "trades": len(closed),
            "wins": wins,
            "losses": len(closed) - wins,
            "win_rate": wins / len(closed) * 100 if closed else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(closed), 3) if closed else 0,
        }
