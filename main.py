#!/usr/bin/env python3
"""
AI Crypto Futures Intraday Scanner
───────────────────────────────────
Scans top crypto futures pairs on Binance and generates hybrid
ML + LLM trading signals for intraday (scalp / day-trade) setups.

Now with market context: OI, L/S ratio, stablecoin dominance,
total market cap, and liquidation zone estimation.

Usage:
    python main.py            # run the scanner loop
    python main.py --once     # single scan then exit
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.display import Display
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.llm_analyzer import LLMAnalyzer
from src.market_context import MarketContext
from src.ml_model import MLSignalModel
from src.entry_refiner import EntryRefiner
from src.executor import TradeExecutor
from src.risk_manager import RiskManager
from src.signal_generator import SignalGenerator
from src.trade_monitor import TradeMonitor

# ── logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("scanner.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# detailed trade log for analysis
_trade_logger = logging.getLogger("trades")
_trade_logger.setLevel(logging.INFO)
_trade_handler = logging.FileHandler("trades.log")
_trade_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_trade_logger.addHandler(_trade_handler)
_trade_logger.propagate = False  # don't print to console


class CryptoScanner:
    """Orchestrates the full scan cycle for all configured pairs."""

    def __init__(self):
        self._archive_logs()
        for f in ("scanner.log", "trades.log"):
            if not os.path.exists(f):
                open(f, "a").close()
        self.fetcher = BinanceDataFetcher(config)
        self.indicators = TechnicalIndicators()
        self.features = FeatureEngineer()
        self.ml_model = MLSignalModel(config.ML_MODEL_PATH)
        self.llm = LLMAnalyzer(config) if config.USE_LLM else None
        self.market_ctx = MarketContext(config)
        self.signal_gen = SignalGenerator(config)
        self.entry_refiner = EntryRefiner(config, self.fetcher)
        self.risk_mgr = RiskManager(config)

        self._llm_cache: dict = {}
        self._llm_timestamps: dict[str, float] = {}
        self._active_signals: dict = {}
        self._open_trades: dict = {}
        self._trade_monitor = TradeMonitor(config)
        self.executor = TradeExecutor(
            config,
            exchange=self.fetcher.exchange if config.TRADING_MODE == "live" else None,
        )

    # ── single pair ──────────────────────────────────────────

    def scan_pair(self, symbol: str):
        """Analyse one pair and return a Signal (or None on error)."""
        logger.info("Scanning %s …", symbol)

        data = self.fetcher.fetch_all_data(symbol)
        if data["primary"].empty:
            logger.warning("No primary data for %s — skipping", symbol)
            return None

        # indicators on every timeframe
        for key in ("primary", "secondary", "trend"):
            if not data[key].empty:
                data[key] = self.indicators.calculate_all(data[key])

        # ── market context (OI, L/S, dominance, liq zones) ───
        ctx = self.market_ctx.get_symbol_context(symbol, data["primary"])
        ctx["funding_rate"] = data.get("funding_rate", 0)

        # ── ML features (multi-TF + context) ─────────────────
        primary_with_ctx = self.features.add_realtime_context(
            data["primary"], ctx,
        )
        X, feat_names = self.features.get_feature_matrix_mtf(
            primary_with_ctx, data["secondary"], data["trend"],
        )
        ml_result = self.ml_model.predict(X)

        # ── LLM analysis (rate-limited, with full context) ───
        llm_result = self._get_llm(symbol, data, ctx)

        # current state
        last_row = data["primary"].iloc[-1]
        price = float(last_row["close"])
        high = float(last_row["high"])
        low = float(last_row["low"])
        atr = float(last_row.get("atr", 0))
        rsi = float(last_row.get("rsi", 50)) if pd.notna(last_row.get("rsi")) else 50.0
        volume_ratio = float(last_row.get("volume_ratio", 1.0)) if pd.notna(last_row.get("volume_ratio")) else 1.0
        adx = float(last_row.get("ADX_14", 25.0)) if pd.notna(last_row.get("ADX_14")) else 25.0
        ob_imbalance = ctx.get("bid_ask_imbalance", 0)

        ema9 = float(last_row.get("ema_9", 0)) if pd.notna(last_row.get("ema_9")) else 0
        ema_trend = (1 if price > ema9 else -1) if ema9 > 0 else 0

        signal = self.signal_gen.generate(
            symbol=symbol,
            ml_result=ml_result,
            llm_result=llm_result,
            current_price=price,
            atr=atr,
            volume_ratio=volume_ratio,
            adx=adx,
            bid_ask_imbalance=ob_imbalance,
            ema_trend=ema_trend,
        )

        # store candle data for accurate paper trading
        signal.candle_high = high
        signal.candle_low = low
        signal.candle_atr = atr
        signal.candle_rsi = rsi

        # ── check freshness: how many past bars had same direction ──
        if signal.direction != "NEUTRAL" and len(X) > 6:
            signal.age_bars = self._check_signal_age(X, signal.direction)

        # ── detailed logging for analysis ────────────────────
        llm_reason = llm_result.get("reasoning", "")
        ml_sig = ml_result.get("signal", 0)
        ml_conf_pct = ml_result.get("confidence", 0) * 100
        ml_only_dir = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(ml_sig, "NEUTRAL")
        ml_only_would_open = ml_conf_pct >= self.signal_gen.config.PREDICTION_THRESHOLD * 100 and ml_sig != 0
        llm_dir_raw = llm_result.get("direction", "NEUTRAL")
        llm_conf_raw = llm_result.get("confidence", 0)
        llm_would_open = llm_dir_raw in ("LONG", "SHORT") and llm_conf_raw >= 7
        llm_inverted = {"LONG": "SHORT", "SHORT": "LONG"}.get(llm_dir_raw, "NEUTRAL")
        ml_inverted = {"LONG": "SHORT", "SHORT": "LONG"}.get(ml_only_dir, "NEUTRAL")

        # potential SL/TP for logging (for each scenario)
        min_sl_dist = price * config.MIN_SL_PCT / 100
        sl_dist = max(atr * config.SL_ATR_MULTIPLIER, min_sl_dist)
        tp_dist = atr * config.TP_ATR_MULTIPLIER

        def _sl_tp(d):
            if d == "LONG": return price - sl_dist, price + tp_dist
            elif d == "SHORT": return price + sl_dist, price - tp_dist
            return 0, 0

        combined_dir = signal.direction
        combined_inv = {"LONG": "SHORT", "SHORT": "LONG"}.get(combined_dir, "NEUTRAL")
        combined_would_open = signal.confidence >= self.signal_gen.config.PREDICTION_THRESHOLD and combined_dir != "NEUTRAL"

        ml_sl, ml_tp = _sl_tp(ml_only_dir)
        ml_inv_sl, ml_inv_tp = _sl_tp(ml_inverted)
        llm_sl, llm_tp = _sl_tp(llm_dir_raw)
        llm_inv_sl, llm_inv_tp = _sl_tp(llm_inverted)
        comb_sl, comb_tp = _sl_tp(combined_dir)
        comb_inv_sl, comb_inv_tp = _sl_tp(combined_inv)

        _trade_logger.info(
            "SCAN %s | price=%.6g | atr=%.4f | vol=%.2f | adx=%.1f | ob=%.2f | "
            "ml=%s(%.1f%%) | llm=%s(%d/10) | combined=%s(%.1f%%) | "
            "ml_only=%s(%s) sl=%.6g tp=%.6g | ml_inv=%s sl=%.6g tp=%.6g | "
            "llm_only=%s(%s) sl=%.6g tp=%.6g | llm_inv=%s sl=%.6g tp=%.6g | "
            "comb_inv=%s sl=%.6g tp=%.6g | "
            "filter=%s | status=%s | age=%d | regime=%s | "
            "rsi=%.1f | funding=%.6f | ls_ratio=%.2f | llm_reason=%s",
            symbol, price, atr, volume_ratio, adx, ob_imbalance,
            {1: "BUY", -1: "SELL", 0: "HOLD"}.get(ml_sig, "?"),
            ml_conf_pct,
            llm_result.get("direction", "?"),
            llm_result.get("confidence", 0),
            signal.direction,
            signal.confidence * 100,
            ml_only_dir, "OPEN" if ml_only_would_open else "SKIP", ml_sl, ml_tp,
            ml_inverted, ml_inv_sl, ml_inv_tp,
            llm_dir_raw, "OPEN" if llm_would_open else "SKIP", llm_sl, llm_tp,
            llm_inverted, llm_inv_sl, llm_inv_tp,
            combined_inv, comb_inv_sl, comb_inv_tp,
            getattr(signal, "filter_reason", "") or "PASS",
            "FRESH" if getattr(signal, "is_new", True) and signal.age_bars == 0
            else f"NEW_{signal.age_bars * 5}m" if getattr(signal, "is_new", True) and signal.age_bars <= 2
            else f"LATE_{signal.age_bars * 5}m" if getattr(signal, "is_new", True)
            else "ACTIVE",
            signal.age_bars,
            signal.market_regime,
            float(last_row.get("rsi", 0)) if pd.notna(last_row.get("rsi")) else 0,
            ctx.get("funding_rate", 0) if isinstance(ctx.get("funding_rate"), (int, float)) else 0,
            ctx.get("long_short_ratio", 0),
            llm_reason.replace("\n", " ") if llm_reason else "—",
        )

        # ── 1m entry refinement (only for strong, fresh signals) ──
        threshold = self.signal_gen.config.PREDICTION_THRESHOLD
        if (
            signal.direction != "NEUTRAL"
            and signal.confidence >= threshold
            and signal.age_bars <= 2
            and config.USE_1M_ENTRY
        ):
            refined = self.entry_refiner.refine(
                symbol=symbol,
                direction=signal.direction,
                signal_price=price,
                atr_5m=atr,
            )
            signal.entry_price = refined.entry_price
            signal.stop_loss = refined.stop_loss
            signal.take_profit = refined.take_profit
            if refined.method != "MARKET":
                signal.llm_reasoning = (
                    f"[1m {refined.method}: entry improved by "
                    f"{refined.improvement_pct:+.3f}%, "
                    f"waited {refined.wait_bars} bars] "
                    + signal.llm_reasoning
                )

        # ── track active signals ─────────────────────────────
        signal = self._track_signal(signal)

        # ── INSTANT execution: only FRESH and NEW signals (not LATE) ──
        threshold = self.signal_gen.config.PREDICTION_THRESHOLD
        if (
            signal.direction != "NEUTRAL"
            and signal.confidence >= threshold
            and getattr(signal, "is_new", False)
            and signal.age_bars <= 2          # FRESH (0) or NEW (1-2), skip LATE (3+)
            and not self.executor.has_position(symbol)
            and self.executor.check_daily_limit()
        ):
            opened = self.executor.open_trade(
                symbol=symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                leverage=signal.leverage,
                confidence=signal.confidence,
            )
            if opened:
                _trade_logger.info(
                    "OPEN %s %s | entry=%.6g | sl=%.6g | tp=%.6g | "
                    "lev=x%d | conf=%.1f%% | mode=%s | reasoning=%s",
                    signal.direction, symbol,
                    signal.entry_price, signal.stop_loss, signal.take_profit,
                    signal.leverage, signal.confidence * 100,
                    config.TRADING_MODE,
                    signal.llm_reasoning[:100],
                )

        return signal

    def _check_signal_age(self, X, direction: str, lookback: int = 6) -> int:
        """Check how many recent bars had the same ML direction."""
        count = 0
        target = 1 if direction == "LONG" else -1
        for i in range(2, min(lookback + 2, len(X))):
            row = X.iloc[[-i]]
            pred = self.ml_model.predict(row)
            if pred["signal"] == target and pred["confidence"] >= 0.50:
                count += 1
            else:
                break
        return count

    # ── signal tracking ──────────────────────────────────────

    def _track_signal(self, signal):
        """
        Track when a signal first appeared. On subsequent scans,
        keep the original entry/SL/TP from the first occurrence.
        """
        sym = signal.symbol
        threshold = self.signal_gen.config.PREDICTION_THRESHOLD
        prev = self._active_signals.get(sym)

        if signal.direction != "NEUTRAL" and signal.confidence >= threshold:
            if prev and prev["direction"] == signal.direction:
                # signal continues — stamp original levels
                signal.first_seen = prev["first_seen"]
                signal.original_entry = prev["entry"]
                signal.original_sl = prev["sl"]
                signal.original_tp = prev["tp"]
                signal.is_new = False
            else:
                # new signal — record it
                from datetime import datetime
                now = datetime.now()
                self._active_signals[sym] = {
                    "direction": signal.direction,
                    "entry": signal.entry_price,
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "first_seen": now,
                }
                signal.first_seen = now
                signal.original_entry = signal.entry_price
                signal.original_sl = signal.stop_loss
                signal.original_tp = signal.take_profit
                signal.is_new = True
        else:
            # signal dropped below threshold — clear tracking
            if sym in self._active_signals:
                del self._active_signals[sym]

        return signal

    # ── full scan ────────────────────────────────────────────

    def run_scan(self) -> list:
        signals = []
        t0 = time.time()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        from src.display import console

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            TextColumn("{task.completed}/{task.total} pairs"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Scanning …", total=len(config.TRADING_PAIRS))

            with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
                futures = {
                    pool.submit(self.scan_pair, sym): sym
                    for sym in config.TRADING_PAIRS
                }
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        sig = future.result()
                        if sig:
                            signals.append(sig)
                    except Exception as exc:
                        logger.error("Error scanning %s: %s", sym, exc)
                    progress.advance(task)
                    progress.update(task, description=f"Scanned {sym}")

        # ── monitor open trades ──────────────────────────────
        self._update_open_trades(signals)

        scan_time = time.time() - t0
        Display.show_signals(
            signals, scan_time, config.SCAN_INTERVAL,
            open_trades=self._open_trades,
        )
        return signals

    # ── trade monitoring ─────────────────────────────────────

    def _update_open_trades(self, signals: list):
        """Sync trade monitor with executor positions, update health & trailing."""

        # sync: if executor has position but monitor doesn't, add it
        for sym, pos in self.executor.get_open_positions().items():
            if sym not in self._open_trades:
                self._open_trades[sym] = self._trade_monitor.create_state(
                    sym, pos["direction"], pos["entry_price"],
                    pos["stop_loss"], pos["take_profit"],
                )

        # close all open trades during dead hours (e.g. NYSE open)
        dead_hours = config.FILTER_DEAD_HOURS
        if dead_hours:
            from datetime import datetime, timezone
            current_hour = datetime.now(timezone.utc).hour
            if current_hour in dead_hours and self._open_trades:
                for sym in list(self._open_trades.keys()):
                    sig_match = next((s for s in signals if s.symbol == sym), None)
                    price = sig_match.entry_price if sig_match else 0
                    if price > 0:
                        result = self.executor.close_trade(sym, price, "DEAD_HOUR")
                        if result:
                            _trade_logger.info(
                                "CLOSE %s (DEAD_HOUR) | pnl=%.2f%%",
                                sym, result.get("pnl_pct", 0),
                            )
                    del self._open_trades[sym]
                return

        for sig in signals:
            sym = sig.symbol

            # update existing trades with REAL candle high/low
            if sym in self._open_trades:
                state = self._open_trades[sym]

                price = sig.entry_price
                high = getattr(sig, "candle_high", price)
                low = getattr(sig, "candle_low", price)
                atr = getattr(sig, "candle_atr", 0)
                rsi = getattr(sig, "candle_rsi", 50)

                state = self._trade_monitor.update_trade(
                    state,
                    high=high,
                    low=low,
                    close=price,
                    atr=atr,
                    rsi=rsi,
                    adx=float(sig.ml_confidence * 100),  # rough proxy
                    volume_ratio=1.0,
                    ml_signal=sig.ml_signal,
                    ml_confidence=sig.ml_confidence,
                )
                self._open_trades[sym] = state

                # check exit using REAL high/low
                should_exit, reason, _ = self._trade_monitor.check_exit(
                    state, high, low, price,
                )
                if should_exit:
                    closed_direction = state.direction
                    result = self.executor.close_trade(sym, price, reason)
                    if result:
                        _trade_logger.info(
                            "CLOSE %s %s | entry=%.6g | exit=%.6g | "
                            "reason=%s | pnl=%.2f%% | bars=%d | "
                            "trail_active=%s | health=%s | %s",
                            state.direction, sym,
                            state.entry_price, price,
                            reason, result.get("pnl_pct", 0),
                            state.bars_held,
                            state.trailing_active,
                            state.health,
                            state.health_reason,
                        )
                    del self._open_trades[sym]

                    # flip: if closed due to ML reversal, open opposite immediately
                    if reason == "EARLY_EXIT" and "ML reversed" in state.health_reason:
                        new_dir = "LONG" if closed_direction == "SHORT" else "SHORT"
                        if (sig.direction == new_dir
                                and sig.confidence >= self.signal_gen.config.PREDICTION_THRESHOLD
                                and self.executor.check_daily_limit()):
                            opened = self.executor.open_trade(
                                symbol=sym,
                                direction=sig.direction,
                                entry_price=sig.entry_price,
                                stop_loss=sig.stop_loss,
                                take_profit=sig.take_profit,
                                leverage=sig.leverage,
                                confidence=sig.confidence,
                            )
                            if opened:
                                _trade_logger.info(
                                    "FLIP %s → %s %s @ %.6g (confidence=%.1f%%)",
                                    closed_direction, new_dir, sym, price, sig.confidence * 100,
                                )

    # ── main loop ────────────────────────────────────────────

    @staticmethod
    def _archive_logs():
        """Move old logs to archive folder on startup."""
        import shutil
        from datetime import datetime

        archive_dir = "logs_archive"
        os.makedirs(archive_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for fname in ("scanner.log", "trades.log", "paper_trades.json"):
            if os.path.exists(fname) and os.path.getsize(fname) > 0:
                dest = os.path.join(archive_dir, f"{stamp}_{fname}")
                shutil.copy2(fname, dest)
                # truncate the original file (keep handler valid)
                with open(fname, "w"):
                    pass

    def run(self, once: bool = False):
        Display.show_info("Starting AI Crypto Futures Scanner …")
        Display.show_info(f"Pairs: {', '.join(config.TRADING_PAIRS)}")
        Display.show_info(f"Primary TF: {config.PRIMARY_TIMEFRAME}")
        Display.show_info(
            f"ML model: {'loaded' if self.ml_model.is_trained else 'NOT trained — run train.py first'}"
        )
        Display.show_info(f"LLM: {'enabled' if config.USE_LLM else 'disabled'}")
        Display.show_info("Market context: OI + L/S + Stablecoin Dom + Liq Zones")
        Display.show_info(f"1m entry refinement: {'ON' if config.USE_1M_ENTRY else 'OFF'}")
        mode_color = "green" if config.TRADING_MODE == "paper" else "bold red"
        Display.show_info(
            f"Trading mode: [{mode_color}]{config.TRADING_MODE.upper()}[/{mode_color}]"
            f" | Balance: ${config.TRADE_BALANCE_USDT} | Max positions: {config.MAX_OPEN_TRADES}"
        )
        Display.show_info(f"Scan interval: {config.SCAN_INTERVAL}s")
        print()

        while True:
            try:
                self.run_scan()
                if once:
                    break
                self._wait_for_next_candle()
            except KeyboardInterrupt:
                Display.show_info("Scanner stopped by user.")
                break
            except Exception as exc:
                logger.error("Scan loop error: %s", exc)
                time.sleep(30)

    @staticmethod
    def _wait_for_next_candle():
        """Sleep until next 5m candle closes + 3 sec buffer."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        # seconds into current 5m candle
        secs_into = (now.minute % 5) * 60 + now.second
        # seconds until next 5m boundary
        wait = (5 * 60) - secs_into + 3  # +3 sec buffer for Binance to finalize
        if wait > 5 * 60:
            wait = 3
        mins = wait // 60
        secs = wait % 60
        Display.show_info(
            f"Next scan at candle close in {mins}m {secs}s …"
        )
        time.sleep(wait)

    # ── helpers ──────────────────────────────────────────────

    def _get_llm(
        self, symbol: str, data: dict, context: dict | None = None,
    ) -> dict:
        default = (
            self.llm.default_response()
            if self.llm
            else LLMAnalyzer.default_response()
        )

        if not self.llm or not config.USE_LLM:
            return default

        now = time.time()
        last = self._llm_timestamps.get(symbol, 0)
        if now - last < config.LLM_ANALYSIS_INTERVAL * 60:
            return self._llm_cache.get(symbol, default)

        result = self.llm.analyze_market(symbol, data, context=context)
        self._llm_cache[symbol] = result
        self._llm_timestamps[symbol] = now
        return result


# ── entry point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Crypto Futures Scanner")
    parser.add_argument(
        "--once", action="store_true", help="Run a single scan and exit",
    )
    args = parser.parse_args()

    scanner = CryptoScanner()
    scanner.run(once=args.once)


if __name__ == "__main__":
    main()
