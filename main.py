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
from src.risk_manager import RiskManager
from src.signal_generator import SignalGenerator

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


class CryptoScanner:
    """Orchestrates the full scan cycle for all configured pairs."""

    def __init__(self):
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
        self._active_signals: dict = {}  # symbol → {direction, entry, sl, tp, first_seen}

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
        atr = float(last_row.get("atr", 0))
        volume_ratio = float(last_row.get("volume_ratio", 1.0)) if pd.notna(last_row.get("volume_ratio")) else 1.0
        adx = float(last_row.get("ADX_14", 25.0)) if pd.notna(last_row.get("ADX_14")) else 25.0
        ob_imbalance = ctx.get("bid_ask_imbalance", 0)

        signal = self.signal_gen.generate(
            symbol=symbol,
            ml_result=ml_result,
            llm_result=llm_result,
            current_price=price,
            atr=atr,
            volume_ratio=volume_ratio,
            adx=adx,
            bid_ask_imbalance=ob_imbalance,
        )

        # ── check freshness: how many past bars had same direction ──
        if signal.direction != "NEUTRAL" and len(X) > 6:
            signal.age_bars = self._check_signal_age(X, signal.direction)

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

        for symbol in config.TRADING_PAIRS:
            try:
                sig = self.scan_pair(symbol)
                if sig:
                    signals.append(sig)
            except Exception as exc:
                logger.error("Error scanning %s: %s", symbol, exc)

        scan_time = time.time() - t0
        Display.show_signals(signals, scan_time, config.SCAN_INTERVAL)
        return signals

    # ── main loop ────────────────────────────────────────────

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
        Display.show_info(f"Scan interval: {config.SCAN_INTERVAL}s")
        print()

        while True:
            try:
                self.run_scan()
                if once:
                    break
                time.sleep(config.SCAN_INTERVAL)
            except KeyboardInterrupt:
                Display.show_info("Scanner stopped by user.")
                break
            except Exception as exc:
                logger.error("Scan loop error: %s", exc)
                time.sleep(30)

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
