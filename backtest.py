#!/usr/bin/env python3
"""
Realistic Backtester for the AI Crypto Futures Scanner.
───────────────────────────────────────────────────────
Walk-forward OUT-OF-SAMPLE testing: trains on first 70% of data,
tests on the last 30% that the model has NEVER seen.

Realistic simulation:
  - Entry at NEXT candle's open (not current close)
  - Binance Futures commissions (0.04% maker + 0.04% taker)
  - Slippage estimate (configurable, default 0.01%)
  - Only one trade per pair at a time
  - Cooldown between trades

Usage:
    python backtest.py                          # all pairs, realistic
    python backtest.py --pair BTC/USDT          # single pair
    python backtest.py --candles 10000          # more history
    python backtest.py --train-ratio 0.75       # 75% train / 25% test
    python backtest.py --no-commission          # disable commissions
"""

import argparse
import logging
import sys
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.ml_model import MLSignalModel, _LABEL_TO_CLASS, _CLASS_TO_LABEL
from src.trade_monitor import TradeMonitor, TradeState

from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
console = Console()


# ═════════════════════════════════════════════════════════════
#  Data structures
# ═════════════════════════════════════════════════════════════

COMMISSION_PCT = 0.04       # Binance Futures VIP0 maker/taker each
SLIPPAGE_PCT = 0.01         # estimated slippage per entry/exit


@dataclass
class Trade:
    symbol: str
    direction: str
    signal_price: float     # price when signal was generated (close)
    entry_price: float      # actual entry (next candle open + slippage)
    entry_time: datetime
    stop_loss: float
    take_profit: float
    confidence: float
    max_bars: int = 12

    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0       # raw PnL
    pnl_net_pct: float = 0.0   # after commission + slippage
    commission_pct: float = 0.0
    bars_held: int = 0


@dataclass
class BacktestResult:
    symbol: str
    trades: list = field(default_factory=list)
    train_samples: int = 0
    test_samples: int = 0
    model_accuracy: float = 0.0

    @property
    def total(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl_net_pct > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl_net_pct <= 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total * 100 if self.total else 0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_net_pct for t in self.trades)

    @property
    def total_pnl_gross(self) -> float:
        return sum(t.pnl_pct for t in self.trades)

    @property
    def total_commission(self) -> float:
        return sum(t.commission_pct for t in self.trades)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total if self.total else 0

    @property
    def avg_win(self) -> float:
        w = [t.pnl_net_pct for t in self.trades if t.pnl_net_pct > 0]
        return float(np.mean(w)) if w else 0

    @property
    def avg_loss(self) -> float:
        lo = [t.pnl_net_pct for t in self.trades if t.pnl_net_pct <= 0]
        return float(np.mean(lo)) if lo else 0

    @property
    def profit_factor(self) -> float:
        gp = sum(t.pnl_net_pct for t in self.trades if t.pnl_net_pct > 0)
        gl = abs(sum(t.pnl_net_pct for t in self.trades if t.pnl_net_pct <= 0))
        return gp / gl if gl > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0
        equity = np.cumsum([t.pnl_net_pct for t in self.trades])
        peak = np.maximum.accumulate(equity)
        return float((peak - equity).max())

    @property
    def sharpe(self) -> float:
        if len(self.trades) < 2:
            return 0
        rets = [t.pnl_net_pct for t in self.trades]
        std = np.std(rets)
        if std == 0:
            return 0
        trades_per_day = 20  # rough estimate for intraday
        return float(np.mean(rets) / std * np.sqrt(252 * trades_per_day))

    @property
    def avg_bars(self) -> float:
        return float(np.mean([t.bars_held for t in self.trades])) if self.trades else 0


# ═════════════════════════════════════════════════════════════
#  Walk-forward model trainer (lightweight, in-memory)
# ═════════════════════════════════════════════════════════════

def _train_ensemble_on_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list,
    sample_weights: Optional[np.ndarray] = None,
) -> dict:
    """Train XGBoost + LightGBM + CatBoost in memory (no saving)."""
    mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_c = X_train.loc[mask]
    y_c = y_train.loc[mask].map(_LABEL_TO_CLASS)
    sw = sample_weights[mask.values] if sample_weights is not None else None

    models = {}

    models["xgboost"] = xgb.XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    models["xgboost"].fit(X_c, y_c, sample_weight=sw, verbose=False)

    models["lightgbm"] = lgb.LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multiclass", num_class=3,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    models["lightgbm"].fit(X_c, y_c, sample_weight=sw)

    models["catboost"] = CatBoostClassifier(
        iterations=500, depth=7, learning_rate=0.03,
        l2_leaf_reg=1.0, random_seed=42, verbose=0,
        loss_function="MultiClass", classes_count=3,
    )
    models["catboost"].fit(X_c, y_c, sample_weight=sw, verbose=0)

    return models


def _predict_with_ensemble(models: dict, X_row: pd.DataFrame) -> dict:
    """Average probabilities from all models in the ensemble."""
    X_clean = X_row.replace([np.inf, -np.inf], np.nan).fillna(0)

    all_proba = []
    for name, model in models.items():
        try:
            proba = model.predict_proba(X_clean)[0]
            if len(proba) == 3:
                all_proba.append(proba)
        except Exception:
            pass

    if not all_proba:
        return {"signal": 0, "confidence": 0.0}

    avg_proba = np.mean(all_proba, axis=0)
    pred_cls = int(np.argmax(avg_proba))
    return {
        "signal": _CLASS_TO_LABEL[pred_cls],
        "confidence": float(avg_proba[pred_cls]),
    }


# ═════════════════════════════════════════════════════════════
#  Backtester
# ═════════════════════════════════════════════════════════════

class Backtester:
    def __init__(
        self,
        train_ratio: float = 0.70,
        min_confidence: float = 0.30,
        sl_atr_mult: float = 1.0,
        tp_atr_mult: float = 2.0,
        max_hold_bars: int = 12,
        cooldown_bars: int = 3,
        use_commission: bool = True,
        slippage_pct: float = SLIPPAGE_PCT,
        use_filters: bool = True,
    ):
        self.fetcher = BinanceDataFetcher(config)
        self.indicators = TechnicalIndicators()
        self.features = FeatureEngineer()

        self.train_ratio = train_ratio
        self.min_confidence = min_confidence
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.max_hold_bars = max_hold_bars
        self.cooldown_bars = cooldown_bars
        self.use_commission = use_commission
        self.slippage_pct = slippage_pct
        self.use_filters = use_filters

    def backtest_pair(
        self,
        symbol: str,
        total_candles: int = 10000,
    ) -> BacktestResult:
        console.print(f"\n  [cyan]{symbol}[/cyan]")

        # ── 1. fetch data ────────────────────────────────────
        console.print(f"    Fetching 5m …", end=" ")
        df_5m = self.fetcher.fetch_ohlcv_extended(
            symbol, config.PRIMARY_TIMEFRAME, total_candles=total_candles,
        )
        if df_5m.empty or len(df_5m) < 500:
            console.print(f"[red]Not enough data[/red]")
            return BacktestResult(symbol=symbol)
        console.print(f"{len(df_5m)} candles")

        console.print(f"    Fetching 15m + 1h …", end=" ")
        df_15m = self.fetcher.fetch_ohlcv_extended(
            symbol, config.SECONDARY_TIMEFRAME,
            total_candles=max(200, total_candles // 3),
        )
        df_1h = self.fetcher.fetch_ohlcv_extended(
            symbol, config.TREND_TIMEFRAME,
            total_candles=max(200, total_candles // 12),
        )
        console.print("done")

        # ── 2. indicators ────────────────────────────────────
        df_5m = self.indicators.calculate_all(df_5m)
        if not df_15m.empty:
            df_15m = self.indicators.calculate_all(df_15m)
        if not df_1h.empty:
            df_1h = self.indicators.calculate_all(df_1h)

        # ── 3. features ──────────────────────────────────────
        X_full, feat_names = self.features.get_feature_matrix_mtf(
            df_5m, df_15m, df_1h,
        )

        # ── 4. labels ────────────────────────────────────────
        y_full = self.features.create_labels(
            df_5m,
            tp_multiplier=config.LABEL_TP_MULTIPLIER,
            sl_multiplier=config.LABEL_SL_MULTIPLIER,
            max_bars=config.LABEL_MAX_BARS,
        )

        # align
        common = X_full.index.intersection(y_full.index)
        X_full = X_full.loc[common]
        y_full = y_full.loc[common]

        # ── 5. train / test split (temporal) ─────────────────
        split_idx = int(len(X_full) * self.train_ratio)
        X_train = X_full.iloc[:split_idx]
        y_train = y_full.iloc[:split_idx]
        X_test = X_full.iloc[split_idx:]

        # test portion of price data
        test_start = X_test.index[0]
        test_end = X_test.index[-1]
        df_test = df_5m.loc[test_start:test_end]

        console.print(
            f"    Train: {len(X_train)} candles | "
            f"Test: {len(X_test)} candles (OUT-OF-SAMPLE)"
        )

        if len(X_train) < 200 or len(X_test) < 100:
            console.print(f"    [red]Not enough data for split[/red]")
            return BacktestResult(symbol=symbol)

        # ── 6. train model on TRAIN data only ────────────────
        console.print(f"    Training ensemble (XGB+LGB+CB) …", end=" ")
        sw = self.features.compute_sample_weights(y_train)
        model = _train_ensemble_on_data(X_train, y_train, feat_names, sw)
        console.print("done")

        # ── 7. walk forward on TEST data ─────────────────────
        console.print(f"    Simulating trades on test data …")
        trades = self._simulate(model, df_test, X_test, symbol)

        console.print(f"    Result: {len(trades)} trades")

        return BacktestResult(
            symbol=symbol,
            trades=trades,
            train_samples=len(X_train),
            test_samples=len(X_test),
        )

    def _simulate(
        self,
        model: dict,
        df: pd.DataFrame,
        X: pd.DataFrame,
        symbol: str,
    ) -> list[Trade]:
        """Walk through test candles and simulate trading."""
        trades: list[Trade] = []
        in_trade = False
        current_trade: Optional[Trade] = None
        trade_state: Optional[TradeState] = None
        monitor = TradeMonitor(config)
        bars_since_exit = self.cooldown_bars
        pending_signal: Optional[dict] = None

        for i in range(len(df)):
            ts = df.index[i]
            row = df.iloc[i]
            price_open = float(row["open"])
            price_close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            atr = float(row["atr"]) if pd.notna(row.get("atr")) else 0

            # ── execute pending signal at this candle's OPEN ──
            if pending_signal is not None and not in_trade:
                sig = pending_signal
                pending_signal = None

                if sig["direction"] == "LONG":
                    entry = price_open * (1 + self.slippage_pct / 100)
                else:
                    entry = price_open * (1 - self.slippage_pct / 100)

                sl_dist = sig["atr"] * self.sl_atr_mult
                tp_dist = sig["atr"] * self.tp_atr_mult

                if sig["direction"] == "LONG":
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                else:
                    sl = entry + sl_dist
                    tp = entry - tp_dist

                current_trade = Trade(
                    symbol=symbol,
                    direction=sig["direction"],
                    signal_price=sig["signal_price"],
                    entry_price=entry,
                    entry_time=ts,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=sig["confidence"],
                    max_bars=self.max_hold_bars,
                )
                trade_state = monitor.create_state(
                    symbol, sig["direction"], entry, sl, tp,
                )
                in_trade = True

            # ── check open trade (with trailing + health) ────
            if in_trade and current_trade is not None and trade_state is not None:
                # get ML signal for health check
                rsi = float(row["rsi"]) if pd.notna(row.get("rsi")) else 50
                adx_val = float(row["ADX_14"]) if pd.notna(row.get("ADX_14")) else 25
                vol_r = float(row["volume_ratio"]) if pd.notna(row.get("volume_ratio")) else 1.0

                ml_sig = 0
                ml_conf = 0.0
                if ts in X.index:
                    loc = X.index.get_loc(ts)
                    pred = _predict_with_ensemble(model, X.iloc[[loc]])
                    ml_sig = pred["signal"]
                    ml_conf = pred["confidence"]

                trade_state = monitor.update_trade(
                    trade_state, high, low, price_close, atr,
                    rsi=rsi, adx=adx_val, volume_ratio=vol_r,
                    ml_signal=ml_sig, ml_confidence=ml_conf,
                )

                # update trade's SL to trailing SL
                current_trade.stop_loss = trade_state.current_sl

                should_exit, reason, exit_price = monitor.check_exit(
                    trade_state, high, low, price_close,
                )

                if should_exit and reason:
                    # apply slippage on exit
                    if current_trade.direction == "LONG":
                        exit_price *= (1 - self.slippage_pct / 100)
                    else:
                        exit_price *= (1 + self.slippage_pct / 100)

                    # PnL
                    if current_trade.direction == "LONG":
                        raw_pnl = (exit_price - current_trade.entry_price) / current_trade.entry_price * 100
                    else:
                        raw_pnl = (current_trade.entry_price - exit_price) / current_trade.entry_price * 100

                    # commission: entry + exit
                    comm = 2 * COMMISSION_PCT if self.use_commission else 0

                    current_trade.exit_price = exit_price
                    current_trade.exit_time = ts
                    current_trade.exit_reason = reason
                    current_trade.pnl_pct = round(raw_pnl, 4)
                    current_trade.pnl_net_pct = round(raw_pnl - comm, 4)
                    current_trade.commission_pct = round(comm, 4)

                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    bars_since_exit = 0
                    continue

            # ── generate signal (for NEXT candle entry) ──────
            if not in_trade and pending_signal is None:
                bars_since_exit += 1
                if bars_since_exit < self.cooldown_bars:
                    continue
                if atr == 0 or ts not in X.index:
                    continue

                loc = X.index.get_loc(ts)
                X_row = X.iloc[[loc]]

                pred = _predict_with_ensemble(model, X_row)
                signal = pred["signal"]
                confidence = pred["confidence"]

                if signal == 0 or confidence < self.min_confidence:
                    continue

                direction = "LONG" if signal == 1 else "SHORT"

                # ── filters (volume + ADX) ───────────────────
                vol_ratio = float(row["volume_ratio"]) if pd.notna(row.get("volume_ratio")) else 1.0
                adx_val = float(row["ADX_14"]) if pd.notna(row.get("ADX_14")) else 25.0

                if self.use_filters:
                    from src.config import config as _cfg
                    if _cfg.FILTER_MIN_VOLUME_RATIO > 0 and vol_ratio < _cfg.FILTER_MIN_VOLUME_RATIO:
                        continue
                    if _cfg.FILTER_MIN_ADX > 0 and adx_val < _cfg.FILTER_MIN_ADX:
                        continue

                # queue signal — will be executed at NEXT candle's open
                pending_signal = {
                    "direction": direction,
                    "confidence": confidence,
                    "signal_price": price_close,
                    "atr": atr,
                }

        # close any remaining trade
        if in_trade and current_trade is not None:
            last_close = float(df.iloc[-1]["close"])
            if current_trade.direction == "LONG":
                raw = (last_close - current_trade.entry_price) / current_trade.entry_price * 100
            else:
                raw = (current_trade.entry_price - last_close) / current_trade.entry_price * 100
            comm = 2 * COMMISSION_PCT if self.use_commission else 0
            current_trade.exit_price = last_close
            current_trade.exit_time = df.index[-1]
            current_trade.exit_reason = "END"
            current_trade.pnl_pct = round(raw, 4)
            current_trade.pnl_net_pct = round(raw - comm, 4)
            current_trade.commission_pct = round(comm, 4)
            trades.append(current_trade)

        return trades

    # ── run all pairs ────────────────────────────────────────

    def run(
        self,
        pairs: list[str] | None = None,
        total_candles: int = 10000,
    ) -> list[BacktestResult]:
        pairs = pairs or config.TRADING_PAIRS
        results = []

        console.print(Panel(
            f"[bold cyan]Realistic Out-of-Sample Backtester[/bold cyan]\n\n"
            f"Pairs: {len(pairs)} | Candles: {total_candles}\n"
            f"Train/Test split: {self.train_ratio:.0%} / {1 - self.train_ratio:.0%} "
            f"(model trained on first {self.train_ratio:.0%}, tested on unseen {1 - self.train_ratio:.0%})\n"
            f"SL: {self.sl_atr_mult}x ATR | TP: {self.tp_atr_mult}x ATR | "
            f"Max hold: {self.max_hold_bars} bars ({self.max_hold_bars * 5} min)\n"
            f"Min confidence: {self.min_confidence:.0%} | "
            f"Commission: {'0.04% x2' if self.use_commission else 'OFF'} | "
            f"Slippage: {self.slippage_pct:.2f}%\n"
            f"Entry: NEXT candle open (realistic)\n"
            f"SL+TP same candle: SL assumed first (conservative)\n"
            f"Filters (volume/ADX): {'ON' if self.use_filters else 'OFF'}",
            box=box.DOUBLE,
        ))

        t0 = _time.time()

        if len(pairs) > 1:
            # parallel execution
            from concurrent.futures import ThreadPoolExecutor, as_completed
            console.print(f"  [dim]Running {len(pairs)} pairs in parallel …[/dim]\n")

            with ThreadPoolExecutor(max_workers=min(4, len(pairs))) as pool:
                futures = {
                    pool.submit(self.backtest_pair, sym, total_candles): sym
                    for sym in pairs
                }
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        console.print(f"  [red]{sym} error: {exc}[/red]")
        else:
            for symbol in pairs:
                try:
                    result = self.backtest_pair(symbol, total_candles)
                    results.append(result)
                except Exception as exc:
                    console.print(f"  [red]{symbol} error: {exc}[/red]")

        elapsed = _time.time() - t0
        console.print(f"\n  Total backtest time: {elapsed:.1f}s")
        self._print_report(results)
        return results

    # ── reporting ────────────────────────────────────────────

    def _print_report(self, results: list[BacktestResult]):
        console.print()

        table = Table(
            title="Out-of-Sample Backtest Results",
            box=box.ROUNDED,
            header_style="bold magenta",
        )
        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Trades", justify="center", width=8)
        table.add_column("Win Rate", justify="center", width=10)
        table.add_column("Net PnL%", justify="right", width=11)
        table.add_column("Gross PnL%", justify="right", width=11)
        table.add_column("Commiss.%", justify="right", width=10)
        table.add_column("Profit F.", justify="center", width=10)
        table.add_column("Max DD%", justify="right", width=9)
        table.add_column("Avg Bars", justify="center", width=9)
        table.add_column("Test Size", justify="center", width=10)

        for r in results:
            if r.total == 0:
                table.add_row(
                    r.symbol, "0", "—", "—", "—", "—", "—", "—", "—",
                    str(r.test_samples),
                )
                continue

            wr = r.win_rate
            wr_s = "green" if wr >= 50 else "yellow" if wr >= 40 else "red"
            pnl_s = "green" if r.total_pnl > 0 else "red"
            gpnl_s = "green" if r.total_pnl_gross > 0 else "red"

            table.add_row(
                r.symbol,
                str(r.total),
                f"[{wr_s}]{wr:.1f}%[/{wr_s}]",
                f"[{pnl_s}]{r.total_pnl:+.2f}%[/{pnl_s}]",
                f"[{gpnl_s}]{r.total_pnl_gross:+.2f}%[/{gpnl_s}]",
                f"[red]-{r.total_commission:.2f}%[/red]",
                f"{r.profit_factor:.2f}",
                f"[red]{r.max_drawdown:.2f}%[/red]",
                f"{r.avg_bars:.1f}",
                str(r.test_samples),
            )

        console.print(table)

        # ── aggregate ────────────────────────────────────────
        all_trades = [t for r in results for t in r.trades]
        if not all_trades:
            console.print("[dim]No trades generated.[/dim]")
            return

        agg = BacktestResult(symbol="ALL", trades=all_trades)

        tp_n = sum(1 for t in all_trades if t.exit_reason == "TP")
        sl_n = sum(1 for t in all_trades if t.exit_reason == "SL")
        trail_n = sum(1 for t in all_trades if t.exit_reason == "TRAIL_SL")
        early_n = sum(1 for t in all_trades if t.exit_reason == "EARLY_EXIT")
        to_n = sum(1 for t in all_trades if t.exit_reason == "TIMEOUT")

        longs = [t for t in all_trades if t.direction == "LONG"]
        shorts = [t for t in all_trades if t.direction == "SHORT"]
        l_wr = sum(1 for t in longs if t.pnl_net_pct > 0) / len(longs) * 100 if longs else 0
        s_wr = sum(1 for t in shorts if t.pnl_net_pct > 0) / len(shorts) * 100 if shorts else 0
        l_pnl = sum(t.pnl_net_pct for t in longs)
        s_pnl = sum(t.pnl_net_pct for t in shorts)

        c = "green" if agg.total_pnl > 0 else "red"

        console.print(Panel(
            f"[bold]Trades:[/bold] {agg.total}  "
            f"(LONG: {len(longs)} | SHORT: {len(shorts)})\n"
            f"[bold]Win Rate:[/bold] {agg.win_rate:.1f}%  "
            f"(LONG: {l_wr:.1f}% | SHORT: {s_wr:.1f}%)\n"
            f"[bold]Net PnL:[/bold] [{c}]{agg.total_pnl:+.2f}%[/{c}]  "
            f"(LONG: {l_pnl:+.2f}% | SHORT: {s_pnl:+.2f}%)\n"
            f"[bold]Gross PnL:[/bold] {agg.total_pnl_gross:+.2f}%  "
            f"[bold]Commissions:[/bold] [red]-{agg.total_commission:.2f}%[/red]\n"
            f"[bold]Profit Factor:[/bold] {agg.profit_factor:.2f}  "
            f"[bold]Sharpe:[/bold] {agg.sharpe:.2f}  "
            f"[bold]Max DD:[/bold] [red]{agg.max_drawdown:.2f}%[/red]\n"
            f"[bold]Avg Win:[/bold] [green]{agg.avg_win:+.3f}%[/green]  "
            f"[bold]Avg Loss:[/bold] [red]{agg.avg_loss:+.3f}%[/red]  "
            f"[bold]Avg Bars:[/bold] {agg.avg_bars:.1f} (~{agg.avg_bars * 5:.0f} min)\n"
            f"\n[bold]Exit Reasons:[/bold] "
            f"TP: {tp_n} | SL: {sl_n} | "
            f"Trail SL: {trail_n} | Early: {early_n} | "
            f"Timeout: {to_n}",
            title="Aggregate Performance (OUT-OF-SAMPLE)",
            border_style="cyan",
        ))

        equity = np.cumsum([t.pnl_net_pct for t in all_trades])
        self._print_equity(equity)

    @staticmethod
    def _print_equity(equity: np.ndarray, width: int = 60, height: int = 12):
        if len(equity) < 2:
            return
        n = len(equity)
        idx = np.linspace(0, n - 1, min(width, n), dtype=int)
        s = equity[idx]
        mn, mx = s.min(), s.max()
        if mx == mn:
            return
        lines = []
        for row in range(height, -1, -1):
            thr = mn + (mx - mn) * row / height
            bar = "".join("█" if v >= thr else " " for v in s)
            lines.append(f"  {thr:+7.2f}% │{bar}│")
        x_line = " " * 10 + "└" + "─" * len(s) + "┘"
        x_lbl = " " * 10 + f"1{' ' * (len(s) - 2)}{n}"
        console.print(Panel(
            "\n".join(lines) + "\n" + x_line + "\n" + x_lbl,
            title="Equity Curve — Net PnL % (after commissions)",
            border_style="dim",
        ))


# ═════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Realistic Out-of-Sample Backtest")
    p.add_argument("--pair", type=str, help="Single pair (e.g. BTC/USDT)")
    p.add_argument("--candles", type=int, default=10000, help="Total candles (default 10000)")
    p.add_argument("--train-ratio", type=float, default=0.70, help="Train portion (default 0.70)")
    p.add_argument("--confidence", type=float, default=config.PREDICTION_THRESHOLD, help=f"Min confidence (default {config.PREDICTION_THRESHOLD})")
    p.add_argument("--sl", type=float, default=config.SL_ATR_MULTIPLIER, help=f"SL in ATR (default {config.SL_ATR_MULTIPLIER})")
    p.add_argument("--tp", type=float, default=config.TP_ATR_MULTIPLIER, help=f"TP in ATR (default {config.TP_ATR_MULTIPLIER})")
    p.add_argument("--max-bars", type=int, default=config.MAX_HOLD_BARS, help=f"Max hold bars (default {config.MAX_HOLD_BARS})")
    p.add_argument("--no-commission", action="store_true", help="Disable commissions")
    p.add_argument("--no-filters", action="store_true", help="Disable volume/ADX filters")
    p.add_argument("--slippage", type=float, default=SLIPPAGE_PCT, help="Slippage %% (default 0.01)")
    args = p.parse_args()

    bt = Backtester(
        train_ratio=args.train_ratio,
        min_confidence=args.confidence,
        sl_atr_mult=args.sl,
        tp_atr_mult=args.tp,
        max_hold_bars=args.max_bars,
        use_commission=not args.no_commission,
        slippage_pct=args.slippage,
        use_filters=not args.no_filters,
    )
    pairs = [args.pair] if args.pair else None
    bt.run(pairs=pairs, total_candles=args.candles)


if __name__ == "__main__":
    main()
