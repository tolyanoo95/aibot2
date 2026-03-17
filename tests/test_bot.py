"""
Unit tests for Volume Bars Trading Bot.
Covers: volume bars, Move SL, guards, positions, fees, thread safety.

Run: NUMBA_DISABLE_JIT=1 python -m pytest tests/test_bot.py -v
"""
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_volbars import (
    Position, VolumeBarsBot, MOVE_SL_STEPS, MOVE_SL_TRAIL,
    SL_MULT, TP_MULT, ADX_MIN, LONG_MOM, SHORT_MOM, ATR_EXP_MAX,
    MAKER_FEE, TAKER_FEE, COOLDOWN_BARS,
)


# ─── Helpers ────────────────────────────────────────────────────────

def make_bot():
    """Create bot with mocked fetcher (no API calls)."""
    with patch("main_volbars.BinanceDataFetcher"):
        bot = VolumeBarsBot(paper=True)
    bot._trades_log = "/dev/null"
    bot._paper_trades_file = "/tmp/test_paper_trades.json"
    return bot


def make_position(symbol="BTC/USDT", direction="LONG", entry=74000.0, atr=350.0,
                  sl_step=0, hard_sl=None, total_size=1):
    if hard_sl is None:
        hard_sl = entry - SL_MULT * atr if direction == "LONG" else entry + SL_MULT * atr
    tp = entry + TP_MULT * atr if direction == "LONG" else entry - TP_MULT * atr
    return Position(
        symbol=symbol, direction=direction,
        entries=[(entry, time.time())], avg_price=entry,
        total_size=total_size, hard_sl=hard_sl, tp=tp,
        entry_time=time.time(), entry_atr=atr,
        max_price=entry, min_price=entry,
        sl_step=sl_step, best_price=entry,
    )


def make_vol_bars(n=100, base_price=74000.0, atr=350.0):
    """Create fake volume bars DataFrame with indicators."""
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    np.random.seed(42)
    closes = base_price + np.cumsum(np.random.randn(n) * atr * 0.1)
    df = pd.DataFrame({
        "open": closes - np.random.rand(n) * atr * 0.05,
        "high": closes + np.abs(np.random.randn(n)) * atr * 0.2,
        "low": closes - np.abs(np.random.randn(n)) * atr * 0.2,
        "close": closes,
        "volume": np.random.randint(1000, 5000, n).astype(float),
        "atr": np.full(n, atr),
        "roc_12": np.random.randn(n) * 0.5,
        "ADX_14": np.random.randint(15, 50, n).astype(float),
        "rsi": 50 + np.random.randn(n) * 10,
        "htf_supertrend": np.ones(n),
    }, index=dates)
    return df


# ═══════════════════════════════════════════════════════════════════
# 1. VOLUME BAR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

class TestVolumeBar:
    def test_bar_completes_at_threshold(self):
        """Volume bar completes when cum_vol >= threshold."""
        bot = make_bot()
        bot.vol_bars["BTC/USDT"] = make_vol_bars()
        bot.vol_thresholds["BTC/USDT"] = 1000
        bot.vol_buffers["BTC/USDT"] = {
            "cum_vol": 0, "bar_open": None, "bar_high": None,
            "bar_low": None, "bar_start": None,
        }
        bot.time_data["BTC/USDT"] = make_vol_bars()
        bot.htf_vol_bars["BTC/USDT"] = make_vol_bars(20)

        initial_bars = len(bot.vol_bars["BTC/USDT"])
        bot._complete_volume_bar(
            "BTC/USDT", 74000, 74200, 73800, 74100, 1500,
            pd.Timestamp("2026-03-17 12:00:00"),
        )
        assert len(bot.vol_bars["BTC/USDT"]) == initial_bars + 1

    def test_monotonic_timestamps_no_crash(self):
        """Non-monotonic timestamps should not crash (aggTrade scenario)."""
        bot = make_bot()
        bot.vol_bars["BTC/USDT"] = make_vol_bars()
        bot.vol_thresholds["BTC/USDT"] = 1000
        bot.vol_buffers["BTC/USDT"] = {
            "cum_vol": 0, "bar_open": None, "bar_high": None,
            "bar_low": None, "bar_start": None,
        }
        bot.time_data["BTC/USDT"] = make_vol_bars()
        bot.htf_vol_bars["BTC/USDT"] = make_vol_bars(20)

        # Two bars with timestamps that could be out of order
        bot._complete_volume_bar(
            "BTC/USDT", 74000, 74200, 73800, 74100, 1500,
            pd.Timestamp("2026-03-17 12:00:01"),
        )
        bot._complete_volume_bar(
            "BTC/USDT", 74100, 74300, 73900, 74200, 1500,
            pd.Timestamp("2026-03-17 12:00:00"),  # earlier timestamp!
        )
        # Should not crash
        assert len(bot.vol_bars["BTC/USDT"]) >= 100

    def test_bar_ohlc_correct(self):
        """OHLC of completed bar should match input data."""
        bot = make_bot()
        bot.vol_bars["BTC/USDT"] = make_vol_bars()
        bot.vol_thresholds["BTC/USDT"] = 1000
        bot.vol_buffers["BTC/USDT"] = {
            "cum_vol": 0, "bar_open": None, "bar_high": None,
            "bar_low": None, "bar_start": None,
        }
        bot.time_data["BTC/USDT"] = make_vol_bars()

        bot._complete_volume_bar(
            "BTC/USDT", 74000, 74500, 73500, 74200, 2000,
            pd.Timestamp("2026-03-17 12:00:00"),
        )
        last = bot.vol_bars["BTC/USDT"].iloc[-1]
        assert last["open"] == 74000
        assert last["high"] == 74500
        assert last["low"] == 73500
        assert last["close"] == 74200
        assert last["volume"] == 2000


# ═══════════════════════════════════════════════════════════════════
# 2. MOVE SL LOGIC
# ═══════════════════════════════════════════════════════════════════

class TestMoveSL:
    def test_step1_scalp(self):
        """Step1: price +0.25 ATR → SL moves to +0.30 ATR."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350)
        bot.positions.append(pos)
        original_sl = pos.hard_sl

        # Price reaches +0.25 ATR = 74087.5
        bot._process_price("BTC/USDT", 74088.0)

        assert pos.sl_step == 1
        assert pos.hard_sl > original_sl
        expected_sl = 74000 + 0.30 * 350  # 74105
        assert abs(pos.hard_sl - expected_sl) < 1.0

    def test_max_min_protection_long(self):
        """Later steps should never downgrade SL (LONG)."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350)
        bot.positions.append(pos)

        # Step1: SL → +0.30 ATR = 74105
        bot._process_price("BTC/USDT", 74088.0)
        sl_after_step1 = pos.hard_sl

        # Step2: move_to=0.25 ATR = 74087.5 (lower than step1 → should keep step1)
        bot._process_price("BTC/USDT", 74175.0)
        assert pos.hard_sl >= sl_after_step1

    def test_max_min_protection_short(self):
        """Later steps should never upgrade SL for SHORT (SL should decrease)."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350, direction="SHORT")
        bot.positions.append(pos)

        # Step0: SL → entry - 0.05*ATR = 73982.5
        bot._process_price("BTC/USDT", 74000 - 0.05 * 350 - 1)
        sl_after_step0 = pos.hard_sl

        # Step1: SL → entry - 0.15*ATR = 73947.5 (lower = better for SHORT)
        bot._process_price("BTC/USDT", 74000 - 0.25 * 350 - 1)
        assert pos.hard_sl <= sl_after_step0

    def test_activation_by_current_price_not_max(self):
        """Move SL activates by current price, not max_price."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350)
        bot.positions.append(pos)

        # Price reaches +0.25 ATR → step1 triggers
        bot._process_price("BTC/USDT", 74088.0)
        assert pos.sl_step == 1

        # max_price is now 74088, but current price is back at entry
        # Step2 needs +0.50 ATR = 74175 — should NOT trigger at current price
        bot._process_price("BTC/USDT", 74000.0)
        assert pos.sl_step == 1  # still step 1, not triggered by max_price

    def test_trail_after_all_steps(self):
        """After all steps, trail SL follows best_price - 1.0 ATR."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350, sl_step=len(MOVE_SL_STEPS))
        pos.sl_moved = True
        pos.hard_sl = 74000 + 1.0 * 350
        pos.best_price = 74000 + 3.0 * 350
        bot.positions.append(pos)

        # Price goes higher
        new_high = 74000 + 3.5 * 350
        bot._process_price("BTC/USDT", new_high)

        expected_trail_sl = new_high - MOVE_SL_TRAIL * 350
        assert abs(pos.hard_sl - expected_trail_sl) < 1.0


# ═══════════════════════════════════════════════════════════════════
# 3. GUARDS
# ═══════════════════════════════════════════════════════════════════

class TestGuards:
    def setup_method(self):
        self.bot = make_bot()
        self.vdf = make_vol_bars()

    def _set_signal_conditions(self, roc=1.0, adx=30, rsi_slope=5, htf=1, atr_exp=1.0):
        j = len(self.vdf) - 1
        self.vdf.loc[self.vdf.index[j], "roc_12"] = roc
        self.vdf.loc[self.vdf.index[j], "ADX_14"] = adx
        rsi_vals = self.vdf["rsi"].values.copy()
        rsi_vals[j] = 55
        rsi_vals[j-6] = 55 - rsi_slope
        self.vdf["rsi"] = rsi_vals
        self.vdf.loc[self.vdf.index[j], "htf_supertrend"] = htf
        atr_arr = self.vdf["atr"].values.copy()
        atr_ma = np.full(len(atr_arr), atr_arr[0])
        atr_arr[j] = atr_arr[0] * atr_exp
        self.vdf["atr"] = atr_arr
        self.bot.vol_bars["BTC/USDT"] = self.vdf

    def test_long_signal_passes(self):
        """Valid LONG signal passes all guards."""
        self._set_signal_conditions(roc=0.5, adx=25, rsi_slope=5, htf=1, atr_exp=1.0)
        sig = self.bot.check_signal("BTC/USDT")
        assert sig is not None
        assert sig["direction"] == "LONG"

    def test_short_signal_passes(self):
        """Valid SHORT signal passes all guards."""
        self._set_signal_conditions(roc=-0.5, adx=25, rsi_slope=-5, htf=-1, atr_exp=1.0)
        sig = self.bot.check_signal("BTC/USDT")
        assert sig is not None
        assert sig["direction"] == "SHORT"

    def test_roc_too_weak(self):
        """Weak ROC → no signal."""
        self._set_signal_conditions(roc=0.15)
        sig = self.bot.check_signal("BTC/USDT")
        assert sig is None

    def test_htf_blocks_long_in_downtrend(self):
        """HTF Supertrend downtrend blocks LONG."""
        self._set_signal_conditions(roc=0.5, htf=-1)
        sig = self.bot.check_signal("BTC/USDT")
        assert sig is None

    def test_htf_blocks_short_in_uptrend(self):
        """HTF Supertrend uptrend blocks SHORT."""
        self._set_signal_conditions(roc=-0.5, htf=1, rsi_slope=-5)
        sig = self.bot.check_signal("BTC/USDT")
        assert sig is None

    def test_adx_too_low(self):
        """ADX < 20 → blocked."""
        self._set_signal_conditions(roc=0.5, adx=15)
        sig = self.bot.check_signal("BTC/USDT")
        assert sig is None

    def test_rsi_slope_blocks_long(self):
        """RSI falling blocks LONG."""
        self._set_signal_conditions(roc=0.5, rsi_slope=-5)
        sig = self.bot.check_signal("BTC/USDT")
        assert sig is None

    def test_atr_expansion_blocks(self):
        """ATR expansion > 1.5 → blocked."""
        self._set_signal_conditions(roc=0.5, atr_exp=1.8)
        sig = self.bot.check_signal("BTC/USDT")
        assert sig is None


# ═══════════════════════════════════════════════════════════════════
# 4. POSITION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

class TestPositions:
    def test_hard_sl_closes_long(self):
        """LONG position closes when price hits hard SL."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350)
        bot.positions.append(pos)

        sl_price = 74000 - SL_MULT * 350 - 1  # below SL
        bot._process_price("BTC/USDT", sl_price)

        assert len(bot.positions) == 0  # closed

    def test_tp_closes_long(self):
        """LONG position closes when price hits TP."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350)
        bot.positions.append(pos)

        tp_price = 74000 + TP_MULT * 350 + 1  # above TP
        bot._process_price("BTC/USDT", tp_price)

        assert len(bot.positions) == 0

    def test_hard_sl_closes_short(self):
        """SHORT position closes when price hits hard SL."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350, direction="SHORT")
        bot.positions.append(pos)

        sl_price = 74000 + SL_MULT * 350 + 1
        bot._process_price("BTC/USDT", sl_price)

        assert len(bot.positions) == 0

    def test_tp_closes_short(self):
        """SHORT position closes when price hits TP."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350, direction="SHORT")
        bot.positions.append(pos)

        tp_price = 74000 - TP_MULT * 350 - 1
        bot._process_price("BTC/USDT", tp_price)

        assert len(bot.positions) == 0

    def test_position_survives_normal_price(self):
        """Position stays open when price is between SL and TP."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350)
        bot.positions.append(pos)

        # Price below step1 activation (+0.25 ATR = 74087.5)
        bot._process_price("BTC/USDT", 74050.0)

        assert len(bot.positions) == 1

    def test_cooldown_after_close(self):
        """Cooldown is set after position close."""
        bot = make_bot()
        bot.vol_bar_counts["BTC/USDT"] = 10
        pos = make_position(entry=74000, atr=350)
        bot.positions.append(pos)

        bot._process_price("BTC/USDT", 74000 - SL_MULT * 350 - 1)

        assert bot.cooldowns.get("BTC/USDT", 0) == 10 + COOLDOWN_BARS

    def test_vol_drop_timeout(self):
        """Vol drop and timeout exits work."""
        bot = make_bot()
        vdf = make_vol_bars()
        # Make last 3 bars low volume
        vdf.loc[vdf.index[-3:], "volume"] = 1.0
        bot.vol_bars["BTC/USDT"] = vdf
        bot.vol_bar_counts["BTC/USDT"] = 10

        pos = make_position(entry=74000, atr=350)
        pos.bars_held = 5  # >= 3 for vol_drop
        bot.positions.append(pos)

        bot._check_vol_drop_timeout("BTC/USDT", 74100.0)

        assert len(bot.positions) == 0  # closed by vol_drop

    def test_timeout_at_24_bars(self):
        """Position times out at 24 bars."""
        bot = make_bot()
        bot.vol_bars["BTC/USDT"] = make_vol_bars()
        bot.vol_bar_counts["BTC/USDT"] = 10

        pos = make_position(entry=74000, atr=350)
        pos.bars_held = 24
        bot.positions.append(pos)

        bot._check_vol_drop_timeout("BTC/USDT", 74100.0)

        assert len(bot.positions) == 0


# ═══════════════════════════════════════════════════════════════════
# 5. FEE CALCULATION
# ═══════════════════════════════════════════════════════════════════

class TestFees:
    def test_fee_dca1(self):
        """Fee for DCA=1: maker + taker = 0.07%."""
        fee_pct = (MAKER_FEE + TAKER_FEE) * 100 * 1
        assert abs(fee_pct - 0.07) < 0.001

    def test_fee_dca2(self):
        """Fee for DCA=2: 0.14%."""
        fee_pct = (MAKER_FEE + TAKER_FEE) * 100 * 2
        assert abs(fee_pct - 0.14) < 0.001

    def test_fee_dca3(self):
        """Fee for DCA=3: 0.21%."""
        fee_pct = (MAKER_FEE + TAKER_FEE) * 100 * 3
        assert abs(fee_pct - 0.21) < 0.001

    def test_net_pnl_positive(self):
        """NET PnL = gross - fee. Positive trade stays positive after fee."""
        gross_pnl = 0.30  # %
        fee_pct = (MAKER_FEE + TAKER_FEE) * 100  # 0.07%
        net = gross_pnl - fee_pct
        assert net > 0
        assert abs(net - 0.23) < 0.01

    def test_net_pnl_micro_trade_negative(self):
        """Very small gross PnL becomes negative after fee."""
        gross_pnl = 0.05  # %
        fee_pct = (MAKER_FEE + TAKER_FEE) * 100  # 0.07%
        net = gross_pnl - fee_pct
        assert net < 0


# ═══════════════════════════════════════════════════════════════════
# 6. THREAD SAFETY
# ═══════════════════════════════════════════════════════════════════

class TestThreadSafety:
    def test_parallel_process_price_no_crash(self):
        """100 parallel _process_price calls should not crash."""
        bot = make_bot()
        pos = make_position(entry=74000, atr=350)
        bot.positions.append(pos)

        errors = []

        def call_price(p):
            try:
                bot._process_price("BTC/USDT", p)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(100):
            price = 74000 + np.random.randn() * 100
            t = threading.Thread(target=call_price, args=(price,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Errors: {errors}"

    def test_save_state_no_deadlock(self):
        """_save_state during active trading should not deadlock."""
        bot = make_bot()
        bot.vol_bars["BTC/USDT"] = make_vol_bars()
        bot.vol_thresholds["BTC/USDT"] = 1000
        pos = make_position(entry=74000, atr=350)
        bot.positions.append(pos)

        done = threading.Event()

        def save_loop():
            for _ in range(10):
                bot._save_state()
                time.sleep(0.01)
            done.set()

        def trade_loop():
            for i in range(50):
                bot._process_price("BTC/USDT", 74000 + i * 0.5)
                time.sleep(0.005)

        t1 = threading.Thread(target=save_loop)
        t2 = threading.Thread(target=trade_loop)
        t1.start()
        t2.start()

        t1.join(timeout=10)
        t2.join(timeout=10)

        assert done.is_set(), "Deadlock detected — save_state never completed"


# ═══════════════════════════════════════════════════════════════════
# MOVE SL STEPS CONFIG VALIDATION
# ═══════════════════════════════════════════════════════════════════

class TestConfig:
    def test_move_sl_steps_count(self):
        """Should have 4 steps."""
        assert len(MOVE_SL_STEPS) == 4

    def test_step1_is_scalp(self):
        """Step1 activation=0.25, move_to=0.30."""
        assert MOVE_SL_STEPS[0] == (0.25, 0.30)

    def test_steps_activation_increasing(self):
        """Activation levels should be increasing."""
        activations = [s[0] for s in MOVE_SL_STEPS]
        for i in range(1, len(activations)):
            assert activations[i] >= activations[i-1]

    def test_trail_positive(self):
        """Trail distance should be positive."""
        assert MOVE_SL_TRAIL > 0
