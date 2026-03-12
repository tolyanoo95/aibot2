#!/usr/bin/env python3
"""
Test external filters against baseline (ADX>20, no lock, vol_drop+flip+move_sl).

Tests:
  1. Taker Buy Ratio guard (needs re-cached data with taker_buy_vol)
  2. Fear & Greed Index regime filter
  3. BTC cascade guard (block alt LONGs when BTC dumps)
  4. Funding Rate regime filter

Usage:
    python test_external_filters.py
    python test_external_filters.py --years 2024    # test specific year
    python test_external_filters.py --years all      # test 2023+2024+2025
"""
import argparse, os, pickle, functools, logging, warnings, json, time
import numpy as np, pandas as pd
import requests
from rich.console import Console
from rich.table import Table
from rich import box
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
print = functools.partial(print, flush=True)
console = Console()

from src.config import config
from src.indicators import TechnicalIndicators
from backtest_volbars import resample_to_volume_bars
from backtest_wf import simulate_dca_trades

CACHE_FILE = "data/ohlcv_cache_15m.pkl"
FNG_CACHE = "data/fng_history.json"
FUNDING_CACHE = "data/funding_history.pkl"

ADX_MIN = 20
SL_MULT = 2.0
TP_MULT = 4.0
LONG_MOM = 0.30
SHORT_MOM = 0.10
MAX_DCA = 3
MAX_OPEN = 11


def fetch_fng_history():
    """Download full Fear & Greed Index history from alternative.me."""
    if os.path.exists(FNG_CACHE):
        with open(FNG_CACHE) as f:
            data = json.load(f)
        console.print(f"[green]F&G cache loaded: {len(data)} days[/green]")
        return data

    console.print("Downloading Fear & Greed Index history...")
    resp = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=30)
    resp.raise_for_status()
    raw = resp.json()["data"]
    data = {}
    for item in raw:
        ts = int(item["timestamp"])
        dt = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        data[dt] = int(item["value"])

    os.makedirs("data", exist_ok=True)
    with open(FNG_CACHE, "w") as f:
        json.dump(data, f)
    console.print(f"[green]F&G saved: {len(data)} days ({min(data.keys())} to {max(data.keys())})[/green]")
    return data


def fetch_funding_history():
    """Download funding rate history for all pairs from Binance."""
    if os.path.exists(FUNDING_CACHE):
        with open(FUNDING_CACHE, "rb") as f:
            data = pickle.load(f)
        console.print(f"[green]Funding cache loaded: {len(data)} pairs[/green]")
        return data

    import ccxt
    exchange = ccxt.binanceusdm({
        "apiKey": config.BINANCE_API_KEY,
        "secret": config.BINANCE_SECRET,
        "enableRateLimit": True,
    })

    console.print("Downloading funding rate history...")
    all_funding = {}
    for symbol in config.TRADING_PAIRS:
        console.print(f"  {symbol}...")
        pair_sym = symbol.replace("/", "").replace(":USDT", "")
        rates = []
        end_time = int(time.time() * 1000)
        start_time = int((datetime(2022, 1, 1)).timestamp() * 1000)
        since = start_time

        while since < end_time:
            try:
                raw = exchange.fapiPublicGetFundingRate({
                    "symbol": pair_sym, "startTime": since, "limit": 1000
                })
                if not raw:
                    break
                for r in raw:
                    rates.append({
                        "timestamp": int(r["fundingTime"]),
                        "rate": float(r["fundingRate"]),
                    })
                since = int(raw[-1]["fundingTime"]) + 1
                time.sleep(0.2)
            except Exception as e:
                console.print(f"    Error: {e}")
                break

        if rates:
            df = pd.DataFrame(rates)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df[~df.index.duplicated(keep="last")]
            all_funding[symbol] = df
            console.print(f"    {len(df)} funding rates ({df.index[0]} to {df.index[-1]})")

    os.makedirs("data", exist_ok=True)
    with open(FUNDING_CACHE, "wb") as f:
        pickle.dump(all_funding, f)
    console.print(f"[green]Funding saved: {len(all_funding)} pairs[/green]")
    return all_funding


def load_and_prepare():
    """Load cached OHLCV, build volume bars with indicators."""
    with open(CACHE_FILE, "rb") as f:
        raw = pickle.load(f)
    indicators = TechnicalIndicators()
    all_pair_data = {}
    for symbol, df in raw.items():
        tdf = indicators.calculate_all(df.copy())
        vdf = resample_to_volume_bars(df)
        if len(vdf) < 1300:
            continue
        warmup = min(1920, len(df))
        htf_vdf = resample_to_volume_bars(df, initial_threshold=df["volume"].iloc[:warmup].median() * 10)
        if len(htf_vdf) > 20:
            htf_vdf = indicators.calculate_all(htf_vdf)
            for col in ["ema_9", "ema_21", "ema_50"]:
                if col in htf_vdf.columns:
                    vdf[f"htf_{col}"] = htf_vdf[col].reindex(vdf.index, method="ffill")
        vdf = indicators.calculate_all(vdf)
        time_atr = tdf["atr"].reindex(vdf.index, method="ffill")
        vdf["atr"] = time_atr.values

        if "taker_buy_vol" in df.columns:
            tbr = (df["taker_buy_vol"] / df["volume"].replace(0, np.nan)).fillna(0.5)
            vdf["taker_buy_ratio"] = tbr.reindex(vdf.index, method="ffill")
            taker_sell = df["volume"] - df["taker_buy_vol"]
            cvd = (df["taker_buy_vol"] - taker_sell).cumsum()
            vdf["cvd"] = cvd.reindex(vdf.index, method="ffill")
            vdf["cvd_slope_12"] = vdf["cvd"].diff(12)

        if "num_trades" in df.columns:
            nt = df["num_trades"].reindex(vdf.index, method="ffill")
            nt_ma = nt.rolling(20, min_periods=1).median()
            vdf["num_trades_ratio"] = (nt / nt_ma.replace(0, 1)).fillna(1.0)

        all_pair_data[symbol] = vdf
    console.print(f"Loaded {len(all_pair_data)} pairs from cache")
    has_taker = any("taker_buy_ratio" in df.columns for df in all_pair_data.values())
    console.print(f"  taker_buy_vol available: {has_taker}")
    return all_pair_data, raw


def generate_signals(df_test, atr_test, atr_ma20, symbol,
                     fng_values=None, funding_data=None, btc_roc_1h=None,
                     filter_fng=False, filter_funding=False,
                     filter_btc_cascade=False, filter_taker=False,
                     filter_num_trades=False):
    """Generate signals with optional external filters."""
    signals = []
    for j in range(len(df_test)):
        roc = float(df_test["roc_12"].iloc[j]) if "roc_12" in df_test.columns else 0

        if roc > LONG_MOM:
            direction = "LONG"
        elif roc < -SHORT_MOM:
            direction = "SHORT"
        else:
            continue

        # HTF trend filter
        htf_e9 = float(df_test["htf_ema_9"].iloc[j]) if "htf_ema_9" in df_test.columns else 0
        htf_e21 = float(df_test["htf_ema_21"].iloc[j]) if "htf_ema_21" in df_test.columns else 0
        if htf_e9 > 0 and htf_e21 > 0:
            if direction == "LONG" and htf_e9 < htf_e21:
                continue
            if direction == "SHORT" and htf_e9 > htf_e21:
                continue

        # ATR expansion
        atr_exp = atr_test[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
        if atr_exp > 1.5:
            continue

        # ADX guard
        adx = float(df_test["ADX_14"].iloc[j]) if "ADX_14" in df_test.columns else 25
        if adx < ADX_MIN:
            continue

        # RSI slope
        rsi_s6 = float(df_test["rsi"].diff(6).iloc[j]) if "rsi" in df_test.columns else 0
        if direction == "LONG" and rsi_s6 < 0:
            continue
        if direction == "SHORT" and rsi_s6 > 0:
            continue

        # === EXTERNAL FILTERS ===

        # Fear & Greed filter
        if filter_fng and fng_values is not None:
            bar_date = df_test.index[j]
            dt_str = bar_date.strftime("%Y-%m-%d") if hasattr(bar_date, 'strftime') else ""
            fng_val = fng_values.get(dt_str, 50)
            if direction == "LONG" and fng_val < 20:
                continue
            if direction == "SHORT" and fng_val > 80:
                continue

        # Funding Rate regime filter
        if filter_funding and funding_data is not None and symbol in funding_data:
            bar_time = df_test.index[j]
            fund_df = funding_data[symbol]
            mask = fund_df.index <= bar_time
            if mask.any():
                last_rate = fund_df.loc[mask].iloc[-1]["rate"]
                if direction == "LONG" and last_rate > 0.0005:
                    continue
                if direction == "SHORT" and last_rate < -0.0005:
                    continue

        # BTC cascade guard (block alt LONGs when BTC dumps)
        if filter_btc_cascade and btc_roc_1h is not None and symbol != "BTC/USDT":
            bar_time = df_test.index[j]
            mask = btc_roc_1h.index <= bar_time
            if mask.any():
                btc_roc = btc_roc_1h.loc[mask].iloc[-1]
                if direction == "LONG" and btc_roc < -1.0:
                    continue

        # Taker Buy Ratio guard
        if filter_taker and "taker_buy_ratio" in df_test.columns:
            tbr = float(df_test["taker_buy_ratio"].iloc[j])
            if direction == "LONG" and tbr < 0.40:
                continue
            if direction == "SHORT" and tbr > 0.60:
                continue

        # Num trades guard
        if filter_num_trades and "num_trades_ratio" in df_test.columns:
            ntr = float(df_test["num_trades_ratio"].iloc[j])
            if ntr < 0.7:
                continue

        signals.append({"bar_idx": j, "symbol": symbol, "direction": direction, "confidence": 0.90})

    return signals


def run_backtest(data, fng_values=None, funding_data=None, btc_roc_1h=None,
                 filter_fng=False, filter_funding=False,
                 filter_btc_cascade=False, filter_taker=False,
                 filter_num_trades=False,
                 train_bars=1000, test_bars=300, year_filter=None):
    """Run full walk-forward backtest with optional filters."""
    all_trades = []
    fold = 0
    min_len = min(len(df) for df in data.values())
    start = 0

    while start + train_bars + test_bars <= min_len:
        fold += 1
        train_end = start + train_bars
        test_end = train_end + test_bars
        fold_trades = []

        for symbol, df_full in data.items():
            if len(df_full) < test_end:
                continue
            df_test = df_full.iloc[train_end:test_end]
            if len(df_test) < 20:
                continue

            if year_filter:
                test_year = df_test.index[len(df_test)//2].year if hasattr(df_test.index[0], 'year') else 0
                if test_year != year_filter:
                    start += test_bars
                    break

            atr_test = df_test["atr"].values
            atr_ma20 = pd.Series(atr_test).rolling(20, min_periods=1).mean().values

            signals = generate_signals(
                df_test, atr_test, atr_ma20, symbol,
                fng_values=fng_values, funding_data=funding_data,
                btc_roc_1h=btc_roc_1h,
                filter_fng=filter_fng, filter_funding=filter_funding,
                filter_btc_cascade=filter_btc_cascade, filter_taker=filter_taker,
                filter_num_trades=filter_num_trades,
            )

            trades = simulate_dca_trades(
                df_test, signals, tp_mult=TP_MULT, dca_step_mult=1.0,
                max_entries=MAX_DCA, hard_sl_mult=SL_MULT, max_hold=24,
                max_open=MAX_OPEN, cooldown=3, threshold=0.10,
                full_size_dca=True, move_sl_at=2.0, move_sl_to=1.0,
            )
            fold_trades.extend(trades)
        else:
            all_trades.extend(fold_trades)
            start += test_bars
            continue
        continue

    if not all_trades:
        return {"trades": 0, "wr": 0, "net": 0, "tp": 0, "sl": 0, "timeout": 0, "vol_drop": 0}

    n = len(all_trades)
    total_pnl = sum(t.pnl_pct for t in all_trades)
    wr = sum(1 for t in all_trades if t.pnl_pct > 0) / n * 100
    pos_size = 1000 * 5 / 11
    net = pos_size * total_pnl / 100 - 2 * pos_size * 0.0002 * n
    tp = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl = sum(1 for t in all_trades if t.exit_reason in ("SL", "HARD_SL"))
    timeout = sum(1 for t in all_trades if t.exit_reason == "TIMEOUT")
    vol_drop = sum(1 for t in all_trades if t.exit_reason == "VOL_DROP")

    return {"trades": n, "wr": wr, "net": net, "tp": tp, "sl": sl,
            "timeout": timeout, "vol_drop": vol_drop, "pnl_pct": total_pnl}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, default="all", help="Year to test: 2023, 2024, 2025, or 'all'")
    args = parser.parse_args()

    console.print("[bold]Loading data...[/bold]")
    data, raw_ohlcv = load_and_prepare()

    console.print("\n[bold]Fetching external data...[/bold]")
    fng_values = fetch_fng_history()
    funding_data = fetch_funding_history()

    btc_roc_1h = None
    if "BTC/USDT" in raw_ohlcv:
        btc_df = raw_ohlcv["BTC/USDT"]
        btc_close_1h = btc_df["close"].resample("1h").last().dropna()
        btc_roc_1h = btc_close_1h.pct_change(1) * 100

    has_taker = any("taker_buy_ratio" in df.columns for df in data.values())

    if args.years == "all":
        years_to_test = [None]
        year_labels = ["All"]
    else:
        years_to_test = [int(args.years)]
        year_labels = [args.years]

    for year, year_label in zip(years_to_test, year_labels):
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold yellow]  Testing: {year_label}[/bold yellow]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")

        tests = [
            ("Baseline (ADX>20, no lock)", {}),
            ("+ Fear & Greed (F<20/G>80)", {"filter_fng": True}),
            ("+ Funding Rate regime", {"filter_funding": True}),
            ("+ BTC cascade (alt LONG)", {"filter_btc_cascade": True}),
        ]

        if has_taker:
            tests.extend([
                ("+ Taker Buy Ratio", {"filter_taker": True}),
                ("+ Num Trades filter", {"filter_num_trades": True}),
                ("+ Taker + Num Trades", {"filter_taker": True, "filter_num_trades": True}),
            ])

        tests.extend([
            ("+ F&G + Funding + BTC", {"filter_fng": True, "filter_funding": True, "filter_btc_cascade": True}),
        ])

        if has_taker:
            tests.append(
                ("+ ALL filters", {"filter_fng": True, "filter_funding": True,
                                   "filter_btc_cascade": True, "filter_taker": True,
                                   "filter_num_trades": True}),
            )

        results = []
        for name, kwargs in tests:
            console.print(f"  Running: {name}...")
            r = run_backtest(
                data, fng_values=fng_values, funding_data=funding_data,
                btc_roc_1h=btc_roc_1h, year_filter=year,
                **kwargs
            )
            r["name"] = name
            results.append(r)
            console.print(f"    Trades={r['trades']} WR={r['wr']:.1f}% NET=${r['net']:+,.0f} "
                          f"TP={r['tp']} SL={r['sl']} TO={r['timeout']} VD={r['vol_drop']}")

        table = Table(title=f"External Filters Comparison ({year_label})", box=box.ROUNDED)
        table.add_column("Filter", style="bold")
        table.add_column("Trades", justify="right")
        table.add_column("WR", justify="right")
        table.add_column("NET", justify="right")
        table.add_column("vs Baseline", justify="right")
        table.add_column("TP", justify="right")
        table.add_column("SL", justify="right")

        base_net = results[0]["net"] if results else 0
        for r in results:
            delta = r["net"] - base_net
            delta_pct = (delta / abs(base_net) * 100) if base_net != 0 else 0
            style = "green" if delta > 0 else ("red" if delta < 0 else "")
            vs = f"${delta:+,.0f} ({delta_pct:+.1f}%)" if r["name"] != results[0]["name"] else "—"
            table.add_row(
                r["name"], str(r["trades"]), f"{r['wr']:.1f}%",
                f"${r['net']:+,.0f}", vs,
                str(r["tp"]), str(r["sl"]),
                style=style if r["name"] != results[0]["name"] else "",
            )
        console.print(table)


if __name__ == "__main__":
    main()
