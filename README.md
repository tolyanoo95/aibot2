# AI Crypto Futures Intraday Scanner

An AI-powered multi-pair scanner for cryptocurrency futures intraday trading on Binance. Combines a trained **ML ensemble (XGBoost + LightGBM + CatBoost)** with **OpenAI LLM analysis** (hybrid approach) to generate actionable trading signals with entry, stop-loss, and take-profit levels.

## How It Works

```
Binance Futures API          CoinGecko API
     │                            │
     ▼                            ▼
┌─────────────┐          ┌───────────────┐
│ OHLCV Data  │          │ USDT/USDC Dom │
│ OI, L/S     │          │ Total MCap    │
│ Funding Rate│          │ Stablecoin %  │
│ Order Book  │          └───────┬───────┘
└──────┬──────┘                  │
       │                         │
       ▼                         ▼
┌──────────────────────────────────────┐
│         Technical Indicators          │
│  RSI, MACD, Bollinger, EMA, ATR,     │
│  ADX, StochRSI, OBV, Volume Ratio    │
└──────────────────┬───────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌─────────────┐        ┌─────────────┐
│  ML Engine  │        │  LLM Engine │
│  (XGBoost)  │        │  (GPT-4o)   │
│             │        │             │
│ 37 features │        │ Full market │
│ 3-class     │        │ context     │
│ BUY/SELL/   │        │ + decision  │
│ HOLD        │        │ framework   │
│ Weight: 60% │        │ Weight: 40% │
└──────┬──────┘        └──────┬──────┘
       │                      │
       └──────────┬───────────┘
                  ▼
       ┌───────────────────┐
       │ Signal Filters    │
       │ Volume, ADX,      │
       │ Order Book, Time  │
       └────────┬──────────┘
                ▼
       ┌───────────────────┐
       │ 1m Entry Refiner  │
       │ EMA/BB/RSI touch  │
       └────────┬──────────┘
                ▼
       ┌───────────────────┐
       │ Signal Output     │
       │ FRESH / NEW / LATE│
       │ Entry, SL, TP     │
       └───────────────────┘
```

## Features

### Multi-Pair Scanner
- Scans **10 top futures pairs** simultaneously (BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT)
- Configurable pair list in `src/config.py`

### Multi-Timeframe Analysis
- **Primary (5m)** — main signal generation, 21 base technical features
- **Secondary (15m)** — confirmation, 6 higher-TF features
- **Trend (1h)** — macro trend filter, 6 higher-TF features

### Technical Indicators (21+)
RSI, MACD (line + histogram), Bollinger Bands (%B), EMAs (9/21/50), ATR, ADX (with DMP/DMN), Stochastic RSI, OBV, Volume Ratio, EMA crossovers, price changes (1/5/15 bars), candle anatomy (body, wicks), RSI divergence

### Market Context
| Data | Source | Purpose |
|------|--------|---------|
| Open Interest + history | Binance `/futures/data/` | Market conviction, OI change trend |
| Long/Short Ratio | Binance `/futures/data/` | Crowd positioning (contrarian signal) |
| Funding Rate | Binance Futures API | Cost of holding, sentiment |
| Order Book Imbalance | Binance `/fapi/v1/depth` | Bid/ask pressure, wall detection |
| USDT Dominance | CoinGecko `/global` | Risk-on/off sentiment |
| USDC Dominance | CoinGecko `/global` | Institutional flow |
| Total Stablecoin Dom | CoinGecko `/global` | Money flow in/out of crypto |
| Total Market Cap | CoinGecko `/global` | Overall market health |
| Liquidation Zones | Computed from swing points | Squeeze/cascade risk levels |

### ML Ensemble (XGBoost + LightGBM + CatBoost)
- **3 models vote together** — averaged probabilities for more stable signals
- **37 features** total (21 base + 12 multi-TF + 7 context including order book)
- **Triple-barrier labeling** — labels based on future price movement (ATR-scaled)
- **Time-series cross-validation** — 5-fold per model, respects temporal ordering
- **Class-balanced sample weights** — handles BUY/SELL/HOLD imbalance
- Trained on ~100K samples across 10 pairs (~35 days of 5m data)
- Each model learns differently → ensemble reduces false signals

### LLM Analysis (OpenAI)
- **Professional decision framework** prompt with 6-step analysis:
  1. Trend First (1H EMA direction)
  2. Momentum Confirmation (RSI + ADX)
  3. Smart Money Signals (L/S ratio, funding)
  4. Liquidation Magnets (price targets)
  5. Order Book Pressure (bid/ask imbalance)
  6. Macro Filter (stablecoin dominance)
- Returns JSON: direction, confidence (1-10), reasoning, key levels, risk assessment, market regime
- Rate-limited per pair (configurable interval)
- Acts as a "sanity filter" — blocks signals when macro context disagrees

### Signal Filters
| Filter | What it does | Default |
|--------|-------------|---------|
| **Volume** | Blocks signals when volume < 80% of average | 0.8 |
| **ADX** | Blocks signals when ADX < 15 (no trend) | 15 |
| **EMA Trend** | Blocks LONG when EMA9 < EMA21 (bearish), SHORT when EMA9 > EMA21 (bullish) | ON |
| **Order Book** | Blocks signals against strong book pressure (>0.5 imbalance) | 0.5 |
| **NYSE Protection** | Blocks new signals + closes open trades during NYSE open (13:00–15:00 UTC) | `[13, 14]` |

### 1-Minute Entry Refinement
After a 5m signal fires, the bot checks 1m candles for a better entry:
- **RSI oversold/overbought** on 1m → enter on pullback
- **EMA-9 touch** on 1m → enter at support/resistance
- **Bollinger Band touch** on 1m → enter at extreme
- SL/TP based on 5m ATR (consistent with signal generator)
- **Minimum SL floor** — SL never less than 0.3% from entry (prevents noise stop-outs)
- Falls back to market entry if no pullback found

### Signal Tracking & Freshness
| Status | Meaning |
|--------|---------|
| **FRESH SIGNAL** | Just appeared — best entry point |
| **NEW ~Xm** | Active for X minutes — still OK to enter |
| **LATE ~Xm** | Signal running for 15+ min — move may be partially done |
| **ACTIVE HH:MM** | Tracked since first appearance, shows original entry levels |

### Trade Monitoring (Trailing Stop + Health + Early Exit)
Active trade management — works in both live scanner and backtester:

| Feature | What it does |
|---------|-------------|
| **Trailing Stop** | Moves SL in your favor after price moves 2.0x ATR; trails at 1.5x ATR distance |
| **Opposite Signal** | Auto-closes trade if ML reverses direction with >65% confidence (only if in profit) |
| **Health Check** | Monitors RSI, ADX, volume each scan: HEALTHY → WEAKENING → CLOSE_EARLY |
| **Auto-Track** | Live scanner auto-opens monitoring when FRESH signal >55% appears |

Live display shows:
```
╭───────────────── Open Trades (1) ──────────────────╮
│  BNB/USDT SHORT  Entry 617.85 → Now 616.20 +0.27% │
│  Trail SL: 617.00  |  HEALTHY  |  15min            │
╰────────────────────────────────────────────────────╯
```

### Hybrid Signal Generation
- ML signal (60% weight) + LLM signal (40% weight) — weighted voting
- **Agreement bonus** (+20% confidence) when both agree
- **Disagreement = NEUTRAL** — when ML and LLM contradict, bot stays out
- ATR-based SL/TP (SL = 1.5x ATR, TP = 2.5x ATR → R:R = 1:1.67, backtest-optimized)
- **Minimum SL floor** of 0.3% — prevents tight stops in low-ATR periods
- Dynamic leverage based on confidence and risk level

### Paper & Live Trading
| Mode | What it does |
|------|-------------|
| **`TRADING_MODE=paper`** | Logs trades to `paper_trades.json`, no real orders. Safe for testing. |
| **`TRADING_MODE=live`** | Auto-executes market orders on Binance with SL/TP. Real money! |

Both modes feature:
- **Instant execution** — trade opens as soon as pair is scanned (~6-8 sec), not after all 10 finish
- **Position sizing** based on `TRADE_BALANCE_USDT` and risk per trade
- **Max open trades** limit (default 3 simultaneous positions)
- **Daily loss limit** — stops trading if daily loss exceeds 3%
- Paper trades persist between restarts (`paper_trades.json`)

### Parallel Execution
- **Live scanner**: 10 pairs scanned simultaneously (~8 sec total vs ~60 sec sequential)
- **Training**: 10 pairs fetched & processed in parallel with live status table
- **Backtesting**: 10 pairs backtested in parallel with live progress display

### Risk Management
- Configurable risk per trade (default 1%)
- Maximum daily loss limit (default 3%)
- Position sizing based on SL distance
- Leverage capped by confidence and risk assessment

### Backtesting
- **Out-of-sample**: trains on 70%, tests on unseen 30%
- **Realistic execution**: entry at next candle open, commissions (0.08%), slippage (0.01%)
- **Trailing stop + early exit** active during backtest (same as live)
- **Filters included**: volume and ADX filters active during backtest
- **LLM backtest**: separate script for testing with GPT analysis per trade
- **Exit reasons**: TP, SL, Trail SL, Early Exit, Timeout
- **Backtest result**: ~+99% net PnL over ~10 days (out-of-sample, after commissions, Win Rate 66%, Sharpe 16, Max DD 3.8%)

### Auto-Retrain
- `auto_retrain.py` — trains new model on fresh data
- Compares accuracy with current model
- Replaces only if new model is better (or equal)
- History logged in `models/retrain_history.json`
- Cron-ready: schedule every 2 days

## Project Structure

```
aibot/
├── src/
│   ├── config.py           # All settings (API keys, pairs, timeframes, thresholds)
│   ├── data_fetcher.py     # Binance Futures data fetching (OHLCV, funding, ticker)
│   ├── indicators.py       # Technical indicator calculations (pandas_ta)
│   ├── features.py         # Feature engineering, labeling, class balancing
│   ├── ml_model.py         # XGBoost classifier (train, predict, save/load)
│   ├── llm_analyzer.py     # OpenAI integration (decision framework prompt → JSON)
│   ├── market_context.py   # OI, L/S ratio, CoinGecko dominance, order book, liq zones
│   ├── signal_generator.py # Hybrid signal combination + filters (ML + LLM → Signal)
│   ├── entry_refiner.py    # 1m entry refinement (pullback, EMA/BB touch)
│   ├── trade_monitor.py    # Trailing stop, health analysis, early exit
│   ├── executor.py         # Paper/live trade execution (Binance orders)
│   ├── risk_manager.py     # Position sizing, daily loss limits
│   └── display.py          # Rich console output (tables, panels, trade monitoring)
├── models/                 # Saved ML models (.json + .pkl metadata)
├── main.py                 # Live scanner (loop or --once)
├── train.py                # ML training pipeline
├── backtest.py             # Realistic out-of-sample backtester
├── backtest_llm.py         # LLM-enhanced backtest (per-trade GPT analysis)
├── auto_retrain.py         # Auto-retrain with accuracy comparison
├── deploy.sh               # One-command server deployment script
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
BINANCE_API_KEY=your_key
BINANCE_SECRET=your_secret
BINANCE_TESTNET=false
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
```

### 3. Train the ML model

```bash
python train.py
```

### 4. Run the scanner

```bash
python main.py          # continuous scanning every 5 min
python main.py --once   # single scan and exit
```

### 5. Backtest

```bash
python backtest.py                          # all 10 pairs, out-of-sample
python backtest.py --pair BTC/USDT          # single pair
python backtest.py --no-filters             # compare without filters
python backtest_llm.py --pair ETH/USDT      # with LLM analysis per trade
```

### 6. Deploy to server

```bash
scp -r . root@YOUR_SERVER:/root/aibot
ssh root@YOUR_SERVER "cd /root/aibot && bash deploy.sh"
```

## Configuration

All settings in `src/config.py`, overridable via `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `TRADING_PAIRS` | 10 top pairs | Futures pairs to scan |
| `PRIMARY_TIMEFRAME` | `5m` | Main signal timeframe |
| `PREDICTION_THRESHOLD` | `0.55` | Min confidence for signals (backtest-optimized) |
| `SL_ATR_MULTIPLIER` | `1.5` | Stop-loss = ATR × this |
| `TP_ATR_MULTIPLIER` | `2.5` | Take-profit = ATR × this (R:R = 1:1.67) |
| `MIN_SL_PCT` | `0.3` | Minimum SL distance in % (floor for low-ATR) |
| `MAX_HOLD_BARS` | `12` | Max holding time (60 min on 5m) |
| `SCAN_INTERVAL` | `300` | Seconds between scans |
| `USE_LLM` | `True` | Enable/disable LLM |
| `USE_1M_ENTRY` | `True` | Enable 1m entry refinement |
| `FILTER_MIN_VOLUME_RATIO` | `0.8` | Volume filter threshold |
| `FILTER_MIN_ADX` | `15.0` | ADX filter threshold |
| `FILTER_TREND_EMA` | `True` | Block signals against EMA9/EMA21 trend |
| `FILTER_DEAD_HOURS` | `[13, 14]` | Block signals + close trades during NYSE open (UTC hours) |
| `TRAILING_ACTIVATION_ATR` | `2.0` | Activate trailing after 2.0x ATR profit |
| `TRAILING_DISTANCE_ATR` | `1.5` | Trail SL at 1.5x ATR behind best price |
| `TRADING_MODE` | `paper` | `paper` = log only, `live` = real Binance orders |
| `TRADE_BALANCE_USDT` | `100` | Balance for position sizing |
| `MAX_OPEN_TRADES` | `3` | Max simultaneous positions |
| `RISK_PER_TRADE` | `0.01` | 1% risk per trade |
| `MAX_DAILY_LOSS` | `0.03` | 3% daily loss limit |

## Tech Stack

- **Python 3.12+**
- **ccxt** — unified crypto exchange API (Binance USDM Futures)
- **pandas + pandas_ta** — data manipulation and technical indicators
- **XGBoost + LightGBM + CatBoost** — ensemble of 3 gradient boosting classifiers
- **scikit-learn** — time-series cross-validation, metrics
- **OpenAI API** — GPT-4o-mini with professional trading decision framework
- **CoinGecko API** — global market data (free, no auth)
- **Binance Futures API** — OI, L/S ratio, order book (free, no auth)
- **Rich** — beautiful terminal output with signal tracking

## Disclaimer

This software is for **educational and research purposes only**. Cryptocurrency futures trading involves significant risk. Past performance of the ML model does not guarantee future results. Always do your own research and start with small amounts. The authors are not responsible for any financial losses incurred through the use of this software.
