# AI Crypto Futures Intraday Scanner

An AI-powered multi-pair scanner for cryptocurrency futures intraday trading on Binance. Combines a trained **XGBoost ML model** with **OpenAI LLM analysis** (hybrid approach) to generate actionable trading signals with entry, stop-loss, and take-profit levels.

## How It Works

```
Binance Futures API          CoinGecko API
     │                            │
     ▼                            ▼
┌─────────────┐          ┌───────────────┐
│ OHLCV Data  │          │ USDT/USDC Dom │
│ OI, L/S     │          │ Total MCap    │
│ Funding Rate│          │ Stablecoin %  │
└──────┬──────┘          └───────┬───────┘
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
│ 36 features │        │ Full market │
│ 3-class     │        │ context     │
│ BUY/SELL/   │        │ prompt      │
│ HOLD        │        │             │
│ Weight: 60% │        │ Weight: 40% │
└──────┬──────┘        └──────┬──────┘
       │                      │
       └──────────┬───────────┘
                  ▼
       ┌───────────────────┐
       │ Signal Generator  │
       │ Weighted voting   │
       │ + agreement bonus │
       │ + SL/TP via ATR   │
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
| USDT Dominance | CoinGecko `/global` | Risk-on/off sentiment |
| USDC Dominance | CoinGecko `/global` | Institutional flow |
| Total Stablecoin Dom | CoinGecko `/global` | Money flow in/out of crypto |
| Total Market Cap | CoinGecko `/global` | Overall market health |
| Liquidation Zones | Computed from swing points | Squeeze/cascade risk levels |

### ML Model (XGBoost)
- **36 features** total (21 base + 12 multi-TF + 6 context)
- **Triple-barrier labeling** — labels based on future price movement (ATR-scaled), not arbitrary thresholds
- **Time-series cross-validation** — 5-fold, respects temporal ordering (no data leakage)
- **Class-balanced sample weights** — handles BUY/SELL/HOLD imbalance
- Trained on ~100K samples across 10 pairs (~35 days of 5m data)

### LLM Analysis (OpenAI)
- Receives full structured prompt: indicators, last 10 candles, 1h trend, OI, L/S ratio, stablecoin dominance, total market cap, liquidation zones
- Returns JSON: direction, confidence (1-10), reasoning, key levels, risk assessment, market regime
- Rate-limited per pair (configurable interval)
- Acts as a "sanity filter" — blocks signals when macro context disagrees

### Hybrid Signal Generation
- ML signal (60% weight) + LLM signal (40% weight) — weighted voting
- **Agreement bonus** (+20% confidence) when both agree
- **Disagreement = NEUTRAL** — when ML and LLM contradict, bot stays out
- ATR-based SL/TP (default R:R = 1:2)
- Dynamic leverage based on confidence and risk level

### Risk Management
- Configurable risk per trade (default 1%)
- Maximum daily loss limit (default 3%)
- Position sizing based on SL distance
- Leverage capped by confidence and risk assessment

## Project Structure

```
aibot/
├── src/
│   ├── config.py           # All settings (API keys, pairs, timeframes, thresholds)
│   ├── data_fetcher.py     # Binance Futures data fetching (OHLCV, funding, ticker)
│   ├── indicators.py       # Technical indicator calculations (pandas_ta)
│   ├── features.py         # Feature engineering, labeling, class balancing
│   ├── ml_model.py         # XGBoost classifier (train, predict, save/load)
│   ├── llm_analyzer.py     # OpenAI integration (structured prompt → JSON)
│   ├── market_context.py   # OI, L/S ratio, CoinGecko dominance, liq zones
│   ├── signal_generator.py # Hybrid signal combination (ML + LLM → Signal)
│   ├── risk_manager.py     # Position sizing, daily loss limits
│   └── display.py          # Rich console output (tables, panels)
├── models/                 # Saved ML models (.json + .pkl metadata)
├── main.py                 # Scanner entry point (loop or --once)
├── train.py                # ML training pipeline
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

This fetches ~35 days of historical data across 3 timeframes for all 10 pairs, computes features, creates triple-barrier labels, and trains XGBoost with time-series cross-validation.

### 4. Run the scanner

```bash
python main.py          # continuous scanning every 5 min
python main.py --once   # single scan and exit
```

## Configuration

All settings are in `src/config.py` and can be overridden via `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `TRADING_PAIRS` | 10 top pairs | List of futures pairs to scan |
| `PRIMARY_TIMEFRAME` | `5m` | Main signal timeframe |
| `SCAN_INTERVAL` | `300` (5 min) | Seconds between scans |
| `USE_LLM` | `True` | Enable/disable LLM analysis |
| `PREDICTION_THRESHOLD` | `0.6` | Minimum confidence for signals |
| `RISK_PER_TRADE` | `0.01` (1%) | Risk per trade |
| `MAX_DAILY_LOSS` | `0.03` (3%) | Daily loss limit |
| `DEFAULT_LEVERAGE` | `5` | Base leverage |
| `MAX_LEVERAGE` | `20` | Maximum allowed leverage |

## Tech Stack

- **Python 3.12+**
- **ccxt** — unified crypto exchange API (Binance USDM Futures)
- **pandas + pandas_ta** — data manipulation and technical indicators
- **XGBoost** — gradient boosting classifier for signal prediction
- **scikit-learn** — time-series cross-validation, metrics
- **OpenAI API** — GPT-4o-mini for market analysis
- **CoinGecko API** — global market data (free, no auth)
- **Binance Futures API** — OI, L/S ratio (free, no auth)
- **Rich** — beautiful terminal output

## Disclaimer

This software is for **educational and research purposes only**. Cryptocurrency futures trading involves significant risk. Past performance of the ML model does not guarantee future results. Always do your own research. The authors are not responsible for any financial losses incurred through the use of this software.
