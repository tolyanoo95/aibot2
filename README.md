# AI Crypto Futures Multi-Model Trading Bot

An AI-powered multi-model trading bot for cryptocurrency futures on Binance. Uses a **Regime Classifier** to detect market state (Trend/Reversal/Range), then applies the specialized model for each regime.

## Architecture

```
Binance Futures API
     │
     ▼
┌──────────────────────────┐
│   Data Pipeline           │
│   OHLCV (5m/15m/1h)     │
│   Funding Rate, OI, L/S  │
│   Order Book, Liq Zones  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Feature Engineering     │
│   58 base + 13 reversal  │
│   + 7 range + 12 regime  │
│   = 90 total features    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Regime Classifier       │
│   TREND / REVERSAL / RANGE│
│   LightGBM (72.9%)       │
└─────┬──────┬──────┬──────┘
      │      │      │
      ▼      ▼      ▼
┌────────┐┌────────┐┌────────┐
│ Trend  ││Reversal││ Range  │
│ Model  ││ Model  ││ Model  │
│XGB+LGB ││XGB+LGB ││XGB+LGB │
│+CatB   ││        ││        │
│71.6%   ││ 90.8%  ││ 76.0%  │
└───┬────┘└───┬────┘└───┬────┘
    │         │         │
    ▼         ▼         ▼
┌──────────────────────────┐
│   Signal Generator        │
│   Dynamic SL/TP per regime│
│   Trend: 2.5x ATR        │
│   Reversal: 1.5x ATR     │
│   Range: 1.0x ATR        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Trade Executor          │
│   Position sizing by conf │
│   Reversal protection     │
│   Drawdown monitor        │
│   Paper / Live mode       │
└──────────────────────────┘
```

## Models

| Model | Type | Accuracy | Purpose |
|-------|------|----------|---------|
| **Trend Model** | XGBoost + LightGBM + CatBoost | 71.6% | Follow trends (BUY/SELL) |
| **Regime Classifier** | LightGBM | 72.9% | Detect market state |
| **Reversal Model** | XGBoost + LightGBM | 90.8% | Catch reversals |
| **Range Model** | XGBoost + LightGBM | 76.0% | Scalp in sideways market |

## Quick Start

```bash
# Clone
git clone https://github.com/tolyanoo95/aibot2.git
cd aibot2

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your Binance API keys

# Train all models (parallel, ~7 min)
python train_all.py

# Run single scan
python main.py --once

# Run continuous scanner
python main.py
```

## Training

```bash
# Train all 4 models at once (recommended)
python train_all.py

# Or train individually
python train.py            # Trend Model
python train_regime.py     # Regime Classifier
python train_reversal.py   # Reversal Model
python train_range.py      # Range Model
```

## Configuration

Key settings in `src/config.py` and `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `TRADING_MODE` | paper | paper or live |
| `PREDICTION_THRESHOLD` | 0.70 | Min confidence to open trade |
| `SL_ATR_MULTIPLIER` | 2.5 | Stop-loss distance (dynamic per regime) |
| `TP_ATR_MULTIPLIER` | 2.5 | Take-profit distance (dynamic per regime) |
| `MAX_OPEN_TRADES` | 11 | Max simultaneous positions |
| `MAX_HOLD_BARS` | 12 | Max holding time (60 min on 5m) |
| `TRAIN_CANDLES` | 30000 | Training data (~104 days) |

## Risk Management

- **3 consecutive SL** → 2 hour pause on all trading
- **Per-pair SL cooldown** → 40 min after SL on specific pair
- **Dynamic confidence threshold** → raises after consecutive losses
- **Position sizing by confidence** → 90%=1.5x, 80%=1x, 70%=0.5x
- **Drawdown monitor** → halve positions if peak PnL drops 3%
- **BTC as leader** → skip alts if BTC is uncertain

## File Structure

```
├── main.py                 # Main scanner loop
├── train_all.py            # Train all models (parallel)
├── train.py                # Train Trend Model
├── train_regime.py         # Train Regime Classifier
├── train_reversal.py       # Train Reversal Model
├── train_range.py          # Train Range Model
├── auto_retrain.py         # Automatic retraining (cron)
├── backtest.py             # Backtesting
├── src/
│   ├── config.py           # Configuration
│   ├── data_fetcher.py     # Binance data
│   ├── indicators.py       # Technical indicators
│   ├── features.py         # Feature engineering (90 features)
│   ├── ml_model.py         # Trend Model (ensemble)
│   ├── regime_classifier.py # Regime Classifier
│   ├── reversal_model.py   # Reversal Model
│   ├── range_model.py      # Range Model
│   ├── signal_generator.py # Signal generation
│   ├── executor.py         # Trade execution
│   ├── trade_monitor.py    # Trade monitoring
│   ├── entry_refiner.py    # 1m entry refinement
│   ├── market_context.py   # OI, L/S, liquidations
│   ├── llm_analyzer.py     # LLM analysis (optional)
│   └── display.py          # Console display
├── models/                 # Trained models
└── .env                    # API keys
```

## Trading Pairs

BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT, LTC (USDT futures)
