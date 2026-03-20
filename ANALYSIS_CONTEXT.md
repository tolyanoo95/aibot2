# Data for Strategy Analysis

## Files in this folder

### Real trading data
- `all_trades.txt` — trades.log from all 7 bots (4147 lines). Format per bot section:
  - `===aibot-v2===` header for each bot
  - `OPEN SHORT LINK/USDT @ 9.0350 SL=9.0804 TP=8.9442 ... ATR=0.02270 ADX=21 roc=-0.30%`
  - `CLOSE SHORT LINK/USDT @ 9.0804 | HARD_SL | PnL -0.57% (gross -0.50% fee -0.07%) | DCA:1 | Bars:1 | MFE:+0.10% MAE:0.62%`
  - `SL_MOVED SHORT BTC/USDT step 1/4 new_sl=69728.46 entry=69800.00 ATR=238.48`
  - `BLOCKED SHORT DOT/USDT @ 1.5410 | RSI_slope=+7.9 | roc=-0.77% ADX=22 ...`

- `hold_data.txt` — 5804 lines of SIGNAL_DATA + HOLD_DATA (every 5s during trades) + OPEN/CLOSE from bot 0.25.
  Contains 68 real-time metrics from 4 WebSocket streams at entry and during each trade.

### Bot configs (7 bots, Move SL activation differs)
| Bot | Dir | Move SL step1 | Step0 |
|---|---|---|---|
| 0.25 | aibot-v2 | (0.25, 0.30) | — |
| 0.26 | aibot-v2-026 | (0.26, 0.31) | — |
| 0.27 | aibot-v2-027 | (0.27, 0.32) | — |
| 0.28 | aibot-v2-028 | (0.28, 0.33) | — |
| 0.29 | aibot-v2-029 | (0.29, 0.34) | — |
| 0.30 | aibot-v2-030 | (0.30, 0.35) | — |
| 0.30+S50 | aibot-v2-030s50 | (0.30, 0.35) | (0.10, -0.50) |
All have rest steps: (0.5, 0.25), (1.0, 0.5), (2.0, 1.0) + trail 1.0

### Source code
- `main_volbars.py` — live bot (~2300 lines, 68 metrics, 4 WS streams)
- `backtest_volbars.py` — walk-forward backtest

## Known facts from analysis

### Current results (287 trades, 7 bots, Mar 19-20, 2 days)
- 244 wins ($+289), 43 losses ($-1051), NET: $-762
- Avg win: $1.18, Avg loss: $24.43
- R:R = 1:21, Required WR for breakeven: 95%, Actual WR: 85%
- Commission: 0.07% per trade (maker entry + taker exit)
- Position: $455 margin × 5x leverage = $2,275 notional

### Price behavior after signal
- MFE (our direction): avg 0.14%
- MAE (against us): avg 0.37% — 2.7x more against
- On losses: MAE 1.03% — 10x more against

### What doesn't work
- Move SL scalp: R:R 1:21, needs 95% WR
- Entry confirmation (check metrics before entering): blocks 11 wins per 1 loss
- Hold exit filters: too noisy on 5s timeframe
- Hedge with 0.07% fee: double commission kills profit
- Smaller SL: more stops hit, worse overall

### What might work (needs validation)
1. No Move SL: backtest 30d = +$505/mo, 1200d = +$94/mo
2. Mean reversion (trade OPPOSITE to signal): showed +$63 on 41 trades but only 2 days
3. ADX 35+ only: 96% WR (but only 23 trades)
4. Reactive hedge: promising simulation but 0 real trades yet
5. Limit orders (0.04% fee): doubles net profit on scalps

## Task
Analyze all_trades.txt and hold_data.txt to find the best strategy. Consider:
1. Which pairs are profitable, which are not?
2. Is momentum the right approach or should we try mean reversion?
3. What ADX/ROC thresholds work best?
4. Can 68 microstructure metrics predict win vs loss?
5. What R:R ratio is achievable?
6. Propose a concrete strategy with backtestable parameters.
