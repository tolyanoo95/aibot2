import pandas as pd
import numpy as np

# Load the live features we just downloaded
df = pd.read_csv("data/live_features_temp.csv")
print(f"Total rows collected so far: {len(df)}")
print(f"Time range: {df['ts'].min()} to {df['ts'].max()}")

# Basic stats on OI and Liquidations
print("\n--- Basic Statistics ---")
print(df[['btc_delta_10s', 'sol_obi_10', 'sol_liq_buy', 'sol_liq_sell', 'sol_oi_change']].describe())

# How often do liquidations happen?
liq_rows = df[(df['sol_liq_buy'] > 0) | (df['sol_liq_sell'] > 0)]
print(f"\nNumber of 10s intervals with liquidations: {len(liq_rows)} ({len(liq_rows)/len(df)*100:.2f}%)")

if len(liq_rows) > 0:
    print("Average size of liquidations:")
    print(liq_rows[['sol_liq_buy', 'sol_liq_sell']].mean())

# Distribution of OI changes
print("\nOI Change distribution (percentiles):")
print(df['sol_oi_change'].quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))

# Let's look at extreme BTC delta events and what OI was doing
print("\n--- Extreme BTC Delta Events (Top 1% Long/Short) ---")
top_1_long = df['btc_delta_10s'].quantile(0.99)
top_1_short = df['btc_delta_10s'].quantile(0.01)

long_events = df[df['btc_delta_10s'] > top_1_long]
short_events = df[df['btc_delta_10s'] < top_1_short]

print(f"Threshold for top 1% BTC Buy Delta: {top_1_long:.2f}")
print(f"Average OI change during extreme buys: {long_events['sol_oi_change'].mean():.2f}")
print(f"Median OI change during extreme buys: {long_events['sol_oi_change'].median():.2f}")

print(f"\nThreshold for top 1% BTC Sell Delta: {top_1_short:.2f}")
print(f"Average OI change during extreme sells: {short_events['sol_oi_change'].mean():.2f}")
print(f"Median OI change during extreme sells: {short_events['sol_oi_change'].median():.2f}")

