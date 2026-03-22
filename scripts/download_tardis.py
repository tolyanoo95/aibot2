"""
Download liquidations + derivative_ticker (OI, funding) from Tardis.dev
Uses 14 API keys in parallel for speed.
"""
import os
import sys
import time
import logging
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

KEYS = [
    "TD.KDeubOKBOaOItHn6.rq2Tkeqseo2dt9x.rz-lob9od0fD--Y.2LWH9ZScU7GUvW3.dWSpVV5T-vKB6Mf.TGq-",
    "TD.sSGNjEpohBl9i0FV.zz0-s-b6DZ9c2lt.uoNpgBPGaTITrA9.YlW26JKS0PSfoOG.2pEnnkHAnTpOloD.3wMo",
]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
           "DOTUSDT", "LINKUSDT", "DOGEUSDT", "AVAXUSDT", "LTCUSDT", "XRPUSDT"]

DATA_TYPES = ["liquidations", "derivative_ticker"]
EXCHANGE = "bybit"
BASE_URL = "https://datasets.tardis.dev/v1"
DOWNLOAD_DIR = "data/tardis"

START = datetime(2026, 2, 1)
END = datetime(2026, 2, 28)


def build_tasks():
    tasks = []
    d = START
    while d <= END:
        year, month, day = d.strftime("%Y"), d.strftime("%m"), d.strftime("%d")
        for dtype in DATA_TYPES:
            if dtype == "liquidations":
                # liquidations: one PERPETUALS file per day has all symbols
                url = f"{BASE_URL}/{EXCHANGE}/{dtype}/{year}/{month}/{day}/PERPETUALS.csv.gz"
                out = os.path.join(DOWNLOAD_DIR, EXCHANGE, dtype, year, month, day, "PERPETUALS.csv.gz")
                tasks.append((url, out, f"{dtype}/{d:%Y-%m-%d}/PERPETUALS"))
            else:
                for sym in SYMBOLS:
                    url = f"{BASE_URL}/{EXCHANGE}/{dtype}/{year}/{month}/{day}/{sym}.csv.gz"
                    out = os.path.join(DOWNLOAD_DIR, EXCHANGE, dtype, year, month, day, f"{sym}.csv.gz")
                    tasks.append((url, out, f"{dtype}/{d:%Y-%m-%d}/{sym}"))
        d += timedelta(days=1)
    return tasks


MAX_RETRIES = 3

def download_one(url, out_path, label, api_key):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return label, "skip", 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=180, stream=True)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                sz = os.path.getsize(out_path)
                return label, "ok", sz
            elif r.status_code == 401:
                return label, "HTTP 401 (no access)", 0
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                return label, f"HTTP {r.status_code}", 0
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return label, f"error: {e}", 0
    return label, "max retries", 0


def main():
    tasks = build_tasks()
    logging.info(f"Total tasks: {len(tasks)} | Keys: {len(KEYS)} | Workers: {len(KEYS)}")

    done, failed, skipped = 0, 0, 0
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=len(KEYS)) as pool:
        futures = {}
        for i, (url, out, label) in enumerate(tasks):
            key = KEYS[i % len(KEYS)]
            f = pool.submit(download_one, url, out, label, key)
            futures[f] = label

        for f in as_completed(futures):
            label, status, sz = f.result()
            if status == "ok":
                done += 1
                total_bytes += sz
                logging.info(f"  ✅ {label}  ({sz/1024:.0f} KB)")
            elif status == "skip":
                skipped += 1
                logging.info(f"  ⏭️  {label}  (already exists)")
            else:
                failed += 1
                logging.warning(f"  ❌ {label}  {status}")

    logging.info(f"\nDone: {done} | Skipped: {skipped} | Failed: {failed} | Total: {total_bytes/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
