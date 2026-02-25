#!/usr/bin/env python3
"""
Train ALL models at once: Trend + Regime + Reversal + Range.
Stops the bot before training to avoid rate limits.

Usage:
    python train_all.py
"""

import logging
import sys
import time
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
console = Console()


def train_all():
    t0 = time.time()
    console.print(Panel.fit(
        f"[bold]Multi-Model Training Pipeline[/bold]\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Models: Trend + Regime + Reversal + Range",
        border_style="green",
    ))

    results = {}

    # ── 1. Trend Model ─────────────────────────────────────
    console.print("\n[bold cyan]═══ 1/4 TREND MODEL ═══[/bold cyan]")
    try:
        from train import train_model_and_return_metrics
        metrics = train_model_and_return_metrics()
        results["Trend"] = {
            "accuracy": metrics.get("cv_accuracy", 0),
            "std": metrics.get("cv_std", 0),
            "status": "OK",
        }
        console.print(f"[green]Trend Model: {metrics.get('cv_accuracy', 0):.4f}[/green]")
    except Exception as exc:
        results["Trend"] = {"accuracy": 0, "std": 0, "status": f"FAIL: {exc}"}
        console.print(f"[red]Trend Model FAILED: {exc}[/red]")

    # ── 2. Regime Classifier ───────────────────────────────
    console.print("\n[bold cyan]═══ 2/4 REGIME CLASSIFIER ═══[/bold cyan]")
    try:
        from train_regime import train_regime
        metrics = train_regime()
        results["Regime"] = {
            "accuracy": metrics.get("cv_accuracy", 0),
            "std": metrics.get("cv_std", 0),
            "status": "OK",
        }
        console.print(f"[green]Regime Classifier: {metrics.get('cv_accuracy', 0):.4f}[/green]")
    except Exception as exc:
        results["Regime"] = {"accuracy": 0, "std": 0, "status": f"FAIL: {exc}"}
        console.print(f"[red]Regime Classifier FAILED: {exc}[/red]")

    # ── 3. Reversal Model ──────────────────────────────────
    console.print("\n[bold cyan]═══ 3/4 REVERSAL MODEL ═══[/bold cyan]")
    try:
        from train_reversal import train_reversal
        metrics = train_reversal()
        results["Reversal"] = {
            "accuracy": metrics.get("cv_accuracy", 0),
            "std": metrics.get("cv_std", 0),
            "status": "OK",
        }
        console.print(f"[green]Reversal Model: {metrics.get('cv_accuracy', 0):.4f}[/green]")
    except Exception as exc:
        results["Reversal"] = {"accuracy": 0, "std": 0, "status": f"FAIL: {exc}"}
        console.print(f"[red]Reversal Model FAILED: {exc}[/red]")

    # ── 4. Range Model ─────────────────────────────────────
    console.print("\n[bold cyan]═══ 4/4 RANGE MODEL ═══[/bold cyan]")
    try:
        from train_range import train_range
        metrics = train_range()
        results["Range"] = {
            "accuracy": metrics.get("cv_accuracy", 0),
            "std": metrics.get("cv_std", 0),
            "status": "OK",
        }
        console.print(f"[green]Range Model: {metrics.get('cv_accuracy', 0):.4f}[/green]")
    except Exception as exc:
        results["Range"] = {"accuracy": 0, "std": 0, "status": f"FAIL: {exc}"}
        console.print(f"[red]Range Model FAILED: {exc}[/red]")

    # ── Summary ────────────────────────────────────────────
    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    table = Table(title=f"Training Complete ({mins}m {secs}s)")
    table.add_column("Model", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Status")

    for name, r in results.items():
        status_style = "green" if r["status"] == "OK" else "red"
        table.add_row(
            name,
            f"{r['accuracy']:.4f}" if r["accuracy"] > 0 else "—",
            f"±{r['std']:.4f}" if r["std"] > 0 else "—",
            f"[{status_style}]{r['status']}[/{status_style}]",
        )

    console.print()
    console.print(table)

    ok_count = sum(1 for r in results.values() if r["status"] == "OK")
    console.print(f"\n[bold]{ok_count}/4 models trained successfully[/bold]")

    if ok_count == 4:
        console.print("[green]All models ready! Start bot with: systemctl start aibot[/green]")
    else:
        console.print("[yellow]Some models failed — check logs above[/yellow]")

    return results


if __name__ == "__main__":
    train_all()
