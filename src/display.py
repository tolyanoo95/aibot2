"""
Rich-powered console display for the scanner output.
"""

from datetime import datetime

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class Display:
    """Pretty-print scanner results to the terminal."""

    # ── main table ───────────────────────────────────────────

    @staticmethod
    def show_signals(signals: list, scan_time: float = 0, interval: int = 300):
        console.clear()

        # header
        console.print(
            Panel(
                "[bold cyan]AI Crypto Futures Scanner[/bold cyan]\n"
                f"[dim]{datetime.now():%Y-%m-%d %H:%M:%S} | "
                f"Scan time: {scan_time:.1f}s | "
                f"Pairs: {len(signals)}[/dim]",
                box=box.DOUBLE,
            )
        )

        table = Table(
            title="Trading Signals",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Direction", justify="center", width=12)
        table.add_column("Confidence", justify="center", width=12)
        table.add_column("Status", justify="center", width=16)
        table.add_column("Entry", justify="right", width=14)
        table.add_column("SL", justify="right", width=14)
        table.add_column("TP", justify="right", width=14)
        table.add_column("Lev", justify="center", width=6)
        table.add_column("ML", justify="center", width=8)
        table.add_column("LLM", justify="center", width=8)
        table.add_column("Regime", justify="center", width=10)

        sorted_sigs = sorted(signals, key=lambda s: s.confidence, reverse=True)

        for s in sorted_sigs:
            table.add_row(
                s.symbol,
                _dir_fmt(s.direction),
                _conf_fmt(s.confidence),
                _status_fmt(s),
                _price(s.entry_price),
                _price(s.stop_loss),
                _price(s.take_profit),
                f"x{s.leverage}" if s.direction != "NEUTRAL" else "—",
                _ml_fmt(s.ml_signal),
                _llm_fmt(s.llm_direction),
                s.market_regime or "—",
            )

        console.print(table)

        # best opportunity panel
        active = [s for s in sorted_sigs if s.direction != "NEUTRAL"]
        if active:
            best = active[0]
            rr = (
                abs(best.take_profit - best.entry_price)
                / abs(best.stop_loss - best.entry_price)
                if best.stop_loss != best.entry_price
                else 0
            )

            # tracking info
            age = getattr(best, "age_bars", 0)
            if getattr(best, "is_new", True):
                if age == 0:
                    status_line = "[bold green]FRESH SIGNAL[/bold green]  — just appeared"
                elif age <= 2:
                    status_line = (
                        f"[green]NEW SIGNAL[/green]  — "
                        f"ML active for ~{age * 5} min"
                    )
                else:
                    status_line = (
                        f"[yellow]LATE ENTRY[/yellow]  — "
                        f"signal active for ~{age * 5} min, move may be partially done"
                    )
            else:
                first = getattr(best, "first_seen", None)
                since = first.strftime("%H:%M") if first else "?"
                status_line = (
                    f"[yellow]Active since {since}[/yellow]  |  "
                    f"Original Entry: {_price(getattr(best, 'original_entry', 0))}"
                )

            # original levels if signal is continuing
            orig_entry = getattr(best, "original_entry", 0)
            orig_sl = getattr(best, "original_sl", 0)
            orig_tp = getattr(best, "original_tp", 0)

            orig_line = ""
            if not getattr(best, "is_new", True) and orig_entry:
                orig_line = (
                    f"\n[dim]Original levels: "
                    f"Entry {_price(orig_entry)}  →  "
                    f"SL {_price(orig_sl)}  |  "
                    f"TP {_price(orig_tp)}[/dim]"
                )

            console.print(
                Panel(
                    f"{status_line}\n"
                    f"[bold]{best.symbol}  {best.direction}[/bold]\n"
                    f"Confidence: {best.confidence * 100:.1f}%  |  "
                    f"Leverage: x{best.leverage}  |  "
                    f"R:R = 1:{rr:.1f}\n"
                    f"Entry {_price(best.entry_price)}  →  "
                    f"SL {_price(best.stop_loss)}  |  "
                    f"TP {_price(best.take_profit)}"
                    f"{orig_line}\n"
                    f"\n[dim italic]{best.llm_reasoning}[/dim italic]",
                    title="Top Opportunity",
                    border_style="green" if best.direction == "LONG" else "red",
                )
            )
        else:
            console.print(
                Panel(
                    "[dim]No active signals — market conditions unclear.[/dim]",
                    title="Status",
                    border_style="dim",
                )
            )

        minutes = interval // 60
        console.print(
            f"\n[dim]Next scan in ~{minutes} min … Press Ctrl+C to stop.[/dim]"
        )

    # ── utility prints ───────────────────────────────────────

    @staticmethod
    def show_info(msg: str):
        console.print(f"[cyan]INFO:[/cyan] {msg}")

    @staticmethod
    def show_error(msg: str):
        console.print(f"[bold red]ERROR:[/bold red] {msg}")


# ── formatting helpers ───────────────────────────────────────

def _dir_fmt(d: str) -> str:
    if d == "LONG":
        return "[bold green]▲ LONG[/bold green]"
    if d == "SHORT":
        return "[bold red]▼ SHORT[/bold red]"
    return "[dim]— NEUTRAL[/dim]"


def _status_fmt(s) -> str:
    if s.direction == "NEUTRAL":
        return "[dim]—[/dim]"
    age = getattr(s, "age_bars", 0)
    is_new = getattr(s, "is_new", True)
    if not is_new:
        first = getattr(s, "first_seen", None)
        since = first.strftime("%H:%M") if first else "?"
        return f"[dim]ACTIVE {since}[/dim]"
    if age == 0:
        return "[bold green]FRESH SIGNAL[/bold green]"
    if age <= 2:
        return f"[green]NEW ~{age * 5}m[/green]"
    return f"[bold yellow]LATE ~{age * 5}m[/bold yellow]"


def _conf_fmt(c: float) -> str:
    pct = f"{c * 100:.1f}%"
    if c >= 0.7:
        return f"[bold green]{pct}[/bold green]"
    if c >= 0.4:
        return f"[yellow]{pct}[/yellow]"
    return f"[dim]{pct}[/dim]"


def _ml_fmt(sig: int) -> str:
    return {1: "[green]BUY[/green]", -1: "[red]SELL[/red]"}.get(
        sig, "[dim]HOLD[/dim]"
    )


def _llm_fmt(d: str) -> str:
    return {
        "LONG": "[green]LONG[/green]",
        "SHORT": "[red]SHORT[/red]",
    }.get(d, "[dim]NEUT[/dim]")


def _risk_fmt(r: str) -> str:
    return {
        "LOW": "[green]LOW[/green]",
        "MEDIUM": "[yellow]MED[/yellow]",
        "HIGH": "[red]HIGH[/red]",
    }.get(r, "[dim]—[/dim]")


def _price(p: float) -> str:
    if not p:
        return "—"
    decimals = 2 if p > 10 else 4 if p > 0.01 else 6
    return f"{p:.{decimals}f}"
