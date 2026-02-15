"""
Position sizing and daily loss-limit tracking.
"""


class RiskManager:
    """Basic risk management for intraday futures trading."""

    def __init__(self, config):
        self.config = config
        self.daily_pnl: float = 0.0
        self.trades_today: int = 0

    def calculate_position_size(
        self,
        balance: float,
        entry: float,
        stop_loss: float,
        leverage: int,
    ) -> float:
        """
        Risk-based position sizing.
        Returns the position size in base asset units.
        """
        risk_amount = balance * self.config.RISK_PER_TRADE
        sl_distance = abs(entry - stop_loss)

        if sl_distance == 0 or entry == 0:
            return 0.0

        size_by_risk = risk_amount / sl_distance
        max_size = balance * leverage / entry

        return min(size_by_risk, max_size)

    def check_daily_limit(self, balance: float) -> bool:
        """Return True if we are still within the daily loss limit."""
        max_loss = balance * self.config.MAX_DAILY_LOSS
        return self.daily_pnl > -max_loss

    def update_pnl(self, pnl: float):
        self.daily_pnl += pnl
        self.trades_today += 1

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.trades_today = 0
