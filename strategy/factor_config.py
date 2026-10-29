"""Configuration for the cross-sectional factor strategy."""

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class FactorStrategyConfig:
    initial_capital: float = 100_000
    benchmark_index: str = "sh.000300"
    feature_set: str = "alpha158_price_volume_subset_v1"

    liquid_universe_size: int = 500
    minimum_history: int = 65
    require_price_above_ma60: bool = True
    minimum_score: float = 55.0

    max_positions: int = 8
    max_position_pct: float = 0.12
    rebalance_frequency: str = "weekly"
    rebalance_day: str = "monday"
    exit_rank: int = 60
    minimum_holding_days: int = 10

    use_market_timing: bool = True
    market_ma_period: int = 20
    stop_loss_pct: Optional[float] = 0.07
    take_profit_pct: Optional[float] = 0.18

    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    stamp_tax_rate_before_20230828: float = 0.001
    stamp_tax_rate_after_20230828: float = 0.0005
    slippage: float = 0.001
    min_commission: float = 5.0

    factor_weights: dict = field(
        default_factory=lambda: {
            "momentum_rank": 0.30,
            "trend_quality_rank": 0.20,
            "value_rank": 0.20,
            "low_volatility_rank": 0.15,
            "volume_confirmation_rank": 0.10,
            "short_reversal_rank": 0.05,
            "alpha158_pv_rank": 0.00,
            "alpha158_ridge_rank": 0.00,
        }
    )

    def to_dict(self):
        return asdict(self)
