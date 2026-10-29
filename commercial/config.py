"""商业分析白名单参数：只暴露安全、可解释的回测设置。

不接受任意Python代码、不开放任意参数注入；所有取值都经过范围校验。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from strategy.config import StrategyConfig

BENCHMARK_CODE = "sh.000300"

STRATEGY_PRESETS = {
    # 内置策略：经典趋势评分（Alice v4 生产基线口径）
    "classic_trend": "Alice经典趋势评分策略（5日动量+量价+平稳度等加权）",
    # 内置策略：截面Alpha158排名（动量/趋势质量/价值/低波/量能确认分位排名加权，
    # 主板池、最低65日历史、buffered-rank调仓，见 engine/factor_backtest.py）
    "cross_sectional_alpha158": (
        "截面Alpha158排名策略（多因子分位排名加权，主板池，最低65日历史）"
    ),
}

# 截面策略专属白名单参数：键→(下界, 上界, 类型)；不在此列的参数一律拒绝，
# 不开放任意参数注入。
CROSS_SECTIONAL_PARAM_WHITELIST = {
    "minimum_score": (0.0, 100.0, float),
    "exit_rank": (1, 500, int),
    "minimum_holding_days": (0, 60, int),
    "market_ma_period": (5, 120, int),
    "liquid_universe_size": (10, 2000, int),
    "minimum_history": (65, 250, int),
}


class ParameterError(ValueError):
    """白名单参数校验失败。"""


def _bounded(name, value, low, high, cast=float):
    try:
        number = cast(value)
    except (TypeError, ValueError):
        raise ParameterError(f"{name} 必须是数字") from None
    if not (low <= number <= high):
        raise ParameterError(f"{name} 必须在 [{low}, {high}] 内")
    return number


@dataclass
class CommercialRequest:
    """一次商业分析的全部输入。"""

    cache: str
    start: str
    end: str
    strategy: str = "classic_trend"
    candidates: str | None = None
    initial_capital: float = 100_000.0
    max_positions: int = 8
    max_position_pct: float = 0.12
    rebalance_freq: str = "weekly"
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    slippage: float = 0.001
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    use_market_timing: bool = True
    run_stress: bool = True
    run_factor_diagnostics: bool = True
    run_rolling_oos: bool = True
    run_segment_consistency: bool = True
    rolling_windows: int = 3
    rolling_anchored: bool = False
    max_scenarios: int = 8
    trials: int = 1
    output_root: str = os.path.join(PROJECT_DIR, "output", "commercial_reports")
    allow_incomplete_data: bool = False
    brand_name: str = ""
    brand_footer: str = ""
    quiet: bool = False
    strategy_params: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    def validate(self) -> None:
        import pandas as pd

        if self.strategy not in STRATEGY_PRESETS:
            raise ParameterError(
                f"不支持的策略预设: {self.strategy}；可选: {', '.join(STRATEGY_PRESETS)}"
            )
        try:
            start = pd.Timestamp(self.start)
            end = pd.Timestamp(self.end)
        except (ValueError, TypeError) as exc:
            raise ParameterError(f"开始/结束日期无法解析: {exc}") from None
        if pd.isna(start) or pd.isna(end):
            raise ParameterError("开始/结束日期无法解析")
        if start > end:
            raise ParameterError("开始日期不能晚于结束日期")
        self.initial_capital = _bounded(
            "initial_capital", self.initial_capital, 10_000, 100_000_000
        )
        self.max_positions = int(_bounded("max_positions", self.max_positions, 1, 50, int))
        self.max_position_pct = _bounded("max_position_pct", self.max_position_pct, 0.01, 1.0)
        if self.rebalance_freq not in {"daily", "weekly", "monthly"}:
            raise ParameterError("rebalance_freq 只支持 daily/weekly/monthly")
        self.commission_rate = _bounded("commission_rate", self.commission_rate, 0.0, 0.02)
        self.stamp_tax_rate = _bounded("stamp_tax_rate", self.stamp_tax_rate, 0.0, 0.02)
        self.slippage = _bounded("slippage", self.slippage, 0.0, 0.05)
        self.stop_loss_pct = _bounded("stop_loss_pct", self.stop_loss_pct, 0.01, 0.5)
        self.take_profit_pct = _bounded("take_profit_pct", self.take_profit_pct, 0.01, 1.0)
        self.max_scenarios = int(_bounded("max_scenarios", self.max_scenarios, 1, 8, int))
        # 本轮尝试的方案数：Deflated Sharpe的多重检验扣除用（1=退化为PSR）。
        self.trials = int(_bounded("trials", self.trials, 1, 1000, int))
        self.brand_name = str(self.brand_name or "")[:60]
        self.brand_footer = str(self.brand_footer or "")[:120]
        if self.strategy_params:
            if self.strategy != "cross_sectional_alpha158":
                raise ParameterError("strategy_params 仅支持 cross_sectional_alpha158 策略")
            validated = {}
            for key, value in self.strategy_params.items():
                if key not in CROSS_SECTIONAL_PARAM_WHITELIST:
                    raise ParameterError(
                        f"不支持的策略参数: {key}；"
                        f"可选: {', '.join(CROSS_SECTIONAL_PARAM_WHITELIST)}"
                    )
                low, high, cast = CROSS_SECTIONAL_PARAM_WHITELIST[key]
                validated[key] = _bounded(key, value, low, high, cast)
            self.strategy_params = validated

    def params_digest(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_engine_config(request: CommercialRequest, **overrides):
    """构造回测引擎可用的配置视图，不修改全局 StrategyConfig。"""
    changes = {
        "initial_capital": request.initial_capital,
        "max_positions": request.max_positions,
        "max_position_pct": request.max_position_pct,
        "rebalance_freq": request.rebalance_freq,
        "commission_rate": request.commission_rate,
        "min_commission": 5.0,
        "stamp_tax_rate": request.stamp_tax_rate,
        "stamp_tax_rate_before_20230828": max(request.stamp_tax_rate, 0.001)
        if request.extras.get("historical_stamp_tax", True)
        else request.stamp_tax_rate,
        "stamp_tax_rate_after_20230828": request.stamp_tax_rate,
        "slippage": request.slippage,
        "stop_loss_pct": request.stop_loss_pct,
        "take_profit_pct": request.take_profit_pct,
        "use_market_timing": request.use_market_timing,
    }
    changes.update(overrides)
    return type("CommercialConfigView", (StrategyConfig,), changes)


def build_factor_strategy_config(request: CommercialRequest):
    """把商业请求映射为截面Alpha158策略配置（strategy.factor_config）。

    共享参数（资金/持仓/成本/择时/止盈止损）沿用商业请求；截面专属参数
    （minimum_score/exit_rank/minimum_holding_days等）来自白名单strategy_params，
    未提供时使用FactorStrategyConfig默认值。返回实例，不改全局配置。
    """
    from strategy.factor_config import FactorStrategyConfig

    stamp_before = (
        max(request.stamp_tax_rate, 0.001)
        if request.extras.get("historical_stamp_tax", True)
        else request.stamp_tax_rate
    )
    config = FactorStrategyConfig(
        initial_capital=request.initial_capital,
        max_positions=request.max_positions,
        max_position_pct=request.max_position_pct,
        benchmark_index=BENCHMARK_CODE,
        rebalance_frequency=request.rebalance_freq,
        commission_rate=request.commission_rate,
        slippage=request.slippage,
        min_commission=5.0,
        stamp_tax_rate=request.stamp_tax_rate,
        stamp_tax_rate_before_20230828=stamp_before,
        stamp_tax_rate_after_20230828=request.stamp_tax_rate,
        stop_loss_pct=request.stop_loss_pct,
        take_profit_pct=request.take_profit_pct,
        use_market_timing=request.use_market_timing,
    )
    for key, value in (request.strategy_params or {}).items():
        setattr(config, key, value)
    if request.use_market_timing and "market_ma_period" not in (request.strategy_params or {}):
        # 截面策略的大盘均线周期沿用经典策略的ma_period口径（默认20）。
        config.market_ma_period = StrategyConfig.ma_period
    return config


def request_from_args(args) -> CommercialRequest:
    request = CommercialRequest(
        cache=args.cache,
        start=args.start,
        end=args.end,
        strategy=args.strategy,
        candidates=args.candidates,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        max_position_pct=args.max_position_pct,
        rebalance_freq=args.rebalance_freq,
        commission_rate=args.commission_rate,
        stamp_tax_rate=args.stamp_tax_rate,
        slippage=args.slippage,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        use_market_timing=not args.no_market_timing,
        run_stress=not args.no_stress,
        run_segment_consistency=not args.no_segment,
        run_factor_diagnostics=not args.no_factor_diagnostics,
        run_rolling_oos=not args.no_rolling_oos,
        rolling_windows=args.rolling_windows,
        rolling_anchored=bool(getattr(args, "rolling_anchored", False)),
        max_scenarios=args.max_scenarios,
        trials=getattr(args, "trials", 1),
        output_root=args.output_root,
        allow_incomplete_data=args.allow_incomplete_data,
        quiet=args.quiet,
    )
    strategy_params = {}
    for item in getattr(args, "strategy_param", None) or []:
        key, sep, value = str(item).partition("=")
        if not key.strip() or not sep:
            raise ParameterError(f"--strategy-param 格式应为 KEY=VALUE: {item}")
        strategy_params[key.strip()] = value.strip()
    request.strategy_params = strategy_params
    request.validate()
    return request
