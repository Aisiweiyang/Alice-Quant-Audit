#!/usr/bin/env python3
"""Train-period parameter selection followed by untouched out-of-sample evaluation."""

import argparse
from dataclasses import replace
import json
import os
import pickle
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from engine.factor_backtest import (
    load_market,
    prepare_feature_store,
    prepare_price_panels,
    run_factor_backtest,
)
from strategy.factor_config import FactorStrategyConfig
from strategy.alpha158_subset import FEATURE_SET_NAME, feature_manifest
from research.factor_diagnostics import write_factor_diagnostics
from research.ridge_model import build_walk_forward_ridge_scores, write_ridge_model


WEIGHT_STYLES = {
    "balanced": {
        "momentum_rank": 0.30,
        "trend_quality_rank": 0.20,
        "value_rank": 0.20,
        "low_volatility_rank": 0.15,
        "volume_confirmation_rank": 0.10,
        "short_reversal_rank": 0.05,
        "alpha158_pv_rank": 0.00,
        "alpha158_ridge_rank": 0.00,
    },
    "momentum_trend": {
        "momentum_rank": 0.38,
        "trend_quality_rank": 0.24,
        "value_rank": 0.12,
        "low_volatility_rank": 0.11,
        "volume_confirmation_rank": 0.10,
        "short_reversal_rank": 0.05,
        "alpha158_pv_rank": 0.00,
        "alpha158_ridge_rank": 0.00,
    },
    "defensive_value": {
        "momentum_rank": 0.23,
        "trend_quality_rank": 0.15,
        "value_rank": 0.27,
        "low_volatility_rank": 0.25,
        "volume_confirmation_rank": 0.07,
        "short_reversal_rank": 0.03,
        "alpha158_pv_rank": 0.00,
        "alpha158_ridge_rank": 0.00,
    },
    "trend_quality": {
        "momentum_rank": 0.31,
        "trend_quality_rank": 0.29,
        "value_rank": 0.15,
        "low_volatility_rank": 0.12,
        "volume_confirmation_rank": 0.10,
        "short_reversal_rank": 0.03,
        "alpha158_pv_rank": 0.00,
        "alpha158_ridge_rank": 0.00,
    },
    "alpha158_pv": {
        "momentum_rank": 0.25,
        "trend_quality_rank": 0.18,
        "value_rank": 0.15,
        "low_volatility_rank": 0.12,
        "volume_confirmation_rank": 0.08,
        "short_reversal_rank": 0.02,
        "alpha158_pv_rank": 0.20,
        "alpha158_ridge_rank": 0.00,
    },
    "alpha158_ridge": {
        "momentum_rank": 0.20,
        "trend_quality_rank": 0.15,
        "value_rank": 0.12,
        "low_volatility_rank": 0.10,
        "volume_confirmation_rank": 0.05,
        "short_reversal_rank": 0.03,
        "alpha158_pv_rank": 0.00,
        "alpha158_ridge_rank": 0.35,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="截面因子策略训练/样本外研究")
    parser.add_argument("--cache", required=True, help="受信任的本地pickle行情缓存")
    parser.add_argument("--output-dir", required=True, help="研究结果目录")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-06")
    parser.add_argument("--baseline-metrics", help="原策略全区间metrics.json")
    parser.add_argument("--feature-cache", help="可选：复用已经生成的特征缓存")
    return parser.parse_args()


def objective(metrics):
    if metrics.get("total_trades", 0) < 15:
        return -100.0
    return (
        float(metrics["sharpe_ratio"])
        + 0.03 * float(metrics["cagr_pct"])
        + 0.04 * float(metrics["max_drawdown_pct"])
    )


def candidate_configs():
    base = FactorStrategyConfig()
    risk_rules = (
        {"stop_loss_pct": 0.07, "take_profit_pct": 0.18, "exit_rank": 60},
        {"stop_loss_pct": 0.06, "take_profit_pct": None, "exit_rank": 40},
    )
    for style, weights in WEIGHT_STYLES.items():
        for market_ma in (20, 60):
            for risk_index, risk_rule in enumerate(risk_rules, start=1):
                config = replace(
                    base,
                    factor_weights=weights.copy(),
                    market_ma_period=market_ma,
                    **risk_rule,
                )
                yield f"{style}_ma{market_ma}_risk{risk_index}", config


def save_run(output_dir, label, portfolio, metrics):
    target = os.path.join(output_dir, label)
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
    pd.DataFrame(portfolio.equity_curve).to_csv(
        os.path.join(target, "equity.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(portfolio.trades).to_csv(
        os.path.join(target, "trades.csv"), index=False, encoding="utf-8-sig"
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_market(args.cache)
    base = FactorStrategyConfig()

    print("[1/4] 构建逐调仓日因子截面...")
    with open(
        os.path.join(args.output_dir, "alpha158_subset_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(feature_manifest(), handle, ensure_ascii=False, indent=2)
    feature_cache = args.feature_cache or os.path.join(
        args.output_dir, f"feature_store_{FEATURE_SET_NAME}.pkl"
    )
    if os.path.isfile(feature_cache):
        with open(feature_cache, "rb") as handle:
            features = pickle.load(handle)
        print(f"    复用因子缓存: {feature_cache}")
    else:
        features = prepare_feature_store(data, args.start, args.end, base, verbose=True)
        with open(feature_cache, "wb") as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    因子缓存已保存: {feature_cache}")
    prices = prepare_price_panels(data)

    print("    生成训练期 Alpha158 因子诊断...")
    write_factor_diagnostics(features, data, args.train_end, args.output_dir, forward_days=5)
    print("    训练滚动 Ridge，并冻结样本外模型...")
    features, ridge_model = build_walk_forward_ridge_scores(features, data, args.train_end)
    write_ridge_model(ridge_model, os.path.join(args.output_dir, "alpha158_ridge_model.json"))
    ridge_feature_cache = os.path.join(
        args.output_dir, f"feature_store_{FEATURE_SET_NAME}_ridge.pkl"
    )
    with open(ridge_feature_cache, "wb") as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("[2/4] 仅在训练区间选择参数...")
    rows = []
    configs = {}
    for label, config in candidate_configs():
        _, metrics = run_factor_backtest(
            data,
            args.start,
            args.train_end,
            config=config,
            feature_store=features,
            price_panels=prices,
            verbose=False,
        )
        score = objective(metrics)
        configs[label] = config
        rows.append({"label": label, "objective": score, **metrics})
        pd.DataFrame(rows).to_csv(
            os.path.join(args.output_dir, "training_sweep_partial.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"    {label:<38} | CAGR {metrics['cagr_pct']:>6.2f}% | "
            f"DD {metrics['max_drawdown_pct']:>6.2f}% | Sharpe {metrics['sharpe_ratio']:>4.2f}"
        )

    sweep = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    sweep.to_csv(
        os.path.join(args.output_dir, "training_sweep.csv"), index=False, encoding="utf-8-sig"
    )
    selected_label = str(sweep.iloc[0]["label"])
    selected = configs[selected_label]
    with open(
        os.path.join(args.output_dir, "selected_config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"label": selected_label, **selected.to_dict()},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[3/4] 冻结参数并运行样本外: {selected_label}")
    test_portfolio, test_metrics = run_factor_backtest(
        data,
        args.test_start,
        args.end,
        config=selected,
        feature_store=features,
        price_panels=prices,
        verbose=False,
    )
    save_run(args.output_dir, "out_of_sample", test_portfolio, test_metrics)

    print("[4/4] 运行全区间最终比较...")
    full_portfolio, full_metrics = run_factor_backtest(
        data,
        args.start,
        args.end,
        config=selected,
        feature_store=features,
        price_panels=prices,
        verbose=False,
    )
    save_run(args.output_dir, "full_period", full_portfolio, full_metrics)

    ridge_result = None
    ridge_rows = sweep[sweep["label"].str.startswith("alpha158_ridge_")]
    if not ridge_rows.empty:
        ridge_label = str(ridge_rows.iloc[0]["label"])
        ridge_config = configs[ridge_label]
        ridge_test_portfolio, ridge_test_metrics = run_factor_backtest(
            data,
            args.test_start,
            args.end,
            config=ridge_config,
            feature_store=features,
            price_panels=prices,
            verbose=False,
        )
        ridge_full_portfolio, ridge_full_metrics = run_factor_backtest(
            data,
            args.start,
            args.end,
            config=ridge_config,
            feature_store=features,
            price_panels=prices,
            verbose=False,
        )
        save_run(
            args.output_dir,
            os.path.join("experiments", ridge_label, "out_of_sample"),
            ridge_test_portfolio,
            ridge_test_metrics,
        )
        save_run(
            args.output_dir,
            os.path.join("experiments", ridge_label, "full_period"),
            ridge_full_portfolio,
            ridge_full_metrics,
        )
        ridge_result = {
            "label": ridge_label,
            "selection_rule": "仅在2025训练期内选择Ridge候选；2026只评估，不参与选择",
            "training": json.loads(ridge_rows.iloc[0].to_json(force_ascii=False)),
            "out_of_sample": ridge_test_metrics,
            "full_period": ridge_full_metrics,
        }

    comparison = {
        "selection_rule": "参数仅由2025训练区间目标函数选出；2026区间未参与选择",
        "selected_label": selected_label,
        "training": json.loads(sweep.iloc[0].to_json(force_ascii=False)),
        "out_of_sample": test_metrics,
        "full_period": full_metrics,
    }
    if args.baseline_metrics:
        with open(args.baseline_metrics, "r", encoding="utf-8") as handle:
            comparison["legacy_full_period"] = json.load(handle)
    if ridge_result is not None:
        comparison["alpha158_ridge_experiment"] = ridge_result
    with open(os.path.join(args.output_dir, "comparison.json"), "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
