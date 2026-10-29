"""商业交付报告生成：独立目录、唯一report_id、全量哈希清单。

每次分析生成 output/commercial_reports/<report_id>/，旧报告不会被覆盖。
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_DIR, "output", ".matplotlib"))

import numpy as np
import pandas as pd

from analysis.backtest_report import equity_svg
from commercial.reliability_score import DIMENSION_SPECS, SCORE_DECLARATION
from run_metadata import build_run_id, prepare_run_directory

DISCLAIMER = "仅供研究，不构成投资建议。不承诺收益，不连接券商，不替代持牌投资顾问。"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def export_report_zip(report_root: str, report_id: str, zip_path: str | None = None) -> str:
    """把指定报告目录打包为zip（zip内顶层为report_id目录）；返回zip路径。"""
    import zipfile

    source = os.path.join(os.path.abspath(report_root), report_id)
    if not os.path.isdir(source):
        raise FileNotFoundError(f"报告目录不存在: {source}")
    zip_path = zip_path or os.path.join(os.path.abspath(report_root), f"{report_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for base, _dirs, files in os.walk(source):
            for name in files:
                absolute = os.path.join(base, name)
                relative = os.path.relpath(absolute, os.path.abspath(report_root))
                bundle.write(absolute, relative)
    return zip_path


def new_report_id() -> str:
    """唯一report_id：版本_时间戳_毫秒，配合目录防覆盖校验。"""
    return build_run_id()


def create_report_directory(output_root: str, report_id: str | None = None) -> tuple[str, str]:
    report_id = report_id or new_report_id()
    report_dir = prepare_run_directory(output_root, report_id)
    return report_dir, report_id


def _git_snapshot() -> dict:
    def git(*args):
        try:
            result = subprocess.run(
                ["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )


def _fmt(value, suffix="", dash="—"):
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value}{suffix}"


def _pct(value):
    return _fmt(value, "%")


# 全指标快照表（quantstats tearsheet风格的"一屏两列"）：统一从metrics字典取值，
# 与第4/5节明细表同源，缺哪个字段显示"—"。
SECTION4_SNAPSHOT_FIELDS = (
    ("区间收益", "total_return_pct", "pct"),
    ("年化收益", "cagr_pct", "pct"),
    ("最大回撤", "max_drawdown_pct", "pct"),
    ("年化波动", "annual_volatility_pct", "pct"),
    ("Sharpe", "sharpe_ratio", "num"),
    ("Sortino", "sortino_ratio", "num"),
    ("Calmar", "calmar_ratio", "num"),
    ("信息比率", "information_ratio", "num"),
    ("Alpha(年化)", "alpha_annualized_pct", "pct"),
    ("Beta", "beta", "num"),
    ("尾部比率", "tail_ratio", "num"),
    ("回收因子", "recovery_factor", "num"),
    ("VaR95(日)", "var_95_pct", "pct"),
    ("CVaR95(日)", "cvar_95_pct", "pct"),
    ("偏度", "skewness", "num"),
    ("超额峰度", "excess_kurtosis", "num"),
    ("Ulcer Index", "ulcer_index_pct", "num"),
    ("Martin比率", "martin_ratio", "num"),
    ("Omega比率", "omega_ratio", "num"),
    ("交易MC回撤p50", "tmc_max_drawdown_pct_p50", "pct"),
    ("交易MC回撤p95", "tmc_max_drawdown_pct_p95", "pct"),
    ("交易MC终值p5", "tmc_terminal_return_p5_pct", "pct"),
    ("最好单笔", "best_trade_pct", "pct"),
    ("最差单笔", "worst_trade_pct", "pct"),
    ("单笔期望值", "expectancy_pct", "pct"),
    ("SQN", "sqn", "num"),
    ("平均持仓(天)", "avg_trade_duration_days", "num"),
    ("最长持仓(天)", "max_trade_duration_days", "num"),
    ("MC回撤p50", "mc_max_drawdown_pct_p50", "pct"),
    ("MC回撤p95", "mc_max_drawdown_pct_p95", "pct"),
    ("上行捕获", "up_capture_pct", "pct"),
    ("下行捕获", "down_capture_pct", "pct"),
    ("Deflated Sharpe", "deflated_sharpe", "num"),
    ("试验次数N", "deflated_sharpe_trials", "num"),
    ("Sharpe 95%CI下界", "sharpe_ci_low", "num"),
    ("Sharpe 95%CI上界", "sharpe_ci_high", "num"),
    ("胜率", "win_rate_pct", "pct"),
    ("盈亏比", "profit_factor", "num"),
    ("最大连续亏损(天)", "max_consecutive_losses", "num"),
    ("最大连续盈利(天)", "max_consecutive_wins", "num"),
    ("完整交易(笔)", "total_trades", "num"),
    ("未售出持仓", "unsold_final_positions", "num"),
)


def render_snapshot_table(metrics: dict) -> str:
    cells = []
    for label, key, kind in SECTION4_SNAPSHOT_FIELDS:
        value = metrics.get(key)
        text = "—" if value is None else (_pct(value) if kind == "pct" else _fmt(value))
        cells.append(f"<th>{html.escape(label)}</th><td>{text}</td>")
    if len(cells) % 2:
        cells.append("<th></th><td></td>")
    rows = "".join(f"<tr>{cells[i]}{cells[i + 1]}</tr>" for i in range(0, len(cells), 2))
    return (
        "<h3 style='margin-top:14px'>全指标快照（一屏两列）</h3>"
        f"<table><tbody>{rows}</tbody></table>"
        "<p class='muted'>与第4/5节同源口径；基准类指标在基准缺失时如实留空。</p>"
    )


def _grade_badge(score) -> str:
    if score is None:
        return '<span class="badge badge-fail">无法评分</span>'
    tone = "badge-ok" if score >= 70 else "badge-warn" if score >= 50 else "badge-fail"
    return f'<span class="badge {tone}">{score}分</span>'


def _grouped_bar_chart(
    plt, frame: pd.DataFrame, label_column: str, value_columns: list, title: str, ylabel: str, path
) -> None:
    """分组柱状图：每个标签一组、每组len(value_columns)根柱。"""
    labels = [str(v) for v in frame[label_column]]
    positions = np.arange(len(labels))
    width = 0.8 / max(len(value_columns), 1)
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(labels)), 4.2))
    for index, column in enumerate(value_columns):
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy()
        ax.bar(positions + index * width - 0.4 + width / 2, values, width, label=str(column))
    ax.set_title(title)
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _heatmap_chart(
    plt,
    matrix: np.ndarray,
    row_labels: list,
    column_labels: list,
    title: str,
    colorbar_label: str,
    path,
) -> None:
    """热力图：NaN格显示为浅灰；红绿色标对称于0。"""
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#e2e8f0")
    finite = masked.compressed()
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    scale = max(scale, 1e-6)
    fig, ax = plt.subplots(
        figsize=(max(9, 0.42 * len(column_labels) + 3), max(4, 0.36 * len(row_labels) + 1.5))
    )
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-scale, vmax=scale)
    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels([str(c) for c in column_labels], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([str(r) for r in row_labels], fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _audit_equity_charts(plt, equity, charts_dir) -> list[str]:
    """从净值序列派生的四张审计图：回撤水下图/日收益分布/滚动Sharpe/资金暴露。

    全部只依赖equity列（date/total/holdings_value），不改变任何指标口径；
    单张失败不影响其余图表。
    """
    written = []
    try:
        frame = equity.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date")
        totals = pd.to_numeric(frame["total"], errors="coerce")
        returns = totals.pct_change().dropna()
        dates = frame["date"].iloc[1:].reset_index(drop=True)

        # 1) 回撤水下图：距历史峰值的回撤百分比。
        peak = totals.cummax()
        drawdown = (totals - peak) / peak * 100
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.fill_between(frame["date"], drawdown, 0, color="#dc2626", alpha=0.55)
        ax.set_title("回撤水下图：净值相对历史峰值的回撤")
        ax.set_ylabel("回撤 (%)")
        fig.tight_layout()
        fig.savefig(charts_dir / "drawdown.png", dpi=130)
        plt.close(fig)
        written.append("charts/drawdown.png")

        # 2) 日收益分布直方图与同均值方差正态曲线（观察肥尾/偏斜）。
        if len(returns) >= 20:
            values = returns.to_numpy(dtype=float) * 100
            mu, sigma = float(np.mean(values)), float(np.std(values, ddof=1))
            fig, ax = plt.subplots(figsize=(9, 3.6))
            ax.hist(
                values,
                bins=min(40, max(10, len(values) // 5)),
                density=True,
                color="#60a5fa",
                alpha=0.75,
            )
            grid = np.linspace(values.min(), values.max(), 200)
            if sigma > 0:
                normal = np.exp(-((grid - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
                ax.plot(grid, normal, color="#172033", linewidth=1.4, label="同均值方差正态曲线")
                ax.legend(fontsize=8)
            ax.set_title("日收益分布：实际直方图 vs 正态参考（肥尾一目了然）")
            ax.set_xlabel("日收益 (%)")
            ax.set_ylabel("密度")
            fig.tight_layout()
            fig.savefig(charts_dir / "return_distribution.png", dpi=130)
            plt.close(fig)
            written.append("charts/return_distribution.png")

        # 3) 滚动Sharpe（60个交易日）：稳定性随时间的变化。
        if len(returns) >= 61:
            rolling_sharpe = (
                returns.rolling(60).mean() / returns.rolling(60).std(ddof=1) * np.sqrt(252)
            )
            fig, ax = plt.subplots(figsize=(9, 3.6))
            ax.plot(dates, rolling_sharpe, color="#2563eb", linewidth=1.4)
            ax.axhline(0, color="#334155", linewidth=0.8)
            ax.set_title("滚动Sharpe（60个交易日窗口）")
            ax.set_ylabel("Sharpe")
            fig.tight_layout()
            fig.savefig(charts_dir / "rolling_sharpe.png", dpi=130)
            plt.close(fig)
            written.append("charts/rolling_sharpe.png")

        # 4) 资金暴露曲线：持仓市值/总净值，反映择时与空仓时段。
        totals_safe = totals.where(totals > 0)
        exposure = pd.to_numeric(frame["holdings_value"], errors="coerce") / totals_safe * 100
        if exposure.notna().any():
            fig, ax = plt.subplots(figsize=(9, 3.6))
            ax.fill_between(frame["date"], exposure, 0, color="#16a34a", alpha=0.45)
            ax.axhline(
                float(exposure.mean()),
                color="#334155",
                linewidth=1.0,
                linestyle="--",
                label=f"平均暴露 {exposure.mean():.0f}%",
            )
            ax.set_title("资金暴露：持仓市值占总净值比例")
            ax.set_ylabel("暴露 (%)")
            ax.set_ylim(0, 105)
            ax.legend(fontsize=8, loc="lower right")
            fig.tight_layout()
            fig.savefig(charts_dir / "exposure.png", dpi=130)
            plt.close(fig)
            written.append("charts/exposure.png")
    except Exception:
        pass  # 审计图失败静默跳过，不影响其余图表与数据交付
    return written


def write_charts(report_dir: str, analysis: dict, charts_dir: Path | None = None) -> list[str]:
    """输出charts/目录；matplotlib失败不阻断报告。"""
    charts_dir = charts_dir or (Path(report_dir) / "charts")
    charts_dir.mkdir(exist_ok=True)
    written = []
    base = analysis.get("base") or {}
    equity = base.get("equity")
    benchmark = base.get("benchmark")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        if equity is not None and not equity.empty:
            from analysis.visualize import (
                plot_equity_curve,
                plot_monthly_returns,
                plot_trade_distribution,
            )

            plot_equity_curve(equity, benchmark, str(charts_dir / "equity_curve.png"))
            written.append("charts/equity_curve.png")
            plot_monthly_returns(equity, str(charts_dir / "monthly_returns.png"))
            written.append("charts/monthly_returns.png")
            trades = base.get("trades")
            if trades is not None and not trades.empty:
                plot_trade_distribution(trades, str(charts_dir / "trade_distribution.png"))
                written.append("charts/trade_distribution.png")
            written.extend(_audit_equity_charts(plt, equity, charts_dir))

        stress_frame = analysis.get("stress_frame")
        if stress_frame is not None and not stress_frame.empty:
            ok = stress_frame[stress_frame.get("status", "ok") == "ok"]
            if not ok.empty:
                fig, ax = plt.subplots(figsize=(9, 4.2))
                colors = [
                    "#b91c1c" if v < 0 else "#15803d" for v in ok["return_delta_pct"].fillna(0)
                ]
                ax.bar(ok["label"], ok["return_delta_pct"].fillna(0), color=colors)
                ax.set_title("压力场景相对基准的收益变化（百分点）")
                ax.axhline(0, color="#334155", linewidth=0.8)
                ax.set_ylabel("收益变化 (pp)")
                plt.xticks(rotation=30, ha="right", fontsize=9)
                fig.tight_layout()
                fig.savefig(charts_dir / "scenario_deltas.png", dpi=130)
                plt.close(fig)
                written.append("charts/scenario_deltas.png")

        rolling = analysis.get("rolling_oos") or {}
        rolling_windows = [
            window for window in (rolling.get("windows") or []) if window.get("status") == "ok"
        ]
        if rolling_windows:
            labels = [str(window.get("window")) for window in rolling_windows]
            values = [float(window.get("total_return_pct") or 0.0) for window in rolling_windows]
            fig, ax = plt.subplots(figsize=(9, 4.2))
            ax.bar(labels, values, color=["#b91c1c" if v < 0 else "#15803d" for v in values])
            ax.set_title("滚动样本外：各不重叠窗口区间收益")
            ax.axhline(0, color="#334155", linewidth=0.8)
            ax.set_ylabel("区间收益 (%)")
            plt.xticks(rotation=20, ha="right", fontsize=9)
            fig.tight_layout()
            fig.savefig(charts_dir / "rolling_windows.png", dpi=130)
            plt.close(fig)
            written.append("charts/rolling_windows.png")

        # 因子诊断图表（alphalens口径的可视化）：IC衰减与分位数收益。
        diagnostics = analysis.get("factor_diagnostics")
        if diagnostics is not None and not diagnostics.empty:
            ok_factors = (
                diagnostics[diagnostics.get("status", "ok") == "ok"]
                if "status" in diagnostics.columns
                else diagnostics
            )
            horizon_columns = [
                c for c in ("ic_1d", "ic_5d", "ic_10d", "ic_20d") if c in ok_factors.columns
            ]
            if not ok_factors.empty and len(horizon_columns) >= 2:
                _grouped_bar_chart(
                    plt,
                    ok_factors,
                    "feature",
                    horizon_columns,
                    title="IC衰减：各因子不同持有期的平均RankIC",
                    ylabel="平均RankIC",
                    path=charts_dir / "ic_decay.png",
                )
                written.append("charts/ic_decay.png")
            quantile_columns = [
                f"q{i}_mean_fwd_pct"
                for i in range(1, 6)
                if f"q{i}_mean_fwd_pct" in ok_factors.columns
            ]
            if not ok_factors.empty and len(quantile_columns) >= 2:
                _grouped_bar_chart(
                    plt,
                    ok_factors,
                    "feature",
                    quantile_columns,
                    title="分位数收益：Q1(最低组)→Q5(最高组)平均未来收益",
                    ylabel="平均未来收益 (%)",
                    path=charts_dir / "quantile_returns.png",
                )
                written.append("charts/quantile_returns.png")

            # 热力图：因子×信号日的逐日RankIC；因子×分位的平均未来收益。
            ic_by_date = (analysis.get("factor_metadata") or {}).get("ic_by_date") or {}
            if len(ic_by_date) >= 2:
                factor_names = sorted(ic_by_date)
                date_labels = sorted({d for series in ic_by_date.values() for d, _ in series})
                lookup = {name: dict(series) for name, series in ic_by_date.items()}
                matrix = np.full((len(factor_names), len(date_labels)), np.nan)
                for i, name in enumerate(factor_names):
                    for j, date in enumerate(date_labels):
                        value = lookup[name].get(date)
                        if value is not None:
                            matrix[i, j] = value
                _heatmap_chart(
                    plt,
                    matrix,
                    factor_names,
                    date_labels,
                    title="RankIC热力图：因子 × 信号日",
                    colorbar_label="RankIC",
                    path=charts_dir / "ic_by_date.png",
                )
                written.append("charts/ic_by_date.png")

            # 多空分位累计收益曲线（alphalens口径）：逐信号日Q5−Q1 spread简单累加，
            # 展示因子截面结构的累计贡献；未计交易成本，末端与quintile_spread均值自洽。
            long_short = (analysis.get("factor_metadata") or {}).get(
                "long_short_spread_by_date"
            ) or {}
            plottable = {
                name: points for name, points in sorted(long_short.items()) if len(points) >= 2
            }
            if plottable:
                fig, ax = plt.subplots(figsize=(9.5, 4.4))
                for name, points in plottable.items():
                    labels = [p[0] for p in points]
                    cumulative = np.cumsum([float(p[1]) for p in points])
                    ax.plot(labels, cumulative, linewidth=1.2, label=str(name))
                ax.axhline(0, color="#334155", linewidth=0.8)
                ax.set_title("多空组合累计收益：Q5组 − Q1组（按信号日累加，未计成本）")
                ax.set_ylabel("累计spread (%)")
                ax.legend(fontsize=7, ncol=2)
                plt.xticks(rotation=45, ha="right", fontsize=7)
                fig.tight_layout()
                fig.savefig(charts_dir / "long_short.png", dpi=130)
                plt.close(fig)
                written.append("charts/long_short.png")

            if not ok_factors.empty and len(quantile_columns) >= 2:
                quantile_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"][: len(quantile_columns)]
                qmatrix = (
                    ok_factors[quantile_columns]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=float)
                )
                _heatmap_chart(
                    plt,
                    qmatrix,
                    ok_factors["feature"].tolist(),
                    quantile_labels,
                    title="分位数收益热力图：因子 × 分位组（区间平均）",
                    colorbar_label="平均未来收益 (%)",
                    path=charts_dir / "quantile_heatmap.png",
                )
                written.append("charts/quantile_heatmap.png")
    except Exception as exc:  # 图表失败不影响数据文件交付
        (charts_dir / "charts_error.txt").write_text(f"图表生成失败：{exc}", encoding="utf-8")
    analysis["charts_written"] = written
    return written


def scenario_csv(analysis: dict) -> pd.DataFrame:
    frame = analysis.get("stress_frame")
    if frame is None or frame.empty:
        return pd.DataFrame(
            [
                {
                    "scenario": "not_run",
                    "label": "本次运行未包含压力测试或被数据阻断",
                }
            ]
        )
    return frame


def factor_diagnostics_csv(analysis: dict) -> pd.DataFrame:
    diagnostics = analysis.get("factor_diagnostics")
    if diagnostics is None:
        return pd.DataFrame(
            [
                {
                    "feature": "not_available",
                    "status": "数据不足或本次运行未包含因子诊断",
                }
            ]
        )
    return diagnostics


def build_manifest(report_dir: str, analysis: dict, chart_files: list[str]) -> dict:
    """汇总报告产物哈希清单（manifest.json自身不入清单）。

    自引用哈希在数学上无法闭合：清单一旦包含自身哈希，写入即失效；历史
    双写版本记录的是已废弃前版的哈希，反而制造"哈希不一致"假象。因此
    outputs_sha256覆盖除manifest.json外的全部正式产物，校验方按同一约定
    跳过manifest自身。
    """
    outputs = {}
    for name in sorted(os.listdir(report_dir)):
        if name == "manifest.json":
            continue
        path = os.path.join(report_dir, name)
        if os.path.isfile(path):
            outputs[name] = sha256_file(path)
    for relative in chart_files:
        path = os.path.join(report_dir, relative)
        if os.path.isfile(path):
            outputs[relative.replace("\\", "/")] = sha256_file(path)
    request = analysis["request"]
    return {
        "report_id": analysis["report_id"],
        "product": "Alice Quant Audit · Alice量化策略审计台",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "inputs": {
            "cache": getattr(request, "cache", None),
            "cache_sha256": analysis.get("input_cache_sha256"),
            "candidates": getattr(request, "candidates", None),
            "source_info": analysis.get("source_info"),
            "candidate_info": analysis.get("candidate_info"),
        },
        "parameters": {
            "start": request.start,
            "end": request.end,
            "strategy": request.strategy,
            "initial_capital": request.initial_capital,
            "max_positions": request.max_positions,
            "max_position_pct": request.max_position_pct,
            "rebalance_freq": request.rebalance_freq,
            "commission_rate": request.commission_rate,
            "stamp_tax_rate": request.stamp_tax_rate,
            "slippage": request.slippage,
            "stop_loss_pct": request.stop_loss_pct,
            "take_profit_pct": request.take_profit_pct,
            "use_market_timing": request.use_market_timing,
            "trials": getattr(request, "trials", 1),
            "brand_name": getattr(request, "brand_name", "") or "",
            "brand_footer": getattr(request, "brand_footer", "") or "",
            "params_digest_sha256": request.params_digest(),
        },
        "benchmark_used": analysis.get("benchmark_code"),
        "blocked": analysis.get("blocked", False),
        "block_message": analysis.get("block_message"),
        "timings_seconds": analysis.get("timings") or {},
        "git": _git_snapshot(),
        "outputs_sha256": outputs,
        "notes": analysis.get("notes", []),
    }


def render_executive_summary(analysis: dict) -> str:
    score = analysis.get("score") or {}
    audit = analysis.get("data_audit") or {}
    base = (analysis.get("base") or {}).get("metrics") or {}
    lines = [
        "# Alice 量化策略审计报告 · 一页结论",
        "",
        f"- 报告编号：`{analysis['report_id']}`",
        f"- 回测区间：{analysis['request'].start} 至 {analysis['request'].end}",
        f"- 数据质量评级：**{audit.get('rating', '未知')}**",
        f"- 策略可信度评分：**{score.get('score', '无法评分')}**（{score.get('grade_label', '')}）",
        "",
        f"> {SCORE_DECLARATION}",
        "",
        "## 核心结论",
    ]
    if analysis.get("blocked"):
        lines.append("")
        lines.append(f"**本次分析被阻断**：{analysis.get('block_message')}")
        lines.append("数据缺陷会扭曲回测结论，未生成任何策略表现数据。")
    else:
        lines += [
            "",
            f"- 区间收益 {_pct(base.get('total_return_pct'))}，"
            f"最大回撤 {_pct(base.get('max_drawdown_pct'))}，"
            f"Sharpe {_fmt(base.get('sharpe_ratio'))}"
            f"（{'可信' if base.get('sharpe_reliable') else '样本不足、不可信'}）",
            f"- Sortino {_fmt(base.get('sortino_ratio'))}，Calmar {_fmt(base.get('calmar_ratio'))}，"
            f"最大连续亏损 {_fmt(base.get('max_consecutive_losses'))} 天",
            f"- 完整交易 {_fmt(base.get('total_trades'))} 笔；"
            f"相对沪深300超额 {_pct(base.get('excess_return_pct'))}",
        ]
        if base.get("mc_max_drawdown_pct_p95") is not None:
            lines.append(
                f"- 蒙特卡洛重构：最大回撤中位数 {_pct(base.get('mc_max_drawdown_pct_p50'))}，"
                f"95分位 {_pct(base.get('mc_max_drawdown_pct_p95'))}"
                "（历史日收益重排的统计重构，非预测）"
            )
        if base.get("deflated_sharpe") is not None:
            lines.append(
                f"- Deflated Sharpe {_fmt(base.get('deflated_sharpe'))}，"
                f"试验数N={_fmt(base.get('deflated_sharpe_trials'))}为用户声明值，"
                "系统无法验证实际尝试过的方案数，N>1时请确保如实申报"
                "（N=1时退化为PSR）"
            )
        if base.get("sharpe_ci_includes_zero"):
            lines.append(
                f"- Sharpe 95%自助置信区间 [{_fmt(base.get('sharpe_ci_low'))}, "
                f"{_fmt(base.get('sharpe_ci_high'))}] 包含零：正收益不具统计显著性，"
                "勿据单次回测下结论"
            )
        rolling = analysis.get("rolling_oos") or {}
        if rolling.get("status") == "ok":
            lines.append(f"- 滚动样本外：{rolling.get('detail', '')}")
        stress = analysis.get("stress_summary") or {}
        if stress.get("status") == "ok":
            lines.append(
                f"- 压力测试：{'对假设过度敏感' if stress.get('over_sensitive') else '未发现极端敏感性'}"
                + (
                    f"；影响最大的假设是「{stress['most_impactful']['label']}」"
                    if stress.get("most_impactful")
                    else ""
                )
            )
    if score.get("failed_dimensions"):
        lines.append(f"- 未通过维度：{', '.join(score['failed_dimensions'])}")
    if score.get("insufficient_dimensions"):
        lines.append(f"- 缺少数据维度：{', '.join(score['insufficient_dimensions'])}（不伪造结论）")
    warnings = audit.get("warnings") or []
    if warnings:
        lines += ["", "## 数据与口径警告"] + [f"- {w}" for w in warnings[:10]]
    lines += [
        "",
        "## 主要问题与改进建议",
    ]
    for issue in (analysis.get("issues") or [])[:8]:
        lines.append(f"- {issue}")
    lines += [
        "",
        "## 研究限制",
        "- 单一历史区间的回测不能代表未来表现；",
        "- 合成数据或CSV自备数据可能存在口径差异，详见数据质量附件；",
        "- 内置策略存在过拟合风险，评分不消除该风险；",
        "",
        f"> {DISCLAIMER}",
        "",
    ]
    return "\n".join(lines)


def render_html(analysis: dict) -> str:
    score = analysis.get("score") or {}
    audit = analysis.get("data_audit") or {}
    base = analysis.get("base") or {}
    metrics = base.get("metrics") or {}
    stats = base.get("statistics") or {}
    request = analysis["request"]
    esc = html.escape
    brand_name = str(getattr(request, "brand_name", "") or "")
    brand_footer = str(getattr(request, "brand_footer", "") or "")
    data_source = str(analysis.get("data_source") or "未声明")
    source_notice = (
        "SYNTHETIC 合成演示数据：本报告不代表真实行情或真实收益。"
        if "SYNTHETIC" in data_source.upper()
        else "数据评级仅覆盖程序检查项，不等于严格历史时点数据或未来收益认证。"
    )

    dimension_names = {key: name for key, name, _ in DIMENSION_SPECS}
    status_labels = {
        "pass": "通过",
        "partial": "部分",
        "fail": "未通过",
        "insufficient_data": "缺少数据",
    }
    dimension_rows = (
        "".join(
            f"<tr><td>{esc(dimension_names.get(d['key'], d['key']))}</td>"
            f"<td>{d['earned_points']} / {d['max_points']}</td>"
            f"<td>{status_labels.get(d['status'], d['status'])}</td>"
            f"<td>{esc('；'.join(d['reasons'])[:300])}</td></tr>"
            for d in score.get("dimensions", [])
        )
        or "<tr><td colspan='4'>评分未生成</td></tr>"
    )

    warnings = "".join(f"<li>{esc(w)}</li>" for w in (audit.get("warnings") or [])[:15])
    blockers = "".join(f"<li>{esc(b)}</li>" for b in (audit.get("blockers") or []))
    issues = "".join(f"<li>{esc(i)}</li>" for i in (analysis.get("issues") or []))
    manifest = analysis.get("manifest") or {}

    stress_rows = ""
    stress_frame = analysis.get("stress_frame")
    stress_records = stress_frame.to_dict("records") if stress_frame is not None else []
    for row in stress_records:
        if row.get("status") != "ok":
            stress_rows += (
                f"<tr><td>{esc(str(row.get('label')))}</td>"
                f"<td colspan='7'>未运行或数据不足</td></tr>"
            )
            continue
        stress_rows += (
            f"<tr><td>{esc(str(row.get('label')))}</td>"
            f"<td>{_pct(row.get('total_return_pct'))}</td>"
            f"<td>{_pct(row.get('max_drawdown_pct'))}</td>"
            f"<td>{_fmt(row.get('sharpe_ratio'))}</td>"
            f"<td>{_fmt(row.get('return_delta_pct'), ' pp')}</td>"
            f"<td>{_fmt(row.get('sharpe_delta'))}</td>"
            f"<td>{_fmt(row.get('annualized_turnover'), ' 倍')}</td></tr>"
        )
    stress_section = (
        f"<table><thead><tr><th>场景</th><th>收益</th><th>最大回撤</th><th>Sharpe</th>"
        f"<th>收益变化</th><th>Sharpe变化</th><th>年化换手</th></tr></thead><tbody>{stress_rows}</tbody></table>"
        if stress_rows
        else "<p class='muted'>本次运行未包含压力测试。</p>"
    )
    rolling = analysis.get("rolling_oos") or {}
    if rolling.get("status") == "ok":
        roll_rows = ""
        for window in rolling.get("windows", []):
            if window.get("status") != "ok":
                roll_rows += (
                    f"<tr><td>{esc(str(window.get('window')))}</td>"
                    f"<td colspan='5'>数据不足</td></tr>"
                )
                continue
            roll_rows += (
                f"<tr><td>{esc(str(window.get('window')))} "
                f"({esc(str(window.get('start')))} ~ {esc(str(window.get('end')))})</td>"
                f"<td>{_pct(window.get('total_return_pct'))}</td>"
                f"<td>{_pct(window.get('max_drawdown_pct'))}</td>"
                f"<td>{_fmt(window.get('sharpe_ratio'))}"
                f"{'⚠' if not window.get('sharpe_reliable') else ''}</td>"
                f"<td>{'[' + _fmt(window.get('sharpe_ci_low')) + ', ' + _fmt(window.get('sharpe_ci_high')) + ']' if window.get('sharpe_ci_low') is not None else '—'}"
                f"{'⚠含零' if window.get('sharpe_ci_includes_zero') else ''}</td>"
                f"<td>{_fmt(window.get('total_trades'), ' 笔')}</td></tr>"
            )
        stress_section += (
            f"<h3 style='margin-top:14px'>滚动样本外（不重叠窗口）</h3>"
            f"<p class='muted'>{esc(rolling.get('detail', ''))}</p>"
            f"<table><thead><tr><th>窗口</th><th>收益</th><th>最大回撤</th>"
            f"<th>Sharpe</th><th>Sharpe 95%CI(块自助)</th><th>交易数</th></tr></thead><tbody>{roll_rows}</tbody></table>"
            + (
                "<p><img src='charts/rolling_windows.png' alt='滚动窗口区间收益柱状图' "
                "style='max-width:100%;border:1px solid #e2e8f0;border-radius:8px'></p>"
                if "charts/rolling_windows.png" in (analysis.get("charts_written") or [])
                else ""
            )
            + f"<p class='muted'>{esc(rolling.get('note', ''))}</p>"
        )

    block_banner = ""
    if analysis.get("blocked"):
        if (analysis.get("effective_config") or {}).get("mode") == "data_audit_only":
            block_banner = (
                "<div class='notice'><strong>本次仅做数据审计，未运行回测。</strong></div>"
            )
        else:
            block_banner = (
                "<div class='notice danger'><strong>本次分析被阻断：</strong>"
                f"{esc(analysis.get('block_message') or '')}"
                "<br>数据缺陷足以扭曲回测结论，因此没有生成任何策略表现数据。</div>"
            )
        body_sections = ""
    else:
        eq_frame = base.get("equity")
        chart_svg = equity_svg(eq_frame, base.get("benchmark")) if eq_frame is not None else ""

        # 前5大回撤明细（pyfolio gen_drawdown_table口径，指标与明细同源）。
        dd_rows = ""
        for period in metrics.get("drawdown_periods") or []:
            recovered = period.get("recovery_days") is not None
            dd_rows += (
                f"<tr><td>{esc(str(period.get('peak_date')))}</td>"
                f"<td>{esc(str(period.get('valley_date')))}</td>"
                f"<td>{esc(str(period.get('recovery_date')) if recovered else '未恢复')}</td>"
                f"<td>{_pct(period.get('depth_pct'))}</td>"
                f"<td>{_fmt(period.get('decline_days'), ' 天')}</td>"
                f"<td>{_fmt(period.get('recovery_days'), ' 天') if recovered else '未恢复'}</td></tr>"
            )
        dd_section = (
            "<h3 style='margin-top:14px'>最大回撤明细（前5大回撤区间，pyfolio口径）</h3>"
            "<table><thead><tr><th>峰值日</th><th>谷底日</th><th>恢复日</th><th>深度</th>"
            "<th>下行天数</th><th>恢复天数</th></tr></thead>"
            f"<tbody>{dd_rows}</tbody></table>"
            if dd_rows
            else ""
        )

        capture_text = "—"
        if (
            metrics.get("up_capture_pct") is not None
            and metrics.get("down_capture_pct") is not None
        ):
            capture_text = (
                f"{_fmt(metrics.get('up_capture_pct'))} / {_fmt(metrics.get('down_capture_pct'))}"
            )

        # 蒙特卡洛重构小节（块自助重排；已实现回撤只是等价路径中的一条）。
        mc_section = ""
        if metrics.get("mc_max_drawdown_pct_p95") is not None:
            mc_section = (
                "<h3 style='margin-top:14px'>蒙特卡洛重构：回撤与终值的分布（块自助重排）</h3>"
                "<table><thead><tr><th>指标</th><th>分位数</th><th>数值</th></tr></thead><tbody>"
                f"<tr><td>最大回撤</td><td>中位数 p50</td><td>{_pct(metrics.get('mc_max_drawdown_pct_p50'))}</td></tr>"
                f"<tr><td>最大回撤</td><td>95分位 p95</td><td>{_pct(metrics.get('mc_max_drawdown_pct_p95'))}</td></tr>"
                f"<tr><td>终值收益</td><td>5分位 p5</td><td>{_pct(metrics.get('mc_terminal_return_p5_pct'))}</td></tr>"
                f"<tr><td>重排路径数</td><td>—</td><td>{_fmt(metrics.get('mc_paths'))}</td></tr>"
                "</tbody></table>"
                "<p class='muted'>对历史日收益做1000次块自助重排（保留短期依赖，固定种子可复现），"
                '把已实现回撤放到"同收益特征的等价路径"分布中衡量。这是历史样本的统计重构，'
                "不是对未来回撤或收益的预测。</p>"
            )

        # 交易级蒙特卡洛重排小节（tmc_*，第二口径，与上方日收益块自助并列）。
        tmc_section = ""
        if metrics.get("tmc_max_drawdown_pct_p95") is not None:
            tmc_section = (
                "<h3 style='margin-top:14px'>交易级蒙特卡洛重排：回撤分布（第二口径）</h3>"
                "<table><thead><tr><th>指标</th><th>分位数</th><th>数值</th></tr></thead><tbody>"
                f"<tr><td>最大回撤</td><td>中位数 p50</td><td>{_pct(metrics.get('tmc_max_drawdown_pct_p50'))}</td></tr>"
                f"<tr><td>最大回撤</td><td>95分位 p95</td><td>{_pct(metrics.get('tmc_max_drawdown_pct_p95'))}</td></tr>"
                f"<tr><td>终值收益</td><td>5分位 p5</td><td>{_pct(metrics.get('tmc_terminal_return_p5_pct'))}</td></tr>"
                f"<tr><td>已完成回合</td><td>—</td><td>{_fmt(metrics.get('tmc_round_trips'))}</td></tr>"
                f"<tr><td>重排路径数</td><td>—</td><td>{_fmt(metrics.get('tmc_paths'))}</td></tr>"
                "</tbody></table>"
                "<p class='muted'>把已完成回合的净盈亏（金额口径，含双边摩擦）随机重排后"
                "依次叠加到资金曲线上（固定种子可复现）。加法不变性保证任意重排的终值都"
                '与已实现总收益一致，因此终值p5就是已实现收益本身；回撤分布回答"同样的'
                '盈亏换个实现顺序会有多深"。与日收益块自助的差异：块自助重排日收益、保留'
                "时序依赖结构，但隐含日频可调仓；交易重排贴合离散调仓的实现，但丢弃回合间"
                "的时序信息。两口径分位数可互相印证：这是历史样本的统计重构，不是对未来"
                "回撤或收益的预测。</p>"
            )

        # 压力场景图（场景跑完才生成；缺图时如实留空）。
        scenario_deltas_img = (
            "<p><img src='charts/scenario_deltas.png' alt='压力场景收益与Sharpe变化图' "
            "style='max-width:100%;border:1px solid #e2e8f0;border-radius:8px'></p>"
            if "charts/scenario_deltas.png" in (analysis.get("charts_written") or [])
            else ""
        )

        # Deflated Sharpe（--trials声明试验数；缺省N=1时退化为PSR）。
        dsr_value = metrics.get("deflated_sharpe")
        dsr_row = ""
        dsr_note = ""
        if dsr_value is not None:
            dsr_row = (
                f"<tr><th>Deflated Sharpe</th><td>{_fmt(dsr_value)}</td>"
                f"<th>试验次数N</th><td>{_fmt(metrics.get('deflated_sharpe_trials'))}</td></tr>"
            )
            dsr_note = (
                "Deflated Sharpe为观测Sharpe扣除N次试验多重检验后仍显著的概率"
                "（0-1，越接近1越可信；N=1时退化为PSR，与块自助CI互为印证）；"
                "它是统计证据强度度量，不是对未来收益的预测。"
                "试验数N为用户声明值，系统无法验证实际尝试过的方案数，"
                "N>1时请确保如实申报（compare模式的N=完成回测方案数，"
                "为系统实测，不受此声明约束）。"
            )

        body_sections = f"""
<section class="panel"><h2>4. 收益与回撤</h2><section class="chart">{chart_svg}</section>
<table><tbody>
<tr><th>区间收益</th><td>{_pct(metrics.get("total_return_pct"))}</td><th>年化收益</th><td>{
            _pct(metrics.get("cagr_pct"))
        }</td></tr>
<tr><th>最大回撤</th><td>{_pct(metrics.get("max_drawdown_pct"))}</td><th>年化波动</th><td>{
            _pct(metrics.get("annual_volatility_pct"))
        }</td></tr>
<tr><th>Sharpe</th><td>{_fmt(metrics.get("sharpe_ratio"))} {
            "✅可信" if metrics.get("sharpe_reliable") else "⚠️样本不足"
        }</td><th>完整交易</th><td>{_fmt(metrics.get("total_trades"), " 笔")}</td></tr>
<tr><th>Sortino</th><td>{_fmt(metrics.get("sortino_ratio"))}</td><th>Calmar</th><td>{
            _fmt(metrics.get("calmar_ratio"))
        }</td></tr>
<tr><th>最大连续亏损/盈利</th><td>{_fmt(metrics.get("max_consecutive_losses"))} / {
            _fmt(metrics.get("max_consecutive_wins"))
        } 天</td><th>Sharpe 95%CI(块自助)</th><td>{
            "["
            + _fmt(metrics.get("sharpe_ci_low"))
            + ", "
            + _fmt(metrics.get("sharpe_ci_high"))
            + "]"
            if metrics.get("sharpe_ci_low") is not None
            else "—"
        }{" ⚠包含零" if metrics.get("sharpe_ci_includes_zero") else ""}</td></tr>
<tr><th>尾部比率</th><td>{_fmt(metrics.get("tail_ratio"))}</td><th>回收因子</th><td>{
            _fmt(metrics.get("recovery_factor"))
        }</td></tr>
<tr><th>VaR95(日)</th><td>{_pct(metrics.get("var_95_pct"))}</td><th>CVaR95(日)</th><td>{
            _pct(metrics.get("cvar_95_pct"))
        }</td></tr>
<tr><th>偏度</th><td>{_fmt(metrics.get("skewness"))}</td><th>超额峰度</th><td>{
            _fmt(metrics.get("excess_kurtosis"))
        }</td></tr>
<tr><th>Ulcer Index</th><td>{_fmt(metrics.get("ulcer_index_pct"))}</td><th>Martin比率</th><td>{
            _fmt(metrics.get("martin_ratio"))
        }</td></tr>
<tr><th>Omega比率(阈值0)</th><td>{_fmt(metrics.get("omega_ratio"))}</td><th></th><td></td></tr>
<tr><th>最好/最差单笔</th><td>{_pct(metrics.get("best_trade_pct"))} / {
            _pct(metrics.get("worst_trade_pct"))
        }</td><th>单笔期望值</th><td>{_pct(metrics.get("expectancy_pct"))}</td></tr>
<tr><th>平均/最长持仓</th><td>{_fmt(metrics.get("avg_trade_duration_days"))} / {
            _fmt(metrics.get("max_trade_duration_days"))
        } 天</td><th>SQN</th><td>{_fmt(metrics.get("sqn"))}</td></tr>
{dsr_row}
</tbody></table>
{render_snapshot_table(metrics)}
{dd_section}
{mc_section}
{tmc_section}
<p class="muted">Sortino仅惩罚下行波动；Calmar=年化收益/最大回撤；Sharpe置信区间采用块自助（保留收益短期依赖），区间包含零说明正收益不具统计显著性。尾部比率=日收益P95/P5绝对值之比（empyrical口径，&gt;1表示右尾厚于左尾）；回收因子=区间收益/最大回撤深度（pyfolio/quantstats常用口径，越大代表修复回撤越快）。{
            dsr_note
        }以上均为历史统计，不评价未来收益。</p>
{
            "".join(
                f"<p><img src='{name}' alt='{name}' style='max-width:100%;border:1px solid #e2e8f0;border-radius:8px'></p>"
                for name in (
                    "charts/drawdown.png",
                    "charts/return_distribution.png",
                    "charts/rolling_sharpe.png",
                    "charts/exposure.png",
                    "charts/equity_curve.png",
                    "charts/monthly_returns.png",
                    "charts/trade_distribution.png",
                )
                if name in (analysis.get("charts_written") or [])
            )
        }</section>
<section class="panel"><h2>5. 相对沪深300表现</h2><table><tbody>
<tr><th>策略收益</th><td>{_pct(metrics.get("total_return_pct"))}</td><th>基准收益</th><td>{
            _pct(metrics.get("benchmark_return_pct"))
        }</td></tr>
<tr><th>超额收益</th><td>{_pct(metrics.get("excess_return_pct"))}</td><th>基准口径</th><td>{
            esc(str(analysis.get("benchmark_code")))
        }</td></tr>
<tr><th>Alpha(年化)</th><td>{_pct(metrics.get("alpha_annualized_pct"))}</td><th>Beta</th><td>{
            _fmt(metrics.get("beta"))
        }</td></tr>
<tr><th>信息比率</th><td>{_fmt(metrics.get("information_ratio"))}</td><th>上行/下行捕获</th><td>{
            capture_text
        }</td></tr>
</tbody></table><p class="muted">超额收益为区间简单差值，不代表风险调整后优势。Alpha/Beta为对基准日收益的CAPM回归（回归口径参考pyfolio performance.alpha_beta，Alpha为回归截距复利年化）；信息比率=超额日收益均值/标准差×√252（quantstats口径）；上行/下行捕获=基准涨/跌日的策略收益均值与基准收益均值之比（pyfolio capture口径），100%表示与基准同涨同跌。基准缺失或样本不足时以上字段如实留空。</p></section>
<section class="panel"><h2>6. 成本与换手</h2><table><tbody>
<tr><th>累计交易摩擦</th><td>¥{_fmt(stats.get("total_friction"))}</td><th>摩擦/初始资金</th><td>{
            _pct(stats.get("friction_pct_of_initial"))
        }</td></tr>
<tr><th>年化换手估算</th><td>{
            _fmt(stats.get("annualized_turnover"), " 倍")
        }</td><th>平均资金暴露</th><td>{_pct(stats.get("average_exposure_pct"))}</td></tr>
<tr><th>佣金率</th><td>{_pct(request.commission_rate * 100)}</td><th>印花税率(卖出)</th><td>{
            _pct(request.stamp_tax_rate * 100)
        }</td></tr>
<tr><th>滑点</th><td>{_pct(request.slippage * 100)}</td><th>止盈/止损</th><td>{
            _pct(request.take_profit_pct * 100)
        } / {_pct(request.stop_loss_pct * 100)}</td></tr>
</tbody></table></section>
<section class="panel"><h2>7. 压力测试</h2>{stress_section}{scenario_deltas_img}
<p class="muted">{
            esc(((analysis.get("stress_summary") or {}).get("interpretation")) or "")
        }</p></section>
"""

    factor_section = analysis.get("factor_section_html") or (
        "<p class='muted'>本次运行未包含因子诊断；缺少数据时不伪造结论。</p>"
    )
    exposure_section = analysis.get("exposure_section_html") or (
        "<p class='muted'>风险暴露需要行业/市值字段，当前数据不足，显示为数据不足。</p>"
    )
    repro_rows = (
        "".join(
            f"<tr><td><code>{esc(k)}</code></td><td class='hash'>{esc((v or '')[:16])}{'…' if v else ''}</td></tr>"
            for k, v in (manifest.get("outputs_sha256") or {}).items()
        )
        or "<tr><td colspan='2'>无</td></tr>"
    )

    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alice量化策略审计台 · {esc(analysis["report_id"])}</title><style>
:root{{--bg:#f4f7fb;--ink:#172033;--muted:#64748b;--line:#e2e8f0;--card:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei",system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:30px 20px 60px}}h1{{margin:0;font-size:28px}}h2{{font-size:19px;margin:0 0 12px}}
.sub{{color:var(--muted);margin:6px 0 18px}}.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.metric{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}}
.metric span{{color:var(--muted);font-size:12px;display:block}}.metric strong{{font-size:22px;display:block;margin-top:6px}}
table{{width:100%;border-collapse:collapse;margin-top:8px}}th,td{{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left;font-size:14px;vertical-align:top}}th{{color:var(--muted);font-weight:500}}
.score-hero{{display:flex;align-items:center;gap:18px;flex-wrap:wrap}}.score-num{{font-size:52px;font-weight:700}}
.badge{{padding:4px 12px;border-radius:999px;font-size:13px}}.badge-ok{{background:#dcfce7;color:#15803d}}.badge-warn{{background:#fef3c7;color:#b45309}}.badge-fail{{background:#fee2e2;color:#b91c1c}}
.notice{{border-left:4px solid #f59e0b;background:#fff7ed;padding:12px 14px;border-radius:8px;margin-top:14px}}.notice.danger{{border-color:#dc2626;background:#fef2f2}}
.disclaimer{{border:1px dashed #94a3b8;border-radius:10px;padding:14px;color:#475569;margin-top:16px;font-size:13px}}
.muted{{color:var(--muted);font-size:13px}}.hash{{font-family:Consolas,monospace;font-size:12px}}
.chart svg{{display:block;width:100%;height:auto}}ul{{margin:6px 0;padding-left:20px}}li{{font-size:14px;margin:3px 0}}
@media(max-width:760px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<h1>Alice量化策略审计台 · 策略可信度报告</h1>
<p class="sub">{esc(("【" + str(brand_name) + "】") if brand_name else "")}报告编号 <code>{esc(analysis["report_id"])}</code> · 生成于 {esc(str(manifest.get("created_at", "")))} · 仅供研究，不构成投资建议</p>
<div class="notice">{esc(DISCLAIMER)}</div>
<div class="notice"><strong>{esc(source_notice)}</strong> 数据来源声明：{esc(data_source)}</div>
{block_banner}

<section class="panel"><h2>1. 一页结论</h2><div class="score-hero">
<div><div class="score-num">{score.get("score") if score.get("score") is not None else "—"}</div>
<div class="muted">策略可信度评分 / 100</div></div>
<div>{_grade_badge(score.get("score"))}<div style="margin-top:8px">{esc(score.get("grade_label", ""))}</div></div>
<div class="muted" style="flex:1;min-width:240px">数据质量评级 <strong>{esc(audit.get("rating", "—"))}</strong> ·
回测区间 {esc(request.start)} 至 {esc(request.end)}</div></div>
<p class="muted">{esc(SCORE_DECLARATION)}{esc(score.get("usage_note", ""))}</p>
<table><thead><tr><th>评分维度</th><th>得分</th><th>状态</th><th>扣分原因 / 说明</th></tr></thead><tbody>{dimension_rows}</tbody></table></section>

<section class="panel"><h2>2. 数据质量</h2><table><tbody>
<tr><th>评级</th><td>{esc(audit.get("rating", "—"))}</td><th>股票数量</th><td>{_fmt((audit.get("coverage") or {}).get("stock_count"), " 只")}</td></tr>
<tr><th>区间交易日</th><td>{_fmt((audit.get("coverage") or {}).get("trade_sessions_in_range"), " 天")}</td><th>中位覆盖率</th><td>{_pct(((audit.get("coverage") or {}).get("median_session_coverage") or 0) * 100)}</td></tr>
<tr><th>异常OHLC行</th><td>{_fmt((audit.get("coverage") or {}).get("invalid_ohlc_rows_total"))}</td><th>重复记录</th><td>{_fmt((audit.get("coverage") or {}).get("duplicate_rows_total"))}</td></tr>
<tr><th>停牌/ST字段</th><td>{"完整" if (audit.get("fields") or {}).get("historical_trade_status") else "不完整"}</td><th>PE/PB覆盖率</th><td>{_pct(((audit.get("fields") or {}).get("valuation_coverage") or 0) * 100)}</td></tr>
<tr><th>幸存者偏差风险</th><td>{esc(((audit.get("survivorship_bias_risk") or {}).get("level") or "—"))}</td><th>基准</th><td>{esc(str(analysis.get("benchmark_code")))}</td></tr>
</tbody></table>
{f"<ul class='issues'>{blockers}</ul>" if blockers else ""}
{f"<ul>{warnings}</ul>" if warnings else ""}</section>

<section class="panel"><h2>3. 回测范围</h2><table><tbody>
<tr><th>区间</th><td>{esc(request.start)} 至 {esc(request.end)}</td><th>策略</th><td>{esc(request.strategy)}</td></tr>
<tr><th>初始资金</th><td>¥{_fmt(request.initial_capital, "", "0")}</td><th>持仓上限</th><td>{request.max_positions} 只 × {int(request.max_position_pct * 100)}%</td></tr>
<tr><th>调仓频率</th><td>{esc(request.rebalance_freq)}</td><th>市场择时</th><td>{"启用" if request.use_market_timing else "关闭"}</td></tr>
</tbody></table></section>
{body_sections}

<section class="panel"><h2>8. 因子诊断</h2>{factor_section}</section>
<section class="panel"><h2>9. 风险暴露</h2>{exposure_section}</section>
<section class="panel"><h2>10. 主要问题</h2>{f"<ul>{issues}</ul>" if issues else "<p class='muted'>未记录到显著问题。</p>"}</section>
<section class="panel"><h2>11. 改进建议</h2>{f"<ul>{''.join('<li>' + esc(s) + '</li>' for s in analysis.get('recommendations', []))}</ul>" if analysis.get("recommendations") else "<p class='muted'>无。</p>"}</section>
<section class="panel"><h2>12. 研究限制</h2><ul>
<li>回测基于历史数据，不预测未来收益；单一区间存在过拟合风险。</li>
<li>停牌/涨跌停按保守规则近似，真实成交可能更差。</li>
<li>CSV/自备数据字段口径可能与Alice标准缓存存在差异，见数据质量附件。</li>
{"".join(f"<li>{esc(n)}</li>" for n in analysis.get("notes", []))}
</ul>
{"".join(f'<div class="notice">{esc(n)}</div>' for n in analysis.get("notes_banner", []))}</section>

<section class="panel"><h2>13. 可复现信息</h2><table><tbody>
<tr><th>Git分支</th><td>{esc(str((manifest.get("git") or {}).get("branch")))}</td><th>Git提交</th><td class="hash">{esc(str((manifest.get("git") or {}).get("commit"))[:12])}</td></tr>
<tr><th>参数哈希</th><td class="hash">{esc(str(manifest.get("parameters", {}).get("params_digest_sha256", ""))[:16])}…</td><th>数据文件哈希</th><td class="hash">{esc(str(manifest.get("inputs", {}).get("cache_sha256", ""))[:16])}…</td></tr>
</tbody></table><p class="muted">产物哈希清单：</p><table>{repro_rows}</table>
<p class="links"><a href="executive_summary.md">一页结论 MD</a> · <a href="metrics.json">指标 JSON</a> ·
<a href="data_quality.json">数据质量 JSON</a> · <a href="strategy_audit.json">评分 JSON</a> ·
<a href="scenario_comparison.csv">压力对比 CSV</a> · <a href="factor_diagnostics.csv">因子诊断 CSV</a> ·
<a href="effective_config.json">有效参数</a> · <a href="manifest.json">清单与哈希</a></p></section>

<section class="panel disclaimer"><strong>14. 免责声明</strong><br>
{esc(brand_footer + "<br>" if brand_footer else "")}本报告由 Alice Quant Audit（Alice量化策略审计台）自动生成，仅供研究与学习使用，不构成任何投资建议或收益承诺。
该评分评价研究流程的可信度，不评价未来收益。本工具不连接券商、不执行自动交易、不保存任何账户凭证；
依据本报告做出任何决策的责任由使用者自行承担。投资有风险，入市需谨慎。</section>
</main></body></html>"""
    return document


def compare_frame(analysis: dict) -> pd.DataFrame:
    """逐方案对比表。"""
    rows = []
    for result in analysis.get("plan_results", []):
        if result.get("status") != "ok":
            rows.append(
                {
                    "label": result.get("label"),
                    "status": "insufficient_data",
                    "detail": result.get("detail", ""),
                }
            )
            continue
        metrics = result.get("metrics") or {}
        score = result.get("score") or {}
        rows.append(
            {
                "label": result.get("label"),
                "status": "ok",
                "total_return_pct": metrics.get("total_return_pct"),
                "cagr_pct": metrics.get("cagr_pct"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "sharpe_reliable": metrics.get("sharpe_reliable"),
                "deflated_sharpe": (result.get("deflated_sharpe") or {}).get("deflated_sharpe"),
                "total_trades": metrics.get("total_trades"),
                "annualized_turnover": result.get("annualized_turnover"),
                "total_friction": result.get("total_friction"),
                "return_delta_vs_baseline": result.get("return_delta_vs_baseline"),
                "score": score.get("score"),
                "data_rating": (analysis.get("data_audit") or {}).get("rating"),
            }
        )
    return pd.DataFrame(rows)


def render_dispersion_correlation_table(dispersion: dict) -> str:
    """方案两两日收益相关矩阵HTML小节。

    数据源：overfitting_audit.dispersion.correlations（compare分散度诊断
    产出的逐对相关清单）。分散度不足的红色警示由调用方现有逻辑渲染，
    本函数只负责矩阵本体；无法构成≥2方案矩阵时返回空串。
    """
    correlations = dispersion.get("correlations") or []
    labels: list[str] = []
    for item in correlations:
        for name in item.get("pair") or []:
            if isinstance(name, str) and name and name not in labels:
                labels.append(name)
    if len(labels) < 2:
        return ""
    lookup = {}
    for item in correlations:
        pair = item.get("pair") or []
        if len(pair) == 2 and all(isinstance(n, str) and n for n in pair):
            lookup[tuple(sorted(pair))] = item.get("correlation")
    esc = html.escape

    def _cell(i: int, j: int) -> str:
        if i == j:
            return "1.0000"
        corr = lookup.get(tuple(sorted((labels[i], labels[j]))))
        return "—" if corr is None else f"{corr:.4f}"

    head = "".join(f"<th>{esc(str(name))}</th>" for name in labels)
    body = "".join(
        "<tr><th>"
        + esc(str(labels[i]))
        + "</th>"
        + "".join(f"<td>{_cell(i, j)}</td>" for j in range(len(labels)))
        + "</tr>"
        for i in range(len(labels))
    )
    return (
        "<p class='muted'>方案两两日收益相关矩阵（Pearson，按共同交易日对齐；"
        "重叠观测不足或无离散度时如实显示—，对角线恒为1）：</p>"
        f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def render_compare_html(analysis: dict) -> str:
    esc = html.escape
    audit = analysis.get("data_audit") or {}
    request = analysis["request"]
    brand_name = str(request.brand_name or "")
    frame = compare_frame(analysis)
    ok_rows = ""
    for row in frame.to_dict("records"):
        if row.get("status") != "ok":
            ok_rows += (
                f"<tr><td>{esc(str(row.get('label')))}</td>"
                f"<td colspan='10'>{esc(str(row.get('detail', '未完成')))}</td></tr>"
            )
            continue
        ok_rows += (
            f"<tr><td><b>{esc(str(row.get('label')))}</b></td>"
            f"<td>{_pct(row.get('total_return_pct'))}</td>"
            f"<td>{_pct(row.get('max_drawdown_pct'))}</td>"
            f"<td>{_fmt(row.get('sharpe_ratio'))}</td>"
            f"<td>{_fmt(row.get('deflated_sharpe'))}</td>"
            f"<td>{_fmt(row.get('total_trades'), ' 笔')}</td>"
            f"<td>{_fmt(row.get('annualized_turnover'), ' 倍')}</td>"
            f"<td>{_fmt(row.get('return_delta_vs_baseline'), ' pp')}</td>"
            f"<td>{_fmt(row.get('score'))}</td></tr>"
        )
    plan_rows = "".join(
        f"<tr><td>{esc(str(p.get('label')))}</td>"
        f"<td class='hash'>{esc(json.dumps(p.get('overrides') or {}, ensure_ascii=False, sort_keys=True))}</td></tr>"
        for p in analysis.get("plans", [])
    )
    warnings = "".join(f"<li>{esc(w)}</li>" for w in (audit.get("warnings") or [])[:10])

    # 过拟合审计小节：PBO全局结果 + 各方案DSR见对比表DSR列。
    overfit = analysis.get("overfitting_audit") or {}
    pbo = overfit.get("pbo") or {}
    if pbo.get("status") == "ok":
        histogram = pbo.get("oos_relative_rank_histogram") or []
        hist_header = "".join(
            f"<th>{i / 10:.1f}–{(i + 1) / 10:.1f}</th>" for i in range(len(histogram))
        )
        hist_cells = "".join(f"<td>{int(v)}</td>" for v in histogram)
        # 分散度前置诊断：方案组缺乏分散度时明确警示PBO无统计意义。
        dispersion = overfit.get("dispersion") or {}
        if dispersion.get("validity") == "insufficient_dispersion":
            dispersion_note = f"<div class='notice' style='border-color:#dc2626;background:#fef2f2'>⚠ {esc(str(dispersion.get('detail') or ''))}</div>"
        elif dispersion:
            dispersion_note = f"<p class='muted'>{esc(str(dispersion.get('detail') or ''))}</p>"
        else:
            dispersion_note = ""
        correlation_table = render_dispersion_correlation_table(dispersion)
        degradation = pbo.get("degradation_regression") or {}
        degradation_note = ""
        if degradation.get("slope") is not None:
            degradation_note = (
                "性能退化回归：样本内最优方案的样本外Sharpe对样本内Sharpe"
                f"线性回归，斜率{_fmt(degradation.get('slope'))}、"
                f"截距{_fmt(degradation.get('intercept'))}、"
                f"{_fmt(degradation.get('pairs'))}个组合；"
                "斜率明显小于1提示样本内优势在样本外大幅缩水。"
            )
        overfit_section = f"""
<section class="panel"><h2>过拟合审计（PBO / Deflated Sharpe）</h2>
{dispersion_note}
{correlation_table}
<table><tbody>
<tr><th>PBO过拟合概率</th><td>{_fmt(pbo.get("pbo_probability"))}</td><th>CSCV组合数</th><td>{_fmt(pbo.get("cscv_combinations"))}</td></tr>
<tr><th>子块数 / 每块天数</th><td>{_fmt(pbo.get("subblock_count"))} / {_fmt(pbo.get("subblock_days"))}</td><th>ω均值</th><td>{_fmt(pbo.get("mean_oos_relative_rank"))}</td></tr>
</tbody></table>
<p class="muted">IS最优方案的样本外相对排名ω分布直方图（10档）：</p>
<table><thead><tr>{hist_header}</tr></thead><tbody><tr>{hist_cells}</tr></tbody></table>
<p class="muted">PBO衡量的是这组方案整体过拟合概率，不是对未来收益的预测。
PBO只反映本窗口内的排名延续性，同一方案组在不同窗口的PBO可能大幅波动，
单窗口数字不可外推。
{degradation_note}
对比表中DSR列为各方案的Deflated Sharpe（N={_fmt(overfit.get("n_trials"))}次试验的多重检验扣除，0-1，越接近1说明扣除多重检验后Sharpe仍显著；统计证据强度度量，不预测未来收益）。
方法：CSCV/DSR按Bailey & López de Prado论文公式自行实现，方案不足或时段不足时如实输出"数据不足"。</p></section>"""
    else:
        overfit_section = f"""
<section class="panel"><h2>过拟合审计（PBO / Deflated Sharpe）</h2>
<p class="muted">{esc(str(pbo.get("detail") or "数据不足，不输出PBO/DSR结论。"))}</p></section>"""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>策略对比报告</title><style>
:root{{--bg:#f4f7fb;--ink:#172033;--muted:#64748b;--line:#e2e8f0;--card:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei",system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:30px 20px 60px}}h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:19px}}
.sub,.muted{{color:var(--muted);font-size:13px}}.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}}
table{{width:100%;border-collapse:collapse;margin-top:8px}}th,td{{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left;font-size:14px;vertical-align:top}}th{{color:var(--muted);font-weight:500}}
.notice{{border-left:4px solid #f59e0b;background:#fff7ed;padding:12px 14px;border-radius:8px;margin-top:14px;font-size:13px}}
.hash{{font-family:Consolas,monospace;font-size:11px;word-break:break-all}}
</style></head><body><main>
<h1>批量策略对比报告</h1>
<p class="sub">{esc(("【" + brand_name + "】") if brand_name else "")}报告编号 <code>{esc(analysis["report_id"])}</code> · 区间 {esc(request.start)} 至 {esc(request.end)} · 仅供研究，不构成投资建议</p>
<div class="notice">{esc(DISCLAIMER)}　评分声明：{esc(SCORE_DECLARATION)}</div>
<section class="panel"><h2>数据质量（全部方案共享）</h2>
<p>评级 <b>{esc(audit.get("rating", "—"))}</b> · 股票数 {_fmt((audit.get("coverage") or {}).get("stock_count"))} ·
中位覆盖率 {_pct(((audit.get("coverage") or {}).get("median_session_coverage") or 0) * 100)}</p>
{"<ul>" + warnings + "</ul>" if warnings else ""}</section>
<section class="panel"><h2>方案对比</h2>
<table><thead><tr><th>方案</th><th>收益</th><th>最大回撤</th><th>Sharpe</th><th>DSR</th><th>交易数</th>
<th>年化换手</th><th>vs基线(pp)</th><th>评分</th></tr></thead><tbody>{ok_rows}</tbody></table>
<p class="muted">基线为计划中的第一个方案；对比模式评分未含压力测试与样本外维度，仅供横向参考，不代表完整可信度结论。DSR=Deflated Sharpe（多重检验扣除后的显著概率）。</p></section>
{overfit_section}
<section class="panel"><h2>方案参数（用户显式指定）</h2>
<table><tbody>{plan_rows}</tbody></table>
<p class="muted">本工具不做自动参数搜索；以上方案全部来自用户提供的计划文件。</p></section>
<section class="panel"><h2>可复现信息</h2>
<p class="muted">完整哈希清单见 <a href="manifest.json">manifest.json</a>，逐方案指标见 <a href="scenario_comparison.csv">scenario_comparison.csv</a>。</p></section>
<section class="panel" style="border:1px dashed #94a3b8;border-radius:10px"><b>免责声明</b><br>
本报告由 Alice Quant Audit 自动生成，仅供研究，不构成投资建议；历史回测不代表未来收益。</section>
</main></body></html>"""


def render_compare_summary(analysis: dict) -> str:
    request = analysis["request"]
    audit = analysis.get("data_audit") or {}
    lines = [
        "# 批量策略对比 · 一页结论",
        "",
        f"- 报告编号：`{analysis['report_id']}`",
        f"- 区间：{request.start} 至 {request.end}；数据评级：**{audit.get('rating', '—')}**",
        f"- 方案数：{len(analysis.get('plan_results', []))}（基线为第一个方案）",
        "",
        "| 方案 | 收益 | 最大回撤 | Sharpe | 评分 | vs基线(pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in analysis.get("plan_results", []):
        if result.get("status") != "ok":
            lines.append(f"| {result.get('label')} | 未完成 | — | — | — | — |")
            continue
        m = result["metrics"]
        score = (result.get("score") or {}).get("score")
        lines.append(
            f"| {result['label']} | {m.get('total_return_pct')}% | "
            f"{m.get('max_drawdown_pct')}% | {m.get('sharpe_ratio')} | "
            f"{score if score is not None else '—'} | "
            f"{result.get('return_delta_vs_baseline')} |"
        )
    overfit = analysis.get("overfitting_audit") or {}
    pbo = overfit.get("pbo") or {}
    dispersion = overfit.get("dispersion") or {}
    if pbo.get("pbo_probability") is not None:
        if dispersion.get("validity") == "ok":
            validity_note = "分散度诊断通过，PBO具备参考意义"
        elif dispersion.get("detail"):
            validity_note = f"⚠ {dispersion['detail']}"
        else:
            validity_note = "无分散度诊断记录"
        lines.append(
            f"- 过拟合审计：PBO={pbo.get('pbo_probability')}"
            f"（CSCV组合数{pbo.get('cscv_combinations')}；{validity_note}；"
            "PBO随窗口可能大幅波动，单窗口数字不可外推）"
        )
        degradation = pbo.get("degradation_regression") or {}
        if degradation.get("slope") is not None:
            lines.append(
                f"- 性能退化回归：OOS~IS斜率={degradation.get('slope')}"
                f"（{degradation.get('pairs')}个组合；斜率明显小于1提示"
                "样本内优势在样本外大幅缩水）"
            )
    lines += [
        "",
        "> 对比模式评分未含压力测试与样本外维度；完整评估请对单一方案运行 full 分析。",
        "",
        f"> {DISCLAIMER}",
        "",
    ]
    return "\n".join(lines)


def write_report_bundle(report_dir: str, analysis: dict) -> dict:
    """写全部交付物；返回最终manifest。

    原子化：全部内容先写入 <report_dir>/.staging，成功后整体移入正式目录；
    中途失败留下 ERROR.txt 与半成品隔离，正式目录不会出现混合状态。
    """
    import shutil

    root = Path(report_dir)
    staging = root / ".staging"
    staging.mkdir(exist_ok=True)
    try:
        if analysis.get("mode") == "compare":
            _write_json(
                staging / "metrics.json",
                {
                    "mode": "compare",
                    "plans": [
                        {
                            "label": r.get("label"),
                            "status": r.get("status"),
                            "metrics": r.get("metrics") or {},
                        }
                        for r in analysis.get("plan_results", [])
                    ],
                    "overfitting_audit": analysis.get("overfitting_audit") or {},
                },
            )
            _write_json(
                staging / "strategy_audit.json",
                {
                    "mode": "compare",
                    "plans": [
                        {
                            "label": r.get("label"),
                            "score": r.get("score") or {},
                            "overrides": r.get("overrides") or {},
                        }
                        for r in analysis.get("plan_results", [])
                    ],
                },
            )
            _write_json(staging / "effective_config.json", analysis.get("effective_config") or {})
            _write_json(staging / "data_quality.json", analysis.get("data_audit") or {})
            compare_frame(analysis).to_csv(
                staging / "scenario_comparison.csv", index=False, encoding="utf-8-sig"
            )
            (staging / "executive_summary.md").write_text(
                render_compare_summary(analysis), encoding="utf-8"
            )
            html_document = render_compare_html(analysis)
            (staging / "index.html").write_text(html_document, encoding="utf-8")
            for entry in sorted(staging.iterdir()):
                target = root / entry.name
                if entry.is_dir():
                    import shutil as _sh

                    if target.exists():
                        _sh.rmtree(target)
                    _sh.move(str(entry), str(target))
                else:
                    os.replace(str(entry), str(target))
            staging.rmdir()
            manifest = build_manifest(report_dir, analysis, ["index.html"])
            _write_json(root / "manifest.json", manifest)
            analysis["manifest"] = manifest
            try:
                from commercial.experiment_index import append_experiment_index

                append_experiment_index(report_dir, analysis)
            except Exception:  # 索引失败绝不影响报告本体
                pass
            return manifest
        _write_json(
            staging / "metrics.json",
            (analysis.get("base") or {}).get("metrics")
            or {"status": "not_run", "reason": analysis.get("block_message", "")},
        )
        _write_json(staging / "data_quality.json", analysis.get("data_audit") or {})
        _write_json(staging / "strategy_audit.json", analysis.get("score") or {})
        _write_json(staging / "effective_config.json", analysis.get("effective_config") or {})
        scenario_csv(analysis).to_csv(
            staging / "scenario_comparison.csv", index=False, encoding="utf-8-sig"
        )
        factor_diagnostics_csv(analysis).to_csv(
            staging / "factor_diagnostics.csv", index=False, encoding="utf-8-sig"
        )
        (staging / "executive_summary.md").write_text(
            render_executive_summary(analysis), encoding="utf-8"
        )
        if analysis.get("factor_metadata") is not None:
            _write_json(staging / "factor_diagnostics_metadata.json", analysis["factor_metadata"])
        if analysis.get("rolling_oos") is not None:
            _write_json(staging / "rolling_oos.json", analysis["rolling_oos"])
        chart_files = write_charts(str(staging), analysis)

        # 因子诊断图表生成成功时，嵌入因子小节（相对路径随bundle整体搬移仍有效）。
        charts_written = analysis.get("charts_written") or []
        factor_images = "".join(
            f"<p><img src='{name}' alt='{name}' style='max-width:100%;"
            "border:1px solid #e2e8f0;border-radius:8px'></p>"
            for name in (
                "charts/ic_decay.png",
                "charts/quantile_returns.png",
                "charts/ic_by_date.png",
                "charts/quantile_heatmap.png",
                "charts/long_short.png",
            )
            if name in charts_written
        )
        if factor_images and analysis.get("factor_section_html"):
            analysis["factor_section_html"] += factor_images

        html_document = render_html(analysis)
        # 整体搬入正式目录。
        for entry in sorted(staging.iterdir()):
            target = root / entry.name
            if entry.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(entry), str(target))
            else:
                os.replace(str(entry), str(target))
        staging.rmdir()

        # index.html先落盘，manifest最后一次性写入：哈希覆盖全部正式产物
        # （含index.html）；manifest自身不入清单（自引用哈希无法闭合）。
        (root / "index.html").write_text(html_document, encoding="utf-8")
        manifest = build_manifest(report_dir, analysis, chart_files + ["index.html"])
        _write_json(root / "manifest.json", manifest)
        analysis["manifest"] = manifest
        try:
            from commercial.experiment_index import append_experiment_index

            append_experiment_index(report_dir, analysis)
        except Exception:  # 索引失败绝不影响报告本体
            pass
        return manifest
    except Exception as exc:
        try:
            (root / "ERROR.txt").write_text(
                f"报告生成失败: {type(exc).__name__}: {exc}\n"
                "正式目录未写入混合状态；.staging 保留现场供排查。\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        raise
