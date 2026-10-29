#!/usr/bin/env python3
"""Alice Quant Audit 统一商业分析入口。

产品定位：A股量化策略可信度评估与研究报告生成工具。
仅供研究，不构成投资建议；不连接券商；不执行用户上传的任意代码。

示例：
  python commercial_mvp.py audit --cache local_data/market.pkl --start 2025-01-01 --end 2025-06-30
  python commercial_mvp.py full  --cache local_data/market.pkl --start 2025-01-01 --end 2025-12-31
  python commercial_mvp.py reports --list
"""

import argparse
import json
import os
import shutil
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from commercial.config import ParameterError, request_from_args
from commercial.pipeline import run_data_audit_only, run_full_analysis
from run_metadata import project_version


def add_common_arguments(parser):
    parser.add_argument(
        "--cache",
        required=True,
        help="行情数据：Alice pickle缓存、CSV目录或含code列的单个CSV",
    )
    parser.add_argument("--start", required=True, help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--candidates",
        help="可选：标准化候选名单(JSON/CSV)；提供后只回测名单内股票",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(PROJECT_DIR, "output", "commercial_reports"),
        help="报告根目录；每次分析创建独立子目录，不覆盖旧报告",
    )
    parser.add_argument("--report-id", help="可选：指定报告编号；默认自动生成唯一编号")
    parser.add_argument(
        "--strategy",
        default="classic_trend",
        help="内置策略预设：classic_trend 或 cross_sectional_alpha158",
    )
    parser.add_argument(
        "--strategy-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="截面Alpha158专属参数（可重复，仅--strategy cross_sectional_alpha158时可用）："
        "minimum_score/exit_rank/minimum_holding_days/market_ma_period/"
        "liquid_universe_size/minimum_history",
    )
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--max-position-pct", type=float, default=0.12)
    parser.add_argument(
        "--rebalance-freq", default="weekly", choices=["daily", "weekly", "monthly"]
    )
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--stop-loss", type=float, default=0.05)
    parser.add_argument("--take-profit", type=float, default=0.10)
    parser.add_argument("--no-market-timing", action="store_true", help="关闭大盘择时门禁")
    parser.add_argument("--no-stress", action="store_true", help="跳过压力测试")
    parser.add_argument(
        "--no-segment",
        action="store_true",
        help="跳过前后分段一致性检查（数据规模大时可加速；默认开启）",
    )
    parser.add_argument(
        "--no-factor-diagnostics", action="store_true", help="跳过因子诊断（数据规模大时可加速）"
    )
    parser.add_argument("--no-rolling-oos", action="store_true", help="跳过多窗口滚动样本外验证")
    parser.add_argument(
        "--rolling-windows", type=int, default=3, help="滚动样本外窗口数(2-4，默认3)"
    )
    parser.add_argument(
        "--rolling-anchored",
        action="store_true",
        help="滚动样本外用锚定窗口（起点固定为区间起点、终点递进）；缺省不重叠等分",
    )
    parser.add_argument("--max-scenarios", type=int, default=8, help="压力场景数量上限(≤8)")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="本轮尝试的策略方案数N（Deflated Sharpe多重检验扣除；缺省1=退化为PSR）",
    )
    parser.add_argument(
        "--allow-incomplete-data",
        action="store_true",
        help="允许缺少历史ST/停牌字段的缓存继续回测（结果标记为LIMITED）",
    )
    parser.add_argument("--brand-name", default="", help="可选：报告页眉品牌名(≤60字)")
    parser.add_argument("--brand-footer", default="", help="可选：报告页脚品牌说明(≤120字)")
    parser.add_argument("--quiet", action="store_true", help="减少进度输出")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(
        prog="commercial_mvp.py",
        description="Alice量化策略审计台 · A股量化策略可信度评估与研究报告生成",
    )
    parser.add_argument("--version", action="version", version=project_version())
    subparsers = parser.add_subparsers(dest="command")

    audit_parser = subparsers.add_parser(
        "audit",
        help="只做数据可信度检查，输出READY/LIMITED/NOT_READY评级报告",
    )
    add_common_arguments(audit_parser)

    full_parser = subparsers.add_parser(
        "full",
        help="完整分析：数据审计+可信回测+压力测试+可信度评分+HTML报告",
    )
    add_common_arguments(full_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="批量策略对比：按用户显式提供的计划文件(2-8个方案)分别回测",
    )
    add_common_arguments(compare_parser)
    compare_parser.add_argument(
        "--plan",
        required=True,
        help='对比计划JSON：{"plans":[{"label":..., "overrides":{...}}, ...]}，最多8个',
    )

    export_parser = subparsers.add_parser(
        "export",
        help="把指定报告目录打包为zip（只读原报告）",
    )
    export_parser.add_argument("report_id", help="要导出的report_id")
    export_parser.add_argument("--zip", dest="zip_path", help="输出zip路径；默认 <report_id>.zip")
    export_parser.add_argument(
        "--output-root",
        default=os.path.join(PROJECT_DIR, "output", "commercial_reports"),
    )

    reports_parser = subparsers.add_parser(
        "reports",
        help="查看历史报告（只读，不修改任何报告）",
    )
    reports_parser.add_argument(
        "--list",
        action="store_true",
        help="列出output/commercial_reports下的历史报告",
    )
    reports_parser.add_argument("--json", action="store_true", help="以JSON输出")
    reports_parser.add_argument(
        "--trend",
        action="store_true",
        help="输出评分/数据评级随时间趋势表（时间正序，只读历史报告）",
    )
    reports_parser.add_argument(
        "--windows",
        action="store_true",
        help="只读输出最近一次多窗口扫描汇总（output/realdata_runs/下的"
        "multi_window_scan_*.md，由scripts/multi_window_scan.py生成）",
    )
    reports_parser.add_argument(
        "--compare",
        nargs="+",
        metavar="REPORT_ID",
        help="只读对比2-8份历史报告的自报指标（如 --compare id1 id2）",
    )
    reports_parser.add_argument(
        "--prune",
        action="store_true",
        help="归档旧报告到_archive/（zip+原目录，绝不删除）；缺省仅预览",
    )
    reports_parser.add_argument(
        "--keep",
        type=int,
        default=20,
        help="prune时保留的最新报告份数（默认20）",
    )
    reports_parser.add_argument(
        "--archive",
        action="store_true",
        help="与--prune联用：实际执行移动；缺省仅预览",
    )
    reports_parser.add_argument("--limit", type=int, default=50, help="最多列出N条(最新优先)")
    reports_parser.add_argument(
        "--output-root",
        default=os.path.join(PROJECT_DIR, "output", "commercial_reports"),
    )
    return parser


ARCHIVE_DIR_NAME = "_archive"


def collect_reports(root: str) -> list[dict]:
    """只读收集历史报告摘要（report_id、时间、评分、评级、阻断、滚动结论）。"""
    rows = []
    if not os.path.isdir(root):
        return rows
    for name in sorted(os.listdir(root), reverse=True):
        if name == ARCHIVE_DIR_NAME or name.startswith(ARCHIVE_DIR_NAME + os.sep):
            continue  # 归档目录不进入历史列表/趋势/对比
        manifest_path = os.path.join(root, name, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        score_value = None
        score_state = "missing"
        grade_label = None
        audit_path = os.path.join(root, name, "strategy_audit.json")
        if os.path.isfile(audit_path):
            try:
                with open(audit_path, "r", encoding="utf-8") as handle:
                    audit_payload = json.load(handle)
                score_value = audit_payload.get("score")
                grade_label = audit_payload.get("grade_label")
                score_state = "none" if score_value is None else "ok"
            except (json.JSONDecodeError, OSError):
                score_state = "error"
        rolling_verdict = None
        rolling_path = os.path.join(root, name, "rolling_oos.json")
        if os.path.isfile(rolling_path):
            try:
                with open(rolling_path, "r", encoding="utf-8") as handle:
                    rolling_verdict = json.load(handle).get("verdict")
            except (json.JSONDecodeError, OSError):
                rolling_verdict = None
        # 过拟合审计证据存在性：compare报告的metrics.json含overfitting_audit，
        # full报告含deflated_sharpe(DSR)字段；便于跨报告找证据，不解读数值。
        has_overfitting_audit = False
        metrics_path = os.path.join(root, name, "metrics.json")
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as handle:
                    metrics_payload = json.load(handle)
                has_overfitting_audit = isinstance(metrics_payload, dict) and (
                    "overfitting_audit" in metrics_payload or "deflated_sharpe" in metrics_payload
                )
            except (json.JSONDecodeError, OSError):
                has_overfitting_audit = False
        rows.append(
            {
                "report_id": name,
                "created_at": manifest.get("created_at", ""),
                "score": score_value,
                "score_state": score_state,
                "grade_label": grade_label,
                "rating": _rating_of(root, name),
                "blocked": manifest.get("blocked"),
                "rolling_verdict": rolling_verdict,
                "has_overfitting_audit": has_overfitting_audit,
                "strategy": (manifest.get("parameters") or {}).get("strategy"),
                "period": f"{(manifest.get('parameters') or {}).get('start', '?')}"
                f"~{(manifest.get('parameters') or {}).get('end', '?')}",
            }
        )
    return rows


def _report_detail(root: str, report_id: str) -> dict | None:
    """只读读取单份报告的自报指标（manifest/metrics/audit），无则返回None。"""
    base = os.path.join(root, report_id)
    manifest_path = os.path.join(base, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    detail = {"report_id": report_id}
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        params = manifest.get("parameters") or {}
        detail.update(
            {
                "created_at": manifest.get("created_at", ""),
                "strategy": params.get("strategy"),
                "period": f"{params.get('start', '?')}~{params.get('end', '?')}",
                "blocked": manifest.get("blocked"),
            }
        )
    except (json.JSONDecodeError, OSError):
        return None
    metrics_path = os.path.join(base, "metrics.json")
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except (json.JSONDecodeError, OSError):
            metrics = {}
        for key in (
            "total_return_pct",
            "max_drawdown_pct",
            "sharpe_ratio",
            "sharpe_reliable",
            "sortino_ratio",
            "calmar_ratio",
            "total_trades",
            "sharpe_ci_low",
            "sharpe_ci_high",
        ):
            detail[key] = metrics.get(key)
    audit_path = os.path.join(base, "strategy_audit.json")
    if os.path.isfile(audit_path):
        try:
            with open(audit_path, "r", encoding="utf-8") as handle:
                audit = json.load(handle)
            detail["score"] = audit.get("score")
            detail["grade_label"] = audit.get("grade_label")
        except (json.JSONDecodeError, OSError):
            pass
    detail["rating"] = _rating_of(root, report_id)
    return detail


def collect_report_details(root: str, report_ids: list[str]) -> list[dict]:
    """按用户给定顺序收集2-8份报告的自报指标；未知id以占位行如实呈现。"""
    details = []
    for report_id in report_ids[:8]:
        found = _report_detail(root, report_id)
        details.append(found or {"report_id": report_id, "missing": True})
    return details


def _print_report_compare(details: list[dict]) -> None:
    headers = (
        "报告编号",
        "评分",
        "数据评级",
        "收益%",
        "回撤%",
        "Sharpe",
        "Sortino",
        "Calmar",
        "交易",
        "策略",
        "区间",
    )
    rows = []
    for item in details:
        if item.get("missing"):
            rows.append([item["report_id"], "—", "报告不存在"] + ["—"] * 8)
            continue
        rows.append(
            [
                item["report_id"],
                str(item.get("score")) if item.get("score") is not None else "—",
                str(item.get("rating") or "—"),
                _num_or_dash(item.get("total_return_pct")),
                _num_or_dash(item.get("max_drawdown_pct")),
                _num_or_dash(item.get("sharpe_ratio")),
                _num_or_dash(item.get("sortino_ratio")),
                _num_or_dash(item.get("calmar_ratio")),
                str(item.get("total_trades") if item.get("total_trades") is not None else "—"),
                str(item.get("strategy") or "—"),
                str(item.get("period") or "—"),
            ]
        )
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print(
        "\n说明：只读汇总各报告的自报指标，不做统一重放；不同数据/参数的报告"
        "横向对比仅供参考。评分评价研究流程可信度，不评价未来收益。"
    )


def _num_or_dash(value) -> str:
    return "—" if value is None else str(value)


def _prune_reports(root: str, keep: int, execute: bool) -> int:
    """归档旧报告：保留最新keep份，其余zip后移入_archive/。

    绝不删除文件；不配合--archive时仅预览。
    """
    rows = collect_reports(root)  # 最新在前，不含_archive
    targets = rows[max(0, int(keep)) :]
    if not targets:
        print(f"报告总数不超过{keep}份，无需归档。")
        return 0
    archive_dir = os.path.join(root, ARCHIVE_DIR_NAME)
    mode = "归档" if execute else "预览（加 --archive 实际执行）"
    print(f"{mode}：共{len(targets)}份将移入_archive/（保留最新{keep}份）：")
    for row in targets:
        report_id = row["report_id"]
        print(
            f"  {report_id}  评分={row['score'] if row['score'] is not None else '—'}"
            f"  {row['created_at']}"
        )
        if not execute:
            continue
        os.makedirs(archive_dir, exist_ok=True)
        from commercial.report import export_report_zip

        zip_path = os.path.join(archive_dir, f"{report_id}.zip")
        export_report_zip(root, report_id, zip_path)
        shutil.move(os.path.join(root, report_id), os.path.join(archive_dir, report_id))
    if execute:
        print(f"已归档{len(targets)}份（zip+原目录）到 {archive_dir}。")
    return 0


def cmd_reports(args) -> int:
    root = os.path.abspath(os.path.expanduser(args.output_root))
    if getattr(args, "prune", False):
        return _prune_reports(
            root, keep=int(getattr(args, "keep", 20)), execute=bool(getattr(args, "archive", False))
        )
    if getattr(args, "windows", False):
        return _print_windows_scan()
    compare_ids = getattr(args, "compare", None)
    if compare_ids:
        if not (2 <= len(compare_ids) <= 8):
            print("对比需要2-8个report_id")
            return 2
        details = collect_report_details(root, compare_ids)
        if args.json:
            print(json.dumps(details, ensure_ascii=False, indent=2))
        else:
            _print_report_compare(details)
        return 0
    rows = collect_reports(root)
    if not rows:
        print(f"{root} 下没有有效报告")
        return 0
    rows = rows[: max(1, int(args.limit))]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if args.trend:
        return _print_trend(rows)
    for row in rows:
        score = "无法评分" if row["score"] is None else str(row["score"])
        rolling = f"  滚动={row['rolling_verdict']}" if row["rolling_verdict"] else ""
        overfit = "  过拟合审计=有" if row.get("has_overfitting_audit") else ""
        print(
            f"{row['report_id']}  {row['created_at']}  评分={score}  "
            f"数据评级={row['rating']}  阻断={row['blocked']}"
            f"  区间={row['period']}{rolling}{overfit}"
        )
    return 0


def _print_windows_scan(scan_dir: str | None = None) -> int:
    """只读打印最近一次多窗口扫描汇总；不修改任何文件。"""
    scan_dir = scan_dir or os.path.join(PROJECT_DIR, "output", "realdata_runs")
    candidates = (
        sorted(
            (
                name
                for name in os.listdir(scan_dir)
                if name.startswith("multi_window_scan_") and name.endswith(".md")
            ),
            reverse=True,
        )
        if os.path.isdir(scan_dir)
        else []
    )
    if not candidates:
        print("还没有多窗口扫描汇总；先用 scripts/multi_window_scan.py 生成")
        return 0
    path = os.path.join(scan_dir, candidates[0])
    with open(path, encoding="utf-8") as handle:
        print(handle.read(), end="")
    print(f"\n[来源：{path}]")
    return 0


def _print_trend(rows: list[dict]) -> int:
    """评分/数据评级随时间趋势（时间正序；只读历史manifest，不做任何比较运行）。"""
    ordered = sorted(rows, key=lambda r: r.get("created_at") or "")
    scores = [r["score"] for r in ordered if isinstance(r.get("score"), (int, float))]
    print("报告评分/数据评级趋势（旧 → 新）")
    print(f"{'时间':<21} {'评分':>4}  {'等级':<10} {'数据评级':<8} 阻断  报告编号")
    for row in ordered:
        score = "—" if row["score"] is None else str(row["score"])
        grade = str(row.get("grade_label") or "—")
        print(
            f"{str(row['created_at']):<21} {score:>4}  {grade:<10} "
            f"{str(row['rating']):<8} {'是' if row['blocked'] else '否':<4} "
            f"{row['report_id']}"
        )
    if len(scores) >= 2:
        direction = (
            "上升" if scores[-1] > scores[0] else ("下降" if scores[-1] < scores[0] else "持平")
        )
        print(
            f"\n共{len(ordered)}份报告，可评分{len(scores)}份；"
            f"评分从{scores[0]}到{scores[-1]}（{direction}）。"
            "评分衡量研究流程可信度，不评价未来收益。"
        )
    else:
        print(f"\n共{len(ordered)}份报告，可评分不足2份，暂无趋势可言。")
    return 0


def _rating_of(root, name):
    audit_path = os.path.join(root, name, "data_quality.json")
    if os.path.isfile(audit_path):
        with open(audit_path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("rating", "?")
    return "?"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    if args.command == "export":
        from commercial.report import export_report_zip

        try:
            zip_path = export_report_zip(args.output_root, args.report_id, args.zip_path)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"已导出: {zip_path}")
        return 0
    if args.command == "reports":
        return cmd_reports(args)
    try:
        request = request_from_args(args)
    except ParameterError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2
    try:
        if args.command == "audit":
            analysis = run_data_audit_only(request, getattr(args, "report_id", None))
        elif args.command == "compare":
            from commercial.compare import run_compare

            analysis = run_compare(request, args.plan, getattr(args, "report_id", None))
        else:
            analysis = run_full_analysis(request, getattr(args, "report_id", None))
    except ParameterError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2
    print(f"\n报告目录: {analysis['report_dir']}")
    print(f"报告编号: {analysis['report_id']}")
    print(f"数据评级: {analysis['data_audit']['rating']}")
    if analysis.get("mode") == "compare":
        results = analysis.get("plan_results") or []
        ok = [r for r in results if r.get("status") == "ok"]
        print(f"对比模式: {len(ok)}/{len(results)}个方案完成")
        if ok:
            best = max(ok, key=lambda r: r["metrics"].get("total_return_pct", -1e9))
            print(f"区间收益最高: {best['label']} ({best['metrics'].get('total_return_pct')}%)")
    if analysis.get("blocked"):
        label = "模式" if analysis.get("audit_only") else "状态: 阻断"
        print(f"{label}: {analysis['block_message']}")
    score = analysis.get("score") or {}
    print(
        f"可信度评分: {score.get('score') if score.get('score') is not None else '无法评分'}"
        f"（{score.get('grade_label', '')}）"
    )
    print(f"清单: {os.path.join(analysis['report_dir'], 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
