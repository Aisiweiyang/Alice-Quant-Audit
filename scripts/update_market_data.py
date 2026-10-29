#!/usr/bin/env python3
"""Create or incrementally update a versioned Alice market-data cache."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from data.cache_store import (
    build_metadata,
    load_market_cache,
    merge_market_data,
    save_market_cache,
    universe_fingerprint,
)
from data.market_providers import create_provider
from data.quality import (
    audit_market_data,
    completion_map,
    pending_downloads,
    save_quality_report,
)
from data.universe import (
    historical_main_board_codes,
    load_security_master,
    normalise_security_master,
    save_security_master,
)
from strategy.config import StrategyConfig as cfg


CURRENT_UNIVERSE_WARNING = "当前股票池用于历史区间时存在幸存者偏差；正式研究应传入历史时点股票池。"


def parse_args():
    parser = argparse.ArgumentParser(description="创建或增量更新 Alice 行情缓存")
    parser.add_argument("start", help="数据开始日期")
    parser.add_argument("end", help="数据结束日期")
    parser.add_argument("--cache", required=True, help="输出或待更新的pickle缓存")
    parser.add_argument("--adjust", default="2", choices=["1", "2", "3"])
    parser.add_argument(
        "--universe",
        choices=["historical-main", "current-liquid", "current-main", "file"],
        default="historical-main",
    )
    parser.add_argument("--universe-file", help="每行一个代码，或JSON代码列表")
    parser.add_argument("--top-n", type=int, default=cfg.top_volume_n)
    parser.add_argument(
        "--provider",
        choices=["baostock", "akshare-tencent", "akshare-eastmoney"],
        default="akshare-tencent",
        help="v4默认腾讯AKShare；BaoStock可用时可切回baostock",
    )
    parser.add_argument("--security-master", help="已有证券主数据CSV，可复用v2生成的文件")
    parser.add_argument("--batch-size", type=int, help="每批下载并保存的股票数")
    parser.add_argument(
        "--priority-universe",
        choices=["none", "hs300"],
        default="none",
        help="优先下载指定当前股票池；只调整顺序，不改变完整股票池",
    )
    parser.add_argument("--request-delay", type=float, default=0.8, help="AKShare股票请求间隔秒数")
    parser.add_argument("--retries", type=int, default=3, help="单个网络请求最大尝试次数")
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.90,
        help="正式数据质量验收的最低交易日覆盖率，默认0.90",
    )
    parser.add_argument(
        "--allow-slow-full-market",
        action="store_true",
        help="确认接受AKShare下载超过100只股票可能耗时数十到数百小时",
    )
    parser.add_argument(
        "--allow-mixed-source",
        action="store_true",
        help="允许向不同来源的旧缓存续写；默认拒绝，通常不建议使用",
    )
    parser.add_argument(
        "--refresh-universe", action="store_true", help="更新现有缓存的股票池；会重拉完整区间"
    )
    return parser.parse_args()


def _read_codes(path):
    if not path:
        raise ValueError("--universe file 需要 --universe-file")
    with open(path, "r", encoding="utf-8") as handle:
        if path.lower().endswith(".json"):
            values = json.load(handle)
        else:
            values = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    if not isinstance(values, list) or not values:
        raise ValueError("股票池文件必须包含非空代码列表")
    return sorted(set(str(code) for code in values))


def _load_master_file(path):
    return normalise_security_master(pd.read_csv(path, encoding="utf-8-sig"))


def _select_codes(args, start, end, provider):
    if args.universe == "file":
        return _read_codes(args.universe_file), [], None
    if args.universe == "historical-main":
        if args.security_master:
            master = _load_master_file(args.security_master)
        elif provider.supports_security_master:
            master = provider.security_master()
        else:
            raise RuntimeError(
                "AKShare模式无法重建含退市股票的历史股票池。请使用 "
                "--security-master 指向v2生成的 .security_master.csv，"
                "或改用 --universe file。"
            )
        codes = historical_main_board_codes(master, start, end)
        if not codes:
            raise RuntimeError("历史主板股票池为空")
        return codes, [], master
    if args.universe == "current-main":
        return provider.stock_list("main"), [CURRENT_UNIVERSE_WARNING], None
    codes = provider.liquid_universe(args.top_n)
    if not codes:
        raise RuntimeError("当前流动性股票池获取失败")
    return codes, [CURRENT_UNIVERSE_WARNING], None


def _batches(codes, batch_size):
    if batch_size <= 0:
        raise ValueError("batch-size必须大于0")
    values = list(codes)
    for offset in range(0, len(values), batch_size):
        yield values[offset : offset + batch_size]


def _ordered_codes(codes, priority_codes=None):
    priority = set(priority_codes or ())
    return sorted(set(codes), key=lambda code: (code not in priority, code))


def _group_pending_jobs(jobs, priority_codes=None):
    """Coalesce jobs to avoid one provider login per IPO or delisting date.

    Querying slightly beyond a security's listing lifetime is harmless: the
    provider simply returns no rows outside that lifetime. Full downloads can
    therefore share one broad window. Incremental head/tail repairs are grouped
    by month to limit duplicate rows while keeping login counts small.
    """
    grouped = {}
    for code, start, end, reason in jobs:
        job_start = pd.Timestamp(start).normalize()
        job_end = pd.Timestamp(end).normalize()
        if reason == "full":
            key = (reason, "all")
        elif reason == "head":
            key = (reason, job_end.strftime("%Y-%m"))
        elif reason == "tail":
            key = (reason, job_start.strftime("%Y-%m"))
        else:
            key = (
                reason,
                job_start.strftime("%Y-%m"),
                job_end.strftime("%Y-%m"),
            )
        bucket = grouped.setdefault(
            key,
            {
                "start": job_start,
                "end": job_end,
                "codes": [],
                "reason": reason,
            },
        )
        bucket["start"] = min(bucket["start"], job_start)
        bucket["end"] = max(bucket["end"], job_end)
        bucket["codes"].append(code)
    return [
        (
            item["start"],
            item["end"],
            item["reason"],
            _ordered_codes(item["codes"], priority_codes),
        )
        for _, item in sorted(grouped.items())
    ]


def _write_checkpoint(
    cache_path,
    data,
    *,
    adjust,
    universe_mode,
    source,
    warnings,
    requested_codes,
    status,
    security_master_info=None,
    requested_start=None,
    requested_end=None,
    security_master=None,
    quality_summary=None,
    completed_code_state=None,
):
    metadata = build_metadata(
        data,
        benchmark=cfg.benchmark_index,
        adjust=adjust,
        universe_mode=universe_mode,
        source=source,
        warnings=warnings,
    )
    metadata["universe"]["requested_count"] = len(requested_codes)
    metadata["universe"]["requested_codes_sha256"] = universe_fingerprint(requested_codes)
    metadata["universe"]["requested_codes"] = list(requested_codes)
    if completed_code_state is not None:
        completed_values = set(completed_code_state)
        complete = {code: code in completed_values for code in requested_codes}
    elif requested_start is not None and requested_end is not None:
        complete = completion_map(
            data,
            requested_codes,
            requested_start,
            requested_end,
            security_master,
        )
    else:
        complete = {code: code in data and not data[code].empty for code in requested_codes}
    completed = sum(complete.values())
    incomplete = [code for code, is_complete in complete.items() if not is_complete]
    metadata["download"] = {
        "status": status,
        "completed_codes": completed,
        "requested_codes": len(requested_codes),
        "incomplete_codes": incomplete[:100],
        "incomplete_count": len(incomplete),
    }
    metadata["requested_period"] = {
        "start": pd.Timestamp(requested_start).date().isoformat()
        if requested_start is not None
        else None,
        "end": pd.Timestamp(requested_end).date().isoformat()
        if requested_end is not None
        else None,
    }
    if quality_summary is not None:
        metadata["quality"] = quality_summary
    if security_master_info:
        metadata["security_master"] = security_master_info
    save_market_cache(cache_path, data, metadata)
    return metadata


def main():
    args = parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if start > end:
        raise SystemExit("start不能晚于end")
    if not 0 < args.minimum_coverage <= 1:
        raise SystemExit("minimum-coverage必须在0到1之间")
    provider = create_provider(
        args.provider,
        delay=args.request_delay,
        retries=args.retries,
    )
    batch_size = args.batch_size or (5 if args.provider.startswith("akshare") else 100)

    resolved_cache = os.path.abspath(os.path.expanduser(args.cache))
    existing = {}
    existing_metadata = None
    security_master = None
    if os.path.isfile(resolved_cache):
        existing, existing_metadata = load_market_cache(args.cache)

    if existing and not args.refresh_universe:
        saved_universe = (existing_metadata or {}).get("universe") or {}
        codes = saved_universe.get("requested_codes") or [
            code for code in existing if code != cfg.benchmark_index
        ]
        codes = sorted(set(codes))
        warnings = list((existing_metadata or {}).get("warnings") or [])
        universe_mode = saved_universe.get("mode", args.universe)
        cached_adjust = str((existing_metadata or {}).get("adjust", args.adjust))
        if cached_adjust != args.adjust:
            raise SystemExit(
                f"缓存复权方式是{cached_adjust}，本次参数是{args.adjust}；拒绝混合不同复权数据"
            )
        security_master = load_security_master(args.cache)
    else:
        codes, warnings, security_master = _select_codes(args, start, end, provider)
        universe_mode = args.universe
        if existing:
            existing = {
                code: frame
                for code, frame in existing.items()
                if code == cfg.benchmark_index or code in codes
            }

    warnings = list(dict.fromkeys(warnings + list(provider.warnings)))
    source = provider.name
    cached_source = str((existing_metadata or {}).get("source") or "")
    if existing and cached_source and cached_source != source:
        if not args.allow_mixed_source:
            raise SystemExit(
                f"缓存来源是 {cached_source}，本次来源是 {source}；拒绝混合不同数据口径。"
                "请换一个新的 --cache 路径，或确认风险后添加 --allow-mixed-source。"
            )
        warnings.append(f"缓存混合数据源: {cached_source} + {source}")
        source = f"{cached_source}+{source}"
    if args.provider.startswith("akshare") and len(codes) > 100 and not args.allow_slow_full_market:
        raise SystemExit(
            f"已选{len(codes)}只股票。AKShare腾讯按月分页，全区间可能运行数十到数百小时；"
            "建议先用 --universe file 做小批验证。确认仍要继续时添加 "
            "--allow-slow-full-market。"
        )
    master_info = None
    if security_master is not None:
        master_info = save_security_master(args.cache, security_master)
    elif existing_metadata:
        master_info = existing_metadata.get("security_master")

    merged = dict(existing)

    benchmark_jobs = pending_downloads(
        merged,
        [cfg.benchmark_index],
        start,
        end,
        grace_days=4,
    )
    for _, benchmark_start, benchmark_end, reason in benchmark_jobs:
        print(f"  基准指数{reason}补数: {benchmark_start.date()} → {benchmark_end.date()}")
        benchmark = provider.index(
            cfg.benchmark_index,
            benchmark_start.date().isoformat(),
            benchmark_end.date().isoformat(),
        )
        if benchmark is not None and not benchmark.empty:
            merged = merge_market_data(
                merged, {cfg.benchmark_index: benchmark}, cfg.benchmark_index
            )
    if cfg.benchmark_index not in merged:
        raise RuntimeError(f"无法获取基准指数 {cfg.benchmark_index}")

    priority_codes = set()
    if args.priority_universe == "hs300":
        try:
            priority_codes = set(provider.stock_list("hs300") or ())
        except Exception as exc:
            message = f"沪深300优先列表获取失败，继续按普通顺序下载: {exc}"
            warnings.append(message)
            print(f"  {message}")
        else:
            matched = len(priority_codes.intersection(codes))
            print(f"  优先队列: 当前沪深300成分股 {matched}只")

    benchmark_dates = merged[cfg.benchmark_index]["date"]
    jobs = pending_downloads(
        merged,
        codes,
        start,
        end,
        security_master,
        benchmark_dates=benchmark_dates,
    )
    incomplete_at_start = {code for code, *_rest in jobs}
    completed_code_state = set(codes) - incomplete_at_start
    groups = _group_pending_jobs(jobs, priority_codes=priority_codes)
    for group_number, (job_start, job_end, reason, group_codes) in enumerate(groups, start=1):
        for batch_number, batch in enumerate(_batches(group_codes, batch_size), start=1):
            print(
                f"  {reason}补数 {group_number}.{batch_number}: {len(batch)}只 "
                f"({job_start.date()} → {job_end.date()})"
            )
            updates = provider.batch(
                batch,
                job_start.date().isoformat(),
                job_end.date().isoformat(),
                adjust=args.adjust,
                verbose=True,
            )
            merged = merge_market_data(merged, updates, cfg.benchmark_index)
            batch_completion = completion_map(
                merged,
                batch,
                start,
                end,
                security_master,
            )
            completed_code_state.update(
                code for code, is_complete in batch_completion.items() if is_complete
            )
            _write_checkpoint(
                args.cache,
                merged,
                adjust=args.adjust,
                universe_mode=universe_mode,
                source=source,
                warnings=warnings,
                requested_codes=codes,
                status="partial",
                security_master_info=master_info,
                requested_start=start,
                requested_end=end,
                security_master=security_master,
                completed_code_state=completed_code_state,
            )

    completed = completion_map(merged, codes, start, end, security_master)
    completed_codes = sum(completed.values())
    final_status = "complete" if completed_codes == len(codes) else "incomplete"
    quality = audit_market_data(
        merged,
        benchmark=cfg.benchmark_index,
        start=start,
        end=end,
        requested_codes=codes,
        security_master=security_master,
        minimum_coverage=args.minimum_coverage,
    )
    quality_json, quality_csv = save_quality_report(args.cache, quality)
    metadata = _write_checkpoint(
        args.cache,
        merged,
        adjust=args.adjust,
        universe_mode=universe_mode,
        source=source,
        warnings=warnings,
        requested_codes=codes,
        status=final_status,
        security_master_info=master_info,
        requested_start=start,
        requested_end=end,
        security_master=security_master,
        quality_summary=quality["summary"],
    )
    print(f"行情缓存: {resolved_cache}")
    print(f"完成股票: {metadata['download']['completed_codes']}/{len(codes)}")
    print(f"数据说明: {resolved_cache}.metadata.json")
    print(f"质量报告: {quality_json}")
    print(f"逐股明细: {quality_csv}")
    if not quality["summary"]["formal_ready"]:
        print(
            f"质量验收未通过: {quality['summary']['failed_codes']}只股票存在覆盖或价格问题，"
            "请先查看质量报告，不要直接用于正式回测。"
        )


if __name__ == "__main__":
    main()
