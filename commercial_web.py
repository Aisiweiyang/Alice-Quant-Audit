#!/usr/bin/env python3
"""Alice Quant Audit 本地网页入口（完全本地，无外部依赖）。

- 只绑定 127.0.0.1，不监听外网；无登录、无支付、无账号体系；
- 不执行任何用户代码；只调用与命令行相同的白名单分析管线；
- 仅供研究，不构成投资建议。

启动：
  python commercial_web.py            # http://127.0.0.1:8765
  python commercial_web.py --port 9000
"""

import argparse
import json
import mimetypes
import os
import sys
import threading
import uuid
from datetime import datetime
from html import escape as esc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from commercial.config import ParameterError, CommercialRequest
from commercial.pipeline import run_data_audit_only, run_full_analysis
from commercial_mvp import collect_reports

DISCLAIMER = "仅供研究，不构成投资建议。不承诺收益，不连接券商，不执行任何用户代码。"

JOBS = {}
JOBS_LOCK = threading.Lock()
REPORTS_ROOT = os.path.join(PROJECT_DIR, "output", "commercial_reports")
# 运行时状态与报告产物分离；web_jobs.json 曾放在报告目录，保留旧路径兼容读取。
STATE_DIR = os.path.join(PROJECT_DIR, "output", "state")
LEGACY_JOBS_FILE = os.path.join(REPORTS_ROOT, "web_jobs.json")
JOBS_FILE = os.path.join(STATE_DIR, "web_jobs.json")


def _persist_jobs():
    """任务历史写入STATE_DIR/web_jobs.json；失败不影响分析服务。"""
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with JOBS_LOCK:
            snapshot = {k: dict(v) for k, v in JOBS.items()}
        with open(JOBS_FILE, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=1)
    except OSError:
        pass


def _load_jobs():
    """优先读取新状态路径；不存在时从旧报告目录迁移（成功后移除旧文件）。"""
    source = JOBS_FILE
    if not os.path.isfile(source) and os.path.isfile(LEGACY_JOBS_FILE):
        source = LEGACY_JOBS_FILE
    try:
        with open(source, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        if isinstance(stored, dict):
            JOBS.update(stored)
    except (OSError, json.JSONDecodeError):
        return
    if source == LEGACY_JOBS_FILE:
        _persist_jobs()
        if os.path.isfile(JOBS_FILE):
            try:
                os.remove(LEGACY_JOBS_FILE)
            except OSError:
                pass


def create_job(kind: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "steps": [],
            "report_dir": None,
            "report_id": None,
            "error": None,
        }
    return job_id


def job_progress(job_id: str):
    def progress(message: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["steps"].append(message)
                job["status"] = "running"
        _persist_jobs()

    return progress


def _overfitting_summary(analysis: dict) -> dict:
    """从compare分析结果提取过拟合审计摘要（只存数字与标签，不存大对象）。

    PBO与DSR都只描述这组方案的历史统计证据强度，不是对未来收益的预测。
    """
    audit = analysis.get("overfitting_audit") or {}
    pbo = audit.get("pbo") or {}
    plans = []
    for result in analysis.get("plan_results") or []:
        if result.get("status") != "ok":
            continue
        dsr = result.get("deflated_sharpe") or {}
        plans.append(
            {
                "label": result.get("label"),
                "dsr": dsr.get("deflated_sharpe"),
                "sr_observed": dsr.get("sr_observed"),
                "dsr_status": dsr.get("status"),
            }
        )
    degradation = pbo.get("degradation_regression") or {}
    return {
        "n_trials": audit.get("n_trials"),
        "pbo_status": pbo.get("status"),
        "pbo_probability": pbo.get("pbo_probability"),
        "cscv_combinations": pbo.get("cscv_combinations"),
        "pbo_detail": pbo.get("detail"),
        "degradation_slope": degradation.get("slope"),
        "degradation_intercept": degradation.get("intercept"),
        "degradation_pairs": degradation.get("pairs"),
        "dispersion_validity": (audit.get("dispersion") or {}).get("validity"),
        "dispersion_detail": (audit.get("dispersion") or {}).get("detail"),
        "plans": plans,
    }


def run_job(
    job_id: str, request: CommercialRequest, mode: str, plan_path: str | None = None
) -> None:
    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
        if mode == "audit":
            analysis = run_data_audit_only(request, progress=job_progress(job_id))
        elif mode == "compare":
            from commercial.compare import run_compare

            analysis = run_compare(request, plan_path, progress=job_progress(job_id))
        else:
            analysis = run_full_analysis(request, progress=job_progress(job_id))
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "done"
            job["report_dir"] = analysis.get("report_dir")
            job["report_id"] = analysis.get("report_id")
            score = analysis.get("score") or {}
            job["score"] = score.get("score")
            job["grade"] = score.get("grade_label")
            job["rating"] = (analysis.get("data_audit") or {}).get("rating")
            job["blocked"] = analysis.get("blocked")
            if analysis.get("mode") == "compare":
                ok = [r for r in analysis.get("plan_results", []) if r.get("status") == "ok"]
                job["compare_summary"] = (
                    f"{len(ok)}/{len(analysis.get('plan_results', []))}个方案完成"
                )
                job["overfitting"] = _overfitting_summary(analysis)
        _persist_jobs()
    except SystemExit as exc:
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "error"
            job["error"] = str(exc)
        _persist_jobs()
    except Exception as exc:  # 任何异常都回报到页面，不中断服务
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
        _persist_jobs()


PAGE_STYLE = """
:root{--bg:#f4f7fb;--ink:#172033;--muted:#64748b;--line:#e2e8f0;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Microsoft YaHei",system-ui,sans-serif}
main{max-width:880px;margin:auto;padding:28px 18px 60px}
h1{font-size:26px;margin:0 0 6px}.muted{color:var(--muted);font-size:13px}
.notice{border-left:4px solid #f59e0b;background:#fff7ed;padding:11px 14px;
border-radius:8px;margin:14px 0;font-size:13px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px;margin-top:14px}
label{display:block;margin:10px 0 4px;font-size:14px}
input[type=text],select{width:100%;padding:8px;border:1px solid var(--line);
border-radius:8px;font-size:14px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
button{margin-top:16px;background:#2563eb;color:#fff;border:0;border-radius:9px;
padding:11px 22px;font-size:15px;cursor:pointer}
ul{padding-left:18px}li{font-size:13px;margin:3px 0}
a{color:#2563eb}
table{width:100%;border-collapse:collapse}th,td{padding:8px 6px;font-size:13px;
border-bottom:1px solid var(--line);text-align:left}
.badge{padding:2px 10px;border-radius:999px;font-size:12px;background:#e2e8f0}
@media(max-width:640px){.row{grid-template-columns:1fr}}
"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>{PAGE_STYLE}</style></head><body><main>
<h1>Alice量化策略审计台 · 本地工作台</h1>
<p class="muted">{DISCLAIMER}</p>
<div class="notice">本页面仅运行在本机(127.0.0.1)。数据保存在你自己的电脑上，
不连接券商。仅加载自己生成或完全信任的 .pkl；交换数据优先使用 CSV。</div>
{body}
</main></body></html>"""


def render_home() -> str:
    body = """
<div class="panel">
<form method="post" action="/run">
<h3>新建分析</h3>
<label>行情数据路径（Alice .pkl 缓存、CSV目录或含code列的CSV）</label>
<input type="text" name="cache" required
       placeholder="examples\\commercial_demo\\demo_cache.pkl">
<div class="row">
<div><label>开始日期</label><input type="text" name="start" required
       placeholder="2025-07-01" value="2025-07-01"></div>
<div><label>结束日期</label><input type="text" name="end" required
       placeholder="2025-12-31" value="2025-12-31"></div>
</div>
<div class="row">
<div><label>初始资金</label><input type="text" name="initial_capital" value="100000"></div>
<div><label>最大持仓数</label><input type="text" name="max_positions" value="8"></div>
</div>
<div class="row">
<div><label>调仓频率</label><select name="rebalance_freq">
<option value="weekly">每周</option><option value="daily">每日</option>
<option value="monthly">每月</option></select></div>
<div><label>分析模式</label><select name="mode">
<option value="full">完整分析（审计+回测+压力+因子+评分+报告）</option>
<option value="audit">仅数据审计</option>
<option value="compare">批量策略对比（需填方案文件路径）</option></select></div>
</div>
<div class="row">
<div><label>策略预设</label><select name="strategy">
<option value="classic_trend">classic_trend（经典趋势评分）</option>
<option value="cross_sectional_alpha158">cross_sectional_alpha158（截面Alpha158排名）</option>
</select></div>
<div><label>方案尝试数 trials（1-1000；DSR多重检验扣除用，1=退化为PSR）</label>
<input type="text" name="trials" value="1" inputmode="numeric"></div>
</div>
<label>截面策略参数（可选，仅cross_sectional_alpha158生效；格式 KEY=VALUE，多组逗号分隔）</label>
<input type="text" name="strategy_params"
       placeholder="minimum_score=60,exit_rank=15">
<label>方案文件路径（compare模式必填，JSON格式 plans 数组）</label>
<input type="text" name="plan_path" placeholder="D:/plans/my_compare.json">
<label><input type="checkbox" name="no_stress" value="1"> 跳过压力测试</label>
<label><input type="checkbox" name="no_segment" value="1"> 跳过前后分段一致性检查（数据规模大时可加速）</label>
<label><input type="checkbox" name="no_factor" value="1"> 跳过因子诊断</label>
<label><input type="checkbox" name="no_rolling" value="1"> 跳过滚动样本外</label>
<label><input type="checkbox" name="rolling_anchored" value="1"> 滚动样本外用锚定窗口（起点固定、终点递进）</label>
<button type="submit">开始分析</button>
</form></div>
<div class="panel"><h3>历史报告</h3>
<p class="muted"><a href="/reports">查看全部历史报告 →</a> ·
<a href="/reports/trend">评分趋势 →</a> ·
<a href="/reports/compare">跨报告对比 →</a> ·
<a href="/experiments">实验索引 →</a> ·
<a href="/validate">compare方案校验（只读） →</a></p></div>
"""
    return _page("Alice量化策略审计台", body)


def _render_overfitting_panel(overfitting: dict) -> str:
    """compare任务完成后的过拟合审计摘要面板：PBO值/组合数/各方案DSR。"""
    if not overfitting:
        return ""
    pbo_status = overfitting.get("pbo_status")
    if pbo_status == "ok" and overfitting.get("pbo_probability") is not None:
        pbo_text = (
            f"PBO={overfitting.get('pbo_probability')}"
            f"（CSCV组合数{overfitting.get('cscv_combinations')}）"
        )
        if overfitting.get("dispersion_validity") == "insufficient_dispersion":
            pbo_text += " ⚠ 方案组缺乏分散度，PBO无统计意义"
    else:
        pbo_text = f"PBO：数据不足（{overfitting.get('pbo_detail') or '方案数或时段不足'}）"
    rows = (
        "".join(
            f"<tr><td>{esc(str(plan.get('label') or ''))}</td>"
            f"<td>{'—' if plan.get('dsr') is None else plan.get('dsr')}</td>"
            f"<td>{'—' if plan.get('sr_observed') is None else plan.get('sr_observed')}</td></tr>"
            for plan in overfitting.get("plans") or []
        )
        or "<tr><td colspan='3'>没有完成回测的方案</td></tr>"
    )
    dispersion_line = ""
    if overfitting.get("dispersion_detail"):
        if overfitting.get("dispersion_validity") == "insufficient_dispersion":
            # 不足：红色警示（琥珀色notice默认样式留给中性/不足说明）
            dispersion_line = (
                "<div class='notice' style='border-color:#dc2626;background:#fef2f2'>"
                f"⚠ {esc(str(overfitting.get('dispersion_detail')))}</div>"
            )
        else:
            dispersion_line = (
                f"<p class='muted'>{esc(str(overfitting.get('dispersion_detail')))}</p>"
            )
    degradation_line = ""
    if overfitting.get("degradation_slope") is not None:
        degradation_line = (
            "<p class='muted'>性能退化回归：样本外Sharpe≈"
            f"{overfitting.get('degradation_slope')}×样本内Sharpe+"
            f"{overfitting.get('degradation_intercept')}"
            f"（{overfitting.get('degradation_pairs')}个组合；"
            "斜率明显小于1提示样本内优势在样本外大幅缩水）。</p>"
        )
    return (
        f"<div class='panel'><h3>过拟合审计摘要（{overfitting.get('n_trials')}个方案）</h3>"
        f"<p>{pbo_text}</p>{dispersion_line}"
        f"<table><thead><tr><th>方案</th><th>Deflated Sharpe（0-1概率）</th>"
        f"<th>观测Sharpe（日频）</th></tr></thead><tbody>{rows}</tbody></table>"
        f"{degradation_line}"
        f"<p class='muted'>PBO衡量这组方案整体过拟合概率，DSR是扣除多重检验后"
        f"Sharpe仍显著的概率；两者都只描述历史统计证据，不是对未来收益的预测。"
        f"PBO随回测窗口可能大幅波动，单窗口数字不可外推。</p></div>"
    )


def render_status(job_id: str) -> str:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return _page("任务不存在", "<div class='panel'>任务不存在或服务已重启。</div>")
    steps = "".join(f"<li>{str(s)}</li>" for s in job["steps"][-30:])
    status = job["status"]
    extra = ""
    if status == "done":
        report_link = f"/report/{job['report_id']}/index.html"
        extra = (
            f"<div class='panel'><h3>分析完成</h3>"
            f"<p>评分：<b>{job.get('score')}</b>（{job.get('grade')}）· "
            f"数据评级：<b>{job.get('rating')}</b>"
            f"{' · 已阻断' if job.get('blocked') else ''}</p>"
            f"<p><a href='{report_link}'>打开HTML报告 →</a> "
            f"<a href='/reports'>历史报告 →</a></p></div>"
            f"{_render_overfitting_panel(job.get('overfitting') or {})}"
        )
    elif status == "error":
        extra = f"<div class='panel'><h3>运行失败</h3><p>{job.get('error')}</p></div>"
    refresh = "" if status in {"done", "error"} else '<meta http-equiv="refresh" content="2">'
    body = (
        f"<div class='panel'><h3>任务 {job['id']} · 状态：{status}</h3>"
        f"<ul>{steps or '<li>排队中…</li>'}</ul></div>{extra}"
        f"<p><a href='/'>← 返回首页</a></p>"
    )
    return (
        f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>{refresh}"
        f"<title>任务状态</title><style>{PAGE_STYLE}</style></head><body><main>"
        f"<h1>Alice量化策略审计台</h1><p class='muted'>{DISCLAIMER}</p>{body}"
        f"</main></body></html>"
    )


def render_experiments(strategy: str = "", rating: str = "") -> str:
    """实验索引只读页：渲染experiments_index.json摘要表（新→旧）。

    支持?strategy=与?rating=查询参数做只读过滤（不修改索引文件）；
    过滤生效时页面显示当前条件与清除链接（P1-5）。
    """
    path = os.path.join(REPORTS_ROOT, "experiments_index.json")
    strategy_key = strategy.strip().lower()
    rating_key = rating.strip().lower()
    filter_banner = ""
    if strategy_key or rating_key:
        parts = []
        if strategy_key:
            parts.append(f"策略={esc(strategy.strip())}")
        if rating_key:
            parts.append(f"数据评级={esc(rating.strip())}")
        filter_banner = (
            f"<p class='muted'>当前过滤：{' · '.join(parts)}"
            f"（<a href='/experiments'>清除过滤</a>）</p>"
        )
    body_rows = ""
    if not os.path.isfile(path):
        body_rows = "<tr><td colspan='8'>还没有记录（实验索引在报告写入时自动生成）</td></tr>"
    else:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
            if not isinstance(records, list):
                raise ValueError("索引格式异常")
        except (OSError, json.JSONDecodeError, ValueError):
            body_rows = (
                "<tr><td colspan='8'>索引文件损坏或格式异常；"
                "按约定不覆盖历史，请人工检查该文件</td></tr>"
            )
        else:
            shown = 0
            for entry in reversed(records):
                if strategy_key and str(entry.get("strategy") or "").lower() != strategy_key:
                    continue
                if rating_key and str(entry.get("data_rating") or "").lower() != rating_key:
                    continue
                shown += 1
                metrics = entry.get("key_metrics") or {}
                report_id = str(entry.get("report_id") or "")
                link = (
                    f"<a href='/report/{esc(report_id)}/index.html'>{esc(report_id)}</a>"
                    if report_id
                    else "—"
                )
                body_rows += (
                    f"<tr><td>{esc(str(entry.get('created_at') or ''))}</td>"
                    f"<td>{link}</td>"
                    f"<td>{esc(str(entry.get('mode') or '—'))}</td>"
                    f"<td>{esc(str(entry.get('strategy') or '—'))}</td>"
                    f"<td>{esc(str(entry.get('start') or ''))}~{esc(str(entry.get('end') or ''))}</td>"
                    f"<td>{'—' if entry.get('score') is None else entry.get('score')}</td>"
                    f"<td>{esc(str(entry.get('data_rating') or '—'))}</td>"
                    f"<td>{'—' if metrics.get('total_return_pct') is None else metrics.get('total_return_pct')}</td></tr>"
                )
            if shown == 0 and (strategy_key or rating_key):
                body_rows = "<tr><td colspan='8'>没有符合当前过滤条件的记录</td></tr>"
    body = (
        f"<div class='panel'><h3>实验索引（追加式，新→旧）</h3>{filter_banner}"
        f"<table><thead><tr><th>时间</th><th>报告编号</th><th>模式</th>"
        f"<th>策略</th><th>区间</th><th>评分</th><th>数据评级</th>"
        f"<th>收益%</th></tr></thead><tbody>{body_rows}</tbody></table>"
        f"<p class='muted'>索引只追加不删除；评分衡量研究流程可信度，"
        f"不评价未来收益。仅供研究，不构成投资建议。</p></div>"
        f"<p><a href='/reports'>← 返回历史报告</a></p>"
    )
    return _page("实验索引", body)


def render_reports() -> str:
    rows = collect_reports(REPORTS_ROOT)[:100]
    items = (
        "".join(
            f"<tr><td><a href='/report/{r['report_id']}/index.html'>{r['report_id']}</a></td>"
            f"<td>{r['created_at']}</td>"
            f"<td>{'—' if r['score'] is None else r['score']}</td>"
            f"<td>{r['rating']}</td><td>{'是' if r['blocked'] else '否'}</td>"
            f"<td><a href='/export/{r['report_id']}'>下载zip</a></td></tr>"
            for r in rows
        )
        or "<tr><td colspan='6'>还没有报告</td></tr>"
    )
    body = (
        f"<div class='panel'><h3>历史报告（最新在前）</h3>"
        f"<table><thead><tr><th>报告编号</th><th>时间</th><th>评分</th>"
        f"<th>数据评级</th><th>阻断</th><th>导出</th></tr></thead><tbody>{items}</tbody></table>"
        f"<p class='muted'><a href='/reports/trend'>查看评分趋势（旧→新） →</a></p></div>"
        f"<p><a href='/'>← 返回首页</a></p>"
    )
    return _page("历史报告", body)


def render_reports_trend() -> str:
    """评分/数据评级随时间趋势（只读历史manifest，时间正序）。"""
    rows = [r for r in collect_reports(REPORTS_ROOT)[:100]]
    rows.reverse()  # collect_reports最新在前；趋势按时间正序展示。
    scored = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    items = (
        "".join(
            f"<tr><td>{r['created_at']}</td>"
            f"<td><a href='/report/{r['report_id']}/index.html'>{r['report_id']}</a></td>"
            f"<td>{'—' if r['score'] is None else r['score']}</td>"
            f"<td>{r.get('grade_label') or '—'}</td>"
            f"<td>{r['rating']}</td><td>{r.get('strategy') or '—'}</td></tr>"
            for r in rows
        )
        or "<tr><td colspan='6'>还没有报告</td></tr>"
    )
    trend_text = ""
    if len(scored) >= 2:
        direction = (
            "上升" if scored[-1] > scored[0] else ("下降" if scored[-1] < scored[0] else "持平")
        )
        trend_text = (
            f"<p class='muted'>共{len(rows)}份报告，可评分{len(scored)}份；"
            f"评分从{scored[0]}到{scored[-1]}（{direction}）。</p>"
        )
    body = (
        f"<div class='panel'><h3>评分与数据评级趋势（旧 → 新）</h3>{trend_text}"
        f"<table><thead><tr><th>时间</th><th>报告编号</th><th>评分</th>"
        f"<th>等级</th><th>数据评级</th><th>策略</th></tr></thead>"
        f"<tbody>{items}</tbody></table>"
        f"<p class='muted'>评分衡量研究流程可信度，不评价未来收益。</p></div>"
        f"<p><a href='/reports'>← 返回历史报告</a></p>"
    )
    return _page("报告趋势", body)


def render_reports_compare(ids: list[str]) -> str:
    """跨报告只读对比页：汇总各报告自报指标，不做统一重放、不执行任何回测。"""
    from commercial_mvp import collect_report_details

    safe_ids = [str(i)[:80] for i in ids if str(i).strip()][:8]
    details = collect_report_details(REPORTS_ROOT, safe_ids) if safe_ids else []
    numeric = (
        "total_return_pct",
        "max_drawdown_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
    )
    rows = ""
    for item in details:
        if item.get("missing"):
            rows += f"<tr><td>{esc(item['report_id'])}</td><td colspan='10'>报告不存在</td></tr>"
            continue
        cells = "".join(
            f"<td>{'—' if item.get(key) is None else item.get(key)}</td>"
            for key in ("score", "rating") + numeric + ("total_trades",)
        )
        rows += (
            f"<tr><td><a href='/report/{esc(item['report_id'])}/index.html'>"
            f"{esc(item['report_id'])}</a></td>{cells}"
            f"<td>{esc(str(item.get('strategy') or '—'))}</td>"
            f"<td>{esc(str(item.get('period') or '—'))}</td></tr>"
        )
    body = (
        f"<div class='panel'><h3>跨报告对比（只读，各报告自报指标）</h3>"
        f"<table><thead><tr><th>报告编号</th><th>评分</th><th>数据评级</th>"
        f"<th>收益%</th><th>回撤%</th><th>Sharpe</th><th>Sortino</th>"
        f"<th>Calmar</th><th>交易</th><th>策略</th><th>区间</th></tr></thead>"
        f"<tbody>{rows or '<tr><td colspan=11>输入report_id后对比</td></tr>'}"
        f"</tbody></table>"
        f"<p class='muted'>不做统一重放；不同数据/参数的报告横向对比仅供参考。"
        f"评分评价研究流程可信度，不评价未来收益。</p>"
        f"<form method='get' action='/reports/compare'>"
        f"<label>report_id（逗号分隔，2-8个）</label>"
        f"<input type='text' name='ids' placeholder='id1,id2'>"
        f"<button type='submit'>对比</button></form></div>"
        f"<p><a href='/reports'>← 返回历史报告</a></p>"
    )
    return _page("跨报告对比", body)


def render_validate_form(error: str = "", result: str = "") -> str:
    """compare方案文件只读校验页：粘贴JSON→显示白名单/数量/范围错误；不执行任何回测。"""
    example = esc(
        '{"plans": [\n'
        '  {"label": "基线", "overrides": {}},\n'
        '  {"label": "高止损", "overrides": {"stop_loss_pct": 0.08}}\n'
        "]}",
        quote=True,
    )
    feedback = ""
    if error:
        feedback = f"<div class='notice' style='border-color:#dc2626;background:#fef2f2'>校验未通过：{error}</div>"
    elif result:
        feedback = (
            f"<div class='notice' style='border-color:#16a34a;background:#f0fdf4'>{result}</div>"
        )
    body = f"""
<div class="panel">
<h3>compare 方案文件校验（只读，不执行任何回测）</h3>
<p class="muted">粘贴对比计划JSON进行白名单/数量/范围校验；本页面只做格式检查，
不会读取行情、不会启动分析。实际运行时后端会再次校验（双保险）。</p>
<form method="post" action="/validate">
<label>方案JSON（2-8个方案；参数白名单与命令行一致）</label>
<textarea name="plan_json" rows="12" style="width:100%;font-family:Consolas,monospace;
font-size:13px;border:1px solid #cbd5e1;border-radius:8px;padding:8px"
placeholder="{example}"></textarea>
<button type="submit">只读校验</button>
</form></div>
{feedback}
<p><a href='/'>← 返回首页</a></p>"""
    return _page("方案校验", body)


def handle_validate(form: dict) -> str:
    """只读校验compare计划JSON：复用compare.parse_plan_payload，不执行任何分析。"""
    from commercial.compare import parse_plan_payload

    raw = (form.get("plan_json") or "").strip()
    if not raw:
        return render_validate_form(error="内容为空：请粘贴方案JSON")
    import json as _json

    try:
        payload = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        return render_validate_form(error=f"JSON解析失败（行{exc.lineno}列{exc.colno}）：{exc.msg}")
    try:
        plans = parse_plan_payload(payload)
    except ParameterError as exc:
        return render_validate_form(error=str(exc))
    except Exception as exc:  # 校验页不抛异常到服务层
        return render_validate_form(error=f"{type(exc).__name__}: {exc}")
    detail = "".join(
        f"<li><b>{esc(p['label'])}</b>："
        f"<code>{esc(_json.dumps(p['overrides'], ensure_ascii=False, sort_keys=True))}</code></li>"
        for p in plans
    )
    return render_validate_form(
        result=f"校验通过：{len(plans)}个方案均符合白名单与范围要求（未执行任何回测）。"
        f"<ul>{detail}</ul>"
    )


def parse_form(payload: bytes) -> dict:
    fields = parse_qs(payload.decode("utf-8"), keep_blank_values=True)
    return {k: v[0] for k, v in fields.items()}


def safe_report_id(raw: str) -> str | None:
    """URL中的report_id净化：只允许安全字符（防路径穿越），非法返回None。

    与run_metadata.validate_run_id同一规则；创建报告目录时已强制该规则，
    因此历史合法报告全部能通过，而".."、"%"、路径分隔符等构造会被拒绝。
    """
    from run_metadata import validate_run_id

    try:
        return validate_run_id(raw)
    except ValueError:
        return None


def request_from_form(form: dict) -> CommercialRequest:
    def flag(name):
        return form.get(name) == "1"

    # 截面策略专属参数：逗号分隔的KEY=VALUE；键与范围由request.validate()白名单把关。
    strategy_params: dict = {}
    raw_params = (form.get("strategy_params") or "").strip()
    if raw_params:
        for item in raw_params.split(","):
            item = item.strip()
            if not item:
                continue
            key, sep, value = item.partition("=")
            if not key.strip() or not sep or not value.strip():
                raise ParameterError(f"策略参数格式应为 KEY=VALUE（多组用逗号分隔）: {item}")
            strategy_params[key.strip()] = value.strip()

    request = CommercialRequest(
        cache=form.get("cache", "").strip(),
        start=form.get("start", "").strip(),
        end=form.get("end", "").strip(),
        strategy=(form.get("strategy") or "").strip() or "classic_trend",
        candidates=(form.get("candidates") or "").strip() or None,
        initial_capital=float(form.get("initial_capital") or 100_000),
        max_positions=int(form.get("max_positions") or 8),
        rebalance_freq=form.get("rebalance_freq", "weekly"),
        run_stress=not flag("no_stress"),
        run_factor_diagnostics=not flag("no_factor"),
        run_rolling_oos=not flag("no_rolling"),
        run_segment_consistency=not flag("no_segment"),
        rolling_anchored=flag("rolling_anchored"),
        trials=int(form.get("trials") or 1),
        strategy_params=strategy_params,
        output_root=REPORTS_ROOT,
        quiet=True,
    )
    request.validate()
    return request


class AuditHandler(BaseHTTPRequestHandler):
    server_version = "AliceAuditLocal/1.0"

    def log_message(self, fmt, *args):  # 安静模式：只打印关键错误
        pass

    def _send_html(self, content: str, code: int = 200):
        data = content.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload, code: int = 200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_report_file(self, report_id: str, relative: str) -> None:
        base = os.path.abspath(REPORTS_ROOT)
        target = os.path.abspath(os.path.join(base, report_id, relative))
        if not target.startswith(base + os.sep) or not os.path.isfile(target):
            self._send_html(_page("404", "<div class='panel'>文件不存在。</div>"), 404)
            return
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        with open(target, "rb") as handle:
            data = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_html(render_home())
        elif path == "/reports":
            self._send_html(render_reports())
        elif path == "/experiments":
            query = parse_qs(parsed.query)
            self._send_html(
                render_experiments(
                    strategy=(query.get("strategy") or [""])[0],
                    rating=(query.get("rating") or [""])[0],
                )
            )
        elif path == "/reports/trend":
            self._send_html(render_reports_trend())
        elif path == "/reports/compare":
            query = parse_qs(parsed.query)
            raw = (query.get("ids") or [""])[0]
            self._send_html(
                render_reports_compare([part.strip() for part in raw.split(",") if part.strip()])
            )
        elif path == "/validate":
            self._send_html(render_validate_form())
        elif path.startswith("/status"):
            query = parse_qs(parsed.query)
            job_id = (query.get("job") or [""])[0]
            self._send_html(render_status(job_id))
        elif path.startswith("/api/status"):
            query = parse_qs(parsed.query)
            job_id = (query.get("job") or [""])[0]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self._send_json({"error": "job not found"}, 404)
            else:
                self._send_json(job)
        elif path.startswith("/export/"):
            from commercial.report import export_report_zip

            report_id = safe_report_id(path[len("/export/") :])
            if report_id is None:
                # 路径净化：report_id只允许安全字符，防止".."等构造
                # 打包报告根目录之外的任意文件夹。
                self._send_html(_page("404", "<div class='panel'>报告不存在。</div>"), 404)
                return
            try:
                zip_path = export_report_zip(REPORTS_ROOT, report_id)
            except FileNotFoundError:
                self._send_html(_page("404", "<div class='panel'>报告不存在。</div>"), 404)
                return
            with open(zip_path, "rb") as handle:
                data = handle.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{report_id}.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path.startswith("/report/"):
            parts = path[len("/report/") :].split("/", 1)
            if len(parts) != 2 or not parts[1]:
                self._send_html(_page("404", "<div class='panel'>路径不正确。</div>"), 404)
                return
            self._serve_report_file(parts[0], parts[1])
        else:
            self._send_html(_page("404", "<div class='panel'>页面不存在。</div>"), 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/validate":
            try:
                length = int(self.headers.get("Content-Length") or 0)
                form = parse_form(self.rfile.read(min(length, 262144)))
            except (ValueError, UnicodeDecodeError):
                form = {}
            self._send_html(handle_validate(form))
            return
        if path != "/run":
            self._send_html(_page("404", "<div class='panel'>页面不存在。</div>"), 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            form = parse_form(self.rfile.read(min(length, 65536)))
            mode = form.get("mode", "full")
            request = request_from_form(form)
        except ParameterError as exc:
            self._send_html(
                _page(
                    "参数错误",
                    f"<div class='panel'><h3>参数错误</h3><p>{exc}</p>"
                    f"<p><a href='/'>← 返回修改</a></p></div>",
                ),
                400,
            )
            return
        except (TypeError, ValueError):
            self._send_html(
                _page(
                    "参数错误",
                    "<div class='panel'><h3>参数错误</h3><p>数字字段格式不正确。</p>"
                    "<p><a href='/'>← 返回修改</a></p></div>",
                ),
                400,
            )
            return
        if not os.path.exists(os.path.abspath(os.path.expanduser(request.cache))):
            self._send_html(
                _page(
                    "数据不存在",
                    f"<div class='panel'><h3>数据路径不存在</h3><p>{request.cache}</p>"
                    f"<p><a href='/'>← 返回修改</a></p></div>",
                ),
                400,
            )
            return
        plan_path = (form.get("plan_path") or "").strip() or None
        if mode == "compare" and not plan_path:
            self._send_html(
                _page(
                    "参数错误",
                    "<div class='panel'><h3>参数错误</h3><p>compare模式需要方案文件路径。</p>"
                    "<p><a href='/'>← 返回修改</a></p></div>",
                ),
                400,
            )
            return
        job_id = create_job(mode)
        thread = threading.Thread(
            target=run_job, args=(job_id, request, mode, plan_path), daemon=True
        )
        thread.start()
        self.send_response(303)
        self.send_header("Location", f"/status?job={job_id}")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Alice量化策略审计台 本地工作台")
    parser.add_argument(
        "--host", default="127.0.0.1", help="仅默认绑定本机回环地址；不建议对外开放"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    os.makedirs(REPORTS_ROOT, exist_ok=True)
    _load_jobs()
    server = ThreadingHTTPServer((args.host, args.port), AuditHandler)
    print(f"Alice量化策略审计台 本地工作台已启动：http://{args.host}:{args.port}")
    print(DISCLAIMER)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
