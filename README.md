# Alice Quant Audit · Alice 商业化 MVP 开源版

Alice Quant Audit 用来检查 A 股量化策略的回测结果。导入行情后，可以先查数据质量，
再比较策略、费用和参数对结果的影响，最后生成带图表和明细的 HTML 报告。

项目在本机运行，提供网页和命令行两种入口。

**作者：[Aisiweiyang](https://github.com/Aisiweiyang)** · [MIT 许可证](LICENSE)

## 能做什么

- 检查 CSV 或本机缓存中的缺失记录、价格异常、停牌状态、估值和基准数据。
- 运行趋势策略和截面多因子策略，同时比较 2–8 个参数方案。
- 改变费用、参数和回测区间，观察结果是否稳定；计算 PBO、DSR 等过拟合诊断指标。
- 导出 HTML 报告、图表和指标明细，用文件哈希核对输入与产物。
- 在网页中发起分析、查看历史报告和比较结果。

## 快速开始

建议 Python **3.12**。在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe examples/commercial_demo/generate_demo_data.py
.venv\Scripts\python.exe commercial_web.py
```

打开 <http://127.0.0.1:8765>。网页的数据路径填写
`examples/commercial_demo/demo_cache.pkl`，日期使用 `2025-07-01` 至 `2025-12-31`。
演示数据是固定随机种子的 **SYNTHETIC 合成数据**，不代表真实证券表现。

Linux / macOS 将上述解释器替换为 `.venv/bin/python`。
Windows 也可使用 `scripts/install.ps1` 与 `scripts/run_commercial_demo.ps1`。

命令行完整演示（先生成合成数据）：

```bash
python commercial_mvp.py full --cache examples/commercial_demo/demo_cache.pkl --start 2025-07-01 --end 2025-12-31
python commercial_mvp.py reports --list
```

请使用已经激活的虚拟环境，或把 `python` 替换为虚拟环境解释器的完整路径。
报告写入 `output/commercial_reports/<report_id>/index.html`；已有报告不覆盖。
分析自己的数据前，先把 `full` 换成 `audit` 运行检查。

比较三个截面策略参数（演示方案，不是参数推荐）：

```bash
python commercial_mvp.py compare --cache examples/commercial_demo/demo_cache.pkl --start 2025-07-01 --end 2025-12-31 --strategy cross_sectional_alpha158 --plan examples/compare_plan.json
```

## 验证

```bash
python -m pip install -r requirements-dev.txt
python examples/commercial_demo/generate_demo_data.py
python -m unittest discover -s tests -v
python -m ruff check .
python scripts/check_public_tree.py
```

测试使用合成数据和小型固定样例，不下载真实行情。
`tests/baselines/real_baselines.json` 只保留历史回归锚点，相关测试检查
结构与比对逻辑；公开版不附带其数据，也不宣称复现这些历史收益。
当前实际验证范围见 [发布验证记录](docs/VALIDATION.md)。
`requirements-tested.txt` 记录本轮本机验证的完整依赖版本；通常先按 requirements.txt 安装。

## 使用边界

- 本项目评价研究过程，不承诺收益，不提供买卖建议，不连接券商自动交易。
- `READY` 表示满足程序的有限检查，**不是**严格 PIT、完整公司行动或未来收益的认证。
- 总分是可解释的启发式汇总，必须结合缺失项、阻断项和样本范围阅读。
- PBO 只针对输入的候选收益矩阵；DSR 依赖试验数量等假设，不能消除未记录的选型偏差。
- 不同窗口、账户或参数方案不能直接视为独立实验；历史重跑不等于未来验证。
- `.pkl` 反序列化可以执行代码。**只加载自己生成或完全信任的缓存**；交换数据优先 CSV。
- 网页是无登录的本机工具，只绑定回环地址；不要通过代理或端口映射公开服务。
- 不分发真实行情、账户、持仓、凭证或旧私有仓库历史；使用者自行取得数据使用授权。

更多说明：[架构与 Alice 关系](docs/ARCHITECTURE.md)、[方法与局限](docs/METHODOLOGY.md)、
[贡献指南](CONTRIBUTING.md)、[安全说明](SECURITY.md)。

## 目录

```text
commercial/          审计、比较、诊断和报告编排
engine/ strategy/    内置研究回测与策略
data/                缓存、数据清洗与质量检查
analysis/ research/  指标及研究辅助函数
integration/         候选清单接口
paper_trade/         离线模拟账户和对账辅助，无实盘网关
examples/            合成数据生成器
tests/               单元、集成和黄金基线测试
scripts/             安装、演示与检查入口
```
