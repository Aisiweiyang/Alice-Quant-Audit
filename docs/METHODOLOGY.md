# 方法与局限

这是方法与工程工具，不是经过外部认证的研究评级系统。
可信度分数由数据质量、时间一致性、样本外检查、执行、成本、稳定性和可复现性
等启发式规则组成；不能把一个数字解释为真实盈利概率。

CSCV/PBO 衡量指定候选矩阵里样本内赢家在互补分段中排名退化的比例。
候选、分段、收益频率和搜索范围都影响结果，单窗口结论不可外推。
DSR 依赖收益分布和试验数量等输入；没有完整研究历史时，用户申报的 trials
不能被当成已认证的有效独立假设数量。退化为 PSR 或不足样本时须阅读原始状态。

- PBO：[Bailey 等原文](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- DSR：[Bailey 与 López de Prado 原文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

`overfitting.py` 保留原公式来源说明。项目中的外部架构参考包括 Qlib、LEAN、
VeighNa、NautilusTrader 与 backtesting.py 的公开设计思想；本次导出未引入其源码。
第三方依赖保留各自许可证，项目 MIT 不授予数据再分发权。

内置 A 股研究模型不能保证完整处理所有历史规则、公司行动、股息税、
容量与历史时点股票池。正式数据审计通过也不代表解决全部这些问题。
演示中的基准、股票代码与交易日序列都是合成设置。
