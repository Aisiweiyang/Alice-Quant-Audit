"""
策略配置 —— 所有可调参数集中管理
修改策略不需要改代码，改这个文件即可
"""


class StrategyConfig:
    """生产基线策略配置。"""

    # ========== 选股条件 ==========
    # 成交额排名前N（流动性的保证）
    top_volume_n = 500

    # 连续上涨天数
    consecutive_up_days = 2

    # 单日最大涨幅%（过滤急涨庄股）
    max_daily_gain_pct = 3.0

    # 入选最低综合评分
    min_score = 5.0

    # 主板过滤开关
    main_board_only = True

    # ========== 估值过滤（历史行情已由Baostock提供PE/PB） ==========
    use_pe_filter = True
    pe_min = 0  # PE>0 排除亏损
    pe_max = 60  # PE<60（收紧，过滤高估值）

    use_pb_filter = True
    pb_min = 0.0
    pb_max = 10.0

    use_roe_filter = False
    roe_min = 5.0  # ROE > 5%

    # ========== 因子权重 ==========
    # avg_volume_5d 曾误命名为 turnover（实为5日均量而非换手率），已更名。
    factor_weights = {
        "momentum": 0.25,  # 5日动量（最重要）
        "reversal": 0.05,  # 短期反转
        "adx": 0.00,  # ADX（暂不启用，数据噪声大）
        "volatility": 0.05,  # 波动率
        "avg_volume_5d": 0.10,  # 5日均量(百万股)
        "volume_ratio": 0.00,  # 量比（暂不启用）
        "vol_price_corr": 0.15,  # 量价相关性
        "steadiness": 0.15,  # 涨幅平稳度
        "price_position": 0.00,  # 价格位置（暂不启用）
        "rsi": 0.10,  # RSI(14)
        "bb": 0.05,  # 布林带位置
        "macd": 0.10,  # MACD信号
    }

    # ========== 回测设置 ==========
    initial_capital = 100_000  # 初始资金
    max_positions = 8  # 最大持仓数（集中持仓）
    max_position_pct = 0.12  # 单只股票最大仓位(12%)
    rebalance_freq = "weekly"  # 调仓频率: daily/weekly/monthly
    rebalance_day = "monday"  # 调仓日

    # ========== 大盘择时 ==========
    use_market_timing = True  # 大盘走弱时空仓
    ma_period = 20  # 均线周期
    ma_cascade = False  # 级联清仓关闭（太激进，会错过反弹）
    ma_slope_lookback = 0  # MA斜率关闭（噪音大）

    # ========== 个股趋势过滤 ==========
    stock_ma_period = 50  # 个股收盘价必须 > MA(N) 才能入选
    stock_ma_crossover = False  # MA20>MA50关闭（过滤太多）
    anti_falling_knife = False  # 防抄底关闭（A股急跌后反弹是常态）
    anti_falling_lookback = 3
    anti_falling_threshold = 0.05

    # ========== 止盈止损 ==========
    take_profit_pct = 0.10  # 止盈 10%
    stop_loss_pct = 0.05  # 止损 5%

    # ========== 交易成本 ==========
    commission_rate = 0.0003  # 佣金 万三
    # 证券交易印花税仅卖出收取，2023-08-28起减半。
    # stamp_tax_rate 保留为兼容字段；v4实际使用下面的历史费率。
    stamp_tax_rate = 0.0005
    stamp_tax_rate_before_20230828 = 0.001
    stamp_tax_rate_after_20230828 = 0.0005
    slippage = 0.001  # 滑点 千一
    min_commission = 5.0  # 最低佣金5元

    # ========== 模拟盘事前风控 ==========
    emergency_stop = False  # True时禁止新增买入，仍允许风险降低卖出
    max_total_position_pct = 0.95  # 总持仓市值上限
    max_order_value_pct = 0.15  # 单笔委托金额/总权益上限
    max_active_orders = 20  # 同时处于活动状态的订单上限
    daily_loss_limit_pct = 0.03  # 当日亏损达到3%后禁止新增买入

    # ========== 可视化 ==========
    benchmark_index = "sh.000300"  # 对比基准：沪深300

    @classmethod
    def to_dict(cls):
        """导出为字典（方便保存和加载）"""
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("_")}

    @classmethod
    def update(cls, **kwargs):
        """批量更新参数"""
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)

    @classmethod
    def reset_weights(cls):
        """重置因子权重为默认值"""
        cls.factor_weights = {
            "momentum": 0.20,
            "reversal": 0.10,
            "volatility": 0.10,
            "avg_volume_5d": 0.10,
            "vol_price_corr": 0.10,
            "steadiness": 0.10,
            "rsi": 0.10,
            "bb": 0.10,
            "macd": 0.10,
        }
