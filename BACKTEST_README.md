# CAN SLIM 回测框架

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `backtest_simple.py` | 简化版回测引擎 |
| `run_backtest.py` | 回测运行器（带CAN SLIM评分） |

## 🚀 快速开始

### 1. 运行演示回测

使用模拟信号快速测试回测引擎：

```bash
python run_backtest.py --demo
```

### 2. 运行完整回测

使用真实 CAN SLIM 评分进行回测：

```bash
# 回测2023年全年
python run_backtest.py --start 2023-01-01 --end 2023-12-31 --capital 100000

# 自定义参数
python run_backtest.py \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --capital 100000 \
    --top 50 \
    --threshold 70
```

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--start` | 回测开始日期 | 2023-01-01 |
| `--end` | 回测结束日期 | 今天 |
| `--capital` | 初始资金 | 100000 |
| `--top` | 扫描股票数量 | 50 |
| `--threshold` | 买入阈值（总分） | 70 |
| `--demo` | 演示模式 | False |

## 📊 回测配置

在 `backtest_simple.py` 中修改 `BacktestConfig`：

```python
config = BacktestConfig(
    initial_capital=100000,  # 初始资金
    commission_rate=0.001,   # 佣金 0.1%
    slippage=0.0005,         # 滑点 0.05%
    max_positions=10,        # 最大持仓数
    hold_days=30,            # 持仓周期
    score_threshold=70,      # 买入阈值
    stop_loss=0.08,          # 止损 8%
    take_profit=0.20         # 止盈 20%
)
```

## 📈 输出指标

- **总交易次数**: 完成的交易数量
- **胜率**: 盈利交易占比
- **总盈亏**: 累计盈亏金额
- **平均盈利**: 平均每笔交易盈亏
- **盈亏比**: 平均盈利/平均亏损
- **最大回撤**: 最大资金回撤比例
- **年化收益率**: 策略年化收益
- **最终资产**: 回测结束时总资产

## 🔄 回测流程

1. **股票池**: 标普500成分股（简化版）
2. **信号生成**: 每月初扫描 CAN SLIM 评分
3. **买入条件**: 总分 ≥ 70，按分数排序取前10
4. **卖出条件**: 
   - 止损: -8%
   - 止盈: +20%
   - 持仓到期: 30天
   - 回测结束强制平仓
5. **仓位管理**: 等权重分配，最多10只
6. **交易成本**: 佣金0.1% + 滑点0.05%

## 📝 结果报告

回测完成后会生成 JSON 报告：

```
backtest_report_2023-01-01_2024-01-01.json
```

包含完整配置、结果指标和生成时间。

## ⚠️ 注意事项

1. **数据限制**: 免费版 yfinance 可能有请求频率限制
2. **回测偏差**: 历史表现不代表未来收益
3. **信号滞后**: 回测中使用的是月末评分信号
4. **简化处理**: 未考虑分红、拆股等事件

## 🔧 进阶用法

### 自定义股票池

修改 `backtest_simple.py` 中的 `fetch_sp500_list()`:

```python
def fetch_sp500_list():
    return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]  # 自定义股票
```

### 修改策略参数

编辑 `BacktestConfig` 调整策略:

```python
config = BacktestConfig(
    max_positions=5,      # 减少持仓数
    hold_days=60,         # 延长持仓
    score_threshold=80,   # 提高买入门槛
    stop_loss=0.05,       # 收紧止损
    take_profit=0.30      # 提高止盈
)
```

### 集成到 GitHub Actions

在 `.github/workflows/canslim-daily.yml` 中添加回测步骤:

```yaml
- name: Run Backtest
  run: |
    python run_backtest.py --start 2023-01-01 --end $(date +%Y-%m-%d)
```

## 📚 参考

- [CAN SLIM 策略说明](https://github.com/pinarocko346-creator/canslim-strategy)
- [欧奈尔投资法则](https://www.investors.com/)

---
*注意: 本回测框架仅供学习和研究使用，不构成投资建议。*
