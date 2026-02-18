# CAN SLIM Stock Screener 🦐

基于威廉·欧奈尔(William J. O'Neil)经典成长股投资策略的量化筛选器。

## 什么是 CAN SLIM?

CAN SLIM 是欧奈尔在《How to Make Money in Stocks》中提出的7维成长股选股法则：

| 字母 | 含义 | 筛选标准 |
|------|------|----------|
| **C** | Current Quarterly Earnings | 当季收益增长 > 20% |
| **A** | Annual Earnings Growth | 年度增长趋势，ROE > 17% |
| **N** | New Products/Management/Highs | 接近52周新高 (<10%) |
| **S** | Supply and Demand | 成交量放大 (>1.5x) |
| **L** | Leader or Laggard | 行业龙头，RSI > 50，站均线上方 |
| **I** | Institutional Sponsorship | 机构持仓 (市值10B-100B最优) |
| **M** | Market Direction | 大盘趋势 (SPY vs 50日均线) |

## 快速开始

### 环境要求
- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (推荐) 或 pip

### 安装依赖
```bash
# 使用 uv (推荐)
uv pip install yfinance pandas numpy matplotlib

# 或使用 pip
pip install yfinance pandas numpy matplotlib
```

### 运行筛选器
```bash
# 默认运行（分析预设观察列表）
uv run canslim_scanner.py

# 显示前20名，最低门槛30分
uv run canslim_scanner.py --top 20 --min-score 30

# 指定股票列表
uv run canslim_scanner.py --watchlist AAPL MSFT NVDA TSLA

# JSON输出
uv run canslim_scanner.py --output json

# 导出结果到文件
uv run canslim_scanner.py --export results.json
```

## 评分规则

| 指标 | 优秀 | 良好 | 一般 | 得分 |
|------|------|------|------|------|
| 营收增长 | >25% | >15% | >0% | 25/15/5 |
| ROE | >17% | - | - | 15 |
| 距52周高 | <10% | <20% | - | 20/10 |
| 成交量比 | >1.5x | >1.2x | >1.0x | 15/10/5 |
| RSI | >50 | - | - | 10 |
| 站50日线 | 是 | - | - | 10 |
| 站200日线 | 是 | - | - | 5 |
| 市值 | 10B-100B | >100B | >1B | 15/10/5 |
| 市场趋势 | 上升 | - | - | 10 |

**满分**: 100分

## 项目结构

```
canslim-strategy/
├── canslim_scanner.py    # 主筛选器脚本
├── backtest/             # 回测模块 (TODO)
├── charts/               # 生成的图表 (TODO)
├── README.md             # 本文件
└── .gitignore
```

## 示例输出

```
==========================================================================================
🏆 CAN SLIM 精选榜 (Top 10)
==========================================================================================

排名   代码       名称                   得分    通过               价格        市值      距高     RSI  
------------------------------------------------------------------------------------------
1    NVDA     NVIDIA Corporation     85     C+,N,L50,L200      $188.50    4.59T    11.2%   47   
2    AAPL     Apple Inc.             80     C+,N,L50           $263.38    3.87T    8.7%    56   
...
```

## 注意事项

1. **数据源**: 使用 Yahoo Finance (yfinance)，免费但可能有15-20分钟延迟
2. **财报数据**: 部分小盘股财报数据可能不完整，C/A项可能为N/A
3. **市场时机**: CAN SLIM 策略在上升趋势中效果最佳，注意M项（市场方向）
4. **仅供参考**: 本工具用于学习研究，不构成投资建议

## TODO

- [ ] 接入更完整的财报数据（EPS、收入增长历史）
- [ ] 杯柄形态（Cup and Handle）自动识别
- [ ] 行业相对强弱排名
- [ ] 历史回测功能
- [ ] 可视化图表生成
- [ ] 邮件/通知推送

## 参考资源

- 📚 [How to Make Money in Stocks](https://www.amazon.com/How-Make-Money-Stocks-Winning/dp/0071614133) - 欧奈尔原著
- 📊 [Investor's Business Daily](https://www.investors.com/) - 欧奈尔创立的财经媒体
- 🦐 [虾头碎碎念] 本策略由AI助手"虾头"开发维护

## License

MIT License - 自由使用，风险自担

---

**⚠️ 免责声明**: 本工具仅供学习研究，不构成投资建议。股市有风险，投资需谨慎。
