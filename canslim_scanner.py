# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yfinance>=0.2.40",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
# ]
# ///

"""
CAN SLIM Stock Screener - 欧奈尔成长股量化筛选器

基于威廉·欧奈尔(William J. O'Neil)的CAN SLIM投资策略：
C = Current Quarterly Earnings (当季每股收益增长 > 20%)
A = Annual Earnings Growth (年度收益增长趋势)
N = New Products/Management/Highs (接近52周新高)
S = Supply and Demand (成交量放大)
L = Leader or Laggard (行业相对强弱)
I = Institutional Sponsorship (机构持仓)
M = Market Direction (市场趋势)

Usage:
    uv run canslim_scanner.py
    uv run canslim_scanner.py --top 10 --min-score 40
    uv run canslim_scanner.py --watchlist AAPL MSFT NVDA --output json
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# 默认观察列表 - 优质成长股
DEFAULT_WATCHLIST = [
    # 科技巨头
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    # 软件/SaaS
    "NFLX", "CRM", "NOW", "SNOW", "DDOG", "NET", "ZS", "CRWD", "OKTA",
    # 半导体
    "AMD", "AVGO", "QCOM", "MU", "LRCX", "KLAC", "AMAT",
    # 金融科技/加密货币
    "COIN", "HOOD", "SQ", "PYPL", "SOFI",
    # 新兴市场/高成长
    "PLTR", "MSTR", "APP", "DUOL", "CELH", "ELF", "SMCI",
    # 中国科技股
    "BABA", "PDD", "JD", "BIDU", "NIO", "XPEV", "LI"
]


@dataclass
class CANSLIMScore:
    """CAN SLIM评分结果"""
    ticker: str
    name: str = ""
    price: float = 0.0
    market_cap: float = 0.0
    
    # C - Current Quarterly Earnings
    c_earnings_growth: Optional[float] = None
    c_revenue_growth: Optional[float] = None
    c_score: int = 0
    
    # A - Annual Earnings Growth
    a_annual_growth: Optional[float] = None
    a_roe: Optional[float] = None
    a_score: int = 0
    
    # N - New Highs
    n_distance_from_high: Optional[float] = None
    n_new_high_flag: bool = False
    n_score: int = 0
    
    # S - Supply and Demand
    s_volume_surge: Optional[float] = None
    s_avg_volume: Optional[float] = None
    s_score: int = 0
    
    # L - Leader (RSI, Trend)
    l_rsi: Optional[float] = None
    l_above_sma50: bool = False
    l_above_sma200: bool = False
    l_score: int = 0
    
    # I - Institutional (简化为市值指标)
    i_market_cap_billions: float = 0.0
    i_score: int = 0
    
    # M - Market Direction (外部传入)
    m_market_score: int = 0
    
    # 总分
    total_score: int = 0
    passed_criteria: List[str] = None
    
    def __post_init__(self):
        if self.passed_criteria is None:
            self.passed_criteria = []


def calculate_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """计算RSI指标"""
    if len(prices) < period + 1:
        return None
    
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0)
    loss = -deltas.where(deltas < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None


def get_stock_data(ticker: str, period: str = "1y") -> Tuple[Optional[yf.Ticker], Optional[pd.DataFrame]]:
    """获取股票基础数据"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty or len(hist) < 50:
            return None, None
        return stock, hist
    except Exception as e:
        return None, None


def analyze_c_current(stock: yf.Ticker, score: CANSLIMScore) -> None:
    """分析C - Current Quarterly Earnings/Revenue"""
    try:
        # 尝试获取季度收入数据
        quarterly_income = stock.quarterly_income_stmt
        if quarterly_income is not None and not quarterly_income.empty:
            if 'TotalRevenue' in quarterly_income.index:
                revenue = quarterly_income.loc['TotalRevenue'].dropna()
                if len(revenue) >= 4:
                    recent = revenue.iloc[0]
                    year_ago = revenue.iloc[3]
                    if year_ago != 0 and not pd.isna(year_ago):
                        growth = ((recent - year_ago) / abs(year_ago)) * 100
                        score.c_revenue_growth = round(growth, 2)
                        
                        # 评分: >25% (+25), >15% (+15), >0% (+5)
                        if growth > 25:
                            score.c_score = 25
                            score.passed_criteria.append("C+")
                        elif growth > 15:
                            score.c_score = 15
                            score.passed_criteria.append("C")
                        elif growth > 0:
                            score.c_score = 5
            
            # 尝试获取EPS增长
            if 'NetIncome' in quarterly_income.index:
                net_income = quarterly_income.loc['NetIncome'].dropna()
                if len(net_income) >= 4:
                    recent = net_income.iloc[0]
                    year_ago = net_income.iloc[3]
                    if year_ago != 0 and not pd.isna(year_ago):
                        growth = ((recent - year_ago) / abs(year_ago)) * 100
                        score.c_earnings_growth = round(growth, 2)
    except:
        pass


def analyze_a_annual(stock: yf.Ticker, score: CANSLIMScore) -> None:
    """分析A - Annual Earnings Growth"""
    try:
        info = stock.info
        
        # 使用ROE作为替代指标
        roe = info.get('returnOnEquity')
        if roe:
            score.a_roe = round(roe * 100, 2)
            if score.a_roe > 17:  # 欧奈尔标准: ROE > 17%
                score.a_score = 15
                score.passed_criteria.append("A")
        
        # 年收入增长
        annual_revenue = info.get('revenueGrowth')
        if annual_revenue:
            score.a_annual_growth = round(annual_revenue * 100, 2)
    except:
        pass


def analyze_n_new_highs(hist: pd.DataFrame, score: CANSLIMScore) -> None:
    """分析N - New Highs (接近52周新高)"""
    try:
        current_price = hist['Close'].iloc[-1]
        high_52w = hist['High'].max()
        
        if high_52w > 0:
            distance = (high_52w - current_price) / high_52w * 100
            score.n_distance_from_high = round(distance, 2)
            
            # 距离高点 < 10% 视为强势
            if distance < 10:
                score.n_new_high_flag = True
                score.n_score = 20
                score.passed_criteria.append("N")
            elif distance < 20:
                score.n_score = 10
    except:
        pass


def analyze_s_supply_demand(hist: pd.DataFrame, score: CANSLIMScore) -> None:
    """分析S - Supply and Demand (成交量)"""
    try:
        recent_volume = hist['Volume'].tail(10).mean()
        avg_volume = hist['Volume'].tail(50).mean()
        
        if avg_volume > 0:
            ratio = recent_volume / avg_volume
            score.s_volume_surge = round(ratio, 2)
            score.s_avg_volume = round(avg_volume, 0)
            
            # 成交量放大 > 1.3倍
            if ratio > 1.5:
                score.s_score = 15
                score.passed_criteria.append("S")
            elif ratio > 1.2:
                score.s_score = 10
            elif ratio > 1.0:
                score.s_score = 5
    except:
        pass


def analyze_l_leader(hist: pd.DataFrame, score: CANSLIMScore) -> None:
    """分析L - Leader (RSI, Trend)"""
    try:
        current_price = hist['Close'].iloc[-1]
        
        # RSI
        score.l_rsi = calculate_rsi(hist['Close'])
        
        # 50日/200日均线
        sma50 = hist['Close'].rolling(50).mean().iloc[-1]
        sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
        
        score.l_above_sma50 = current_price > sma50
        if sma200:
            score.l_above_sma200 = current_price > sma200
        
        # 评分
        if score.l_rsi and score.l_rsi > 50:
            score.l_score += 10
        if score.l_above_sma50:
            score.l_score += 10
            score.passed_criteria.append("L50")
        if score.l_above_sma200:
            score.l_score += 5
            score.passed_criteria.append("L200")
    except:
        pass


def analyze_i_institutional(score: CANSLIMScore) -> None:
    """分析I - Institutional Sponsorship (机构持仓)"""
    # 简化为市值指标
    cap_b = score.market_cap / 1e9
    score.i_market_cap_billions = round(cap_b, 2)
    
    # 偏好中大型成长股
    if cap_b > 100:  # 大型股
        score.i_score = 10
    elif cap_b > 10:  # 中型股
        score.i_score = 15
        score.passed_criteria.append("I")
    elif cap_b > 1:  # 小型股
        score.i_score = 5


def analyze_stock(ticker: str) -> Optional[CANSLIMScore]:
    """完整分析一只股票"""
    stock, hist = get_stock_data(ticker)
    if not stock or hist is None:
        return None
    
    try:
        info = stock.info
        score = CANSLIMScore(ticker=ticker)
        score.name = info.get('shortName', ticker)
        score.price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        score.market_cap = info.get('marketCap', 0)
        
        # 逐项分析
        analyze_c_current(stock, score)
        analyze_a_annual(stock, score)
        analyze_n_new_highs(hist, score)
        analyze_s_supply_demand(hist, score)
        analyze_l_leader(hist, score)
        analyze_i_institutional(score)
        
        # 计算总分
        score.total_score = (
            score.c_score + score.a_score + score.n_score + 
            score.s_score + score.l_score + score.i_score + score.m_market_score
        )
        
        return score
    except Exception as e:
        return None


def check_market_direction() -> Tuple[bool, float]:
    """检查市场方向 (SPY vs 50日均线)"""
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="6mo")
        if len(hist) < 50:
            return False, 0
        
        current = hist['Close'].iloc[-1]
        sma50 = hist['Close'].rolling(50).mean().iloc[-1]
        sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
        
        distance_pct = (current / sma50 - 1) * 100
        
        # 价格在50日线上方视为趋势良好
        is_uptrend = current > sma50
        if sma200:
            is_uptrend = is_uptrend and (current > sma200)
        
        return is_uptrend, round(distance_pct, 2)
    except:
        return False, 0


def format_market_cap(cap: float) -> str:
    """格式化市值显示"""
    if cap >= 1e12:
        return f"{cap/1e12:.2f}T"
    elif cap >= 1e9:
        return f"{cap/1e9:.1f}B"
    elif cap >= 1e6:
        return f"{cap/1e6:.1f}M"
    return f"{cap:.0f}"


def print_results_table(results: List[CANSLIMScore], top_n: int = 10) -> None:
    """打印结果表格"""
    print("\n" + "=" * 90)
    print(f"🏆 CAN SLIM 精选榜 (Top {min(top_n, len(results))})")
    print("=" * 90)
    
    print(f"\n{'排名':<4} {'代码':<8} {'名称':<20} {'得分':<5} {'通过':<15} {'价格':<10} {'市值':<8} {'距高':<6} {'RSI':<5}")
    print("-" * 90)
    
    for i, r in enumerate(results[:top_n], 1):
        name_short = r.name[:18] if len(r.name) > 18 else r.name
        passed_str = ','.join(r.passed_criteria[:3])
        near_high = f"{r.n_distance_from_high:.1f}%" if r.n_distance_from_high else "N/A"
        rsi = f"{r.l_rsi:.0f}" if r.l_rsi else "N/A"
        
        print(f"{i:<4} {r.ticker:<8} {name_short:<20} {r.total_score:<5} {passed_str:<15} "
              f"${r.price:<9.2f} {format_market_cap(r.market_cap):<8} {near_high:<6} {rsi:<5}")


def print_detailed_analysis(results: List[CANSLIMScore], top_n: int = 5) -> None:
    """打印详细分析"""
    print("\n" + "=" * 90)
    print("📋 详细分析")
    print("=" * 90)
    
    for i, r in enumerate(results[:top_n], 1):
        print(f"\n{i}. {r.ticker} - {r.name}")
        print(f"   💯 总分: {r.total_score}/100 | 通过: {', '.join(r.passed_criteria)}")
        print(f"   💰 价格: ${r.price:.2f} | 市值: {format_market_cap(r.market_cap)}")
        
        # C
        if r.c_revenue_growth:
            status = "✅" if r.c_revenue_growth > 20 else ("🟡" if r.c_revenue_growth > 0 else "❌")
            print(f"   📈 营收增长: {r.c_revenue_growth:.1f}% {status}")
        if r.c_earnings_growth:
            print(f"   💵 利润增长: {r.c_earnings_growth:.1f}%")
        
        # A
        if r.a_roe:
            status = "✅" if r.a_roe > 17 else "🟡"
            print(f"   📊 ROE: {r.a_roe:.1f}% {status}")
        
        # N
        if r.n_distance_from_high is not None:
            status = "✅" if r.n_new_high_flag else "🟡"
            print(f"   🎯 距52周高: {r.n_distance_from_high:.1f}% {status}")
        
        # S
        if r.s_volume_surge:
            status = "✅" if r.s_volume_surge > 1.3 else "🟡"
            print(f"   📊 成交量比: {r.s_volume_surge:.1f}x {status}")
        
        # L
        if r.l_rsi:
            status = "✅" if r.l_rsi > 50 else "🟡"
            print(f"   💪 RSI: {r.l_rsi:.1f} {status}")
        trend_status = "✅" if r.l_above_sma50 else "❌"
        print(f"   📈 50日均线: {'上方' if r.l_above_sma50 else '下方'} {trend_status}")


def export_to_json(results: List[CANSLIMScore], filepath: str) -> None:
    """导出结果为JSON"""
    data = [asdict(r) for r in results]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已导出: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='CAN SLIM 成长股量化筛选器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
评分标准:
  C (Current)    : 营收增长>25%(+25), >15%(+15)
  A (Annual)     : ROE>17%(+15)
  N (New Highs)  : 距52周高<10%(+20), <20%(+10)
  S (Supply/Demand): 成交量>1.5x(+15), >1.2x(+10)
  L (Leader)     : RSI>50(+10), 站50日线上(+10), 站200日线上(+5)
  I (Institutional): 市值10B-100B(+15), >100B(+10)
  M (Market)     : 市场趋势加成(0-10)
        """
    )
    parser.add_argument('--watchlist', nargs='+', help='指定股票列表')
    parser.add_argument('--top', type=int, default=10, help='显示前N名 (默认10)')
    parser.add_argument('--min-score', type=int, default=25, help='最低得分门槛 (默认25)')
    parser.add_argument('--output', choices=['text', 'json'], default='text', help='输出格式')
    parser.add_argument('--export', type=str, help='导出JSON文件路径')
    
    args = parser.parse_args()
    
    watchlist = args.watchlist if args.watchlist else DEFAULT_WATCHLIST
    
    print("=" * 90)
    print("🦐 CAN SLIM 成长股量化筛选器 v1.0")
    print("   基于威廉·欧奈尔(William J. O'Neil)投资策略")
    print("=" * 90)
    
    # 检查市场方向
    market_ok, market_pct = check_market_direction()
    market_status = "✅ 上升趋势" if market_ok else "⚠️ 震荡/下降"
    print(f"\n📈 市场方向 (SPY): {market_status} ({market_pct:+.1f}% vs 50日均线)")
    
    if not market_ok:
        print("   ⚠️  建议: 市场趋势不佳，谨慎操作或降低仓位")
    
    print(f"\n🔍 正在分析 {len(watchlist)} 只股票...")
    print("-" * 90)
    
    results = []
    for i, ticker in enumerate(watchlist, 1):
        print(f"[{i:2d}/{len(watchlist)}] {ticker:6s} ... ", end='', flush=True)
        score = analyze_stock(ticker)
        if score:
            # 根据市场趋势调整M分
            if market_ok:
                score.m_market_score = 10
                score.total_score += 10
            results.append(score)
            print(f"得分: {score.total_score:2d} | 通过: {','.join(score.passed_criteria)}")
        else:
            print("跳过 (数据不足)")
    
    # 筛选和排序
    results = [r for r in results if r.total_score >= args.min_score]
    results.sort(key=lambda x: x.total_score, reverse=True)
    
    if not results:
        print("\n⚠️ 没有股票达到最低得分门槛")
        return
    
    # 输出
    if args.output == 'json':
        print(json.dumps([asdict(r) for r in results[:args.top]], ensure_ascii=False, indent=2))
    else:
        print_results_table(results, args.top)
        print_detailed_analysis(results, min(5, args.top))
    
    # 导出
    if args.export:
        export_to_json(results, args.export)
    
    print("\n" + "=" * 90)
    print("⚠️  免责声明: 本工具仅供学习研究，不构成投资建议")
    print("    股市有风险，投资需谨慎")
    print("=" * 90)


if __name__ == "__main__":
    main()
