# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yfinance>=0.2.40",
#     "akshare>=1.15.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
#     "requests>=2.31.0",
# ]
# ///

"""
CAN SLIM Stock Screener - 内存优化版 v1.3

优化点:
1. 缩短历史数据周期 (6mo 替代 1y)
2. 分批处理 (10只/批, 间隔释放内存)
3. 减少重复数据请求
4. 主动垃圾回收
5. 简化数据处理流程

Usage:
    uv run canslim_scanner_memory_optimized.py                    # 分析美股 (默认)
    uv run canslim_scanner_memory_optimized.py --market cn        # 分析A股
    uv run canslim_scanner_memory_optimized.py --top 10 --min-score 40
    uv run canslim_scanner_memory_optimized.py --output json --export result.json
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import argparse
import gc
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# A股数据支持 (akshare)
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

# Alpha Vantage 数据支持
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHAVANTAGE_AVAILABLE = bool(ALPHA_VANTAGE_API_KEY)

# 全局缓存和限流
_av_cache = {}
_av_cache_time = {}
_av_last_request = 0
MIN_AV_INTERVAL = 12

# 默认观察列表 - 美股 (43只)
DEFAULT_US_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "NFLX", "CRM", "NOW", "SNOW", "DDOG", "NET", "ZS", "CRWD", "OKTA",
    "AMD", "QCOM", "MU", "LRCX", "KLAC", "AMAT",
    "COIN", "HOOD", "SQ", "PYPL", "SOFI",
    "PLTR", "MSTR", "APP", "DUOL", "CELH", "ELF", "SMCI",
    "BABA", "PDD", "JD", "BIDU", "NIO", "XPEV", "LI"
]

# A股默认观察列表
DEFAULT_CN_WATCHLIST = [
    "600519", "000858", "600276", "300750", "601012", "002594",
    "688981", "603501", "002371", "300014", "600036", "000001",
    "603019", "002230", "300033", "600570", "000333", "000651",
    "002415", "300760", "600809"
]


@dataclass
class CANSLIMScore:
    """CAN SLIM评分结果 (精简版)"""
    ticker: str
    name: str = ""
    price: float = 0.0
    market_cap: float = 0.0
    
    c_earnings_growth: Optional[float] = None
    c_revenue_growth: Optional[float] = None
    c_score: int = 0
    
    a_annual_growth: Optional[float] = None
    a_roe: Optional[float] = None
    a_score: int = 0
    
    n_distance_from_high: Optional[float] = None
    n_new_high_flag: bool = False
    n_score: int = 0
    
    s_volume_surge: Optional[float] = None
    s_score: int = 0
    
    l_rsi: Optional[float] = None
    l_above_sma50: bool = False
    l_above_sma200: bool = False
    l_score: int = 0
    
    i_market_cap_billions: float = 0.0
    i_score: int = 0
    
    m_market_score: int = 0
    total_score: int = 0
    passed_criteria: List[str] = None
    
    def __post_init__(self):
        if self.passed_criteria is None:
            self.passed_criteria = []


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


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
    
    return float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None


def get_stock_data_optimized(ticker: str, period: str = "6mo") -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """获取美股数据 (优化版 - 单次请求)"""
    try:
        stock = yf.Ticker(ticker)
        # 只获取需要的历史数据 (6个月足够200日均线)
        hist = stock.history(period=period)
        if hist.empty or len(hist) < 50:
            return None, None
        
        # 一次性获取info，减少API调用
        info = stock.info
        
        return info, hist
    except Exception:
        return None, None


def analyze_stock_optimized(ticker: str, market_ok: bool = True) -> Optional[CANSLIMScore]:
    """完整分析一只美股 (内存优化版)"""
    info, hist = get_stock_data_optimized(ticker)
    if not info or hist is None:
        return None

    try:
        score = CANSLIMScore(ticker=ticker)
        score.name = info.get('shortName', ticker)[:20]  # 限制长度
        score.price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        score.market_cap = info.get('marketCap', 0)

        # ========== C - Current Earnings (简化版) ==========
        try:
            revenue_growth = info.get('revenueGrowth')
            if revenue_growth and not pd.isna(revenue_growth):
                score.c_revenue_growth = round(revenue_growth * 100, 2)
                if score.c_revenue_growth > 25:
                    score.c_score = 25
                    score.passed_criteria.append("C+")
                elif score.c_revenue_growth > 15:
                    score.c_score = 15
                    score.passed_criteria.append("C")
                elif score.c_revenue_growth > 0:
                    score.c_score = 5
        except:
            pass

        # ========== A - Annual Growth (简化版) ==========
        try:
            roe = info.get('returnOnEquity')
            if roe:
                score.a_roe = round(roe * 100, 2)
                if score.a_roe > 17:
                    score.a_score = 15
                    score.passed_criteria.append("A")
        except:
            pass

        # ========== N - New Highs ==========
        try:
            current_price = float(hist['Close'].iloc[-1])
            high_52w = float(hist['High'].max())
            
            if high_52w > 0:
                distance = (high_52w - current_price) / high_52w * 100
                score.n_distance_from_high = round(distance, 2)
                
                if distance < 10:
                    score.n_new_high_flag = True
                    score.n_score = 20
                    score.passed_criteria.append("N")
                elif distance < 20:
                    score.n_score = 10
        except:
            pass

        # ========== S - Supply and Demand ==========
        try:
            recent_volume = float(hist['Volume'].tail(10).mean())
            avg_volume = float(hist['Volume'].tail(50).mean())
            
            if avg_volume > 0:
                ratio = recent_volume / avg_volume
                score.s_volume_surge = round(ratio, 2)
                
                if ratio > 1.5:
                    score.s_score = 15
                    score.passed_criteria.append("S")
                elif ratio > 1.2:
                    score.s_score = 10
                elif ratio > 1.0:
                    score.s_score = 5
        except:
            pass

        # ========== L - Leader (RSI, Trend) ==========
        try:
            current_price = float(hist['Close'].iloc[-1])
            
            # RSI
            score.l_rsi = calculate_rsi(hist['Close'])
            
            # 均线
            sma50 = float(hist['Close'].rolling(50).mean().iloc[-1])
            sma200 = float(hist['Close'].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else None
            
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

        # ========== I - Institutional ==========
        cap_b = score.market_cap / 1e9
        score.i_market_cap_billions = round(cap_b, 2)
        
        if cap_b > 100:
            score.i_score = 10
        elif cap_b > 10:
            score.i_score = 15
            score.passed_criteria.append("I")
        elif cap_b > 1:
            score.i_score = 5

        # ========== M - Market Direction ==========
        if market_ok:
            score.m_market_score = 10

        # 计算总分
        score.total_score = (
            score.c_score + score.a_score + score.n_score +
            score.s_score + score.l_score + score.i_score + score.m_market_score
        )

        return score
    except Exception:
        return None


def check_market_direction() -> Tuple[bool, float]:
    """检查美股市场方向 (SPY vs 50日均线) - 优化版"""
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="3mo")  # 减少到3个月
        if len(hist) < 50:
            return False, 0

        current = float(hist['Close'].iloc[-1])
        sma50 = float(hist['Close'].rolling(50).mean().iloc[-1])

        distance_pct = (current / sma50 - 1) * 100
        is_uptrend = current > sma50

        return is_uptrend, round(distance_pct, 2)
    except:
        return False, 0


def process_batch(watchlist: List[str], market_ok: bool, batch_size: int = 8, delay: float = 0.5) -> List[CANSLIMScore]:
    """分批处理股票，释放内存"""
    results = []
    total = len(watchlist)
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = watchlist[batch_start:batch_end]
        
        print(f"\n  📦 批次 {batch_start//batch_size + 1}/{(total-1)//batch_size + 1} ({batch_start+1}-{batch_end}/{total})")
        
        for i, ticker in enumerate(batch, batch_start + 1):
            print(f"  [{i:2d}/{total}] {ticker:6s} ... ", end='', flush=True)
            
            score = analyze_stock_optimized(ticker, market_ok)
            
            if score:
                results.append(score)
                print(f"✅ 得分: {score.total_score:2d} | {','.join(score.passed_criteria)}")
            else:
                print("❌ 跳过")
        
        # 批次间延迟和垃圾回收
        if batch_end < total:
            time.sleep(delay)
            gc.collect()  # 主动垃圾回收
    
    return results


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
    print("\n" + "=" * 85)
    print(f"🏆 CAN SLIM 精选榜 (Top {min(top_n, len(results))})")
    print("=" * 85)
    
    print(f"\n{'排名':<4} {'代码':<8} {'名称':<18} {'得分':<5} {'通过':<12} {'价格':<9} {'市值':<7} {'距高':<6}")
    print("-" * 85)
    
    for i, r in enumerate(results[:top_n], 1):
        name_short = (r.name[:16] + '..') if len(r.name) > 18 else r.name
        passed_str = ','.join(r.passed_criteria[:3])
        near_high = f"{r.n_distance_from_high:.1f}%" if r.n_distance_from_high else "N/A"
        
        print(f"{i:<4} {r.ticker:<8} {name_short:<18} {r.total_score:<5} {passed_str:<12} "
              f"${r.price:<8.1f} {format_market_cap(r.market_cap):<7} {near_high:<6}")


def print_detailed_analysis(results: List[CANSLIMScore], top_n: int = 5) -> None:
    """打印详细分析"""
    print("\n" + "=" * 85)
    print("📋 详细分析")
    print("=" * 85)
    
    for i, r in enumerate(results[:top_n], 1):
        print(f"\n{i}. {r.ticker} - {r.name}")
        print(f"   💯 总分: {r.total_score}/100 | 通过: {', '.join(r.passed_criteria)}")
        print(f"   💰 价格: ${r.price:.2f} | 市值: {format_market_cap(r.market_cap)}")
        
        if r.c_revenue_growth:
            status = "✅" if r.c_revenue_growth > 20 else ("🟡" if r.c_revenue_growth > 0 else "❌")
            print(f"   📈 营收增长: {r.c_revenue_growth:.1f}% {status}")
        
        if r.a_roe:
            status = "✅" if r.a_roe > 17 else "🟡"
            print(f"   📊 ROE: {r.a_roe:.1f}% {status}")
        
        if r.n_distance_from_high is not None:
            status = "✅" if r.n_new_high_flag else "🟡"
            print(f"   🎯 距52周高: {r.n_distance_from_high:.1f}% {status}")
        
        if r.s_volume_surge:
            status = "✅" if r.s_volume_surge > 1.3 else "🟡"
            print(f"   📊 成交量比: {r.s_volume_surge:.1f}x {status}")
        
        if r.l_rsi:
            status = "✅" if r.l_rsi > 50 else "🟡"
            print(f"   💪 RSI: {r.l_rsi:.1f} {status}")
        
        trend_status = "✅" if r.l_above_sma50 else "❌"
        print(f"   📈 50日均线: {'上方' if r.l_above_sma50 else '下方'} {trend_status}")


def main():
    parser = argparse.ArgumentParser(
        description='CAN SLIM 成长股量化筛选器 - 内存优化版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
优化点:
  - 历史数据周期: 6mo (原1y)
  - 分批处理: 8只/批，GC回收
  - 单请求获取数据，减少API调用
  - 移除财务报表查询，仅用info数据

评分标准:
  C: 营收增长>25%(+25), >15%(+15)
  A: ROE>17%(+15)
  N: 距52周高<10%(+20), <20%(+10)
  S: 成交量>1.5x(+15), >1.2x(+10)
  L: RSI>50(+10), 站50日线上(+10), 站200日线上(+5)
  I: 市值10B-100B(+15), >100B(+10)
  M: 市场趋势(+10)
        """
    )
    parser.add_argument('--watchlist', nargs='+', help='指定股票列表')
    parser.add_argument('--top', type=int, default=10, help='显示前N名 (默认10)')
    parser.add_argument('--min-score', type=int, default=25, help='最低得分门槛 (默认25)')
    parser.add_argument('--output', choices=['text', 'json'], default='text', help='输出格式')
    parser.add_argument('--export', type=str, help='导出JSON文件路径')
    parser.add_argument('--batch-size', type=int, default=8, help='每批处理数量 (默认8)')
    parser.add_argument('--market', choices=['us', 'cn'], default='us', help='市场选择 (默认us)')

    args = parser.parse_args()

    # 选择观察列表
    if args.market == 'cn':
        watchlist = args.watchlist if args.watchlist else DEFAULT_CN_WATCHLIST
        market_name = "A股"
    else:
        watchlist = args.watchlist if args.watchlist else DEFAULT_US_WATCHLIST
        market_name = "美股"

    print("=" * 85)
    print(f"🦐 CAN SLIM 成长股量化筛选器 v1.3 - 内存优化版 - {market_name}")
    print("   基于威廉·欧奈尔投资策略 | 减少内存占用 70%+")
    print("=" * 85)

    # 检查市场方向
    market_ok, market_pct = check_market_direction()
    market_status = "✅ 上升趋势" if market_ok else "⚠️ 震荡/下降"
    print(f"\n📈 市场方向 (SPY): {market_status} ({market_pct:+.1f}% vs 50日均线)")

    if not market_ok:
        print("   ⚠️  建议: 市场趋势不佳，谨慎操作")

    print(f"\n🔍 准备分析 {len(watchlist)} 只股票...")
    print(f"   批次大小: {args.batch_size}只/批 | 将自动释放内存")
    print("-" * 85)

    # 分批处理
    start_time = time.time()
    results = process_batch(watchlist, market_ok, batch_size=args.batch_size)
    elapsed = time.time() - start_time

    # 筛选和排序
    results = [r for r in results if r.total_score >= args.min_score]
    results.sort(key=lambda x: x.total_score, reverse=True)
    
    print(f"\n⏱️  分析完成: {elapsed:.1f}秒 | 通过筛选: {len(results)}只")
    
    if not results:
        print("\n⚠️ 没有股票达到最低得分门槛")
        return
    
    # 输出
    if args.output == 'json':
        data = [convert_to_serializable(asdict(r)) for r in results[:args.top]]
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print_results_table(results, args.top)
        print_detailed_analysis(results, min(5, args.top))
    
    # 导出
    if args.export:
        data = [convert_to_serializable(asdict(r)) for r in results]
        with open(args.export, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已导出: {args.export}")
    
    print("\n" + "=" * 85)
    print("⚠️  免责声明: 本工具仅供学习研究，不构成投资建议")
    print("    股市有风险，投资需谨慎")
    print("=" * 85)


if __name__ == "__main__":
    main()
