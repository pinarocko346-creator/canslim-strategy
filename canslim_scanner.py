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
    uv run canslim_scanner.py                    # 分析美股 (默认)
    uv run canslim_scanner.py --market cn        # 分析A股
    uv run canslim_scanner.py --market all       # 分析美股+A股
    uv run canslim_scanner.py --top 10 --min-score 40
    uv run canslim_scanner.py --watchlist AAPL MSFT NVDA --output json
    uv run canslim_scanner.py --market cn --watchlist 600519 000858 300750
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# A股数据支持 (akshare)
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

# Alpha Vantage 数据支持 (美股技术指标/基本面)
import os
import time
import requests
from datetime import datetime, timedelta

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHAVANTAGE_AVAILABLE = bool(ALPHA_VANTAGE_API_KEY)

# Alpha Vantage 缓存
_av_cache = {}
_av_cache_time = {}
_av_last_request = 0
MIN_AV_INTERVAL = 12  # 免费版 5次/分钟

def _av_rate_limit():
    """Alpha Vantage 限流"""
    global _av_last_request
    elapsed = time.time() - _av_last_request
    if elapsed < MIN_AV_INTERVAL:
        time.sleep(MIN_AV_INTERVAL - elapsed)
    _av_last_request = time.time()

def _av_get_cache(key: str):
    if key in _av_cache and key in _av_cache_time:
        if time.time() - _av_cache_time[key] < 300:  # 5分钟缓存
            return _av_cache[key]
    return None

def _av_set_cache(key: str, data):
    _av_cache[key] = data
    _av_cache_time[key] = time.time()

def get_av_quote(symbol: str) -> Optional[Dict]:
    """Alpha Vantage 实时报价"""
    if not ALPHAVANTAGE_AVAILABLE:
        return None
    
    cache_key = f"av_quote_{symbol}"
    cached = _av_get_cache(cache_key)
    if cached:
        return cached
    
    _av_rate_limit()
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            result = {
                "price": float(quote.get("05. price", 0)),
                "change": float(quote.get("09. change", 0)),
                "change_percent": quote.get("10. change percent", "0%"),
                "volume": int(quote.get("06. volume", 0)),
            }
            _av_set_cache(cache_key, result)
            return result
    except:
        pass
    return None

def get_av_fundamentals(symbol: str) -> Optional[Dict]:
    """Alpha Vantage 基本面数据"""
    if not ALPHAVANTAGE_AVAILABLE:
        return None
    
    cache_key = f"av_fund_{symbol}"
    cached = _av_get_cache(cache_key)
    if cached:
        return cached
    
    _av_rate_limit()
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data and "Symbol" in data:
            result = {
                "sector": data.get("Sector", ""),
                "industry": data.get("Industry", ""),
                "market_cap": int(data.get("MarketCapitalization", 0)),
                "pe_ratio": float(data.get("PERatio", 0)) if data.get("PERatio") else None,
                "pb_ratio": float(data.get("PriceToBookRatio", 0)) if data.get("PriceToBookRatio") else None,
                "roe": float(data.get("ReturnOnEquityTTM", 0)) * 100 if data.get("ReturnOnEquityTTM") else None,
                "revenue_growth": float(data.get("QuarterlyRevenueGrowthYOY", 0)) * 100 if data.get("QuarterlyRevenueGrowthYOY") else None,
                "earnings_growth": float(data.get("QuarterlyEarningsGrowthYOY", 0)) * 100 if data.get("QuarterlyEarningsGrowthYOY") else None,
            }
            _av_set_cache(cache_key, result)
            return result
    except:
        pass
    return None


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# 默认观察列表 - 美股优质成长股
DEFAULT_US_WATCHLIST = [
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
    # 中国科技股 (ADR)
    "BABA", "PDD", "JD", "BIDU", "NIO", "XPEV", "LI"
]

# A股默认观察列表 - 优质成长股
DEFAULT_CN_WATCHLIST = [
    # 白酒/消费
    "600519",   # 贵州茅台
    "000858",   # 五粮液
    "600276",   # 恒瑞医药
    # 新能源
    "300750",   # 宁德时代
    "601012",   # 隆基绿能
    "002594",   # 比亚迪
    # 科技/半导体
    "688981",   # 中芯国际
    "603501",   # 韦尔股份
    "002371",   # 北方华创
    "300014",   # 亿纬锂能
    # 金融
    "600036",   # 招商银行
    "000001",   # 平安银行
    # 互联网/AI
    "603019",   # 中科曙光
    "002230",   # 科大讯飞
    "300033",   # 同花顺
    "600570",   # 恒生电子
    # 制造业
    "000333",   # 美的集团
    "000651",   # 格力电器
    "002415",   # 海康威视
    # 医药
    "300760",   # 迈瑞医疗
    "600809",   # 山西汾酒
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
    """获取美股数据 (yfinance)"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty or len(hist) < 50:
            return None, None
        return stock, hist
    except Exception as e:
        return None, None


def get_cn_stock_data(code: str) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """获取A股数据 (akshare)

    Returns:
        info: 股票基本信息 dict
        hist: 历史行情 DataFrame (列名兼容 yfinance: Open, High, Low, Close, Volume)
    """
    if not AKSHARE_AVAILABLE:
        print("⚠️  akshare 未安装，无法获取A股数据")
        return None, None

    try:
        # 获取历史行情
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20240101", adjust="qfq")
        if df is None or len(df) < 50:
            return None, None

        # 列名转换为 yfinance 格式以便兼容
        df = df.rename(columns={
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        # 获取股票基本信息 - 使用历史数据的最新价格和代码作为名称
        # 避免使用 stock_zh_a_spot_em() 因为它会加载全市场数据
        try:
            current_price = df['Close'].iloc[-1]
            info = {
                'shortName': code,  # 使用代码作为名称，避免查询全市场
                'currentPrice': current_price,
                'marketCap': 0,
            }
        except:
            info = {'shortName': code, 'currentPrice': 0, 'marketCap': 0}

        return info, df
    except Exception as e:
        return None, None


def analyze_c_current(stock: yf.Ticker, score: CANSLIMScore, ticker: str = "") -> None:
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
    
    # 使用 Alpha Vantage 补充数据
    if ticker and ALPHAVANTAGE_AVAILABLE:
        av_fund = get_av_fundamentals(ticker)
        if av_fund:
            # Alpha Vantage 提供季度增长数据
            if not score.c_revenue_growth and av_fund.get('revenue_growth'):
                score.c_revenue_growth = av_fund['revenue_growth']
                if score.c_revenue_growth > 25:
                    score.c_score = max(score.c_score, 25)
                    if "C+" not in score.passed_criteria:
                        score.passed_criteria.append("C+")
                elif score.c_revenue_growth > 15:
                    score.c_score = max(score.c_score, 15)
                    if "C" not in score.passed_criteria and "C+" not in score.passed_criteria:
                        score.passed_criteria.append("C")
            
            if not score.c_earnings_growth and av_fund.get('earnings_growth'):
                score.c_earnings_growth = av_fund['earnings_growth']


def analyze_a_annual(stock: yf.Ticker, score: CANSLIMScore, ticker: str = "") -> None:
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
    
    # 使用 Alpha Vantage 补充数据
    if ticker and ALPHAVANTAGE_AVAILABLE:
        av_fund = get_av_fundamentals(ticker)
        if av_fund:
            # Alpha Vantage 的 ROE 更精确
            if not score.a_roe and av_fund.get('roe'):
                score.a_roe = av_fund['roe']
                if score.a_roe > 17:
                    score.a_score = 15
                    if "A" not in score.passed_criteria:
                        score.passed_criteria.append("A")


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
    """完整分析一只美股"""
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
        analyze_c_current(stock, score, ticker)
        analyze_a_annual(stock, score, ticker)
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


def analyze_cn_stock(code: str) -> Optional[CANSLIMScore]:
    """完整分析一只A股 (使用akshare)"""
    info, hist = get_cn_stock_data(code)
    if not info or hist is None:
        return None

    try:
        score = CANSLIMScore(ticker=code)
        score.name = info.get('shortName', code)
        score.price = info.get('currentPrice', 0)
        # A股市值需要另外获取，暂时设为0
        score.market_cap = info.get('marketCap', 0)

        # A股目前主要支持技术分析 (N, S, L)
        # C和A需要财务报表数据，akshare可以扩展

        analyze_n_new_highs(hist, score)
        analyze_s_supply_demand(hist, score)
        analyze_l_leader(hist, score)
        # A股市值数据需要另外获取，暂时跳过I评分

        # 计算总分 (A股目前主要基于技术面)
        score.total_score = (
            score.c_score + score.a_score + score.n_score +
            score.s_score + score.l_score + score.i_score + score.m_market_score
        )

        return score
    except Exception as e:
        return None


def check_market_direction() -> Tuple[bool, float]:
    """检查美股市场方向 (SPY vs 50日均线)"""
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


def check_cn_market_direction() -> Tuple[bool, float]:
    """检查A股市场方向 (上证指数 vs 50日均线)"""
    if not AKSHARE_AVAILABLE:
        return False, 0

    try:
        # 获取上证指数历史数据
        df = ak.index_zh_a_hist(symbol="000001", period="daily", start_date="20240801")
        if df is None or len(df) < 50:
            return False, 0

        df = df.rename(columns={'收盘': 'Close'})
        current = df['Close'].iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1]

        distance_pct = (current / sma50 - 1) * 100
        is_uptrend = current > sma50

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
    data = [convert_to_serializable(asdict(r)) for r in results]
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

市场选择:
  us   - 美股 (yfinance, 默认)
  cn   - A股 (akshare)
  all  - 美股+A股
        """
    )
    parser.add_argument('--watchlist', nargs='+', help='指定股票列表')
    parser.add_argument('--top', type=int, default=10, help='显示前N名 (默认10)')
    parser.add_argument('--min-score', type=int, default=25, help='最低得分门槛 (默认25)')
    parser.add_argument('--output', choices=['text', 'json'], default='text', help='输出格式')
    parser.add_argument('--export', type=str, help='导出JSON文件路径')
    parser.add_argument('--market', choices=['us', 'cn', 'all'], default='us', help='市场选择 (默认us)')

    args = parser.parse_args()

    # 根据市场选择设置观察列表
    if args.market == 'cn':
        watchlist = args.watchlist if args.watchlist else DEFAULT_CN_WATCHLIST
        is_cn_market = True
        market_name = "A股"
    elif args.market == 'all':
        watchlist = (args.watchlist if args.watchlist else DEFAULT_US_WATCHLIST) + DEFAULT_CN_WATCHLIST
        is_cn_market = False
        market_name = "美股+A股"
    else:
        watchlist = args.watchlist if args.watchlist else DEFAULT_US_WATCHLIST
        is_cn_market = False
        market_name = "美股"

    print("=" * 90)
    print(f"🦐 CAN SLIM 成长股量化筛选器 v1.2 - {market_name}")
    print("   基于威廉·欧奈尔(William J. O'Neil)投资策略")
    if ALPHAVANTAGE_AVAILABLE and args.market != 'cn':
        print("   📊 Alpha Vantage 数据增强已启用")
    print("=" * 90)

    # 检查市场方向
    if args.market == 'cn':
        market_ok, market_pct = check_cn_market_direction()
        market_label = "上证指数"
    else:
        market_ok, market_pct = check_market_direction()
        market_label = "SPY"

    market_status = "✅ 上升趋势" if market_ok else "⚠️ 震荡/下降"
    print(f"\n📈 市场方向 ({market_label}): {market_status} ({market_pct:+.1f}% vs 50日均线)")

    if not market_ok:
        print("   ⚠️  建议: 市场趋势不佳，谨慎操作或降低仓位")

    print(f"\n🔍 正在分析 {len(watchlist)} 只股票...")
    print("-" * 90)

    results = []
    for i, ticker in enumerate(watchlist, 1):
        print(f"[{i:2d}/{len(watchlist)}] {ticker:6s} ... ", end='', flush=True)

        # 根据股票代码判断市场并使用对应分析函数
        if args.market == 'cn' or (args.market == 'all' and ticker.isdigit()):
            score = analyze_cn_stock(ticker)
        else:
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
        data = [convert_to_serializable(asdict(r)) for r in results[:args.top]]
        print(json.dumps(data, ensure_ascii=False, indent=2))
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
