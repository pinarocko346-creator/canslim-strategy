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

基于威廉·欧奈尔(William J. O'Neil)的CAN SLIM投资策略
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import argparse
import pickle
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from functools import wraps
from pathlib import Path

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

import time
import requests

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHAVANTAGE_AVAILABLE = bool(ALPHA_VANTAGE_API_KEY)


# ============================================================================
# A. 异常处理和重试装饰器
# ============================================================================

def retry_on_failure(max_retries=3, delay=1):
    """装饰器：失败时重试"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if result is not None and not (hasattr(result, 'empty') and result.empty):
                        return result
                except Exception as e:
                    print(f"⚠️  尝试 {attempt+1}/{max_retries} 失败: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator


# ============================================================================
# B. 数据验证函数
# ============================================================================

def validate_stock_data(hist, info, ticker):
    """验证数据完整性和合理性"""
    if hist is None or hist.empty:
        return False, "无历史数据"
    if len(hist) < 50:
        return False, "历史数据不足50天"
    if info is None:
        return False, "无基本信息"
    current_price = hist['Close'].iloc[-1]
    if current_price <= 0 or pd.isna(current_price):
        return False, "价格数据异常"
    return True, "数据正常"


# ============================================================================
# D. 本地缓存机制 (SQLite)
# ============================================================================

class DataCache:
    """股票数据本地缓存管理器"""
    
    def __init__(self, cache_dir: str = None, cache_duration_hours: int = 24):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.canslim_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.db_path = self.cache_dir / "stock_cache.db"
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_cache (
                ticker TEXT PRIMARY KEY,
                market TEXT,
                hist_data BLOB,
                info_data BLOB,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        cache_time = datetime.fromtimestamp(timestamp)
        return datetime.now() - cache_time < self.cache_duration
    
    def get(self, ticker: str, market: str = "us"):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT hist_data, info_data, timestamp FROM stock_cache WHERE ticker = ? AND market = ?",
                (ticker, market)
            )
            row = cursor.fetchone()
            conn.close()
            if row and self._is_cache_valid(row[2]):
                hist = pickle.loads(row[0]) if row[0] else None
                info = pickle.loads(row[1]) if row[1] else None
                return info, hist
        except Exception as e:
            pass
        return None, None
    
    def set(self, ticker: str, info: Any, hist: pd.DataFrame, market: str = "us"):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            hist_blob = pickle.dumps(hist) if hist is not None else None
            info_blob = pickle.dumps(info) if info is not None else None
            cursor.execute('''
                INSERT OR REPLACE INTO stock_cache (ticker, market, hist_data, info_data, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (ticker, market, hist_blob, info_blob, time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            pass


_data_cache = DataCache()


# ============================================================================
# C. 备用数据源 - Alpha Vantage
# ============================================================================

_av_cache = {}
_av_cache_time = {}
_av_last_request = 0
MIN_AV_INTERVAL = 12

def _av_rate_limit():
    global _av_last_request
    elapsed = time.time() - _av_last_request
    if elapsed < MIN_AV_INTERVAL:
        time.sleep(MIN_AV_INTERVAL - elapsed)
    _av_last_request = time.time()

def _av_get_cache(key: str):
    if key in _av_cache and key in _av_cache_time:
        if time.time() - _av_cache_time[key] < 300:
            return _av_cache[key]
    return None

def _av_set_cache(key: str, data):
    _av_cache[key] = data
    _av_cache_time[key] = time.time()

@retry_on_failure(max_retries=2, delay=2)
def get_av_daily(symbol: str) -> Optional[pd.DataFrame]:
    if not ALPHAVANTAGE_AVAILABLE:
        return None
    cache_key = f"av_daily_{symbol}"
    cached = _av_get_cache(cache_key)
    if cached is not None:
        return cached
    _av_rate_limit()
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full", "apikey": ALPHA_VANTAGE_API_KEY}
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        _av_set_cache(cache_key, df)
        return df
    return None

@retry_on_failure(max_retries=2, delay=2)
def get_av_fundamentals(symbol: str) -> Optional[Dict]:
    if not ALPHAVANTAGE_AVAILABLE:
        return None
    cache_key = f"av_fund_{symbol}"
    cached = _av_get_cache(cache_key)
    if cached:
        return cached
    _av_rate_limit()
    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": ALPHA_VANTAGE_API_KEY}
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    if data and "Symbol" in data:
        result = {
            "sector": data.get("Sector", ""),
            "industry": data.get("Industry", ""),
            "market_cap": int(data.get("MarketCapitalization", 0)),
            "roe": float(data.get("ReturnOnEquityTTM", 0)) * 100 if data.get("ReturnOnEquityTTM") else None,
            "revenue_growth": float(data.get("QuarterlyRevenueGrowthYOY", 0)) * 100 if data.get("QuarterlyRevenueGrowthYOY") else None,
        }
        _av_set_cache(cache_key, result)
        return result
    return None


# ============================================================================
# 数据获取函数（带缓存和备用源）
# ============================================================================

@retry_on_failure(max_retries=3, delay=1)
def get_stock_data_yf(ticker: str, period: str = "1y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty or len(hist) < 50:
        return None, None
    return stock, hist

def get_stock_data(ticker: str, period: str = "1y", use_cache: bool = True):
    global _data_cache
    if use_cache:
        cached_info, cached_hist = _data_cache.get(ticker, market="us")
        if cached_info is not None and cached_hist is not None:
            is_valid, msg = validate_stock_data(cached_hist, cached_info, ticker)
            if is_valid:
                return cached_info, cached_hist, "cache"
    stock, hist = get_stock_data_yf(ticker, period)
    if stock is not None and hist is not None:
        is_valid, msg = validate_stock_data(hist, stock.info, ticker)
        if is_valid:
            if use_cache:
                _data_cache.set(ticker, stock.info, hist, market="us")
            return stock, hist, "yfinance"
    if ALPHAVANTAGE_AVAILABLE:
        av_hist = get_av_daily(ticker)
        if av_hist is not None and len(av_hist) >= 50:
            av_info = get_av_fundamentals(ticker)
            if av_info:
                is_valid, msg = validate_stock_data(av_hist, av_info, ticker)
                if is_valid:
                    if use_cache:
                        _data_cache.set(ticker, av_info, av_hist, market="us")
                    return av_info, av_hist, "alphavantage"
    return None, None, "failed"

@retry_on_failure(max_retries=3, delay=1)
def get_cn_stock_data_akshare(code: str):
    if not AKSHARE_AVAILABLE:
        return None, None
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20240101", adjust="qfq")
    if df is None or len(df) < 50:
        return None, None
    df = df.rename(columns={'日期': 'Date', '开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', '成交量': 'Volume'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    try:
        current_price = df['Close'].iloc[-1]
        info = {'shortName': code, 'currentPrice': current_price, 'marketCap': 0}
    except:
        info = {'shortName': code, 'currentPrice': 0, 'marketCap': 0}
    return info, df

def get_cn_stock_data(code: str, use_cache: bool = True):
    global _data_cache
    if use_cache:
        cached_info, cached_hist = _data_cache.get(code, market="cn")
        if cached_info is not None and cached_hist is not None:
            is_valid, msg = validate_stock_data(cached_hist, cached_info, code)
            if is_valid:
                return cached_info, cached_hist, "cache"
    info, hist = get_cn_stock_data_akshare(code)
    if info is not None and hist is not None:
        is_valid, msg = validate_stock_data(hist, info, code)
        if is_valid:
            if use_cache:
                _data_cache.set(code, info, hist, market="cn")
            return info, hist, "akshare"
    return None, None, "failed"


# 其他函数保持不变...
DEFAULT_US_WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO"]
DEFAULT_CN_WATCHLIST = ["600519", "000858", "300750", "601012", "002594"]

@dataclass
class CANSLIMScore:
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
    s_avg_volume: Optional[float] = None
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

# ============================================================================
# 分析函数
# ============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
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

def analyze_c_current(stock: Any, score: CANSLIMScore, ticker: str = "", source: str = "") -> None:
    if source == "yfinance" and isinstance(stock, yf.Ticker):
        quarterly_income = retry_on_failure(max_retries=2, delay=1)(lambda: stock.quarterly_income_stmt)()
        if quarterly_income is not None and not quarterly_income.empty:
            revenue_row = 'TotalRevenue' if 'TotalRevenue' in quarterly_income.index else ('Total Revenue' if 'Total Revenue' in quarterly_income.index else None)
            if revenue_row:
                revenue = quarterly_income.loc[revenue_row].dropna()
                if len(revenue) >= 4:
                    recent, year_ago = revenue.iloc[0], revenue.iloc[3]
                    if year_ago != 0 and not pd.isna(year_ago) and year_ago != recent:
                        growth = ((recent - year_ago) / abs(year_ago)) * 100
                        score.c_revenue_growth = round(growth, 2)
                        if growth > 25:
                            score.c_score = 25
                            score.passed_criteria.append("C+")
                        elif growth > 15:
                            score.c_score = 15
                            score.passed_criteria.append("C")
                        elif growth > 0:
                            score.c_score = 5
            netincome_row = 'NetIncome' if 'NetIncome' in quarterly_income.index else ('Net Income' if 'Net Income' in quarterly_income.index else None)
            if netincome_row:
                net_income = quarterly_income.loc[netincome_row].dropna()
                if len(net_income) >= 4:
                    recent, year_ago = net_income.iloc[0], net_income.iloc[3]
                    if year_ago != 0 and not pd.isna(year_ago) and year_ago != recent:
                        growth = ((recent - year_ago) / abs(year_ago)) * 100
                        score.c_earnings_growth = round(growth, 2)
        if score.c_revenue_growth is None:
            annual_income = retry_on_failure(max_retries=2, delay=1)(lambda: stock.income_stmt)()
            if annual_income is not None and not annual_income.empty and 'TotalRevenue' in annual_income.index:
                revenue = annual_income.loc['TotalRevenue'].dropna()
                if len(revenue) >= 2:
                    recent, year_ago = revenue.iloc[0], revenue.iloc[1]
                    if year_ago != 0 and not pd.isna(year_ago) and year_ago != recent:
                        growth = ((recent - year_ago) / abs(year_ago)) * 100
                        score.c_revenue_growth = round(growth, 2)
                        if growth > 20 and score.c_score < 15:
                            score.c_score = 15
                            if "C" not in score.passed_criteria:
                                score.passed_criteria.append("C")
        if score.c_revenue_growth is None:
            try:
                info = stock.info
                if info:
                    revenue_growth = info.get('revenueGrowth')
                    if revenue_growth and not pd.isna(revenue_growth):
                        score.c_revenue_growth = round(revenue_growth * 100, 2)
                        if score.c_revenue_growth > 20 and score.c_score < 10:
                            score.c_score = 10
            except:
                pass
    if ticker and ALPHAVANTAGE_AVAILABLE:
        av_fund = get_av_fundamentals(ticker)
        if av_fund:
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

def analyze_a_annual(stock: Any, score: CANSLIMScore, ticker: str = "", source: str = "") -> None:
    try:
        if source == "yfinance" and isinstance(stock, yf.Ticker):
            info = stock.info
        elif source == "alphavantage" and isinstance(stock, dict):
            info = stock
        else:
            info = None
        if info:
            roe = info.get('returnOnEquity') if source == "yfinance" else info.get('roe')
            if roe:
                if source == "yfinance":
                    roe = roe * 100
                score.a_roe = round(roe, 2)
                if score.a_roe > 17:
                    score.a_score = 15
                    score.passed_criteria.append("A")
            annual_revenue = info.get('revenueGrowth')
            if annual_revenue and source == "yfinance":
                score.a_annual_growth = round(annual_revenue * 100, 2)
    except:
        pass
    if ticker and ALPHAVANTAGE_AVAILABLE and not score.a_roe:
        av_fund = get_av_fundamentals(ticker)
        if av_fund:
            if not score.a_roe and av_fund.get('roe'):
                score.a_roe = av_fund['roe']
                if score.a_roe > 17:
                    score.a_score = 15
                    if "A" not in score.passed_criteria:
                        score.passed_criteria.append("A")

def analyze_n_new_highs(hist: pd.DataFrame, score: CANSLIMScore) -> None:
    try:
        current_price = hist['Close'].iloc[-1]
        high_52w = hist['High'].max()
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

def analyze_s_supply_demand(hist: pd.DataFrame, score: CANSLIMScore) -> None:
    try:
        recent_volume = hist['Volume'].tail(10).mean()
        avg_volume = hist['Volume'].tail(50).mean()
        if avg_volume > 0:
            ratio = recent_volume / avg_volume
            score.s_volume_surge = round(ratio, 2)
            score.s_avg_volume = round(avg_volume, 0)
            if ratio > 1.5:
                score.s_score = 15
                score.passed_criteria.append("S")
            elif ratio > 1.2:
                score.s_score = 10
            elif ratio > 1.0:
                score.s_score = 5
    except:
        pass

def calculate_rs_rating(ticker: str, period: str = "252d") -> float:
    try:
        stock = yf.Ticker(ticker)
        stock_hist = stock.history(period=period)
        if stock_hist.empty or len(stock_hist) < 50:
            return 0.0
        stock_return = (stock_hist['Close'].iloc[-1] / stock_hist['Close'].iloc[0] - 1) * 100
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period=period)
        spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100
        relative_strength = stock_return - spy_return
        if relative_strength >= 20:
            rs_score = 100
        elif relative_strength <= -20:
            rs_score = 0
        else:
            rs_score = 50 + (relative_strength / 20) * 50
        return max(0, min(100, rs_score))
    except Exception as e:
        return 0.0

def analyze_l_leader(hist: pd.DataFrame, score: CANSLIMScore, ticker: str = "") -> None:
    try:
        current_price = hist['Close'].iloc[-1]
        if ticker:
            rs_rating = calculate_rs_rating(ticker)
            score.l_rsi = rs_rating
            if rs_rating >= 90:
                score.l_score += 6
            elif rs_rating >= 80:
                score.l_score += 5
            elif rs_rating >= 70:
                score.l_score += 3
            elif rs_rating >= 50:
                score.l_score += 1
        high_52w = hist['High'].max()
        if high_52w > 0:
            distance_from_high = (high_52w - current_price) / high_52w * 100
            if distance_from_high <= 5:
                score.l_score += 2
            elif distance_from_high <= 10:
                score.l_score += 1
        sma50 = hist['Close'].rolling(50).mean().iloc[-1]
        sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
        score.l_above_sma50 = current_price > sma50
        if sma200:
            score.l_above_sma200 = current_price > sma200
        if score.l_above_sma50:
            score.passed_criteria.append("L50")
        if score.l_above_sma200:
            score.passed_criteria.append("L200")
    except:
        pass

def analyze_i_institutional(stock: Any, hist: pd.DataFrame, score: CANSLIMScore, source: str = "") -> None:
    try:
        if source == "yfinance" and isinstance(stock, yf.Ticker):
            info = stock.info
        elif source == "alphavantage" and isinstance(stock, dict):
            info = stock
        else:
            info = None
        if info:
            market_cap = info.get('marketCap', 0)
            if source == "alphavantage":
                market_cap = info.get('market_cap', 0)
            cap_b = market_cap / 1e9
            score.i_market_cap_billions = round(cap_b, 2)
            if market_cap >= 100e9:
                score.i_score += 5
            elif market_cap >= 10e9:
                score.i_score += 4
            elif market_cap >= 2e9:
                score.i_score += 3
            elif market_cap >= 500e6:
                score.i_score += 2
            elif market_cap > 0:
                score.i_score += 1
        if not hist.empty and len(hist) >= 20:
            avg_volume = hist['Volume'].tail(20).mean()
            if avg_volume >= 10e6:
                score.i_score += 5
            elif avg_volume >= 5e6:
                score.i_score += 4
            elif avg_volume >= 1e6:
                score.i_score += 3
            elif avg_volume >= 500e3:
                score.i_score += 2
            elif avg_volume > 0:
                score.i_score += 1
        if len(hist) >= 63:
            recent_returns = hist['Close'].tail(63).pct_change().dropna()
            if len(recent_returns) > 0:
                volatility = recent_returns.std() * np.sqrt(252) * 100
                if 15 <= volatility <= 40:
                    score.i_score += 5
                elif 10 <= volatility < 15 or 40 < volatility <= 50:
                    score.i_score += 3
                elif volatility < 10 or 50 < volatility <= 60:
                    score.i_score += 1
        if score.i_score >= 10:
            score.passed_criteria.append("I")
    except Exception as e:
        pass

def analyze_stock(ticker: str) -> Optional[CANSLIMScore]:
    stock, hist, source = get_stock_data(ticker)
    if stock is None or hist is None:
        return None
    try:
        if source == "yfinance":
            info = stock.info
        elif source == "alphavantage":
            info = stock
        else:
            info = None
        score = CANSLIMScore(ticker=ticker)
        score.name = info.get('shortName', ticker) if info else ticker
        score.price = info.get('currentPrice', info.get('regularMarketPrice', 0)) if info else 0
        score.market_cap = info.get('marketCap', 0) if info else 0
        analyze_c_current(stock, score, ticker, source)
        analyze_a_annual(stock, score, ticker, source)
        analyze_n_new_highs(hist, score)
        analyze_s_supply_demand(hist, score)
        analyze_l_leader(hist, score, ticker)
        analyze_i_institutional(stock, hist, score, source)
        score.total_score = (
            score.c_score + score.a_score + score.n_score +
            score.s_score + score.l_score + score.i_score + score.m_market_score
        )
        return score
    except Exception as e:
        return None

def analyze_cn_stock(code: str) -> Optional[CANSLIMScore]:
    info, hist, source = get_cn_stock_data(code)
    if info is None or hist is None:
        return None
    try:
        score = CANSLIMScore(ticker=code)
        score.name = info.get('shortName', code)
        score.price = info.get('currentPrice', 0)
        score.market_cap = info.get('marketCap', 0)
        analyze_n_new_highs(hist, score)
        analyze_s_supply_demand(hist, score)
        analyze_l_leader(hist, score)
        score.total_score = (
            score.c_score + score.a_score + score.n_score +
            score.s_score + score.l_score + score.i_score + score.m_market_score
        )
        return score
    except Exception as e:
        return None

@retry_on_failure(max_retries=3, delay=1)
def check_market_direction():
    spy = yf.Ticker("SPY")
    hist = spy.history(period="6mo")
    if len(hist) < 50:
        return False, 0
    current = hist['Close'].iloc[-1]
    sma50 = hist['Close'].rolling(50).mean().iloc[-1]
    sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
    distance_pct = (current / sma50 - 1) * 100
    is_uptrend = current > sma50
    if sma200:
        is_uptrend = is_uptrend and (current > sma200)
    return is_uptrend, round(distance_pct, 2)

@retry_on_failure(max_retries=3, delay=1)
def check_cn_market_direction():
    if not AKSHARE_AVAILABLE:
        return False, 0
    df = ak.index_zh_a_hist(symbol="000001", period="daily", start_date="20240801")
    if df is None or len(df) < 50:
        return False, 0
    df = df.rename(columns={'收盘': 'Close'})
    current = df['Close'].iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1]
    distance_pct = (current / sma50 - 1) * 100
    is_uptrend = current > sma50
    return is_uptrend, round(distance_pct, 2)

def format_market_cap(cap: float) -> str:
    if cap >= 1e12:
        return f"{cap/1e12:.2f}T"
    elif cap >= 1e9:
        return f"{cap/1e9:.1f}B"
    elif cap >= 1e6:
        return f"{cap/1e6:.1f}M"
    return f"{cap:.0f}"

def convert_to_serializable(obj: Any) -> Any:
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

def print_results_table(results, top_n: int = 10):
    print("\n" + "=" * 90)
    print(f"🏆 CAN SLIM 精选榜 (Top {min(top_n, len(results))})")
    print("=" * 90)
    print(f"{'排名':<4} {'代码':<8} {'名称':<20} {'得分':<5} {'通过':<15} {'价格':<10} {'市值':<8}")
    print("-" * 90)
    for i, r in enumerate(results[:top_n], 1):
        name_short = r.name[:18] if len(r.name) > 18 else r.name
        passed_str = ','.join(r.passed_criteria[:3])
        print(f"{i:<4} {r.ticker:<8} {name_short:<20} {r.total_score:<5} {passed_str:<15} ${r.price:<9.2f} {format_market_cap(r.market_cap):<8}")

def export_to_json(results, filepath: str):
    data = [convert_to_serializable(asdict(r)) for r in results]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已导出: {filepath}")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CAN SLIM 成长股量化筛选器')
    parser.add_argument('--watchlist', nargs='+', help='指定股票列表')
    parser.add_argument('--top', type=int, default=10, help='显示前N名 (默认10)')
    parser.add_argument('--min-score', type=int, default=25, help='最低得分门槛 (默认25)')
    parser.add_argument('--output', choices=['text', 'json'], default='text', help='输出格式')
    parser.add_argument('--export', type=str, help='导出JSON文件路径')
    parser.add_argument('--market', choices=['us', 'cn', 'all'], default='us', help='市场选择 (默认us)')
    parser.add_argument('--no-cache', action='store_true', help='禁用缓存')

    args = parser.parse_args()

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
    print(f"🦐 CAN SLIM 成长股量化筛选器 v1.3 - {market_name}")
    print("   基于威廉·欧奈尔(William J. O'Neil)投资策略")
    print("   ✨ 新增: 异常处理/重试/缓存/备用数据源")
    if ALPHAVANTAGE_AVAILABLE and args.market != 'cn':
        print("   📊 Alpha Vantage 数据增强已启用")
    print("=" * 90)

    # 清理过期缓存
    _data_cache.clear_expired()

    if args.market == 'cn':
        market_ok, market_pct = check_cn_market_direction()
        market_label = "上证指数"
    else:
        market_ok, market_pct = check_market_direction()
        market_label = "SPY"

    market_status = "✅ 上升趋势" if market_ok else "⚠️ 震荡/下降"
    print(f"\n📈 市场方向 ({market_label}): {market_status} ({market_pct:+.1f}% vs 50日均线)")

    if not market_ok:
        print("   ⚠️ 建议: 市场趋势不佳，谨慎操作或降低仓位")

    print(f"\n🔍 正在分析 {len(watchlist)} 只股票...")
    print("-" * 90)

    results = []
    for i, ticker in enumerate(watchlist, 1):
        print(f"[{i:2d}/{len(watchlist)}] {ticker:6s} ... ", end='', flush=True)

        if args.market == 'cn' or (args.market == 'all' and ticker.isdigit()):
            score = analyze_cn_stock(ticker)
        else:
            score = analyze_stock(ticker)

        if score:
            if market_ok:
                score.m_market_score = 10
                score.total_score += 10
            results.append(score)
            print(f"得分: {score.total_score:2d} | 通过: {','.join(score.passed_criteria)}")
        else:
            print("跳过 (数据不足)")
    
    results = [r for r in results if r.total_score >= args.min_score]
    results.sort(key=lambda x: x.total_score, reverse=True)
    
    if not results:
        print("\n⚠️ 没有股票达到最低得分门槛")
        return
    
    if args.output == 'json':
        data = [convert_to_serializable(asdict(r)) for r in results[:args.top]]
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print_results_table(results, args.top)
    
    if args.export:
        export_to_json(results, args.export)
    
    print("\n" + "=" * 90)
    print("⚠️ 免责声明: 本工具仅供学习研究，不构成投资建议")
    print("    股市有风险，投资需谨慎")
    print("=" * 90)


if __name__ == "__main__":
    main()
