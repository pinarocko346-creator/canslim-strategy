# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2.31.0",
# ]
# ///

"""
Alpha Vantage 数据源模块
用于获取美股技术指标和基本面数据

免费 API 限制: 5次/分钟, 500次/天
获取 API Key: https://www.alphavantage.co/support/#api-key
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import requests

# 从环境变量读取 API Key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# 缓存控制
_cache = {}
_cache_time = {}
CACHE_DURATION = 300  # 5分钟缓存

# 最后请求时间（用于限流）
_last_request_time = 0
MIN_REQUEST_INTERVAL = 12  # 免费版: 5次/分钟 = 12秒间隔


def _rate_limit():
    """限流控制 - 确保不超过 5次/分钟"""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - elapsed
        print(f"⏳ 等待 {sleep_time:.1f} 秒 (Alpha Vantage 限流)...")
        time.sleep(sleep_time)
    _last_request_time = time.time()


def _get_cache(key: str) -> Optional[Dict]:
    """获取缓存数据"""
    if key in _cache and key in _cache_time:
        if time.time() - _cache_time[key] < CACHE_DURATION:
            return _cache[key]
    return None


def _set_cache(key: str, data: Dict):
    """设置缓存数据"""
    _cache[key] = data
    _cache_time[key] = time.time()


def get_quote(symbol: str) -> Optional[Dict]:
    """获取实时报价 (Global Quote)"""
    if not ALPHA_VANTAGE_API_KEY:
        print("⚠️  未设置 ALPHA_VANTAGE_API_KEY")
        return None
    
    cache_key = f"quote_{symbol}"
    cached = _get_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit()
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            result = {
                "symbol": quote.get("01. symbol"),
                "price": float(quote.get("05. price", 0)),
                "change": float(quote.get("09. change", 0)),
                "change_percent": quote.get("10. change percent", "0%"),
                "volume": int(quote.get("06. volume", 0)),
                "latest_trading_day": quote.get("07. latest trading day"),
            }
            _set_cache(cache_key, result)
            return result
        else:
            print(f"⚠️  Alpha Vantage 返回空数据: {data.get('Note', data.get('Information', 'Unknown'))}")
            return None
    except Exception as e:
        print(f"❌ 获取报价失败: {e}")
        return None


def get_technical_indicator(symbol: str, indicator: str, interval: str = "daily", time_period: int = 14) -> Optional[Dict]:
    """
    获取技术指标
    
    常用指标:
    - RSI
    - SMA (简单移动平均)
    - EMA (指数移动平均)
    - MACD
    - BBANDS (布林带)
    - VWAP (成交量加权平均价)
    """
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    cache_key = f"ti_{symbol}_{indicator}_{interval}_{time_period}"
    cached = _get_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit()
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": indicator,
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        # 技术指标返回格式不同，直接返回原始数据
        _set_cache(cache_key, data)
        return data
    except Exception as e:
        print(f"❌ 获取技术指标失败: {e}")
        return None


def get_fundamentals(symbol: str) -> Optional[Dict]:
    """获取基本面数据 (公司概况)"""
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    cache_key = f"fundamentals_{symbol}"
    cached = _get_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit()
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data and "Symbol" in data:
            _set_cache(cache_key, data)
            return data
        else:
            print(f"⚠️  Alpha Vantage 基本面数据为空")
            return None
    except Exception as e:
        print(f"❌ 获取基本面数据失败: {e}")
        return None


def get_income_statement(symbol: str) -> Optional[Dict]:
    """获取利润表 (用于 CANSLIM C - Current Earnings)"""
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    cache_key = f"income_{symbol}"
    cached = _get_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit()
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "INCOME_STATEMENT",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if "quarterlyReports" in data:
            _set_cache(cache_key, data)
            return data
        return None
    except Exception as e:
        print(f"❌ 获取利润表失败: {e}")
        return None


def get_earnings_calendar() -> Optional[List[Dict]]:
    """获取财报日历 (即将发布财报的股票)"""
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    cache_key = "earnings_calendar"
    cached = _get_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit()
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "EARNINGS_CALENDAR",
        "horizon": "3month",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        # 返回的是 CSV 格式
        if response.status_code == 200:
            # 解析 CSV
            lines = response.text.strip().split('\n')
            headers = lines[0].split(',')
            results = []
            for line in lines[1:10]:  # 只取前10条
                values = line.split(',')
                if len(values) >= 4:
                    results.append({
                        'symbol': values[0],
                        'name': values[1],
                        'report_date': values[2],
                        'fiscal_date_ending': values[3],
                    })
            _set_cache(cache_key, results)
            return results
        return None
    except Exception as e:
        print(f"❌ 获取财报日历失败: {e}")
        return None


# 简单测试
if __name__ == "__main__":
    if not ALPHA_VANTAGE_API_KEY:
        print("❌ 请设置环境变量 ALPHA_VANTAGE_API_KEY")
        print("获取免费 API Key: https://www.alphavantage.co/support/#api-key")
        exit(1)
    
    print("🦐 Alpha Vantage 测试")
    print("=" * 50)
    
    # 测试 1: 实时报价
    print("\n1. 实时报价 (AAPL):")
    quote = get_quote("AAPL")
    if quote:
        print(f"   价格: ${quote['price']}")
        print(f"   涨跌: {quote['change']} ({quote['change_percent']})")
        print(f"   成交量: {quote['volume']:,}")
    
    # 测试 2: 技术指标 (RSI)
    print("\n2. RSI 指标 (AAPL):")
    rsi_data = get_technical_indicator("AAPL", "RSI", time_period=14)
    if rsi_data and "Technical Analysis: RSI" in rsi_data:
        dates = list(rsi_data["Technical Analysis: RSI"].keys())[:3]
        for date in dates:
            print(f"   {date}: RSI = {rsi_data['Technical Analysis: RSI'][date]['RSI']}")
    
    # 测试 3: 基本面
    print("\n3. 基本面 (AAPL):")
    fundamentals = get_fundamentals("AAPL")
    if fundamentals:
        print(f"   行业: {fundamentals.get('Industry', 'N/A')}")
        print(f"   市值: {fundamentals.get('MarketCapitalization', 'N/A')}")
        print(f"   P/E: {fundamentals.get('PERatio', 'N/A')}")
        print(f"   营收增长: {fundamentals.get('QuarterlyRevenueGrowthYOY', 'N/A')}")
    
    print("\n✅ 测试完成!")
