#!/usr/bin/env python3
"""
CAN SLIM 回测框架 - 简化版
验证策略在历史数据上的表现
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0  # 初始资金
    commission_rate: float = 0.001     # 佣金 0.1%
    slippage: float = 0.0005           # 滑点 0.05%
    max_positions: int = 10            # 最大持仓数
    hold_days: int = 30                # 持仓周期
    score_threshold: int = 70          # 买入阈值
    stop_loss: float = 0.08            # 止损 8%
    take_profit: float = 0.20          # 止盈 20%

@dataclass
class Trade:
    """交易记录"""
    ticker: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""

class SimpleBacktest:
    """简化回测引擎"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Trade] = {}
        self.trades: List[Trade] = []
        self.daily_values: List[Tuple[datetime, float]] = []
        
    def calculate_metrics(self, trades: List[Trade]) -> Dict:
        """计算回测指标"""
        if not trades:
            return {"error": "无交易记录"}
            
        closed_trades = [t for t in trades if t.exit_date is not None]
        if not closed_trades:
            return {"error": "无完成的交易"}
            
        profits = [t.pnl for t in closed_trades]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        total_profit = sum(profits)
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # 计算最大回撤
        values = [v for _, v in self.daily_values]
        peak = 0
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        # 年化收益率
        if len(self.daily_values) >= 2:
            start_val = self.daily_values[0][1]
            end_val = self.daily_values[-1][1]
            days = (self.daily_values[-1][0] - self.daily_values[0][0]).days
            annual_return = ((end_val / start_val) ** (365.25 / days) - 1) if days > 0 else 0
        else:
            annual_return = 0
            
        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": f"{win_rate:.2%}",
            "total_pnl": f"${total_profit:,.2f}",
            "avg_profit": f"${np.mean(profits):,.2f}",
            "profit_factor": f"{profit_factor:.2f}",
            "max_drawdown": f"{max_dd:.2%}",
            "annual_return": f"{annual_return:.2%}",
            "final_value": f"${self.cash:,.2f}"
        }
    
    def run_quick_backtest(self, stock_data: Dict[str, pd.DataFrame], 
                          signals: Dict[str, List[datetime]]) -> Dict:
        """
        快速回测 - 基于已有信号
        
        Args:
            stock_data: {ticker: price_df} 价格数据
            signals: {ticker: [buy_dates]} 买入信号日期
        """
        print(f"开始回测... 初始资金: ${self.config.initial_capital:,.2f}")
        
        # 获取所有交易日
        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df.index)
        sorted_dates = sorted(all_dates)
        
        for current_date in sorted_dates:
            # 检查现有持仓是否需要卖出
            for ticker, trade in list(self.positions.items()):
                if ticker not in stock_data:
                    continue
                    
                df = stock_data[ticker]
                current_data = df[df.index <= current_date]
                if len(current_data) < 2:
                    continue
                    
                current_price = current_data['Close'].iloc[-1]
                entry_price = trade.entry_price
                
                # 止损/止盈检查
                pnl_pct = (current_price - entry_price) / entry_price
                hold_days = (current_date - trade.entry_date).days
                
                exit_reason = None
                if pnl_pct <= -self.config.stop_loss:
                    exit_reason = "止损"
                elif pnl_pct >= self.config.take_profit:
                    exit_reason = "止盈"
                elif hold_days >= self.config.hold_days:
                    exit_reason = "持仓到期"
                
                if exit_reason:
                    # 执行卖出
                    exit_price = current_price * (1 - self.config.slippage)
                    trade.exit_date = current_date
                    trade.exit_price = exit_price
                    trade.pnl = (exit_price - entry_price) * trade.shares
                    trade.pnl_pct = pnl_pct
                    trade.exit_reason = exit_reason
                    
                    self.cash += exit_price * trade.shares * (1 - self.config.commission_rate)
                    self.trades.append(trade)
                    del self.positions[ticker]
                    
            # 检查新买入信号
            for ticker, dates in signals.items():
                if current_date in dates and ticker in stock_data:
                    if ticker in self.positions:
                        continue
                    if len(self.positions) >= self.config.max_positions:
                        break
                        
                    df = stock_data[ticker]
                    current_data = df[df.index <= current_date]
                    if len(current_data) < 2:
                        continue
                    
                    # 计算买入金额（等权重）
                    position_size = self.cash / (self.config.max_positions - len(self.positions))
                    entry_price = current_data['Close'].iloc[-1] * (1 + self.config.slippage)
                    shares = int(position_size / entry_price)
                    
                    if shares > 0 and self.cash >= entry_price * shares * (1 + self.config.commission_rate):
                        trade = Trade(
                            ticker=ticker,
                            entry_date=current_date,
                            entry_price=entry_price,
                            shares=shares
                        )
                        self.positions[ticker] = trade
                        self.cash -= entry_price * shares * (1 + self.config.commission_rate)
            
            # 记录每日净值
            total_value = self.cash
            for ticker, trade in self.positions.items():
                if ticker in stock_data:
                    df = stock_data[ticker]
                    current_data = df[df.index <= current_date]
                    if len(current_data) > 0:
                        current_price = current_data['Close'].iloc[-1]
                        total_value += current_price * trade.shares
            
            self.daily_values.append((current_date, total_value))
        
        # 强制平仓所有剩余持仓
        for ticker, trade in list(self.positions.items()):
            if ticker in stock_data:
                df = stock_data[ticker]
                if len(df) > 0:
                    exit_price = df['Close'].iloc[-1] * (1 - self.config.slippage)
                    trade.exit_date = df.index[-1]
                    trade.exit_price = exit_price
                    trade.pnl = (exit_price - trade.entry_price) * trade.shares
                    trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
                    trade.exit_reason = "回测结束"
                    self.trades.append(trade)
        
        self.positions.clear()
        
        return self.calculate_metrics(self.trades)


def fetch_sp500_list() -> List[str]:
    """获取标普500成分股列表（简化版 - 使用常见股票）"""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "UNH", "JNJ", "XOM", "JPM", "V", "PG", "HD", "CVX", "MA", "LLY",
        "ABBV", "PFE", "KO", "PEP", "COST", "TMO", "AVGO", "DIS", "ABT",
        "WMT", "BAC", "MRK", "CSCO", "ACN", "VZ", "NEE", "TXN", "CMCSA",
        "ADBE", "CRM", "PM", "NKE", "RTX", "HON", "BMY", "QCOM", "T",
        "UNP", "NFLX", "AMD", "UPS", "LIN", "LOW", "AMGN", "SPGI", "INTU"
    ]


def run_backtest_demo():
    """运行回测演示"""
    print("=" * 60)
    print("CAN SLIM 策略回测演示")
    print("=" * 60)
    
    # 配置
    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        max_positions=5,
        hold_days=20,
        score_threshold=70
    )
    
    # 获取股票列表
    tickers = fetch_sp500_list()[:10]  # 先用10只测试
    print(f"\n股票池: {', '.join(tickers)}")
    
    # 下载数据
    print("\n下载历史数据...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    stock_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 50:
                stock_data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} 天数据")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")
    
    if not stock_data:
        print("错误: 无法获取数据")
        return
    
    # 生成模拟信号（实际应从 canslim_scanner.py 获取）
    print("\n生成买入信号...")
    signals = {}
    for ticker, df in stock_data.items():
        # 简化的信号生成：每月第一天买入
        buy_dates = [d for d in df.index if d.day == 1]
        if buy_dates:
            signals[ticker] = buy_dates[:3]  # 每只股票最多3个信号
            print(f"  {ticker}: {len(buy_dates[:3])} 个信号")
    
    # 运行回测
    print("\n开始回测...")
    backtest = SimpleBacktest(config)
    results = backtest.run_quick_backtest(stock_data, signals)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key:20s}: {value}")
    
    # 保存结果
    output_file = f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    run_backtest_demo()
