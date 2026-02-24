#!/usr/bin/env python3
"""
CAN SLIM 策略回测运行器

使用方法:
    python run_backtest.py --start 2023-01-01 --end 2024-01-01 --capital 100000
"""

import argparse
import json
from datetime import datetime, timedelta
from backtest_simple import SimpleBacktest, BacktestConfig, fetch_sp500_list
from canslim_scanner import CanslimScanner


def run_backtest_with_canslim(start_date: str, end_date: str, 
                              capital: float = 100000,
                              top_n: int = 50,
                              score_threshold: int = 70):
    """
    使用 CAN SLIM 评分运行回测
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        capital: 初始资金
        top_n: 扫描股票数量
        score_threshold: 买入阈值
    """
    print("=" * 70)
    print("CAN SLIM 策略回测")
    print("=" * 70)
    print(f"回测期间: {start_date} ~ {end_date}")
    print(f"初始资金: ${capital:,.2f}")
    print(f"买入阈值: {score_threshold}分")
    print("=" * 70)
    
    # 获取股票列表
    print("\n📊 获取股票池...")
    tickers = fetch_sp500_list()[:top_n]
    print(f"   共 {len(tickers)} 只股票")
    
    # 配置回测
    config = BacktestConfig(
        initial_capital=capital,
        score_threshold=score_threshold,
        max_positions=10,
        hold_days=30,
        stop_loss=0.08,
        take_profit=0.20
    )
    
    # 逐月扫描获取信号
    print("\n🔍 逐月扫描 CAN SLIM 信号...")
    scanner = CanslimScanner()
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    all_signals = {}
    current_date = start_dt
    
    while current_date <= end_dt:
        print(f"\n   📅 {current_date.strftime('%Y-%m')}")
        
        # 扫描当月信号
        month_signals = []
        for ticker in tickers:
            try:
                result = scanner.analyze_stock(ticker)
                if result and result['total_score'] >= score_threshold:
                    month_signals.append({
                        'ticker': ticker,
                        'score': result['total_score'],
                        'date': current_date
                    })
            except Exception as e:
                continue
        
        # 按分数排序，取前10
        month_signals.sort(key=lambda x: x['score'], reverse=True)
        top_signals = month_signals[:10]
        
        for signal in top_signals:
            ticker = signal['ticker']
            if ticker not in all_signals:
                all_signals[ticker] = []
            all_signals[ticker].append(current_date)
            print(f"      ✓ {ticker}: {signal['score']}分")
        
        # 下一个月
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    print(f"\n   共找到 {sum(len(v) for v in all_signals.values())} 个买入信号")
    
    # 下载价格数据
    print("\n💾 下载价格数据...")
    import yfinance as yf
    
    stock_data = {}
    for ticker in all_signals.keys():
        try:
            df = yf.download(
                ticker, 
                start=start_date, 
                end=end_date,
                progress=False
            )
            if len(df) > 20:
                stock_data[ticker] = df
        except Exception as e:
            print(f"   ✗ {ticker}: {e}")
    
    print(f"   成功获取 {len(stock_data)} 只股票数据")
    
    # 运行回测
    print("\n🚀 运行回测...")
    backtest = SimpleBacktest(config)
    results = backtest.run_quick_backtest(stock_data, all_signals)
    
    # 显示结果
    print("\n" + "=" * 70)
    print("📈 回测结果")
    print("=" * 70)
    
    for key, value in results.items():
        print(f"   {key:20s}: {value}")
    
    # 保存报告
    report = {
        "backtest_config": {
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": capital,
            "score_threshold": score_threshold,
            "max_positions": config.max_positions,
            "hold_days": config.hold_days
        },
        "results": results,
        "generated_at": datetime.now().isoformat()
    }
    
    output_file = f"backtest_report_{start_date}_{end_date}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细报告已保存: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='CAN SLIM 策略回测')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, 
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='初始资金 (默认: 100000)')
    parser.add_argument('--top', type=int, default=50,
                       help='扫描股票数量 (默认: 50)')
    parser.add_argument('--threshold', type=int, default=70,
                       help='买入阈值 (默认: 70)')
    parser.add_argument('--demo', action='store_true',
                       help='运行快速演示 (不使用真实CAN SLIM评分)')
    
    args = parser.parse_args()
    
    if args.demo:
        # 运行简化演示
        from backtest_simple import run_backtest_demo
        run_backtest_demo()
    else:
        # 运行完整回测
        run_backtest_with_canslim(
            start_date=args.start,
            end_date=args.end,
            capital=args.capital,
            top_n=args.top,
            score_threshold=args.threshold
        )


if __name__ == "__main__":
    main()
