#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yfinance>=0.2.40",
# ]
# ///

"""
Daily Briefing Bot - 每日简报发送脚本
独立运行，不依赖 agent，直接推送 Discord
"""

import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Discord Webhook
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1473795489718734870/ET-se53d6MS02GOc_E4c3GUFNwq9KtVQS15eo6pimbD4aQF0d675x0fuTIHzEiRj2ESh"

# 工作目录
WORKSPACE = Path.home() / ".openclaw/workspace/canslim-strategy"


def run_command(cmd: str, cwd: Path = None) -> tuple:
    """运行 shell 命令，返回 (stdout, stderr, returncode)"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=60
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", str(e), 1


def get_weather() -> str:
    """获取上海天气"""
    stdout, _, _ = run_command('curl -s "wttr.in/Shanghai?format=%c+%t"')
    return stdout if stdout else "☀️ +20°C"


def get_market_trend() -> dict:
    """获取 SPY 和 QQQ 市场趋势"""
    try:
        import yfinance as yf
        
        def get_trend(ticker):
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="3mo")
                if len(hist) < 50:
                    return None
                price = hist['Close'].iloc[-1]
                sma50 = hist['Close'].rolling(50).mean().iloc[-1]
                diff = (price - sma50) / sma50 * 100
                
                # 判断趋势
                if diff > 1:
                    trend = "📈 上升"
                elif diff < -1:
                    trend = "📉 下降"
                else:
                    trend = "⚠️ 震荡"
                
                return {"trend": trend, "diff": diff, "price": price}
            except Exception as e:
                print(f"Error getting {ticker}: {e}", file=sys.stderr)
                return None
        
        spy = get_trend("SPY")
        qqq = get_trend("QQQ")
        
        result = []
        if spy:
            result.append(f"SPY {spy['trend']} ({spy['diff']:+.1f}%)")
        if qqq:
            result.append(f"QQQ {qqq['trend']} ({qqq['diff']:+.1f}%)")
        
        raw = " | ".join(result) if result else "SPY/QQQ: 数据获取失败"
        
        return {
            "spy": f"SPY {spy['trend']}" if spy else "⚠️ 震荡",
            "qqq": f"QQQ {qqq['trend']}" if qqq else "⚠️ 震荡",
            "raw": raw
        }
    except Exception as e:
        print(f"Market trend error: {e}", file=sys.stderr)
        return {
            "spy": "⚠️ 震荡",
            "qqq": "⚠️ 震荡",
            "raw": "SPY: 数据获取失败 | QQQ: 数据获取失败"
        }


def get_canslim_stocks() -> list:
    """获取 CANSLIM 选股结果"""
    # 先导出到临时文件，避免 stdout 被日志污染
    temp_file = "/tmp/canslim_output.json"
    
    stdout, stderr, rc = run_command(
        f"uv run canslim_scanner.py --top 5 --output json --export {temp_file}",
        cwd=WORKSPACE
    )
    
    if rc != 0:
        print(f"CANSLIM scanner failed: {stderr}", file=sys.stderr)
        return []
    
    try:
        with open(temp_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"JSON parse error: {e}", file=sys.stderr)
        return []
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)


def format_stock_line(stock: dict, rank: int) -> str:
    """格式化单行股票信息"""
    ticker = stock.get('ticker', 'N/A')
    score = stock.get('total_score', 0)
    price = stock.get('price', 0)
    dist_high = stock.get('n_distance_from_high', 0)
    
    # 距离52周高格式化
    if dist_high is not None:
        dist_str = f"距高{dist_high:.1f}%"
    else:
        dist_str = "距高N/A"
    
    return f"{rank}. {ticker} {score}分 ${price:.0f} {dist_str}"


def build_message() -> str:
    """构建完整简报消息"""
    today = datetime.now().strftime("%m-%d")
    weather = get_weather()
    market = get_market_trend()
    stocks = get_canslim_stocks()
    
    lines = [
        f"🌅 简报 | {today}",
        f"🌤️ 上海: {weather}",
        f"📈 {market['raw']}",
        ""
    ]
    
    # CANSLIM Top 5
    if stocks:
        lines.append("🦐 CAN SLIM Top 5")
        for i, stock in enumerate(stocks[:5], 1):
            lines.append(format_stock_line(stock, i))
        
        # 板块重点
        sectors = set()
        for stock in stocks[:5]:
            ticker = stock.get('ticker', '')
            if ticker in ['AMAT', 'LRCX', 'KLAC']:
                sectors.add("半导体设备")
            elif ticker == 'MU':
                sectors.add("存储芯片")
            elif ticker == 'NVDA':
                sectors.add("AI芯片")
            elif ticker == 'AAPL':
                sectors.add("消费电子")
        
        if sectors:
            lines.append("")
            lines.append(f"💡 重点: {', '.join(sectors)}强势")
    else:
        lines.append("🦐 CAN SLIM: 数据获取失败")
    
    lines.append("")
    lines.append("⚠️ 免责声明: 仅供参考，不构成投资建议")
    
    return '\n'.join(lines)


def send_discord(message: str) -> bool:
    """发送 Discord webhook"""
    payload = json.dumps({"content": message}, ensure_ascii=False)
    
    cmd = [
        "curl", "-X", "POST",
        "-H", "Content-Type: application/json",
        "-d", payload,
        DISCORD_WEBHOOK
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Discord 发送成功")
            return True
        else:
            print(f"❌ Discord 发送失败: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"❌ Discord 发送异常: {e}", file=sys.stderr)
        return False


def main():
    print("=" * 50)
    print("🦐 Daily Briefing Bot - 启动")
    print("=" * 50)
    
    # 构建消息
    message = build_message()
    
    print("\n📋 生成的简报:")
    print("-" * 50)
    print(message)
    print("-" * 50)
    
    # 统计字符数
    char_count = len(message)
    print(f"\n📊 字符数: {char_count} (Discord限制: 2000)")
    
    if char_count > 2000:
        print("⚠️ 警告: 消息超过 Discord 限制，将被截断")
        message = message[:1997] + "..."
    
    # 发送
    print("\n📤 正在发送 Discord...")
    success = send_discord(message)
    
    if success:
        print("\n✅ 简报发送完成")
        return 0
    else:
        print("\n❌ 简报发送失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
