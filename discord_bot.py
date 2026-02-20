#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "discord.py>=2.3.0",
#     "yfinance>=0.2.40",
#     "akshare>=1.15.0",
#     "pandas>=2.0.0",
# ]
# ///

"""
Discord Bot for CANSLIM Strategy
Discord 交互命令机器人

用法:
    uv run discord_bot.py

命令:
    !stock <代码> [us/cn]  - 分析股票 (默认us)
    !canslim [us/cn/all]   - CANSLIM扫描
    !market                - 市场趋势
    !help                  - 帮助信息
"""

import os
import sys
import json
import asyncio
import discord
from discord.ext import commands
from datetime import datetime
from typing import Optional

# Discord Bot Token (从环境变量读取)
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

# 配置命令前缀
bot = commands.Bot(command_prefix='!', intents=discord.Intents.default())


class CanslimBot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name='stock')
    async def stock(self, ctx, ticker: str, market: str = "us"):
        """分析单只股票"""
        await ctx.send(f"🦐 正在分析 **{ticker.upper()}** ({market.upper()})...")
        
        try:
            # 调用 canslim_scanner.py 分析单只股票
            import subprocess
            
            cmd = f"cd ~/.openclaw/workspace/canslim-strategy && uv run canslim_scanner.py --market {market} --watchlist {ticker.upper()} --top 1 --min-score 0 --output json"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout:
                # 解析结果
                lines = result.stdout.strip().split('\n')
                # 找到 JSON 部分
                json_start = None
                for i, line in enumerate(lines):
                    if line.strip().startswith('['):
                        json_start = i
                        break
                
                if json_start is not None:
                    json_str = '\n'.join(lines[json_start:])
                    data = json.loads(json_str)
                    if data and len(data) > 0:
                        stock = data[0]
                        msg = f"""📊 **{stock['ticker']}** - {stock['name']}
💯 得分: **{stock['total_score']}**/100
💰 价格: ${stock['price']:.2f}
🎯 距52周高: {stock['n_distance_from_high']:.1f}%
💪 RSI: {stock['l_rsi']:.1f}
✅ 通过: {', '.join(stock['passed_criteria'])}"""
                        await ctx.send(msg)
                    else:
                        await ctx.send("⚠️ 未获取到数据")
                else:
                    await ctx.send("⚠️ 解析结果失败")
            else:
                await ctx.send(f"❌ 分析失败: {result.stderr[:200]}")
                
        except Exception as e:
            await ctx.send(f"❌ 错误: {str(e)[:200]}")

    @commands.command(name='canslim')
    async def canslim(self, ctx, market: str = "us"):
        """运行 CANSLIM 扫描"""
        await ctx.send(f"🦐 正在运行 CANSLIM 扫描 ({market.upper()})...")
        
        try:
            import subprocess
            cmd = f"cd ~/.openclaw/workspace/canslim-strategy && uv run canslim_scanner.py --market {market} --top 5 --min-score 40 --output json"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                json_start = None
                for i, line in enumerate(lines):
                    if line.strip().startswith('['):
                        json_start = i
                        break
                
                if json_start is not None:
                    json_str = '\n'.join(lines[json_start:])
                    data = json.loads(json_str)
                    
                    if data and len(data) > 0:
                        msg = f"🏆 **CANSLIM Top {min(5, len(data))}** ({market.upper()})\n\n"
                        for i, stock in enumerate(data[:5], 1):
                            msg += f"**{i}. {stock['ticker']}** - {stock['total_score']}分 | ${stock['price']:.0f} | 距高{stock['n_distance_from_high']:.1f}%\n"
                        await ctx.send(msg)
                    else:
                        await ctx.send("⚠️ 没有股票达到最低得分门槛")
                else:
                    await ctx.send("⚠️ 解析结果失败")
            else:
                await ctx.send(f"❌ 扫描失败")
                
        except Exception as e:
            await ctx.send(f"❌ 错误: {str(e)[:200]}")

    @commands.command(name='market')
    async def market(self, ctx):
        """查看市场趋势"""
        await ctx.send("🦐 正在获取市场数据...")
        
        try:
            import yfinance as yf
            
            # SPY
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="3mo")
            spy_current = spy_hist['Close'].iloc[-1]
            spy_sma50 = spy_hist['Close'].rolling(50).mean().iloc[-1]
            spy_diff = (spy_current / spy_sma50 - 1) * 100
            
            # QQQ
            qqq = yf.Ticker("QQQ")
            qqq_hist = qqq.history(period="3mo")
            qqq_current = qqq_hist['Close'].iloc[-1]
            qqq_sma50 = qqq_hist['Close'].rolling(50).mean().iloc[-1]
            qqq_diff = (qqq_current / qqq_sma50 - 1) * 100
            
            msg = f"""📈 **市场趋势**

**SPY**: ${spy_current:.2f} ({spy_diff:+.1f}% vs 50日MA)
**QQQ**: ${qqq_current:.2f} ({qqq_diff:+.1f}% vs 50日MA)

趋势: {'📈 上升' if spy_diff > 0 else '📉 下降'}"""
            
            await ctx.send(msg)
            
        except Exception as e:
            await ctx.send(f"❌ 获取失败: {str(e)[:200]}")

    @commands.command(name='help')
    async def help_command(self, ctx):
        """显示帮助信息"""
        msg = """🦐 **CANSLIM Bot 命令列表**

`!stock <代码> [us/cn]` - 分析单只股票
  例: `!stock AAPL` 或 `!stock 600519 cn`

`!canslim [us/cn/all]` - 运行 CANSLIM 扫描
  例: `!canslim us`

`!market` - 查看市场趋势 (SPY/QQQ)

`!help` - 显示此帮助信息

---
⚠️ 免责声明: 仅供参考，不构成投资建议
"""
        await ctx.send(msg)


@bot.event
async def on_ready():
    print(f'🦐 Bot 已登录: {bot.user.name} ({bot.user.id})')
    print('------')


async def main():
    if not DISCORD_TOKEN:
        print("❌ 错误: 未设置 DISCORD_BOT_TOKEN 环境变量")
        print("请设置: export DISCORD_BOT_TOKEN='你的Bot Token'")
        sys.exit(1)
    
    await bot.add_cog(CanslimBot(bot))
    await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
