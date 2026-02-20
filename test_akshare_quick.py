# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "akshare>=1.15.0",
# ]
# ///

"""
AkShare 快速测试
"""

import akshare as ak

print(f"✅ AkShare 版本: {ak.__version__}")
print("✅ 安装成功！可以开始使用 A 股数据")

# 快速测试 - 获取上证指数
print("\n🦐 测试: 上证指数历史数据（最近5天）")
df = ak.index_zh_a_hist(symbol="000001", period="daily", 
                        start_date="20250215", end_date="20250220")
print(df[['日期', '收盘', '涨跌幅']].to_string(index=False))
