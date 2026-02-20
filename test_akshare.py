# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "akshare>=1.15.0",
# ]
# ///

"""
AkShare 测试脚本 - 验证 A 股数据获取
"""

import akshare as ak
import json
from datetime import datetime

def test_spot():
    """测试实时行情"""
    print("=" * 60)
    print("🦐 AkShare 测试 - 实时行情")
    print("=" * 60)
    
    try:
        # 获取 A 股实时行情（前10只）
        df = ak.stock_zh_a_spot_em()
        print(f"✅ 获取成功！共 {len(df)} 只股票")
        print(f"\n前 5 只股票:")
        print(df[['代码', '名称', '最新价', '涨跌幅', '换手率']].head())
        return True
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        return False

def test_hist():
    """测试历史数据"""
    print("\n" + "=" * 60)
    print("🦐 AkShare 测试 - 历史数据 (600519 贵州茅台)")
    print("=" * 60)
    
    try:
        df = ak.stock_zh_a_hist(symbol="600519", period="daily", 
                                start_date="20250101", adjust="qfq")
        print(f"✅ 获取成功！共 {len(df)} 条记录")
        print(f"\n最近 5 天:")
        print(df[['日期', '开盘', '收盘', '最高', '最低', '成交量']].tail())
        return True
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        return False

def test_financial():
    """测试财务数据"""
    print("\n" + "=" * 60)
    print("🦐 AkShare 测试 - 财务报表")
    print("=" * 60)
    
    try:
        df = ak.stock_yjbb_em(date="20241231")
        print(f"✅ 获取成功！共 {len(df)} 条记录")
        print(f"\n前 5 条业绩报表:")
        print(df[['股票代码', '股票简称', '营业收入', '净利润', '营收同比增长']].head())
        return True
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        return False

def test_fund_flow():
    """测试资金流向"""
    print("\n" + "=" * 60)
    print("🦐 AkShare 测试 - 个股资金流向 (000001 平安银行)")
    print("=" * 60)
    
    try:
        df = ak.stock_individual_fund_flow(symbol="000001", market="sh")
        print(f"✅ 获取成功！共 {len(df)} 条记录")
        print(f"\n最近 5 天资金流向:")
        print(df.head())
        return True
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print(f"🦐 AkShare 版本: {ak.__version__}")
    print("=" * 60)
    
    results = []
    
    results.append(("实时行情", test_spot()))
    results.append(("历史数据", test_hist()))
    results.append(("财务报表", test_financial()))
    results.append(("资金流向", test_fund_flow()))
    
    print("\n" + "=" * 60)
    print("🦐 测试结果汇总")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_pass = all(r[1] for r in results)
    
    if all_pass:
        print("\n✅ 所有测试通过！AkShare 可以正常使用")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查网络或 akshare 版本")
        return 1

if __name__ == "__main__":
    exit(main())
