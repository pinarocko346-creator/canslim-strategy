#!/bin/bash
# Cron wrapper for daily brief - 输出日志方便排查

# 加载用户配置（获取环境变量如 ALPHA_VANTAGE_API_KEY）
[ -f "$HOME/.zshrc" ] && source "$HOME/.zshrc" 2>/dev/null
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null

LOG_FILE="$HOME/.openclaw/workspace/canslim-strategy/logs/daily_brief_$(date +%Y%m%d).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=== $(date): Daily Brief Start ===" >> "$LOG_FILE"
echo "Alpha Vantage Key: ${ALPHA_VANTAGE_API_KEY:0:5}..." >> "$LOG_FILE"
cd "$HOME/.openclaw/workspace/canslim-strategy"
uv run daily_brief.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
echo "=== $(date): Exit code $EXIT_CODE ===" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
exit $EXIT_CODE
