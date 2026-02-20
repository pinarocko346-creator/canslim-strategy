#!/bin/bash
# Cron wrapper for daily brief - 输出日志方便排查
LOG_FILE="$HOME/.openclaw/workspace/canslim-strategy/logs/daily_brief_$(date +%Y%m%d).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=== $(date): Daily Brief Start ===" >> "$LOG_FILE"
cd "$HOME/.openclaw/workspace/canslim-strategy"
python3 daily_brief.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
echo "=== $(date): Exit code $EXIT_CODE ===" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
exit $EXIT_CODE
