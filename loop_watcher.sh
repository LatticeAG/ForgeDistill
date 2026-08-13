#!/usr/bin/env bash
# loop_watcher.sh - tail prod-loop.log and ping the Discord thread after each stage.
set -uo pipefail
LOG=/tmp/prod-loop.log
TARGET="discord:1537388273654308942"
LAST=0
LASTLINE=""
SENT_LINES=""

ping() { # $1 = message text
  hermes send --to "$TARGET" "$1" 2>/dev/null && echo "  [watcher] pinged: $1" | tee -a /tmp/loop-watcher.log
}

echo "[watcher] started $(date +%H:%M:%S)" | tee /tmp/loop-watcher.log
# initial state ping
ping "🔁 **ForgeDistill prod loop is running** - I'll ping you after every stage.

Current stage: Grok 4.6 Extreme spec already landed (v0.3 hardening, 46KB) -> K3 review -> Grok Ultimate Build -> verify -> repeat until PROD-READINESS is green."

while true; do
  # new lines since last read
  NEW=$(tail -n +$((LAST+1)) "$LOG" 2>/dev/null | head -100)
  if [ -n "$NEW" ]; then
    while IFS= read -r line; do
      case "$line" in
        *"=== ITERATION"*)
          ping "🔄 **Iteration ${line##*ITERATION }** starting - Grok spec -> K3 -> build -> verify cycle $(( $(grep -c '=== ITERATION' "$LOG") ))"
          ;;
        *"K3 review finished"*)
          ping "✅ **K3 review done** (Modal-RR / Kimi K3). Verdict written to SPEC-V0.3-K3-REVIEW.md. Next: Grok Ultimate Build."
          ;;
        *"build "*"finished"*)
          ping "✅ **Grok Ultimate Build done** (${line##*\(}s). Next: independent verification (pytest, imports, plan counts, stress, key scan)."
          ;;
        *"CODE CLOSURE GREEN"*)
          ping "✅ **Code closure GREEN** - all gates pass. Committing, then real 500-trace fleet run + eval card."
          ;;
        *"code closure not green"*)
          ping "⚠️ **Code closure NOT green** - looping again. See VERIFY-STATE.txt. Next iteration: Grok spec -> K3 -> build."
          ;;
        *"real run + eval card green"*)
          ping "✅ **Real run GREEN** - 500 traces, eval card passes --require-gates. Next: fresh-clone smoke test."
          ;;
        *"real run blocked"*)
          ping "⚠️ **Real run blocked** (quota/fleet) - code closure reached but live-run numbers are partial. Finishing smoke test."
          ;;
        *"PROD LOOP DONE"*)
          ping "🎉 **PROD LOOP DONE** - ForgeDistill is production-ready. Full report incoming."
          ;;
        *"FAILED:"*)
          ping "❌ **PROD LOOP FAILED**: ${line#*FAILED: }. Check /tmp/prod-loop.log."
          ;;
      esac
    done <<< "$NEW"
  fi
  LAST=$(wc -l < "$LOG")
  # exit when loop wrote its final marker
  if [ -f /home/ubuntu/nanbeige-agentic/PROD-LOOP-DONE.md ] || [ -f /home/ubuntu/nanbeige-agentic/FAILED ]; then
    sleep 5
    ping "🏁 **Loop finished** - marker: $(cat /home/ubuntu/nanbeige-agentic/PROD-LOOP-DONE.md 2>/dev/null || cat /home/ubuntu/nanbeige-agentic/FAILED 2>/dev/null)"
    exit 0
  fi
  sleep 20
done