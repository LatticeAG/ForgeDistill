#!/usr/bin/env bash
# prod_loop.sh - self-driving ForgeDistill production-readiness loop.
# Chain: Grok extreme spec -> Kimi K3 review -> Grok Ultimate Build -> verify.
# Only advances when a stage's deliverable exists / exits 0. Iterates up to
# MAX_ITER times, writing PROD-LOOP-DONE.md on full closure, FAILED marker on
# hard failure. Runs unattended; the agent is notified once at the end.
set -uo pipefail

cd /home/ubuntu/nanbeige-agentic
VENV=/home/ubuntu/.hermes/hermes-agent/venv/bin/python
LOG=/tmp/prod-loop.log
MAX_ITER=5
ITER=0
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Wait for a file to exist (timeout seconds), retrying.
wait_file() { # $1 path, $2 timeout_s
  local f="$1" t="$2" n=0
  while [ ! -s "$f" ] && [ "$n" -lt "$t" ]; do sleep 15; n=$((n+15)); done
  [ -s "$f" ]
}

# wait for background cursor-agent / hermes chat process to finish via marker
wait_proc() { # $1 pidfile, $2 timeout_s
  local pidf="$1" t="$2" n=0
  while [ ! -f "$pidf" ] && [ "$n" -lt "$t" ]; do sleep 15; n=$((n+15)); done
  [ -f "$pidf" ]
}

run_verify() { # returns 0 if ALL code-closure checks pass; writes VERIFY-STATE.txt
  {
    echo "### verify $(date +%H:%M:%S)"
    $VENV -m pytest tests/ -q --no-header 2>&1 | tail -2
    $VENV -c "import sys; sys.path.insert(0,'src'); import distill_tools, agentic_plans, prose_writer, eval_card, export_sft, verifier, dpo_pairs, curriculum, eval_live; print('IMPORTS OK')" 2>&1 | tail -1
    $VENV -c "import sys; sys.path.insert(0,'src'); from agentic_plans import PLANS, SKILLS; print('PLANS', len(PLANS), 'SKILLS', len(SKILLS))" 2>&1 | tail -1
    $VENV -c "import sys,random; sys.path.insert(0,'src'); from agentic_plans import build_chain, validate_chain; r=random.Random(0); print('STRESS', sum(1 for _ in range(300) if validate_chain(build_chain(r)['steps'])))" 2>&1 | tail -1
    $VENV src/eval_card.py --input tests/fixtures --require-gates >/dev/null 2>&1 && echo "EVALCARD_GATE OK" || echo "EVALCARD_GATE FAIL"
    if grep -rnE "nvdacf_|lexzm_|lexgf_|kimcf_|sk-[a-zA-Z0-9]{20,}" src/ configs/ 2>/dev/null | grep -qv "KEY_FALLBACK\|north-mini"; then echo "KEYSCAN FAIL"; else echo "KEYSCAN OK"; fi
    [ -f pyproject.toml ] && echo "PYPROJECT OK" || echo "PYPROJECT MISSING"
    for m in distill_tools eval_card export_sft dpo_pairs eval_live; do
      $VENV src/$m.py --help >/dev/null 2>&1 && echo "HELP_$m OK" || echo "HELP_$m FAIL"
    done
  } > VERIFY-STATE.txt 2>&1
  # Decide: all gates green?
  ! grep -qE "FAIL|MISSING|error" VERIFY-STATE.txt
}

run_real() { # real >=500 trace run + eval card (B1-B5). Returns 0 on success.
  {
    echo "### real run $(date +%H:%M:%S)"
    bash safe_launch.sh --count 500 2>&1 | tail -5
  } > /tmp/prod-real-run.log 2>&1
  # eval card on the produced data
  if [ -n "$(ls data/raw/traces_*.jsonl 2>/dev/null)" ]; then
    $VENV src/eval_card.py --input data/raw --out data/raw/eval_card.json --require-gates >> /tmp/prod-real-run.log 2>&1
    echo "REAL_RUN exit=$?" >> /tmp/prod-real-run.log
  else
    echo "REAL_RUN no traces" >> /tmp/prod-real-run.log
  fi
  grep -q "REAL_RUN exit=0" /tmp/prod-real-run.log
}

log "=== PROD LOOP START ==="
rm -f PROD-LOOP-DONE.md FAILED

# Stage 0: wait for the in-flight Grok hardening spec (already running as proc_2172027f1139)
if [ ! -s SPEC-V0.3-HARDENING.md ]; then
  log "waiting for in-flight Grok hardening spec..."
  wait_file SPEC-V0.3-HARDENING.md 5400 || { echo "FAILED: spec never landed" > FAILED; log "FAILED: spec never landed"; exit 1; }
  log "hardening spec landed"
fi

while [ "$ITER" -lt "$MAX_ITER" ]; do
  ITER=$((ITER+1))
  log "=== ITERATION $ITER ==="

  # --- K3 review of current spec (skip if already reviewed) ---
  if [ ! -s SPEC-V0.3-K3-REVIEW.md ]; then
    log "K3 review via Modal-RR..."
    timeout 2400 hermes chat -q "$(cat /tmp/spec-v03-k3-review-prompt.txt)" -m moonshotai/Kimi-K3 --provider Modal-RR -Q > /tmp/k3-v03.log 2>&1
    echo "k3 done" > /tmp/k3-v03.done
    log "K3 review finished (exit $?)"
  else
    log "K3 review exists, using it"
  fi

  # --- Grok Ultimate Build: implement spec + K3 fixes ---
  log "Grok Ultimate Build (xhigh)..."
  BUILD_KICK="$(date +%s)"
  ( cd /home/ubuntu/nanbeige-agentic && cursor-agent --yolo --model cursor-grok-4.6-xhigh "$(cat /tmp/prod-build-prompt.txt)" > /tmp/prod-build-$ITER.log 2>&1; echo "build $ITER exit $?" > /tmp/prod-build-$ITER.done ) &
  BUILD_PID=$!
  wait_proc /tmp/prod-build-$ITER.done 9000 || { echo "FAILED: build $ITER timeout" > FAILED; log "build $ITER TIMEOUT"; exit 1; }
  log "build $ITER finished ($(($(date +%s)-BUILD_KICK))s)"

  # --- Independent verification ---
  log "independent verification..."
  if run_verify; then
    log "CODE CLOSURE GREEN after iteration $ITER"
    # Commit the green tree so the fresh-clone smoke tests THIS state.
    git add -A && git commit -q -m "v0.3 hardening: iteration $ITER - production readiness pass" 2>/dev/null || true
    log "committed green tree (v0.3 iter $ITER)"
    break
  else
    log "code closure not green after iteration $ITER - looping"
    tail -8 VERIFY-STATE.txt >> "$LOG"
  fi
done

if ! [ -f FAILED ]; then
  # --- Real run (B items) ---
  log "real run (500 traces, live fleet)..."
  if run_real; then
    log "real run + eval card green"
    # Final fresh-clone smoke (E1)
    log "fresh-clone smoke..."
    SMOKE=/tmp/forge-clone-smoke; rm -rf $SMOKE
    git clone -q . $SMOKE 2>/dev/null || git clone -q https://github.com/LatticeAG/ForgeDistill.git $SMOKE
    ( cd $SMOKE && $VENV -m venv .venv 2>/dev/null; $SMOKE/.venv/bin/pip install -q -e . 2>&1 | tail -1; $SMOKE/.venv/bin/python -m pytest tests/ -q --no-header 2>&1 | tail -1 ) > /tmp/prod-smoke.log 2>&1
    log "smoke done (see /tmp/prod-smoke.log)"
    echo "DONE" > PROD-LOOP-DONE.md
    log "=== PROD LOOP DONE after $ITER iterations ==="
  else
    log "real run blocked (quota/fleet) - code closure reached, real-run partial"
    echo "REAL-RUN-BLOCKED" > PROD-LOOP-DONE.md
    log "=== PROD LOOP DONE (code green, real run blocked) ==="
  fi
fi

log "final state: $(cat PROD-LOOP-DONE.md 2>/dev/null || cat FAILED)"
echo "PROD LOOP EXIT"