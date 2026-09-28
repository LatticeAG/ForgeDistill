#!/usr/bin/env python3
"""distill_tools.py - reversed-distillation generator for agentic tool-calling.

Pipeline per trace (reversed-v2):
  1. Deterministic chain from agentic_plans.build_chain (correct by construction)
  2. Teacher writes N <thought> blocks + FINAL_ANSWER only (no tool_call JSON)
  3. Harness assembles thoughts + our tool calls + real results
  4. Grounding + prose validators; format errors backoff the model, not the provider

Legacy generate_trace (teacher-emits-calls) is kept and tagged distill_version=legacy-v1.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import multiprocessing
import os
import random
import re
import sys
import time
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from mock_tools import TOOL_DEFINITIONS, execute_tool_calls
from seed_bank import pick_class, sample_seed
from agentic_plans import (
    PLANS,
    build_chain,
    build_external_chain,
    trajectory_hash,
    validate_chain,
    validate_external_chain,
)
from archive_data import archive_raw, default_archive_dir, files_to_archive
from curriculum import pick_plan_index, split_plan_ids
from dpo_pairs import build_pair
from eval_card import compute_card, load_traces, stamp_eval, _load_jsonl_glob
from lineage import kept_record, lineage_id, map_reject_reason, reject_record
from verifier import deterministic_repair, should_llm_verify, build_verify_prompt, parse_verify_reply
from prose_writer import (
    build_prose_prompt, build_answer_prompt, parse_teacher_output,
    parse_thoughts, parse_final, assemble_trace,
    validate_prose_trace, validate_answer_grounding, DISTILL_VERSION,
)

ROSTER_PATH = Path("configs/roster.yaml")
OUT_DIR = Path("data/raw")
EXTERNAL_CHAINS_PATH = Path("data/seeds/external_chains.jsonl")

# Top-level roster keys that are never provider ids (flat or wrapped).
RESERVED_ROSTER_KEYS = frozenset({
    "backoff", "providers", "roles", "curriculum", "verify", "dpo",
})


def iter_provider_items(roster: dict):
    """Yield (provider_id, conf) from a flat or providers:-wrapped roster."""
    if not isinstance(roster, dict):
        return
    src = roster.get("providers") if isinstance(roster.get("providers"), dict) else roster
    for prov, conf in src.items():
        if prov in RESERVED_ROSTER_KEYS:
            continue
        if not isinstance(conf, dict):
            continue
        yield prov, conf


def shard_keeps(traj_hash: str, shard_i: int, shard_n: int) -> bool:
    """True iff this trajectory belongs to shard i/N."""
    if shard_n <= 1:
        return True
    return int(str(traj_hash)[:8], 16) % shard_n == shard_i


def _fold_usage(acc: dict, res: dict | None) -> None:
    """Add one call's usage into acc. Missing usage counts as 0, never interpolated."""
    u = (res or {}).get("usage") or {}
    acc["input"] += int(u.get("input") or 0)
    acc["output"] += int(u.get("output") or 0)


def parse_shard(s: str) -> tuple[int, int]:
    """Parse 0-indexed i/N. Raises SystemExit(2) on bad input."""
    raw = str(s)
    parts = raw.split("/")
    if len(parts) != 2:
        print(f"invalid --shard {raw}; expected i/N")
        raise SystemExit(2)
    try:
        i = int(parts[0])
        n = int(parts[1])
    except ValueError:
        print(f"invalid --shard {raw}; expected i/N")
        raise SystemExit(2)
    if n < 1 or i < 0 or i >= n:
        print(f"invalid --shard {raw}; expected i/N")
        raise SystemExit(2)
    return i, n


def parse_mp(n: int) -> int:
    n = int(n)
    if n < 1:
        print("invalid --mp; expected integer >= 1")
        raise SystemExit(2)
    return n


def split_counts(total: int, n: int) -> list[int]:
    """Partition total into n buckets. split_counts(21, 2) == [11, 10]."""
    n = int(n)
    total = int(total)
    if n < 1:
        print("invalid --mp; expected integer >= 1")
        raise SystemExit(2)
    q, r = divmod(total, n)
    return [q + (1 if i < r else 0) for i in range(n)]


def merge_token_usage(out_dir) -> dict:
    """Sum .token_usage_*.json sidecars. Zeros if none exist. Never invents counts."""
    out_dir = Path(out_dir)
    merged = {"input": 0, "output": 0, "by_route": {}}
    if not out_dir.is_dir():
        return merged
    for p in sorted(out_dir.glob(".token_usage_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        merged["input"] += int(data.get("input", 0) or 0)
        merged["output"] += int(data.get("output", 0) or 0)
        by_route = data.get("by_route") or {}
        if not isinstance(by_route, dict):
            continue
        for route, usage in by_route.items():
            slot = merged["by_route"].setdefault(str(route), {"input": 0, "output": 0})
            if not isinstance(usage, dict):
                continue
            slot["input"] += int(usage.get("input", 0) or 0)
            slot["output"] += int(usage.get("output", 0) or 0)
    return merged


def _atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

# Keys: read from environment variables only (see .env / env vars).
# Never commit credentials. Provider key names are declared in configs/roster.yaml.

CONTRACT = (
    "When you need to use tools, first output a short <thought> block "
    "(1-2 sentences explaining which tools you will call and why), then a "
    "<tool_call> block containing a JSON array of tool calls:\n"
    "<thought>\nBrief reasoning here.\n</thought>\n"
    '<tool_call>\n[{"name": "tool_name", "arguments": {"param": "value"}}]\n</tool_call>\n'
    "If a tool call fails, read the error payload and retry with corrected "
    "arguments in your next turn. If no tool is needed, answer directly without "
    "any tags."
)

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def validate_trace(trace: dict) -> str | None:
    """Return an error string if the trace must NOT enter the training set.

    Rules:
      - messages list present, starts with system, roles alternate validly
        (system -> user -> assistant -> (tool|user|assistant) -> ...)
      - every user/tool message has non-empty content
      - every assistant message has non-empty content (no silent turns)
      - any assistant turn containing <tool_call> must be followed by a tool
        message carrying results for those calls
      - system prompt must contain tool definitions (teachers that ignored
        the contract produce garbage - reject)
    """
    msgs = trace.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return "messages missing or too short"
    if msgs[0].get("role") != "system":
        return "first message is not system"
    if len(msgs[0].get("content", "")) < 50:
        return "system prompt too short (no tool defs?)"
    prev_role = None
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content") or ""
        if role not in ("system", "user", "assistant", "tool"):
            return f"invalid role {role!r} at {i}"
        if not content.strip():
            return f"empty content for {role} turn at {i}"
        if prev_role is None:
            if role != "system":
                return f"first message is not system at {i}"
        elif prev_role == "system" and role != "user":
            return f"system not followed by user at {i}"
        elif prev_role == "user" and role not in ("assistant",):
            return f"user not followed by assistant at {i}"
        elif prev_role == "assistant" and role not in ("tool", "user", "assistant"):
            return f"assistant followed by invalid role at {i}"
        elif prev_role == "tool" and role not in ("assistant", "user", "tool"):
            return f"tool not followed by assistant/user at {i}"
        if "</tool_call>" in content and role == "assistant":
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            if not nxt or nxt.get("role") != "tool":
                return f"tool_call at {i} has no tool result message"
        prev_role = role
    return None


class ModelHealth:
    """Per-route health state machine.

    Rate-limits (429) and transport errors quarantine a *provider* route.
    Format/parse/grounding errors only backoff the *specific model* and
    never quarantine the provider (other models on the same gateway stay live).
    """

    def __init__(self):
        self.status = "healthy"          # healthy | backoff | quarantined
        self.consecutive_failures = 0
        self.quarantine_until = 0.0
        self.backoff_until = 0.0
        self.last_retry_after = None
        self.generated = 0
        self.failed = 0
        self.format_fail = 0

    def note_success(self):
        self.status = "healthy"
        self.consecutive_failures = 0
        self.backoff_until = 0.0
        self.generated += 1

    def note_429(self, retry_after=None, cap=60):
        self.consecutive_failures += 1
        self.failed += 1
        if retry_after:
            self.last_retry_after = retry_after
            self.backoff_until = time.time() + retry_after
        else:
            delay = min(cap, 5 * (2 ** min(self.consecutive_failures - 1, 4)))
            self.backoff_until = time.time() + delay
        if self.consecutive_failures >= 5:
            self.status = "quarantined"
            self.quarantine_until = time.time() + 600  # 10 min
        else:
            self.status = "backoff"

    def note_failure(self):
        self.consecutive_failures += 1
        self.failed += 1
        if self.consecutive_failures >= 5:
            self.status = "quarantined"
            self.quarantine_until = time.time() + 600
        else:
            self.status = "backoff"
            self.backoff_until = time.time() + min(60, 5 * 2 ** (self.consecutive_failures - 1))

    def note_format_fail(self):
        """Parse/grounding/plan errors: backoff this model only, never quarantine."""
        self.format_fail += 1
        self.failed += 1
        self.status = "backoff"
        delay = min(8.0, 0.5 * (2 ** min(self.format_fail - 1, 4)))
        self.backoff_until = time.time() + delay

    def available(self):
        now = time.time()
        if self.status == "quarantined":
            if now >= self.quarantine_until:
                self.status = "healthy"
                self.consecutive_failures = 0
                return True
            return False
        if self.status == "backoff" and now < self.backoff_until:
            return False
        return True


class Distiller:
    def __init__(self, roster: dict, out_dir: Path, seed: int = 42,
                 provider_filter: list[str] | None = None,
                 model_filter: list[str] | None = None,
                 curriculum_mode: str = "uniform",
                 holdout_frac: float = 0.15,
                 write_eval_card: bool = True,
                 stamp_trace_eval: bool = True,
                 verify_sample: float = 0.2,
                 no_verify: bool = False,
                 cross_teacher: bool = False,
                 cross_teacher_rate: float = 0.3,
                 dpo_enabled: bool = False,
                 dpo_rate: float = 1.0,
                 shard_i: int = 0,
                 shard_n: int = 1,
                 external_chains_path: str | None = None,
                 external_frac: float = 0.0):
        self.roster = roster
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = random.Random(seed)  # fallback only; workers get their own
        self.provider_filter = provider_filter
        self.model_filter = set(model_filter) if model_filter else None
        self.curriculum_mode = curriculum_mode or "uniform"
        self.holdout_frac = float(holdout_frac)
        self.write_eval_card = write_eval_card
        self.stamp_trace_eval = stamp_trace_eval
        self.verify_sample = float(verify_sample)
        self.no_verify = bool(no_verify)
        self.verify_skipped = 0
        self.cross_teacher = bool(cross_teacher)
        self.cross_teacher_rate = float(cross_teacher_rate)
        self.cross_teacher_fallback = 0
        self._cross_disabled_logged = False
        self._pin_unused_logged = False
        self.dpo_enabled = bool(dpo_enabled)
        self.dpo_rate = float(dpo_rate)
        self.shard_i = int(shard_i)
        self.shard_n = max(1, int(shard_n))
        self.holdout_ids: set[str] = set()
        hold_path = self.out_dir / "holdout_plan_ids.json"
        if hold_path.exists():
            loaded = json.loads(hold_path.read_text(encoding="utf-8"))
            self.holdout_ids = set(loaded)
        elif self.holdout_frac > 0:
            _train, hold = split_plan_ids(PLANS, self.holdout_frac, seed)
            self.holdout_ids = set(hold)
            _atomic_write_json(hold_path, hold)
        self.external_prompts = self._load_external()
        self.external_frac = float(external_frac)
        self.external_chains: list[dict] = []
        self._external_empty_logged = False
        ext_path = (Path(external_chains_path) if external_chains_path
                    else EXTERNAL_CHAINS_PATH)
        if external_chains_path and not ext_path.exists():
            print(f"external chains file not found: {ext_path}", file=sys.stderr)
            raise SystemExit(2)
        if ext_path.exists():
            for line in ext_path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("steps"), list):
                    self.external_chains.append(rec)
        elif not external_chains_path:
            print(f"[external] no {EXTERNAL_CHAINS_PATH}; external chain pool empty",
                  file=sys.stderr)
        self.used_trajs: set[str] = set()
        for p in self.out_dir.glob("traces_*.jsonl"):
            for line in p.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                h = obj.get("traj_hash")
                if h:
                    self.used_trajs.add(h)
                elif obj.get("prompt"):
                    self.used_trajs.add(obj["prompt"])
        # OpenAI strict tool protocol is a per-provider roster flag.
        self.providers: dict[str, dict] = dict(iter_provider_items(roster))
        self.strict_tool_ids: dict[str, bool] = {
            prov: bool(conf.get("strict_tool_protocol", False))
            for prov, conf in self.providers.items()
        }
        # Token accounting: per-provider/model input/output usage
        self.token_usage: dict[str, dict] = {}  # "prov/model" -> {"input": int, "output": int}
        self.health: dict[str, ModelHealth] = {}
        self.model_health: dict[str, ModelHealth] = {}  # "prov/model"
        self.semaphores: dict[str, asyncio.Semaphore] = {}
        self.checkpoints: dict[str, int] = {}   # per-model offset -> line count
        self.headers: dict[str, dict] = {}
        self.clients: dict[str, httpx.AsyncClient] = {}
        self._client_lock: asyncio.Lock | None = None
        for prov, conf in self.providers.items():
            self.health[prov] = ModelHealth()
            self.semaphores[prov] = asyncio.Semaphore(conf.get("concurrency", 3))
            key_env = conf.get("key_env") or ""
            key = os.environ.get(key_env, "") if key_env else ""
            self.headers[prov] = {"Authorization": f"Bearer {key}"} if key else {}
            self.checkpoints[prov] = self._load_checkpoint(prov)
            for model in (conf.get("models") or {}):
                self.model_health[f"{prov}/{model}"] = ModelHealth()

    # ---- checkpointing -------------------------------------------------
    def _load_external(self) -> list[str]:
        p = Path("data/seeds/external.jsonl")
        if not p.exists():
            return []
        prompts = []
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                prompts.append(json.loads(line)["prompt"])
            except Exception:
                continue
        return prompts

    def _checkpoint_path(self, prov: str) -> Path:
        return self.out_dir / f"checkpoint_{prov}.json"

    def _load_checkpoint(self, prov: str) -> int:
        p = self._checkpoint_path(prov)
        if p.exists():
            try:
                return int(p.read_text().strip())
            except ValueError:
                return 0
        return 0

    def _locked_append(self, path, line: str) -> None:
        """POSIX flock around a jsonl append. Checkpoint updates share the same lock."""
        path = Path(path)
        name = path.name
        prov = None
        if name.startswith("traces_") and name.endswith(".jsonl"):
            prov = name[len("traces_"):-len(".jsonl")]
        elif name.startswith("dpo_pairs_") and name.endswith(".jsonl"):
            prov = name[len("dpo_pairs_"):-len(".jsonl")]
        lock_path = path.parent / (f".lock_{prov}" if prov else f".lock_{path.stem}")
        lock_path.touch(exist_ok=True)
        payload = line if line.endswith("\n") else line + "\n"
        with open(lock_path, "a+", encoding="utf-8") as lf:
            if fcntl is not None:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(payload)
                if prov is not None and name.startswith("traces_"):
                    self.checkpoints[prov] = self.checkpoints.get(prov, 0) + 1
                    self._checkpoint_path(prov).write_text(str(self.checkpoints[prov]))
            finally:
                if fcntl is not None:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _append_trace(self, prov: str, trace: dict):
        # STRICT GATE: never write malformed/empty data into the training pool.
        if trace.get("distill_version") == DISTILL_VERSION:
            err = validate_prose_trace(trace)
        else:
            err = validate_trace(trace)
        if err:
            self.health[prov].failed += 1
            return False
        path = self.out_dir / f"traces_{prov}.jsonl"
        self._locked_append(path, json.dumps(trace, ensure_ascii=False))
        return True

    def _append_dpo(self, prov: str, pair: dict):
        path = self.out_dir / f"dpo_pairs_{prov}.jsonl"
        self._locked_append(path, json.dumps(pair, ensure_ascii=False))

    def _append_lineage(self, prov: str, row: dict) -> None:
        path = self.out_dir / f"lineage_{prov}.jsonl"
        self._locked_append(path, json.dumps(row, ensure_ascii=False))

    def _append_reject(self, prov: str, row: dict) -> None:
        path = self.out_dir / f"rejects_{prov}.jsonl"
        self._locked_append(path, json.dumps(row, ensure_ascii=False))

    # ---- trace generation ----------------------------------------------
    def _system_prompt(self, cls: str) -> str:
        schemas = json.dumps(TOOL_DEFINITIONS)
        if cls == "irrelevant":
            # Add distractor tools that should NOT be called
            distractors = json.dumps([
                {"type": "function", "function": {"name": "delete_all_data", "description": "DANGEROUS. Deletes everything.", "parameters": {"type": "object", "properties": {"confirm": {"type": "boolean"}}, "required": ["confirm"]}}},
                {"type": "function", "function": {"name": "charge_credit_card", "description": "Charge a credit card.", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]}}},
            ])
            return (
                "You are a helpful assistant. The following tools are registered:\n"
                f"{schemas}\n{distractors}\n"
                "Answer the user directly. ONLY call a tool if it is genuinely "
                "required by the request - in most cases you should simply respond "
                "in text. Never call a tool 'for fun' or to demonstrate ability."
            )
        if cls == "agentic":
            return (
                "You are a tool-calling agent with access to these tools:\n"
                f"{schemas}\n"
                f"{CONTRACT}\n"
                "IMPORTANT - SEQUENTIAL WORKFLOW: These tasks require multiple "
                "dependent steps. You often cannot complete them in a single "
                "tool call. Examples:\n"
                "- To email a user, FIRST call get_user to learn their email "
                "address, THEN call send_email with that exact address.\n"
                "- To count users on a plan, FIRST get_user to learn the plan "
                "string, THEN db_query filtering by that exact plan.\n"
                "- To email weather info, FIRST weather.get, THEN send_email "
                "describing the result.\n"
                "When a tool result gives you a value you need for the next "
                "call (an email address, a plan, a temperature), use that "
                "exact value. Never invent or guess addresses or plans - the "
                "executor will reject fabricated values and you will need to "
                "retry with the real ones."
            )
        return (
            "You are a tool-calling agent with access to these tools:\n"
            f"{schemas}\n"
            f"{CONTRACT}"
        )

    async def _client(self, prov: str) -> httpx.AsyncClient:
        """One AsyncClient per provider (connection reuse at high concurrency)."""
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        async with self._client_lock:
            c = self.clients.get(prov)
            if c is None:
                c = httpx.AsyncClient(
                    timeout=httpx.Timeout(180.0),
                    limits=httpx.Limits(max_connections=600, max_keepalive_connections=120),
                )
                self.clients[prov] = c
            return c

    async def aclose(self):
        for c in list(self.clients.values()):
            try:
                await c.aclose()
            except Exception:
                pass
        self.clients.clear()

    async def _chat(self, prov: str, model: str, msgs: list[dict], max_tokens: int,
                    extra_body: dict | None = None) -> dict:
        """POST one chat completion. Returns {ok, data|error, retry_after}."""
        conf = self.providers[prov]
        url = conf["base_url"].rstrip("/") + "/chat/completions"
        body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "stream": False}
        if extra_body:
            body.update(extra_body)
        try:
            client = await self._client(prov)
            r = await client.post(url, headers=self.headers[prov], json=body)
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}
        if r.status_code == 200:
            try:
                d = r.json()
                msg = (d.get("choices") or [{}])[0].get("message") or {}
                usage = d.get("usage") or {}
                parsed_usage = {
                    "input": int(usage.get("prompt_tokens") or 0),
                    "output": int(usage.get("completion_tokens") or 0),
                }
                key = f"{prov}/{model}"
                u = self.token_usage.setdefault(key, {"input": 0, "output": 0})
                u["input"] += parsed_usage["input"]
                u["output"] += parsed_usage["output"]
                return {"ok": True, "content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
                        "usage": parsed_usage}
            except Exception as e:
                return {"ok": False, "error": f"parse: {e}"}
        retry_after = r.headers.get("retry-after")
        try:
            retry_after = float(retry_after) if retry_after else None
        except ValueError:
            retry_after = None
        return {"ok": False, "http": r.status_code, "error": r.text[:200], "retry_after": retry_after}

    async def _chat_user(self, prov: str, model: str, model_conf: dict, user_content: str) -> dict:
        """One user-turn completion. Teachers receive prose prompts only."""
        conf = model_conf or {}
        max_tokens = int(conf.get("max_tokens", 2000))
        extra = conf.get("chat_template_kwargs")
        extra_body = {"chat_template_kwargs": extra} if extra else None
        msgs = [{"role": "user", "content": user_content}]
        return await self._chat(prov, model, msgs, max_tokens, extra_body)

    def _select_answer_route(self, thought_prov, thought_model, rng):
        """Pin wins. Sampling never overrides a valid pin. None if no other route."""
        work = self._providers_with_models()
        roles = self.roster.get("roles") if isinstance(self.roster.get("roles"), dict) else {}
        pin = (roles or {}).get("answer") or {}
        pprov, pmodel = pin.get("provider"), pin.get("model")
        if pprov and pmodel:
            for p, m, c in work:
                if (p, m) == (pprov, pmodel) and (p, m) != (thought_prov, thought_model):
                    return (p, m, c)
            if not self._pin_unused_logged:
                print("[cross-teacher] roles.answer pin unused; sampling")
                self._pin_unused_logged = True
        others = [(p, m, c) for p, m, c in work if (p, m) != (thought_prov, thought_model)]
        if not others:
            return None
        return rng.choice(others)

    def _parse_turn(self, content: str) -> tuple[str | None, list[dict] | None]:
        """Extract <thought> and <tool_call> from assistant content."""
        thought_m = THOUGHT_RE.search(content)
        thought = thought_m.group(1).strip() if thought_m else None
        tool_m = TOOL_CALL_RE.search(content)
        calls = None
        if tool_m:
            try:
                calls = json.loads(tool_m.group(1).strip())
                if isinstance(calls, dict):
                    calls = [calls]
                if not isinstance(calls, list):
                    calls = None
                else:
                    for c in calls:
                        if not isinstance(c, dict) or "name" not in c:
                            raise ValueError("missing name")
                        if not isinstance(c.get("arguments"), dict):
                            c["arguments"] = json.loads(c.get("arguments", "{}")) if isinstance(c.get("arguments"), str) else {}
            except Exception:
                calls = None
        return thought, calls

    def _assistant_msg(self, prov: str, content: str, calls: list[dict] | None) -> dict:
        """Build an assistant message. For strict providers, attach the
        OpenAI tool_calls array (ids used by subsequent tool messages)."""
        msg = {"role": "assistant", "content": content}
        if calls and self.strict_tool_ids.get(prov, False):
            msg["tool_calls"] = [
                {"id": f"call_{i:04d}", "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}}
                for i, c in enumerate(calls)
            ]
        return msg

    def _tool_msg(self, prov: str, content: str, calls: list[dict] | None) -> dict:
        """Build a tool result message. Strict providers need tool_call_id
        matching the assistant's tool_calls array."""
        msg = {"role": "tool", "content": content}
        if calls and self.strict_tool_ids.get(prov, False):
            # One tool message per call id (strict protocol is 1:1)
            # -> merge results into a single message keyed by first call id
            msg["tool_call_id"] = f"call_0000"
        return msg

    async def generate_trace(self, prov: str, model: str, model_conf: dict, cls: str, prompt: str) -> dict:
        """One full multi-turn trajectory from seed to final answer."""
        sys_prompt = self._system_prompt(cls)
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        max_tokens = model_conf.get("max_tokens", 2000)
        extra = model_conf.get("chat_template_kwargs")
        extra_body = {"chat_template_kwargs": extra} if extra else None

        # Turn 1: teacher proposes tools (or answers directly)
        r1 = await self._chat(prov, model, msgs, max_tokens, extra_body)
        if not r1["ok"]:
            return {"ok": False, "error": r1.get("error"), "http": r1.get("http"), "retry_after": r1.get("retry_after")}
        thought1, calls1 = self._parse_turn(r1["content"])
        # If the teacher emitted <tool_call> but the JSON was malformed, the
        # trace is garbage - fail it rather than write a broken tool turn.
        if "<tool_call>" in r1["content"] and calls1 is None:
            return {"ok": False, "error": "malformed <tool_call> JSON in turn 1", "http": "FORMAT"}
        msgs.append({"role": "assistant", "content": r1["content"]})

        trace = {
            "seed_class": cls,
            "prompt": prompt,
            "teacher": f"{prov}/{model}",
            "teacher_mode": model_conf.get("mode", "unknown"),
            "distill_version": "legacy-v1",
            "messages": [],
        }
        msgs_out = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]

        # Case A: no tool call needed (irrelevant/ambiguous class or direct answer)
        if not calls1:
            msgs_out.append({"role": "assistant", "content": r1["content"]})
            trace["messages"] = msgs_out
            return {"ok": True, "trace": trace, "tool_calls_made": 0}

        # Case B: tool calls executed, results fed back
        asst_msg = self._assistant_msg(prov, r1["content"], calls1)
        msgs.append(asst_msg)
        msgs_out.append(dict(asst_msg))
        results = execute_tool_calls(calls1)

        def tool_messages(results: dict, calls: list[dict]) -> list[dict]:
            """Emit tool messages. Strict providers: one per call id (1:1
            protocol). Loose providers: single merged message."""
            if self.strict_tool_ids.get(prov, False):
                out = []
                for i, c in enumerate(calls):
                    payload = results.get(c["name"], {"status": 400, "error": {"code": "UNKNOWN_TOOL", "message": "no result"}})
                    out.append({"role": "tool", "content": json.dumps(payload), "tool_call_id": f"call_{i:04d}"})
                return out
            return [{"role": "tool", "content": json.dumps(results)}]

        tmsgs = tool_messages(results, calls1)
        msgs.extend(tmsgs)
        msgs_out.extend(dict(m) for m in tmsgs)

        # Error recovery: if any call failed, give the teacher a correction turn
        any_fail = any(r.get("status", 200) >= 400 for r in results.values())
        if any_fail:
            msgs.append({"role": "user", "content": "Some tool calls failed. Inspect the error payloads above and retry with corrected arguments."})
            msgs_out.append({"role": "user", "content": "Some tool calls failed. Inspect the error payloads above and retry with corrected arguments."})
            r2 = await self._chat(prov, model, msgs, max_tokens, extra_body)
            if not r2["ok"]:
                return {"ok": False, "error": r2.get("error"), "http": r2.get("http"), "retry_after": r2.get("retry_after")}
            thought2, calls2 = self._parse_turn(r2["content"])
            if "<tool_call>" in r2["content"] and calls2 is None:
                return {"ok": False, "error": "malformed <tool_call> JSON in correction turn", "http": "FORMAT"}
            asst_msg2 = self._assistant_msg(prov, r2["content"], calls2)
            msgs.append(asst_msg2)
            msgs_out.append(dict(asst_msg2))
            if calls2:
                results2 = execute_tool_calls(calls2)
                tmsgs2 = tool_messages(results2, calls2)
                msgs.extend(tmsgs2)
                msgs_out.extend(dict(m) for m in tmsgs2)
                any_fail2 = any(r.get("status", 200) >= 400 for r in results2.values())
                if any_fail2:
                    msgs.append({"role": "user", "content": "Some tool calls still failed. Provide the final answer based on what you have, or explain the failure."})
                    msgs_out.append({"role": "user", "content": "Some tool calls still failed. Provide the final answer based on what you have, or explain the failure."})
                    r3 = await self._chat(prov, model, msgs, max_tokens, extra_body)
                    if not r3["ok"]:
                        return {"ok": False, "error": r3.get("error"), "http": r3.get("http"), "retry_after": r3.get("retry_after")}
                    msgs_out.append({"role": "assistant", "content": r3["content"]})
                else:
                    msgs.append({"role": "user", "content": "Now provide the final answer to the user's original request."})
                    msgs_out.append({"role": "user", "content": "Now provide the final answer to the user's original request."})
                    r3 = await self._chat(prov, model, msgs, max_tokens, extra_body)
                    if not r3["ok"]:
                        return {"ok": False, "error": r3.get("error"), "http": r3.get("http"), "retry_after": r3.get("retry_after")}
                    msgs_out.append({"role": "assistant", "content": r3["content"]})
            else:
                msgs.append({"role": "user", "content": "Now provide the final answer to the user's original request."})
                msgs_out.append({"role": "user", "content": "Now provide the final answer to the user's original request."})
                r3 = await self._chat(prov, model, msgs, max_tokens, extra_body)
                if not r3["ok"]:
                    return {"ok": False, "error": r3.get("error"), "http": r3.get("http"), "retry_after": r3.get("retry_after")}
                msgs_out.append({"role": "assistant", "content": r3["content"]})
        else:
            msgs.append({"role": "user", "content": "Now provide the final answer to the user's original request."})
            msgs_out.append({"role": "user", "content": "Now provide the final answer to the user's original request."})
            r2 = await self._chat(prov, model, msgs, max_tokens, extra_body)
            if not r2["ok"]:
                return {"ok": False, "error": r2.get("error"), "http": r2.get("http"), "retry_after": r2.get("retry_after")}
            msgs_out.append({"role": "assistant", "content": r2["content"]})

        trace["messages"] = msgs_out
        return {"ok": True, "trace": trace, "tool_calls_made": len(calls1)}

    # ---- reversed distillation (Phase 1 deterministic + Phase 2 prose) ---
    async def generate_agentic_trace(self, prov: str, model: str, model_conf: dict,
                                     traj: dict | None = None,
                                     rng: random.Random | None = None) -> dict:
        """Reversed distillation: structure is built by us (guaranteed
        correct), teacher writes only thoughts + final answer.

        Pass traj in from the worker so the uniqueness check and the
        exported trace share ONE chain (no double build_chain).
        """
        if traj is None:
            traj = build_chain(rng or self.rng, exclude_ids=self.holdout_ids)
        if traj.get("seed_class") not in (None, "agentic"):
            return {"ok": False, "error": "non-agentic seed_class is frozen; reversed-v2 is agentic-only",
                    "http": "PLAN", "usage": {"input": 0, "output": 0}}
        plan_err = (
            validate_external_chain(traj["steps"])
            if traj.get("source") == "external"
            else validate_chain(traj["steps"])
        )
        if plan_err:
            return {"ok": False, "error": f"plan: {plan_err}", "http": "PLAN",
                    "usage": {"input": 0, "output": 0}}

        teacher_prompt = build_prose_prompt(traj)
        sys_prompt = self._system_prompt("agentic")
        rng = rng or self.rng
        n_steps = len(traj["steps"])
        usage_acc = {"input": 0, "output": 0}

        r1 = await self._chat_user(prov, model, model_conf, teacher_prompt)
        _fold_usage(usage_acc, r1)
        if not r1["ok"]:
            return {"ok": False, "error": r1.get("error"), "http": r1.get("http"),
                    "retry_after": r1.get("retry_after"), "usage": dict(usage_acc)}

        blob = r1.get("content") or ""
        if r1.get("reasoning"):
            blob_plus = blob + "\n" + r1["reasoning"]
        else:
            blob_plus = blob

        thoughts = parse_thoughts(blob, n_steps) or parse_thoughts(blob_plus, n_steps)
        if not thoughts:
            return {"ok": False, "error": "could not parse teacher prose", "http": "FORMAT",
                    "usage": dict(usage_acc)}

        split = False
        fallback = False
        ans_prov = ans_model = None
        answer_route = None
        if self.cross_teacher:
            answer_route = self._select_answer_route(prov, model, rng)
            if answer_route is None:
                if not self._cross_disabled_logged:
                    print("[cross-teacher] only one route in worklist; split disabled")
                    self._cross_disabled_logged = True
            elif rng.random() < self.cross_teacher_rate:
                split = True
                answer_route = answer_route

        final = None
        if split and answer_route:
            ans_prov, ans_model, ans_conf = answer_route
            ans_prompt = build_answer_prompt(traj, thoughts)
            r2 = await self._chat_user(ans_prov, ans_model, ans_conf, ans_prompt)
            _fold_usage(usage_acc, r2)
            if r2.get("ok"):
                final = parse_final(r2.get("content") or "")
            if not final:
                self.cross_teacher_fallback += 1
                fallback = True
                r_fb = await self._chat_user(prov, model, model_conf, ans_prompt)
                _fold_usage(usage_acc, r_fb)
                if r_fb.get("ok"):
                    final = parse_final(r_fb.get("content") or "")
                if not final:
                    final = parse_final(blob_plus)
        else:
            parsed_one = parse_teacher_output(blob, n_steps) or parse_teacher_output(blob_plus, n_steps)
            if parsed_one:
                thoughts = parsed_one["thoughts"]
                final = parsed_one["final"]

        if not thoughts or not final:
            fail = {"ok": False, "error": "could not parse teacher prose", "http": "FORMAT",
                    "usage": dict(usage_acc)}
            if split and answer_route:
                fail["teacher"] = f"{ans_prov}/{ans_model}"
            return fail
        parsed = {"thoughts": thoughts, "final": final}

        gerr = validate_answer_grounding(traj, parsed["final"])
        repaired = False
        if gerr:
            fixed = deterministic_repair(traj, parsed["final"], gerr)
            if fixed is not None:
                parsed["final"] = fixed
                repaired = True
                gerr = None
        if gerr:
            return {"ok": False, "error": f"grounding: {gerr}", "http": "GROUNDING",
                    "usage": dict(usage_acc)}

        trace, aerr = assemble_trace(
            traj,
            {"teacher": f"{prov}/{model}", "mode": model_conf.get("mode", "unknown")},
            parsed, sys_prompt,
        )
        if aerr or trace is None:
            return {"ok": False, "error": f"assemble: {aerr}", "http": "FORMAT",
                    "usage": dict(usage_acc)}
        trace["traj_hash"] = trajectory_hash(traj)
        trace["forge_spec"] = "0.2"
        if traj.get("tier") and not trace.get("plan_tier"):
            trace["plan_tier"] = traj["tier"]
        verr = validate_prose_trace(trace)
        if verr:
            return {"ok": False, "error": f"prose validate: {verr}", "http": "FORMAT",
                    "usage": dict(usage_acc)}
        vfail = await self._maybe_llm_verify(
            traj, parsed["final"], repaired, rng or self.rng, usage_acc=usage_acc,
        )
        if vfail:
            return {"ok": False, "error": vfail, "http": "VERIFY", "usage": dict(usage_acc)}
        if self.stamp_trace_eval:
            if split and answer_route and not fallback:
                trace["teacher_thoughts"] = f"{prov}/{model}"
                trace["teacher_answer"] = f"{ans_prov}/{ans_model}"
                stamp_eval(trace, traj, repaired=repaired, cross_teacher=True)
            elif fallback:
                stamp_eval(trace, traj, repaired=repaired, cross_teacher=False,
                           cross_teacher_fallback=True)
            else:
                stamp_eval(trace, traj, repaired=repaired, cross_teacher=False)
        return {"ok": True, "trace": trace, "tool_calls_made": n_steps, "repaired": repaired,
                "usage": dict(usage_acc)}

    async def _maybe_llm_verify(self, traj: dict, final: str, repaired: bool,
                                rng: random.Random, usage_acc: dict | None = None) -> str | None:
        """Optional PASS/FAIL gate. HTTP/parse failure: skip and keep the trace."""
        if self.no_verify:
            return None
        if not should_llm_verify(repaired, rng, self.verify_sample):
            return None
        roles = self.roster.get("roles") if isinstance(self.roster.get("roles"), dict) else {}
        vrole = (roles or {}).get("verifier") or {}
        vprov = vrole.get("provider")
        vmodel = vrole.get("model")
        if not vprov or not vmodel or vprov not in self.providers:
            self.verify_skipped += 1
            return None
        mconf = (self.providers[vprov].get("models") or {}).get(vmodel) or {}
        max_tokens = int(mconf.get("max_tokens", 256))
        extra = mconf.get("chat_template_kwargs")
        extra_body = {"chat_template_kwargs": extra} if extra else None
        msgs = [
            {"role": "system", "content": "You are a strict fact checker. Reply with one line."},
            {"role": "user", "content": build_verify_prompt(traj, final)},
        ]
        try:
            r = await self._chat(vprov, vmodel, msgs, max_tokens, extra_body)
        except Exception:
            self.verify_skipped += 1
            return None
        if usage_acc is not None:
            _fold_usage(usage_acc, r)
        if not r.get("ok"):
            self.verify_skipped += 1
            return None
        verdict = parse_verify_reply(r.get("content") or "")
        if verdict is None:
            self.verify_skipped += 1
            return None
        if verdict == "fail":
            return "verifier FAIL"
        return None

    # ---- worker loop ---------------------------------------------------
    def _providers_with_models(self):
        """Flatten roster to (prov, model, model_conf) skipping reserved keys."""
        out = []
        filt = set(self.provider_filter) if self.provider_filter else None
        for prov, conf in self.providers.items():
            if filt is not None and prov not in filt:
                continue
            for model, mconf in (conf.get("models") or {}).items():
                if self.model_filter is not None and model not in self.model_filter:
                    continue
                out.append((prov, model, mconf))
        return out

    async def run(self, count: int, pilot: bool = False):
        worklist = self._providers_with_models()
        if not worklist:
            print("No models in roster!", file=sys.stderr)
            raise SystemExit(2)
        print(f"Roster: {len(worklist)} model routes across {len({p for p,_,_ in worklist})} providers")
        print("Routes: " + ", ".join(f"{p}/{m}" for p, m, _ in worklist))
        print(f"Distill version: {DISTILL_VERSION}")

        tasks = []
        lock = asyncio.Lock()
        produced = 0
        started = time.time()
        self._client_lock = asyncio.Lock()

        async def worker_loop(prov: str, model: str, mconf: dict, worker_id: int):
            nonlocal produced
            rng = random.Random((self.seed + worker_id * 1_000_003) & 0xFFFFFFFF)
            route = f"{prov}/{model}"
            attempt_seq = 0
            while True:
                async with lock:
                    if produced >= count:
                        return
                h = self.health[prov]
                mh = self.model_health[route]
                if not h.available() or not mh.available():
                    await asyncio.sleep(0.5)
                    continue
                traj = None
                thash = None
                progress = (produced + 1) / max(count, 1)
                try:
                    for _try in range(40):
                        cand = None
                        if self.external_chains and rng.random() < self.external_frac:
                            async with lock:
                                rec = self.external_chains.pop() if self.external_chains else None
                                if not self.external_chains and not self._external_empty_logged:
                                    self._external_empty_logged = True
                                    print("[external] chain pool exhausted; "
                                          "continuing on plan chains only")
                            if rec is not None:
                                try:
                                    cand = build_external_chain(rec)
                                except ValueError:
                                    cand = None
                        if cand is None:
                            if self.curriculum_mode == "off":
                                cand = build_chain(rng, exclude_ids=self.holdout_ids)
                            else:
                                idx = pick_plan_index(
                                    rng, PLANS, self.curriculum_mode, progress,
                                    self.holdout_ids,
                                )
                                cand = build_chain(rng, plan_index=idx,
                                                   exclude_ids=self.holdout_ids)
                        fh = trajectory_hash(cand)
                        if not shard_keeps(fh, self.shard_i, self.shard_n):
                            continue
                        async with lock:
                            if fh not in self.used_trajs:
                                self.used_trajs.add(fh)
                                traj = cand
                                thash = fh
                                break
                except RuntimeError:
                    raise
                if traj is None:
                    raise RuntimeError("plan space exhausted")
                async with lock:
                    if produced >= count:
                        if thash:
                            self.used_trajs.discard(thash)
                        return
                    produced += 1
                    my_slot = produced
                async with self.semaphores[prov]:
                    attempt_seq += 1
                    res = await self.generate_agentic_trace(prov, model, mconf, traj=traj, rng=rng)
                usage = res.get("usage") or {"input": 0, "output": 0}
                if res.get("ok"):
                    pair = None
                    if self.dpo_enabled and rng.random() < self.dpo_rate:
                        pair = build_pair(res["trace"], traj, rng)
                        if pair:
                            res["trace"]["dpo_pair_id"] = pair["pair_id"]
                    teacher = str(res["trace"].get("teacher") or route)
                    lid = lineage_id(
                        str(res["trace"].get("traj_hash") or thash or ""),
                        teacher,
                        str(res["trace"].get("distill_version") or DISTILL_VERSION),
                        str(res["trace"].get("forge_spec") or "0.2"),
                    )
                    res["trace"]["lineage_id"] = lid
                    wrote = self._append_trace(prov, res["trace"])
                    if not wrote:
                        mh.note_format_fail()
                        async with lock:
                            produced -= 1
                            if thash:
                                self.used_trajs.discard(thash)
                        gate_err = validate_prose_trace(res["trace"]) or "append rejected"
                        print(f"  [REJ] {route} rejected by validator: {gate_err}")
                        self._append_reject(prov, reject_record(
                            prov=prov,
                            model=model,
                            plan_id=traj.get("plan_id") if traj else None,
                            traj_hash=thash,
                            reason=map_reject_reason(res, gate_err=gate_err),
                            http=res.get("http"),
                            error=str(gate_err),
                            attempt_seq=attempt_seq,
                        ))
                    else:
                        self._append_lineage(prov, kept_record(
                            res["trace"],
                            tokens_in=int(usage.get("input") or 0),
                            tokens_out=int(usage.get("output") or 0),
                            curriculum_mode=self.curriculum_mode,
                            seed=self.seed,
                        ))
                        if pair is not None:
                            self._append_dpo(prov, pair)
                        h.note_success()
                        mh.note_success()
                        if my_slot % 5 == 0 or pilot:
                            rate = produced / max(1, time.time() - started)
                            print(f"  [{my_slot}/{count}] {route} OK ({res['tool_calls_made']} tool calls, {rate:.2f}/s)")
                else:
                    http = res.get("http")
                    if http == 429:
                        h.note_429(res.get("retry_after"))
                    elif http in ("FORMAT", "PLAN", "GROUNDING", "VERIFY"):
                        mh.note_format_fail()
                    else:
                        h.note_failure()
                    async with lock:
                        produced -= 1
                        if thash:
                            self.used_trajs.discard(thash)
                    rej_prov, rej_model = prov, model
                    attempted = res.get("teacher")
                    if isinstance(attempted, str) and "/" in attempted:
                        rej_prov, rej_model = attempted.split("/", 1)
                    self._append_reject(rej_prov, reject_record(
                        prov=rej_prov,
                        model=rej_model,
                        plan_id=traj.get("plan_id") if traj else None,
                        traj_hash=thash,
                        reason=map_reject_reason(res),
                        http=http,
                        error=str(res.get("error") or ""),
                        attempt_seq=attempt_seq,
                    ))
                    if h.status == "quarantined":
                        print(f"  [Q] {route} provider quarantined after {h.consecutive_failures} failures: {res.get('error','')[:100]}")
                    else:
                        print(f"  [!] {route}: {res.get('error','')[:160]}")
                    await asyncio.sleep(0.4)

        wid = 0
        for prov, model, mconf in worklist:
            w = int(mconf.get("weight", 1))
            if pilot:
                w = min(w, max(2, min(8, count)))
            for _ in range(w):
                tasks.append(asyncio.create_task(worker_loop(prov, model, mconf, wid)))
                wid += 1

        try:
            await asyncio.gather(*tasks)
        except RuntimeError as e:
            for t in tasks:
                t.cancel()
            print(f"[FAIL] {e}")
            await self.aclose()
            raise SystemExit(3)
        finally:
            await self.aclose()
        print(f"\nDone. {produced} traces in {time.time()-started:.0f}s")
        self.print_summary()
        sidecar = self.out_dir / f".token_usage_{self.shard_i}.json"
        sidecar.write_text(
            json.dumps(self._token_usage_for_card(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if self.write_eval_card:
            self._write_eval_card()

    def _token_usage_for_card(self) -> dict:
        by_route = {k: dict(v) for k, v in self.token_usage.items()}
        return {
            "input": sum(int(v.get("input", 0) or 0) for v in by_route.values()),
            "output": sum(int(v.get("output", 0) or 0) for v in by_route.values()),
            "by_route": by_route,
        }

    def _write_eval_card(self):
        files = sorted(self.out_dir.glob("traces_*.jsonl"))
        traces = load_traces(files)
        hold = sorted(self.holdout_ids)
        card = compute_card(
            traces,
            token_usage=self._token_usage_for_card(),
            extra={
                "input_paths": [str(p) for p in files],
                "holdout_plan_ids": hold,
                "lineage_rows": _load_jsonl_glob(self.out_dir, "lineage_*.jsonl"),
                "reject_rows": _load_jsonl_glob(self.out_dir, "rejects_*.jsonl"),
            },
        )
        out = self.out_dir / "eval_card.json"
        out.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"Wrote eval card {out} n_traces={card.get('n_traces', 0)}")

    def print_summary(self):
        """Per-model + total token usage and trace counts."""
        print("\n" + "=" * 64)
        print("DISTILLATION SUMMARY")
        print("=" * 64)
        tot_in = tot_out = tot_traces = 0
        # Gather trace counts from checkpoint files
        for prov in self.checkpoints:
            n = self.checkpoints[prov]
            if n:
                tot_traces += n
        rows = []
        for key, u in self.token_usage.items():
            rows.append((key, u["input"], u["output"]))
        if not rows:
            print("(no token usage recorded - providers may not return usage)")
        for key, tin, tout in sorted(rows, key=lambda x: -(x[1] + x[2])):
            tot_in += tin
            tot_out += tout
            print(f"  {key:45s} in {tin:>10,d}  out {tout:>10,d}")
        print("-" * 64)
        print(f"  {'TOTAL':45s} in {tot_in:>10,d}  out {tot_out:>10,d}")
        print(f"  {'TOTAL TOKENS':45s} {tot_in + tot_out:>10,d}")
        print(f"  Traces written: {tot_traces}")
        print(f"  Avg tokens/trace: {int((tot_in + tot_out) / max(1, tot_traces)):,}")
        fmt = sum(mh.format_fail for mh in self.model_health.values())
        gen = sum(mh.generated for mh in self.model_health.values())
        print(f"  Model successes: {gen}   format/grounding fails: {fmt}")
        print(f"  Verifier skipped: {self.verify_skipped}")
        for key, mh in sorted(self.model_health.items()):
            if mh.generated or mh.format_fail or mh.failed:
                print(f"    {key:45s} ok {mh.generated:4d}  fmt {mh.format_fail:4d}  fail {mh.failed:4d}  {mh.status}")


def _mp_child(payload: dict) -> None:
    roster = yaml.safe_load(Path(payload["roster_path"]).read_text())
    d = Distiller(
        roster,
        Path(payload["out_dir"]),
        seed=payload["seed"],
        provider_filter=payload["provider_filter"],
        model_filter=payload["model_filter"],
        curriculum_mode=payload["curriculum_mode"],
        holdout_frac=payload["holdout_frac"],
        write_eval_card=False,
        stamp_trace_eval=payload["stamp_trace_eval"],
        verify_sample=payload["verify_sample"],
        no_verify=payload["no_verify"],
        cross_teacher=payload["cross_teacher"],
        cross_teacher_rate=payload["cross_teacher_rate"],
        dpo_enabled=payload["dpo_enabled"],
        dpo_rate=payload["dpo_rate"],
        shard_i=payload["shard_i"],
        shard_n=payload["shard_n"],
        external_chains_path=payload["external_chains_path"],
        external_frac=payload["external_frac"],
    )
    asyncio.run(d.run(payload["count"], pilot=payload["pilot"]))


def run_mp(
    roster_path: str,
    out_dir: Path,
    count: int,
    nproc: int,
    seed: int,
    holdout_frac: float,
    provider_filter,
    model_filter,
    curriculum_mode: str,
    write_eval_card: bool,
    stamp_trace_eval: bool,
    verify_sample: float,
    no_verify: bool,
    cross_teacher: bool,
    cross_teacher_rate: float,
    dpo_enabled: bool,
    dpo_rate: float,
    pilot: bool,
    external_chains_path: str | None = None,
    external_frac: float = 0.0,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if files_to_archive(out_dir):
        print("archive or wipe before --mp")
        raise SystemExit(1)
    hold_path = out_dir / "holdout_plan_ids.json"
    if holdout_frac > 0 and not hold_path.exists():
        _train, hold = split_plan_ids(PLANS, holdout_frac, seed)
        _atomic_write_json(hold_path, hold)
    counts = split_counts(count, nproc)
    ctx = multiprocessing.get_context("spawn")
    procs = []
    for i, child_count in enumerate(counts):
        if child_count <= 0:
            continue
        payload = {
            "roster_path": str(roster_path),
            "out_dir": str(out_dir),
            "count": int(child_count),
            "seed": int(seed) + i * 7919,
            "shard_i": i,
            "shard_n": int(nproc),
            "holdout_frac": float(holdout_frac),
            "provider_filter": provider_filter,
            "model_filter": model_filter,
            "curriculum_mode": curriculum_mode,
            "stamp_trace_eval": bool(stamp_trace_eval),
            "verify_sample": float(verify_sample),
            "no_verify": bool(no_verify),
            "cross_teacher": bool(cross_teacher),
            "cross_teacher_rate": float(cross_teacher_rate),
            "dpo_enabled": bool(dpo_enabled),
            "dpo_rate": float(dpo_rate),
            "pilot": bool(pilot),
            "external_chains_path": external_chains_path,
            "external_frac": float(external_frac),
        }
        p = ctx.Process(target=_mp_child, args=(payload,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    if any(c not in (0,) for c in codes):
        raise SystemExit(max((c if isinstance(c, int) and c > 0 else 1) for c in codes))
    if write_eval_card:
        files = sorted(out_dir.glob("traces_*.jsonl"))
        traces = load_traces(files)
        hold = []
        if hold_path.exists():
            hold = json.loads(hold_path.read_text(encoding="utf-8"))
        card = compute_card(
            traces,
            token_usage=merge_token_usage(out_dir),
            extra={
                "input_paths": [str(p) for p in files],
                "holdout_plan_ids": hold,
                "lineage_rows": _load_jsonl_glob(out_dir, "lineage_*.jsonl"),
                "reject_rows": _load_jsonl_glob(out_dir, "rejects_*.jsonl"),
            },
        )
        dest = out_dir / "eval_card.json"
        dest.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"Wrote eval card {dest} n_traces={card.get('n_traces', 0)}")


def _resolve_out_dir(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _pilot_defaults(roster: dict) -> tuple[list[str], list[str]]:
    """First provider in roster order and that provider's first model."""
    items = list(iter_provider_items(roster))
    if not items:
        return [], []
    prov, conf = items[0]
    models = list((conf.get("models") or {}).keys())
    if not models:
        return [prov], []
    return [prov], [models[0]]


def main():
    ap = argparse.ArgumentParser(
        description="Reversed-v2 agentic distillation. Teacher writes prose only."
    )
    ap.add_argument("--count", type=int, default=100, help="total traces to generate")
    ap.add_argument("--pilot", action="store_true",
                    help="print every trace (small run); empty filters use first roster provider/model")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed; --mp child i uses seed + i * 7919")
    ap.add_argument("--roster", default=str(ROSTER_PATH),
                    help="YAML roster path (default: configs/roster.yaml). Keys via key_env only")
    ap.add_argument("--providers", default="",
                    help="comma-separated provider ids to use (default: all, or first roster provider with --pilot)")
    ap.add_argument("--models", default="",
                    help="comma-separated model ids (default: all, or first model of the pilot provider with --pilot)")
    ap.add_argument("--wipe", action="store_true",
                    help="archive existing traces via archive_data.py, then proceed. Never unlinks.")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="trace output directory (default: data/raw)")
    ap.add_argument("--legacy-v1", action="store_true",
                    help="rejected: legacy-v1 generator is frozen")
    ap.add_argument("--no-eval-card", action="store_true",
                    help="skip writing aggregate eval_card.json at end of run")
    ap.add_argument("--no-trace-eval", action="store_true",
                    help="omit per-trace eval block (size escape hatch)")
    ap.add_argument("--curriculum", default=None, choices=("off", "uniform", "linear"),
                    help="plan-tier mix (default: roster curriculum.mode or uniform); "
                         "linear interpolates easy 0.50->0.10, medium 0.30->0.20, "
                         "hard 0.15->0.40, expert 0.05->0.30")
    ap.add_argument("--holdout-frac", type=float, default=None,
                    help="fraction of plan ids held out of training (default: 0.15); 0 disables holdout")
    ap.add_argument("--verify-sample", type=float, default=None,
                    help="LLM-verify sample rate for already-passing traces (default: 0.2)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip LLM verifier calls; deterministic repair still runs")
    ap.add_argument("--cross-teacher", action="store_true",
                    help="split thoughts/final across two teacher routes")
    ap.add_argument("--cross-teacher-rate", type=float, default=None,
                    help="probability of a thoughts/answer split when --cross-teacher (default 0.3)")
    ap.add_argument("--dpo", action="store_true",
                    help="write dpo_pairs_{prov}.jsonl next to traces; rejected never in traces_*.jsonl")
    ap.add_argument("--dpo-rate", type=float, default=None,
                    help="probability of building a DPO pair per kept trace (default 1.0)")
    ap.add_argument("--mp", type=int, default=1,
                    help="POSIX processes sharing out-dir via fcntl; 1 = in-process")
    ap.add_argument("--shard", default="0/1",
                    help="i/N (0-indexed) keep when int(traj_hash[:8],16)%%N==i; checked before teacher HTTP")
    ap.add_argument("--external-chains", default=None,
                    help="JSONL pool of corpus-derived chain records "
                         f"(default: {EXTERNAL_CHAINS_PATH})")
    ap.add_argument("--external-frac", type=float, default=0.0,
                    help="fraction of agentic draws from the external chain pool, "
                         "0..1 (default 0.0); pool exhaustion falls back to plans")
    args = ap.parse_args()

    if args.legacy_v1:
        print("legacy-v1 is frozen; reversed-v2 is the only supported generator")
        raise SystemExit(2)

    mp_n = parse_mp(args.mp)
    shard_i, shard_n = parse_shard(args.shard)
    if not (0.0 <= args.external_frac <= 1.0):
        print("invalid --external-frac; expected a float in 0..1")
        raise SystemExit(2)
    if mp_n > 1 and fcntl is None:
        print("fcntl missing; use --shard on separate machines")
        raise SystemExit(2)
    if mp_n > 1 and args.shard != "0/1":
        print("do not combine --mp with explicit --shard; --mp implies --shard i/N per child")
        raise SystemExit(2)

    out_dir = _resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SAFETY GUARD: never silently overwrite existing data.
    # --mp requires an empty run-state dir (traces, checkpoints, dpo, tokens, card, holdout).
    run_state = files_to_archive(out_dir)
    if mp_n > 1 and run_state and not args.wipe:
        print("archive or wipe before --mp")
        raise SystemExit(1)

    existing = list(out_dir.glob("traces_*.jsonl"))
    if existing and not args.wipe:
        print(f"[GUARD] {out_dir} already has {len(existing)} trace file(s).")
        print("[GUARD] Run src/archive_data.py to archive them, then relaunch.")
        print("[GUARD] Or pass --wipe to archive existing traces, then proceed.")
        raise SystemExit(1)

    if args.wipe and files_to_archive(out_dir):
        try:
            archive_raw(
                raw_dir=out_dir,
                archive_dir=default_archive_dir(out_dir),
                label="wipe",
            )
        except Exception as e:
            print(f"[GUARD] archive failed: {e}")
            raise SystemExit(1)
        leftover_state = files_to_archive(out_dir)
        if mp_n > 1 and leftover_state:
            print("archive or wipe before --mp")
            raise SystemExit(1)
        leftover = list(out_dir.glob("traces_*.jsonl"))
        if leftover:
            print(f"[GUARD] archive left {len(leftover)} trace file(s) in place; refusing to continue.")
            raise SystemExit(1)

    try:
        roster = yaml.safe_load(Path(args.roster).read_text())
    except FileNotFoundError:
        print(
            f"roster not found: {args.roster} "
            "(copy configs/roster.example.yaml to configs/roster.yaml, "
            "or --roster pointing at packaged forge_assets/roster.example.yaml)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except yaml.YAMLError as e:
        print(f"roster YAML invalid: {e}", file=sys.stderr)
        raise SystemExit(2)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.pilot and not providers:
        pdef, mdef = _pilot_defaults(roster)
        providers = pdef
        if not models:
            models = mdef
    elif args.pilot and not models:
        _, mdef = _pilot_defaults(roster)
        # If a provider filter is set, use that provider's first model.
        if providers:
            pmap = dict(iter_provider_items(roster))
            conf = pmap.get(providers[0]) or {}
            mk = list((conf.get("models") or {}).keys())
            models = [mk[0]] if mk else mdef
        else:
            models = mdef
    cur_conf = (roster or {}).get("curriculum") or {}
    curriculum_mode = args.curriculum if args.curriculum is not None else cur_conf.get("mode", "uniform")
    holdout_frac = args.holdout_frac if args.holdout_frac is not None else float(cur_conf.get("holdout_frac", 0.15))
    ver_conf = (roster or {}).get("verify") or {}
    verify_sample = args.verify_sample if args.verify_sample is not None else float(ver_conf.get("sample_rate", 0.2))
    dpo_conf = (roster or {}).get("dpo") or {}
    dpo_enabled = True if args.dpo else bool(dpo_conf.get("enabled", False))
    dpo_rate = args.dpo_rate if args.dpo_rate is not None else float(dpo_conf.get("rate", 1.0))
    cross_teacher_rate = args.cross_teacher_rate if args.cross_teacher_rate is not None else 0.3
    distiller_kwargs = dict(
        provider_filter=providers or None,
        model_filter=models or None,
        curriculum_mode=curriculum_mode,
        holdout_frac=holdout_frac,
        write_eval_card=not args.no_eval_card,
        stamp_trace_eval=not args.no_trace_eval,
        verify_sample=verify_sample,
        no_verify=args.no_verify,
        cross_teacher=bool(args.cross_teacher),
        cross_teacher_rate=cross_teacher_rate,
        dpo_enabled=dpo_enabled,
        dpo_rate=dpo_rate,
        external_chains_path=args.external_chains,
        external_frac=args.external_frac,
    )
    if mp_n > 1:
        run_mp(
            roster_path=args.roster,
            out_dir=out_dir,
            count=args.count,
            nproc=mp_n,
            seed=args.seed,
            holdout_frac=holdout_frac,
            provider_filter=providers or None,
            model_filter=models or None,
            curriculum_mode=curriculum_mode,
            write_eval_card=not args.no_eval_card,
            stamp_trace_eval=not args.no_trace_eval,
            verify_sample=verify_sample,
            no_verify=args.no_verify,
            cross_teacher=bool(args.cross_teacher),
            cross_teacher_rate=cross_teacher_rate,
            dpo_enabled=dpo_enabled,
            dpo_rate=dpo_rate,
            pilot=args.pilot,
            external_chains_path=args.external_chains,
            external_frac=args.external_frac,
        )
        return
    d = Distiller(roster, out_dir, seed=args.seed, shard_i=shard_i, shard_n=shard_n,
                  **distiller_kwargs)
    asyncio.run(d.run(args.count, pilot=args.pilot))


if __name__ == "__main__":
    main()