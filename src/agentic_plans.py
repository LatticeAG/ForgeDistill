"""agentic_plans.py - DETERMINISTIC agentic chain definitions.

Core insight: don't ask the teacher to demonstrate multi-step agentic
behavior (weak models fail -> 82% single-round traces). Instead, WE define
the exact tool-call sequence per template with REAL dependencies, execute
it against the mock executor, and the teacher writes only the prose.

Value references (in step args):
  "$S.result.field"     -> field from step S's result (S = step index)
  "$S.result.rows[0].f" -> field from first row of a list result
Refs always point to a PRIOR step (dependency), and only to successful
steps. Error steps carry result with status != 200 and are used for
recovery turns.

Every chain is guaranteed correct BY CONSTRUCTION:
  - send_email 'to' is always the opaque email a prior get_user returned
    (except intentional negative/partial-failure steps tagged expect=error)
  - db_query plan filters use the exact plan string from get_user
  - bad-id / bad-city first steps fail, then the plan corrects OR stops
"""
from __future__ import annotations
import hashlib
import json
import random
import re

from mock_tools import (
    EMAIL_BY_USER,
    EXISTING_FILES,
    USERS,
    execute_one,
)

# ----------------------------------------------------------------------
# Pools. Executor lives in mock_tools; these are plan-building only.
# ----------------------------------------------------------------------
GOOD_IDS = [42, 7, 99, 123]
BAD_IDS = [0, -1, 9999, 2.5]
GOOD_CITIES = ["London", "Paris", "New York", "Tokyo"]
BAD_CITIES = ["Atlantis", "Wakanda"]  # no empty string: that is INVALID_CITY, not CITY_NOT_FOUND
GOOD_FILES = ["data.csv", "report.pdf", "config.json", "logs/error.log", "src/main.py"]
MISSING_FILES = ["missing.csv", "deleted.txt", "absent.log", "tmp/gone.dat"]
SUBJECTS = ["Your account update", "Welcome to the platform", "Quarterly report", "Status notification", "Action required"]
GUESSED_EMAILS = ["alice@example.com", "user@internal.corp", "admin@internal.corp", "bob@gmail.com"]

assert not (set(MISSING_FILES) & EXISTING_FILES)
assert "" not in BAD_CITIES


# ----------------------------------------------------------------------
# Ref resolution
# ----------------------------------------------------------------------
_REF_RE = re.compile(r"^\$(\d+)\.result(?:\.rows\[(\d+)\])?(?:\.([A-Za-z_][A-Za-z0-9_]*))?$")
_REF_RE_SUB = re.compile(r"\$(\d+)\.result(?:\.rows\[(\d+)\])?(?:\.([A-Za-z_][A-Za-z0-9_]*))?")


def _resolve_ref(ref: str, steps: list[dict]) -> object:
    """Resolve "$S.result[.rows[i]][.field]" against executed steps.

    refs index into steps (including error steps), NOT into a success list.
    """
    m = _REF_RE.match(ref)
    if not m:
        raise ValueError(f"bad ref: {ref}")
    si, row, field = int(m.group(1)), m.group(2), m.group(3)
    if si >= len(steps):
        raise ValueError(f"ref step {si} out of range (only {len(steps)} steps)")
    payload = steps[si]["result"]
    if payload.get("status") != 200:
        raise ValueError(f"ref step {si} is an error step - cannot dereference result")
    result = payload["result"]
    if row is not None:
        result = result["rows"][int(row)]
    if field:
        if field not in result:
            raise ValueError(f"ref field '{field}' not in step {si} result: {result}")
        return result[field]
    return result


def _fill_args(args_tpl: dict, steps: list[dict], vars_: dict) -> dict:
    """Fill step args: {var} from vars_, '$S.result...' refs from prior steps
    (both as whole values and inline inside strings)."""
    out = {}
    for k, v in args_tpl.items():
        if isinstance(v, str) and v.startswith("$") and _REF_RE.match(v):
            out[k] = _resolve_ref(v, steps)
        elif isinstance(v, str):
            def _sub(m):
                ref = f"${m.group(1)}.result"
                if m.group(2) is not None:
                    ref += f".rows[{m.group(2)}]"
                if m.group(3) is not None:
                    ref += f".{m.group(3)}"
                return str(_resolve_ref(ref, steps))
            resolved = _REF_RE_SUB.sub(_sub, v)
            # Bare placeholder like "{uid}" keeps the ORIGINAL type (int stays int)
            m = re.fullmatch(r"\{([a-z_0-9]+)\}", resolved)
            if m and m.group(1) in vars_ and vars_[m.group(1)] is not None:
                out[k] = vars_[m.group(1)]
            else:
                out[k] = resolved.format(**vars_)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------
# Chain templates. EVERY step's args that must come from a prior call
# uses a $S.result ref. send_email 'to' is ALWAYS a $ ref on success
# paths, so the dependency is structurally unbreakable.
#
# skills tag the pedagogical target; build_chain samples stratified by
# skill rather than rng.choice(PLANS).
# ----------------------------------------------------------------------
PLANS = [
    # --- get_user -> send_email ----------------------------------------
    {
        "id": "user-email-welcome",
        "skills": ["multi_hop", "stop"],
        "prompt": "Look up user {uid} and send them a welcome email using their registered address.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "Welcome to the platform",
                                            "body": "Hi $0.result.name, welcome aboard! Your plan: $0.result.plan"}},
        ],
    },
    {
        "id": "user-email-notify",
        "skills": ["multi_hop", "stop"],
        "prompt": "Fetch user {uid}'s profile, then send a notification to their email address about '{subject}'.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "{subject}",
                                            "body": "Hello $0.result.name, this is an automated notification."}},
        ],
    },
    {
        "id": "user-email-get-address",
        "skills": ["multi_hop", "stop"],
        "prompt": "Get the email address of user {uid}, then send them '{subject}'.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "{subject}", "body": "Hi $0.result.name."}},
        ],
    },
    {
        "id": "user-email-account",
        "skills": ["multi_hop", "stop"],
        "prompt": "Retrieve user {uid} and email their account about '{subject}'.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "{subject}",
                                            "body": "Dear $0.result.name, regarding {subject}."}},
        ],
    },
    # --- get_user -> db_query (plan learned from user) -----------------
    {
        "id": "user-plan-count",
        "skills": ["multi_hop"],
        "prompt": "Get user {uid}'s plan from their profile, then count users on that same plan.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "db_query", "args": {"sql": "SELECT count(*) FROM users WHERE plan = '$0.result.plan'"}},
        ],
    },
    {
        "id": "user-plan-list",
        "skills": ["multi_hop"],
        "prompt": "Fetch user {uid}, read their plan, then query the database for other users on that plan.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "db_query", "args": {"sql": "SELECT * FROM users WHERE plan = '$0.result.plan'"}},
        ],
    },
    # --- weather -> send_email (weather FIRST) -------------------------
    {
        "id": "weather-then-user-email",
        "skills": ["multi_hop", "reorder"],
        "prompt": "Check the weather in {city}, then email the details to user {uid}.",
        "plan": [
            {"tool": "weather.get", "args": {"city": "{city}"}},
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$1.result.email", "subject": "Weather update",
                                            "body": "In {city} it is $0.result.condition at $0.result.temp_c C."}},
        ],
    },
    {
        "id": "weather-report-email",
        "skills": ["multi_hop", "reorder"],
        "prompt": "Get weather for {city}, then send a weather report to user {uid}'s address.",
        "plan": [
            {"tool": "weather.get", "args": {"city": "{city}"}},
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$1.result.email", "subject": "Weather report",
                                            "body": "Current conditions in {city}: $0.result.condition, $0.result.temp_c C."}},
        ],
    },
    # --- db_query -> send_email ----------------------------------------
    {
        "id": "db-count-email",
        "skills": ["multi_hop"],
        "prompt": "Query the database for order count, then send user {uid} the number.",
        "plan": [
            {"tool": "db_query", "args": {"sql": "SELECT count(*) FROM orders"}},
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$1.result.email", "subject": "Order count",
                                            "body": "Total orders: $0.result.rows[0].count"}},
        ],
    },
    # --- file_exists -> notify -----------------------------------------
    {
        "id": "file-status-email",
        "skills": ["multi_hop"],
        "prompt": "Verify {file} is present, then notify user {uid} about its status.",
        "plan": [
            {"tool": "file_exists", "args": {"filepath": "{file}"}},
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$1.result.email", "subject": "File status",
                                            "body": "File {file} exists: $0.result.exists."}},
        ],
    },
    {
        "id": "user-file-status",
        "skills": ["multi_hop", "digest"],
        "prompt": "Look up user {uid}, check if {file} exists, then email them a full status report.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "file_exists", "args": {"filepath": "{file}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "Status report",
                                            "body": "User $0.result.name (plan $0.result.plan). File {file} exists: $1.result.exists."}},
        ],
    },
    # --- honest "only if" / stop-on-failure (NO silent city/id substitute)
    {
        "id": "bad-id-stop",
        "skills": ["branch"],
        "prompt": "Fetch user {bad_id} and, only if that lookup succeeds, email them about '{subject}'. Also check whether {file} exists. If the user is not found, do not email anyone else and do not try a different id.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{bad_id}"}, "expect": "error"},
            {"tool": "file_exists", "args": {"filepath": "{file}"}},
        ],
    },
    {
        "id": "bad-id-stop-v2",
        "skills": ["branch"],
        "prompt": "Look up user {bad_id}; email them '{subject}' only if that exact id exists. Separately check {file}. Do not substitute a different user id.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{bad_id}"}, "expect": "error"},
            {"tool": "file_exists", "args": {"filepath": "{file}"}},
        ],
    },
    {
        "id": "bad-city-only-if",
        "skills": ["branch"],
        "prompt": "Look up user {uid}, then get weather in {bad_city}. Email them the weather only if the weather lookup works. If there is no weather data, do not email and do not try a different city.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "weather.get", "args": {"city": "{bad_city}"}, "expect": "error"},
        ],
    },
    # --- two-user fan-out ----------------------------------------------
    {
        "id": "two-user-notify",
        "skills": ["fanout", "multi_hop"],
        "prompt": "Look up user {uid} and user {uid2}, then send each a notification.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "get_user", "args": {"user_id": "{uid2}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "Notification", "body": "Hi $0.result.name."}},
            {"tool": "send_email", "args": {"to": "$1.result.email", "subject": "Notification", "body": "Hi $1.result.name."}},
        ],
    },
    {
        "id": "two-user-subject",
        "skills": ["fanout", "multi_hop"],
        "prompt": "Fetch users {uid} and {uid2}, then email both about '{subject}'.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "get_user", "args": {"user_id": "{uid2}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "{subject}", "body": "Hi $0.result.name."}},
            {"tool": "send_email", "args": {"to": "$1.result.email", "subject": "{subject}", "body": "Hi $1.result.name."}},
        ],
    },
    # ==================================================================
    # NEW templates (P1): join / branch / recovery / fanout / reorder /
    # idempotent / schema / digest / disambiguate / stop
    # ==================================================================
    {
        "id": "join-two-users-plan-email",
        "skills": ["join", "multi_hop"],
        "prompt": "Look up user {uid} and user {uid2}. Query how many users share {uid}'s plan, then email {uid} a comparison of both users' plans and that count. Send only one email.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "get_user", "args": {"user_id": "{uid2}"}},
            {"tool": "db_query", "args": {"sql": "SELECT count(*) FROM users WHERE plan = '$0.result.plan'"}},
            {"tool": "send_email", "args": {
                "to": "$0.result.email",
                "subject": "Plan comparison",
                "body": "$0.result.name is on $0.result.plan ($2.result.rows[0].count users). $1.result.name is on $1.result.plan.",
            }},
        ],
    },
    {
        "id": "file-missing-no-email",
        "skills": ["branch"],
        "prompt": "Look up user {uid}. If {missing_file} exists, email them about it. If it does not exist, do not send any email.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "file_exists", "args": {"filepath": "{missing_file}"}},
        ],
    },
    {
        "id": "get-user-str-retry-int",
        "skills": ["recovery"],
        "prompt": "Fetch user {uid} (the id may arrive as a string) and email them '{subject}' at their registered address.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid_str}"}, "expect": "error"},
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$1.result.email", "subject": "{subject}", "body": "Hi $1.result.name."}},
        ],
    },
    {
        "id": "fanout-partial-fail",
        "skills": ["fanout", "branch"],
        "prompt": "Look up users {uid} and {uid2}. Email {uid} about '{subject}'. Also attempt to email {uid2} but the second send will have a blank subject and should fail. Report the mixed outcome; do not hide the failure.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "get_user", "args": {"user_id": "{uid2}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "{subject}", "body": "Hi $0.result.name."}},
            {"tool": "send_email", "args": {"to": "$1.result.email", "subject": "", "body": "Hi $1.result.name."}, "expect": "error"},
        ],
    },
    {
        "id": "plan-filter-email-count",
        "skills": ["multi_hop"],
        "prompt": "Get user {uid}'s plan, count how many users are on that plan, then email {uid} the count.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "db_query", "args": {"sql": "SELECT count(*) FROM users WHERE plan = '$0.result.plan'"}},
            {"tool": "send_email", "args": {
                "to": "$0.result.email",
                "subject": "Plan census",
                "body": "Hi $0.result.name. Your plan $0.result.plan has $1.result.rows[0].count users.",
            }},
        ],
    },
    {
        "id": "user-then-weather-email",
        "skills": ["reorder", "multi_hop"],
        "prompt": "Look up user {uid} first, then check the weather in {city}, then email them the forecast. Do the user lookup before the weather call.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "weather.get", "args": {"city": "{city}"}},
            {"tool": "send_email", "args": {
                "to": "$0.result.email",
                "subject": "Weather update",
                "body": "Hi $0.result.name. In {city} it is $1.result.condition at $1.result.temp_c C.",
            }},
        ],
    },
    {
        "id": "idempotent-reuse-email",
        "skills": ["idempotent", "recovery", "schema"],
        "prompt": "Look up user {uid} once, then email them '{subject}'. If the first send is rejected, retry the send using the email you already learned — do not look the user up again.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "", "body": "Hi $0.result.name."}, "expect": "error"},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "{subject}", "body": "Hi $0.result.name."}},
        ],
    },
    {
        "id": "empty-subject-retry",
        "skills": ["schema", "recovery"],
        "prompt": "Look up user {uid} and email them. If send_email rejects an empty subject, retry with subject '{subject}'.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "", "body": "Hi $0.result.name."}, "expect": "error"},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "{subject}", "body": "Hi $0.result.name, this is the corrected send."}},
        ],
    },
    {
        "id": "weather-db-join-email",
        "skills": ["join", "multi_hop"],
        "prompt": "Get the weather in {city} and the order count from the database, then email user {uid} both facts in a single message.",
        "plan": [
            {"tool": "weather.get", "args": {"city": "{city}"}},
            {"tool": "db_query", "args": {"sql": "SELECT count(*) FROM orders"}},
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {
                "to": "$2.result.email",
                "subject": "Weather and orders",
                "body": "Hi $2.result.name. {city}: $0.result.condition $0.result.temp_c C. Orders: $1.result.rows[0].count.",
            }},
        ],
    },
    {
        "id": "guess-email-then-correct",
        "skills": ["recovery"],
        "prompt": "Email user {uid} about '{subject}'. If you guess their address it will fail — look them up and send to the registered address.",
        "plan": [
            {"tool": "send_email", "args": {"to": "{guessed_email}", "subject": "{subject}", "body": "Hello."}, "expect": "error"},
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$1.result.email", "subject": "{subject}", "body": "Hi $1.result.name."}},
        ],
    },
    {
        "id": "five-step-digest",
        "skills": ["digest", "multi_hop"],
        "prompt": "Build a digest for user {uid}: look them up, check whether {file} exists, get weather in {city}, count users on their plan, then email them one summary. Do not add extra calls after the email.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "file_exists", "args": {"filepath": "{file}"}},
            {"tool": "weather.get", "args": {"city": "{city}"}},
            {"tool": "db_query", "args": {"sql": "SELECT count(*) FROM users WHERE plan = '$0.result.plan'"}},
            {"tool": "send_email", "args": {
                "to": "$0.result.email",
                "subject": "Daily digest",
                "body": "Hi $0.result.name (plan $0.result.plan, $3.result.rows[0].count peers). File {file} exists=$1.result.exists. {city}: $2.result.condition $2.result.temp_c C.",
            }},
        ],
    },
    {
        "id": "two-candidate-id",
        "skills": ["disambiguate", "recovery"],
        "prompt": "The user id might be {bad_id} or {fix_id}. Try {bad_id} first; if that returns 404, try {fix_id}, then email them '{subject}'.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{bad_id}"}, "expect": "error"},
            {"tool": "get_user", "args": {"user_id": "{fix_id}"}},
            {"tool": "send_email", "args": {"to": "$1.result.email", "subject": "{subject}", "body": "Hi $1.result.name."}},
        ],
    },
    {
        "id": "stop-after-welcome",
        "skills": ["stop", "multi_hop"],
        "prompt": "Look up user {uid} and send a single welcome email to their registered address. Stop after that send succeeds — do not make extra tool calls.",
        "plan": [
            {"tool": "get_user", "args": {"user_id": "{uid}"}},
            {"tool": "send_email", "args": {"to": "$0.result.email", "subject": "Welcome to the platform",
                                            "body": "Hi $0.result.name, welcome aboard."}},
        ],
    },
]

VAR_POOLS = {
    "uid": GOOD_IDS, "uid2": GOOD_IDS, "bad_id": BAD_IDS, "fix_id": GOOD_IDS,
    "good_city": GOOD_CITIES, "bad_city": BAD_CITIES, "city": GOOD_CITIES,
    "file": GOOD_FILES, "missing_file": MISSING_FILES, "subject": SUBJECTS,
    "guessed_email": GUESSED_EMAILS,
}

CHAIN_COUNT = len(PLANS)
DERIVED_VARS = {"uid_str"}


def _plans_by_skill() -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for i, p in enumerate(PLANS):
        skills = p.get("skills") or ["multi_hop"]
        for sk in skills:
            buckets.setdefault(sk, []).append(i)
    return buckets


_SKILL_BUCKETS = _plans_by_skill()
SKILLS = list(_SKILL_BUCKETS.keys())


def _pick_vars(plan_tpl: dict, rng: random.Random) -> dict:
    vars_ = {}
    text = plan_tpl["prompt"] + " " + str(plan_tpl["plan"])
    needed = set(re.findall(r"\{([a-z_0-9]+)\}", text))
    if "uid_str" in needed:
        needed.add("uid")
    for name in sorted(needed - DERIVED_VARS):
        if name not in VAR_POOLS:
            continue
        pool = list(VAR_POOLS[name])
        if name == "fix_id" and "bad_id" in vars_:
            pool = [i for i in pool if i != vars_["bad_id"]]
        if name == "uid2" and "uid" in vars_ and isinstance(vars_["uid"], int):
            pool = [i for i in pool if i != vars_["uid"]]
        if name == "good_city" and "bad_city" in vars_:
            bc = str(vars_["bad_city"]).lower()
            pool = [c for c in pool if c.lower() != bc]
        if pool:
            vars_[name] = rng.choice(pool)
    if "uid_str" in needed:
        uid = vars_.get("uid", rng.choice(GOOD_IDS))
        vars_["uid"] = uid
        vars_["uid_str"] = str(uid)
    return vars_


def trajectory_hash(traj: dict) -> str:
    """Stable hash of (plan_id, vars, step args/results). Not just the prompt."""
    payload = {
        "plan_id": traj.get("plan_id", traj.get("plan_index")),
        "vars": traj.get("vars", {}),
        "steps": [
            {"tool": s["tool"], "args": s["args"], "result": s["result"]}
            for s in traj.get("steps", [])
        ],
    }
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_chain(rng: random.Random, skill: str | None = None,
                plan_index: int | None = None) -> dict:
    """Pick a plan stratified by skill, fill vars/refs, execute deterministically.

    Returns prompt, steps, vars, plan_index, plan_id, skills.
    """
    buckets = _SKILL_BUCKETS
    if plan_index is None:
        if skill is None or skill not in buckets:
            skill = rng.choice(list(buckets.keys()))
        plan_index = rng.choice(buckets[skill])
    plan_tpl = PLANS[plan_index]
    vars_ = _pick_vars(plan_tpl, rng)

    steps = []
    for i, step_tpl in enumerate(plan_tpl["plan"]):
        tool = step_tpl["tool"]
        args = _fill_args(step_tpl["args"], steps, vars_)
        result = execute_one(tool, args)
        steps.append({
            "tool": tool,
            "args": args,
            "result": result,
            "expect": step_tpl.get("expect", "success"),
            "step_index": i,
        })

    fmt_vars = {k: v for k, v in vars_.items() if v is not None}
    prompt = plan_tpl["prompt"].format(**fmt_vars)
    return {
        "prompt": prompt,
        "steps": steps,
        "vars": vars_,
        "plan_index": plan_index,
        "plan_id": plan_tpl["id"],
        "skills": list(plan_tpl.get("skills") or []),
    }


def validate_chain(steps: list[dict]) -> str | None:
    """Sanity check a built chain. Returns error string or None.

    Rules:
    - successful send_email 'to' must be a registered opaque email
    - a failing step is OK if expect is error/stop, OR it is immediately
      followed by a successful retry of the same tool
    - a later non-immediate recovery (e.g. guess email -> get_user -> send)
      must be tagged expect=error
    - the FINAL step must succeed unless it is an expected error/stop
    """
    n = len(steps)
    if not steps:
        return "empty chain"
    for i, s in enumerate(steps):
        r = s["result"]
        expect = s.get("expect", "success")
        if r.get("status") == 200:
            if expect == "error":
                return f"step {i}: expected error but got success"
            if s["tool"] == "send_email" and r["result"].get("to") not in EMAIL_BY_USER.values():
                return f"step {i}: send_email to unregistered address"
            continue
        # error step
        if expect in ("error", "stop"):
            continue
        # recovery: a later call of the same tool succeeds (immediate retry
        # OR delayed: guess email -> get_user -> send_email)
        later_ok = any(
            steps[j]["tool"] == s["tool"] and steps[j]["result"].get("status") == 200
            for j in range(i + 1, n)
        )
        if not later_ok:
            return f"step {i}: unexpected error {r.get('error', {}).get('code')} (no recovery)"
    last = steps[-1]
    if last["result"].get("status") != 200 and last.get("expect") not in ("error", "stop"):
        return "chain does not complete (final step failed)"
    return None
