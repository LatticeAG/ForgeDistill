"""Deterministic mock tool executor with dependency enforcement (v2).

Agentic-only design: tools have REAL dependencies between them.
- User email addresses are OPAQUE registry values (u42@internal.corp).
  The teacher CANNOT guess them - they MUST call get_user first to learn
  the address, then pass it to send_email. Parallel guessing fails with 400.
- db_query with plan filter validates against known plans -> teacher must
  get_user first to learn the plan.
- weather.get feeds send_email bodies.
This forces true sequential multi-turn chains: call -> read result -> call
again with derived arguments. That is the agentic training signal.
"""
from __future__ import annotations
import json
import random

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "file_exists",
            "description": "Check whether a file exists on the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user",
            "description": "Fetch a registered user profile by numeric ID. Returns name, opaque email address, and plan. Use this to learn a user's email address before emailing them.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "integer"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "db_query",
            "description": "Run a read-only SQL query against the analytics database. SELECT only. When filtering by plan, use the exact plan string from a user profile.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather.get",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient address with a subject and body. The 'to' address MUST be a registered user's opaque email address previously returned by get_user - you cannot invent or guess addresses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]

EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}

# Files that exist in the mock filesystem
EXISTING_FILES = {"data.csv", "report.pdf", "config.json", "logs/error.log", "src/main.py"}

# Opaque email registry - addresses are NOT guessable from user IDs.
# Teacher must call get_user to learn them. (deliberately non-obvious)
EMAIL_BY_USER = {
    42: "alice.q.8172@internal.corp",
    7: "bob.w.3341@internal.corp",
    99: "carol.y.9024@internal.corp",
    123: "dave.z.1157@internal.corp",
}

USERS = {
    42: {"name": "Alice", "email": EMAIL_BY_USER[42], "plan": "pro"},
    7: {"name": "Bob", "email": EMAIL_BY_USER[7], "plan": "free"},
    99: {"name": "Carol", "email": EMAIL_BY_USER[99], "plan": "pro"},
    123: {"name": "Dave", "email": EMAIL_BY_USER[123], "plan": "enterprise"},
}

KNOWN_PLANS = {"pro", "free", "enterprise"}

WEATHER = {
    "london": {"temp_c": 14, "condition": "cloudy"},
    "paris": {"temp_c": 16, "condition": "sunny"},
    "new york": {"temp_c": 21, "condition": "clear"},
    "tokyo": {"temp_c": 18, "condition": "rain"},
}


def execute_one(name: str, arguments: dict) -> dict:
    """Canonical single-call executor. Used by agentic_plans sequential chains
    and by execute_tool_calls (legacy parallel batch)."""
    return _exec_call(name, arguments, random.Random(0))


def _exec_call(name: str, arguments: dict, rng: random.Random) -> dict:
    """Execute one tool call. Returns the raw result payload (success or error)."""
    if name == "file_exists":
        fp = arguments.get("filepath", "")
        if not isinstance(fp, str) or not fp:
            return {"status": 400, "error": {"code": "INVALID_FILEPATH", "message": "filepath must be a non-empty string"}}
        return {"status": 200, "result": {"exists": fp in EXISTING_FILES, "path": fp}}

    if name == "get_user":
        uid = arguments.get("user_id")
        if not isinstance(uid, int) or uid <= 0:
            return {"status": 404, "error": {"code": "USER_NOT_FOUND", "message": f"user_id must be a positive integer, got {uid!r}"}}
        if uid not in USERS:
            return {"status": 404, "error": {"code": "USER_NOT_FOUND", "message": f"no user registered with id {uid}"}}
        return {"status": 200, "result": USERS[uid]}

    if name == "db_query":
        sql = arguments.get("sql", "")
        low = sql.lower()
        if not isinstance(sql, str) or not sql.strip():
            return {"status": 400, "error": {"code": "INVALID_SQL", "message": "sql must be a non-empty string"}}
        if not low.startswith("select"):
            return {"status": 400, "error": {"code": "INVALID_SQL", "message": "only SELECT queries are allowed"}}
        if ";" in sql.rstrip(";").rstrip():
            return {"status": 400, "error": {"code": "INVALID_SQL", "message": "multiple statements are not supported"}}
        # Dependency enforcement: plan filter must reference a KNOWN plan
        if "plan" in low and "where" in low:
            import re
            m = re.search(r"plan\s*=\s*['\"]([^'\"]+)['\"]", low)
            if m:
                plan = m.group(1)
                if plan not in KNOWN_PLANS:
                    return {"status": 400, "error": {"code": "UNKNOWN_PLAN", "message": f"unknown plan '{plan}'. Get the exact plan string from a user profile first."}}
                return {"status": 200, "result": {"rows": [{"plan": plan, "count": 12}], "sql": sql}}
        return {"status": 200, "result": {"rows": [{"count": 42}], "sql": sql}}

    if name == "weather.get":
        city = arguments.get("city", "")
        key = str(city).strip().lower()
        if not key:
            return {"status": 400, "error": {"code": "INVALID_CITY", "message": "city must be a non-empty string"}}
        if key not in WEATHER:
            return {"status": 404, "error": {"code": "CITY_NOT_FOUND", "message": f"no weather data for city {city!r}"}}
        return {"status": 200, "result": WEATHER[key]}

    if name == "send_email":
        to = arguments.get("to", "")
        subject = arguments.get("subject", "")
        body = arguments.get("body", "")
        if not isinstance(to, str) or not to:
            return {"status": 400, "error": {"code": "INVALID_RECIPIENT", "message": "'to' must be a non-empty string"}}
        # DEPENDENCY ENFORCEMENT: 'to' must be a registered opaque address.
        # Guessing/paraphrasing fails -> teacher must get_user first.
        if to not in EMAIL_BY_USER.values():
            return {"status": 400, "error": {"code": "UNREGISTERED_ADDRESS", "message": f"'{to}' is not a registered user address. Call get_user to fetch the user's email first."}}
        if not subject.strip():
            return {"status": 400, "error": {"code": "INVALID_SUBJECT", "message": "subject must be a non-empty string"}}
        return {"status": 200, "result": {"sent": True, "to": to, "subject": subject}}

    return {"status": 400, "error": {"code": "UNKNOWN_TOOL", "message": f"no such tool: {name}"}}


def execute_tool_calls(calls: list[dict]) -> dict:
    """Execute a batch of tool calls (parallel). Returns {tool_name: result}."""
    rng = random.Random(42)
    results = {}
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if not (isinstance(args, dict) and set(args.keys()) <= {"filepath", "user_id", "sql", "city", "to", "subject", "body"}):
            results[name] = {"status": 400, "error": {"code": "INVALID_ARGS", "message": "arguments do not match tool schema"}}
            continue
        results[name] = execute_one(name, args)
    return results
