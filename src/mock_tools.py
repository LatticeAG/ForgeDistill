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
import hashlib
import json
import random
import re

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
    {
        "type": "function",
        "function": {
            "name": "calendar.list",
            "description": "List calendar events for a registered user id.",
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
            "name": "calendar.create",
            "description": "Create a calendar event inviting a registered opaque email address. Guessing an address fails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "attendee_email": {"type": "string"},
                },
                "required": ["title", "start", "attendee_email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search.query",
            "description": "Search the document index. Returns hits with opaque doc ids that must be fetched via search.get.",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search.get",
            "description": "Fetch a document by opaque doc_id previously returned by search.query.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc.eval",
            "description": "Evaluate a binary arithmetic expression of the form NUMBER OP NUMBER. OP is one of + - * /.",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crm.get_account",
            "description": "Fetch a CRM account by numeric id. Owner email is an opaque registry address.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "integer"}},
                "required": ["account_id"],
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

USER_BY_EMAIL = {email: uid for uid, email in EMAIL_BY_USER.items()}

_CALC_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*$")

# Opaque doc ids are not concatenations of the query string.
_DOC_CENSUS = {
    "doc_id": "doc.k7x2",
    "title": "Quarterly census",
    "snippet": "Published census figures for the period.",
    "body": "The published figure is 48 units.",
}
_DOC_CAP = {
    "doc_id": "doc.m3p9",
    "title": "Capacity brief",
    "snippet": "Seat and capacity notes.",
    "body": "Allocated capacity stands at 12 seats.",
}
_DOC_OPS = {
    "doc_id": "doc.w4n1",
    "title": "Ops handbook",
    "snippet": "Internal operations overview.",
    "body": "Handbook revision 3 is current.",
}

DOC_BY_QUERY = {
    "census": [_DOC_CENSUS],
    "figure": [_DOC_CENSUS],
    "quarterly": [_DOC_CENSUS, _DOC_CAP],
    "capacity": [_DOC_CAP],
    "brief": [_DOC_CAP],
    "handbook": [_DOC_OPS],
    "ops": [_DOC_OPS],
    "xyzzy": [],
    "missingterm": [],
}

DOC_BY_ID = {}
for _docs in DOC_BY_QUERY.values():
    for _d in _docs:
        DOC_BY_ID[_d["doc_id"]] = _d

# 1-2 deterministic pre-seeded events per known user. event_id minted via sha256.
CALENDAR_SEED = {
    42: [
        {"title": "Sprint planning", "start": "2026-08-14T09:00:00"},
        {"title": "1:1 with Bob", "start": "2026-08-14T14:00:00"},
    ],
    7: [
        {"title": "Design review", "start": "2026-08-15T10:00:00"},
    ],
    99: [
        {"title": "Quarterly sync", "start": "2026-08-16T14:30:00"},
        {"title": "Kickoff", "start": "2026-08-17T11:00:00"},
    ],
    123: [
        {"title": "Ops standup", "start": "2026-08-18T09:30:00"},
    ],
}


def mint_event_id(user_id: int, title: str) -> str:
    digest = hashlib.sha256(f"{user_id}:{title}".encode("utf-8")).hexdigest()[:12]
    return f"evt.{digest}"


ACCOUNT_BY_ID = {
    1001: {
        "id": 1001,
        "name": "Northwind",
        "owner": {"name": "Alice", "email": EMAIL_BY_USER[42]},
        "billing": {"plan": "pro", "seats": 12},
    },
    1002: {
        "id": 1002,
        "name": "Contoso",
        "owner": {"name": "Bob", "email": EMAIL_BY_USER[7]},
        "billing": {"plan": "enterprise", "seats": 40},
    },
}

ALLOWED_ARG_KEYS = {
    "filepath", "user_id", "sql", "city", "to", "subject", "body",
    "q", "doc_id", "expr", "account_id", "title", "start", "attendee_email",
    "event_id",
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

    if name == "calendar.list":
        uid = arguments.get("user_id")
        if not isinstance(uid, int) or uid <= 0:
            return {"status": 404, "error": {"code": "USER_NOT_FOUND", "message": f"user_id must be a positive integer, got {uid!r}"}}
        if uid not in USERS:
            return {"status": 404, "error": {"code": "USER_NOT_FOUND", "message": f"no user registered with id {uid}"}}
        events = []
        for ev in CALENDAR_SEED.get(uid, []):
            events.append({
                "event_id": mint_event_id(uid, ev["title"]),
                "title": ev["title"],
                "start": ev["start"],
            })
        return {"status": 200, "result": {"events": events}}

    if name == "calendar.create":
        title = arguments.get("title", "")
        start = arguments.get("start", "")
        attendee = arguments.get("attendee_email", "")
        if not isinstance(title, str) or not title.strip():
            return {"status": 400, "error": {"code": "INVALID_TITLE", "message": "title must be a non-empty string"}}
        if not isinstance(attendee, str) or not attendee:
            return {"status": 400, "error": {"code": "UNREGISTERED_ADDRESS", "message": "attendee_email must be a registered user address"}}
        if attendee not in USER_BY_EMAIL:
            return {"status": 400, "error": {"code": "UNREGISTERED_ADDRESS", "message": f"'{attendee}' is not a registered user address. Call get_user to fetch the user's email first."}}
        uid = USER_BY_EMAIL[attendee]
        return {"status": 200, "result": {
            "created": True,
            "event_id": mint_event_id(uid, title),
            "attendee_email": attendee,
        }}

    if name == "search.query":
        q = arguments.get("q", "")
        if not isinstance(q, str) or not q.strip():
            return {"status": 400, "error": {"code": "INVALID_QUERY", "message": "q must be a non-empty string"}}
        tokens = q.lower().split()
        if not tokens:
            return {"status": 400, "error": {"code": "INVALID_QUERY", "message": "q must be a non-empty string"}}
        id_sets = []
        for tok in tokens:
            ids = {d["doc_id"] for d in DOC_BY_QUERY.get(tok, [])}
            id_sets.append(ids)
        common = set.intersection(*id_sets) if id_sets else set()
        hits = []
        for d in DOC_BY_QUERY.get(tokens[0], []):
            if d["doc_id"] in common:
                hits.append({"doc_id": d["doc_id"], "title": d["title"], "snippet": d["snippet"]})
        return {"status": 200, "result": {"hits": hits}}

    if name == "search.get":
        doc_id = arguments.get("doc_id", "")
        if not isinstance(doc_id, str) or doc_id not in DOC_BY_ID:
            return {"status": 400, "error": {"code": "UNKNOWN_DOC", "message": f"no document registered with id {doc_id!r}"}}
        doc = DOC_BY_ID[doc_id]
        return {"status": 200, "result": {"doc_id": doc["doc_id"], "title": doc["title"], "body": doc["body"]}}

    if name == "calc.eval":
        expr = arguments.get("expr", "")
        if not isinstance(expr, str):
            return {"status": 400, "error": {"code": "INVALID_EXPR", "message": "expr must be a NUMBER OP NUMBER string"}}
        m = _CALC_RE.match(expr)
        if not m:
            return {"status": 400, "error": {"code": "INVALID_EXPR", "message": "expr must be NUMBER OP NUMBER with OP in + - * /"}}
        left_s, op, right_s = m.group(1), m.group(2), m.group(3)
        left = float(left_s) if "." in left_s else int(left_s)
        right = float(right_s) if "." in right_s else int(right_s)
        if op == "/" and right == 0:
            return {"status": 400, "error": {"code": "INVALID_EXPR", "message": "division by zero"}}
        if op == "+":
            value = left + right
        elif op == "-":
            value = left - right
        elif op == "*":
            value = left * right
        else:
            value = left / right
        if isinstance(value, float) and value == int(value):
            value = int(value)
        return {"status": 200, "result": {"expr": expr, "value": value}}

    if name == "crm.get_account":
        aid = arguments.get("account_id")
        if not isinstance(aid, int) or aid not in ACCOUNT_BY_ID:
            return {"status": 404, "error": {"code": "UNKNOWN_ACCOUNT", "message": f"no account registered with id {aid!r}"}}
        return {"status": 200, "result": {"account": ACCOUNT_BY_ID[aid]}}

    return {"status": 400, "error": {"code": "UNKNOWN_TOOL", "message": f"no such tool: {name}"}}


def execute_tool_calls(calls: list[dict]) -> dict:
    """Execute a batch of tool calls (parallel). Returns {tool_name: result}."""
    rng = random.Random(42)
    results = {}
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if not (isinstance(args, dict) and set(args.keys()) <= ALLOWED_ARG_KEYS):
            results[name] = {"status": 400, "error": {"code": "INVALID_ARGS", "message": "arguments do not match tool schema"}}
            continue
        results[name] = execute_one(name, args)
    return results
