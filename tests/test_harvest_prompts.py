"""Offline tests for src/harvest_prompts.py extraction shapes.

The prompt lane stays a prompt source only - nothing here is wired into
generation (the chain lane lives in external_chains.py / agentic_plans).
"""
from __future__ import annotations

import json

from harvest_prompts import (
    SOURCES,
    extract_user_prompts,
    is_good_prompt,
)


def test_sharegpt_from_value_shape():
    # NousResearch/hermes-function-calling-v1 and Team-ACE/ToolACE rows.
    row = {"conversations": [
        {"from": "system", "value": "You are helpful."},
        {"from": "human", "value": "Find the weather in Paris today."},
        {"from": "gpt", "value": "[get_weather()]"},
        {"from": "user", "value": "Now check Berlin too."},
    ]}
    assert extract_user_prompts(row) == [
        "Find the weather in Paris today.", "Now check Berlin too."]


def test_json_array_chat_string_shape():
    row = {"chat": json.dumps([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is the price of BTC?"},
        {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        {"role": "user", "content": "And ETH?"},
    ])}
    assert extract_user_prompts(row) == [
        "What is the price of BTC?", "And ETH?"]


def test_glaive_plain_text_chat_shape():
    row = {"chat": (
        "SYSTEM: You are a helpful assistant with tools.\n"
        "USER: Find me the current stock price of Apple\n"
        "ASSISTANT: <functioncall> get_price(...)\n"
        "USER: What about Microsoft?\n"
        "ASSISTANT: sure\n<|endoftext|>"
    )}
    assert extract_user_prompts(row) == [
        "Find me the current stock price of Apple",
        "What about Microsoft?"]


def test_query_column_shape():
    row = {"query": "Convert 100 USD to EUR please.",
           "tools": "[]", "answers": "[]"}
    assert extract_user_prompts(row) == ["Convert 100 USD to EUR please."]


def test_extract_returns_nothing_on_junk():
    assert extract_user_prompts({"conversations": "not a list"}) == []
    assert extract_user_prompts({}) == []
    assert extract_user_prompts({"row": {"other": 1}}) == []


def test_is_good_prompt_filters():
    assert not is_good_prompt("hi")                       # too short
    assert not is_good_prompt("x" * 600)                  # too long
    assert not is_good_prompt("the sky is blue today")    # not actionable
    assert not is_good_prompt("find me an image please")  # banned keyword
    assert is_good_prompt("Find the weather in Tokyo")
    assert is_good_prompt("Please check my order status")


def test_sources_cover_both_lanes():
    names = {ds for ds, _cfg, _split, _n in SOURCES}
    assert "Team-ACE/ToolACE" in names
    assert "lockon/xlam-function-calling-60k" in names
    assert "glaiveai/glaive-function-calling-v2" in names
    assert "NousResearch/hermes-function-calling-v1" in names


def test_out_is_cwd_relative():
    import harvest_prompts
    from pathlib import Path
    assert harvest_prompts.OUT == Path("data/seeds/external.jsonl")
    assert not harvest_prompts.OUT.is_absolute()
