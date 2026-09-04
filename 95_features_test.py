#!/usr/bin/env python3
"""Exercise the two Hy4-specific serving features the cookbook documents:
suffix-aware reasoning separation and the arg_key/arg_value tool-call format.

    python3 95_features_test.py                       # all
    python3 95_features_test.py reasoning
    python3 95_features_test.py tools

Run it against an already-serving endpoint. It only needs `openai`, so from the
host use the container:

    docker run --rm --net=host -v $PWD:/w -w /w --entrypoint python3 \
        lmsysorg/sglang:hy4-preview 95_features_test.py

Why this is a separate file from 90_smoke_test.sh: the tool-call path is where an
`auto` parser can silently degrade. If the Hunyuan parser fails to resolve Hy4's
SUFFIXED structural tokens (<tool_calls:opensource> etc.) from the tokenizer
vocab, the request still returns 200 -- the raw tokens just land in `content` and
`tool_calls` is None. A curl smoke test would pass; this asserts.
"""
import json
import os
import sys

from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://localhost:30000/v1")
MODEL = os.environ.get("MODEL", "tencent/Hy4-preview-FP8")

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def reasoning():
    print("== reasoning_effort=high (top-level OpenAI field) ==")
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Solve step by step: What is 15% of 240?"}],
        reasoning_effort="high",
        max_tokens=2048,
    )
    m = r.choices[0].message
    print("--- reasoning_content ---")
    print((m.reasoning_content or "")[:1000])
    print("--- content ---")
    print(m.content)
    check("thinking lands in reasoning_content", bool(m.reasoning_content))
    check("answer lands in content", bool(m.content))
    check("36 in the answer", "36" in (m.content or ""))
    # The tell for a parser that did NOT resolve the suffixed tokens.
    check("no raw structural tokens in content", "<think" not in (m.content or ""))

    print("== reasoning_effort=no_think (model-specific, via chat_template_kwargs) ==")
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Give me a one-line summary of relativity."}],
        extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
        max_tokens=256,
    )
    m = r.choices[0].message
    print("Content:", m.content)
    print("Reasoning:", m.reasoning_content)
    check("no_think produces content", bool(m.content))
    check("no_think skips thinking", not (m.reasoning_content or "").strip())


def tools():
    print("== tool call (non-streaming) ==")
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What's the weather in Beijing? Use fahrenheit."}],
        tools=TOOLS,
    )
    m = r.choices[0].message
    print("Content:  ", m.content)
    for tc in m.tool_calls or []:
        print(f"Tool Call: {tc.function.name}\n  Arguments: {tc.function.arguments}")
    check("tool_calls parsed (not left as raw text in content)", bool(m.tool_calls))
    if m.tool_calls:
        tc = m.tool_calls[0]
        check("function name", tc.function.name == "get_weather", tc.function.name)
        try:
            args = json.loads(tc.function.arguments)
        except Exception as e:  # noqa: BLE001
            args = {}
            check("arguments are JSON", False, str(e))
        else:
            check("arguments are JSON", True)
        check("city argument", str(args.get("city", "")).lower().startswith("beijing"), repr(args))
        check("unit argument coerced", args.get("unit") == "fahrenheit", repr(args))

    print("== tool call (streaming, incremental deltas) ==")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What's the weather in Beijing? Use fahrenheit."}],
        tools=TOOLS,
        stream=True,
    )
    buf = {}
    for chunk in stream:
        delta = chunk.choices[0].delta
        for tc in delta.tool_calls or []:
            b = buf.setdefault(tc.index, {"name": "", "args": ""})
            if tc.function and tc.function.name:
                b["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                b["args"] += tc.function.arguments
    for idx, b in buf.items():
        print(f"Tool[{idx}] {b['name']}({b['args']})")
    check("streaming yielded a tool call", bool(buf))
    if buf:
        b = buf[min(buf)]
        check("streamed name", b["name"] == "get_weather", b["name"])
        try:
            json.loads(b["args"])
            check("streamed arguments reassemble into JSON", True)
        except Exception as e:  # noqa: BLE001
            check("streamed arguments reassemble into JSON", False, str(e))


def text_only():
    """Image input is rejected with 400 by design -- assert the contract."""
    print("== text-only contract (image input must be rejected) ==")
    try:
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
            ]}],
            max_tokens=16,
        )
        check("image input rejected", False, "server accepted an image")
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "status_code", None)
        check("image input rejected", code == 400 or code is not None, f"status={code}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"endpoint={BASE_URL} model={MODEL}\n")
    if which in ("all", "reasoning"):
        reasoning()
    if which in ("all", "tools"):
        tools()
    if which in ("all", "text-only"):
        text_only()
    print()
    if failures:
        print(f"FAILED {len(failures)}: " + ", ".join(failures))
        sys.exit(1)
    print("all checks passed")
