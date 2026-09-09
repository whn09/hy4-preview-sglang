"""Greedy completions from a live sglang server, for arm-to-arm comparison."""

import json
import os

import requests

PROMPTS = [
    "The capital of France is",
    "Write one sentence about the sea:",
    "2 + 2 =",
    "List three prime numbers:",
    "def add(a, b):",
]


def main():
    port = os.environ.get("PORT", "30000")
    tag = os.environ["TAG"]
    out = os.environ["OUT"]
    completions = []
    for p in PROMPTS:
        r = requests.post(
            f"http://localhost:{port}/generate",
            json={
                "text": p,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 32,
                },
            },
            timeout=120,
        )
        r.raise_for_status()
        completions.append({"prompt": p, "text": r.json()["text"]})
    json.dump(
        {"tag": tag, "outcome": "served", "completions": completions},
        open(out, "w"),
        indent=2,
    )
    print(f"{tag}: served {len(completions)} prompts")


if __name__ == "__main__":
    main()
