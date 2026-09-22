#!/usr/bin/env python3
"""Measure what a multi-turn chat pays to reuse the conversation prefix.

  python3 chat_turns.py [url]
"""
import json, random, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
WORDS = "the model context window grows each turn so prefix reuse matters here".split()


def post(body):
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


random.seed(3)
conv = "System: you are helpful.\n" + " ".join(random.choice(WORDS) for _ in range(1800))
for turn in range(1, 4):
    conv += f"\nUser: question number {turn}, please answer.\nAssistant:"
    t = time.time()
    out = post({"prompt": conv, "n_predict": 8, "temperature": 0, "cache_prompt": True})
    tm = out["timings"]
    print(f"  turn {turn}: wall={(time.time() - t) * 1000:7.0f}ms  "
          f"prompt_n={tm['prompt_n']:5d}  prompt_ms={tm['prompt_ms']:7.0f}")
    conv += " ok. " + " ".join(random.choice(WORDS) for _ in range(120))
