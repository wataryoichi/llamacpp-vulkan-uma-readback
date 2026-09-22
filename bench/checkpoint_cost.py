#!/usr/bin/env python3
"""Show the readback cliff: prompt time vs new-token count, and vs cached-prefix length.

Start llama-server with a hybrid/recurrent model first, e.g.
  llama-server -m model.gguf -c 8192 -ngl 999 --flash-attn on --verbose
Then:
  python3 checkpoint_cost.py [url]
"""
import json, sys, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"


def post(body):
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["timings"]


def run(prompt, cache):
    for _ in range(3):  # third call is the steady state
        tm = post({"prompt": prompt, "n_predict": 1, "n_probs": 0,
                   "temperature": 0, "cache_prompt": cache})
    return tm


print("A) fresh prompt, n new tokens")
for n in [1, 2, 3, 4, 5, 6, 8, 16, 32]:
    tm = run(" ".join(["stone"] * n), False)
    print(f"  prompt_n={tm['prompt_n']:4d}  prompt_ms={tm['prompt_ms']:7.0f}")

print("B) cached prefix of P tokens, then 1 new token")
for p in [1, 2, 4, 8, 16, 64, 256]:
    base = " ".join(["river"] * p)
    post({"prompt": base, "n_predict": 1, "cache_prompt": True})
    tm = run(base + " x", True)
    print(f"  prefix={p:4d}  prompt_n={tm['prompt_n']:4d}  prompt_ms={tm['prompt_ms']:7.0f}")
