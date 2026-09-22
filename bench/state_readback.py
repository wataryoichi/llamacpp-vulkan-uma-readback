#!/usr/bin/env python3
"""Time one recurrent-state checkpoint directly, by making the server take one.

Needs llama-server started with --verbose; reads the two create_check log lines.
  python3 state_readback.py /tmp/llama-server.log [url]
"""
import json, re, sys, urllib.request

LOG = sys.argv[1]
URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080"


def post(body):
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["timings"]


before = sum(1 for _ in open(LOG))
base = " ".join(["river"] * 8)
post({"prompt": base, "n_predict": 1, "cache_prompt": True})
tm = post({"prompt": base + " x", "n_predict": 1, "cache_prompt": True})
print(f"prompt_ms for 1 appended token: {tm['prompt_ms']:.0f}")

lines = open(LOG).read().splitlines()[before:]
ts = [(m.group(1), l) for l in lines if (m := re.match(r"(\d+\.\d+\.\d+\.\d+)", l)) and "create_check" in l]
for t, l in ts:
    print("  " + l[:150])
if len(ts) >= 2:
    def secs(s):
        h, m, sec, ms = s.split(".")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000
    print(f"  => checkpoint took {(secs(ts[-1][0]) - secs(ts[-2][0])) * 1000:.0f} ms")
