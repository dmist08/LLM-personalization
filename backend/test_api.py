"""Quick test of the full backend generate flow."""
import requests
import json
import time

BASE = "http://localhost:5000/api"

print("=== Backend API Test ===\n")

# 1. Health
r = requests.get(f"{BASE}/health")
print(f"1. Health: {r.status_code} {r.json()}")

# 2. Authors
for pub in ["TOI", "HT"]:
    r = requests.get(f"{BASE}/authors", params={"publication": pub})
    d = r.json()
    names = [a["name"] for a in d["authors"][:3]]
    print(f"2. Authors ({pub}): {d['count']} total — {names}")

# 3. Generate
print("\n3. Generate (this may take ~20s if Modal GPU is warm, ~3min if cold)...")
t0 = time.time()
r = requests.post(f"{BASE}/generate", json={
    "source_text": "Scientists at MIT have developed a new battery technology that can charge electric vehicles in under five minutes, potentially revolutionizing the EV industry.",
    "author_id": "alok_chamaria",
    "publication": "TOI",
}, timeout=300)
elapsed = time.time() - t0

print(f"   Status: {r.status_code} | Time: {elapsed:.1f}s")
d = r.json()
print(f"   Session: {d.get('session_id', 'NONE')[:30]}")
print(f"   Errors: {d.get('errors', [])}")

for method, result in d.get("results", {}).items():
    headline = result.get("headline", "")[:80]
    latency = result.get("latency_ms", 0)
    error = result.get("error", "")
    if error:
        print(f"   {method:22s} | ERROR: {error}")
    else:
        print(f"   {method:22s} | {latency:5d}ms | {headline}")

print("\n=== Done ===")
