"""
load_test.py — Concurrent stress test for the StyleVector API.
===============================================================
Simulates N concurrent users hitting the /api/generate endpoint.
Measures latency, throughput, error rate, and Modal cold-start behavior.

Usage:
    pip install aiohttp  (if not installed)
    python backend/load_test.py --users 5 --rounds 2
    python backend/load_test.py --users 10 --rounds 1 --url https://dkmist08--stylevector-backend-flask-app.modal.run/api

Output:
    Prints a results table + saves to backend/load_test_results.json
"""

import argparse
import asyncio
import json
import time
import statistics
from datetime import datetime

try:
    import aiohttp
except ImportError:
    print("Install aiohttp: pip install aiohttp")
    exit(1)


# ── TEST PAYLOADS (realistic sparse/mid author requests) ─────────────────────
TEST_PAYLOADS = [
    {
        "source_text": "India and Bangladesh on Monday discussed key bilateral issues including border management, trade facilitation, and connectivity projects during a meeting between External Affairs Minister S Jaishankar and his Bangladeshi counterpart. The two sides agreed to expedite pending infrastructure projects and enhance cooperation in counter-terrorism.",
        "publication": "HT",
        "author_id": "nisheeth_upadhyay",
        "author_name": "Nisheeth Upadhyay",
    },
    {
        "source_text": "The Supreme Court on Tuesday dismissed a plea challenging the constitutional validity of the Goods and Services Tax compensation cess, saying the levy was within the legislative competence of Parliament. A bench headed by Chief Justice said the cess was a measure to compensate states for revenue loss.",
        "publication": "HT",
        "author_id": "rezaul_h_laskar",
        "author_name": "Rezaul H Laskar",
    },
    {
        "source_text": "The Sensex surged over 500 points on Wednesday tracking strong global cues after the US Federal Reserve signaled a pause in interest rate hikes. Banking and IT stocks led the rally with HDFC Bank, Infosys and TCS among the top gainers. The Nifty also crossed the 24000 mark for the first time.",
        "publication": "HT",
        "author_id": "shamik_banerjee",
        "author_name": "Shamik Banerjee",
    },
    {
        "source_text": "A massive fire broke out at a chemical factory in the MIDC industrial area of Pune on Thursday, injuring at least seven workers. Fire tenders from three stations rushed to the spot and brought the blaze under control after a four-hour operation. The cause of the fire is being investigated.",
        "publication": "HT",
        "author_id": "shivya_kanojia",
        "author_name": "Shivya Kanojia",
    },
    {
        "source_text": "The Delhi government on Friday announced a new electric vehicle policy aimed at making 25 percent of all new vehicle registrations electric by 2026. The policy includes purchase incentives, charging infrastructure development, and scrapping incentives for old petrol and diesel vehicles.",
        "publication": "TOI",
        "author_id": "trisha_mahajan",
        "author_name": "Trisha Mahajan",
    },
]


async def send_request(session, url, payload, user_id, request_id):
    """Send a single generate request and measure response."""
    start = time.perf_counter()
    result = {
        "user_id": user_id,
        "request_id": request_id,
        "author": payload["author_name"],
        "status": None,
        "latency_s": None,
        "error": None,
        "methods_returned": 0,
    }

    try:
        async with session.post(
            f"{url}/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),  # 3 min timeout (Modal cold start)
        ) as resp:
            elapsed = time.perf_counter() - start
            result["latency_s"] = round(elapsed, 2)
            result["status"] = resp.status

            if resp.status == 200:
                data = await resp.json()
                results = data.get("results", {})
                result["methods_returned"] = len(results)
                # Check each method
                for method, r in results.items():
                    if r and r.get("headline"):
                        result[f"{method}_ok"] = True
                    else:
                        result[f"{method}_ok"] = False
            else:
                body = await resp.text()
                result["error"] = body[:200]

    except asyncio.TimeoutError:
        result["latency_s"] = round(time.perf_counter() - start, 2)
        result["status"] = "TIMEOUT"
        result["error"] = "Request timed out after 180s"
    except Exception as e:
        result["latency_s"] = round(time.perf_counter() - start, 2)
        result["status"] = "ERROR"
        result["error"] = str(e)[:200]

    status_icon = "✓" if result["status"] == 200 else "✗"
    print(f"  [{status_icon}] User {user_id} | Req {request_id} | {result['author']:<20} | "
          f"{result['latency_s']:>6.1f}s | Status: {result['status']} | Methods: {result['methods_returned']}")

    return result


async def run_round(url, num_users, round_num, payloads):
    """Fire num_users concurrent requests."""
    print(f"\n{'─'*70}")
    print(f"  ROUND {round_num} — {num_users} concurrent requests")
    print(f"{'─'*70}")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_users):
            payload = payloads[i % len(payloads)]
            tasks.append(send_request(session, url, payload, i + 1, round_num))

        results = await asyncio.gather(*tasks)

    return results


def print_summary(all_results):
    """Print aggregate statistics."""
    print(f"\n{'='*70}")
    print(f"  LOAD TEST RESULTS SUMMARY")
    print(f"{'='*70}")

    total = len(all_results)
    successes = [r for r in all_results if r["status"] == 200]
    failures = [r for r in all_results if r["status"] != 200]
    latencies = [r["latency_s"] for r in all_results if r["latency_s"] is not None]

    print(f"\n  Total requests:    {total}")
    print(f"  Successful (200):  {len(successes)} ({len(successes)/total*100:.0f}%)")
    print(f"  Failed:            {len(failures)} ({len(failures)/total*100:.0f}%)")

    if latencies:
        print(f"\n  ── Latency ──")
        print(f"  Min:               {min(latencies):.1f}s")
        print(f"  Max:               {max(latencies):.1f}s")
        print(f"  Mean:              {statistics.mean(latencies):.1f}s")
        print(f"  Median:            {statistics.median(latencies):.1f}s")
        if len(latencies) > 1:
            print(f"  Stdev:             {statistics.stdev(latencies):.1f}s")
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else latencies[0]
        print(f"  P95:               {p95:.1f}s")

    if successes:
        methods_ok = {}
        for r in successes:
            for key in ["no_personalization_ok", "rag_bm25_ok", "stylevector_ok", "cold_start_sv_ok"]:
                if key in r:
                    methods_ok.setdefault(key, {"pass": 0, "fail": 0})
                    if r[key]:
                        methods_ok[key]["pass"] += 1
                    else:
                        methods_ok[key]["fail"] += 1

        if methods_ok:
            print(f"\n  ── Method Health ──")
            for method, counts in methods_ok.items():
                name = method.replace("_ok", "").upper()
                total_m = counts["pass"] + counts["fail"]
                pct = counts["pass"] / total_m * 100 if total_m > 0 else 0
                print(f"  {name:<25} {counts['pass']}/{total_m} ({pct:.0f}%)")

    if failures:
        print(f"\n  ── Failures ──")
        for r in failures:
            print(f"  User {r['user_id']} | {r['status']} | {r.get('error', 'unknown')[:80]}")

    # Throughput
    if latencies:
        total_time = max(latencies)  # wall-clock time for concurrent batch
        if total_time > 0:
            throughput = len(successes) / total_time
            print(f"\n  ── Throughput ──")
            print(f"  Requests/sec:      {throughput:.2f}")
            print(f"  Wall-clock time:   {total_time:.1f}s (for {total} concurrent requests)")

    print(f"\n{'='*70}")


async def main():
    parser = argparse.ArgumentParser(description="Load test StyleVector API")
    parser.add_argument("--url", default="https://dkmist08--stylevector-backend-flask-app.modal.run/api",
                        help="API base URL")
    parser.add_argument("--users", type=int, default=5, help="Number of concurrent users per round")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--warmup", action="store_true", help="Send 1 warmup request first (wake Modal)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  STYLEVECTOR API LOAD TEST")
    print(f"  URL:    {args.url}")
    print(f"  Users:  {args.users} concurrent")
    print(f"  Rounds: {args.rounds}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Warmup — wake up Modal container
    if args.warmup:
        print("\n  ⏳ Warmup: sending 1 request to wake Modal container...")
        async with aiohttp.ClientSession() as session:
            warmup_result = await send_request(session, args.url, TEST_PAYLOADS[0], 0, 0)
            if warmup_result["status"] == 200:
                print(f"  ✓ Warmup complete in {warmup_result['latency_s']}s")
            else:
                print(f"  ✗ Warmup failed: {warmup_result.get('error', '')}")
        await asyncio.sleep(2)  # let container stabilize

    all_results = []
    for round_num in range(1, args.rounds + 1):
        results = await run_round(args.url, args.users, round_num, TEST_PAYLOADS)
        all_results.extend(results)
        if round_num < args.rounds:
            print("  Cooling down 3s...")
            await asyncio.sleep(3)

    print_summary(all_results)

    # Save raw results
    out_path = "backend/load_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {"url": args.url, "users": args.users, "rounds": args.rounds},
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Raw results saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
