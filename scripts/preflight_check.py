#!/usr/bin/env python3
"""Pre-launch verification checklist for VP Backend.

Run BEFORE going live after ANY deploy, code change, or cutover.
Catches endpoint mismatches, connectivity issues, and GHL config drift.

Usage:
    python scripts/preflight_check.py

Requires: requests, neo4j (same deps as the app)
"""

import json
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ──────────────────────────────────────────────────────────────────────
# CONFIG — Update these if infrastructure changes
# ──────────────────────────────────────────────────────────────────────
VP_API_URL = "https://vp-api.fastflipai.cloud"
GHL_API_BASE = "https://services.leadconnectorhq.com"
GHL_LOCATION_ID = "ER9V9WFNXLK3NNXnObw9"

# GHL custom value names → expected VP backend endpoint paths
EXPECTED_WEBHOOK_MAP = {
    "AI Text Webhook": "/webhooks/inbound",
    "AI Outreach Webhook": "/webhooks/outreach",
    "AI Call Webhook": "/webhooks/call",
}

# All endpoints that MUST exist on the VP backend
REQUIRED_ENDPOINTS = [
    ("GET", "/health"),
    ("GET", "/ready"),
    ("GET", "/canary/status"),
    ("POST", "/webhooks/inbound"),
    ("POST", "/webhooks/outreach"),
    ("POST", "/webhooks/outbound"),
    ("POST", "/webhooks/call"),
]


def _request(method, url, data=None, headers=None, timeout=10):
    """Simple HTTP request helper."""
    hdrs = headers or {}
    if data:
        hdrs["Content-Type"] = "application/json"
        body = json.dumps(data).encode()
    else:
        body = None
    req = Request(url, data=body, headers=hdrs, method=method)
    try:
        resp = urlopen(req, timeout=timeout)
        return resp.status, json.loads(resp.read().decode())
    except HTTPError as e:
        return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}


# ──────────────────────────────────────────────────────────────────────
# CHECKS
# ──────────────────────────────────────────────────────────────────────
results = []


def check(name, passed, detail=""):
    """Record a check result."""
    status = "PASS" if passed else "FAIL"
    results.append((status, name, detail))
    icon = "+" if passed else "X"
    print(f"  [{icon}] {name}" + (f" — {detail}" if detail else ""))


print("=" * 60)
print("VP Backend Pre-Launch Verification")
print("=" * 60)

# ── 1. Endpoint Existence ──────────────────────────────────────────
print("\n1. ENDPOINT EXISTENCE")
for method, path in REQUIRED_ENDPOINTS:
    url = f"{VP_API_URL}{path}"
    data = {"preflight": True} if method == "POST" else None
    status, body = _request(method, url, data=data)
    check(f"{method} {path}", status == 200, f"HTTP {status}")

# ── 2. Health & Dependencies ───────────────────────────────────────
print("\n2. HEALTH & DEPENDENCIES")
status, body = _request("GET", f"{VP_API_URL}/ready")
if status == 200 and isinstance(body, dict):
    checks = body.get("checks", {})
    check("Redis connected", checks.get("redis") == "ok", str(checks.get("redis")))
    check("Neo4j connected", checks.get("neo4j") == "ok", str(checks.get("neo4j")))
else:
    check("Ready endpoint", False, f"HTTP {status}")

# ── 3. GHL Custom Values Match Endpoints ───────────────────────────
print("\n3. GHL CUSTOM VALUES vs VP ENDPOINTS")
# This needs the GHL PIT token. Try to read from env or pass as arg.
ghl_token = None
try:
    import os
    ghl_token = os.environ.get("GHL_TOKEN")
    if not ghl_token:
        # Try reading from the custom values on the VPS
        ghl_token = "pit-5ae75f37-34a7-4e7f-903a-3f2a3f2575ae"
except Exception:
    pass

if ghl_token:
    cv_url = f"{GHL_API_BASE}/locations/{GHL_LOCATION_ID}/customValues"
    status, body = _request("GET", cv_url, headers={
        "Authorization": f"Bearer {ghl_token}",
        "Version": "2021-07-28",
    })
    if status == 200:
        custom_values = {
            cv["name"]: cv["value"]
            for cv in body.get("customValues", [])
        }
        for cv_name, expected_path in EXPECTED_WEBHOOK_MAP.items():
            cv_value = custom_values.get(cv_name, "")
            expected_url = f"{VP_API_URL}{expected_path}"
            match = cv_value == expected_url
            check(
                f"GHL '{cv_name}' matches endpoint",
                match,
                f"GHL={cv_value} vs Expected={expected_url}",
            )
            if match:
                # Verify the endpoint actually responds
                ep_status, _ = _request("POST", cv_value, data={"preflight": True})
                check(f"  ...and endpoint responds", ep_status == 200, f"HTTP {ep_status}")
    else:
        check("GHL Custom Values API", False, f"HTTP {status} — run from VPS or check token")
else:
    print("  [SKIP] No GHL_TOKEN — set GHL_TOKEN env var or run from VPS")

# ── 4. Pipeline E2E (test payload) ─────────────────────────────────
print("\n4. PIPELINE E2E TEST")
test_payload = {
    "contactId": f"preflight-{int(time.time())}",
    "customData": {
        "Contact Phone": "+5215500000001",
        "Message Body": "Preflight test - ignore",
        "Contact Name": "Preflight Check",
    },
    "direction": "inbound",
    "locationId": GHL_LOCATION_ID,
    "type": "SMS",
}
status, body = _request("POST", f"{VP_API_URL}/webhooks/inbound", data=test_payload)
accepted = status == 200 and body.get("status") == "accepted"
check("Webhook accepts test payload", accepted, body.get("status", "unknown"))
if accepted:
    check("Trace ID returned", bool(body.get("trace_id")), body.get("trace_id", "none"))

# ── 5. Canary Status ──────────────────────────────────────────────
print("\n5. CANARY STATUS")
status, body = _request("GET", f"{VP_API_URL}/canary/status")
if status == 200:
    check("Canary endpoint", True, f"enabled={body.get('canary_enabled')}")
    error_rate = body.get("canary_error_rate", 0)
    check("Error rate < 10%", error_rate < 0.1, f"{error_rate*100:.1f}%")
else:
    check("Canary endpoint", False, f"HTTP {status}")

# ── SUMMARY ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for s, _, _ in results if s == "PASS")
failed = sum(1 for s, _, _ in results if s == "FAIL")
total = len(results)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")

if failed > 0:
    print("\nFAILED CHECKS:")
    for s, name, detail in results:
        if s == "FAIL":
            print(f"  [X] {name} — {detail}")
    print("\n*** DO NOT GO LIVE — fix failures first ***")
    sys.exit(1)
else:
    print("\n*** ALL CHECKS PASSED — safe to go live ***")
    sys.exit(0)
