# Rollback Runbook: FastAPI -> n8n

> **Total time:** ~5 minutes
> **When:** Only if FastAPI pipeline is critically failing AND doing nothing is worse

## IMPORTANT: Current n8n System is NON-FUNCTIONAL

The 89-node n8n Main Router has been non-functional since before Phase 14. Rolling back means going from a working system (FastAPI) to a broken one (n8n). **Only rollback if FastAPI is WORSE than doing nothing.**

Before rolling back, consider:
1. Is the issue transient? (Redis restart, Neo4j blip) -- wait 5 minutes first
2. Is it a single processor bug? -- The circuit breaker will auto-fallback
3. Is it a GHL delivery issue? -- That affects n8n too, rollback won't help

## When to Rollback

- Error rate > 10% sustained for 30+ minutes
- Complete pipeline failure (zero messages being processed)
- Critical bug in safety gates (messages going to wrong contacts, PII leak)
- Celery workers completely unresponsive (no heartbeat for 5+ minutes)

---

## Step 1: Reactivate n8n Main Router (1 minute)

```bash
# Source n8n API key
source "C:/Users/conta/Desktop/MCP Servers/n8n-mcp-agent/.env"

# Activate the 89-node Main Router
curl -s -X POST -H "X-N8N-API-KEY: $N8N_API_KEY" \
  "https://n8n.fastflipai.cloud/api/v1/workflows/{MAIN_ROUTER_ID}/activate"

# Deactivate the simplified forwarder
curl -s -X POST -H "X-N8N-API-KEY: $N8N_API_KEY" \
  "https://n8n.fastflipai.cloud/api/v1/workflows/{FORWARDER_ID}/deactivate"
```

Verify: Check n8n UI -- Main Router shows as active.

---

## Step 2: Update GHL Webhook (2 minutes)

> USER ACTION REQUIRED -- No API available for GHL webhook management.

1. Open **GHL Dashboard -> Settings -> Webhooks** (or the Workflow trigger)
2. Change the webhook URL:
   - **From:** `https://vp-api.fastflipai.cloud/webhooks/inbound`
   - **To:** n8n webhook URL (the Main Router's webhook path)
3. Save

---

## Step 3: Disable FastAPI Canary (1 minute)

1. In Coolify dashboard, update vp-api environment:
   - Set `CANARY_ENABLED=false`
2. Redeploy vp-api (Coolify auto-redeploys on env change)
3. Wait for container to be healthy:
   ```bash
   curl -s https://vp-api.fastflipai.cloud/health | python -m json.tool
   ```

---

## Step 4: Verify Rollback (1 minute)

1. Send a test message via GHL (or WhatsApp to test number)
2. Confirm it appears in **n8n execution log** (check n8n UI)
3. Confirm vp-api is NOT processing messages:
   ```bash
   docker logs vp-api --tail 5 --since 1m 2>&1 | grep process_message
   # Should show nothing new
   ```

---

## Post-Rollback Actions

1. **Investigate** the issue that caused the rollback
   - Check FastAPI structured logs: `docker logs vp-api --tail 500 2>&1`
   - Check Celery worker logs: `docker logs vp-worker --tail 500 2>&1`
   - Check Redis: `redis-cli -h vp-redis ping`
   - Check Neo4j: `curl http://10.0.2.8:7474`
2. **Fix** the issue in the FastAPI codebase
3. **Re-run** the test suite: `pytest tests/ -v --cov=app --cov-fail-under=80`
4. **Re-deploy** via Coolify
5. **Re-attempt cutover** following CUTOVER_RUNBOOK.md from Phase 1

---

## Emergency Contacts

| Role | Who | Contact |
|------|-----|---------|
| System Owner | _(fill in)_ | _(fill in)_ |
| n8n Admin | _(fill in)_ | _(fill in)_ |
| GHL Admin | _(fill in)_ | _(fill in)_ |

---

## Workflow IDs Reference

| Workflow | ID | Notes |
|----------|----|-------|
| Old Main Router | _(fill after Task 3)_ | 89 nodes, original n8n pipeline |
| Simplified Forwarder | _(fill after Task 3)_ | 5-6 nodes, webhook -> FastAPI forward |
| FastAPI Webhook | N/A | `POST /webhooks/inbound` on vp-api |
