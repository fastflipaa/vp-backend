# Cutover Runbook: n8n -> FastAPI

> **Target:** Vive Polanco lead processing pipeline
> **From:** n8n Main Router (89-node workflow, currently NON-FUNCTIONAL)
> **To:** FastAPI vp-api (Python + Celery + Redis + Neo4j)
> **Estimated time:** Phase 1-3 in 1 day, Phase 4 over 1 week, Phase 5 after burn-in

## Prerequisites

- [ ] pytest suite passes at 80%+ coverage (`pytest --cov=app --cov-fail-under=80`)
- [ ] Phase 17 canary validation completed successfully
- [ ] All containers healthy: vp-api, vp-worker, vp-beat, vp-redis, Neo4j
- [ ] Simplified n8n forwarder workflow active (Plan 18-03 Task 3)
- [ ] Old 89-node Main Router deactivated

## Workflow IDs (from Plan 18-03 Task 3)

| Workflow | ID | Status |
|----------|----|--------|
| Old Main Router | `M4Q1cbWwc9QZfLJM` | INACTIVE |
| Simplified Forwarder | `bJjQCpCnBe2EtDrt` | ACTIVE |
| FastAPI Webhook | `POST /webhooks/inbound` | RUNNING |

---

## Phase 1: Enable Full Canary (All Contacts)

**Duration:** 10 minutes + 24h monitoring

1. Set `CANARY_ENABLED=true` in Coolify env vars for vp-api
2. Redeploy vp-api (Coolify auto-redeploys on env change)
3. Verify canary is enabled:
   ```bash
   curl -s https://vp-api.fastflipai.cloud/canary/status | python -m json.tool
   ```
   Expected: `{"enabled": true, ...}`
4. n8n forwarder continues to receive GHL webhooks and forward ALL contacts to FastAPI
5. **Monitor for 24 hours minimum** -- check structured logs for errors:
   ```bash
   docker logs vp-api --tail 100 --since 1h 2>&1 | grep -i error
   ```

**Go/No-Go:** Error rate < 1%, no dropped messages, response latency < 15s average.

---

## Phase 2: Verify n8n Forwarder

**Duration:** 15 minutes

1. The simplified n8n forwarder workflow was already created and activated by Plan 18-03 Task 3
2. Verify the old 89-node Main Router is deactivated:
   ```bash
   source "MCP_ENV" && curl -s -H "X-N8N-API-KEY: $N8N_API_KEY" \
     "https://n8n.fastflipai.cloud/api/v1/workflows/M4Q1cbWwc9QZfLJM" | python -c \
     "import sys,json; w=json.load(sys.stdin); print(f'Main Router: active={w[\"active\"]}')"
   ```
3. Verify the new forwarder is active:
   ```bash
   source "MCP_ENV" && curl -s -H "X-N8N-API-KEY: $N8N_API_KEY" \
     "https://n8n.fastflipai.cloud/api/v1/workflows/bJjQCpCnBe2EtDrt" | python -c \
     "import sys,json; w=json.load(sys.stdin); print(f'Forwarder: active={w[\"active\"]}')"
   ```
4. Test: Send test message via GHL -> appears in FastAPI structured logs with trace_id

---

## Phase 3: GHL Webhook Switch (USER ACTION REQUIRED)

**Duration:** 5 minutes

> This step requires manual action in the GHL Dashboard -- no API available for webhook management.

1. In **GHL Dashboard -> Settings -> Webhooks** (or Workflow trigger):
   - **Current URL:** n8n webhook URL (the forwarder)
   - **New URL:** `https://vp-api.fastflipai.cloud/webhooks/inbound`
2. After switch, verify with a test message:
   - Send a message to the Vive Polanco WhatsApp number
   - Confirm trace_id appears in FastAPI logs (NOT in n8n execution log)
   ```bash
   docker logs vp-api --tail 20 2>&1 | grep trace_id
   ```
3. Confirm n8n is no longer receiving webhooks (execution log should stop updating)

---

## Phase 4: Shadow Burn-in (1 Week)

**Duration:** 7 days

1. Keep the simplified n8n forwarder workflow available but inactive (for rollback)
2. Monitor daily:

| Metric | Target | Check Command |
|--------|--------|---------------|
| Error rate | < 1% | `docker logs vp-api --since 24h 2>&1 \| grep -c error` |
| Response latency | < 15s avg | Check `pipeline_complete` log entries for `duration_ms` |
| Gate block rate | Stable | Check `gate_decision` log entries |
| Fallback rate | < 5% | `docker logs vp-api --since 24h 2>&1 \| grep -c fallback_sent` |
| Dropped messages | 0 | Compare GHL inbound count vs FastAPI processed count |

3. After 1 week with no issues, proceed to Phase 5

---

## Phase 5: Deactivate Shadow

**Duration:** 15 minutes

1. Deactivate the simplified n8n forwarder workflow (keep for emergency rollback):
   ```bash
   source "MCP_ENV" && curl -s -X POST -H "X-N8N-API-KEY: $N8N_API_KEY" \
     "https://n8n.fastflipai.cloud/api/v1/workflows/bJjQCpCnBe2EtDrt/deactivate"
   ```
2. Optional cleanup (can be done later):
   - Remove `_run_shadow_comparison()` call from `gate_tasks.process_gates_shadow()`
   - Remove `n8n_gate_decisions` from webhook payload schemas
3. Archive :ShadowComparison nodes in Neo4j (let them age out naturally, no TTL needed)

---

## Final Verification Checklist

- [ ] All messages have trace_ids in structured logs
- [ ] Error rate < 1% over 24 hours
- [ ] Average response latency < 15 seconds
- [ ] No dropped messages (GHL inbound count == FastAPI processed count)
- [ ] Gate agreement rates stable
- [ ] Fallback rate < 5%
- [ ] n8n Main Router preserved (inactive) for emergency rollback
- [ ] Prompt versioning active (Interaction nodes carry prompt_version)

---

## A/B Comparison Queries (Neo4j)

These Cypher queries enable prompt versioning analysis from day one. Run in Neo4j Browser at `http://10.0.2.8:7474`.

### Response Rate by Prompt Version

```cypher
MATCH (i:Interaction)
WHERE i.prompt_version IS NOT NULL
WITH i.prompt_version AS version,
     count(*) AS total,
     count(CASE WHEN i.response IS NOT NULL THEN 1 END) AS responded
RETURN version, total, responded,
       round(toFloat(responded) / total * 100, 1) AS response_rate_pct
ORDER BY version
```

### Average Conversation Length by Prompt Version

```cypher
MATCH (l:Lead)-[:HAS_INTERACTION]->(i:Interaction)
WHERE i.prompt_version IS NOT NULL
WITH l.contact_id AS contact, i.prompt_version AS version, count(i) AS msg_count
RETURN version, avg(msg_count) AS avg_conversation_length, count(contact) AS leads
ORDER BY version
```

### Conversion Rate by Prompt Version

```cypher
// Leads that reached HANDOFF, SCHEDULING, or QUALIFIED states
MATCH (l:Lead)-[:HAS_INTERACTION]->(i:Interaction)
WHERE i.prompt_version IS NOT NULL AND l.state IN ['HANDOFF', 'SCHEDULING', 'QUALIFIED']
WITH i.prompt_version AS version, count(DISTINCT l) AS converted
MATCH (i2:Interaction) WHERE i2.prompt_version = version
WITH version, converted, count(DISTINCT i2) AS total_interactions
RETURN version, converted, total_interactions,
       round(toFloat(converted) / total_interactions * 100, 1) AS conversion_pct
```

### Prompt Version Distribution Over Time

```cypher
MATCH (i:Interaction)
WHERE i.prompt_version IS NOT NULL AND i.created_at IS NOT NULL
WITH i.prompt_version AS version,
     date(i.created_at) AS day,
     count(*) AS count
RETURN version, day, count
ORDER BY day DESC, version
LIMIT 100
```
