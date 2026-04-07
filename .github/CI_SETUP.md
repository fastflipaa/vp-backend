# CI / Deploy Gate Setup

This file documents the **one-time setup** for the GitHub Actions deploy
gate (`.github/workflows/test-and-deploy.yml`).

## Why this exists

On 2026-04-05 a state machine fix was committed to `main` and deployed
to production without running the existing test suite. The fix
introduced `CLOSED.to(RE_ENGAGE)` -- an invalid transition from a
final state -- which caused python-statemachine to raise
`InvalidDefinition` on every `process_message` task. The system was
100% broken for 46 hours. The existing test
`tests/state_machine/test_conversation_sm.py::test_restoration_to_final_states[CLOSED]`
would have caught the bug in 0.21 seconds. Nobody ran it.

The CI gate makes it impossible to deploy a commit that breaks
existing tests. No human discipline required.

## Required GitHub repo secrets (one-time)

The deploy job SSHes into the production VPS to call Coolify's deploy
helper. It needs the same SSH key that the Python agents use locally.

### `VPS_SSH_KEY`

The full contents of the private key at `~/.ssh/vps_key` (an RSA key
already used by `paramiko.RSAKey.from_private_key_file()` throughout
the operator's tooling).

#### How to add it

1. Open the key on the operator's local machine:
   ```bash
   cat ~/.ssh/vps_key
   ```
2. Copy the entire output, including the
   `-----BEGIN OPENSSH PRIVATE KEY-----` and
   `-----END OPENSSH PRIVATE KEY-----` lines.
3. In GitHub: **Settings -> Secrets and variables -> Actions ->
   New repository secret**.
4. Name: `VPS_SSH_KEY`
5. Value: paste the key contents.
6. Click **Add secret**.

That's it. No other secrets are needed.

## What runs on each push

1. **`test` job** -- installs dependencies, runs `pytest tests/ -m "not slow"`,
   then runs an explicit one-liner that instantiates `ConversationSM` for
   every state value (catches python-statemachine validation regressions).
2. **`integration-tests` job** -- runs `pytest tests/ -m slow` which includes
   testcontainers-backed integration tests (real Neo4j / Redis spun up for
   the test run).
3. **`deploy` job** -- only runs if BOTH test jobs passed AND the push is
   to `main` (not a PR). SSHes to the VPS and calls
   `/usr/local/bin/coolify-deploy <uuid>` for vp-api and vp-worker. The
   helper uses Coolify's `queue_application_deployment()` to enqueue a
   build with the new commit.

## What's on the VPS

- `/usr/local/bin/coolify-deploy` -- bash wrapper script
- `/var/www/html/_coolify_deploy.php` (inside the `coolify` container) --
  PHP helper that calls `queue_application_deployment()` from Coolify's
  Laravel helpers

Both were created on 2026-04-07 during the outage recovery.

## Verifying the gate works

After adding the secret, push any small change to `main` and watch
**Actions** tab in GitHub. You should see:
1. `test` (~2-3 min) -- green
2. `integration-tests` (~5-8 min) -- green
3. `deploy` (~10 sec) -- green

If you push a commit that breaks a test, `test` goes red and `deploy`
never runs. Production stays on the previous commit.

## Bypassing the gate (emergency only)

If for some reason you need to deploy without CI (e.g. CI itself is
broken, GitHub is down), you can SSH to the VPS and run the deploy
helper directly:

```bash
ssh root@72.62.64.164 /usr/local/bin/coolify-deploy qok8cs0g4wgooks400cwoo4o
ssh root@72.62.64.164 /usr/local/bin/coolify-deploy xk4o0g40o8cc848c40gg48ck
```

This is the same path the GitHub Action takes -- it just bypasses the
test gate. Use sparingly. Add a Slack post-mortem describing why CI
was bypassed.
