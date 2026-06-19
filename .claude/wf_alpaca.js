export const meta = {
  name: 'tsmom-alpaca-wiring',
  description: 'Harden the Alpaca paper-trading submit path of the TSMOM robot for safe unattended monthly running, + home setup docs, with adversarial verification (mocked Alpaca, no network)',
  phases: [
    { title: 'Build Alpaca submit hardening + docs' },
    { title: 'Adversarial verify' },
    { title: 'Integration check' },
  ],
}

const ROOT = '/Users/pierre.renaudot/test/trend_robot'
const PY = '/Users/pierre.renaudot/test/.venv/bin/python'
const PYTEST = '/Users/pierre.renaudot/test/.venv/bin/pytest'

const CONTEXT = `
=== TSMOM ROBOT — ALPACA PAPER SUBMIT HARDENING (milestone 2 of paper trading) ===
The robot at ${ROOT} already has: the live/ package (broker.py with a FUNCTIONAL AlpacaBroker — get_account/get_positions/submit_order map alpaca-py paper objects), run_live.py (dry-run default; --no-dry-run --broker alpaca ALREADY loops intents and calls broker.submit_order), the executor (plan_orders/summarize_plan), state.py, and a pre-registration harness (validation/preregistration.py with verify_config_matches + a frozen decision_record.json: direction=long_only, rebalance=monthly).

THE GOAL: make the LIVE submit path SAFE for UNATTENDED MONTHLY running on the user's PERSONAL Mac, and document home setup. The raw submit works but is unsafe for a scheduled job: re-running re-trades, it ignores the rebalance cadence, it doesn't check the market is open, it doesn't verify the running config still matches the pre-registered decision, and it discards the broker's order results.

IMPORTANT TEST CONSTRAINT: this environment has NO Alpaca credentials and must NOT hit the broker network. ALL tests use a MOCK Alpaca client (injected) — never a real TradingClient. The user will do the real connectivity test at home.

PYTHON (venv, alpaca-py 0.43.4 installed): ${PY}
PYTEST: ${PYTEST}
Work from ${ROOT}.

READ FIRST (do not rewrite working code — EXTEND it):
- trend_robot/live/broker.py (AlpacaBroker, LocalPaperBroker, Broker Protocol, OrderResult, Position, AccountSnapshot).
- run_live.py (main(argv): arg parsing, _build_broker, the plan/submit/preview/state flow).
- trend_robot/live/state.py (save_run_state, load_last_state, has_run_for).
- trend_robot/live/executor.py (plan_orders, summarize_plan).
- trend_robot/validation/preregistration.py (load_decision, verify_config_matches) + decision_record.json.
- trend_robot/backtest/engine.py (_rebalance_mask — the cadence logic to mirror for "is this a rebalance period?").
=== END CONTEXT ===
`

const BUILD_SPEC = `
Build/extend the following. Typed, docstringed. Dry-run stays the safe default. No hard-coded market values.

1) trend_robot/live/broker.py — extend AlpacaBroker (do NOT break existing behaviour):
   - Add an optional injected client for testing: __init__(..., client=None). When client is provided, use it directly and SKIP the env-credential check and the real TradingClient construction (so tests inject a mock with no network/creds). When client is None, keep the current env-key + paper TradingClient behaviour unchanged.
   - Add is_market_open() -> bool: query the Alpaca clock (self._client.get_clock().is_open). Tolerate clients without get_clock by returning True with a logged note (so the local path/tests don't break).
   - Add recent_orders(limit=20) -> list[OrderResult] (best-effort; map the client's recent orders) for audit. If unsupported, return [].
   - Keep get_account/get_positions/submit_order. submit_order already returns OrderResult — keep it.

2) trend_robot/live/scheduling.py (new) — cadence + idempotence helpers (pure):
   - period_key(asof: str, cadence: str) -> str: a stable key identifying the rebalance period for the cadence — 'YYYY-MM-DD' for daily, ISO 'YYYY-Www' for weekly, 'YYYY-MM' for monthly. Raise on unknown cadence.
   - These let the runner act AT MOST ONCE per period (idempotence + cadence gating in one): if the saved state already has a successful live submission for the current period_key, skip.

3) run_live.py — harden the LIVE submit path (keep dry-run default & behaviour):
   - New flags: --status (connect to the broker, print account + positions, submit NOTHING, exit 0 — the home connectivity test); --force (override the period idempotence guard); --state-dir already exists.
   - Config-drift guard: if a decision_record.json exists at the project root (or --decision path), call verify_config_matches(cfg, record). On a LIVE submit (--no-dry-run), REFUSE if the running config drifted from the frozen pre-registered variant (clear error) — we only paper-trade the pre-registered strategy that the forward hold-out will judge. For a dry-run, just WARN.
   - Cadence/idempotence guard (LIVE submit only): compute period_key(asof, cfg.rebalance); inspect state (a per-period marker, e.g. state file keyed by period or a 'last_submitted_period' record). If already submitted for this period and not --force, SKIP submission with a clear message (still print the preview + write state). This makes a daily-scheduled job safe: it only trades once per month for monthly cadence.
   - Market-open check (LIVE alpaca): call broker.is_market_open(); if closed, log a clear warning that DAY orders will queue for the next session (do not hard-block; allow --force semantics to proceed).
   - Capture results: collect the list of OrderResult from submit_order and store them in the run-state record (symbol, side, qty, status, broker_order_id), plus the period_key and a 'submitted_period' marker.
   - --status mode: build the broker, print account snapshot + current positions in a small table, write nothing to broker, return.
   - Keep main(argv) testable.

4) Documentation — create ${ROOT}/ALPACA_SETUP.md (home setup guide):
   - Create an Alpaca PAPER account, get API key/secret (paper).
   - Install on a personal Mac: clone, python -m venv, pip install -r requirements.txt.
   - Set env vars APCA_API_KEY_ID / APCA_API_SECRET_KEY (paper keys).
   - Connectivity test: python run_live.py --status --broker alpaca.
   - Dry-run preview: python run_live.py --dry-run --live.
   - Go live (paper): python run_live.py --no-dry-run --broker alpaca --live  (explain idempotence: safe to run daily; trades once per month for monthly cadence).
   - Scheduling on macOS: a concrete launchd .plist example (run daily on weekdays; the period guard ensures one trade/month) AND a cron alternative. Note the venv python absolute path and CWD must be the project root.
   - How this feeds the PRISTINE FORWARD HOLD-OUT: each monthly run accrues post-decision bars; after ~252 trading days run python run_holdout.py --live for the first pristine §6.5 read.
   - Safety: it is PAPER only (no real money); kill switch = stop the scheduler / it never runs --no-dry-run without you setting it; config-drift guard prevents trading a non-pre-registered variant. NOT financial advice.

5) Tests — tests/test_live_submit.py (MOCK Alpaca, NO network):
   - A MockTradingClient (plain class) returning canned account/positions/clock/order objects; inject via AlpacaBroker(client=mock). Assert get_account/get_positions/submit_order map correctly; is_market_open reads the clock.
   - run_live.main on the live path using a mock broker (monkeypatch _build_broker or AlpacaBroker, or inject): asserts intents are submitted, OrderResults are recorded in state, submitted count > 0.
   - Idempotence: a second main() call for the SAME period does NOT resubmit (submitted == 0 / skipped), and --force overrides.
   - Config-drift guard: a cfg/record mismatch REFUSES a live submit (raises / errors) but a dry-run only warns.
   - --status mode submits nothing.
   - period_key: correct keys for daily/weekly/monthly; unknown cadence raises.
   Run: cd ${ROOT} && ${PYTEST} tests/test_live_submit.py -q  (green). Plus the existing suite must stay green.
`

const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    files_written: { type: 'array', items: { type: 'string' } },
    smoke_test: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['files_written', 'smoke_test', 'notes'],
}
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    passed: { type: 'boolean' },
    criteria: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { name: { type: 'string' }, met: { type: 'boolean' }, evidence: { type: 'string' } },
      required: ['name', 'met', 'evidence'] } },
    issues: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['passed', 'criteria', 'issues', 'summary'],
}

const CRITERIA = `
ACCEPTANCE CRITERIA (verify by RUNNING real code with the venv; MOCK Alpaca only, no network; default met=false without evidence):
1. AlpacaBroker accepts an injected client (no creds/network needed for tests); is_market_open() reads the clock; get_account/get_positions/submit_order map a mock client's objects correctly.
2. scheduling.period_key gives correct stable keys for daily/weekly/monthly and raises on unknown cadence.
3. run_live.py LIVE submit is IDEMPOTENT per cadence period: a second run for the same period_key does NOT resubmit (unless --force); the first run records OrderResults + the period marker in state.
4. Config-drift guard: a live submit with a config that does not match decision_record.json is REFUSED; a dry-run only warns.
5. --status mode prints account + positions and submits NOTHING.
6. Dry-run remains the default and still sends nothing; existing dry-run behaviour unbroken.
7. ALPACA_SETUP.md exists and covers: paper account + keys, env vars, install, --status connectivity test, going live, a macOS launchd example + cron alternative, how runs feed the pristine forward hold-out, and PAPER-only safety.
8. tests/test_live_submit.py covers the above and PASSES; the full pre-existing suite still passes (no regressions). No real network/credential calls anywhere in tests.
`

phase('Build Alpaca submit hardening + docs')
const build = await agent(
  `Harden the Alpaca paper submit path for safe unattended monthly running + write the home setup doc. Implement fully, then smoke-test (pytest with a MOCK Alpaca client; a dry-run still works offline).\n\n${CONTEXT}\n\nWHAT TO BUILD:\n${BUILD_SPEC}\n\nREAD the existing modules first and EXTEND them. NEVER hit the Alpaca network in tests (inject a mock client). Run the smoke tests with the venv before finishing. Return the structured result.`,
  { label: 'build:alpaca', phase: 'Build Alpaca submit hardening + docs', schema: BUILD_SCHEMA }
)

phase('Adversarial verify')
let verdict = await agent(
  `You are an ADVERSARIAL verifier. Independently check every criterion by READING the code and RUNNING real checks with the venv (MOCK Alpaca only — NO network, NO creds). Default met=false without concrete evidence.\n\n${CONTEXT}\n\n${CRITERIA}\n\nFor each criterion run an actual check, capture output, record met + evidence. passed=true ONLY if every criterion is met. List concrete actionable issues for anything unmet. Pay special attention: confirm NO test makes a real network/broker call, and that the idempotence + config-drift guards actually prevent unsafe submits.`,
  { label: 'verify:alpaca', phase: 'Adversarial verify', schema: VERIFY_SCHEMA }
)
let attempt = 0
while (verdict && !verdict.passed && attempt < 2) {
  attempt++
  log(`alpaca: verify FAILED (attempt ${attempt}) — fixing ${verdict.issues.length} issue(s): ${verdict.issues.slice(0, 3).join(' | ')}`)
  await agent(
    `Fix these UNMET issues at the source (edit real files):\n\nISSUES:\n${verdict.issues.map((s, i) => `${i + 1}. ${s}`).join('\n')}\n\nFAILED CRITERIA:\n${verdict.criteria.filter(c => !c.met).map(c => `- ${c.name}: ${c.evidence}`).join('\n')}\n\n${CONTEXT}\n\nBUILD SPEC (reference):\n${BUILD_SPEC}\n\nFix, re-run pytest (mock Alpaca) + an offline dry-run, return the structured result.`,
    { label: `fix:alpaca#${attempt}`, phase: 'Adversarial verify', schema: BUILD_SCHEMA }
  )
  verdict = await agent(
    `Re-verify the Alpaca submit hardening adversarially (same criteria). Run real checks with the venv (mock Alpaca, no network).\n\n${CONTEXT}\n\n${CRITERIA}\n\nRecord met + evidence per criterion; passed only if all met.`,
    { label: `verify:alpaca#${attempt}`, phase: 'Adversarial verify', schema: VERIFY_SCHEMA }
  )
}

phase('Integration check')
const integration = await agent(
  `Final integration for the Alpaca submit hardening at ${ROOT}. Be honest and adversarial. MOCK Alpaca only (no network/creds).\n\n${CONTEXT}\n\nDo ALL and report:\n1) Clean any stray scratch files at the project root (do NOT touch real source/tests).\n2) Run the FULL suite: cd ${ROOT} && ${PYTEST} -q (report counts; must be green, no regressions).\n3) Confirm the offline DRY-RUN still works & sends nothing: cd ${ROOT} && ${PY} run_live.py --dry-run (paste the tail).\n4) Confirm --status fails GRACEFULLY without creds: cd ${ROOT} && ${PY} run_live.py --status --broker alpaca (should raise the clear 'requires API credentials' error, NOT a traceback about network — paste it).\n5) Verify the idempotence + config-drift guards by reading the code paths and citing the lines.\n6) Confirm ALPACA_SETUP.md exists and contains: paper keys/env, --status test, going-live command, a macOS launchd example, and the link to run_holdout.py for the pristine forward read.\n7) Completeness audit: AlpacaBroker client injection + is_market_open, scheduling.period_key, run_live --status/--force/idempotence/drift-guard/result-capture, tests present & green, no hard-coded market values, NO real network in tests. List anything MISSING or DEVIATING and whether the milestone meets its acceptance criteria.\n`,
  { label: 'integration', phase: 'Integration check', schema: VERIFY_SCHEMA }
)

return {
  build: build ? { files: build.files_written, smoke: build.smoke_test } : null,
  verify: verdict ? { passed: verdict.passed, issues: verdict.issues, summary: verdict.summary } : null,
  integration: integration ? { passed: integration.passed, issues: integration.issues, summary: integration.summary, criteria: integration.criteria } : null,
}
