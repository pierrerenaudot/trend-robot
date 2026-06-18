export const meta = {
  name: 'tsmom-robot-finish',
  description: 'Finish & adversarially validate the TSMOM robot: verify T7, complete T8 tests, implement T9 validation harness, build T10 cost-stress + final report',
  phases: [
    { title: 'T7 — verify reproducibility' },
    { title: 'T8 — pytest suite (complete)' },
    { title: 'T9 — Validation harness (implement)' },
    { title: 'T10 — Cost stress & final report' },
    { title: 'Integration check' },
  ],
}

const ROOT = '/Users/pierre.renaudot/test/trend_robot'
const PY = '/Users/pierre.renaudot/test/.venv/bin/python'
const PYTEST = '/Users/pierre.renaudot/test/.venv/bin/pytest'

const SPEC = `
=== TSMOM RESEARCH ROBOT — SPEC EXCERPTS (source of truth) ===
PRINCIPLES (non-negotiable): research/production parity; validation-first; ZERO look-ahead (decision at t uses only info up to t; signals lagged >=1 bar before execution); realistic+pessimistic costs (sensitivity-tested); reproducibility (global seed, deterministic backtests). NO market values hard-coded — everything via config.yaml/Config.

PARAMETERS (in config.yaml): initial_capital 2000; universe [SPY,EFA,EEM,TLT,IEF,GLD,DBC]; direction long_short (opt long_only); rebalance weekly (opt daily/monthly); lookbacks [21,63,126,252]; vol_window 60; asset_vol_target 0.10; portfolio_vol_target 0.10; max_gross_leverage 2.0; kelly_fraction 1.0; cost_bps_per_side 2; cost_stress_levels [5,10]; periods_per_year 252; train_test_ratio 0.70 (last 30% locked OOS test); wf_train_years 5 / wf_test_years 1 / wf_step_years 1; cv_embargo 0.01; seed 42.

SECTION 5 — COST MODEL: cost per trade = cost_bps_per_side * |delta notional|; optional sqrt impact term present in code. MANDATORY sensitivity test: replay backtest at cost_stress_levels. A strategy that only survives low costs is fragile.

SECTION 6 — VALIDATION PROTOCOL (the heart of the project; no shortcuts):
6.1 Locked train/test: last 30% of history = out-of-sample test set, untouched until the very end; all dev/tuning on first 70%.
6.2 Walk-forward: rolling windows — train wf_train_years, test wf_test_years, step wf_step_years; concatenate test segments into one "as-in-production" out-of-sample track. Stability across windows = robustness; instability = overfitting.
6.3 Purged CV + embargo (Lopez de Prado 2018): naive CV is INVALID in finance due to temporal overlap of observations. Implement PURGING (drop train samples whose information window overlaps the test fold) and EMBARGO (cv_embargo fraction of samples blocked immediately AFTER each test fold to stop leakage).
6.4 Multiple-testing correction: maintain an n_trials counter (number of configurations tested) and feed it into deflated_sharpe_ratio. More trials => higher significance bar. Optional/advanced: White Reality Check (2000), Hansen SPA (2005).
6.5 Success criterion: retain a variant ONLY if, ON THE LOCKED TEST SET: (a) Deflated Sharpe clearly positive after correction, AND (b) walk-forward performance is stable (not driven by a single period). Otherwise DO NOT deploy and DO NOT re-tune on the test set.

SECTION 8 — MANDATORY PYTEST TESTS:
- No look-ahead: on the SIGNAL, the SIZING, and the ENGINE (removing data after t does not change the decision at t).
- Cost application: higher turnover => higher costs => lower equity.
- Engine consistency: weight=1 on one asset, zero costs => equity == that asset's cumulative return.
- Sizing math: gross exposure <= cap; realized vol ~ target.
- Metrics exactness: Sharpe/drawdown verified on known analytic cases.
- Data integrity: contract format respected, NaN handled.

SECTION 11 — human-judgment points: the final retain/reject decision and interpretation of validation results are HUMAN judgments; the system must report verdicts but never auto-deploy or re-tune on the locked test set.
=== END SPEC EXCERPTS ===
`

const CURRENT_STATE = `
=== CURRENT PROJECT STATE (already built — do NOT rebuild these; READ them to use their REAL signatures) ===
Project root: ${ROOT}. Package import root = ${ROOT} (run with CWD=${ROOT}; "import trend_robot.<...>" works).

ALREADY IMPLEMENTED & WORKING (T1-T7):
- trend_robot/config.py : @dataclass(frozen=True) Config (fields: initial_capital, universe, direction, rebalance, lookbacks, vol_window, asset_vol_target, portfolio_vol_target, max_gross_leverage, kelly_fraction, cost_bps_per_side, cost_stress_levels, periods_per_year, train_test_ratio, wf_train_years, wf_test_years, wf_step_years, cv_embargo, seed). load_config(path)->Config (validates, raises ConfigError). set_global_seed(seed). ConfigError.
- trend_robot/data/provider.py : DataProvider(Protocol).get_prices(tickers,start,end)->pd.DataFrame ; make_cache_key ; cache_path ; read_cache ; write_cache ; CachedProvider(provider, cache_dir).get_prices(...).
- trend_robot/data/synthetic_provider.py : SyntheticProvider(...).get_prices(tickers,start,end)->pd.DataFrame  (deterministic, seeded GBM; tz-naive DatetimeIndex on business days, columns=tickers, adjusted-close-like). USE THIS for all tests/offline (Yahoo is rate-limited / HTTP 429).
- trend_robot/data/yfinance_provider.py : YFinanceProvider (graceful on 429).
- trend_robot/signals/tsmom.py : tsmom_signal(prices, lookbacks, direction='long_short')->pd.DataFrame in [-1,1]. Pure.
- trend_robot/portfolio/sizing.py : target_weights(signals, returns, cfg)->pd.DataFrame. Pure. (returns = daily pct-change returns DataFrame.)
- trend_robot/backtest/costs.py : bps_to_fraction(bps)->float and the linear+optional-sqrt-impact cost model.
- trend_robot/backtest/engine.py : @dataclass BacktestResult(equity: pd.Series, weights: pd.DataFrame, turnover: pd.Series, trades: pd.DataFrame[date,asset,delta_weight,cost]); run_backtest(prices, target_weights, cfg)->BacktestResult. Target weights are shift(1)-lagged (no look-ahead); rebalance cadence; costs on turnover; mark-to-market.
- trend_robot/metrics/performance.py : performance_metrics(result, cfg)->dict (CAGR, annual vol, Sharpe, Sortino, Calmar/MAR, max drawdown + duration, profit factor, hit rate, avg annual turnover, avg exposure, per-asset P&L).
- trend_robot/metrics/deflated_sharpe.py : observed_sharpe(returns)->float ; expected_max_sharpe(n_trials, var_trials=1.0)->float ; deflated_sharpe_ratio(returns, n_trials, skew, kurtosis)->float (decreases as n_trials grows).
- trend_robot/reporting/report.py : build_report(...) (charts + metrics table; presentation only).
- run_research.py : end-to-end pipeline (load config -> seed -> prices via YFinance w/ cache, fallback SyntheticProvider on 429 -> returns -> tsmom_signal -> target_weights -> run_backtest -> performance_metrics -> build_report). RUNS and produces outputs/ (equity_curve.png, drawdown.png, exposure.png, contribution.png, metrics_table.csv/html). Deflated Sharpe already wired with n_trials.
- tests/test_metrics.py : metrics-exactness tests (17 tests currently pass).

NOT YET DONE / STUBBED (this is the remaining work):
- trend_robot/validation/splits.py : PLACEHOLDER only — __all__=["train_test_split","walk_forward_splits"] but NO implementations. MUST IMPLEMENT in T9.
- trend_robot/validation/purged_cv.py : PLACEHOLDER only — __all__=["purged_cv_splits"] but NO implementation. MUST IMPLEMENT in T9.
- A trials counter for n_trials (T9).
- The full section-8 pytest suite beyond metrics (T8): no-look-ahead (signal/sizing/engine), cost application, engine consistency, sizing math, data integrity.
- Cost-stress sensitivity routine + final section-6.5 validation report on the locked test set (T10) — entirely missing.
=== END CURRENT STATE ===
`

const COMMON = `
PROJECT ROOT: ${ROOT}   (do ALL work from here)
PYTHON (use this EXACT venv interpreter — all deps installed): ${PY}
PYTEST: ${PYTEST}

ENVIRONMENT: Yahoo Finance is RATE-LIMITED (HTTP 429). NEVER depend on live downloads in tests/verification — use SyntheticProvider (deterministic, seeded). Reproducibility is mandatory (seed via cfg.seed / set_global_seed).

RULES:
- Match the spec interfaces and the REAL existing signatures (READ the relevant existing module BEFORE writing code that uses it; do not assume).
- NO market values hard-coded — everything via config.yaml/Config.
- Type annotations + docstrings on public functions. Pure functions for signal/portfolio.
- Do NOT rewrite already-working modules (T1-T7) unless fixing a genuine defect.
- Always run a real smoke test with the venv python/pytest before declaring done; report exact command + key output.
`

const BUILD_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    files_written: { type: 'array', items: { type: 'string' } },
    smoke_test: { type: 'string', description: 'exact command(s) run + key output proving it works' },
    notes: { type: 'string' },
  },
  required: ['files_written', 'smoke_test', 'notes'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    passed: { type: 'boolean' },
    criteria: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          name: { type: 'string' },
          met: { type: 'boolean' },
          evidence: { type: 'string', description: 'command run + observed output, or code citation' },
        },
        required: ['name', 'met', 'evidence'],
      },
    },
    issues: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['passed', 'criteria', 'issues', 'summary'],
}

const TASKS = [
  {
    key: 'T7', title: 'T7 — verify reproducibility',
    build: `T7 is ALREADY implemented and runs end-to-end (run_research.py produces outputs/ + a metrics table). Your job is NOT to rewrite it. Confirm REPRODUCIBILITY: from ${ROOT}, run "${PY} run_research.py" TWICE and compare the printed metrics / metrics_table.csv across the two runs — they must be IDENTICAL (deterministic, seeded). If and ONLY if they differ, find and fix the source of nondeterminism (missing/late seeding, dict ordering, unseeded RNG, timestamp in output) at the source — minimal change. Also confirm it does not crash despite Yahoo 429 (must fall back to SyntheticProvider and log the data source used). Report the two metric sets and whether they matched.`,
    criteria: `Acceptance (spec 9/T7): produces a REPRODUCIBLE equity curve + metrics table. Verify by running run_research.py twice from ${ROOT} and confirming identical metrics (diff the two metrics_table.csv or captured outputs). Confirm it ran without crashing and report which data source was used (synthetic fallback expected given 429).`,
  },
  {
    key: 'T8', title: 'T8 — pytest suite (complete)',
    build: `T8 — Complete the section-8 pytest suite. tests/test_metrics.py already covers metrics exactness (keep it). READ the existing modules for real signatures, then ADD test files (synthetic/deterministic data only — use SyntheticProvider or hand-built DataFrames; NEVER hit Yahoo) covering EVERY remaining section-8 category:
- No look-ahead on the SIGNAL (tsmom_signal): signal at t unchanged when data after t is removed (truncate-and-compare).
- No look-ahead on the SIZING (target_weights): weights at t unchanged when future data removed.
- No look-ahead on the ENGINE (run_backtest): decision/held-book for a bar does not use that bar's or later target; weights are shift(1)-lagged. Prove the lag.
- Cost application: higher cost_bps (or higher turnover) => higher total cost => lower final equity (monotonic).
- Engine consistency: constant weight=1 on a single asset with zero costs => equity replicates that asset's cumulative return within float tolerance.
- Sizing math: gross exposure sum(|w|) <= max_gross_leverage on every date; realized portfolio vol ~ portfolio_vol_target (order of magnitude).
- Data integrity: provider/contract — tz-naive date index, columns=tickers, NaN preserved (not silently filled).
Run the FULL suite from ${ROOT}: ${PYTEST} -q. Smoke test = the full green pytest output with the new test count.`,
    criteria: `Acceptance (spec 9/T8): ALL section-8 categories are covered by real tests AND the FULL suite passes (${PYTEST} -q from ${ROOT}), using synthetic data only (no network). Specifically confirm there exist passing tests for: no-look-ahead on signal, on sizing, AND on engine; cost application monotonicity; engine consistency (weight=1, zero cost => asset return); sizing cap + vol-target; data integrity (NaN + contract). Any failing/missing category = fail.`,
  },
  {
    key: 'T9', title: 'T9 — Validation harness (implement)',
    build: `T9 — Implement the validation harness (spec section 6). The files trend_robot/validation/splits.py and purged_cv.py are currently EMPTY PLACEHOLDERS — implement them for real. READ config.py (cfg fields), engine, metrics/deflated_sharpe first.
Implement:
- splits.py: train_test_split(index_or_df, cfg) -> locked split where the LAST (1 - train_test_ratio) fraction is the untouched out-of-sample test set (return train/test index slices or boolean masks). walk_forward_splits(index, cfg) -> generator/list of (train_idx, test_idx) rolling windows using wf_train_years/wf_test_years/wf_step_years (convert years to bars via periods_per_year). Provide a helper to concatenate the per-window test segments into one contiguous, non-overlapping out-of-sample track.
- purged_cv.py: a PurgedKFold-style splitter (purged_cv_splits(...) and/or a class) implementing PURGING (remove train samples whose information window overlaps the test fold; account for label/lookback horizon overlap) AND EMBARGO (drop cv_embargo fraction of samples immediately after each test fold). Per Lopez de Prado 2018.
- A trials counter: a small class/function (e.g. TrialCounter) that counts configurations evaluated and exposes n_trials to feed deflated_sharpe_ratio (wire-compatible with metrics/deflated_sharpe.deflated_sharpe_ratio's n_trials arg).
Add pytest tests (synthetic data) in tests/ for: locked split is exactly the last 30% (within rounding) and train is the first 70%; walk-forward windows have correct lengths/step and concatenated test segments are contiguous and non-overlapping and cover the expected span; purging actually removes the overlapping train indices (assert counts); embargo removes the expected number of samples; trials counter increments and feeds DSR. Run ${PYTEST} -q from ${ROOT}. Smoke test = green pytest + a quick printed demo of walk-forward windows + purge/embargo counts on synthetic data.`,
    criteria: `Acceptance (spec section 6): (1) locked train/test split correct — test = LAST 30%, train = first 70%, no overlap. (2) walk-forward yields correct rolling windows (lengths from wf_*_years via periods_per_year) and concatenated test track is contiguous + non-overlapping. (3) purged CV genuinely PURGES overlapping train samples AND applies EMBARGO of cv_embargo (demonstrate the removed counts numerically). (4) a trials counter exists and is wired to feed n_trials into deflated_sharpe_ratio. (5) New pytest tests for all the above pass via ${PYTEST} from ${ROOT}. splits.py and purged_cv.py must contain REAL implementations (not placeholders).`,
  },
  {
    key: 'T10', title: 'T10 — Cost stress & final report',
    build: `T10 — Cost sensitivity + final validation report (spec sections 5 & 6.5). Depends on T9. READ run_research.py, engine, metrics, validation/splits, validation/purged_cv, deflated_sharpe first.
Implement:
- A cost-sensitivity routine that replays the SAME backtest at the base cost and at each cfg.cost_stress_levels ([5,10] bps), collecting key metrics at each level into a comparison table (CAGR/Sharpe/max DD/total cost/final equity). Put it in a dedicated module (e.g. trend_robot/validation/stress.py) reusing run_backtest (do not duplicate strategy logic) — vary cost by constructing a Config copy with a different cost_bps_per_side (dataclasses.replace, since Config is frozen).
- A final validation report that, ON THE LOCKED TEST SET (use validation.splits.train_test_split): computes the Deflated Sharpe (corrected with the T9 trials counter / n_trials) and walk-forward stability (per-window Sharpe/return dispersion), then EXPLICITLY evaluates the section-6.5 criterion: retain ONLY if DSR clearly positive AND walk-forward stable. Print a clear RETAIN/REJECT verdict plus a section-11 note that this is a human judgment and the system never auto-deploys or re-tunes on the test set.
- Wire it into an entry point: add a "--validate" flag to run_research.py OR create run_validation.py (preferred) that runs cost-stress + the locked-test-set validation report end-to-end on the (synthetic-fallback) data.
Smoke test: run the validation entry point with ${PY} from ${ROOT} end-to-end; show the cost-stress table and the section-6.5 verdict printout.`,
    criteria: `Acceptance (spec 9/T10 + 6.5): (1) backtest replayed at base + all cost_stress_levels producing a comparison table, with higher cost => worse performance (lower final equity / higher total cost). (2) Deflated Sharpe (with n_trials from the trials counter) AND walk-forward stability computed ON THE LOCKED TEST SET, and the section-6.5 criterion explicitly evaluated with a clear RETAIN/REJECT verdict + a human-judgment / no-auto-deploy note. (3) A working entry point (run_validation.py or run_research.py --validate) runs it end-to-end via ${PY} from ${ROOT} without crashing (synthetic fallback OK). Verify by running it and reading the printed table + verdict.`,
  },
]

function buildPrompt(task) {
  return `You are completing one task of the TSMOM robot build. Implement it FULLY and correctly against the EXISTING codebase, then smoke-test it.\n\n${COMMON}\n\n${CURRENT_STATE}\n\n${SPEC}\n\nYOUR TASK NOW:\n${task.build}\n\nDeliver real, working, typed Python. READ the existing modules you depend on first. Run the smoke test with the venv python/pytest before finishing. Return the structured result.`
}

function verifyPrompt(task) {
  return `You are an ADVERSARIAL verifier. Do NOT trust the builder — independently check each acceptance criterion by READING the code and RUNNING real checks with the venv python/pytest. Default to met=false unless you have concrete evidence (command output or code citation). Use synthetic data only (Yahoo is 429).\n\n${COMMON}\n\n${CURRENT_STATE}\n\n${SPEC}\n\nTASK UNDER REVIEW: ${task.title}\n\nACCEPTANCE CRITERIA:\n${task.criteria}\n\nFor each criterion run an actual check (write a tiny script + execute with ${PY}, or run ${PYTEST}), capture output, record met + evidence. passed=true ONLY if every criterion is met. List concrete, actionable issues for anything unmet.`
}

function fixPrompt(task, verdict) {
  return `You are fixing a partially-failing task. The adversarial verifier found these UNMET issues — fix them at the source (edit the real files), do not paper over:\n\nISSUES:\n${verdict.issues.map((s, i) => `${i + 1}. ${s}`).join('\n')}\n\nFAILED CRITERIA EVIDENCE:\n${verdict.criteria.filter(c => !c.met).map(c => `- ${c.name}: ${c.evidence}`).join('\n')}\n\n${COMMON}\n\n${CURRENT_STATE}\n\n${SPEC}\n\nORIGINAL TASK:\n${task.build}\n\nFix the issues, re-run the smoke test with the venv python/pytest, and return the structured result.`
}

const results = []
for (const task of TASKS) {
  phase(task.title)
  const build = await agent(buildPrompt(task), { label: `build:${task.key}`, phase: task.title, schema: BUILD_SCHEMA })
  let verdict = await agent(verifyPrompt(task), { label: `verify:${task.key}`, phase: task.title, schema: VERIFY_SCHEMA })
  let attempt = 0
  while (verdict && !verdict.passed && attempt < 2) {
    attempt++
    log(`${task.key}: verify FAILED (attempt ${attempt}) — fixing ${verdict.issues.length} issue(s): ${verdict.issues.slice(0, 3).join(' | ')}`)
    await agent(fixPrompt(task, verdict), { label: `fix:${task.key}#${attempt}`, phase: task.title, schema: BUILD_SCHEMA })
    verdict = await agent(verifyPrompt(task), { label: `verify:${task.key}#${attempt}`, phase: task.title, schema: VERIFY_SCHEMA })
  }
  results.push({ key: task.key, title: task.title, passed: !!(verdict && verdict.passed), issues: verdict ? verdict.issues : ['verifier returned null'], summary: verdict ? verdict.summary : '', files: build ? build.files_written : [] })
  log(`${task.key} -> ${verdict && verdict.passed ? 'PASSED' : 'NOT PASSED'}`)
}

phase('Integration check')
const integration = await agent(
  `Final whole-project integration & completeness check for the TSMOM robot at ${ROOT}. Be honest and adversarial.\n\n${COMMON}\n\n${CURRENT_STATE}\n\n${SPEC}\n\nDo ALL of the following and report findings:\n1) Clean stray scratch files (e.g. any _smoke_*.py at the project root) — but do NOT touch real source/tests.\n2) Run the FULL test suite from ${ROOT}: ${PYTEST} -q  (report exact pass/fail counts).\n3) Run the end-to-end pipeline from ${ROOT}: ${PY} run_research.py (must not crash; report data source used + that metrics table + charts were produced).\n4) Run the T10 validation entry point (run_validation.py or run_research.py --validate) end-to-end and confirm the cost-stress table + section-6.5 RETAIN/REJECT verdict print.\n5) Completeness audit vs spec: confirm EVERY module in the arborescence exists with REAL (non-placeholder) implementations — especially validation/splits.py and validation/purged_cv.py; confirm key interface signatures match; grep for hard-coded market values outside config.yaml; confirm zero-look-ahead is enforced; confirm all section-8 test categories exist. List anything MISSING or DEVIATING and whether each task T7-T10 meets its acceptance criteria.\n`,
  { label: 'integration', phase: 'Integration check', schema: VERIFY_SCHEMA }
)

return {
  tasks: results,
  integration: integration ? { passed: integration.passed, issues: integration.issues, summary: integration.summary, criteria: integration.criteria } : null,
}
