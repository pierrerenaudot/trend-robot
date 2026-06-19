export const meta = {
  name: 'tsmom-preregistration-holdout',
  description: 'Build a pre-registration + pristine forward hold-out validation harness for the TSMOM robot, then freeze the current decision and report',
  phases: [
    { title: 'Build pre-registration + hold-out' },
    { title: 'Adversarial verify' },
    { title: 'Integration: freeze + report' },
  ],
}

const ROOT = '/Users/pierre.renaudot/test/trend_robot'
const PY = '/Users/pierre.renaudot/test/.venv/bin/python'
const PYTEST = '/Users/pierre.renaudot/test/.venv/bin/pytest'
const DECISION_DATE = '2026-06-19'   // today; the pristine forward hold-out = bars strictly after this

const CONTEXT = `
=== TSMOM ROBOT — PRE-REGISTRATION + PRISTINE FORWARD HOLD-OUT ===
The TSMOM research robot at ${ROOT} is built & validated (T1-T10) + has a paper-trading dry-run (live/ package). An investigation found: (1) a Deflated Sharpe mis-calibration (now FIXED in metrics/deflated_sharpe.py — var_trials auto-estimated to the per-bar scale), and (2) that the long_short SHORT legs drove walk-forward instability; config.yaml is now direction=long_only, rebalance=monthly, which gives a BORDERLINE §6.5 RETAIN.

THE PROBLEM THIS MILESTONE SOLVES: that RETAIN is NOT a clean out-of-sample verdict, because the variant (long_only/monthly) was SELECTED using exploration that looked at data INCLUDING the locked test set. Once you have peeked at all historical data to make a choice, NO past window is truly pristine anymore. The only genuinely untouched out-of-sample evidence is the FUTURE: freeze the decision now, then evaluate the frozen config ONLY on bars that arrive AFTER the decision date. This forward hold-out is exactly what the paper-trading track accrues.

WHAT TO BUILD: a pre-registration mechanism (freeze the chosen config + decision date + the honest number of configurations already explored) and a forward hold-out evaluator that runs the section-6.5 read ONLY on post-decision-date bars, plus a clearly-labelled (non-pristine) RETROSPECTIVE mode so a number can be produced today on existing data.

PYTHON (venv, deps installed; DSR already fixed): ${PY}
PYTEST: ${PYTEST}
Work from ${ROOT} (import root; "import trend_robot.<...>" and "import run_research" work when CWD=${ROOT}). Yahoo is reachable but sometimes 429 — synthetic fallback exists in run_research._load_prices; tests must use synthetic only.

REUSE THESE EXISTING PIECES (READ them first for exact signatures):
- trend_robot/config.py: load_config(path)->Config, set_global_seed(seed); Config is a frozen dataclass (fields incl. universe, direction, rebalance, lookbacks, vol_window, asset_vol_target, portfolio_vol_target, max_gross_leverage, kelly_fraction, cost_bps_per_side, periods_per_year, train_test_ratio, wf_*_years, cv_embargo, seed).
- trend_robot/signals/tsmom.py: tsmom_signal(prices, lookbacks, direction)->DataFrame.
- trend_robot/portfolio/sizing.py: target_weights(signals, returns, cfg)->DataFrame.
- trend_robot/backtest/engine.py: run_backtest(prices, target_weights, cfg)->BacktestResult(equity, weights, turnover, trades).
- trend_robot/metrics/performance.py: performance_metrics(result, cfg)->dict (keys incl. sharpe, cagr, max_drawdown, total_cost, ...).
- trend_robot/metrics/deflated_sharpe.py: deflated_sharpe_ratio(returns, n_trials, skew, kurtosis, var_trials=None)->float  (var_trials now auto-estimated; pandas .kurt() is EXCESS kurtosis so add 3.0 before passing, matching final_report.py).
- trend_robot/validation/splits.py: train_test_split, walk_forward_splits(index, cfg)->list[WalkForwardWindow(fold, train_index, test_index)].
- trend_robot/validation/final_report.py: evaluate_final_validation / _backtest_returns_on / _walk_forward_stability — STUDY this; your forward/retrospective evaluator should mirror its slice-evaluation pattern (compute weights on the FULL price history for no-look-ahead, then slice to the hold-out index, run_backtest on the slice, performance_metrics, DSR on the slice's net-of-cost returns). Reuse format/threshold conventions (DSR threshold 0.60).
- run_research.py: _date_window(history_years, end)->(start,end), _load_prices(cfg, start, end, cache_dir, *, prefer_yfinance=False)->(prices, source).
=== END CONTEXT ===
`

const BUILD_SPEC = `
Build the following. Real, typed, docstringed Python. No hard-coded market values (flow from Config). Tests synthetic-only.

1) trend_robot/validation/preregistration.py
   - strategy_fingerprint(cfg) -> dict: the STRATEGY-RELEVANT subset of Config that defines the variant (universe, direction, rebalance, lookbacks, vol_window, asset_vol_target, portfolio_vol_target, max_gross_leverage, kelly_fraction, cost_bps_per_side, periods_per_year, seed). Deterministic ordering.
   - config_hash(fingerprint: dict) -> str: stable sha256 hex of the JSON-canonicalized fingerprint (sort_keys=True).
   - @dataclass(frozen=True) DecisionRecord: decision_date(str ISO), config_fingerprint(dict), config_hash(str), n_trials_spent(int), created_at(str ISO), notes(str). The forward hold-out is defined as bars with index date STRICTLY AFTER decision_date.
   - freeze_decision(cfg, *, decision_date: str, n_trials_spent: int, notes: str, path) -> DecisionRecord: build + write the record as pretty JSON. REFUSE to silently overwrite an existing record whose config_hash differs (raise a clear error unless an explicit overwrite flag is passed) — pre-registration must be tamper-evident.
   - load_decision(path) -> DecisionRecord.
   - verify_config_matches(cfg, record) -> bool: True iff strategy_fingerprint(cfg)'s hash == record.config_hash (detects config drift since freezing).

2) trend_robot/validation/holdout.py
   - @dataclass(frozen=True) HoldoutReport: mode(str: 'forward'|'retrospective'), decision_date, holdout_start, holdout_end, n_holdout_bars(int), min_bars(int), sufficient(bool), metrics(dict), dsr_threshold(float), dsr_preregistered(float, computed at n_trials=1), dsr_carried(float, computed at n_trials=record.n_trials_spent), stability (walk-forward stability over the hold-out slice if >=2 complete windows fit, else None), retain_preregistered(bool), retain_carried(bool).
   - evaluate_holdout(prices, cfg, record, *, mode='forward', retrospective_months: int|None=None, min_bars: int|None=None, dsr_threshold=0.60) -> HoldoutReport:
       * Compute target_weights on the FULL price history (no look-ahead: each date's signal uses only prior prices), exactly like final_report.
       * Determine the hold-out index:
           - mode='forward': dates strictly AFTER record.decision_date.
           - mode='retrospective': the last retrospective_months months of available data (clearly NOT pristine — label it). Convert months to ~ (periods_per_year/12 * months) bars or use a date offset.
       * min_bars default = one year (cfg.periods_per_year) if None. sufficient = n_holdout_bars >= min_bars.
       * If sufficient: slice prices+weights to the hold-out index, run_backtest, performance_metrics; compute DSR on the slice net-of-cost returns at BOTH n_trials=1 (a single pre-registered test on fresh data) AND n_trials=record.n_trials_spent (conservative, accounting for the selection search). Compute walk-forward stability over the hold-out slice ONLY if >=2 complete windows fit (else stability=None and note it).
       * If NOT sufficient: return a report with sufficient=False and metrics minimal; the caller will print "accrue more data".
       * retain_* = (dsr_* > threshold) AND (stability is None or stability.is_stable).  Be explicit that with stability=None the walk-forward leg is 'not yet assessable'.
   - format_holdout_report(report, record) -> str: a readable block. MUST include a banner stating whether this is a PRISTINE FORWARD hold-out (genuinely untouched) or a NON-PRISTINE RETROSPECTIVE check, the decision hash/date/n_trials, the DSR at both n_trials settings, the data-sufficiency status (e.g. "126/252 bars — need N more"), and a section-11 human-judgment / no-auto-deploy note. Make the pristine-vs-retrospective distinction unmissable.

3) run_holdout.py (at ${ROOT})
   - argparse: --config, --decision (path to decision_record.json; default ${ROOT}/decision_record.json), --live, --end (YYYY-MM-DD), --cache-dir, --history-years (default 15), --mode {forward,retrospective} (default forward), --retrospective-months (int, default 12), --min-bars (int, optional), --log-level.
   - Flow: load_config -> set_global_seed -> load_decision -> verify_config_matches (WARN loudly if the current config drifted from the frozen one) -> _load_prices -> evaluate_holdout -> print format_holdout_report. Provide main(argv=None) for tests.

4) tests/test_preregistration.py (synthetic only):
   - strategy_fingerprint / config_hash: same cfg -> same hash; changing direction or lookbacks -> different hash.
   - freeze_decision/load_decision round-trip; refusing to overwrite a differing record (raises); overwrite flag works.
   - verify_config_matches: True for matching cfg, False after a change.
   - evaluate_holdout forward mode: only bars strictly after decision_date are in the hold-out; with a decision_date at/after the last bar -> sufficient=False (0 bars); with a decision_date early enough -> sufficient=True and metrics computed.
   - retrospective mode: selects ~ the last N months; report.mode=='retrospective'.
   - no-look-ahead: the hold-out evaluation uses weights computed from full history (the forward-slice metrics are unchanged whether or not bars BEFORE the holdout are passed as part of the panel — i.e. weights at holdout dates depend only on data up to those dates). Demonstrate.
   - DSR carried (n_trials_spent) <= DSR preregistered (n_trials=1) for the same slice.
   - run_holdout.main(['--mode','retrospective', ...]) on synthetic runs end-to-end and prints a report (no network).
   Run: cd ${ROOT} && ${PYTEST} tests/test_preregistration.py -q  (green). Use a SyntheticProvider-backed panel or a hand-built one; never hit Yahoo.
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
ACCEPTANCE CRITERIA (verify by RUNNING real code; default met=false without evidence; synthetic only):
1. preregistration.py: strategy_fingerprint/config_hash are deterministic & sensitive (changing direction/lookbacks changes the hash); freeze/load round-trips; freezing refuses to overwrite a DIFFERENT record unless explicitly allowed; verify_config_matches detects drift.
2. holdout.py: the FORWARD hold-out contains ONLY bars strictly after decision_date; with decision_date >= last bar -> sufficient=False (0 forward bars); the section-6.5-style read (DSR + optional walk-forward stability) is computed correctly on the slice, reusing the canonical engine/metrics/DSR (weights computed on FULL history -> no look-ahead). DSR is reported at BOTH n_trials=1 and n_trials=n_trials_spent.
3. The pristine-vs-retrospective distinction is explicit and unmissable in format_holdout_report; retrospective mode is clearly labelled NOT pristine.
4. No-look-ahead: forward-slice metrics depend only on data up to each date (demonstrate the weights at hold-out dates are invariant to removing later data / to truncation).
5. run_holdout.py runs end-to-end (synthetic, no network) for both --mode forward and --mode retrospective and prints the report; main(argv) is testable.
6. tests/test_preregistration.py covers the above and PASSES via ${PYTEST}; the pre-existing full suite still passes (no regressions).
7. No hard-coded market values; genuine reuse of existing validation/metrics modules.
`

phase('Build pre-registration + hold-out')
const build = await agent(
  `Build the pre-registration + pristine forward hold-out harness. Implement fully, then smoke-test (pytest + a synthetic run of run_holdout in both modes).\n\n${CONTEXT}\n\nWHAT TO BUILD:\n${BUILD_SPEC}\n\nREAD the reused modules first (especially validation/final_report.py). Run the smoke tests with the venv before finishing. Return the structured result.`,
  { label: 'build:prereg', phase: 'Build pre-registration + hold-out', schema: BUILD_SCHEMA }
)

phase('Adversarial verify')
let verdict = await agent(
  `You are an ADVERSARIAL verifier. Independently check every criterion by READING the code and RUNNING real checks with the venv. Default met=false without concrete evidence. Synthetic only (no network).\n\n${CONTEXT}\n\n${CRITERIA}\n\nFor each criterion run an actual check, capture output, record met + evidence. passed=true ONLY if every criterion is met. List concrete actionable issues for anything unmet.`,
  { label: 'verify:prereg', phase: 'Adversarial verify', schema: VERIFY_SCHEMA }
)
let attempt = 0
while (verdict && !verdict.passed && attempt < 2) {
  attempt++
  log(`prereg: verify FAILED (attempt ${attempt}) — fixing ${verdict.issues.length} issue(s): ${verdict.issues.slice(0, 3).join(' | ')}`)
  await agent(
    `Fix these UNMET issues at the source (edit real files):\n\nISSUES:\n${verdict.issues.map((s, i) => `${i + 1}. ${s}`).join('\n')}\n\nFAILED CRITERIA:\n${verdict.criteria.filter(c => !c.met).map(c => `- ${c.name}: ${c.evidence}`).join('\n')}\n\n${CONTEXT}\n\nBUILD SPEC (reference):\n${BUILD_SPEC}\n\nFix, re-run pytest + the synthetic run_holdout, return the structured result.`,
    { label: `fix:prereg#${attempt}`, phase: 'Adversarial verify', schema: BUILD_SCHEMA }
  )
  verdict = await agent(
    `Re-verify the pre-registration + hold-out harness adversarially (same criteria). Run real checks with the venv.\n\n${CONTEXT}\n\n${CRITERIA}\n\nRecord met + evidence per criterion; passed only if all met.`,
    { label: `verify:prereg#${attempt}`, phase: 'Adversarial verify', schema: VERIFY_SCHEMA }
  )
}

phase('Integration: freeze + report')
const integration = await agent(
  `Final integration for the pre-registration + hold-out harness at ${ROOT}. Be honest and adversarial.\n\n${CONTEXT}\n\nDo ALL and report:\n1) Clean any stray scratch files at the project root (do NOT touch real source/tests).\n2) Run the FULL suite: cd ${ROOT} && ${PYTEST} -q (report counts; must be green, no regressions).\n3) FREEZE THE CURRENT DECISION: using freeze_decision, write ${ROOT}/decision_record.json for the CURRENT config (load_config on ${ROOT}/config.yaml — it is direction=long_only, rebalance=monthly) with decision_date='${DECISION_DATE}', n_trials_spent=6 (honest count: the baseline + the 2x2 direction×cadence grid + the candidate read explored to select this variant), and a notes string explaining the variant was selected via test-inclusive exploration so only post-${DECISION_DATE} data is a pristine OOS. This file is a versioned pre-registration artifact (it SHOULD be committed — do not gitignore it).\n4) Run the PRISTINE FORWARD hold-out now: cd ${ROOT} && ${PY} run_holdout.py --live --end 2026-06-19  (expect: 0 forward bars -> 'insufficient, accrue forward data' — that is the correct honest state; the forward track will be filled by paper trading). Paste the output.\n5) Run the (clearly non-pristine) RETROSPECTIVE check for context: cd ${ROOT} && ${PY} run_holdout.py --live --end 2026-06-19 --mode retrospective --retrospective-months 12  and paste the DSR/verdict, with the NOT-pristine caveat surfaced.\n6) Completeness audit: preregistration.py + holdout.py + run_holdout.py + tests present and correct; decision_record.json written & loadable; verify_config_matches matches the frozen record; reuse of engine/metrics/DSR genuine; no hard-coded market values. List anything MISSING or DEVIATING.\n`,
  { label: 'integration', phase: 'Integration: freeze + report', schema: VERIFY_SCHEMA }
)

return {
  build: build ? { files: build.files_written, smoke: build.smoke_test } : null,
  verify: verdict ? { passed: verdict.passed, issues: verdict.issues, summary: verdict.summary } : null,
  integration: integration ? { passed: integration.passed, issues: integration.issues, summary: integration.summary, criteria: integration.criteria } : null,
}
