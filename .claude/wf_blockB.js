export const meta = {
  name: 'tsmom-block-b-quality',
  description: 'Quality & harness work for the TSMOM robot: CI tests workflow, White Reality Check + Hansen SPA, shorter forward walk-forward windows, and a results-preserving speedup',
  phases: [
    { title: 'B1 — CI tests workflow' },
    { title: 'B2 — White Reality Check + Hansen SPA' },
    { title: 'B3 — Shorter forward walk-forward windows' },
    { title: 'B4 — Results-preserving speedup' },
    { title: 'Integration check' },
  ],
}

const REPO = '/Users/pierre.renaudot/test'           // git repo root (.github lives here)
const ROOT = '/Users/pierre.renaudot/test/trend_robot' // project (config.yaml, run_*.py, tests/)
const PY = '/Users/pierre.renaudot/test/.venv/bin/python'
const PYTEST = '/Users/pierre.renaudot/test/.venv/bin/pytest'

const CONTEXT = `
=== TSMOM ROBOT — QUALITY & HARNESS (block B) ===
The TSMOM research robot is complete & validated. Repo layout (IMPORTANT):
- GIT REPO ROOT: ${REPO}   (this is where .github/ lives; there is already .github/workflows/paper-trading.yml here)
- PROJECT DIR:   ${ROOT}   (contains config.yaml, requirements.txt, run_research.py, run_validation.py, run_live.py, run_holdout.py, tests/, and the package trend_robot/)
- The package import root is ${ROOT}; "import trend_robot.<...>" and "import run_research" work only with CWD=${ROOT}.

PYTHON (venv, all deps installed): ${PY}
PYTEST: ${PYTEST}
The full suite currently passes (127 tests, ~7-11 min). Yahoo may be 429 — tests use synthetic data only, never the network.

RELEVANT EXISTING MODULES (READ before editing):
- trend_robot/config.py : Config (frozen dataclass) + load_config + set_global_seed. Fields incl. lookbacks, vol_window, asset_vol_target, portfolio_vol_target, max_gross_leverage, kelly_fraction, cost_bps_per_side, periods_per_year, train_test_ratio, wf_train_years, wf_test_years, wf_step_years, cv_embargo, seed, direction, rebalance, universe, initial_capital, cost_stress_levels.
- trend_robot/signals/tsmom.py : tsmom_signal(prices, lookbacks, direction).
- trend_robot/portfolio/sizing.py : target_weights(signals, returns, cfg) -> DataFrame. Has a per-date Python loop (_ewma_cov_at / _portfolio_vol per as-of date) -> the main perf hot spot.
- trend_robot/backtest/engine.py : run_backtest(prices, target_weights, cfg) -> BacktestResult. Has a sequential per-bar loop (path-dependent equity/drift) -> inherently sequential.
- trend_robot/metrics/deflated_sharpe.py : observed_sharpe, expected_max_sharpe, deflated_sharpe_ratio (var_trials auto-estimated).
- trend_robot/validation/splits.py : train_test_split, walk_forward_splits(index, cfg) using cfg.wf_*_years * cfg.periods_per_year, WalkForwardWindow, concat_test_segments.
- trend_robot/validation/final_report.py : evaluate_final_validation, _walk_forward_stability (uses walk_forward_splits over the whole index).
- trend_robot/validation/holdout.py : evaluate_holdout(prices, cfg, record, *, mode, retrospective_months, min_bars, dsr_threshold) + format_holdout_report. Computes the §6.5-style read on a forward/retrospective slice; walk-forward stability over the slice uses walk_forward_splits(slice, cfg) -> with the spec's 5y/1y windows it needs ~7 years of slice for 2 windows.
- run_holdout.py : CLI for the hold-out (flags incl. --mode, --retrospective-months, --min-bars, --decision, --history-years, --live, --end).
- trend_robot/validation/trials.py : TrialCounter (n_trials).
=== END CONTEXT ===
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

const TASKS = [
  {
    key: 'B1', title: 'B1 — CI tests workflow',
    build: `B1 — Add a Continuous-Integration workflow that runs the pytest suite on push & PR. Create ${REPO}/.github/workflows/tests.yml:
- Triggers: on push to main and on pull_request (any branch). Add workflow_dispatch too.
- Job on ubuntu-latest, defaults.run.working-directory: trend_robot.
- Steps: actions/checkout@v4 ; actions/setup-python@v5 (python-version "3.13", cache: pip, cache-dependency-path: trend_robot/requirements.txt) ; pip install -r requirements.txt ; run the suite: "${PYTEST} -q" but using the runner's python (i.e. just "pytest -q" since setup-python is on PATH) from trend_robot. Use a sensible job timeout (e.g. timeout-minutes: 25) since the suite is ~10 min.
- permissions: contents: read. Add a concurrency group keyed by ref with cancel-in-progress: true (so new pushes cancel stale CI runs).
- Do NOT put Alpaca secrets here (tests need none). Keep it minimal and correct.
Validate the YAML parses (e.g. ${PY} -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('${REPO}/.github/workflows/*.yml')]; print('ok')"). Do not break the existing paper-trading.yml. Smoke test = YAML validation output + a confirmation that "pytest -q" is the command the job runs.`,
    criteria: `(1) ${REPO}/.github/workflows/tests.yml exists, valid YAML, triggers on push+pull_request(+dispatch). (2) It sets working-directory trend_robot, sets up Python 3.13 with pip cache, installs requirements, and runs the pytest suite. (3) It has a job timeout and a cancel-in-progress concurrency group, permissions contents: read, and no secrets. (4) The existing paper-trading.yml is untouched and still valid. Verify by parsing both YAML files and citing the relevant keys.`,
  },
  {
    key: 'B2', title: 'B2 — White Reality Check + Hansen SPA',
    build: `B2 — Implement the advanced multiple-testing tests from spec section 6.4: White's Reality Check (2000) and Hansen's SPA test (2005), using a stationary bootstrap (Politis & Romano 1994). New module trend_robot/validation/multiple_testing.py:
- stationary_bootstrap_indices(n, avg_block, n_boot, rng) -> ndarray (n_boot x n) of resampled time indices (geometric block lengths; wrap-around). Deterministic given a seeded numpy Generator.
- whites_reality_check(perf: np.ndarray | pd.DataFrame, *, avg_block=..., n_boot=..., seed=...) -> dict: perf is a (T x K) matrix of per-period performance statistics for K candidate strategies relative to a benchmark (e.g. excess returns, or loss differentials d_k,t where positive = model beats benchmark). Test statistic V = max_k sqrt(T) * mean_k. Bootstrap the null distribution (recentred per White) and return {statistic, p_value, k_best}. p_value = fraction of bootstrap maxima >= V.
- hansens_spa(perf, *, avg_block=..., n_boot=..., seed=...) -> dict: Hansen's SPA (studentized statistic T_SPA = max_k sqrt(T)*mean_k / std_k), with the consistent recentring (omega_k threshold). Return at least {statistic, p_value} (the consistent p-value; optionally lower/upper). Document the recentring used.
- Both pure & deterministic (seeded). Type annotations + docstrings citing the references.
Add tests trend_robot tests/test_multiple_testing.py (synthetic, seeded):
  * all-null candidates (zero-mean noise) => large p-values (e.g. > 0.10) most of the time;
  * one genuinely superior candidate (positive mean) among many null => small p-value (e.g. < 0.05);
  * adding many useless candidates INCREASES White's p-value for a fixed good model (data-snooping penalty) — demonstrate monotonic-ish behavior;
  * stationary_bootstrap_indices shape/determinism;
  * SPA p-value <= or ~ consistent with White on a clear case.
Run: cd ${ROOT} && ${PYTEST} tests/test_multiple_testing.py -q. Smoke test = green tests + a tiny printed demo (null vs one-good-model p-values).`,
    criteria: `(1) multiple_testing.py implements a stationary bootstrap, White's Reality Check, and Hansen's SPA — pure, seeded/deterministic, with reference-citing docstrings. (2) Statistical sanity holds on seeded synthetic data: all-null => high p-value; one truly-good-among-many => low p-value; more useless candidates => higher White p-value (data-snooping penalty). (3) tests/test_multiple_testing.py covers these and PASSES. (4) The full pre-existing suite still passes (no regressions / no import breakage). Verify by running the new tests AND citing the statistical-sanity outputs with ${PY}.`,
  },
  {
    key: 'B3', title: 'B3 — Shorter forward walk-forward windows',
    build: `B3 — Make the hold-out's walk-forward stability leg assessable on a SHORT forward window (the spec's 5y/1y windows need ~7 years of forward data for 2 windows; we want leg (b) usable in ~1.5-2 years). Without changing the spec defaults for the locked-test §6.5:
- Extend walk_forward_splits (or add a sibling) so the window lengths can be supplied explicitly in BARS (not only via cfg years), e.g. walk_forward_splits(index, cfg, *, train_bars=None, test_bars=None, step_bars=None) — when provided, these override cfg.wf_*_years*periods_per_year; otherwise behaviour is unchanged (backward compatible; existing tests must still pass).
- In trend_robot/validation/holdout.py evaluate_holdout: add optional params wf_train_months / wf_test_months / wf_step_months (or *_bars) that, when set, are used for the FORWARD/RETROSPECTIVE walk-forward stability computation (convert months->bars via periods_per_year/12). Default None => current behaviour. Surface clearly in format_holdout_report which window lengths were used.
- In run_holdout.py: add CLI flags --wf-train-months / --wf-test-months / --wf-step-months (default unset) wired through.
Add tests (synthetic) in tests/test_holdout_wf.py (or extend an existing hold-out test): with short windows (e.g. train 6mo / test 3mo) on a ~2-year slice, walk-forward stability becomes assessable (>=2 windows) whereas the 5y/1y default yields 0 windows; backward-compat: omitting the overrides reproduces the prior result exactly.
Run: cd ${ROOT} && ${PYTEST} tests/ -k "holdout or walk or wf" -q. Smoke test = green + a printed demo showing #windows with short vs default windows on a 2-year synthetic slice.`,
    criteria: `(1) walk_forward_splits accepts explicit bar-length overrides; with none given it is byte-for-byte backward compatible (existing splits/validation tests pass). (2) evaluate_holdout + run_holdout expose short-window overrides (months) for the FORWARD walk-forward leg; defaults unchanged. (3) On a ~2-year synthetic slice, short windows yield >=2 walk-forward windows (leg (b) assessable) while the 5y/1y default yields 0 — demonstrated numerically. (4) The window lengths used are shown in the report. (5) New tests pass AND the full pre-existing suite still passes. Verify with ${PY}/${PYTEST}.`,
  },
  {
    key: 'B4', title: 'B4 — Results-preserving speedup',
    build: `B4 — Speed up the suite/backtest WITHOUT changing results. The hot spot is the per-as-of-date loop in trend_robot/portfolio/sizing.py (EWMA covariance recomputed per date); engine.run_backtest's per-bar loop is path-dependent and largely must stay sequential (optimize only safely).
- Profile first (e.g. time target_weights on a multi-year synthetic panel) to confirm the hot spot. Report the before timing.
- Vectorize / precompute where it PRESERVES results: e.g. compute the EWMA covariance terms with vectorized pandas/numpy (ewm on returns and cross-products) instead of recomputing a full covariance from scratch at every date; reuse arrays. Keep the public signatures identical (target_weights(signals, returns, cfg), run_backtest(...)).
- HARD REQUIREMENT — numerical equivalence: target_weights and run_backtest outputs must match the pre-change outputs within a tight tolerance (<=1e-6 on weights/equity), and ALL 127 existing tests must still pass. If a given optimization cannot preserve results within tolerance + green tests, DO NOT apply it — keep correctness over speed and say so.
- Capture before/after timing of the full suite (or of target_weights on a fixed synthetic panel) to show the speedup.
Smoke test: a script that builds a fixed seeded synthetic panel, computes target_weights + run_backtest BEFORE (git-stashed/baseline values you capture first) vs AFTER and asserts max abs diff <= 1e-6; plus the full ${PYTEST} -q result and before/after timing.`,
    criteria: `(1) A real perf hot spot was profiled and reported (before timing). (2) The optimization PRESERVES results: target_weights & run_backtest match baseline within <=1e-6 (demonstrate with a numerical diff on a seeded synthetic panel) AND the full 127-test suite still passes. (3) A measurable speedup is shown (before/after timing) OR, if no safe speedup was achievable, that is reported honestly with evidence and nothing is degraded. (4) Public signatures unchanged. Verify by running the numerical-equivalence check AND ${PYTEST} -q with ${PY}.`,
  },
]

function buildPrompt(task) {
  return `You are completing one quality/harness task for the TSMOM robot. Implement it fully, then smoke-test with the venv.\n\n${CONTEXT}\n\nYOUR TASK:\n${task.build}\n\nREAD the modules you touch first. Never hit the network in tests. Run the smoke test before finishing. Return the structured result.`
}
function verifyPrompt(task) {
  return `You are an ADVERSARIAL verifier. Independently check each criterion by READING the code and RUNNING real checks with the venv (synthetic data only, no network). Default met=false without concrete evidence (command output / code citation).\n\n${CONTEXT}\n\nTASK UNDER REVIEW: ${task.title}\n\nACCEPTANCE CRITERIA:\n${task.criteria}\n\nFor each criterion run an actual check, capture output, record met + evidence. passed=true ONLY if every criterion is met. List concrete actionable issues for anything unmet.`
}
function fixPrompt(task, verdict) {
  return `Fix these UNMET issues at the source (edit real files, do not paper over):\n\nISSUES:\n${verdict.issues.map((s, i) => `${i + 1}. ${s}`).join('\n')}\n\nFAILED CRITERIA:\n${verdict.criteria.filter(c => !c.met).map(c => `- ${c.name}: ${c.evidence}`).join('\n')}\n\n${CONTEXT}\n\nORIGINAL TASK:\n${task.build}\n\nFix, re-run the smoke test, return the structured result.`
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
  results.push({ key: task.key, title: task.title, passed: !!(verdict && verdict.passed), issues: verdict ? verdict.issues : ['verifier null'], summary: verdict ? verdict.summary : '' })
  log(`${task.key} -> ${verdict && verdict.passed ? 'PASSED' : 'NOT PASSED'}`)
}

phase('Integration check')
const integration = await agent(
  `Final integration check for block B (quality & harness) of the TSMOM robot. Be honest and adversarial. Synthetic data only.\n\n${CONTEXT}\n\nDo ALL and report:\n1) Clean any stray scratch files at ${ROOT} root (do NOT touch real source/tests).\n2) Run the FULL suite: cd ${ROOT} && ${PYTEST} -q (report exact counts; must be green, no regressions, including the new B2/B3 tests).\n3) Validate BOTH workflow YAMLs parse: ${PY} -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('${REPO}/.github/workflows/*.yml')]; print('yaml ok')" and confirm tests.yml runs pytest and paper-trading.yml is unchanged.\n4) Confirm B3: on a ~2-year synthetic slice, short walk-forward windows yield >=2 windows while the 5y/1y default yields 0 (cite numbers).\n5) Confirm B4: target_weights & run_backtest still match baseline within 1e-6 (or that no unsafe optimization was applied) and report the before/after timing.\n6) Confirm B2: White & Hansen SPA give sane p-values on the null vs one-good-model cases.\n7) Completeness audit: list every file added/changed, confirm no hard-coded market values, no network in tests, public signatures unchanged for sizing/engine. State whether each of B1-B4 meets its acceptance criteria and flag anything MISSING or DEVIATING.\n`,
  { label: 'integration', phase: 'Integration check', schema: VERIFY_SCHEMA }
)

return {
  tasks: results,
  integration: integration ? { passed: integration.passed, issues: integration.issues, summary: integration.summary, criteria: integration.criteria } : null,
}
