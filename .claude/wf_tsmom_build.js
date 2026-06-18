export const meta = {
  name: 'tsmom-robot-build',
  description: 'Build & adversarially validate the TSMOM trading research robot (tasks T1-T10) per the spec',
  phases: [
    { title: 'T1 — Skeleton & config' },
    { title: 'T2 — Data layer' },
    { title: 'T3 — TSMOM signal' },
    { title: 'T4 — Sizing & portfolio' },
    { title: 'T5 — Costs & backtest engine' },
    { title: 'T6 — Metrics & Deflated Sharpe' },
    { title: 'T7 — run_research.py' },
    { title: 'T8 — pytest suite' },
    { title: 'T9 — Validation harness' },
    { title: 'T10 — Cost stress & final report' },
    { title: 'Integration check' },
  ],
}

const ROOT = '/Users/pierre.renaudot/test/trend_robot'
const PY = '/Users/pierre.renaudot/test/.venv/bin/python'
const PYTEST = '/Users/pierre.renaudot/test/.venv/bin/pytest'

const SPEC = `
=== TSMOM RESEARCH ROBOT — IMPLEMENTATION SPEC (source of truth) ===
Scope: research + validation only (data layer, TSMOM signal, vol-targeted portfolio, realistic backtest, honest metrics incl. Deflated Sharpe, rigorous validation harness). OUT of scope: live execution, OMS, broker, paper/live trading, intraday.

NON-NEGOTIABLE PRINCIPLES:
- Research/production parity: code must be reusable as-is later. No throwaway shortcuts.
- Validation-first: validation pipeline quality matters more than signal sophistication.
- ZERO look-ahead: any decision at date t uses only info available at t (signals lagged >=1 bar before execution).
- Realistic & pessimistic costs: modeled + sensitivity-tested.
- Reproducibility: global seed, versioned config, deterministic backtests.

STACK & CONVENTIONS:
- Python 3.11+ (here 3.14). Libs: pandas, numpy, scipy, matplotlib, yfinance, pyyaml, pytest, pyarrow. No heavy backtest framework.
- Centralized config: config.yaml + typed dataclass Config. NO market values hard-coded anywhere — everything via config.
- Quality: type annotations, docstrings on public funcs, pure functions for signal/portfolio (no side effects).

PROJECT ARBORESCENCE (project root = trend_robot/, inside it a package also named trend_robot/):
trend_robot/
  config.yaml
  trend_robot/
    __init__.py
    config.py            # dataclass Config + YAML loading + global seed
    data/
      __init__.py
      provider.py        # DataProvider Protocol + disk cache helper
      yfinance_provider.py
      synthetic_provider.py   # deterministic offline provider (added: Yahoo is rate-limited here)
    signals/
      __init__.py
      tsmom.py
    portfolio/
      __init__.py
      sizing.py
    backtest/
      __init__.py
      engine.py
      costs.py
    metrics/
      __init__.py
      performance.py
      deflated_sharpe.py
    validation/
      __init__.py
      splits.py
      purged_cv.py
    reporting/
      __init__.py
      report.py
  tests/
    __init__.py (optional)
    ...
  run_research.py        # entry point: end-to-end backtest + report

PARAMETERS (defaults, all live in config.yaml; add a global integer seed=42):
- initial_capital: 2000
- universe: [SPY, EFA, EEM, TLT, IEF, GLD, DBC]
- direction: long_short   (option: long_only -> truncate negative signals to 0)
- rebalance: weekly       (options: daily, monthly)
- lookbacks: [21, 63, 126, 252]   (days = 1/3/6/12 months, averaged)
- vol_window: 60          (days; EWMA com~60 recommended)
- asset_vol_target: 0.10  (10% annualized, per asset before aggregation)
- portfolio_vol_target: 0.10  (10% annualized, portfolio level)
- max_gross_leverage: 2.0
- kelly_fraction: 1.0
- cost_bps_per_side: 2
- cost_stress_levels: [5, 10]
- periods_per_year: 252
- train_test_ratio: 0.70   (last 30% locked as out-of-sample test)
- wf_train_years: 5, wf_test_years: 1, wf_step_years: 1
- cv_embargo: 0.01   (1% of samples)
- seed: 42

KEY INTERFACES (implement EXACTLY these signatures):
- DataProvider (typing.Protocol):
    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame
    Returns DataFrame indexed by date (tz-naive, trading days), columns=tickers, values=ADJUSTED close (dividends/splits). No silent forward-fill; gaps explicit (NaN). Never future data (no value after the end date).
- tsmom_signal(prices: pd.DataFrame, lookbacks: list[int], direction: str = 'long_short') -> pd.DataFrame
    Signal in [-1,1] per asset/date. For each lookback L: s_L = sign(P_t/P_{t-L} - 1). Final s = mean(s_L) over lookbacks. long_only -> s = max(s,0). No look-ahead (signal at t uses only prices up to t).
- target_weights(signals: pd.DataFrame, returns: pd.DataFrame, cfg: Config) -> pd.DataFrame
    1) estimate ex-ante vol per asset (vol_window, EWMA recommended, annualized by sqrt(periods_per_year));
    2) raw weight w_i = s_i * (asset_vol_target / sigma_i);
    3) estimate portfolio ex-ante vol sigma_p (covariance or approximation) -> scale factor k = portfolio_vol_target/sigma_p; w = k*w;
    4) apply kelly_fraction;
    5) cap so that sum(|w_i|) <= max_gross_leverage (renormalize if exceeded).
    Zero weight when signal is 0 or vol undefined (insufficient history).
- @dataclass BacktestResult: equity: pd.Series; weights: pd.DataFrame; turnover: pd.Series; trades: pd.DataFrame (date, asset, delta_weight, cost)
- run_backtest(prices: pd.DataFrame, target_weights: pd.DataFrame, cfg: Config) -> BacktestResult
    Rebalance at cfg.rebalance cadence toward target weights. Target weights are SHIFTED >=1 bar before application (no look-ahead). Apply cost model on turnover. Mark to market.
- costs.py: cost = cost_bps_per_side * |delta notional|; optional sqrt impact term impact = c*sigma_i*sqrt(|delta notional|/ADV) present in code (calibratable later, negligible at research scale).
- performance_metrics(result: BacktestResult, cfg: Config) -> dict
    CAGR, annualized vol, Sharpe, Sortino, Calmar/MAR, max drawdown + duration, profit factor, hit rate, avg annual turnover, avg exposure. Plus per-asset P&L attribution.
- deflated_sharpe_ratio(returns: pd.Series, n_trials: int, skew: float, kurtosis: float) -> float
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): corrects observed Sharpe for number of trials and non-normality (skew, kurtosis). Must decrease as n_trials increases.
- validation/splits.py: locked train/test split (train_test_ratio); walk-forward (wf_train_years/wf_test_years/wf_step_years), concatenate test segments.
- validation/purged_cv.py: purged cross-validation + embargo (Lopez de Prado 2018): purging removes train samples overlapping the test window; embargo (cv_embargo) blocks leakage right after the test window.
- reporting/report.py: charts (equity curve, drawdown, exposure over time, per-asset contribution) + metrics table; save PNG/HTML. Presentation only, no calculation logic.

STRATEGY FORMULAS:
- Signal: per asset & lookback L: s_L = sign(P_t/P_{t-L} - 1). s = mean(s_L) in [-1,1]. long_only: s = max(s,0).
- Ex-ante vol: sigma_i = annualized vol of daily returns over vol_window (EWMA recommended), annualize by sqrt(periods_per_year).
- Weighting: w_i = s_i*(asset_vol_target/sigma_i); k = portfolio_vol_target/sigma_p; w = k*w*kelly_fraction; renormalize if sum(|w_i|) > max_gross_leverage.
- Execution: target weights computed at t applied at t+1 (lag). Rebalance at cadence. Turnover between rebalances -> costs.

COST MODEL: cost per trade = cost_bps_per_side * |delta notional|. Optional impact term in code. Mandatory sensitivity test: replay backtest at cost_stress_levels.

VALIDATION PROTOCOL (critical, no shortcuts):
6.1 Locked train/test: last 30% of history = out-of-sample test, untouched until the very end. All dev/tuning on first 70%.
6.2 Walk-forward: rolling windows train wf_train_years, test wf_test_years, step wf_step_years; concatenate test segments for "as-in-production" performance. Stability across windows = robustness; instability = overfitting.
6.3 Purged CV + embargo: naive CV is invalid in finance (temporal overlap). Implement purging + embargo (cv_embargo).
6.4 Multiple-testing correction: keep an n_trials counter (number of configs tested) and feed it to deflated_sharpe_ratio. Optional/advanced: White Reality Check, Hansen SPA.
6.5 Success criterion: retain a variant only if, ON THE LOCKED TEST SET: (a) Deflated Sharpe clearly positive after correction, and (b) walk-forward performance stable (not driven by one period). Otherwise do not deploy and do not re-tune on the test set.

MANDATORY PYTEST TESTS (section 8):
- No look-ahead: on signal, sizing, and engine (removing data after t does not change decision at t).
- Cost application: higher turnover => higher costs => lower equity.
- Engine consistency: weight=1 on one asset, zero costs => equity == that asset's return.
- Sizing math: gross exposure <= cap; realized vol ~ target.
- Metrics exactness: Sharpe/drawdown verified on known analytic cases.
- Data integrity: contract format respected, NaN handled.
=== END SPEC ===
`

const COMMON = `
PROJECT ROOT: ${ROOT}   (create with: mkdir -p, then do ALL work from this directory; package import root is this directory so "import trend_robot.<...>" works when CWD=${ROOT})
PYTHON (use this EXACT venv interpreter — all deps installed): ${PY}
PYTEST (use this EXACT path): ${PYTEST}

ENVIRONMENT CONSTRAINTS (important):
- Yahoo Finance is RATE-LIMITED here (HTTP 429). Do NOT depend on live downloads for any smoke test, verification, or pytest. Tests and offline runs MUST use deterministic SYNTHETIC price data (seeded). The YFinanceProvider must cache to parquet and degrade gracefully when downloads fail (return what it can, never crash the pipeline).
- Reproducibility is mandatory: seed numpy/random from cfg.seed.

RULES:
- Follow the spec arborescence and interface signatures EXACTLY.
- NO market values hard-coded in code — everything flows from config.yaml via the typed Config.
- Before writing any module that depends on earlier modules, READ those module files first to use their REAL signatures (do not assume).
- Pure functions for signal & portfolio (no side effects / no I/O).
- Type annotations + docstrings on public functions.
- Always run a smoke test with ${PY} before declaring done, and report the exact command + output.
`

const BUILD_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    files_written: { type: 'array', items: { type: 'string' }, description: 'absolute or repo-relative paths created/modified' },
    smoke_test: { type: 'string', description: 'exact command(s) run with the venv python and their key output proving the module works' },
    notes: { type: 'string', description: 'design decisions, deviations, anything the verifier should know' },
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
          evidence: { type: 'string', description: 'concrete evidence: command run + observed output, or code citation' },
        },
        required: ['name', 'met', 'evidence'],
      },
    },
    issues: { type: 'array', items: { type: 'string' }, description: 'concrete, actionable problems to fix (empty if passed)' },
    summary: { type: 'string' },
  },
  required: ['passed', 'criteria', 'issues', 'summary'],
}

const TASKS = [
  {
    key: 'T1', title: 'T1 — Skeleton & config',
    build: `T1 — Skeleton & config. Create the FULL arborescence from the spec (all directories with __init__.py, and create the module files needed by later tasks; you may stub the not-yet-implemented modules with a clear placeholder docstring, but config.py must be FULLY implemented now). Implement:
- config.yaml at ${ROOT}/config.yaml containing every parameter from the spec parameter table (incl. seed: 42).
- trend_robot/config.py: a typed @dataclass Config with one field per parameter (correct types: ints, floats, lists, str), a load_config(path) -> Config that parses YAML and VALIDATES (e.g. ratios in (0,1), positive windows, non-empty universe, direction in {long_short, long_only}, rebalance in {daily, weekly, monthly}); raise a clear error on invalid config. Also a set_global_seed(seed) -> None utility seeding numpy + random.
Smoke test: run from ${ROOT}: ${PY} -c "from trend_robot.config import load_config; c=load_config('config.yaml'); print(c)".`,
    criteria: `Acceptance (spec section 9 / T1): Config loads from config.yaml, is typed and validated. Verify: the arborescence exists exactly per spec; config.yaml contains ALL parameters; load_config returns a typed Config with correct values; validation rejects at least one malformed config (test by feeding a bad value). No market values hard-coded in config.py.`,
  },
  {
    key: 'T2', title: 'T2 — Data layer',
    build: `T2 — Data layer (spec 3.1). Implement:
- trend_robot/data/provider.py: DataProvider Protocol (typing.Protocol) with get_prices(tickers, start, end) -> pd.DataFrame per contract; plus a disk cache helper (parquet) — e.g. a CachedProvider wrapper or cache utility keyed by (tickers, start, end) writing to a cache dir, so identical calls do NOT re-download.
- trend_robot/data/yfinance_provider.py: YFinanceProvider implementing the protocol using yfinance with ADJUSTED close, parquet cache, and GRACEFUL degradation on failure/rate-limit (never raise on 429; return available data or empty frame with a logged warning). Align calendars across tickers; preserve NaN (no silent forward-fill); never return data after the end date.
- trend_robot/data/synthetic_provider.py: SyntheticProvider implementing the SAME protocol — deterministic seeded geometric-brownian-motion price series for given tickers/date range (used for offline runs & tests since Yahoo is rate-limited). Must honor the same contract (tz-naive dates, columns=tickers, no future data).
Smoke test with SYNTHETIC provider only (do NOT hit yahoo): generate prices for the default universe over a date range, print shape/head, and demonstrate cache hit avoids recompute/redownload (use a counting stub provider wrapped by the cache).`,
    criteria: `Acceptance (spec 3.1): (1) get_prices returns the exact contract (date index tz-naive, columns=tickers, ADJUSTED closes). (2) The cache prevents any re-download on an identical call (verify e.g. by counting underlying fetch calls with a counting stub provider — do not depend on yahoo). (3) A test confirms NO look-ahead: no value dated after the end date. (4) NaN preserved, not silently filled. Run targeted checks with ${PY} (synthetic provider + a counting stub for the cache). Yahoo must NOT be required.`,
  },
  {
    key: 'T3', title: 'T3 — TSMOM signal',
    build: `T3 — TSMOM signal (spec 3.2). Depends on T2. READ trend_robot/config.py and the data provider first. Implement trend_robot/signals/tsmom.py: tsmom_signal(prices, lookbacks, direction='long_short') -> pd.DataFrame. Pure function. For each lookback L: s_L = sign(P_t/P_{t-L} - 1); final signal = mean over lookbacks, in [-1,1]; long_only -> max(s,0). No look-ahead. Handle insufficient history (NaN/0). Smoke test with synthetic data: a monotone uptrend asset -> signal -> +1; downtrend -> -1.`,
    criteria: `Acceptance (spec 3.2): (1) signal strictly within [-1,1]. (2) monotone-up asset -> +1, monotone-down -> -1 (demonstrate numerically). (3) No look-ahead: signal at t is unchanged if data after t is removed (demonstrate by truncating and comparing). Run checks with ${PY} on synthetic data.`,
  },
  {
    key: 'T4', title: 'T4 — Sizing & portfolio',
    build: `T4 — Sizing & portfolio (spec 3.3). Depends on T3. READ config.py and tsmom.py first. Implement trend_robot/portfolio/sizing.py: target_weights(signals, returns, cfg) -> pd.DataFrame following the 5 steps (ex-ante vol via vol_window EWMA annualized; raw weight = s*(asset_vol_target/sigma); portfolio vol scaling k=portfolio_vol_target/sigma_p; apply kelly_fraction; cap gross exposure <= max_gross_leverage). Pure function. Zero weight when signal=0 or vol undefined. Smoke test on synthetic data: print gross exposure over time and an estimate of realized portfolio vol vs target.`,
    criteria: `Acceptance (spec 3.3): (1) gross exposure sum(|w_i|) NEVER exceeds max_gross_leverage (demonstrate the max over time). (2) ex-post realized portfolio vol is in the right ballpark of portfolio_vol_target (order of magnitude). (3) zero weights when signal is 0 or vol undefined (insufficient history). Run checks with ${PY} on synthetic data.`,
  },
  {
    key: 'T5', title: 'T5 — Costs & backtest engine',
    build: `T5 — Cost model + backtest engine (spec 3.4, 4, 5). Depends on T4. READ prior modules first. Implement:
- trend_robot/backtest/costs.py: cost = cost_bps_per_side * |delta notional|; include the optional sqrt impact term in code (impact = c*sigma_i*sqrt(|delta notional|/ADV)), disabled/negligible by default but present.
- trend_robot/backtest/engine.py: @dataclass BacktestResult (equity: pd.Series, weights: pd.DataFrame, turnover: pd.Series, trades: pd.DataFrame[date,asset,delta_weight,cost]); run_backtest(prices, target_weights, cfg) -> BacktestResult. Target weights SHIFTED >=1 bar before application (no look-ahead). Rebalance at cfg.rebalance cadence. Apply costs on turnover. Mark to market.
Smoke test on synthetic data: run a full backtest, print equity head/tail, total turnover, total cost.`,
    criteria: `Acceptance (spec 3.4): (1) weights shifted >=1 bar (engine no-look-ahead — demonstrate the shift). (2) costs reduce equity monotonically with turnover (compare two cost levels: higher cost => lower final equity). (3) with zero costs and constant weight=1 on a SINGLE asset, equity replicates that asset's cumulative return within float tolerance. Run all three checks with ${PY} on synthetic data and show numbers.`,
  },
  {
    key: 'T6', title: 'T6 — Metrics & Deflated Sharpe',
    build: `T6 — Metrics + Deflated Sharpe (spec 3.5, 7). Depends on T5. READ engine/BacktestResult first. Implement:
- trend_robot/metrics/performance.py: performance_metrics(result, cfg) -> dict with CAGR, annualized vol, Sharpe, Sortino, Calmar/MAR, max drawdown + duration, profit factor, hit rate, avg annual turnover, avg exposure; AND per-asset P&L attribution (e.g. a dict or DataFrame).
- trend_robot/metrics/deflated_sharpe.py: deflated_sharpe_ratio(returns, n_trials, skew, kurtosis) -> float per Bailey & Lopez de Prado 2014 (expected max Sharpe under multiple trials + non-normality adjustment).
Smoke test: verify Sharpe & max drawdown on a known analytic case (e.g. constant return series, or a hand-computed small series) and show DSR decreasing as n_trials grows.`,
    criteria: `Acceptance (spec 3.5): (1) Sharpe/Sortino/drawdown verified against known analytic cases (show a hand-checkable example with expected vs computed). (2) deflated_sharpe_ratio DECREASES as n_trials increases (show the monotonic decrease). (3) per-asset P&L attribution is available and sums coherently. Run with ${PY}.`,
  },
  {
    key: 'T7', title: 'T7 — run_research.py',
    build: `T7 — run_research.py (spec 9/T7). End-to-end backtest on the default universe + report. READ all prior modules. Implement ${ROOT}/run_research.py: load config -> set seed -> get prices (PREFER YFinanceProvider with cache, but FALL BACK to SyntheticProvider if download is empty/fails, logging clearly which data source was used since Yahoo is rate-limited here) -> compute returns -> tsmom_signal -> target_weights -> run_backtest -> performance_metrics -> reporting.report (save equity/drawdown/exposure/contribution charts + metrics table to an output dir). Must be REPRODUCIBLE (seeded). Also implement trend_robot/reporting/report.py (presentation only). Smoke test: run from ${ROOT}: ${PY} run_research.py and show it produces an equity curve + metrics table; run it TWICE and confirm identical metrics (reproducible).`,
    criteria: `Acceptance (spec 9/T7): produces a reproducible equity curve and a metrics table. Verify by running run_research.py end-to-end (must NOT crash even though Yahoo is 429 — fallback to synthetic), confirming it emits a metrics table + saved chart files, and that two consecutive runs give identical metrics (reproducibility). Report which data source was used.`,
  },
  {
    key: 'T8', title: 'T8 — pytest suite',
    build: `T8 — pytest suite (spec 8). READ all modules. Create tests/ covering EVERY section-8 test, each using SYNTHETIC/deterministic data (never yahoo):
- No look-ahead: signal, sizing, AND engine (removing data after t doesn't change decision at t / weights shifted).
- Cost application: higher turnover => higher costs => lower equity.
- Engine consistency: weight=1 on one asset, zero costs => equity == asset return.
- Sizing math: gross exposure <= cap; realized vol ~ target.
- Metrics exactness: Sharpe/drawdown on known analytic cases.
- Data integrity: contract format respected, NaN handled.
Run the full suite from ${ROOT}: ${PYTEST} -q. Smoke test = the full pytest run output.`,
    criteria: `Acceptance (spec 9/T8): ALL tests pass, including the no-look-ahead tests, with the venv pytest, using synthetic data (no network). Verify by running from ${ROOT}: ${PYTEST} -q and confirming every section-8 category is covered AND green. If any fail, that's a fail.`,
  },
  {
    key: 'T9', title: 'T9 — Validation harness',
    build: `T9 — Validation harness (spec 6). Depends on prior. Implement:
- trend_robot/validation/splits.py: locked train/test split (last 30% out-of-sample per train_test_ratio); walk-forward generator (wf_train_years/wf_test_years/wf_step_years) yielding (train_idx, test_idx) windows; a helper to concatenate test segments into one out-of-sample track.
- trend_robot/validation/purged_cv.py: PurgedKFold-style CV with purging (drop train samples overlapping the test window) + embargo (cv_embargo fraction) per Lopez de Prado 2018.
- A trials counter mechanism (e.g. a small class/function) that counts configurations tested and is fed into deflated_sharpe_ratio as n_trials.
Add pytest tests for: split sizes (70/30), walk-forward window coverage/non-overlap of test segments, purging actually removes overlapping train samples, embargo removes the right count. Smoke test: run from ${ROOT}: ${PYTEST} -q on the new validation tests and a quick demo of walk-forward windows on synthetic data.`,
    criteria: `Acceptance (spec 6): (1) locked 70/30 split correct and test set is the LAST 30%. (2) walk-forward produces correct rolling windows; concatenated test track is contiguous/non-overlapping. (3) purged CV genuinely purges overlapping train samples AND applies embargo (cv_embargo) — demonstrate counts. (4) n_trials counter exists and feeds deflated_sharpe_ratio. Verify with ${PY} / ${PYTEST} on synthetic data.`,
  },
  {
    key: 'T10', title: 'T10 — Cost stress & final report',
    build: `T10 — Cost sensitivity tests + final validation report (spec 5, 6.5). Depends on all prior. Implement:
- A cost sensitivity routine that replays the backtest at cfg.cost_stress_levels ([5,10] bps) plus the base level, collecting metrics at each (a dedicated module e.g. trend_robot/validation/stress.py, or extend reporting/run_research).
- A final validation report that, ON THE LOCKED TEST SET: computes Deflated Sharpe (after correction with the trials counter) and walk-forward stability, then evaluates the section-6.5 success criterion (retain only if DSR clearly positive AND walk-forward stable). The report must clearly STATE the verdict but must NOT auto-deploy or re-tune on the test set (human-judgment note per spec section 11).
Wire this into run_research.py (e.g. a --validate flag) or a run_validation.py entry point. Smoke test: run the cost stress + final validation end-to-end on the (synthetic-fallback) data and print the stress table + the section-6.5 verdict.`,
    criteria: `Acceptance (spec 9/T10 + 6.5): (1) backtest replayed at all cost_stress_levels and base, producing a comparison table (higher cost => lower performance). (2) Deflated Sharpe + walk-forward stability computed ON THE LOCKED TEST SET and the section-6.5 criterion explicitly evaluated with a clear retain/reject verdict (and a note that it's human judgment, no auto-deploy). Verify by running the validation entry point with ${PY} end-to-end without crashing.`,
  },
]

function buildPrompt(task) {
  return `You are implementing one task of a multi-task software build. Implement it FULLY and correctly, then smoke-test it.\n\n${COMMON}\n\n${SPEC}\n\nYOUR TASK NOW:\n${task.build}\n\nDeliver real, working, typed Python following the spec exactly. Run the smoke test with the venv python before finishing. Return the structured result (files_written, smoke_test command+output, notes).`
}

function verifyPrompt(task) {
  return `You are an ADVERSARIAL verifier. Do NOT trust the builder's claims — independently check the acceptance criteria by READING the code and RUNNING real checks with the venv python/pytest. Default to met=false unless you have concrete evidence (command output or code citation). Use synthetic data only (Yahoo is rate-limited).\n\n${COMMON}\n\n${SPEC}\n\nTASK UNDER REVIEW: ${task.title}\n\n${task.criteria}\n\nFor each acceptance criterion: run an actual check (write a tiny script and execute it with ${PY}, or run ${PYTEST}), capture output, and record met + evidence. Set passed=true ONLY if every criterion is met. List concrete, actionable issues for anything not met.`
}

function fixPrompt(task, verdict) {
  return `You are fixing a partially-failing task. The adversarial verifier found these UNMET issues — fix them at the source (edit the real files), do not paper over:\n\nISSUES:\n${verdict.issues.map((s, i) => `${i + 1}. ${s}`).join('\n')}\n\nFAILED CRITERIA EVIDENCE:\n${verdict.criteria.filter(c => !c.met).map(c => `- ${c.name}: ${c.evidence}`).join('\n')}\n\n${COMMON}\n\n${SPEC}\n\nORIGINAL TASK:\n${task.build}\n\nFix the issues, re-run the smoke test with the venv python, and return the structured result.`
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
  `Final whole-project integration & completeness check for the TSMOM robot at ${ROOT}.\n\n${COMMON}\n\n${SPEC}\n\nDo ALL of the following and report findings honestly:\n1) Run the FULL test suite from ${ROOT}: ${PYTEST} -q  (report pass/fail counts).\n2) Run the end-to-end research pipeline from ${ROOT}: ${PY} run_research.py (must not crash; report which data source was used and that a metrics table + charts were produced).\n3) Run the validation/cost-stress entry point end-to-end (the T10 deliverable) and confirm the section-6.5 verdict prints.\n4) Completeness audit vs spec: confirm EVERY module in the arborescence exists, every key interface signature matches the spec, NO market values are hard-coded outside config.yaml (grep for suspicious literals), zero-look-ahead is enforced, and all section-8 test categories exist. List anything MISSING or DEVIATING.\n`,
  { label: 'integration', phase: 'Integration check', schema: VERIFY_SCHEMA }
)

return {
  tasks: results,
  integration: integration ? { passed: integration.passed, issues: integration.issues, summary: integration.summary, criteria: integration.criteria } : null,
}
