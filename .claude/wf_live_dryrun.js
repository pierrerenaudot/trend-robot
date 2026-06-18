export const meta = {
  name: 'tsmom-live-dryrun',
  description: 'Build the live/ package + run_live.py --dry-run (Alpaca-paper ready) for the TSMOM robot, with adversarial verification',
  phases: [
    { title: 'Build live package + dry-run' },
    { title: 'Adversarial verify' },
    { title: 'Integration check' },
  ],
}

const ROOT = '/Users/pierre.renaudot/test/trend_robot'
const PY = '/Users/pierre.renaudot/test/.venv/bin/python'
const PYTEST = '/Users/pierre.renaudot/test/.venv/bin/pytest'

const CONTEXT = `
=== TSMOM ROBOT — PAPER-TRADING DRY-RUN MILESTONE ===
The TSMOM research robot is already built & validated (tasks T1-T10) at ${ROOT}. We are now adding a PAPER-TRADING DRY-RUN as an ENGINEERING dress-rehearsal (NOT a deploy signal — the spec defers live/paper trading in section 10, and the current section-6.5 verdict is REJECT). Chosen broker: ALPACA (paper). First milestone: a run_live.py --dry-run that COMPUTES and DISPLAYS today's orders WITHOUT sending anything, structured so wiring the real Alpaca API later is just a plug-in.

PYTHON (venv, all deps incl. alpaca-py 0.43.4 installed): ${PY}
PYTEST: ${PYTEST}
Work from ${ROOT} (package import root; "import trend_robot.<...>" and "import run_research" work when CWD=${ROOT}).

REUSE THESE EXISTING PIECES (parity research/production — READ them first to confirm):
- trend_robot/config.py: Config (frozen dataclass), load_config(path)->Config, set_global_seed(seed). Fields incl. universe, lookbacks, direction, vol_window, asset_vol_target, portfolio_vol_target, max_gross_leverage, kelly_fraction, cost_bps_per_side, initial_capital, periods_per_year, seed.
- trend_robot/signals/tsmom.py: tsmom_signal(prices, lookbacks, direction='long_short')->pd.DataFrame (pure).
- trend_robot/portfolio/sizing.py: target_weights(signals, returns, cfg)->pd.DataFrame (rows=dates, cols=assets; pure). The LAST row is today's target book.
- trend_robot/data/provider.py: DataProvider Protocol get_prices(tickers,start,end)->pd.DataFrame (tz-naive index, cols=tickers, adjusted closes, NaN gaps, never data after end); CachedProvider(provider, cache_dir).
- trend_robot/data/yfinance_provider.py: YFinanceProvider().cached(cache_dir) ; trend_robot/data/synthetic_provider.py: SyntheticProvider(seed=...).
- run_research.py: _date_window(history_years, end)->(start,end) ; _load_prices(cfg, start, end, cache_dir, *, prefer_yfinance=False)->(prices_df, data_source_str). REUSE these to fetch prices "as of" a date (yfinance when --live, synthetic otherwise). Yahoo is sometimes 429 — synthetic fallback must keep dry-run working offline.

ANTI-LOOK-AHEAD DISCIPLINE (must hold in live, mirrors the backtest's shift(1)): the target book is computed from closes UP TO AND INCLUDING asof; the resulting orders are what you would place AFTER asof (next session). The function that computes the target must use only prices with index <= asof.
=== END CONTEXT ===
`

const BUILD_SPEC = `
Build the following. Real, typed, docstringed Python. Pure where noted. No market values hard-coded (flow from Config). Dry-run must require NO Alpaca credentials and NO network (synthetic fallback).

1) trend_robot/live/__init__.py — export the public API.

2) trend_robot/live/broker.py
   - Dataclasses: AccountSnapshot(equity: float, cash: float, buying_power: float); Position(symbol: str, qty: float, avg_price: float, market_value: float); OrderIntent(symbol: str, side: str ['buy'|'sell'], qty: float, est_price: float, notional: float, target_weight: float, current_weight: float, reason: str); OrderResult(symbol: str, side: str, qty: float, status: str, broker_order_id: str | None).
   - Broker (typing.Protocol): get_account()->AccountSnapshot; get_positions()->dict[str, Position]; submit_order(intent: OrderIntent)->OrderResult.
   - LocalPaperBroker (NO external deps): constructed with starting equity (float) and optional initial positions (dict[str,float] qty or dict[str,Position]); get_account/get_positions from in-memory state; submit_order records the intent and returns a simulated OrderResult(status='accepted_simulated', broker_order_id=None). This is the broker used for the dry-run.
   - AlpacaBroker (implements Broker against alpaca-py PAPER endpoint): LAZY-import alpaca inside __init__/methods so the module imports fine without creds. Read keys from env (try APCA_API_KEY_ID/APCA_API_SECRET_KEY then ALPACA_API_KEY/ALPACA_SECRET_KEY); raise a clear, actionable error ONLY when actually constructed without keys (not at import). Use paper=True / paper base url. Map Alpaca account->AccountSnapshot, positions->dict[str,Position]; submit_order builds a MarketOrderRequest (alpaca.trading.requests) and submits via TradingClient. This path is NOT exercised in the dry-run but must be import-clean and structurally correct.

3) trend_robot/live/live_data.py
   - prices_asof(cfg, asof: str, cache_dir, *, prefer_yfinance: bool=False, history_years: int=15)->tuple[pd.DataFrame,str]: compute window via run_research._date_window(history_years, asof) and fetch via run_research._load_prices(...); guarantee the returned frame has NO rows after asof. Return (prices, data_source).
   - last_prices(prices: pd.DataFrame)->pd.Series: last valid (non-NaN) close per column.

4) trend_robot/live/target.py
   - compute_target_book(cfg, prices: pd.DataFrame, asof: str | None=None)->pd.Series: returns=prices.pct_change(fill_method=None); signals=tsmom_signal(prices, cfg.lookbacks, direction=cfg.direction); weights=target_weights(signals, returns, cfg); take the last row with index <= asof (or the very last row if asof is None) and return it as a Series indexed by symbol (today's target weights). MUST be no-look-ahead: only uses prices up to asof. Pure (no I/O).

5) trend_robot/live/executor.py
   - plan_orders(target_w: pd.Series, positions: dict[str,Position], last_px: pd.Series, equity: float, cfg: Config, *, min_trade_notional: float=0.0, allow_fractional: bool=False)->list[OrderIntent]:
       For each symbol in the union of (target_w index) and (current positions):
         target_notional = target_w.get(sym,0.0)*equity ; cur_qty = position qty or 0 ; price = last_px[sym]
         cur_notional = cur_qty*price ; cur_weight = cur_notional/equity (if equity>0 else 0)
         delta_notional = target_notional - cur_notional
         skip with reason 'below_min_trade' if abs(delta_notional) < min_trade_notional
         qty_delta = delta_notional/price ; if not allow_fractional: qty_delta = truncate toward zero to whole shares ; skip (reason 'rounds_to_zero') if 0
         side = 'buy' if qty_delta>0 else 'sell' ; build OrderIntent with abs(qty), est_price=price, notional=abs(qty_delta)*price, target_weight, current_weight, reason='rebalance' (or 'close' if target_w==0 and had a position)
       Handle held symbols NOT in target (target weight 0) -> sell to flat.
       Guard: gross = sum(|target_w|); raise a clear error if gross > cfg.max_gross_leverage + 1e-9 (sizing should already enforce this).
       Return the list (deterministic order, e.g. sorted by symbol).
   - A helper summarize_plan(intents, target_w, equity, cfg)->dict: n_orders, total_buy_notional, total_sell_notional, gross_exposure (sum|target_w|), est_cost (cost_bps_per_side/1e4 * sum(notional)).

6) trend_robot/live/state.py
   - save_run_state(state_dir, record: dict)->Path (JSON; filename keyed by asof, e.g. live_state_<asof>.json; also update a 'latest.json'); load_last_state(state_dir)->dict|None; has_run_for(state_dir, asof)->bool. Records are JSON-serializable (convert Series/np types). Python datetime is allowed here for a generated_at timestamp.

7) run_live.py (at ${ROOT})
   - argparse: --config (default project config.yaml), --asof (YYYY-MM-DD; default today), --equity (float; default = cfg.initial_capital), --positions (optional JSON file {symbol: qty}; default none = flat book), --dry-run / --no-dry-run (DEFAULT: dry-run TRUE), --live (prefer yfinance data; else synthetic), --broker {local,alpaca} (default local), --cache-dir (default ./.cache), --state-dir (default ./live_state), --min-trade-notional (float, default 0), --allow-fractional (flag), --history-years (default 15), --log-level.
   - Flow: load_config -> set_global_seed(cfg.seed) -> resolve asof -> prices_asof(...) -> compute_target_book -> last_prices -> pick broker (dry-run uses LocalPaperBroker seeded with --equity and --positions; if --broker alpaca AND --no-dry-run, use AlpacaBroker) -> get_account/get_positions -> plan_orders -> PRINT a clear, aligned PREVIEW TABLE (symbol, target_w, current_w, side, qty, est_price, notional, reason) + a summary line (n orders, gross exposure, buy/sell notional, est cost) + a banner that this is a DRY-RUN / PAPER / not a deploy signal -> save_run_state.
   - SAFETY: in --dry-run, NEVER call submit_order (just preview). Only when --no-dry-run AND --broker alpaca do you actually submit (guard: refuse --no-dry-run with --broker local). Provide a main(argv=None) callable for tests. Dry-run must run with NO network and NO alpaca creds (synthetic data).

8) tests/tests for the above — tests/test_live.py (synthetic/deterministic only, NEVER network/alpaca):
   - executor math: known target_w + current positions + prices + equity -> correct sides/qty/notional; below-min-trade skipped; symbol held but dropped from target -> sell-to-flat; integer rounding (truncate) vs allow_fractional; leverage guard raises when gross>cap.
   - target book no-look-ahead: compute_target_book at asof is unchanged when prices after asof are removed (truncate-and-compare).
   - run_live.main(['--dry-run', ...]) end-to-end on synthetic: produces a non-empty plan from a flat book, writes state, sends NOTHING (assert submit_order not called / LocalPaperBroker has zero submissions), no network, no alpaca import required.
   - state.py round-trip + has_run_for idempotence.
   - AlpacaBroker: the class/module imports without creds; constructing it without env keys raises a clear error (assert the error message), but importing must not.
   Run: cd ${ROOT} && ${PYTEST} tests/test_live.py -q  (all green). Also smoke: cd ${ROOT} && ${PY} run_live.py --dry-run  (synthetic; show the preview table).
`

const BUILD_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    files_written: { type: 'array', items: { type: 'string' } },
    smoke_test: { type: 'string', description: 'exact commands run (dry-run + pytest) + key output' },
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
          evidence: { type: 'string' },
        },
        required: ['name', 'met', 'evidence'],
      },
    },
    issues: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['passed', 'criteria', 'issues', 'summary'],
}

const CRITERIA = `
ACCEPTANCE CRITERIA (check each by RUNNING real code with the venv; default met=false without evidence):
1. trend_robot/live/ package exists with broker.py, live_data.py, target.py, executor.py, state.py, __init__.py — all import cleanly with ${PY}.
2. run_live.py --dry-run runs end-to-end on SYNTHETIC data with NO network and NO Alpaca creds, prints an order PREVIEW TABLE + summary + dry-run banner, and SENDS NOTHING. (Prove no order is submitted in dry-run.)
3. Executor math is correct: orders move from current positions toward target notional (target_w*equity); correct side/qty/notional; below-min-trade skipped; held-but-dropped symbol -> sell to flat; gross-leverage guard works.
4. No look-ahead: compute_target_book at asof is unchanged when prices after asof are removed (truncate-and-compare).
5. AlpacaBroker imports without creds (lazy alpaca import) and raises a clear error only when constructed without keys; it is structurally correct (builds MarketOrderRequest / TradingClient with paper endpoint).
6. tests/test_live.py exists, covers the above, and PASSES via ${PYTEST} tests/test_live.py -q. The pre-existing suite must still import/collect cleanly (no breakage of T1-T10).
7. Reuse: target/sizing/signal/providers are reused (not reimplemented); no market values hard-coded outside config.
8. (If Yahoo reachable) run_live.py --dry-run --live also works (real prices) and degrades to synthetic if 429.
`

const build = await (async () => {
  phase('Build live package + dry-run')
  return agent(
    `You are building the paper-trading DRY-RUN milestone for the TSMOM robot. Implement it fully, then smoke-test (dry-run + pytest).\n\n${CONTEXT}\n\nWHAT TO BUILD:\n${BUILD_SPEC}\n\nREAD the reused modules first. Run the smoke tests with the venv python/pytest before finishing. Return the structured result.`,
    { label: 'build:live', phase: 'Build live package + dry-run', schema: BUILD_SCHEMA }
  )
})()

phase('Adversarial verify')
let verdict = await agent(
  `You are an ADVERSARIAL verifier. Independently check every criterion by READING the code and RUNNING real checks with the venv python/pytest. Default met=false without concrete evidence (command output / code citation). Use synthetic data; do NOT require network or Alpaca creds.\n\n${CONTEXT}\n\n${CRITERIA}\n\nFor each criterion run an actual check, capture output, record met + evidence. passed=true ONLY if every criterion is met. List concrete actionable issues for anything unmet.`,
  { label: 'verify:live', phase: 'Adversarial verify', schema: VERIFY_SCHEMA }
)

let attempt = 0
while (verdict && !verdict.passed && attempt < 2) {
  attempt++
  log(`live: verify FAILED (attempt ${attempt}) — fixing ${verdict.issues.length} issue(s): ${verdict.issues.slice(0, 3).join(' | ')}`)
  await agent(
    `Fix these UNMET issues at the source (edit real files, do not paper over):\n\nISSUES:\n${verdict.issues.map((s, i) => `${i + 1}. ${s}`).join('\n')}\n\nFAILED CRITERIA EVIDENCE:\n${verdict.criteria.filter(c => !c.met).map(c => `- ${c.name}: ${c.evidence}`).join('\n')}\n\n${CONTEXT}\n\nWHAT TO BUILD (reference):\n${BUILD_SPEC}\n\nFix, re-run the dry-run + pytest, and return the structured result.`,
    { label: `fix:live#${attempt}`, phase: 'Adversarial verify', schema: BUILD_SCHEMA }
  )
  verdict = await agent(
    `Re-verify the paper-trading dry-run milestone adversarially (same criteria). Run real checks with the venv.\n\n${CONTEXT}\n\n${CRITERIA}\n\nRecord met + evidence per criterion; passed only if all met.`,
    { label: `verify:live#${attempt}`, phase: 'Adversarial verify', schema: VERIFY_SCHEMA }
  )
}

phase('Integration check')
const integration = await agent(
  `Final integration check for the TSMOM paper-trading dry-run at ${ROOT}. Be honest and adversarial.\n\n${CONTEXT}\n\nDo ALL and report:\n1) Clean any stray scratch files at the project root (e.g. _smoke_*.py / tmp_*.py) without touching real source/tests.\n2) Run: cd ${ROOT} && ${PYTEST} tests/test_live.py -q  (report counts). Also confirm the existing suite still COLLECTS without import errors: ${PYTEST} --collect-only -q | tail -3.\n3) Run: cd ${ROOT} && ${PY} run_live.py --dry-run  (synthetic) and paste the preview table + summary; confirm NOTHING was submitted and state was written under ./live_state.\n4) If Yahoo is reachable, also run ${PY} run_live.py --dry-run --live and report data source; otherwise note the 429/synthetic fallback.\n5) Completeness audit: live/ package present (broker/live_data/target/executor/state/__init__), AlpacaBroker import-clean without creds, reuse of signal/sizing/providers confirmed, no hard-coded market values, no-look-ahead in compute_target_book, dry-run sends nothing. List anything MISSING or DEVIATING and whether the milestone meets its acceptance criteria.`,
  { label: 'integration', phase: 'Integration check', schema: VERIFY_SCHEMA }
)

return {
  build: build ? { files: build.files_written, smoke: build.smoke_test } : null,
  verify: verdict ? { passed: verdict.passed, issues: verdict.issues, summary: verdict.summary } : null,
  integration: integration ? { passed: integration.passed, issues: integration.issues, summary: integration.summary, criteria: integration.criteria } : null,
}
