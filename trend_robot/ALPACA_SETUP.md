# Alpaca paper-trading — home setup guide

This guide walks through running the TSMOM robot's paper-trading track on your
**personal Mac**, unattended, on a monthly cadence. It is **PAPER ONLY** — no
real money is ever at risk. The point of this track is to accrue *genuinely*
out-of-sample (post-decision) bars so the robot can later produce a pristine
section-6.5 forward read.

> **NOT financial advice.** This is a research engineering exercise. Paper
> trading simulates fills; it does not place real-money orders and makes no
> recommendation to do so.

---

## 1. Create an Alpaca PAPER account & get paper API keys

1. Sign up at <https://alpaca.markets/> and log in.
2. In the dashboard, switch to **Paper Trading** (there is a live/paper toggle).
   Confirm the environment reads *Paper*.
3. Under **Your API Keys** (paper), click **Generate New Key**.
4. Copy the **Key ID** and the **Secret Key**. The secret is shown only once —
   store it in your password manager.

These are *paper* keys. The robot always constructs the broker with
`paper=True`; it never targets the live trading endpoint.

---

## 2. Install on your Mac

```bash
# Clone your copy of the repo, then:
cd /path/to/trend_robot          # the directory containing run_live.py + config.yaml

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # includes alpaca-py (the paper broker)
```

The `--dry-run` preview works with **no** Alpaca install and **no** network
(synthetic-price fallback). `alpaca-py` is only needed to actually talk to the
paper broker.

> Throughout this guide, replace `/path/to/trend_robot` with your real project
> path and `/path/to/trend_robot/.venv/bin/python` with your venv's Python.

---

## 3. Set the paper API credentials as environment variables

The broker reads these (in order): `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`
(falling back to `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`).

For an interactive shell, add to `~/.zshrc` (macOS default shell):

```bash
export APCA_API_KEY_ID="PK................"      # your paper Key ID
export APCA_API_SECRET_KEY="................"    # your paper Secret
```

Then `source ~/.zshrc`. For the scheduled job, the keys are passed explicitly in
the launchd plist / cron entry below (a launchd job does **not** read `~/.zshrc`).

> Never commit your keys. `.gitignore` already excludes local state; keep keys
> out of the repo.

---

## 4. Connectivity test (submits nothing)

Confirm the keys work and the account is reachable. `--status` reads the account
and positions, prints a small table, and **submits nothing**:

```bash
python run_live.py --status --broker alpaca
```

You should see the `BROKER STATUS` banner with your paper equity, cash, buying
power, and any open positions. If credentials are missing you get a clear error
naming the env vars to set.

---

## 5. Dry-run preview (no orders, fully offline-capable)

See exactly what the robot *would* trade today, without sending anything:

```bash
python run_live.py --dry-run --live
```

* `--dry-run` is the **default** — nothing is submitted, ever.
* `--live` prefers cached Yahoo prices (falls back to synthetic if unavailable).
* The preview prints the order table, a summary line, and a DRY-RUN banner.

Run this first whenever you change anything.

---

## 6. Go live (paper) — safe to run daily

```bash
python run_live.py --no-dry-run --broker alpaca --live
```

This is the only command that submits. It is engineered to be **safe to run
every day** even though the strategy rebalances **monthly**:

* **Cadence + idempotence guard.** The runner computes a *period key* for the
  rebalance cadence (`YYYY-MM` for monthly). It records the period it last
  submitted for in the run state. If you run it again **in the same month**, it
  **skips** the submission (it still prints the preview and writes state) — so a
  daily-scheduled job trades **once per month**. A new month submits again
  automatically. Use `--force` to override the guard deliberately.
* **Config-drift guard.** Before a live submit, the runner verifies the running
  `config.yaml` still matches the frozen pre-registered variant in
  `decision_record.json`. If a strategy-defining field drifted, it **refuses**
  the live submit (we only paper-trade the pre-registered strategy the forward
  hold-out will judge). A dry-run only warns.
* **Market-open check.** If the market is closed, it warns that DAY orders will
  **queue for the next session** (it does not hard-block).
* **Result capture.** Every submitted order's broker id and status is recorded
  in the run state under `live_state/`.

---

## 7. Scheduling on macOS

Run **daily on weekdays**; the per-period guard ensures exactly **one trade per
month** for the monthly cadence. The scheduled command must:

* use the **absolute path to the venv Python**, and
* set the **working directory to the project root** (so it finds `config.yaml`,
  `decision_record.json`, `.cache/` and `live_state/`).

### Option A — launchd (recommended on macOS)

Create `~/Library/LaunchAgents/com.trendrobot.paper.plist`. Adjust the paths and
paste your **paper** keys into `EnvironmentVariables`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.trendrobot.paper</string>

    <!-- Absolute venv python + the script. -->
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/trend_robot/.venv/bin/python</string>
        <string>/path/to/trend_robot/run_live.py</string>
        <string>--no-dry-run</string>
        <string>--broker</string>
        <string>alpaca</string>
        <string>--live</string>
    </array>

    <!-- CWD must be the project root. -->
    <key>WorkingDirectory</key>
    <string>/path/to/trend_robot</string>

    <!-- Paper credentials (launchd does NOT read ~/.zshrc). -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>APCA_API_KEY_ID</key>
        <string>PK................</string>
        <key>APCA_API_SECRET_KEY</key>
        <string>................</string>
    </dict>

    <!-- Run daily, Mon–Fri at 15:30 local (an example time). -->
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>30</integer></dict>
    </array>

    <key>StandardOutPath</key>
    <string>/path/to/trend_robot/live_state/launchd.out.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/trend_robot/live_state/launchd.err.log</string>
</dict>
</plist>
```

Load / unload it:

```bash
launchctl load   ~/Library/LaunchAgents/com.trendrobot.paper.plist
launchctl unload ~/Library/LaunchAgents/com.trendrobot.paper.plist   # kill switch
launchctl list | grep trendrobot                                     # is it loaded?
```

> Note on `StartCalendarInterval`: if the Mac is asleep at the scheduled time,
> launchd runs the job once at the next wake. The per-period guard makes that
> harmless — at most one trade per month either way.

### Option B — cron (alternative)

`crontab -e`, then add (one line). cron also does not read `~/.zshrc`, so set the
keys inline and `cd` to the project root first:

```cron
30 15 * * 1-5 cd /path/to/trend_robot && APCA_API_KEY_ID="PK................" APCA_API_SECRET_KEY="................" /path/to/trend_robot/.venv/bin/python run_live.py --no-dry-run --broker alpaca --live >> /path/to/trend_robot/live_state/cron.log 2>&1
```

(`1-5` = Monday–Friday; `30 15` = 15:30 local time.) On recent macOS, grant the
`cron`/`bash` binary **Full Disk Access** in *System Settings → Privacy &
Security* if the job cannot read the project directory.

---

## 8. How this feeds the pristine forward hold-out

The pre-registered decision (`decision_record.json`) froze the chosen variant
(`direction=long_only`, `rebalance=monthly`) and a **decision date**. Only bars
dated **strictly after** that date are genuinely out-of-sample — every earlier
window was peeked at during model selection.

Each monthly paper run accrues fresh post-decision bars. After roughly **252
trading days** (about one year) of accrual, run the first pristine section-6.5
forward read:

```bash
python run_holdout.py --live
```

Until enough forward bars exist, that read is **expected to REJECT / be
inconclusive** — that is the honest state of a young hold-out, not a failure.

---

## 9. Safety summary

* **Paper only.** The broker is always constructed with `paper=True`; no real
  money is involved. This is not financial advice.
* **Kill switch.** Stop the scheduler (`launchctl unload …` or remove the cron
  line). The robot also **never** submits unless you explicitly pass
  `--no-dry-run`; the default is a harmless dry-run preview.
* **Idempotent.** Running the live command repeatedly within a month does not
  re-trade — it trades at most once per rebalance period.
* **Drift-protected.** A live submit is refused if the running config no longer
  matches the frozen pre-registered strategy, protecting the pristine forward
  read from accidentally trading a different variant.

---

## 10. Serverless alternative — GitHub Actions (no machine to keep on)

Instead of (or in addition to) your Mac, the monthly paper run can fire from
**GitHub Actions** — a cron in the cloud, nothing to keep awake. A ready-made
workflow lives at `.github/workflows/paper-trading.yml` (at the **repo root**,
not inside `trend_robot/`).

> Use a **PRIVATE** repository — this wiring carries trading-account context.

### 10.1 Add your paper keys as repository secrets

On GitHub: **Settings → Secrets and variables → Actions → New repository secret**.
Add both (paper keys):

| Secret name | Value |
|---|---|
| `APCA_API_KEY_ID` | your Alpaca **paper** Key ID |
| `APCA_API_SECRET_KEY` | your Alpaca **paper** Secret |

Secrets are encrypted and never printed in logs. Never commit keys to the repo.

### 10.2 Enable Actions & push the workflow

```bash
git add .github/workflows/paper-trading.yml
git commit -m "ci: monthly Alpaca paper-trading workflow"
git push
```

If Actions are disabled: **Settings → Actions → General → Allow all actions**.

### 10.3 Test it without trading (recommended first)

**Actions** tab → **TSMOM paper trading (Alpaca)** → **Run workflow**. Leave
**Dry-run** checked → it runs `run_live.py --dry-run` (no keys needed, sends
nothing) and should go green. This verifies install + data + logic in CI.

Then run it once more with **Dry-run unchecked** to verify the real paper
submission and your secrets (it will place that month's orders, then the
idempotence guard blocks further trades until next month).

### 10.4 How the schedule behaves

* Fires **weekdays at 13:30 UTC** (`cron: "30 13 * * 1-5"`, ~before the US open).
* The per-month **idempotence guard** ⇒ **one trade per month**; the other
  weekday runs are harmless health-checks that print "already submitted".
* State (the idempotence marker + price cache) is persisted between ephemeral
  runners via a **rolling `actions/cache`**, and each run's `live_state/` is also
  uploaded as a 90-day **artifact** for auditing. Alpaca itself remains the
  source of truth for positions/fills.

### 10.5 Things to know (GitHub Actions quirks)

* Scheduled workflows run **only on the default branch** (`main`).
* GitHub **auto-disables** scheduled workflows after **60 days of no repo
  activity** — the daily runs keep it alive; if you pause the project, re-enable
  it from the Actions tab.
* Scheduled runs can be **delayed** under load (fine for a monthly cadence).
* Cost: ~3 min/run × ~22 weekdays ≈ **~66 min/month** (well within the free
  2,000 min/month for private repos; unlimited for public).
* Cron is **UTC** — edit the cron line to change the time.
* **Kill switch:** Actions tab → the workflow → **⋯ → Disable workflow**, or
  delete the YAML file.
* Small caveat: if the cache is ever evicted mid-month, a re-run could place one
  extra (tiny) rebalance — harmless in paper, since the monthly target is
  unchanged and positions are reconciled from the broker.
