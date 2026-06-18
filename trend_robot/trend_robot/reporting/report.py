"""Reporting layer -- charts + metrics table (spec 9 / T8, presentation only).

This module is *presentation only*: it contains NO calculation logic. It takes
an already-computed :class:`~trend_robot.backtest.engine.BacktestResult` and the
pre-computed metrics dictionary from
:func:`trend_robot.metrics.performance.performance_metrics`, and renders them to
disk as PNG charts and an HTML/CSV metrics table.

Artifacts produced by :func:`build_report`
-------------------------------------------
* ``equity_curve.png``      -- marked-to-market equity over time.
* ``drawdown.png``          -- underwater (peak-to-trough) curve.
* ``exposure.png``          -- gross/net exposure ``sum|w_i|`` / ``sum w_i``.
* ``contribution.png``      -- per-asset realized P&L attribution (bar chart).
* ``metrics_table.html``    -- a styled HTML table of the scalar metrics.
* ``metrics_table.csv``     -- the same metrics as machine-readable CSV.

Design notes
------------
* All drawdown/exposure *display* series are derived here purely for plotting
  (a chart axis is a presentation concern); no strategy/risk decision is made.
* Matplotlib uses the non-interactive ``Agg`` backend so the module renders
  headlessly (CI, servers) without a display.
* Every figure is explicitly closed to avoid leaking memory across runs.
* The function is deterministic given its inputs: identical results always yield
  identical artifacts.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
import pandas as pd  # noqa: E402

from trend_robot.backtest.engine import BacktestResult  # noqa: E402

__all__ = ["build_report"]


# ---------------------------------------------------------------------------
# Individual chart renderers (presentation only)
# ---------------------------------------------------------------------------
def _plot_equity_curve(equity: pd.Series, path: Path, title: str) -> None:
    """Render the marked-to-market equity curve to ``path``."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(equity.index, equity.to_numpy(), color="#1f77b4", lw=1.4)
    ax.set_title(f"{title} -- Equity curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_drawdown(equity: pd.Series, path: Path, title: str) -> None:
    """Render the underwater (drawdown) curve to ``path``.

    The drawdown ``equity / running_max - 1`` is computed here strictly for
    display; it is not used for any strategy decision.
    """
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0) * 100.0  # percent, <= 0
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(
        drawdown.index, drawdown.to_numpy(), 0.0, color="#d62728", alpha=0.4
    )
    ax.plot(drawdown.index, drawdown.to_numpy(), color="#d62728", lw=1.0)
    ax.set_title(f"{title} -- Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_exposure(weights: pd.DataFrame, path: Path, title: str) -> None:
    """Render gross and net exposure over time to ``path``.

    Gross exposure is ``sum_i |w_i|`` (leverage); net exposure is ``sum_i w_i``
    (long minus short). Both are display aggregations only.
    """
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(gross.index, gross.to_numpy(), color="#2ca02c", lw=1.2, label="Gross")
    ax.plot(net.index, net.to_numpy(), color="#9467bd", lw=1.0, label="Net")
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
    ax.set_title(f"{title} -- Exposure over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exposure (fraction of equity)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_contribution(
    per_asset_pnl: Mapping[str, float], path: Path, title: str
) -> None:
    """Render the per-asset realized P&L attribution as a bar chart.

    ``per_asset_pnl`` is read straight from the metrics dict (computed upstream);
    this renderer performs no attribution math.
    """
    items = list(per_asset_pnl.items())
    assets = [str(a) for a, _ in items]
    values = [float(v) for _, v in items]
    colors = ["#2ca02c" if v >= 0.0 else "#d62728" for v in values]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(assets, values, color=colors)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(f"{title} -- Per-asset P&L contribution")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Realized P&L (currency)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics table renderers (presentation only)
# ---------------------------------------------------------------------------
def _format_metric_value(value: Any) -> str:
    """Format a single metric value for display (no calculation)."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):  # bool is an int; handle before numeric
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        return f"{value:,.6g}"
    if isinstance(value, (int,)):
        return f"{value:,d}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the scalar metrics (drop nested structures like dicts)."""
    return {
        key: val
        for key, val in metrics.items()
        if not isinstance(val, (dict, list, tuple, pd.Series, pd.DataFrame))
    }


def _write_metrics_table(
    metrics: Mapping[str, Any], html_path: Path, csv_path: Path, title: str
) -> None:
    """Persist the scalar metrics as an HTML table and a CSV file."""
    scalars = _scalar_metrics(metrics)

    # --- CSV (machine-readable) -------------------------------------------
    table = pd.DataFrame(
        {
            "metric": list(scalars.keys()),
            "value": [_format_metric_value(v) for v in scalars.values()],
        }
    )
    table.to_csv(csv_path, index=False)

    # --- HTML (human-readable) --------------------------------------------
    rows = "\n".join(
        f"      <tr><td>{html.escape(str(k))}</td>"
        f"<td style='text-align:right'>{html.escape(_format_metric_value(v))}</td></tr>"
        for k, v in scalars.items()
    )
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)} -- Metrics</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }}
    h1 {{ font-size: 1.25rem; }}
    table {{ border-collapse: collapse; min-width: 28rem; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 12px; }}
    th {{ background: #f2f2f2; text-align: left; }}
    tr:nth-child(even) td {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)} -- Performance metrics</h1>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(doc, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_report(
    result: BacktestResult,
    metrics: Mapping[str, Any],
    output_dir: str | Path,
    title: str = "TSMOM research",
) -> dict[str, Path]:
    """Render charts + a metrics table for a backtest (presentation only).

    This function contains **no calculation logic**: it consumes the already
    computed ``result`` and ``metrics`` and writes presentation artifacts to
    ``output_dir`` (created if needed). Any series derived here (drawdown,
    exposure) are computed solely to draw an axis, not to make a decision.

    Parameters
    ----------
    result:
        Backtest output (equity curve, held weights, turnover, trades) from
        :func:`trend_robot.backtest.engine.run_backtest`.
    metrics:
        Flat metrics mapping from
        :func:`trend_robot.metrics.performance.performance_metrics`, including
        the nested ``per_asset_pnl`` attribution used for the contribution chart.
    output_dir:
        Directory to write artifacts into (created with parents if missing).
    title:
        Human-readable title prefix used in chart titles and the HTML table.

    Returns
    -------
    dict[str, Path]
        Mapping of artifact name to the file path written, with keys
        ``equity_curve``, ``drawdown``, ``exposure``, ``contribution``,
        ``metrics_html`` and ``metrics_csv``.

    Raises
    ------
    TypeError
        If ``result`` is not a :class:`BacktestResult`.
    """
    if not isinstance(result, BacktestResult):
        raise TypeError(
            f"'result' must be a BacktestResult, got {type(result).__name__}."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    equity = result.equity.astype("float64")

    paths: dict[str, Path] = {
        "equity_curve": out / "equity_curve.png",
        "drawdown": out / "drawdown.png",
        "exposure": out / "exposure.png",
        "contribution": out / "contribution.png",
        "metrics_html": out / "metrics_table.html",
        "metrics_csv": out / "metrics_table.csv",
    }

    _plot_equity_curve(equity, paths["equity_curve"], title)
    _plot_drawdown(equity, paths["drawdown"], title)
    _plot_exposure(result.weights, paths["exposure"], title)
    _plot_contribution(
        dict(metrics.get("per_asset_pnl", {})), paths["contribution"], title
    )
    _write_metrics_table(
        metrics, paths["metrics_html"], paths["metrics_csv"], title
    )

    return paths
