"""Validation layer: locked split, walk-forward, purged CV + embargo.

Re-exports the public validation API (spec section 6):

* :func:`train_test_split`, :func:`walk_forward_splits`,
  :func:`concat_test_segments`, :class:`WalkForwardWindow` (splits, 6.1 / 6.2);
* :func:`purged_cv_splits`, :class:`PurgedKFold`, :class:`PurgedSplit`
  (purged CV + embargo, 6.3);
* :class:`TrialCounter` (multiple-testing trials counter, 6.4);
* :func:`cost_stress_test`, :func:`cost_stress_table`, :class:`CostStressRow`
  (cost sensitivity, section 5);
* :func:`evaluate_final_validation`, :func:`format_final_report`,
  :class:`FinalValidationReport`, :class:`WalkForwardStability` (final
  section-6.5 verdict on the locked test set);
* :func:`strategy_fingerprint`, :func:`config_hash`, :func:`freeze_decision`,
  :func:`load_decision`, :func:`verify_config_matches`, :class:`DecisionRecord`
  (pre-registration of the frozen decision, 6.5 / 11);
* :func:`evaluate_holdout`, :func:`format_holdout_report`, :class:`HoldoutReport`
  (pristine forward / non-pristine retrospective hold-out read, 6.5 / 11).
"""

from __future__ import annotations

from trend_robot.validation.final_report import (
    FinalValidationReport,
    WalkForwardStability,
    evaluate_final_validation,
    format_final_report,
)
from trend_robot.validation.holdout import (
    HoldoutReport,
    evaluate_holdout,
    format_holdout_report,
)
from trend_robot.validation.preregistration import (
    DecisionRecord,
    config_hash,
    freeze_decision,
    load_decision,
    strategy_fingerprint,
    verify_config_matches,
)
from trend_robot.validation.purged_cv import (
    PurgedKFold,
    PurgedSplit,
    purged_cv_splits,
)
from trend_robot.validation.splits import (
    WalkForwardWindow,
    concat_test_segments,
    train_test_split,
    walk_forward_splits,
)
from trend_robot.validation.stress import (
    CostStressRow,
    cost_stress_table,
    cost_stress_test,
)
from trend_robot.validation.trials import TrialCounter

__all__: list[str] = [
    "CostStressRow",
    "DecisionRecord",
    "FinalValidationReport",
    "HoldoutReport",
    "PurgedKFold",
    "PurgedSplit",
    "TrialCounter",
    "WalkForwardStability",
    "WalkForwardWindow",
    "concat_test_segments",
    "config_hash",
    "cost_stress_table",
    "cost_stress_test",
    "evaluate_final_validation",
    "evaluate_holdout",
    "format_final_report",
    "format_holdout_report",
    "freeze_decision",
    "load_decision",
    "purged_cv_splits",
    "strategy_fingerprint",
    "train_test_split",
    "verify_config_matches",
    "walk_forward_splits",
]
