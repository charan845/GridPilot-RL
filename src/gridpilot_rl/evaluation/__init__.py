"""Evaluation, CVaR tail-risk benchmarking, and visualization utilities."""

from gridpilot_rl.evaluation.metrics import (
    PolicySpec,
    aggregate_results,
    compute_cvar,
    evaluate_policy_on_episodes,
    evaluate_suite,
    generate_eval_episodes,
    rollout_episode_trace,
)
from gridpilot_rl.evaluation.plotting import (
    plot_dispatch_schedule,
    plot_forecast_ablation,
    plot_multi_objective_comparison,
    plot_pareto_cost_vs_degradation,
    plot_robustness_comparison,
    plot_training_curves,
)

__all__ = [
    "PolicySpec",
    "aggregate_results",
    "compute_cvar",
    "evaluate_policy_on_episodes",
    "evaluate_suite",
    "generate_eval_episodes",
    "plot_dispatch_schedule",
    "plot_forecast_ablation",
    "plot_multi_objective_comparison",
    "plot_pareto_cost_vs_degradation",
    "plot_robustness_comparison",
    "plot_training_curves",
    "rollout_episode_trace",
]
