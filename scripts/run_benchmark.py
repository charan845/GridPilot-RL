"""End-to-end novel benchmark: training, cross-scenario CVaR evaluation, Oracle bound, and forecast ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gridpilot_rl.agents.sb3_runner import (
    get_action_mode_for_algo,
    get_base_controller_for_algo,
    train_agent,
)
from gridpilot_rl.baselines.rule_based import get_baseline
from gridpilot_rl.config import GridConfig, STRESS_PRESETS
from gridpilot_rl.data_loader import DEFAULT_DATA_PATH, load_opsd_splits
from gridpilot_rl.evaluation.metrics import (
    PolicySpec,
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

TRAIN_STRESS_POOL = (
    "nominal_real",
    "nominal_real",
    "high_demand",
    "cloudy_solar_drop",
    "volatile_tariff",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full GridPilot-RL novel benchmark suite.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--timesteps", type=int, default=35_000, help="Training timesteps per main agent.")
    parser.add_argument("--ablation-timesteps", type=int, default=20_000, help="Training timesteps for forecast ablation.")
    parser.add_argument("--n-eval-episodes", type=int, default=12, help="Held-out evaluation episodes per scenario.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_sampler, test_sampler = load_opsd_splits(args.data_path, train_ratio=0.75)
    print(f"[Data] Real OPSD split -> train={len(train_sampler)} hrs ({len(train_sampler)/168:.1f} wks), test={len(test_sampler)} hrs ({len(test_sampler)/168:.1f} wks)")

    models_dir = args.output_dir / "models"
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    logs_dir = args.output_dir / "logs"
    for d in (models_dir, tables_dir, figures_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Main Benchmark: Uncertainty-Aware + Physics Safety Shield
    grid_main = GridConfig(
        forecast_aware=True,
        forecast_uncertainty_aware=True,
        forecast_horizon=6,
        enable_safety_shield=True,
    )
    algos = ("dqn", "ppo", "sac", "residual_ppo", "residual_sac")
    histories: dict[str, list[dict]] = {}
    trained_specs: list[PolicySpec] = []

    for idx, algo in enumerate(algos):
        print(f"\n[1/3 Training] {algo.upper()} ({args.timesteps} timesteps)...")
        res = train_agent(
            algo=algo,
            episode_sampler=train_sampler,
            total_timesteps=args.timesteps,
            grid_config=grid_main,
            stress="nominal_real",
            randomize_stress=TRAIN_STRESS_POOL,
            seed=args.seed + idx,
            save_path=models_dir / f"{algo}_main.zip",
        )
        histories[algo] = res.history
        if res.history:
            costs = [h["total_operating_cost"] for h in res.history]
            print(
                f"  -> {algo.upper()} done | episodes={len(costs)} | "
                f"init_cost=${sum(costs[:5])/min(5,len(costs)):.2f} -> end_cost=${sum(costs[-5:])/min(5,len(costs)):.2f}"
            )
        trained_specs.append(
            PolicySpec(
                name=algo,
                policy=res.model,
                action_mode=get_action_mode_for_algo(algo),
                grid_config=grid_main,
                base_controller=get_base_controller_for_algo(algo, grid_main),
            )
        )

    (logs_dir / "training_history_main.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    plot_training_curves(histories, figures_dir / "training_curves.png")

    # 2. Held-Out Test Evaluation across all 6 scenarios + Baselines + LP Oracle
    all_policies: list[PolicySpec] = [
        PolicySpec("no_battery", get_baseline("no_battery", grid_main), "continuous", grid_main),
        PolicySpec("rule_tou", get_baseline("rule_tou", grid_main), "continuous", grid_main),
        PolicySpec("rule_peak_shave", get_baseline("rule_peak_shave", grid_main), "continuous", grid_main),
        *trained_specs,
        PolicySpec("oracle_lp", None, "continuous", grid_main, is_oracle=True),
    ]
    scenarios = list(STRESS_PRESETS.keys())
    print(f"\n[2/3 Evaluating] {len(all_policies)} controllers across {len(scenarios)} real+stress scenarios ({args.n_eval_episodes} episodes each)...")
    raw_df, summary_df = evaluate_suite(
        policies=all_policies,
        episode_sampler=test_sampler,
        scenarios=scenarios,
        n_episodes=args.n_eval_episodes,
        seed=args.seed + 5000,
        grid_config=grid_main,
    )
    raw_df.to_csv(tables_dir / "eval_episodes_main.csv", index=False)
    summary_df.to_csv(tables_dir / "eval_summary_main.csv", index=False)

    plot_multi_objective_comparison(summary_df, figures_dir / "multi_objective_nominal.png", scenario="nominal_real")
    plot_robustness_comparison(summary_df, figures_dir / "robustness_scenarios.png")
    plot_pareto_cost_vs_degradation(summary_df, figures_dir / "pareto_cost_degradation.png", scenario="nominal_real")

    # 48-hour dispatch slice on first held-out test week
    sample_ep = generate_eval_episodes(test_sampler, "nominal_real", n_episodes=1, seed=args.seed + 9000, grid_config=grid_main)[0]
    trace_policies = [p for p in all_policies if p.name in {"no_battery", "rule_tou", "ppo", "sac", "residual_sac", "oracle_lp"}]
    traces = {}
    for spec in trace_policies:
        _, trace_df = rollout_episode_trace(spec, sample_ep, grid_config=grid_main)
        traces[spec.name] = trace_df
    plot_dispatch_schedule(traces, figures_dir / "dispatch_48h_real.png", hours_to_show=48)

    # 3. Forecast & Uncertainty Ablation: Blind vs Point Forecast vs Uncertainty-Aware
    print(f"\n[3/3 Forecast Ablation] Blind vs Point vs Uncertainty-Aware ({args.ablation_timesteps} timesteps)...")
    ablation_modes = [
        ("forecast_blind", GridConfig(forecast_aware=False, forecast_uncertainty_aware=False, enable_safety_shield=True)),
        ("point_forecast", GridConfig(forecast_aware=True, forecast_uncertainty_aware=False, enable_safety_shield=True)),
        ("uncertainty_aware", GridConfig(forecast_aware=True, forecast_uncertainty_aware=True, enable_safety_shield=True)),
    ]
    ablation_frames: list[pd.DataFrame] = []
    ablation_algos = ("ppo", "sac", "residual_sac")
    ablation_scenarios = ("nominal_real", "cloudy_solar_drop", "forecast_dropout")

    for mode_name, cfg_mode in ablation_modes:
        specs_mode: list[PolicySpec] = []
        for a_idx, algo in enumerate(ablation_algos):
            res = train_agent(
                algo=algo,
                episode_sampler=train_sampler,
                total_timesteps=args.ablation_timesteps,
                grid_config=cfg_mode,
                stress="nominal_real",
                randomize_stress=TRAIN_STRESS_POOL,
                seed=args.seed + 100 + a_idx,
            )
            specs_mode.append(
                PolicySpec(
                    name=algo,
                    policy=res.model,
                    action_mode=get_action_mode_for_algo(algo),
                    grid_config=cfg_mode,
                    base_controller=get_base_controller_for_algo(algo, cfg_mode),
                )
            )
        _, ab_summary = evaluate_suite(
            policies=specs_mode,
            episode_sampler=test_sampler,
            scenarios=ablation_scenarios,
            n_episodes=args.n_eval_episodes,
            seed=args.seed + 8000,
            grid_config=cfg_mode,
        )
        ab_summary["algo"] = ab_summary["policy"]
        ab_summary["forecast_mode"] = mode_name
        ablation_frames.append(ab_summary)

    ablation_df = pd.concat(ablation_frames, ignore_index=True)
    ablation_df.to_csv(tables_dir / "forecast_ablation.csv", index=False)
    plot_forecast_ablation(ablation_df, figures_dir / "forecast_ablation.png")

    # Print key tables
    nom_cols = [
        "policy",
        "total_energy_cost_mean",
        "peak_demand_kw_mean",
        "battery_degradation_cost_mean",
        "total_carbon_kg_mean",
        "total_operating_cost_mean",
        "cvar20_operating_cost",
        "operating_cost_reduction_pct",
        "shield_intervention_rate_mean",
    ]
    print("\n=== Held-Out Real OPSD Test Results (nominal_real) ===")
    print(summary_df[summary_df["scenario"] == "nominal_real"][nom_cols].round(3).to_string(index=False))

    ab_pivot = ablation_df.groupby(["algo", "forecast_mode"], as_index=False)[
        ["total_energy_cost_mean", "peak_demand_kw_mean", "total_operating_cost_mean", "cvar20_operating_cost"]
    ].mean()
    print("\n=== Forecast & Uncertainty Ablation (Mean Across nominal_real / cloudy / forecast_dropout) ===")
    print(ab_pivot.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
