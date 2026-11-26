"""Multi-objective and CVaR tail-risk evaluation harness for GridPilot-RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from gridpilot_rl.baselines.oracle import OracleSchedulePolicy, solve_oracle_schedule
from gridpilot_rl.config import GridConfig, RewardWeights, get_stress_preset
from gridpilot_rl.data_loader import EpisodeSampler, GridEpisodeData
from gridpilot_rl.envs.smart_grid_env import SmartGridEnv


@dataclass
class PolicySpec:
    """Descriptor binding a controller to its name, action mode, and optional Residual base."""

    name: str
    policy: Any
    action_mode: Literal["continuous", "discrete"] = "continuous"
    grid_config: Optional[GridConfig] = None
    base_controller: Optional[Any] = None
    is_oracle: bool = False


def compute_cvar(values: pd.Series, alpha: float = 0.20) -> float:
    """Compute Conditional Value-at-Risk (mean of worst alpha fraction of high-cost episodes)."""
    arr = np.sort(values.to_numpy(dtype=np.float64))[::-1]
    k = max(1, int(np.ceil(alpha * len(arr))))
    return float(np.mean(arr[:k]))


def generate_eval_episodes(
    episode_sampler: EpisodeSampler,
    stress_name: str,
    n_episodes: int = 12,
    seed: int = 1000,
    grid_config: Optional[GridConfig] = None,
) -> list[GridEpisodeData]:
    """Sample deterministic real-data episodes (with optional stress) for paired evaluation."""
    grid = grid_config or GridConfig()
    stress = get_stress_preset(stress_name)
    episodes: list[GridEpisodeData] = []
    for i in range(n_episodes):
        rng = np.random.default_rng(seed + i * 37)
        episodes.append(
            episode_sampler.sample_episode(
                episode_steps=grid.episode_steps,
                forecast_horizon=grid.forecast_horizon,
                stress=stress,
                rng=rng,
            )
        )
    return episodes


def rollout_episode_trace(
    policy_spec: PolicySpec,
    episode_data: GridEpisodeData,
    grid_config: Optional[GridConfig] = None,
    reward_weights: Optional[RewardWeights] = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one deterministic episode and return summary metrics + step-by-step trace."""
    grid = policy_spec.grid_config or grid_config or GridConfig()
    weights = reward_weights or RewardWeights()

    if policy_spec.is_oracle:
        commands = solve_oracle_schedule(episode_data, grid_config=grid, reward_weights=weights)
        active_policy: Any = OracleSchedulePolicy(commands)
    else:
        active_policy = policy_spec.policy

    env = SmartGridEnv(
        grid_config=grid,
        reward_weights=weights,
        action_mode=policy_spec.action_mode,
        base_controller=policy_spec.base_controller,
        fixed_episode_data=episode_data,
    )
    obs, _ = env.reset()
    rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    final_metrics: dict[str, Any] = {}

    while not (terminated or truncated):
        action, _ = active_policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "step": info["step"],
                "hour": info["hour"],
                "demand_kw": info["demand_kw"],
                "solar_kw": info["solar_kw"],
                "price_per_kwh": info["price_per_kwh"],
                "carbon_kg_per_kwh": info["carbon_kg_per_kwh"],
                "battery_power_kw": info["battery_power_kw"],
                "soc": info["soc"],
                "grid_import_kw": info["grid_import_kw"],
                "grid_export_kw": info["grid_export_kw"],
                "shield_intervened": info["shield_intervened"],
                "step_energy_cost": info["step_energy_cost"],
                "step_degradation_cost": info["step_degradation_cost"],
                "step_carbon_kg": info["step_carbon_kg"],
                "peak_demand_kw": info["peak_demand_kw"],
                "reward": reward,
            }
        )
        if terminated and "episode_metrics" in info:
            final_metrics = dict(info["episode_metrics"])

    env.close()
    return final_metrics, pd.DataFrame(rows)


def evaluate_policy_on_episodes(
    policy_spec: PolicySpec,
    episodes: Sequence[GridEpisodeData],
    scenario_name: str,
    grid_config: Optional[GridConfig] = None,
    reward_weights: Optional[RewardWeights] = None,
) -> pd.DataFrame:
    """Evaluate a single policy across a fixed list of real episodes."""
    records: list[dict[str, Any]] = []
    for ep_idx, ep_data in enumerate(episodes):
        metrics, _ = rollout_episode_trace(
            policy_spec=policy_spec,
            episode_data=ep_data,
            grid_config=grid_config,
            reward_weights=reward_weights,
        )
        metrics["policy"] = policy_spec.name
        metrics["scenario"] = scenario_name
        metrics["episode"] = ep_idx
        records.append(metrics)
    return pd.DataFrame(records)


def aggregate_results(
    raw_df: pd.DataFrame,
    reference_policy: str = "no_battery",
) -> pd.DataFrame:
    """Aggregate metrics (mean, std, CVaR-20%) and compute savings vs no-battery baseline."""
    metric_cols = [
        "total_energy_cost",
        "peak_demand_kw",
        "peak_demand_cost",
        "battery_degradation_cost",
        "total_carbon_kg",
        "total_operating_cost",
        "total_throughput_kwh",
        "self_consumption_rate",
        "shield_intervention_rate",
        "total_reward",
    ]
    grouped = raw_df.groupby(["scenario", "policy"], as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [
        f"{col}_{stat}" if stat else col for col, stat in grouped.columns.to_flat_index()
    ]

    # CVaR-20% on total operating cost (worst-quintile tail cost)
    cvar_df = (
        raw_df.groupby(["scenario", "policy"], as_index=False)["total_operating_cost"]
        .agg(cvar20_operating_cost=lambda s: compute_cvar(s, alpha=0.20))
    )
    grouped = grouped.merge(cvar_df, on=["scenario", "policy"], how="left")

    if reference_policy in set(grouped["policy"]):
        ref = grouped[grouped["policy"] == reference_policy].set_index("scenario")
        energy_savings_pct: list[float] = []
        peak_reduction_pct: list[float] = []
        total_savings_pct: list[float] = []
        carbon_reduction_pct: list[float] = []

        for _, row in grouped.iterrows():
            scen = row["scenario"]
            ref_e = float(ref.loc[scen, "total_energy_cost_mean"])
            ref_p = float(ref.loc[scen, "peak_demand_kw_mean"])
            ref_tot = float(ref.loc[scen, "total_operating_cost_mean"])
            ref_co2 = float(ref.loc[scen, "total_carbon_kg_mean"])

            energy_savings_pct.append(
                100.0 * (ref_e - float(row["total_energy_cost_mean"])) / max(1e-6, abs(ref_e))
            )
            peak_reduction_pct.append(
                100.0 * (ref_p - float(row["peak_demand_kw_mean"])) / max(1e-6, ref_p)
            )
            total_savings_pct.append(
                100.0 * (ref_tot - float(row["total_operating_cost_mean"])) / max(1e-6, abs(ref_tot))
            )
            carbon_reduction_pct.append(
                100.0 * (ref_co2 - float(row["total_carbon_kg_mean"])) / max(1e-6, ref_co2)
            )

        grouped["energy_cost_reduction_pct"] = energy_savings_pct
        grouped["peak_reduction_pct"] = peak_reduction_pct
        grouped["operating_cost_reduction_pct"] = total_savings_pct
        grouped["carbon_reduction_pct"] = carbon_reduction_pct
        grouped["weekly_savings_usd"] = [
            float(ref.loc[row["scenario"], "total_operating_cost_mean"])
            - float(row["total_operating_cost_mean"])
            for _, row in grouped.iterrows()
        ]

    return grouped


def evaluate_suite(
    policies: Sequence[PolicySpec],
    episode_sampler: EpisodeSampler,
    scenarios: Sequence[str],
    n_episodes: int = 12,
    seed: int = 2000,
    grid_config: Optional[GridConfig] = None,
    reward_weights: Optional[RewardWeights] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all policies across all stress scenarios using paired real episodes."""
    frames: list[pd.DataFrame] = []
    for s_idx, scenario_name in enumerate(scenarios):
        episodes = generate_eval_episodes(
            episode_sampler=episode_sampler,
            stress_name=scenario_name,
            n_episodes=n_episodes,
            seed=seed + s_idx * 1000,
            grid_config=grid_config,
        )
        for spec in policies:
            df = evaluate_policy_on_episodes(
                policy_spec=spec,
                episodes=episodes,
                scenario_name=scenario_name,
                grid_config=grid_config,
                reward_weights=reward_weights,
            )
            frames.append(df)

    raw_df = pd.concat(frames, ignore_index=True)
    summary_df = aggregate_results(raw_df)
    return raw_df, summary_df
