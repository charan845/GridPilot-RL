"""Integration tests for baselines, PPO/SAC/DQN, Residual RL, and CVaR evaluation."""

from __future__ import annotations

import pytest

from gridpilot_rl.agents.sb3_runner import (
    get_action_mode_for_algo,
    get_base_controller_for_algo,
    load_agent,
    train_agent,
)
from gridpilot_rl.baselines.rule_based import get_baseline
from gridpilot_rl.config import GridConfig
from gridpilot_rl.data_loader import DEFAULT_DATA_PATH, load_opsd_splits
from gridpilot_rl.evaluation.metrics import PolicySpec, evaluate_suite


def test_baselines_and_oracle_on_real_opsd() -> None:
    _, test_sampler = load_opsd_splits(DEFAULT_DATA_PATH)
    cfg = GridConfig(episode_steps=168, forecast_aware=True, forecast_uncertainty_aware=True)
    policies = [
        PolicySpec("no_battery", get_baseline("no_battery", cfg), "continuous", cfg),
        PolicySpec("rule_tou", get_baseline("rule_tou", cfg), "continuous", cfg),
        PolicySpec("rule_peak_shave", get_baseline("rule_peak_shave", cfg), "continuous", cfg),
        PolicySpec("oracle_lp", None, "continuous", cfg, is_oracle=True),
    ]
    _, summary = evaluate_suite(
        policies=policies,
        episode_sampler=test_sampler,
        scenarios=["nominal_real"],
        n_episodes=4,
        seed=123,
        grid_config=cfg,
    )
    by_pol = summary.set_index("policy")
    assert by_pol.loc["oracle_lp", "total_operating_cost_mean"] <= by_pol.loc["no_battery", "total_operating_cost_mean"]
    assert by_pol.loc["rule_tou", "total_operating_cost_mean"] < by_pol.loc["no_battery", "total_operating_cost_mean"]
    assert "cvar20_operating_cost" in by_pol.columns


@pytest.mark.parametrize("algo", ["ppo", "sac", "dqn", "residual_ppo", "residual_sac"])
def test_all_rl_agents_train_save_load_eval(algo: str, tmp_path) -> None:
    train_sampler, test_sampler = load_opsd_splits(DEFAULT_DATA_PATH)
    cfg = GridConfig(episode_steps=24, forecast_aware=True, forecast_uncertainty_aware=True, forecast_horizon=4)
    model_path = tmp_path / f"{algo}.zip"

    result = train_agent(
        algo=algo,
        episode_sampler=train_sampler,
        total_timesteps=250,
        grid_config=cfg,
        stress="nominal_real",
        seed=42,
        save_path=model_path,
        device="cpu",
    )
    assert model_path.exists()
    assert len(result.history) >= 1

    loaded = load_agent(algo, model_path, device="cpu")
    spec = PolicySpec(
        name=algo,
        policy=loaded,
        action_mode=get_action_mode_for_algo(algo),
        grid_config=cfg,
        base_controller=get_base_controller_for_algo(algo, cfg),
    )
    _, summary = evaluate_suite(
        policies=[spec],
        episode_sampler=test_sampler,
        scenarios=["nominal_real", "high_demand"],
        n_episodes=2,
        seed=77,
        grid_config=cfg,
    )
    assert len(summary) == 2
    assert set(summary["scenario"]) == {"nominal_real", "high_demand"}
