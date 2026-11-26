"""Unit tests for SmartGridEnv, Physics Safety Shield, OPSD data loader, and LP Oracle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from gridpilot_rl.baselines.oracle import solve_oracle_schedule
from gridpilot_rl.baselines.rule_based import TimeOfUseRuleBaseline
from gridpilot_rl.config import GridConfig, get_stress_preset
from gridpilot_rl.data_loader import DEFAULT_DATA_PATH, EpisodeSampler, load_opsd_splits
from gridpilot_rl.envs.smart_grid_env import SmartGridEnv


def _make_sample_df(rows: int = 240) -> pd.DataFrame:
    hours = np.arange(rows, dtype=np.float32) % 24.0
    demand = 2.5 + 1.5 * np.sin(2 * np.pi * (hours - 14.0) / 24.0)
    solar = np.maximum(0.0, 6.0 * np.sin(np.pi * (hours - 6.0) / 13.0))
    price = 0.22 + 0.08 * np.sin(2 * np.pi * (hours - 13.0) / 24.0)
    carbon = 0.48 - 0.12 * np.sin(np.pi * np.clip(hours - 6.0, 0.0, 13.0) / 13.0)
    return pd.DataFrame(
        {
            "utc_timestamp": pd.date_range("2016-06-01", periods=rows, freq="h", tz="UTC").astype(str),
            "demand_kw": demand,
            "solar_kw": solar,
            "price_per_kwh": price,
            "carbon_kg_per_kwh": carbon,
            "hour": hours,
        }
    )


@pytest.mark.parametrize("action_mode", ["continuous", "discrete"])
@pytest.mark.parametrize(
    "forecast_aware,uncertainty_aware",
    [(False, False), (True, False), (True, True)],
)
def test_gymnasium_check_env(
    action_mode: str, forecast_aware: bool, uncertainty_aware: bool
) -> None:
    sampler = EpisodeSampler(_make_sample_df(120))
    cfg = GridConfig(
        episode_steps=24,
        forecast_aware=forecast_aware,
        forecast_uncertainty_aware=uncertainty_aware,
        forecast_horizon=4,
    )
    env = SmartGridEnv(grid_config=cfg, episode_sampler=sampler, action_mode=action_mode)
    check_env(env, skip_render_check=True)
    env.close()


def test_observation_dimensions() -> None:
    sampler = EpisodeSampler(_make_sample_df(120))
    for aware, unc, expected_dim in [
        (False, False, 11),
        (True, False, 11 + 4 * 6),
        (True, True, 11 + 7 * 6),
    ]:
        cfg = GridConfig(
            episode_steps=24,
            forecast_aware=aware,
            forecast_uncertainty_aware=unc,
            forecast_horizon=6,
        )
        env = SmartGridEnv(grid_config=cfg, episode_sampler=sampler)
        obs, _ = env.reset(seed=7)
        assert obs.shape == (expected_dim,)
        assert not np.isnan(obs).any()


def test_safety_shield_ramp_and_soc_bounds() -> None:
    sampler = EpisodeSampler(_make_sample_df(120))
    cfg = GridConfig(
        episode_steps=48,
        soc_min=0.12,
        soc_max=0.90,
        initial_soc=0.50,
        enable_safety_shield=True,
        ramp_limit_kw=4.0,
        pre_peak_reserve_soc=0.30,
    )
    env = SmartGridEnv(grid_config=cfg, episode_sampler=sampler, action_mode="continuous")
    env.reset(seed=5)

    prev_power = 0.0
    for step in range(48):
        # Alternate violent +1 and -1 requests to verify ramp-rate shield
        raw = 1.0 if step % 2 == 0 else -1.0
        _, _, term, _, info = env.step(np.array([raw], dtype=np.float32))
        assert cfg.soc_min - 1e-6 <= info["soc"] <= cfg.soc_max + 1e-6
        assert abs(info["battery_power_kw"] - prev_power) <= cfg.ramp_limit_kw + 1e-4
        prev_power = info["battery_power_kw"]
        if term:
            break


def test_residual_mode_and_oracle_lp() -> None:
    sampler = EpisodeSampler(_make_sample_df(120))
    cfg = GridConfig(episode_steps=24, forecast_aware=True, forecast_uncertainty_aware=True)
    ep = sampler.sample_episode(episode_steps=24, forecast_horizon=6, stress=get_stress_preset("nominal_real"))

    # Residual RL wrapper with zero residual action must match base controller
    base = TimeOfUseRuleBaseline(cfg)
    env_res = SmartGridEnv(grid_config=cfg, fixed_episode_data=ep, base_controller=base)
    env_raw = SmartGridEnv(grid_config=cfg, fixed_episode_data=ep, base_controller=None)
    obs_res, _ = env_res.reset()
    obs_raw, _ = env_raw.reset()
    for _ in range(24):
        u_base = base.compute_normalized_action(obs_raw)
        obs_res, _, _, _, info_res = env_res.step(np.array([0.0], dtype=np.float32))
        obs_raw, _, _, _, info_raw = env_raw.step(np.array([u_base], dtype=np.float32))
        assert info_res["battery_power_kw"] == pytest.approx(info_raw["battery_power_kw"], abs=1e-5)

    # LP Oracle solves feasible commands in [-1, 1]
    oracle_cmds = solve_oracle_schedule(ep, grid_config=cfg)
    assert oracle_cmds.shape == (24,)
    assert np.all(oracle_cmds >= -1.0 - 1e-6) and np.all(oracle_cmds <= 1.0 + 1e-6)


def test_real_opsd_dataset_chronological_split() -> None:
    assert DEFAULT_DATA_PATH.exists(), "Run scripts/prepare_real_data.py before testing OPSD split."
    train_s, test_s = load_opsd_splits(DEFAULT_DATA_PATH, train_ratio=0.75)
    assert len(train_s) > 5000
    assert len(test_s) > 1500
    last_train_ts = pd.Timestamp(train_s.df["utc_timestamp"].iloc[-1])
    first_test_ts = pd.Timestamp(test_s.df["utc_timestamp"].iloc[0])
    assert last_train_ts < first_test_ts
