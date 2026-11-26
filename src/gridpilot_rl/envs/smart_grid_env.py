"""Custom Gymnasium environment for smart grid energy scheduling.

Features:
- Continuous (PPO/SAC) and discrete (DQN) action modes over identical physics.
- Physics Safety Shield (SoC bounds, inverter ramp-rate limit, pre-peak SoC reserve).
- Physics-Guided Residual RL mode (u = clip(u_base(obs) + alpha * delta_u, -1, 1)).
- Uncertainty-aware multi-horizon look-ahead observations.
- Non-linear electrochemical SoC-stress degradation + marginal carbon emissions.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gridpilot_rl.config import GridConfig, RewardWeights, StressConfig, get_stress_preset
from gridpilot_rl.data_loader import GridEpisodeData, EpisodeSampler


class SmartGridEnv(gym.Env):
    """Smart grid microgrid scheduling environment with BESS, solar PV, and dynamic pricing."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_config: Optional[GridConfig] = None,
        reward_weights: Optional[RewardWeights] = None,
        episode_sampler: Optional[EpisodeSampler] = None,
        stress: Optional[StressConfig | str] = None,
        randomize_stress: Optional[Sequence[StressConfig | str]] = None,
        action_mode: Literal["continuous", "discrete"] = "continuous",
        base_controller: Optional[Any] = None,
        fixed_episode_data: Optional[GridEpisodeData] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.grid = grid_config or GridConfig()
        self.weights = reward_weights or RewardWeights()
        self.episode_sampler = episode_sampler
        self.action_mode = action_mode
        self.base_controller = base_controller
        self.fixed_episode_data = fixed_episode_data
        self.render_mode = render_mode

        if isinstance(stress, str):
            self.base_stress = get_stress_preset(stress)
        elif stress is not None:
            self.base_stress = stress
        else:
            self.base_stress = StressConfig()

        self.stress_pool: list[StressConfig] = []
        if randomize_stress:
            for item in randomize_stress:
                self.stress_pool.append(
                    get_stress_preset(item) if isinstance(item, str) else item
                )

        self.discrete_levels = np.asarray(self.grid.discrete_action_levels, dtype=np.float32)

        if self.action_mode == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        elif self.action_mode == "discrete":
            self.action_space = spaces.Discrete(len(self.discrete_levels))
        else:
            raise ValueError(f"Unsupported action_mode '{self.action_mode}'.")

        # 11 base state features:
        # [norm_soc, demand, solar, price, carbon, sin_hour, cos_hour, peak_so_far,
        #  net_load, prev_battery_power, price_spread_vs_forecast]
        # + optional 4*H point forecasts (demand, solar, price, net_load)
        # + optional 3*H forecast uncertainty intervals
        base_obs_dim = 11
        forecast_dim = 0
        if self.grid.forecast_aware:
            forecast_dim += 4 * self.grid.forecast_horizon
            if self.grid.forecast_uncertainty_aware:
                forecast_dim += 3 * self.grid.forecast_horizon
        obs_dim = base_obs_dim + forecast_dim

        self.observation_space = spaces.Box(
            low=-5.0, high=15.0, shape=(obs_dim,), dtype=np.float32
        )

        # Runtime state
        self._step_idx: int = 0
        self._soc: float = self.grid.initial_soc
        self._prev_battery_power_kw: float = 0.0
        self._peak_demand_kw: float = 0.0
        self._episode_data: Optional[GridEpisodeData] = None
        self._active_stress: StressConfig = self.base_stress
        self._current_obs: Optional[np.ndarray] = None

        # Episode accumulators
        self._total_energy_cost: float = 0.0
        self._total_degradation_cost: float = 0.0
        self._total_carbon_cost: float = 0.0
        self._total_carbon_kg: float = 0.0
        self._total_throughput_kwh: float = 0.0
        self._total_grid_import_kwh: float = 0.0
        self._total_grid_export_kwh: float = 0.0
        self._total_solar_kwh: float = 0.0
        self._total_demand_kwh: float = 0.0
        self._shield_interventions: int = 0
        self._total_reward: float = 0.0

    @property
    def episode_data(self) -> GridEpisodeData:
        if self._episode_data is None:
            raise RuntimeError("Environment must be reset before accessing episode_data.")
        return self._episode_data

    def _select_stress(self) -> StressConfig:
        if self.stress_pool:
            idx = int(self.np_random.integers(0, len(self.stress_pool)))
            return self.stress_pool[idx]
        return self.base_stress

    def _build_obs(self) -> np.ndarray:
        data = self.episode_data
        t = min(self._step_idx, len(data.demand_kw) - 1)
        soc_span = max(1e-6, self.grid.soc_max - self.grid.soc_min)
        norm_soc = (self._soc - self.grid.soc_min) / soc_span

        hour_rad = 2.0 * np.pi * (float(data.hours[t]) / 24.0)
        demand_val = float(data.demand_kw[t])
        solar_val = float(data.solar_kw[t])
        price_val = float(data.price_per_kwh[t])
        net_load_val = demand_val - solar_val
        price_spread = (
            (price_val - float(np.mean(data.price_forecast[t]))) / self.grid.price_scale_per_kwh
            if self.grid.forecast_aware
            else 0.0
        )

        parts: list[np.ndarray] = [
            np.asarray(
                [
                    np.clip(norm_soc, 0.0, 1.0),
                    demand_val / self.grid.demand_scale_kw,
                    solar_val / self.grid.solar_scale_kw,
                    price_val / self.grid.price_scale_per_kwh,
                    float(data.carbon_kg_per_kwh[t]) / self.grid.carbon_scale_kg_per_kwh,
                    float(np.sin(hour_rad)),
                    float(np.cos(hour_rad)),
                    self._peak_demand_kw / self.grid.demand_scale_kw,
                    net_load_val / self.grid.demand_scale_kw,
                    self._prev_battery_power_kw / max(1e-6, self.grid.max_charge_kw),
                    price_spread,
                ],
                dtype=np.float32,
            )
        ]

        if self.grid.forecast_aware:
            net_load_fc = (data.demand_forecast[t] - data.solar_forecast[t]) / self.grid.demand_scale_kw
            parts.append((data.demand_forecast[t] / self.grid.demand_scale_kw).astype(np.float32))
            parts.append((data.solar_forecast[t] / self.grid.solar_scale_kw).astype(np.float32))
            parts.append((data.price_forecast[t] / self.grid.price_scale_per_kwh).astype(np.float32))
            parts.append(net_load_fc.astype(np.float32))
            if self.grid.forecast_uncertainty_aware:
                parts.append(
                    (data.demand_uncertainty[t] / self.grid.demand_scale_kw).astype(np.float32)
                )
                parts.append(
                    (data.solar_uncertainty[t] / self.grid.solar_scale_kw).astype(np.float32)
                )
                parts.append(
                    (data.price_uncertainty[t] / self.grid.price_scale_per_kwh).astype(np.float32)
                )

        obs = np.concatenate(parts, axis=0)
        obs = np.nan_to_num(obs, nan=0.0, posinf=15.0, neginf=-5.0)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(
            np.float32
        )

    def _decode_raw_action(self, action: Any) -> float:
        if self.action_mode == "discrete":
            idx = int(np.asarray(action).item())
            if idx < 0 or idx >= len(self.discrete_levels):
                raise ValueError(f"Discrete action {idx} out of bounds.")
            return float(self.discrete_levels[idx])
        val = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        return float(np.clip(val, -1.0, 1.0))

    def _compose_residual_action(self, raw_cmd: float) -> float:
        if self.base_controller is None or self._current_obs is None:
            return raw_cmd
        u_base = float(self.base_controller.compute_normalized_action(self._current_obs))
        return float(np.clip(u_base + self.grid.residual_alpha * raw_cmd, -1.0, 1.0))

    def _apply_physics_safety_shield(
        self,
        desired_power_kw: float,
        hour: float,
    ) -> tuple[float, bool]:
        """Project desired battery power onto SoC, ramp-rate, and pre-peak reserve constraints."""
        dt = self.grid.timestep_hours
        cap = self.grid.battery_capacity_kwh
        energy_kwh = self._soc * cap

        # Hard SoC limits (always enforced by physical BMS)
        energy_min = self.grid.soc_min * cap
        energy_max = self.grid.soc_max * cap

        # Pre-peak reserve floor (enforced by Safety Shield before evening peak)
        reserve_soc = self.grid.soc_min
        if (
            self.grid.enable_safety_shield
            and self.grid.pre_peak_hours[0] <= hour < self.grid.pre_peak_hours[1]
        ):
            reserve_soc = max(self.grid.soc_min, self.grid.pre_peak_reserve_soc)
        effective_energy_min = reserve_soc * cap

        max_grid_charge_kw = max(
            0.0, (energy_max - energy_kwh) / (self.grid.charge_efficiency * dt)
        )
        max_ac_discharge_kw = max(
            0.0, (energy_kwh - effective_energy_min) * self.grid.discharge_efficiency / dt
        )

        # Ramp-rate bounds when Safety Shield is enabled
        if self.grid.enable_safety_shield:
            low_bound = max(
                -self.grid.max_discharge_kw,
                -max_ac_discharge_kw,
                self._prev_battery_power_kw - self.grid.ramp_limit_kw,
            )
            high_bound = min(
                self.grid.max_charge_kw,
                max_grid_charge_kw,
                self._prev_battery_power_kw + self.grid.ramp_limit_kw,
            )
        else:
            low_bound = max(-self.grid.max_discharge_kw, -max_ac_discharge_kw)
            high_bound = min(self.grid.max_charge_kw, max_grid_charge_kw)

        if low_bound > high_bound:
            # Fallback to physically feasible SoC bounds if reserve/ramp conflict
            low_bound = max(-self.grid.max_discharge_kw, -(energy_kwh - energy_min) * self.grid.discharge_efficiency / dt)
            high_bound = min(self.grid.max_charge_kw, max_grid_charge_kw)

        safe_power_kw = float(np.clip(desired_power_kw, low_bound, high_bound))
        intervened = abs(safe_power_kw - desired_power_kw) > 1e-3
        return safe_power_kw, intervened

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self._step_idx = 0
        self._soc = float(np.clip(self.grid.initial_soc, self.grid.soc_min, self.grid.soc_max))
        self._prev_battery_power_kw = 0.0
        self._peak_demand_kw = 0.0

        self._total_energy_cost = 0.0
        self._total_degradation_cost = 0.0
        self._total_carbon_cost = 0.0
        self._total_carbon_kg = 0.0
        self._total_throughput_kwh = 0.0
        self._total_grid_import_kwh = 0.0
        self._total_grid_export_kwh = 0.0
        self._total_solar_kwh = 0.0
        self._total_demand_kwh = 0.0
        self._shield_interventions = 0
        self._total_reward = 0.0

        if options and "episode_data" in options:
            self._episode_data = options["episode_data"]
            self._active_stress = self.base_stress
        elif self.fixed_episode_data is not None:
            self._episode_data = self.fixed_episode_data
            self._active_stress = self.base_stress
        elif self.episode_sampler is not None:
            self._active_stress = self._select_stress()
            rng = np.random.default_rng(int(self.np_random.integers(0, 2**31 - 1)))
            self._episode_data = self.episode_sampler.sample_episode(
                episode_steps=self.grid.episode_steps,
                forecast_horizon=self.grid.forecast_horizon,
                stress=self._active_stress,
                rng=rng,
            )
        else:
            raise RuntimeError(
                "SmartGridEnv requires either fixed_episode_data, episode_sampler, or options['episode_data']."
            )

        self._current_obs = self._build_obs()
        info = {
            "stress": self._active_stress.name,
            "soc": self._soc,
            "peak_demand_kw": self._peak_demand_kw,
        }
        return self._current_obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        data = self.episode_data
        t = self._step_idx
        dt = self.grid.timestep_hours
        cap = self.grid.battery_capacity_kwh

        demand_kw = float(data.demand_kw[t])
        solar_kw = float(data.solar_kw[t])
        price = float(data.price_per_kwh[t])
        carbon_intensity = float(data.carbon_kg_per_kwh[t])
        hour = float(data.hours[t])

        raw_cmd = self._decode_raw_action(action)
        cmd = self._compose_residual_action(raw_cmd)

        desired_power_kw = (
            cmd * self.grid.max_charge_kw if cmd >= 0.0 else cmd * self.grid.max_discharge_kw
        )
        actual_battery_power_kw, shielded = self._apply_physics_safety_shield(
            desired_power_kw=desired_power_kw,
            hour=hour,
        )
        if shielded:
            self._shield_interventions += 1

        energy_kwh = self._soc * cap
        if actual_battery_power_kw >= 0.0:
            new_energy_kwh = energy_kwh + actual_battery_power_kw * self.grid.charge_efficiency * dt
        else:
            new_energy_kwh = (
                energy_kwh - ((-actual_battery_power_kw) * dt) / self.grid.discharge_efficiency
            )
        new_soc = float(np.clip(new_energy_kwh / cap, self.grid.soc_min, self.grid.soc_max))

        # Microgrid power balance
        net_load_kw = demand_kw - solar_kw + actual_battery_power_kw
        grid_import_kw = max(0.0, net_load_kw)
        grid_export_kw = max(0.0, -net_load_kw)

        import_cost = grid_import_kw * dt * price
        export_revenue = grid_export_kw * dt * price * self.grid.feed_in_tariff_ratio
        step_energy_cost = import_cost - export_revenue

        # Incremental peak demand cost telescopes to episode_peak * penalty
        prev_peak_kw = self._peak_demand_kw
        self._peak_demand_kw = max(prev_peak_kw, grid_import_kw)
        step_peak_cost = (self._peak_demand_kw - prev_peak_kw) * self.grid.peak_demand_penalty_per_kw

        # Non-linear electrochemical aging: throughput + extreme-SoC stress + ramp stress
        step_throughput_kwh = abs(actual_battery_power_kw) * dt
        mid_soc = 0.5 * (self._soc + new_soc)
        extreme_soc_excess = max(0.0, 0.20 - mid_soc) + max(0.0, mid_soc - 0.85)
        soc_stress_factor = 1.0 + self.grid.soc_stress_penalty * (extreme_soc_excess / 0.15)
        ramp_kw = abs(actual_battery_power_kw - self._prev_battery_power_kw)
        step_degradation_cost = (
            step_throughput_kwh * self.grid.degradation_cost_per_kwh * soc_stress_factor
            + ramp_kw * self.grid.ramp_stress_cost_per_kw * (1.0 if step_throughput_kwh > 1e-4 else 0.0)
        )

        # Carbon emissions & shadow carbon cost
        step_carbon_kg = grid_import_kw * dt * carbon_intensity
        step_carbon_cost = step_carbon_kg * self.grid.carbon_price_per_kg

        weighted_cost = (
            self.weights.energy_cost * step_energy_cost
            + self.weights.peak_demand * step_peak_cost
            + self.weights.degradation * step_degradation_cost
            + self.weights.carbon * step_carbon_cost
        )
        reward = float(-self.weights.reward_scale * weighted_cost)

        # Update state and accumulators
        self._soc = new_soc
        self._prev_battery_power_kw = actual_battery_power_kw
        self._step_idx += 1
        self._total_energy_cost += step_energy_cost
        self._total_degradation_cost += step_degradation_cost
        self._total_carbon_cost += step_carbon_cost
        self._total_carbon_kg += step_carbon_kg
        self._total_throughput_kwh += step_throughput_kwh
        self._total_grid_import_kwh += grid_import_kw * dt
        self._total_grid_export_kwh += grid_export_kw * dt
        self._total_solar_kwh += solar_kw * dt
        self._total_demand_kwh += demand_kw * dt
        self._total_reward += reward

        terminated = self._step_idx >= len(data.demand_kw)
        truncated = False
        self._current_obs = self._build_obs()

        info: dict[str, Any] = {
            "step": t,
            "hour": hour,
            "demand_kw": demand_kw,
            "solar_kw": solar_kw,
            "price_per_kwh": price,
            "carbon_kg_per_kwh": carbon_intensity,
            "raw_action_cmd": raw_cmd,
            "action_cmd": cmd,
            "battery_power_kw": actual_battery_power_kw,
            "soc": self._soc,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": grid_export_kw,
            "shield_intervened": shielded,
            "step_energy_cost": step_energy_cost,
            "step_peak_cost": step_peak_cost,
            "step_degradation_cost": step_degradation_cost,
            "step_carbon_kg": step_carbon_kg,
            "peak_demand_kw": self._peak_demand_kw,
            "stress": self._active_stress.name,
        }

        if terminated:
            solar_self_consumed = max(0.0, self._total_solar_kwh - self._total_grid_export_kwh)
            self_consumption_rate = (
                solar_self_consumed / self._total_solar_kwh if self._total_solar_kwh > 1e-6 else 0.0
            )
            peak_total_cost = self._peak_demand_kw * self.grid.peak_demand_penalty_per_kw
            total_operating_cost = (
                self._total_energy_cost
                + peak_total_cost
                + self._total_degradation_cost
                + self.weights.carbon * self._total_carbon_cost
            )
            info["episode_metrics"] = {
                "stress": self._active_stress.name,
                "total_energy_cost": self._total_energy_cost,
                "peak_demand_kw": self._peak_demand_kw,
                "peak_demand_cost": peak_total_cost,
                "battery_degradation_cost": self._total_degradation_cost,
                "total_carbon_kg": self._total_carbon_kg,
                "total_carbon_cost": self._total_carbon_cost,
                "total_operating_cost": total_operating_cost,
                "total_throughput_kwh": self._total_throughput_kwh,
                "total_grid_import_kwh": self._total_grid_import_kwh,
                "total_grid_export_kwh": self._total_grid_export_kwh,
                "self_consumption_rate": self_consumption_rate,
                "shield_intervention_rate": self._shield_interventions / max(1, len(data.demand_kw)),
                "total_reward": self._total_reward,
            }

        return self._current_obs, reward, terminated, truncated, info
