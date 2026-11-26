"""Exact perfect-foresight LP Oracle for computing the theoretical optimal lower bound."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linprog

from gridpilot_rl.config import GridConfig, RewardWeights
from gridpilot_rl.data_loader import GridEpisodeData


def solve_oracle_schedule(
    episode_data: GridEpisodeData,
    grid_config: Optional[GridConfig] = None,
    reward_weights: Optional[RewardWeights] = None,
) -> np.ndarray:
    """Solve a linear program with full foresight to get normalized commands in [-1, 1].

    Variables per timestep t in 0..T-1:
      0..T-1:       c_t (charge kW)
      T..2T-1:      d_t (discharge kW)
      2T..3T-1:     g_imp_t (grid import kW)
      3T..4T-1:     g_exp_t (grid export kW)
      4T..5T-1:     E_t (end-of-step battery energy kWh)
      5T:           P_peak (weekly max grid import kW)
    """
    grid = grid_config or GridConfig()
    weights = reward_weights or RewardWeights()
    T = len(episode_data.demand_kw)
    dt = grid.timestep_hours
    cap = grid.battery_capacity_kwh

    n_vars = 5 * T + 1
    idx_c = lambda t: t
    idx_d = lambda t: T + t
    idx_imp = lambda t: 2 * T + t
    idx_exp = lambda t: 3 * T + t
    idx_e = lambda t: 4 * T + t
    idx_peak = 5 * T

    # Objective vector
    obj = np.zeros(n_vars, dtype=np.float64)
    for t in range(T):
        price = float(episode_data.price_per_kwh[t])
        carbon = float(episode_data.carbon_kg_per_kwh[t])
        obj[idx_imp(t)] = (
            weights.energy_cost * dt * price
            + weights.carbon * dt * carbon * grid.carbon_price_per_kg
        )
        obj[idx_exp(t)] = -weights.energy_cost * dt * price * grid.feed_in_tariff_ratio
        obj[idx_c(t)] = weights.degradation * dt * grid.degradation_cost_per_kwh
        obj[idx_d(t)] = weights.degradation * dt * grid.degradation_cost_per_kwh
    obj[idx_peak] = weights.peak_demand * grid.peak_demand_penalty_per_kw

    # Variable bounds
    bounds: list[tuple[float, Optional[float]]] = []
    for _ in range(T):
        bounds.append((0.0, grid.max_charge_kw))
    for _ in range(T):
        bounds.append((0.0, grid.max_discharge_kw))
    for _ in range(T):
        bounds.append((0.0, None))
    for _ in range(T):
        bounds.append((0.0, None))
    for t in range(T):
        hour = float(episode_data.hours[t])
        min_soc = grid.soc_min
        if grid.enable_safety_shield and grid.pre_peak_hours[0] <= hour < grid.pre_peak_hours[1]:
            min_soc = max(min_soc, grid.pre_peak_reserve_soc)
        bounds.append((min_soc * cap, grid.soc_max * cap))
    bounds.append((0.0, None))

    # Equality constraints: battery SoC dynamics + microgrid power balance
    A_eq = np.zeros((2 * T, n_vars), dtype=np.float64)
    b_eq = np.zeros(2 * T, dtype=np.float64)

    e_init = float(np.clip(grid.initial_soc, grid.soc_min, grid.soc_max)) * cap
    for t in range(T):
        # E_t - E_{t-1} - eta_c*dt*c_t + (dt/eta_d)*d_t = 0
        row = t
        A_eq[row, idx_e(t)] = 1.0
        A_eq[row, idx_c(t)] = -grid.charge_efficiency * dt
        A_eq[row, idx_d(t)] = dt / grid.discharge_efficiency
        if t == 0:
            b_eq[row] = e_init
        else:
            A_eq[row, idx_e(t - 1)] = -1.0
            b_eq[row] = 0.0

        # g_imp_t - g_exp_t - c_t + d_t = demand_t - solar_t
        row_p = T + t
        A_eq[row_p, idx_imp(t)] = 1.0
        A_eq[row_p, idx_exp(t)] = -1.0
        A_eq[row_p, idx_c(t)] = -1.0
        A_eq[row_p, idx_d(t)] = 1.0
        b_eq[row_p] = float(episode_data.demand_kw[t] - episode_data.solar_kw[t])

    # Inequality constraints: peak demand + optional ramp-rate limits
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    for t in range(T):
        row = np.zeros(n_vars, dtype=np.float64)
        row[idx_imp(t)] = 1.0
        row[idx_peak] = -1.0
        ub_rows.append(row)
        ub_rhs.append(0.0)

    if grid.enable_safety_shield:
        for t in range(T):
            # (c_t - d_t) - (c_{t-1} - d_{t-1}) <= ramp_limit
            row_up = np.zeros(n_vars, dtype=np.float64)
            row_dn = np.zeros(n_vars, dtype=np.float64)
            row_up[idx_c(t)] = 1.0
            row_up[idx_d(t)] = -1.0
            row_dn[idx_c(t)] = -1.0
            row_dn[idx_d(t)] = 1.0
            if t > 0:
                row_up[idx_c(t - 1)] = -1.0
                row_up[idx_d(t - 1)] = 1.0
                row_dn[idx_c(t - 1)] = 1.0
                row_dn[idx_d(t - 1)] = -1.0
            ub_rows.extend([row_up, row_dn])
            ub_rhs.extend([grid.ramp_limit_kw, grid.ramp_limit_kw])

    A_ub = np.vstack(ub_rows)
    b_ub = np.asarray(ub_rhs, dtype=np.float64)

    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return np.zeros(T, dtype=np.float32)

    c_opt = res.x[:T]
    d_opt = res.x[T : 2 * T]
    net_bat = c_opt - d_opt
    commands = np.where(
        net_bat >= 0.0,
        net_bat / max(1e-6, grid.max_charge_kw),
        net_bat / max(1e-6, grid.max_discharge_kw),
    )
    return np.clip(commands, -1.0, 1.0).astype(np.float32)


class OracleSchedulePolicy:
    """Pre-solved LP schedule wrapped to match the predict() rollout interface."""

    name: str = "oracle_lp"

    def __init__(self, commands: np.ndarray) -> None:
        self.commands = commands
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        cmd = float(self.commands[min(self._idx, len(self.commands) - 1)])
        self._idx += 1
        return np.array([cmd], dtype=np.float32), None
