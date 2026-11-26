"""Rule-based baseline controllers sharing the Stable-Baselines3 predict() interface."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from gridpilot_rl.config import GridConfig


class BaseRuleController:
    """Base class for rule-based smart grid controllers."""

    name: str = "base_rule"

    def __init__(self, grid_config: Optional[GridConfig] = None) -> None:
        self.grid = grid_config or GridConfig()

    def _decode_obs(self, obs: np.ndarray) -> dict[str, float]:
        soc_span = max(1e-6, self.grid.soc_max - self.grid.soc_min)
        soc = self.grid.soc_min + float(obs[0]) * soc_span
        demand_kw = float(obs[1]) * self.grid.demand_scale_kw
        solar_kw = float(obs[2]) * self.grid.solar_scale_kw
        price = float(obs[3]) * self.grid.price_scale_per_kwh
        carbon = float(obs[4]) * self.grid.carbon_scale_kg_per_kwh
        sin_h, cos_h = float(obs[5]), float(obs[6])
        hour = (np.arctan2(sin_h, cos_h) * 24.0 / (2.0 * np.pi)) % 24.0
        peak_so_far_kw = float(obs[7]) * self.grid.demand_scale_kw
        return {
            "soc": soc,
            "demand_kw": demand_kw,
            "solar_kw": solar_kw,
            "price": price,
            "carbon": carbon,
            "hour": float(hour),
            "peak_so_far_kw": peak_so_far_kw,
        }

    def compute_normalized_action(self, obs: np.ndarray) -> float:
        """Compute normalized battery command in [-1, 1] from observation vector."""
        state = self._decode_obs(np.asarray(obs, dtype=np.float32).reshape(-1))
        return float(np.clip(self._compute_from_state(state), -1.0, 1.0))

    def _compute_from_state(self, state: dict[str, float]) -> float:
        raise NotImplementedError

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        cmd = self.compute_normalized_action(observation)
        return np.array([cmd], dtype=np.float32), state


class IdleBaseline(BaseRuleController):
    """No-battery reference controller (always outputs zero charge/discharge)."""

    name: str = "no_battery"

    def _compute_from_state(self, state: dict[str, float]) -> float:
        return 0.0


class TimeOfUseRuleBaseline(BaseRuleController):
    """Time-of-Use tariff arbitrage and solar self-consumption heuristic."""

    name: str = "rule_tou"

    def __init__(
        self,
        grid_config: Optional[GridConfig] = None,
        low_price_threshold: float = 0.17,
        high_price_threshold: float = 0.265,
        target_charge_soc: float = 0.45,
    ) -> None:
        super().__init__(grid_config)
        self.low_price_threshold = low_price_threshold
        self.high_price_threshold = high_price_threshold
        self.target_charge_soc = target_charge_soc

    def _compute_from_state(self, state: dict[str, float]) -> float:
        soc = state["soc"]
        net_load_kw = state["demand_kw"] - state["solar_kw"]
        price = state["price"]
        hour = state["hour"]

        # 1. Capture surplus solar before exporting at discounted feed-in rate
        if net_load_kw < -0.10 and soc < self.grid.soc_max - 0.02:
            charge_kw = min(-net_load_kw, self.grid.max_charge_kw)
            return charge_kw / self.grid.max_charge_kw

        # 2. Discharge during evening peak hours or tariff spikes
        is_on_peak = (17.0 <= hour < 22.0) or (price >= self.high_price_threshold)
        if is_on_peak and net_load_kw > 0.2 and soc > self.grid.soc_min + 0.03:
            discharge_kw = min(net_load_kw, self.grid.max_discharge_kw)
            return -(discharge_kw / self.grid.max_discharge_kw)

        # 3. Gentle off-peak grid top-up only when tariff is unusually low (never spike peak)
        if price <= self.low_price_threshold and soc < self.target_charge_soc:
            headroom_kw = (self.target_charge_soc - soc) * self.grid.battery_capacity_kwh
            peak_budget = max(0.0, max(2.0, state["peak_so_far_kw"]) - max(0.0, net_load_kw))
            charge_kw = min(2.5, peak_budget, headroom_kw / self.grid.timestep_hours, self.grid.max_charge_kw)
            return charge_kw / self.grid.max_charge_kw

        return 0.0


class PeakShavingRuleBaseline(BaseRuleController):
    """Peak-demand-shaving + solar self-consumption rule-based controller."""

    name: str = "rule_peak_shave"

    def __init__(
        self,
        grid_config: Optional[GridConfig] = None,
        peak_target_kw: float = 2.0,
        charge_cap_kw: float = 2.0,
    ) -> None:
        super().__init__(grid_config)
        self.peak_target_kw = peak_target_kw
        self.charge_cap_kw = charge_cap_kw

    def _compute_from_state(self, state: dict[str, float]) -> float:
        soc = state["soc"]
        net_load_kw = state["demand_kw"] - state["solar_kw"]
        price = state["price"]
        hour = state["hour"]
        dynamic_target = max(self.peak_target_kw, 0.85 * state["peak_so_far_kw"])

        # Absorb solar excess first
        if net_load_kw < -0.10 and soc < self.grid.soc_max - 0.02:
            charge_kw = min(-net_load_kw, self.grid.max_charge_kw)
            return charge_kw / self.grid.max_charge_kw

        # Shave demand spikes above target
        if net_load_kw > dynamic_target and soc > self.grid.soc_min + 0.02:
            shave_kw = min(net_load_kw - dynamic_target, self.grid.max_discharge_kw)
            return -(shave_kw / self.grid.max_discharge_kw)

        # Discharge during evening / high-price hours to offset demand
        if (18.0 <= hour < 22.0 or price >= 0.27) and net_load_kw > 0.3 and soc > self.grid.soc_min + 0.08:
            discharge_kw = min(net_load_kw * 0.75, self.grid.max_discharge_kw)
            return -(discharge_kw / self.grid.max_discharge_kw)

        # Gentle overnight recharge only when very cheap without creating a new peak
        if price <= 0.17 and soc < 0.40 and net_load_kw < dynamic_target:
            available_headroom = max(0.0, dynamic_target - net_load_kw)
            charge_kw = min(self.charge_cap_kw, available_headroom, self.grid.max_charge_kw)
            return charge_kw / self.grid.max_charge_kw

        return 0.0


BASELINE_REGISTRY: dict[str, type[BaseRuleController]] = {
    "no_battery": IdleBaseline,
    "rule_tou": TimeOfUseRuleBaseline,
    "rule_peak_shave": PeakShavingRuleBaseline,
}


def get_baseline(name: str, grid_config: Optional[GridConfig] = None) -> BaseRuleController:
    """Instantiate a rule-based baseline controller by name."""
    if name not in BASELINE_REGISTRY:
        available = ", ".join(sorted(BASELINE_REGISTRY))
        raise ValueError(f"Unknown baseline '{name}'. Available: {available}")
    return BASELINE_REGISTRY[name](grid_config=grid_config)
