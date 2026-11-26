"""Configuration dataclasses and robustness stress presets for GridPilot-RL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class GridConfig:
    """Physical, economic, and safety-shield parameters of the smart grid microgrid."""

    # Simulation timing
    timestep_hours: float = 1.0
    episode_steps: int = 168  # 7 days at hourly resolution

    # Battery energy storage system (BESS) parameters
    battery_capacity_kwh: float = 40.0
    max_charge_kw: float = 10.0
    max_discharge_kw: float = 10.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    soc_min: float = 0.10
    soc_max: float = 0.95
    initial_soc: float = 0.50

    # Physics Safety Shield parameters
    enable_safety_shield: bool = True
    ramp_limit_kw: float = 8.0  # Max hour-to-hour inverter power change (kW)
    pre_peak_reserve_soc: float = 0.25  # Minimum SoC floor reserved ahead of evening peak
    pre_peak_hours: tuple[int, int] = (12, 16)

    # Tariff & electrochemical aging parameters
    feed_in_tariff_ratio: float = 0.25  # Realistic retail export credit (~25% of retail import price)
    degradation_cost_per_kwh: float = 0.025  # Base $/kWh battery throughput cost
    soc_stress_penalty: float = 0.40  # Non-linear aging multiplier at extreme SoC (<0.20 or >0.85)
    ramp_stress_cost_per_kw: float = 0.004  # $/kW penalty for rapid power swings
    peak_demand_penalty_per_kw: float = 0.45  # $/kW incremental weekly peak demand charge
    carbon_price_per_kg: float = 0.06  # $/kg CO2 shadow carbon cost

    # Action discretization (for DQN)
    discrete_action_levels: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

    # Forecast-aware & uncertainty-aware scheduling configuration
    forecast_aware: bool = True
    forecast_uncertainty_aware: bool = True
    forecast_horizon: int = 6  # Look-ahead window in hours

    # Physics-Guided Residual RL mixing coefficient (u = clip(u_base + alpha * delta_u, -1, 1))
    residual_alpha: float = 0.25

    # Observation normalization scales
    demand_scale_kw: float = 15.0
    solar_scale_kw: float = 15.0
    price_scale_per_kwh: float = 0.50
    carbon_scale_kg_per_kwh: float = 0.70


@dataclass
class RewardWeights:
    """Multi-objective weights combining energy cost, peak demand, degradation, and carbon."""

    energy_cost: float = 1.0
    peak_demand: float = 1.0
    degradation: float = 1.0
    carbon: float = 0.5
    reward_scale: float = 0.20


@dataclass
class StressConfig:
    """Out-of-distribution stress perturbations applied to real OPSD episodes for robustness testing."""

    name: str = "nominal_real"
    demand_multiplier: float = 1.0
    solar_multiplier: float = 1.0
    cloud_dropout_prob: float = 0.0
    cloud_dropout_factor: float = 0.35
    price_multiplier: float = 1.0
    price_spike_prob: float = 0.0
    price_spike_multiplier: float = 2.0
    forecast_noise_std: float = 0.08  # Base relative forecast error at horizon H


STRESS_PRESETS: Dict[str, StressConfig] = {
    "nominal_real": StressConfig(name="nominal_real"),
    "high_demand": StressConfig(
        name="high_demand",
        demand_multiplier=1.35,
        forecast_noise_std=0.12,
    ),
    "cloudy_solar_drop": StressConfig(
        name="cloudy_solar_drop",
        solar_multiplier=0.65,
        cloud_dropout_prob=0.35,
        cloud_dropout_factor=0.30,
        forecast_noise_std=0.15,
    ),
    "volatile_tariff": StressConfig(
        name="volatile_tariff",
        price_multiplier=1.15,
        price_spike_prob=0.10,
        price_spike_multiplier=2.25,
        forecast_noise_std=0.14,
    ),
    "forecast_dropout": StressConfig(
        name="forecast_dropout",
        forecast_noise_std=0.35,
    ),
    "compound_stress": StressConfig(
        name="compound_stress",
        demand_multiplier=1.30,
        solar_multiplier=0.70,
        cloud_dropout_prob=0.25,
        cloud_dropout_factor=0.35,
        price_multiplier=1.15,
        price_spike_prob=0.08,
        price_spike_multiplier=2.10,
        forecast_noise_std=0.20,
    ),
}


def get_stress_preset(name: str) -> StressConfig:
    """Return a copy of a named stress perturbation preset."""
    if name not in STRESS_PRESETS:
        available = ", ".join(sorted(STRESS_PRESETS))
        raise ValueError(f"Unknown stress scenario '{name}'. Available: {available}")
    preset = STRESS_PRESETS[name]
    return StressConfig(**preset.__dict__)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    grid: GridConfig = field(default_factory=GridConfig)
    weights: RewardWeights = field(default_factory=RewardWeights)
    stress: StressConfig = field(default_factory=StressConfig)
    seed: int = 42
