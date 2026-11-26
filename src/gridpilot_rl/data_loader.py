"""Real-world OPSD + Energy-Charts data loader, chronological splitter, and rolling forecaster."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gridpilot_rl.config import StressConfig

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "opsd_real_microgrid.csv"
)


@dataclass
class GridEpisodeData:
    """Real microgrid time series, rolling forecasts, and calibrated uncertainty bands."""

    demand_kw: np.ndarray
    solar_kw: np.ndarray
    price_per_kwh: np.ndarray
    carbon_kg_per_kwh: np.ndarray
    hours: np.ndarray
    demand_forecast: np.ndarray  # shape (T, H)
    solar_forecast: np.ndarray  # shape (T, H)
    price_forecast: np.ndarray  # shape (T, H)
    demand_uncertainty: np.ndarray  # shape (T, H)
    solar_uncertainty: np.ndarray  # shape (T, H)
    price_uncertainty: np.ndarray  # shape (T, H)
    start_timestamp: str = ""


def _build_forecasts(
    series: np.ndarray,
    horizon: int,
    noise_std: float,
    rng: np.random.Generator,
    min_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate horizon-dependent imperfect forecasts and calibrated uncertainty stds (T, H)."""
    total_steps = len(series)
    forecasts = np.zeros((total_steps, horizon), dtype=np.float32)
    uncertainties = np.zeros((total_steps, horizon), dtype=np.float32)

    for t in range(total_steps):
        for h in range(1, horizon + 1):
            idx = min(t + h, total_steps - 1)
            base = float(series[idx])
            rel_std = noise_std * np.sqrt(h / max(1, horizon))
            abs_std = abs(base) * rel_std
            if noise_std > 0.0 and base > 0.0:
                noisy = base * (1.0 + rng.normal(0.0, rel_std))
                forecasts[t, h - 1] = max(min_value, noisy)
            else:
                forecasts[t, h - 1] = max(min_value, base)
            uncertainties[t, h - 1] = float(abs_std)
    return forecasts, uncertainties


def _apply_stress(
    demand: np.ndarray,
    solar: np.ndarray,
    price: np.ndarray,
    stress: StressConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply controlled out-of-distribution stress perturbations on top of real data."""
    d = demand * stress.demand_multiplier
    s = solar * stress.solar_multiplier
    if stress.cloud_dropout_prob > 0.0:
        cloud_mask = rng.random(len(s)) < stress.cloud_dropout_prob
        s = np.where(cloud_mask, s * stress.cloud_dropout_factor, s)

    p = price * stress.price_multiplier
    if stress.price_spike_prob > 0.0:
        spike_mask = rng.random(len(p)) < stress.price_spike_prob
        p = np.where(spike_mask, p * stress.price_spike_multiplier, p)

    return np.clip(d, 0.1, None), np.clip(s, 0.0, None), np.clip(p, 0.02, None)


class EpisodeSampler:
    """Samples contiguous real-world episodes from a chronologically split DataFrame."""

    def __init__(self, df: pd.DataFrame, split_name: str = "train") -> None:
        required = {"demand_kw", "solar_kw", "price_per_kwh", "carbon_kg_per_kwh", "hour"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
        if len(df) < 48:
            raise ValueError(f"Dataset split '{split_name}' too short: {len(df)} rows.")
        self.df = df.reset_index(drop=True)
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.df)

    def sample_episode(
        self,
        episode_steps: int = 168,
        forecast_horizon: int = 6,
        stress: Optional[StressConfig] = None,
        rng: Optional[np.random.Generator] = None,
        start_idx: Optional[int] = None,
    ) -> GridEpisodeData:
        """Slice a contiguous real episode window and compute rolling forecasts."""
        rng = rng or np.random.default_rng()
        stress = stress or StressConfig()
        needed = episode_steps + forecast_horizon + 1
        max_start = len(self.df) - needed
        if max_start < 0:
            raise ValueError(
                f"Split '{self.split_name}' has {len(self.df)} rows, need at least {needed}."
            )

        if start_idx is None:
            # Align start indices to midnight (or multiple of 24) when possible for diurnal consistency
            valid_starts = np.arange(0, max_start + 1, 24)
            s_idx = int(rng.choice(valid_starts)) if len(valid_starts) > 0 else int(rng.integers(0, max_start + 1))
        else:
            s_idx = int(np.clip(start_idx, 0, max_start))

        window = self.df.iloc[s_idx : s_idx + needed]
        demand_raw = window["demand_kw"].to_numpy(dtype=np.float32)
        solar_raw = window["solar_kw"].to_numpy(dtype=np.float32)
        price_raw = window["price_per_kwh"].to_numpy(dtype=np.float32)
        carbon = window["carbon_kg_per_kwh"].to_numpy(dtype=np.float32)
        hours = window["hour"].to_numpy(dtype=np.float32)
        start_ts = (
            str(window["utc_timestamp"].iloc[0]) if "utc_timestamp" in window.columns else str(s_idx)
        )

        demand, solar, price = _apply_stress(demand_raw, solar_raw, price_raw, stress, rng)

        demand_fc, demand_unc = _build_forecasts(
            demand, forecast_horizon, stress.forecast_noise_std, rng, min_value=0.1
        )
        solar_fc, solar_unc = _build_forecasts(
            solar, forecast_horizon, stress.forecast_noise_std * 1.35, rng, min_value=0.0
        )
        price_fc, price_unc = _build_forecasts(
            price, forecast_horizon, stress.forecast_noise_std * 0.90, rng, min_value=0.02
        )

        return GridEpisodeData(
            demand_kw=demand[:episode_steps].astype(np.float32),
            solar_kw=solar[:episode_steps].astype(np.float32),
            price_per_kwh=price[:episode_steps].astype(np.float32),
            carbon_kg_per_kwh=carbon[:episode_steps].astype(np.float32),
            hours=hours[:episode_steps].astype(np.float32),
            demand_forecast=demand_fc[:episode_steps],
            solar_forecast=solar_fc[:episode_steps],
            price_forecast=price_fc[:episode_steps],
            demand_uncertainty=demand_unc[:episode_steps],
            solar_uncertainty=solar_unc[:episode_steps],
            price_uncertainty=price_unc[:episode_steps],
            start_timestamp=start_ts,
        )


def load_opsd_splits(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    train_ratio: float = 0.75,
) -> tuple[EpisodeSampler, EpisodeSampler]:
    """Load processed real OPSD microgrid CSV and return chronological (train, test) samplers."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed OPSD dataset not found at {path}. Run scripts/prepare_real_data.py first."
        )
    df = pd.read_csv(path)
    split_idx = int(len(df) * train_ratio)
    # Align split to a 24-hour boundary to keep days intact
    split_idx = max(168, (split_idx // 24) * 24)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return EpisodeSampler(train_df, split_name="train"), EpisodeSampler(test_df, split_name="test")
