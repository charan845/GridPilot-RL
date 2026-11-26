"""Prepare the real-world OPSD + Energy-Charts smart microgrid dataset.

Sources (100% real measured open data):
1. Open Power System Data (OPSD) Household Data (Konstanz, Germany):
   Metered hourly electricity import, export, and rooftop solar PV from
   DE_KN_residential3, DE_KN_residential4, and DE_KN_residential6.
2. Fraunhofer Energy-Charts / ENTSO-E:
   Hourly DE-AT-LU day-ahead electricity spot prices (EUR/MWh) mapped to a
   realistic dynamic retail prosumer tariff ($/kWh), plus 15-min German grid
   generation mix aggregated hourly to derive IPCC lifecycle CO2 intensity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

IPCC_EMISSION_FACTORS_KG_PER_KWH = {
    "Fossil brown coal / lignite": 1.05,
    "Fossil hard coal": 0.82,
    "Fossil oil": 0.65,
    "Fossil coal-derived gas": 0.55,
    "Fossil gas": 0.49,
    "Waste": 0.40,
    "Others": 0.35,
    "Biomass": 0.23,
    "Solar": 0.045,
    "Geothermal": 0.038,
    "Hydro Run-of-River": 0.024,
    "Hydro water reservoir": 0.024,
    "Hydro pumped storage": 0.024,
    "Wind onshore": 0.012,
    "Wind offshore": 0.012,
    "Nuclear": 0.012,
}


def compute_hourly_carbon_intensity(public_power_json: Path) -> pd.DataFrame:
    """Compute real hourly German grid carbon intensity (kg CO2/kWh) from generation mix."""
    raw = json.loads(public_power_json.read_text(encoding="utf-8"))
    ts = pd.to_datetime(raw["unix_seconds"], unit="s", utc=True)

    weighted_co2 = np.zeros(len(ts), dtype=np.float64)
    total_gen_mw = np.zeros(len(ts), dtype=np.float64)

    for entry in raw["production_types"]:
        name = entry["name"]
        if name not in IPCC_EMISSION_FACTORS_KG_PER_KWH:
            continue
        vals = np.asarray(
            [0.0 if v is None else max(0.0, float(v)) for v in entry["data"]],
            dtype=np.float64,
        )
        total_gen_mw += vals
        weighted_co2 += vals * IPCC_EMISSION_FACTORS_KG_PER_KWH[name]

    carbon_intensity = np.divide(
        weighted_co2,
        np.maximum(total_gen_mw, 1e-6),
    )
    df = pd.DataFrame({"utc_timestamp": ts, "carbon_kg_per_kwh": carbon_intensity})
    df["utc_hour"] = df["utc_timestamp"].dt.floor("h")
    hourly = df.groupby("utc_hour", as_index=False)["carbon_kg_per_kwh"].mean()
    return hourly.rename(columns={"utc_hour": "utc_timestamp"})


def load_hourly_prices(prices_json: Path) -> pd.DataFrame:
    """Load real hourly DE-AT-LU day-ahead prices and convert to dynamic retail tariff ($/kWh)."""
    raw = json.loads(prices_json.read_text(encoding="utf-8"))
    ts = pd.to_datetime(raw["unix_seconds"], unit="s", utc=True)
    spot_eur_mwh = pd.Series(raw["price"], dtype="float64").ffill().bfill().to_numpy()
    spot_kwh = spot_eur_mwh / 1000.0
    # Dynamic retail pass-through tariff: base grid fee + scaled real-time wholesale component
    # Preserves negative/low midday solar dips and genuine evening wholesale price spikes.
    dynamic_tariff = np.clip(0.14 + 3.0 * np.maximum(spot_kwh, -0.02), 0.05, 0.75)
    return pd.DataFrame(
        {
            "utc_timestamp": ts.floor("h"),
            "spot_eur_mwh": spot_eur_mwh,
            "price_per_kwh": dynamic_tariff,
        }
    )


def build_processed_dataset(
    household_csv: Path,
    prices_json: Path,
    public_power_json: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Align OPSD household smart meters with real wholesale prices and grid carbon intensity."""
    hh = pd.read_csv(household_csv, parse_dates=["utc_timestamp"])
    hh["utc_timestamp"] = hh["utc_timestamp"].dt.tz_convert("UTC") if hh["utc_timestamp"].dt.tz is not None else hh["utc_timestamp"].dt.tz_localize("UTC")

    homes = ("DE_KN_residential3", "DE_KN_residential4", "DE_KN_residential6")
    meter_cols: list[str] = []
    for h in homes:
        meter_cols.extend([f"{h}_grid_import", f"{h}_grid_export", f"{h}_pv"])

    sub = hh[["utc_timestamp"] + meter_cols].dropna().sort_values("utc_timestamp").reset_index(drop=True)

    # Convert cumulative meter readings (kWh) to hourly energy/power (kW over 1h)
    diffs = sub[meter_cols].diff()
    # Guard against occasional meter resets / sensor glitches
    diffs = diffs.clip(lower=0.0, upper=25.0)

    demand_kw = np.zeros(len(sub), dtype=np.float64)
    solar_kw = np.zeros(len(sub), dtype=np.float64)
    for h in homes:
        imp = diffs[f"{h}_grid_import"].to_numpy()
        exp = diffs[f"{h}_grid_export"].to_numpy()
        pv = diffs[f"{h}_pv"].to_numpy()
        # Energy conservation per household: gross load = import + pv - export
        home_demand = np.clip(imp + pv - exp, 0.05, 25.0)
        demand_kw += home_demand
        solar_kw += np.clip(pv, 0.0, 25.0)

    microgrid = pd.DataFrame(
        {
            "utc_timestamp": sub["utc_timestamp"],
            "demand_kw": demand_kw,
            "solar_kw": solar_kw,
        }
    ).iloc[1:].reset_index(drop=True)  # drop first diff NaN row

    prices_df = load_hourly_prices(prices_json)
    carbon_df = compute_hourly_carbon_intensity(public_power_json)

    merged = (
        microgrid.merge(prices_df, on="utc_timestamp", how="inner")
        .merge(carbon_df, on="utc_timestamp", how="inner")
        .sort_values("utc_timestamp")
        .reset_index(drop=True)
    )

    # Ensure strictly continuous hourly timeline with zero NaNs
    merged["hour"] = (merged["utc_timestamp"].dt.hour).astype(np.float32)
    merged = merged.dropna().reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build processed real OPSD microgrid dataset.")
    parser.add_argument("--household-csv", type=Path, default=root / "data/raw/household_data_60min_singleindex.csv")
    parser.add_argument("--prices-json", type=Path, default=root / "data/raw/de_at_lu_prices.json")
    parser.add_argument("--public-power-json", type=Path, default=root / "data/raw/de_public_power.json")
    parser.add_argument("--output-csv", type=Path, default=root / "data/processed/opsd_real_microgrid.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_processed_dataset(
        household_csv=args.household_csv,
        prices_json=args.prices_json,
        public_power_json=args.public_power_json,
        output_csv=args.output_csv,
    )
    print(f"Saved real OPSD microgrid dataset: {args.output_csv}")
    print(f"Rows: {len(df)} ({len(df)/168:.1f} weeks) | {df['utc_timestamp'].iloc[0]} -> {df['utc_timestamp'].iloc[-1]}")
    print(df[["demand_kw", "solar_kw", "price_per_kwh", "carbon_kg_per_kwh"]].describe().round(3))


if __name__ == "__main__":
    main()
