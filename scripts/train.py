"""Train PPO, SAC, DQN, Residual-PPO, or Residual-SAC on real OPSD data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gridpilot_rl.agents.sb3_runner import SUPPORTED_ALGOS, train_agent
from gridpilot_rl.config import GridConfig
from gridpilot_rl.data_loader import DEFAULT_DATA_PATH, load_opsd_splits
from gridpilot_rl.evaluation.plotting import plot_training_curves

DEFAULT_TRAIN_STRESS = (
    "nominal_real",
    "nominal_real",
    "high_demand",
    "cloudy_solar_drop",
    "volatile_tariff",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GridPilot-RL agents on real OPSD data.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--algos", nargs="+", default=["ppo", "sac", "dqn", "residual_ppo", "residual_sac"], choices=list(SUPPORTED_ALGOS))
    parser.add_argument("--timesteps", type=int, default=35_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forecast-horizon", type=int, default=6)
    parser.add_argument("--no-forecast", action="store_true")
    parser.add_argument("--no-uncertainty", action="store_true")
    parser.add_argument("--no-shield", action="store_true")
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_sampler, _ = load_opsd_splits(args.data_path)
    grid_cfg = GridConfig(
        forecast_aware=not args.no_forecast,
        forecast_uncertainty_aware=not args.no_uncertainty,
        forecast_horizon=args.forecast_horizon,
        enable_safety_shield=not args.no_shield,
    )
    randomize = DEFAULT_TRAIN_STRESS if args.domain_randomize else None
    tag = "uncertainty_aware" if (grid_cfg.forecast_aware and grid_cfg.forecast_uncertainty_aware) else (
        "point_forecast" if grid_cfg.forecast_aware else "forecast_blind"
    )

    models_dir = args.save_dir / "models"
    logs_dir = args.save_dir / "logs"
    figures_dir = args.save_dir / "figures"
    logs_dir.mkdir(parents=True, exist_ok=True)

    histories: dict[str, list[dict]] = {}
    for algo in args.algos:
        print(f"[Train] {algo.upper()} | steps={args.timesteps} | {tag} | shield={grid_cfg.enable_safety_shield}")
        save_path = models_dir / f"{algo}_{tag}.zip"
        res = train_agent(
            algo=algo,
            episode_sampler=train_sampler,
            total_timesteps=args.timesteps,
            grid_config=grid_cfg,
            stress="nominal_real",
            randomize_stress=randomize,
            seed=args.seed,
            save_path=save_path,
        )
        histories[algo] = res.history
        if res.history:
            last = res.history[-1]
            print(
                f"  -> saved {save_path} | episodes={len(res.history)} | "
                f"last_cost=${last['total_operating_cost']:.2f} | peak={last['peak_demand_kw']:.2f} kW"
            )

    (logs_dir / f"training_history_{tag}.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    plot_training_curves(histories, figures_dir / f"training_curves_{tag}.png")


if __name__ == "__main__":
    main()
