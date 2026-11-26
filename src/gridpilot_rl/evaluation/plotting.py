"""Publication-quality matplotlib visualization helpers for GridPilot-RL."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "no_battery": "#64748b",
    "rule_tou": "#f59e0b",
    "rule_peak_shave": "#fb7185",
    "dqn": "#8b5cf6",
    "ppo": "#2563eb",
    "sac": "#10b981",
    "residual_ppo": "#0284c7",
    "residual_sac": "#059669",
    "oracle_lp": "#111827",
}

DISPLAY_NAMES = {
    "no_battery": "No Battery",
    "rule_tou": "Rule (ToU)",
    "rule_peak_shave": "Rule (Peak Shave)",
    "dqn": "DQN",
    "ppo": "PPO",
    "sac": "SAC",
    "residual_ppo": "Residual-PPO",
    "residual_sac": "Residual-SAC",
    "oracle_lp": "Oracle (LP)",
}


def _display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _color(name: str) -> str:
    return PALETTE.get(name, "#0ea5e9")


def plot_training_curves(
    histories: Mapping[str, Sequence[dict]],
    output_path: str | Path,
    window: int = 8,
) -> Path:
    """Plot training convergence (reward and weekly operating cost) across RL agents."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)
    for algo, records in histories.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        steps = df["timesteps"].to_numpy()
        reward = df["total_reward"].rolling(window, min_periods=1).mean()
        cost = df["total_operating_cost"].rolling(window, min_periods=1).mean()

        color = _color(algo)
        label = _display_name(algo)
        axes[0].plot(steps, reward, label=label, color=color, linewidth=2.0)
        axes[1].plot(steps, cost, label=label, color=color, linewidth=2.0)

    axes[0].set_title("Episode Reward During Training", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Environment Timesteps")
    axes[0].set_ylabel("Smoothed Episode Reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8.5)

    axes[1].set_title("Weekly Operating Cost During Training", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Environment Timesteps")
    axes[1].set_ylabel("Operating Cost ($ / week)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_multi_objective_comparison(
    summary_df: pd.DataFrame,
    output_path: str | Path,
    scenario: str = "nominal_real",
) -> Path:
    """Plot Energy Cost, Peak Demand, Degradation, and CVaR-20% Tail Cost."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sub = summary_df[summary_df["scenario"] == scenario].copy()
    order = [
        "no_battery",
        "rule_tou",
        "rule_peak_shave",
        "dqn",
        "ppo",
        "sac",
        "residual_ppo",
        "residual_sac",
        "oracle_lp",
    ]
    sub["sort_key"] = sub["policy"].apply(lambda p: order.index(p) if p in order else 99)
    sub = sub.sort_values("sort_key")

    labels = [_display_name(p) for p in sub["policy"]]
    colors = [_color(p) for p in sub["policy"]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.2), dpi=150)
    panels = [
        ("total_energy_cost_mean", "total_energy_cost_std", "Energy Cost ($ / week)"),
        ("peak_demand_kw_mean", "peak_demand_kw_std", "Peak Grid Demand (kW)"),
        ("battery_degradation_cost_mean", "battery_degradation_cost_std", "Battery Degradation ($ / week)"),
        ("cvar20_operating_cost", None, "CVaR-20% Tail Operating Cost ($ / week)"),
    ]

    for ax, (mean_col, std_col, title) in zip(axes.flat, panels):
        means = sub[mean_col].to_numpy()
        stds = sub[std_col].fillna(0.0).to_numpy() if std_col else None
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=colors,
            alpha=0.92,
            edgecolor="#1e293b",
            linewidth=0.6,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    fig.suptitle(
        f"Multi-Objective & Tail-Risk Benchmark on Real OPSD Data ({scenario})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dispatch_schedule(
    traces_by_policy: Mapping[str, pd.DataFrame],
    output_path: str | Path,
    hours_to_show: int = 48,
) -> Path:
    """Plot a 48-hour dispatch slice on real OPSD load/solar/prices."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    first_df = next(iter(traces_by_policy.values())).iloc[:hours_to_show]
    t = first_df["step"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(13.0, 8.8), sharex=True, dpi=150)

    ax1 = axes[0]
    ax1.plot(t, first_df["demand_kw"], label="OPSD Demand (kW)", color="#0f172a", linewidth=2.0)
    ax1.fill_between(t, 0, first_df["solar_kw"], label="OPSD Solar PV (kW)", color="#facc15", alpha=0.45)
    ax1_r = ax1.twinx()
    ax1_r.step(
        t,
        first_df["price_per_kwh"],
        where="mid",
        label="Dynamic Tariff ($/kWh)",
        color="#ef4444",
        linestyle="--",
        linewidth=1.6,
    )
    ax1.set_ylabel("Power (kW)")
    ax1_r.set_ylabel("Tariff ($/kWh)", color="#ef4444")
    ax1.set_title("48-Hour Real OPSD Microgrid Dispatch Comparison", fontsize=12, fontweight="bold")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", ncol=3, fontsize=8.5)
    ax1.grid(alpha=0.25)

    ax2 = axes[1]
    for policy_name, df in traces_by_policy.items():
        sub = df.iloc[:hours_to_show]
        ls = "--" if policy_name in {"no_battery", "oracle_lp"} else "-"
        ax2.plot(
            sub["step"],
            sub["grid_import_kw"],
            label=_display_name(policy_name),
            color=_color(policy_name),
            linestyle=ls,
            linewidth=1.85,
        )
    ax2.set_ylabel("Grid Import (kW)")
    ax2.set_title("Net Grid Import (Peak Shaving & Price Arbitrage)", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="upper left", ncol=4, fontsize=8.0)
    ax2.grid(alpha=0.25)

    ax3 = axes[2]
    for policy_name, df in traces_by_policy.items():
        if policy_name == "no_battery":
            continue
        sub = df.iloc[:hours_to_show]
        ls = "--" if policy_name == "oracle_lp" else "-"
        ax3.plot(
            sub["step"],
            sub["soc"] * 100.0,
            label=_display_name(policy_name),
            color=_color(policy_name),
            linestyle=ls,
            linewidth=1.85,
        )
    ax3.set_ylabel("Battery SoC (%)")
    ax3.set_xlabel("Hour")
    ax3.set_ylim(0, 105)
    ax3.set_title("Battery State of Charge Trajectory", fontsize=10.5, fontweight="bold")
    ax3.legend(loc="upper left", ncol=4, fontsize=8.0)
    ax3.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_robustness_comparison(
    summary_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot operating cost reduction (%) across out-of-distribution stress scenarios."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sub = summary_df[~summary_df["policy"].isin(["no_battery"])].copy()
    scenarios = list(dict.fromkeys(sub["scenario"]))
    preferred_order = [
        "rule_tou",
        "rule_peak_shave",
        "dqn",
        "ppo",
        "sac",
        "residual_ppo",
        "residual_sac",
        "oracle_lp",
    ]
    policies = [p for p in preferred_order if p in set(sub["policy"])]

    x = np.arange(len(scenarios))
    width = 0.84 / max(1, len(policies))

    fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=150)
    for idx, policy in enumerate(policies):
        vals = []
        for scen in scenarios:
            match = sub[(sub["scenario"] == scen) & (sub["policy"] == policy)]
            vals.append(float(match["operating_cost_reduction_pct"].iloc[0]) if len(match) else 0.0)
        offset = (idx - (len(policies) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=_display_name(policy),
            color=_color(policy),
            edgecolor="#1e293b",
            linewidth=0.5,
        )

    ax.axhline(0.0, color="#334155", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9.0)
    ax.set_ylabel("Total Operating Cost Reduction vs No-Battery (%)")
    ax.set_title("Cross-Scenario Policy Robustness (Real OPSD + Stress Perturbations)", fontsize=12, fontweight="bold")
    ax.legend(ncol=4, loc="upper right", fontsize=8.2)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pareto_cost_vs_degradation(
    summary_df: pd.DataFrame,
    output_path: str | Path,
    scenario: str = "nominal_real",
) -> Path:
    """Scatter plot of Energy Cost vs Battery Degradation (Pareto trade-off)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sub = summary_df[summary_df["scenario"] == scenario].copy()
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)

    for _, row in sub.iterrows():
        pol = str(row["policy"])
        x = float(row["battery_degradation_cost_mean"])
        y = float(row["total_energy_cost_mean"])
        size = 90 + 14 * float(row["peak_demand_kw_mean"])
        ax.scatter(
            x,
            y,
            s=size,
            color=_color(pol),
            edgecolor="#0f172a",
            linewidth=1.0,
            alpha=0.9,
            zorder=3,
        )
        ax.annotate(
            _display_name(pol),
            (x, y),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xlabel("Battery Degradation Cost ($ / week)")
    ax.set_ylabel("Weekly Energy Cost ($ / week)")
    ax.set_title(f"Cost vs. Battery Aging Trade-off ({scenario}; bubble size = Peak kW)", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_forecast_ablation(
    ablation_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Compare Forecast-Blind vs Point-Forecast vs Uncertainty-Aware RL policies."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    modes = ["forecast_blind", "point_forecast", "uncertainty_aware"]
    mode_labels = {
        "forecast_blind": "Forecast-Blind (H=0)",
        "point_forecast": "Point Forecast (H=6)",
        "uncertainty_aware": "Uncertainty-Aware (H=6, mu+sigma)",
    }
    mode_colors = {
        "forecast_blind": "#94a3b8",
        "point_forecast": "#38bdf8",
        "uncertainty_aware": "#0284c7",
    }

    algos = list(dict.fromkeys(ablation_df["algo"]))
    x = np.arange(len(algos))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=150)
    for i, mode in enumerate(modes):
        vals = []
        for algo in algos:
            sub = ablation_df[(ablation_df["algo"] == algo) & (ablation_df["forecast_mode"] == mode)]
            vals.append(float(sub["total_operating_cost_mean"].mean()) if len(sub) else 0.0)
        offset = (i - 1.0) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=mode_labels[mode],
            color=mode_colors[mode],
            edgecolor="#1e293b",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val,
                f"${val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_display_name(a) for a in algos], fontsize=10)
    ax.set_ylabel("Mean Weekly Operating Cost ($)")
    ax.set_title("Forecast & Uncertainty Ablation Across All Scenarios", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
