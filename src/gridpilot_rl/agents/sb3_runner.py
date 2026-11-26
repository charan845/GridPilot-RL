"""Stable-Baselines3 wrappers for PPO, SAC, DQN, and Physics-Guided Residual RL."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import torch
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gridpilot_rl.baselines.rule_based import TimeOfUseRuleBaseline
from gridpilot_rl.config import GridConfig, RewardWeights, StressConfig
from gridpilot_rl.data_loader import EpisodeSampler
from gridpilot_rl.envs.smart_grid_env import SmartGridEnv

SUPPORTED_ALGOS = ("ppo", "sac", "dqn", "residual_ppo", "residual_sac")


def _core_algo(algo: str) -> str:
    algo_lower = algo.lower()
    if algo_lower.startswith("residual_"):
        return algo_lower.replace("residual_", "", 1)
    return algo_lower


def _linear_schedule(initial_lr: float, min_fraction: float = 0.15) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return initial_lr * max(min_fraction, float(progress_remaining))

    return schedule


def get_action_mode_for_algo(algo: str) -> Literal["continuous", "discrete"]:
    """Return continuous or discrete action mode for a given algorithm name."""
    core = _core_algo(algo)
    if core == "dqn":
        return "discrete"
    if core in {"ppo", "sac"}:
        return "continuous"
    raise ValueError(f"Unsupported algorithm '{algo}'. Choose from: {SUPPORTED_ALGOS}")


def get_base_controller_for_algo(
    algo: str, grid_config: Optional[GridConfig] = None
) -> Optional[TimeOfUseRuleBaseline]:
    """Return the domain-heuristic base controller when using Physics-Guided Residual RL."""
    if algo.lower().startswith("residual_"):
        return TimeOfUseRuleBaseline(grid_config=grid_config)
    return None


def make_grid_env(
    episode_sampler: EpisodeSampler,
    grid_config: Optional[GridConfig] = None,
    reward_weights: Optional[RewardWeights] = None,
    stress: Optional[StressConfig | str] = "nominal_real",
    randomize_stress: Optional[Sequence[StressConfig | str]] = None,
    algo: str = "ppo",
    seed: int = 42,
    monitored: bool = True,
) -> SmartGridEnv | Monitor:
    """Construct and seed a SmartGridEnv configured for standard or Residual RL."""
    grid = grid_config or GridConfig()
    env = SmartGridEnv(
        grid_config=grid,
        reward_weights=reward_weights,
        episode_sampler=episode_sampler,
        stress=stress,
        randomize_stress=randomize_stress,
        action_mode=get_action_mode_for_algo(algo),
        base_controller=get_base_controller_for_algo(algo, grid),
    )
    env.reset(seed=seed)
    return Monitor(env) if monitored else env


class EpisodeMetricsCallback(BaseCallback):
    """Collects per-episode operating metrics emitted by SmartGridEnv."""

    def __init__(self, label: str = "", total_timesteps: int = 0, log_interval_pct: int = 25) -> None:
        super().__init__()
        self.history: list[dict[str, Any]] = []
        self.label = label
        self.total_timesteps = total_timesteps
        self.log_interval_pct = log_interval_pct
        self._next_log_pct = log_interval_pct

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_metrics" in info:
                entry = dict(info["episode_metrics"])
                entry["timesteps"] = int(self.num_timesteps)
                self.history.append(entry)
        if self.total_timesteps > 0 and self.history:
            pct = int(100 * self.num_timesteps / self.total_timesteps)
            if pct >= self._next_log_pct:
                recent = [h["total_operating_cost"] for h in self.history[-5:]]
                print(
                    f"    [{self.label}] {self.num_timesteps}/{self.total_timesteps} ({pct}%) | "
                    f"recent_cost=${sum(recent)/len(recent):.2f}",
                    flush=True,
                )
                self._next_log_pct += self.log_interval_pct
        return True


@dataclass
class TrainResult:
    """Container returned by train_agent."""

    algo: str
    model: BaseAlgorithm
    history: list[dict[str, Any]]
    model_path: Optional[Path] = None


def build_sb3_model(
    algo: str,
    env: Any,
    seed: int = 42,
    learning_rate: Optional[float] = None,
    device: str = "auto",
    total_timesteps: int = 35_000,
) -> BaseAlgorithm:
    """Instantiate a configured SB3 PyTorch model (PPO, SAC, DQN, Residual-PPO, Residual-SAC)."""
    algo_lower = algo.lower()
    is_residual = algo_lower.startswith("residual_")
    core = _core_algo(algo_lower)
    policy_kwargs: dict[str, Any] = dict(activation_fn=torch.nn.ReLU, net_arch=[192, 192])
    learning_starts = min(1000, max(50, total_timesteps // 6))

    if core == "ppo":
        policy_kwargs["log_std_init"] = -2.0 if is_residual else -0.5
        lr = learning_rate or (2.5e-4 if is_residual else 3e-4)
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=_linear_schedule(lr),
            n_steps=336,
            batch_size=128,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0005 if is_residual else 0.002,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=0,
        )
        if is_residual:
            with torch.no_grad():
                model.policy.action_net.weight.mul_(1e-3)
                model.policy.action_net.bias.zero_()
        return model

    if core == "sac":
        lr = learning_rate or (2.0e-4 if is_residual else 3e-4)
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=_linear_schedule(lr),
            buffer_size=100_000,
            learning_starts=learning_starts,
            batch_size=128,
            tau=0.005,
            gamma=0.995,
            train_freq=8,
            gradient_steps=2,
            ent_coef=0.004 if is_residual else "auto",
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=0,
        )
        if is_residual:
            with torch.no_grad():
                model.actor.mu.weight.mul_(1e-3)
                model.actor.mu.bias.zero_()
                model.actor.log_std.weight.mul_(1e-3)
                model.actor.log_std.bias.fill_(-2.0)
        return model

    if core == "dqn":
        lr = learning_rate or 3e-4
        return DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=_linear_schedule(lr),
            buffer_size=100_000,
            learning_starts=learning_starts,
            batch_size=128,
            gamma=0.995,
            train_freq=8,
            gradient_steps=2,
            target_update_interval=500,
            exploration_fraction=0.25,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=0,
        )

    raise ValueError(f"Unsupported algorithm '{algo}'. Choose from: {SUPPORTED_ALGOS}")


def train_agent(
    algo: str,
    episode_sampler: EpisodeSampler,
    total_timesteps: int = 35_000,
    grid_config: Optional[GridConfig] = None,
    reward_weights: Optional[RewardWeights] = None,
    stress: Optional[StressConfig | str] = "nominal_real",
    randomize_stress: Optional[Sequence[StressConfig | str]] = None,
    seed: int = 42,
    save_path: Optional[str | Path] = None,
    device: str = "auto",
) -> TrainResult:
    """Train PPO, SAC, DQN, Residual-PPO, or Residual-SAC with EvalCallback checkpointing."""
    core = _core_algo(algo)
    n_envs = 8 if core == "ppo" else 1

    def _make_worker(worker_idx: int) -> Callable[[], Monitor]:
        def _init() -> Monitor:
            return make_grid_env(
                episode_sampler=episode_sampler,
                grid_config=grid_config,
                reward_weights=reward_weights,
                stress=stress,
                randomize_stress=randomize_stress,
                algo=algo,
                seed=seed + worker_idx * 17,
                monitored=True,
            )  # type: ignore[return-value]

        return _init

    env = DummyVecEnv([_make_worker(i) for i in range(n_envs)])
    eval_env = make_grid_env(
        episode_sampler=episode_sampler,
        grid_config=grid_config,
        reward_weights=reward_weights,
        stress="nominal_real",
        algo=algo,
        seed=seed + 999,
        monitored=True,
    )

    model = build_sb3_model(
        algo=algo, env=env, seed=seed, device=device, total_timesteps=total_timesteps
    )
    history_cb = EpisodeMetricsCallback(label=algo.upper(), total_timesteps=total_timesteps)

    with tempfile.TemporaryDirectory() as tmp_dir:
        eval_freq = max(500, total_timesteps // (10 * n_envs))
        eval_cb = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=tmp_dir,
            n_eval_episodes=5,
            eval_freq=eval_freq,
            deterministic=True,
            verbose=0,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([history_cb, eval_cb]),
        )
        best_zip = Path(tmp_dir) / "best_model.zip"
        if best_zip.exists():
            model = load_agent(algo, best_zip, device=device)

    out_path: Optional[Path] = None
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out_path))

    env.close()
    eval_env.close()
    return TrainResult(algo=algo.lower(), model=model, history=history_cb.history, model_path=out_path)


def load_agent(
    algo: str,
    model_path: str | Path,
    device: str = "auto",
) -> BaseAlgorithm:
    """Load a saved SB3 model checkpoint."""
    core = _core_algo(algo)
    cls_map = {"ppo": PPO, "sac": SAC, "dqn": DQN}
    if core not in cls_map:
        raise ValueError(f"Unsupported algorithm '{algo}'.")
    return cls_map[core].load(str(model_path), device=device)
