"""MaskablePPO 학습. 시간 예산(분) 안에서 박스 크기 커리큘럼 10 → 80 cm 를 올려가며 학습.

python -m buildup.rl.train --minutes 20 --envs 8 --out runs/ppo
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from .env import BuildUpEnv, LOOKAHEAD


class GridExtractor(BaseFeaturesExtractor):
    """높이맵 2채널 → CNN, 박스 벡터 → MLP, 합쳐서 features_dim."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        c, nx, ny = observation_space["grid"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, c, nx, ny)).shape[1]
        self.box = nn.Sequential(nn.Linear(LOOKAHEAD * 4, 64), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(n + 64, features_dim), nn.ReLU())

    def forward(self, obs) -> torch.Tensor:
        return self.out(torch.cat([self.cnn(obs["grid"]), self.box(obs["boxes"])], dim=1))


def mask_fn(env: BuildUpEnv):
    return env.action_masks()


def make_env(cell: float, min_dim: int, max_dim: int):
    def _f():
        return ActionMasker(BuildUpEnv(cell_cm=cell, min_dim=min_dim, max_dim=max_dim), mask_fn)
    return _f


class Curriculum(BaseCallback):
    """경과 시간 비율에 따라 max_dim 을 start → end 로 올리고, 예산이 끝나면 학습을 멈춘다. 롤아웃마다 CSV 로그."""

    def __init__(self, minutes: float, start: int, end: int, out: Path, hold_frac: float = 0.15):
        super().__init__()
        self.budget = minutes * 60
        self.start, self.end, self.hold = start, end, hold_frac
        self.out = out
        self.t0 = time.time()
        self.utils: list[float] = []
        self.by_pallet: dict[str, list[float]] = {}
        self.rows: list[dict] = []
        self.cur = start

    def _frac(self) -> float:
        return min(1.0, (time.time() - self.t0) / self.budget)

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_max_dim", self.start)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "fill" in info:
                self.utils.append(info["fill"])
                self.by_pallet.setdefault(info["pallet"], []).append(info["fill"])
        return time.time() - self.t0 < self.budget

    def _on_rollout_end(self) -> None:
        f = self._frac()
        # 처음 hold 구간은 start 유지, 마지막 hold 구간은 end 유지, 사이는 선형
        g = 0.0 if f < self.hold else 1.0 if f > 1 - self.hold else (f - self.hold) / (1 - 2 * self.hold)
        self.cur = int(round(self.start + (self.end - self.start) * g))
        self.training_env.env_method("set_max_dim", self.cur)
        el = time.time() - self.t0
        row = {"elapsed_s": round(el), "timesteps": self.num_timesteps, "max_dim": self.cur,
               "episodes": len(self.utils),
               "util_mean_recent": round(float(np.mean(self.utils[-64:])), 4) if self.utils else None,
               "fps": round(self.num_timesteps / el) if el else None}
        for k, v in self.by_pallet.items():
            row[f"util_{k}"] = round(float(np.mean(v[-32:])), 4)
        self.rows.append(row)
        print(f"[{row['elapsed_s']:5d}s] steps {row['timesteps']:>8} max_dim {self.cur:2d} "
              f"util {row['util_mean_recent']} fps {row['fps']}", flush=True)
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in self.rows for k in r}))
            w.writeheader()
            w.writerows(self.rows)
        self.model.save(self.out / "model")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--cell", type=float, default=10)
    ap.add_argument("--min-dim", type=int, default=10)
    ap.add_argument("--start-dim", type=int, default=10, help="커리큘럼 시작 max_dim")
    ap.add_argument("--end-dim", type=int, default=80)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="이어서 학습할 model.zip")
    ap.add_argument("--ent-coef", type=float, default=0.001,
                    help="엔트로피 계수. 행동 1,800개라 0.01 이면 보너스가 보상을 압도해 무작위로 수렴함")
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args(argv)

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    out = Path(a.out)
    vec_cls = SubprocVecEnv if a.envs > 1 else DummyVecEnv
    env = make_vec_env(make_env(a.cell, a.min_dim, a.start_dim), n_envs=a.envs, seed=a.seed, vec_env_cls=vec_cls)
    if a.resume:
        model = MaskablePPO.load(a.resume, env=env, ent_coef=a.ent_coef, learning_rate=a.lr)
    else:
        model = MaskablePPO(
            "MultiInputPolicy", env, seed=a.seed, verbose=0,
            n_steps=256, batch_size=512, n_epochs=4, learning_rate=a.lr, ent_coef=a.ent_coef,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            policy_kwargs=dict(features_extractor_class=GridExtractor,
                               features_extractor_kwargs=dict(features_dim=256),
                               net_arch=dict(pi=[256], vf=[256])))
    print(f"train: {a.envs} envs, cell {a.cell}, dims {a.min_dim}..{a.start_dim}→{a.end_dim}, budget {a.minutes} min", flush=True)
    cb = Curriculum(a.minutes, a.start_dim, a.end_dim, out)
    try:
        model.learn(total_timesteps=10 ** 9, callback=cb)
    except KeyboardInterrupt:
        pass
    model.save(out / "model")
    print(f"saved → {out / 'model.zip'}  ({cb.num_timesteps} steps, {len(cb.utils)} episodes)", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
