"""학습 모델 vs 휴리스틱 vs 무작위 비교. 같은 박스 세트, 같은 격자.

python -m buildup.rl.evaluate runs/ppo/model.zip --episodes 10 --dims 20,40,60,80
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ..packer import PackConfig, pack
from .env import BuildUpEnv


def run_policy(env: BuildUpEnv, seed: int, pick) -> dict:
    obs, _ = env.reset(seed=seed)
    done, info = False, {}
    while not done:
        mask = env.action_masks()
        a = pick(obs, mask)
        obs, r, done, _, info = env.step(a)
    return info


def heuristic_on(env: BuildUpEnv, seed: int, order: str) -> dict:
    env.reset(seed=seed)                      # 같은 박스 세트/팔레트를 얻기 위해
    hm = env._new_map_padded(env.pallet_name)
    res = pack(list(env.boxes), hm, PackConfig(cell_cm=env.cell, min_support=env.min_support, order=order))
    total = sum(b.volume_cm3 for b in env.boxes)
    return {"utilization": res.utilization, "fill": res.placed_volume_cm3 / min(hm.envelope_volume_cm3, total),
            "placed": len(res.placements), "pallet": env.pallet_name}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--dims", default="20,40,60,80")
    ap.add_argument("--cell", type=float, default=10)
    ap.add_argument("--out", default=None, help="markdown 저장 경로")
    a = ap.parse_args(argv)

    model = None
    if a.model:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(a.model, device="cpu")

    rng = np.random.default_rng(123)
    rows = []
    for md in (int(x) for x in a.dims.split(",")):
        env = BuildUpEnv(cell_cm=a.cell, min_dim=10, max_dim=md)
        res: dict[str, list[float]] = {"random": [], "heur_order": [], "heur_sorted": [], "model": []}
        t_model = 0.0
        for ep in range(a.episodes):
            seed = int(rng.integers(0, 2 ** 31))
            env._ep = ep                       # PMC / PAG 번갈아
            res["random"].append(run_policy(env, seed, lambda o, m: int(rng.choice(np.flatnonzero(m))))["fill"])
            env._ep = ep
            res["heur_order"].append(heuristic_on(env, seed, "none")["fill"])
            env._ep = ep
            res["heur_sorted"].append(heuristic_on(env, seed, "volume")["fill"])
            if model is not None:
                env._ep = ep
                t = time.perf_counter()
                res["model"].append(run_policy(env, seed, lambda o, m: int(model.predict(o, action_masks=m, deterministic=True)[0]))["fill"])
                t_model += time.perf_counter() - t
        row = {"max_dim": md, "n": a.episodes}
        for k, v in res.items():
            if v:
                row[k] = round(float(np.mean(v)), 3)
        if model is not None:
            row["model_s_per_ep"] = round(t_model / a.episodes, 2)
        rows.append(row)
        print(row, flush=True)

    cols = ["max_dim", "n", "random", "heur_order", "heur_sorted", "model", "model_s_per_ep"]
    cols = [c for c in cols if any(c in r for r in rows)]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    text = "\n".join(md)
    print("\n" + text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
