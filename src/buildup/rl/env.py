"""Gymnasium 환경: 박스를 하나씩 받아 팔레트 격자 위 (x, y, 회전) 을 고른다.

- 관측: 높이맵 2채널 (현재 높이 / 허용 높이, 0~1) + 다음 박스 K개의 (l, w, h, 부피) 정규화 벡터
- 행동: Discrete(nx * ny * 2). 무효 행동은 action_masks() 로 가림
- 보상: 놓은 박스 부피 / 컨투어 부피. 못 놓는 박스는 건너뛴다 (보상 0). 박스가 다 떨어지면 종료
- 커리큘럼: max_dim (각 변의 상한 cm). 박스 각 변은 [min_dim, max_dim] 에서 정수로 뽑는다
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..models import Piece
from ..heightmap import HeightMap
from ..uld import load_pallets, load_contours, limit_map, with_overhang

LOOKAHEAD = 4


def make_boxes(rng: np.random.Generator, min_dim: int, max_dim: int, envelope_cm3: float,
               fill: float = 1.4, cap: int = 300) -> list[Piece]:
    """부피 합이 컨투어 부피 × fill 을 넘을 때까지 박스를 뽑는다 (항상 남는 박스가 있도록)."""
    boxes, vol = [], 0.0
    while vol < envelope_cm3 * fill and len(boxes) < cap:
        l, w, h = (int(v) for v in rng.integers(min_dim, max_dim + 1, size=3))
        boxes.append(Piece(awb="RL", pid=f"b{len(boxes)}", l=l, w=w, h=h, weight=l * w * h / 1e6 * 150))
        vol += l * w * h
    return boxes


class BuildUpEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, pallet: str = "PMC", contour: str = "LD_160_FLAT", cell_cm: float = 10.0,
                 min_dim: int = 10, max_dim: int = 80, min_support: float = 0.7,
                 alternate_pallets: tuple[str, ...] = ("PMC", "PAG"), seed: int | None = None,
                 reward_scale: float = 100.0):
        super().__init__()
        self.reward_scale = reward_scale   # 부피 비율(0~1)에 곱함. 엔트로피 보너스에 묻히지 않게 100 = 퍼센트 단위
        self.pallets = load_pallets()
        self.contour = load_contours()[contour]
        self.cell = cell_cm
        self.min_dim, self.max_dim = min_dim, max_dim
        self.min_support = min_support
        self.alternate = alternate_pallets
        self.pallet_name = pallet
        self.rng = np.random.default_rng(seed)
        self._ep = 0
        # 격자 크기는 가장 큰 팔레트 기준으로 고정 (PAG 는 폭이 좁아 바깥 셀이 limit 0 으로 막힘)
        big = max((self.pallets[p] for p in alternate_pallets), key=lambda s: s.length_cm * s.width_cm)
        hm = self._new_map(big.code)
        self.nx, self.ny = hm.nx, hm.ny
        self.n_rot = 2
        self.action_space = spaces.Discrete(self.nx * self.ny * self.n_rot)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, shape=(2, self.nx, self.ny), dtype=np.float32),
            "boxes": spaces.Box(0.0, 1.0, shape=(LOOKAHEAD * 4,), dtype=np.float32),
        })
        self.hm: HeightMap | None = None
        self.boxes: list[Piece] = []
        self.idx = 0
        self._mask_cache: np.ndarray | None = None

    # ── 팔레트 ──
    def _new_map(self, code: str) -> HeightMap:
        spec = self.pallets[code]
        c = self.contour
        hm = HeightMap(spec.length_cm, spec.width_cm, limit_map(spec, c, self.cell), self.cell, c.overhang, c.overhang_slope)
        return hm

    def _new_map_padded(self, code: str) -> HeightMap:
        """작은 팔레트를 큰 격자에 맞춰 넣는다: 바깥 셀은 limit 0 (배치 불가)."""
        hm = self._new_map(code)
        if (hm.nx, hm.ny) == (self.nx, self.ny):
            return hm
        big = HeightMap(hm.length_cm, hm.width_cm, 0.0, self.cell, hm.overhang, hm.overhang_slope)
        # HeightMap 은 팔레트 치수로 격자를 잡으므로 직접 확장
        big.nx, big.ny = self.nx, self.ny
        big.height = np.zeros((self.nx, self.ny), dtype=np.float32)
        big.limit = np.zeros((self.nx, self.ny), dtype=np.float32)
        big.limit[:hm.nx, :hm.ny] = hm.limit
        big.limit0 = big.limit.copy()
        big.floor = np.zeros((self.nx, self.ny), dtype=bool)
        big.floor[:hm.nx, :hm.ny] = hm.floor
        big.geom = hm.geom
        return big

    # ── gym API ──
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._ep += 1
        code = self.alternate[self._ep % len(self.alternate)] if self.alternate else self.pallet_name
        self.pallet_name = code
        self.hm = self._new_map_padded(code)
        self.boxes = make_boxes(self.rng, self.min_dim, self.max_dim, self.hm.envelope_volume_cm3)
        self.idx = 0
        self._mask_cache = None
        self._skip_unplaceable()
        return self._obs(), {}

    def _current(self) -> Piece | None:
        return self.boxes[self.idx] if self.idx < len(self.boxes) else None

    def _mask_for(self, p: Piece) -> np.ndarray:
        m = np.zeros((self.n_rot, self.nx, self.ny), dtype=bool)
        for r, (l, w) in enumerate(((p.l, p.w), (p.w, p.l))):
            res = self.hm.candidates(l, w, p.h, self.min_support)
            if res is None:
                continue
            base, ok = res
            m[r, :ok.shape[0], :ok.shape[1]] = ok
        return m.reshape(-1)

    def action_masks(self) -> np.ndarray:
        if self._mask_cache is None:
            p = self._current()
            self._mask_cache = self._mask_for(p) if p is not None else np.zeros(self.action_space.n, dtype=bool)
            if not self._mask_cache.any():          # 종료 직전 방어: 아무거나 허용 (step 에서 무시)
                self._mask_cache[0] = True
        return self._mask_cache

    def _skip_unplaceable(self):
        """놓을 수 없는 박스는 건너뛴다."""
        while self.idx < len(self.boxes):
            m = self._mask_for(self.boxes[self.idx])
            if m.any():
                self._mask_cache = m
                return
            self.idx += 1
        self._mask_cache = None

    def step(self, action: int):
        p = self._current()
        if p is None:
            return self._obs(), 0.0, True, False, {}
        r, i, j = np.unravel_index(int(action), (self.n_rot, self.nx, self.ny))
        l, w = (p.l, p.w) if r == 0 else (p.w, p.l)
        mask = self.action_masks()
        reward = 0.0
        if mask[int(action)]:
            z = float(self.hm.height[i:i + self.hm.cells(l), j:j + self.hm.cells(w)].max())
            self.hm.place(p, int(i), int(j), z, l, w, p.h)
            reward = p.volume_cm3 / self.hm.envelope_volume_cm3 * self.reward_scale
        self.idx += 1
        self._mask_cache = None
        self._skip_unplaceable()
        done = self._current() is None
        info = {}
        if done:
            env_vol = self.hm.envelope_volume_cm3
            total = sum(b.volume_cm3 for b in self.boxes)
            info = {"utilization": self.hm.placed_volume / env_vol,
                    "fill": self.hm.placed_volume / min(env_vol, total),   # 채울 수 있는 부피 대비
                    "placed": len(self.hm.placements), "pallet": self.pallet_name}
        return self._obs(), float(reward), done, False, info

    def _obs(self):
        hmax = float(self.hm.limit0.max()) or 1.0
        grid = np.stack([self.hm.height / hmax, self.hm.limit / hmax]).astype(np.float32)
        b = np.zeros((LOOKAHEAD, 4), dtype=np.float32)
        for k in range(LOOKAHEAD):
            p = self.boxes[self.idx + k] if self.idx + k < len(self.boxes) else None
            if p is not None:
                b[k] = (p.l / 100, p.w / 100, p.h / 100, p.volume_cm3 / 1e6)
        return {"grid": grid, "boxes": b.reshape(-1)}

    # ── 커리큘럼 ──
    def set_max_dim(self, v: int):
        self.max_dim = int(max(self.min_dim, min(80, v)))
