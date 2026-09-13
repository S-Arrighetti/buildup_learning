from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .models import Piece, Placement
from .uld import geometry, parse_overhang


class HeightMap:
    """팔레트 격자 (바닥 + 오버행 구역).
    height = 현재 적재 높이, limit = 컨투어 허용 높이 (cm). floor = 팔레트 바닥인 셀 (바닥 밖은 지지력 없음).
    좌표계: 팔레트 모서리가 (0, 0). 오버행 구역은 음수 좌표."""

    def __init__(self, length_cm: float, width_cm: float, limit: np.ndarray | float,
                 cell_cm: float = 5.0, overhang: dict[str, float] | float | None = None):
        self.length_cm = float(length_cm)
        self.width_cm = float(width_cm)
        self.overhang = parse_overhang(overhang)
        self.geom = geometry(length_cm, width_cm, cell_cm, self.overhang)
        self.cell = self.geom.cell
        self.nx, self.ny = self.geom.nx, self.geom.ny
        self.floor = self.geom.floor_mask()
        self.height = np.zeros((self.nx, self.ny), dtype=np.float32)
        if np.isscalar(limit):
            self.limit = np.full((self.nx, self.ny), float(limit), dtype=np.float32)
        else:
            assert limit.shape == (self.nx, self.ny), f"limit shape {limit.shape} != {(self.nx, self.ny)}"
            self.limit = limit.astype(np.float32).copy()
        self.limit0 = self.limit.copy()       # 원래 컨투어 (non-stackable 로 limit 이 줄어도 envelope 은 고정)
        self.placements: list[Placement] = []
        self.placed_weight = 0.0
        self.placed_volume = 0.0

    # ── helpers ──
    def cells(self, size_cm: float) -> int:
        return max(1, int(np.ceil(size_cm / self.cell - 1e-9)))

    @property
    def grid_length_cm(self) -> float:
        return self.nx * self.cell

    @property
    def grid_width_cm(self) -> float:
        return self.ny * self.cell

    @property
    def envelope_volume_cm3(self) -> float:
        return float(self.limit0.sum()) * self.cell * self.cell

    @property
    def max_height(self) -> float:
        return float(self.height.max()) if self.height.size else 0.0

    def copy(self) -> "HeightMap":
        hm = HeightMap.__new__(HeightMap)
        hm.__dict__.update(self.__dict__)
        hm.height = self.height.copy()
        hm.limit = self.limit.copy()
        hm.placements = list(self.placements)
        return hm

    # ── placement search ──
    def candidates(self, l: float, w: float, h: float, min_support: float = 0.7):
        """모든 (i, j) 원점에 대해 base 높이와 배치 가능 여부를 벡터로 계산.
        지지 = 밑면 셀 중 (높이 == base) 이면서 (base > 0 이거나 팔레트 바닥인) 셀의 비율."""
        li, lj = self.cells(l), self.cells(w)
        if li > self.nx or lj > self.ny:
            return None
        win = sliding_window_view(self.height, (li, lj))          # (nx-li+1, ny-lj+1, li, lj)
        base = win.max(axis=(2, 3))
        limwin = sliding_window_view(self.limit, (li, lj)).min(axis=(2, 3))
        fits = (base + h) <= limwin + 1e-6
        # 격자 반올림과 별개로 실제 cm 경계 검사 (팔레트 + 오버행 허용치)
        ov = self.overhang
        xs = (np.arange(base.shape[0]) - self.geom.ox) * self.cell
        ys = (np.arange(base.shape[1]) - self.geom.oy) * self.cell
        in_x = (xs >= -ov["x0"] - 1e-6) & (xs + l <= self.length_cm + ov["x1"] + 1e-6)
        in_y = (ys >= -ov["y0"] - 1e-6) & (ys + w <= self.width_cm + ov["y1"] + 1e-6)
        fits &= in_x[:, None] & in_y[None, :]
        b = base[:, :, None, None]
        floor_win = sliding_window_view(self.floor, (li, lj))
        supported = (win >= b - 1e-6) & (floor_win | (b > 1e-6))
        support = supported.mean(axis=(2, 3))
        ok = fits & (support >= min_support)
        return base, ok

    def best_position(self, l: float, w: float, h: float, min_support: float = 0.7):
        """가장 낮게 놓이는 자리, 같으면 뒤쪽(y 작음), 그다음 왼쪽(x 작음). (i, j, z) 반환."""
        res = self.candidates(l, w, h, min_support)
        if res is None:
            return None
        base, ok = res
        if not ok.any():
            return None
        idx = np.argwhere(ok)
        top = base[ok] + h
        order = np.lexsort((idx[:, 0], idx[:, 1], top))
        i, j = idx[order[0]]
        return int(i), int(j), float(base[i, j])

    def place(self, piece: Piece, i: int, j: int, z: float, l: float, w: float, h: float) -> Placement:
        li, lj = self.cells(l), self.cells(w)
        self.height[i:i + li, j:j + lj] = z + h
        if not piece.stackable:
            self.limit[i:i + li, j:j + lj] = np.minimum(self.limit[i:i + li, j:j + lj], z + h)
        p = Placement(pid=piece.pid, awb=piece.awb,
                      x=(i - self.geom.ox) * self.cell, y=(j - self.geom.oy) * self.cell, z=z,
                      l=l, w=w, h=h, weight=piece.weight)
        self.placements.append(p)
        self.placed_weight += piece.weight
        self.placed_volume += piece.volume_cm3
        return p
