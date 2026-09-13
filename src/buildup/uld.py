from __future__ import annotations
import json
from dataclasses import dataclass
from importlib import resources
from math import ceil
import numpy as np


@dataclass(frozen=True)
class ULDSpec:
    code: str
    length_cm: float
    width_cm: float
    max_gross_kg: float
    tare_kg: float

    @property
    def max_net_kg(self) -> float:
        return self.max_gross_kg - self.tare_kg


@dataclass(frozen=True)
class Contour:
    name: str
    height_cm: float
    side: str | None            # "x0" | "x1" | "y0" | "y1" | None
    profile: tuple[tuple[float, float], ...]   # (edge_dist_cm, max_height_cm)


def _load(name: str) -> dict:
    with resources.files("buildup.data").joinpath(name).open("r", encoding="utf-8") as f:
        d = json.load(f)
    d.pop("_note", None)
    return d


def load_pallets(path: str | None = None) -> dict[str, ULDSpec]:
    raw = json.load(open(path, encoding="utf-8")) if path else _load("pallets.json")
    raw.pop("_note", None)
    return {k: ULDSpec(code=k, **v) for k, v in raw.items()}


def load_contours(path: str | None = None) -> dict[str, Contour]:
    raw = json.load(open(path, encoding="utf-8")) if path else _load("contours.json")
    raw.pop("_note", None)
    return {k: Contour(name=k, height_cm=v["height_cm"], side=v.get("side"),
                       profile=tuple(tuple(p) for p in v.get("profile", []))) for k, v in raw.items()}


def grid_shape(spec: ULDSpec, cell_cm: float) -> tuple[int, int]:
    return ceil(spec.length_cm / cell_cm), ceil(spec.width_cm / cell_cm)


def limit_map(spec: ULDSpec, contour: Contour, cell_cm: float) -> np.ndarray:
    """셀별 허용 최대 높이. 컨투어의 깎인 면은 셀의 가장자리 쪽 거리를 써서 보수적으로 계산."""
    nx, ny = grid_shape(spec, cell_cm)
    lim = np.full((nx, ny), float(contour.height_cm), dtype=np.float32)
    if not contour.profile or contour.side is None:
        return lim
    xs = np.array([p[0] for p in contour.profile], dtype=float)
    ys = np.array([p[1] for p in contour.profile], dtype=float)
    if contour.side in ("x0", "x1"):
        idx = np.arange(nx)
        dist = idx * cell_cm if contour.side == "x0" else (nx - 1 - idx) * cell_cm
        vals = np.interp(dist, xs, ys, right=contour.height_cm).astype(np.float32)
        lim[:, :] = vals[:, None]
    else:
        idx = np.arange(ny)
        dist = idx * cell_cm if contour.side == "y0" else (ny - 1 - idx) * cell_cm
        vals = np.interp(dist, xs, ys, right=contour.height_cm).astype(np.float32)
        lim[:, :] = vals[None, :]
    return np.minimum(lim, contour.height_cm)
