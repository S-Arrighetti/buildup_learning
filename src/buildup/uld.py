from __future__ import annotations
import json
from dataclasses import dataclass
from importlib import resources
from math import ceil
import numpy as np

SIDES = ("x0", "x1", "y0", "y1")


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
    side: str | None            # 깎인 면: "x0" | "x1" | "y0" | "y1" | None
    profile: tuple[tuple[float, float], ...]   # (팔레트 가장자리에서의 거리 cm, 허용 높이). 음수 = 오버행 구역
    overhang: dict[str, float]  # 면별 허용 오버행 cm


def parse_overhang(v) -> dict[str, float]:
    if v is None:
        return {s: 0.0 for s in SIDES}
    if isinstance(v, (int, float)):
        return {s: float(v) for s in SIDES}
    return {s: float(v.get(s, 0.0)) for s in SIDES}


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
                       profile=tuple(tuple(p) for p in v.get("profile", [])),
                       overhang=parse_overhang(v.get("overhang_cm")))
            for k, v in raw.items()}


def with_overhang(c: Contour, overhang_cm: float) -> Contour:
    return Contour(c.name, c.height_cm, c.side, c.profile, parse_overhang(overhang_cm))


@dataclass(frozen=True)
class Geometry:
    """격자 기하. 팔레트 바닥 + 오버행 구역. 원점 셀(ox, oy)이 팔레트 모서리."""
    cell: float
    nx: int
    ny: int
    ox: int          # x0 쪽 오버행 셀 수
    oy: int          # y0 쪽 오버행 셀 수
    bx: int          # 팔레트 바닥 길이 셀 수
    by: int          # 팔레트 바닥 폭 셀 수

    def floor_mask(self) -> np.ndarray:
        m = np.zeros((self.nx, self.ny), dtype=bool)
        m[self.ox:self.ox + self.bx, self.oy:self.oy + self.by] = True
        return m


def _cells(v: float, cell: float) -> int:
    return max(0, ceil(v / cell - 1e-9))


def geometry(length_cm: float, width_cm: float, cell_cm: float, overhang: dict[str, float] | None = None) -> Geometry:
    ov = parse_overhang(None) if overhang is None else overhang
    bx, by = max(1, _cells(length_cm, cell_cm)), max(1, _cells(width_cm, cell_cm))
    ox0, ox1 = _cells(ov["x0"], cell_cm), _cells(ov["x1"], cell_cm)
    oy0, oy1 = _cells(ov["y0"], cell_cm), _cells(ov["y1"], cell_cm)
    return Geometry(cell=float(cell_cm), nx=bx + ox0 + ox1, ny=by + oy0 + oy1, ox=ox0, oy=oy0, bx=bx, by=by)


def limit_map(spec: ULDSpec, contour: Contour, cell_cm: float) -> np.ndarray:
    """셀별 허용 최대 높이 (오버행 구역 포함). 깎인 면은 셀의 바깥쪽 거리를 써서 보수적으로."""
    g = geometry(spec.length_cm, spec.width_cm, cell_cm, contour.overhang)
    lim = np.full((g.nx, g.ny), float(contour.height_cm), dtype=np.float32)
    if contour.profile and contour.side is not None:
        xs = np.array([p[0] for p in contour.profile], dtype=float)
        ys = np.array([p[1] for p in contour.profile], dtype=float)
        if contour.side in ("x0", "x1"):
            idx = np.arange(g.nx)
            dist = (idx - g.ox) * cell_cm if contour.side == "x0" else (g.ox + g.bx - 1 - idx) * cell_cm
            vals = np.interp(dist, xs, ys, right=contour.height_cm).astype(np.float32)
            lim[:, :] = vals[:, None]
        else:
            idx = np.arange(g.ny)
            dist = (idx - g.oy) * cell_cm if contour.side == "y0" else (g.oy + g.by - 1 - idx) * cell_cm
            vals = np.interp(dist, xs, ys, right=contour.height_cm).astype(np.float32)
            lim[:, :] = vals[None, :]
    return np.minimum(lim, contour.height_cm)
