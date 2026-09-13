from __future__ import annotations
from dataclasses import dataclass
from .models import Piece, PackResult, Unplaced
from .heightmap import HeightMap


@dataclass
class PackConfig:
    cell_cm: float = 5.0
    min_support: float = 0.7      # 밑면 지지 비율 최소
    allow_tip: bool = False       # 옆으로 눕히기 허용
    order: str = "volume"         # volume | area | weight | none


def orientations(p: Piece, allow_tip: bool):
    seen = set()
    cand = [(p.l, p.w, p.h), (p.w, p.l, p.h)]
    if allow_tip:
        cand += [(p.l, p.h, p.w), (p.h, p.l, p.w), (p.w, p.h, p.l), (p.h, p.w, p.l)]
    for o in cand:
        if o not in seen:
            seen.add(o)
            yield o


def sort_pieces(pieces: list[Piece], order: str) -> list[Piece]:
    if order == "volume":
        return sorted(pieces, key=lambda p: (-p.volume_cm3, -p.weight))
    if order == "area":
        return sorted(pieces, key=lambda p: (-(p.l * p.w), -p.h, -p.weight))
    if order == "weight":
        return sorted(pieces, key=lambda p: (-p.weight, -p.volume_cm3))
    return list(pieces)


def _oversize(p: Piece, hm: HeightMap, allow_tip: bool) -> bool:
    """어떤 방향으로도 팔레트/컨투어 안에 들어갈 수 없는 크기인지."""
    hmax = float(hm.limit.max())
    for (l, w, h) in orientations(p, allow_tip):
        if l <= hm.length_cm and w <= hm.width_cm and h <= hmax:
            return False
    return True


def pack(pieces: list[Piece], hm: HeightMap, cfg: PackConfig | None = None,
         max_net_kg: float | None = None) -> PackResult:
    """휴리스틱 패커: 큰 것부터, 가장 낮게 놓이는 자리에 뒤-왼쪽 우선. hm 을 직접 갱신한다."""
    cfg = cfg or PackConfig()
    unplaced: list[Unplaced] = []
    for p in sort_pieces(pieces, cfg.order):
        if max_net_kg is not None and hm.placed_weight + p.weight > max_net_kg + 1e-6:
            unplaced.append(Unplaced(p, "weight"))
            continue
        best = None
        for (l, w, h) in orientations(p, cfg.allow_tip):
            pos = hm.best_position(l, w, h, cfg.min_support)
            if pos is None:
                continue
            i, j, z = pos
            key = (z + h, j, i)
            if best is None or key < best[0]:
                best = (key, i, j, z, l, w, h)
        if best is None:
            unplaced.append(Unplaced(p, "oversize" if _oversize(p, hm, cfg.allow_tip) else "fit"))
            continue
        _, i, j, z, l, w, h = best
        hm.place(p, i, j, z, l, w, h)
    return PackResult(placements=list(hm.placements), unplaced=unplaced,
                      placed_volume_cm3=hm.placed_volume, envelope_volume_cm3=hm.envelope_volume_cm3,
                      placed_weight_kg=hm.placed_weight, max_height_cm=hm.max_height)
