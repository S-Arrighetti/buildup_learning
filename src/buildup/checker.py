from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from .models import Piece, PackResult
from .heightmap import HeightMap
from .packer import PackConfig, pack
from .booking import BookingLine, PlanEntry, expand_pieces
from .uld import ULDSpec, Contour, limit_map, load_pallets, load_contours, with_overhang


@dataclass
class ULDCheck:
    uld: str
    type: str
    contour: str
    awbs: list[str]
    assigned: int
    placed: int
    weight_kg: float
    max_net_kg: float
    utilization: float
    max_height_cm: float
    estimated_pcs: int
    status: str                 # OK | RISK | OVER
    reasons: list[str]
    result: PackResult
    hm: HeightMap

    @property
    def unplaced_by_awb(self) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        for u in self.result.unplaced:
            d[u.piece.awb] += 1
        return dict(d)


@dataclass
class Suggestion:
    awb: str
    pcs: int
    uld_from: str | None        # None = 플랜에 배정 안 된 화물
    uld_to: str | None          # None = 새 팔레트 필요


@dataclass
class CheckReport:
    ulds: list[ULDCheck]
    unassigned: dict[str, int]                  # booking 에 있지만 플랜에 없는 pcs
    unknown_awbs: list[tuple[str, str]]         # (uld, awb) 플랜에 있지만 booking 에 없음
    warnings: list[str]
    suggestions: list[Suggestion]
    extra_needed: list[Piece]                   # 어디에도 못 넣은 pcs

    @property
    def overall(self) -> str:
        if any(u.status == "OVER" for u in self.ulds) or self.extra_needed:
            return "OVER"
        if any(u.status == "RISK" for u in self.ulds) or self.unassigned:
            return "RISK"
        return "OK"


def _status(res: PackResult, max_net: float, est_pcs: int, risk: float, est_risk: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    n_fit = sum(1 for u in res.unplaced if u.reason == "fit")
    n_w = sum(1 for u in res.unplaced if u.reason == "weight")
    n_os = sum(1 for u in res.unplaced if u.reason == "oversize")
    if n_fit:
        reasons.append(f"공간 부족 {n_fit} pcs")
    if n_w:
        reasons.append(f"중량 초과 {n_w} pcs")
    if n_os:
        reasons.append(f"규격 초과 {n_os} pcs")
    if res.unplaced:
        return "OVER", reasons
    util = res.utilization
    if util >= risk:
        reasons.append(f"용적률 {util:.0%}")
    if est_pcs and util >= est_risk:
        reasons.append(f"추정 치수 {est_pcs} pcs + 용적률 {util:.0%}")
    if max_net and res.placed_weight_kg >= 0.95 * max_net:
        reasons.append(f"중량 {res.placed_weight_kg:.0f}/{max_net:.0f} kg")
    return ("RISK" if reasons else "OK"), reasons


def check(booking: list[BookingLine], plan: list[PlanEntry], cfg: PackConfig | None = None,
          pallets: dict[str, ULDSpec] | None = None, contours: dict[str, Contour] | None = None,
          risk_threshold: float = 0.85, est_risk_threshold: float = 0.75,
          overhang_cm: float | None = None) -> CheckReport:
    """overhang_cm 을 주면 모든 컨투어의 오버행 허용치를 그 값으로 덮어쓴다."""
    cfg = cfg or PackConfig()
    pallets = pallets or load_pallets()
    contours = contours or load_contours()

    pieces, warnings = expand_pieces(booking)
    pool: dict[str, list[Piece]] = defaultdict(list)
    for p in pieces:
        pool[p.awb].append(p)

    ulds: list[ULDCheck] = []
    unknown: list[tuple[str, str]] = []
    for e in plan:
        spec = pallets.get(e.type)
        if spec is None:
            warnings.append(f"{e.uld}: 모르는 ULD 타입 {e.type} → 건너뜀")
            continue
        contour = contours.get(e.contour)
        if contour is None:
            warnings.append(f"{e.uld}: 모르는 컨투어 {e.contour} → LD_160_FLAT 사용")
            contour = contours["LD_160_FLAT"]
        if overhang_cm is not None:
            contour = with_overhang(contour, overhang_cm)
        hm = HeightMap(spec.length_cm, spec.width_cm, limit_map(spec, contour, cfg.cell_cm),
                       cfg.cell_cm, contour.overhang)

        assigned: list[Piece] = []
        for awb, q in e.awbs:
            lst = pool.get(awb)
            if not lst:
                unknown.append((e.uld, awb))
                continue
            take = lst if q is None else lst[:q]
            assigned.extend(take)
            pool[awb] = lst[len(take):]
        res = pack(assigned, hm, cfg, spec.max_net_kg)
        est_pcs = sum(1 for p in assigned if p.estimated)
        status, reasons = _status(res, spec.max_net_kg, est_pcs, risk_threshold, est_risk_threshold)
        ulds.append(ULDCheck(uld=e.uld, type=e.type, contour=contour.name, awbs=[a for a, _ in e.awbs],
                             assigned=len(assigned), placed=len(res.placements),
                             weight_kg=res.placed_weight_kg, max_net_kg=spec.max_net_kg,
                             utilization=res.utilization, max_height_cm=res.max_height_cm,
                             estimated_pcs=est_pcs, status=status, reasons=reasons, result=res, hm=hm))

    unassigned = {awb: len(lst) for awb, lst in pool.items() if lst}

    # ── 대안 제안: 넘친 pcs 와 미배정 pcs 를 여유 있는 다른 팔레트에 넣어본다 (복사본 위에서) ──
    trial = {u.uld: u.hm.copy() for u in ulds}
    max_net = {u.uld: u.max_net_kg for u in ulds}
    todo: list[tuple[str | None, Piece]] = []
    for u in ulds:
        for un in u.result.unplaced:
            if un.reason != "oversize":
                todo.append((u.uld, un.piece))
    for awb, lst in pool.items():
        for p in lst:
            todo.append((None, p))

    counts: dict[tuple[str | None, str | None, str], int] = defaultdict(int)
    extra: list[Piece] = []
    for src, p in todo:
        targets = sorted((uid for uid in trial if uid != src),
                         key=lambda uid: -(trial[uid].envelope_volume_cm3 - trial[uid].placed_volume))
        placed_to = None
        for uid in targets:
            hm = trial[uid]
            r = pack([p], hm, cfg, max_net[uid])
            if not r.unplaced:
                placed_to = uid
                break
        if placed_to is None:
            extra.append(p)
        counts[(src, placed_to, p.awb)] += 1
    suggestions = [Suggestion(awb=awb, pcs=n, uld_from=src, uld_to=dst)
                   for (src, dst, awb), n in counts.items() if dst is not None]
    # 못 넣은 것은 출처별로 묶어서 "새 팔레트" 제안
    extra_counts: dict[tuple[str | None, str], int] = defaultdict(int)
    src_of = {id(p): s for s, p in todo}
    for p in extra:
        extra_counts[(src_of[id(p)], p.awb)] += 1
    suggestions += [Suggestion(awb=awb, pcs=n, uld_from=src, uld_to=None) for (src, awb), n in extra_counts.items()]

    return CheckReport(ulds=ulds, unassigned=unassigned, unknown_awbs=unknown, warnings=warnings,
                       suggestions=suggestions, extra_needed=extra)
