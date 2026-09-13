from __future__ import annotations
from .checker import CheckReport

_MARK = {"OK": "OK  ", "RISK": "RISK", "OVER": "OVER"}


def format_text(rep: CheckReport) -> str:
    lines: list[str] = []
    hdr = f"{'ULD':<10} {'TYPE':<5} {'CONTOUR':<15} {'AWB':>4} {'PCS':>9} {'WEIGHT kg':>14} {'UTIL':>6} {'HMAX':>6}  STATUS  NOTE"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for u in rep.ulds:
        pcs = f"{u.placed}/{u.assigned}"
        wt = f"{u.weight_kg:.0f}/{u.max_net_kg:.0f}"
        note = "; ".join(u.reasons)
        if u.estimated_pcs and "추정" not in note:
            note = (note + "; " if note else "") + f"추정 {u.estimated_pcs} pcs"
        lines.append(f"{u.uld:<10} {u.type:<5} {u.contour:<15} {len(u.awbs):>4} {pcs:>9} {wt:>14} "
                     f"{u.utilization:>6.0%} {u.max_height_cm:>6.0f}  {_MARK[u.status]}    {note}")
    lines.append("")
    lines.append(f"전체 판정: {rep.overall}")

    if rep.unassigned:
        lines.append("")
        lines.append("플랜에 없는 화물 (booking 에만 있음):")
        for awb, n in rep.unassigned.items():
            lines.append(f"  {awb}: {n} pcs")
    if rep.unknown_awbs:
        lines.append("")
        lines.append("booking 에 없는 AWB (플랜에만 있음):")
        for uld, awb in rep.unknown_awbs:
            lines.append(f"  {uld}: {awb}")

    over = [u for u in rep.ulds if u.result.unplaced]
    if over:
        lines.append("")
        lines.append("넘친 화물:")
        for u in over:
            for awb, n in u.unplaced_by_awb.items():
                lines.append(f"  {u.uld}: {awb} {n} pcs")

    if rep.suggestions:
        lines.append("")
        lines.append("제안:")
        for s in rep.suggestions:
            src = s.uld_from or "미배정"
            if s.uld_to:
                lines.append(f"  {src} → {s.uld_to}: {s.awb} {s.pcs} pcs 이동 가능")
            else:
                lines.append(f"  {src}: {s.awb} {s.pcs} pcs 는 추가 팔레트 필요")
    if rep.extra_needed:
        vol = sum(p.volume_cm3 for p in rep.extra_needed) / 1e6
        wt = sum(p.weight for p in rep.extra_needed)
        lines.append(f"  추가 팔레트 필요 합계: {len(rep.extra_needed)} pcs / {vol:.2f} m3 / {wt:.0f} kg")

    if rep.warnings:
        lines.append("")
        lines.append("경고:")
        lines.extend(f"  {w}" for w in rep.warnings)
    return "\n".join(lines)


def to_dict(rep: CheckReport) -> dict:
    return {
        "overall": rep.overall,
        "ulds": [{
            "uld": u.uld, "type": u.type, "contour": u.contour, "awbs": u.awbs,
            "assigned": u.assigned, "placed": u.placed,
            "weight_kg": round(u.weight_kg, 1), "max_net_kg": u.max_net_kg,
            "utilization": round(u.utilization, 4), "max_height_cm": u.max_height_cm,
            "estimated_pcs": u.estimated_pcs, "status": u.status, "reasons": u.reasons,
            "placements": [p.to_dict() for p in u.result.placements],
            "unplaced": [{"pid": x.piece.pid, "awb": x.piece.awb, "reason": x.reason,
                          "l": x.piece.l, "w": x.piece.w, "h": x.piece.h, "weight": x.piece.weight}
                         for x in u.result.unplaced],
            "grid": {"cell_cm": u.hm.cell, "nx": u.hm.nx, "ny": u.hm.ny,
                     "origin_cell": [u.hm.geom.ox, u.hm.geom.oy],   # 팔레트 모서리가 있는 셀
                     "pallet_cm": [u.hm.length_cm, u.hm.width_cm], "overhang_cm": u.hm.overhang,
                     "height": u.hm.height.round(1).tolist(), "limit": u.hm.limit.round(1).tolist()},
        } for u in rep.ulds],
        "unassigned": rep.unassigned,
        "unknown_awbs": rep.unknown_awbs,
        "suggestions": [s.__dict__ for s in rep.suggestions],
        "extra_needed": [{"pid": p.pid, "awb": p.awb, "l": p.l, "w": p.w, "h": p.h, "weight": p.weight}
                         for p in rep.extra_needed],
        "warnings": rep.warnings,
    }
