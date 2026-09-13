from __future__ import annotations
import csv
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from .models import Piece

BOOKING_COLUMNS = ["awb", "qty", "length_cm", "width_cm", "height_cm", "weight_kg", "volume_cbm", "shc"]
PLAN_COLUMNS = ["uld", "type", "contour", "awbs"]
EST_RATIO = (1.25, 1.0, 0.8)     # 치수 없는 화물의 추정 박스 비율 (L:W:H)
NON_STACK_SHC = {"FRA", "TOP", "NST"}   # 위에 못 쌓는 SHC (예시)


@dataclass
class BookingLine:
    awb: str
    qty: int
    l: float | None
    w: float | None
    h: float | None
    weight_kg: float           # 이 행 전체 중량
    volume_cbm: float | None   # 이 행 전체 부피
    shc: str = ""

    @property
    def has_dims(self) -> bool:
        return all(v is not None and v > 0 for v in (self.l, self.w, self.h))


@dataclass
class PlanEntry:
    uld: str
    type: str
    contour: str
    awbs: list[tuple[str, int | None]] = field(default_factory=list)   # (awb, qty or None=전량)


def _num(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s == "":
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if isfinite(f) else None


def _rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        import openpyxl
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
        it = ws.iter_rows(values_only=True)
        header = [str(c).strip().lower() if c is not None else "" for c in next(it)]
        return [dict(zip(header, r)) for r in it if any(c is not None and str(c).strip() for c in r)]
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        rd.fieldnames = [h.strip().lower() for h in rd.fieldnames]
        return [r for r in rd if any((v or "").strip() for v in r.values())]


def read_booking(path: str | Path) -> list[BookingLine]:
    out = []
    for r in _rows(path):
        awb = str(r.get("awb", "")).strip()
        if not awb:
            continue
        out.append(BookingLine(
            awb=awb, qty=int(_num(r.get("qty")) or 0),
            l=_num(r.get("length_cm")), w=_num(r.get("width_cm")), h=_num(r.get("height_cm")),
            weight_kg=_num(r.get("weight_kg")) or 0.0, volume_cbm=_num(r.get("volume_cbm")),
            shc=str(r.get("shc") or "").strip().upper()))
    return out


def read_plan(path: str | Path) -> list[PlanEntry]:
    out = []
    for r in _rows(path):
        uld = str(r.get("uld", "")).strip()
        if not uld:
            continue
        awbs = []
        for tok in str(r.get("awbs") or "").replace(",", ";").split(";"):
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                a, q = tok.split(":", 1)
                awbs.append((a.strip(), int(q)))
            else:
                awbs.append((tok, None))
        out.append(PlanEntry(uld=uld, type=str(r.get("type") or "PMC").strip().upper(),
                             contour=str(r.get("contour") or "LD_160_FLAT").strip().upper(), awbs=awbs))
    return out


def estimate_dims(volume_cm3_each: float, ratio=EST_RATIO) -> tuple[float, float, float]:
    a, b, c = ratio
    s = (volume_cm3_each / (a * b * c)) ** (1 / 3)
    return round(a * s, 1), round(b * s, 1), round(c * s, 1)


def expand_pieces(lines: list[BookingLine]) -> tuple[list[Piece], list[str]]:
    """booking 행을 개별 Piece 로 펼친다. 치수 없으면 CBM/qty 로 추정. 둘 다 없으면 제외 + 경고."""
    pieces: list[Piece] = []
    warnings: list[str] = []
    counter: dict[str, int] = {}
    for ln in lines:
        if ln.qty <= 0:
            warnings.append(f"{ln.awb}: qty 0 → 제외")
            continue
        if ln.has_dims:
            l, w, h, est = ln.l, ln.w, ln.h, False
        elif ln.volume_cbm and ln.volume_cbm > 0:
            l, w, h = estimate_dims(ln.volume_cbm * 1e6 / ln.qty)
            est = True
        else:
            warnings.append(f"{ln.awb}: 치수도 CBM도 없음 → 판정 불가, 제외 ({ln.qty} pcs / {ln.weight_kg:.0f} kg)")
            continue
        per = ln.weight_kg / ln.qty
        stackable = not (set(ln.shc.replace("/", " ").split()) & NON_STACK_SHC)
        for _ in range(ln.qty):
            n = counter.get(ln.awb, 0) + 1
            counter[ln.awb] = n
            pieces.append(Piece(awb=ln.awb, pid=f"{ln.awb}#{n}", l=l, w=w, h=h, weight=per,
                                shc=ln.shc, estimated=est, stackable=stackable))
    return pieces, warnings
