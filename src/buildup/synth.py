"""합성 booking / plan 생성기. 실제 데이터와 무관한 난수 샘플."""
from __future__ import annotations
import csv
import random
from pathlib import Path
from .booking import BOOKING_COLUMNS, PLAN_COLUMNS
from .uld import load_pallets

ENVELOPE_PMC_CM3 = 318 * 244 * 160


def awb_number(rng: random.Random, prefix: str = "180") -> str:
    serial = rng.randint(1000000, 9999999)
    return f"{prefix}-{serial}{serial % 7}"


def _piece_type(rng: random.Random) -> tuple[tuple[float, float, float], int]:
    """(치수, 개수). 카톤은 많고 작게, 크레이트/대형은 적게."""
    r = rng.random()
    if r < 0.65:      # 카톤
        return (rng.randint(30, 80), rng.randint(25, 60), rng.randint(20, 60)), rng.randint(5, 40)
    if r < 0.9:       # 크레이트
        return (rng.randint(90, 140), rng.randint(70, 120), rng.randint(60, 120)), rng.randint(1, 6)
    return (rng.randint(100, 200), rng.randint(80, 120), rng.randint(100, 155)), rng.randint(1, 2)   # 대형


def generate(seed: int = 0, n_awb: int = 12, n_uld: int | None = None, unknown_ratio: float = 0.3,
             fill_lo: float = 0.55, fill_hi: float = 0.95) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    booking: list[dict] = []
    awb_vol: dict[str, float] = {}
    for _ in range(n_awb):
        awb = awb_number(rng)
        known = rng.random() >= unknown_ratio
        shc = rng.choice(["", "", "", "GEN", "FRA", "DGR", "PER"])
        for _ in range(rng.randint(1, 3) if known else 1):
            (l, w, h), qty = _piece_type(rng)
            vol_each = l * w * h
            density = rng.uniform(80, 300)                     # kg/m3
            weight = round(vol_each / 1e6 * density * qty, 1)
            row = {"awb": awb, "qty": qty, "weight_kg": weight, "shc": shc}
            if known:
                row.update(length_cm=l, width_cm=w, height_cm=h, volume_cbm="")
            else:
                row.update(length_cm="", width_cm="", height_cm="", volume_cbm=round(vol_each * qty / 1e6, 3))
            booking.append(row)
            awb_vol[awb] = awb_vol.get(awb, 0.0) + vol_each * qty

    pallets = load_pallets()
    awbs = list(awb_vol)
    rng.shuffle(awbs)
    if n_uld is None:   # 총 부피 기준 자동 (약간 모자라게 잡아서 넘침/미배정 케이스가 섞이게)
        n_uld = max(1, round(sum(awb_vol.values()) / (ENVELOPE_PMC_CM3 * 0.75)))
    plan: list[dict] = []
    k = 0
    for u in range(n_uld):
        typ = "PMC" if rng.random() < 0.7 else "PAG"
        spec = pallets[typ]
        contour = "LD_160_CHAMFER" if rng.random() < 0.2 else "LD_160_FLAT"
        env = spec.length_cm * spec.width_cm * 160
        target = env * rng.uniform(fill_lo, fill_hi)
        acc = 0.0
        chosen = []
        while k < len(awbs) and (not chosen or acc + awb_vol[awbs[k]] <= target):
            chosen.append(awbs[k])
            acc += awb_vol[awbs[k]]
            k += 1
        plan.append({"uld": f"{typ}{10000 + u * 37 + rng.randint(0, 30)}", "type": typ,
                     "contour": contour, "awbs": ";".join(chosen)})
    # 남은 AWB 는 일부러 플랜에 안 넣는다 (미배정 케이스)
    return booking, plan


def write(out_dir: str | Path, booking: list[dict], plan: list[dict]) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bp, pp = out / "booking.csv", out / "plan.csv"
    with open(bp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BOOKING_COLUMNS)
        w.writeheader()
        w.writerows(booking)
    with open(pp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLUMNS)
        w.writeheader()
        w.writerows(plan)
    return bp, pp
