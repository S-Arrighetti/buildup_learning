from __future__ import annotations
import argparse
import json
import sys
from .booking import read_booking, read_plan
from .checker import check
from .packer import PackConfig
from .report import format_text, to_dict
from .uld import load_pallets, load_contours


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="buildup", description="ULD build-up 예비 플랜 검사기")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="booking + plan 을 읽어 팔레트별 적재 가능 여부 판정")
    c.add_argument("booking")
    c.add_argument("plan")
    c.add_argument("--json", help="결과(배치 좌표 포함)를 JSON 으로 저장")
    c.add_argument("--cell", type=float, default=5.0, help="격자 크기 cm (기본 5)")
    c.add_argument("--support", type=float, default=0.7, help="밑면 지지 비율 최소 (기본 0.7)")
    c.add_argument("--tip", action="store_true", help="박스 눕히기 허용")
    c.add_argument("--order", default="volume", choices=["volume", "area", "weight", "none"])
    c.add_argument("--pallets", help="팔레트 정의 JSON (기본 내장)")
    c.add_argument("--contours", help="컨투어 정의 JSON (기본 내장)")
    c.add_argument("--risk", type=float, default=0.85, help="RISK 판정 용적률 (기본 0.85)")
    c.add_argument("--overhang", type=float, default=None, help="오버행 허용 cm (전 면 동일, 컨투어 설정 덮어씀)")
    c.add_argument("--overhang-slope", type=float, default=None,
                   help="오버행 1cm 당 떠야 하는 높이 cm (기본 1.0 = 1:1, 0 = 제한 없음)")
    c.add_argument("--html", help="결과가 내장된 뷰어 HTML 저장 (더블클릭으로 열기)")

    g = sub.add_parser("gen", help="합성 booking.csv / plan.csv 생성")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="data/samples")
    g.add_argument("--awbs", type=int, default=12)
    g.add_argument("--ulds", type=int, default=None, help="팔레트 수 (기본: 총 부피에서 자동)")

    a = ap.parse_args(argv)
    try:  # Windows 콘솔 코드페이지(cp932 등)에서 한글 깨짐 방지
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if a.cmd == "gen":
        from .synth import generate, write
        b, p = generate(a.seed, a.awbs, a.ulds)
        bp, pp = write(a.out, b, p)
        print(f"wrote {bp} ({len(b)} rows), {pp} ({len(p)} ulds)")
        return 0

    cfg = PackConfig(cell_cm=a.cell, min_support=a.support, allow_tip=a.tip, order=a.order)
    rep = check(read_booking(a.booking), read_plan(a.plan), cfg,
                load_pallets(a.pallets), load_contours(a.contours), risk_threshold=a.risk,
                overhang_cm=a.overhang, overhang_slope=a.overhang_slope)
    print(format_text(rep))
    if a.json or a.html:
        d = to_dict(rep)
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
            print(f"\nJSON → {a.json}")
        if a.html:
            from .viewer import write_html
            write_html(d, a.html)
            print(f"HTML → {a.html}")
    return 0 if rep.overall != "OVER" else 2


if __name__ == "__main__":
    sys.exit(main())
