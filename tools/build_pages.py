"""GitHub Pages 용 정적 사이트 생성: 합성 샘플이 내장된 뷰어 + 빈 뷰어(드래그&드롭용)."""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buildup.synth import generate, write            # noqa: E402
from buildup.booking import read_booking, read_plan   # noqa: E402
from buildup.checker import check                     # noqa: E402
from buildup.report import to_dict                    # noqa: E402
from buildup.viewer import write_html, template       # noqa: E402


def main(out: str = "site", seed: int = 1) -> None:
    site = ROOT / out
    site.mkdir(parents=True, exist_ok=True)
    b, p = generate(seed=seed, n_awb=12)
    bp, pp = write(site / "sample", b, p)
    rep = check(read_booking(bp), read_plan(pp))
    write_html(to_dict(rep), site / "index.html")          # 샘플 내장
    (site / "viewer.html").write_text(template(), encoding="utf-8")   # 빈 뷰어
    (site / ".nojekyll").write_text("", encoding="utf-8")
    print(f"site → {site}  (overall {rep.overall}, {len(rep.ulds)} ulds)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
