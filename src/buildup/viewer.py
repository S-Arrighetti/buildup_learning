"""결과 JSON 을 내장한 단일 HTML 뷰어 생성. 템플릿은 data/viewer.html (서버 없이 더블클릭으로 열림)."""
from __future__ import annotations
import json
from importlib import resources
from pathlib import Path

PLACEHOLDER = "/*__DATA__*/null"


def template() -> str:
    return resources.files("buildup.data").joinpath("viewer.html").read_text(encoding="utf-8")


def render_html(report_dict: dict) -> str:
    payload = json.dumps(report_dict, ensure_ascii=False).replace("</", "<\\/")
    tpl = template()
    assert PLACEHOLDER in tpl, "viewer.html 에 데이터 자리표시자가 없음"
    return tpl.replace(PLACEHOLDER, payload)


def write_html(report_dict: dict, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(render_html(report_dict), encoding="utf-8")
    return p
