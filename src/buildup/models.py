from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Piece:
    """개별 박스 1개. 치수 cm, 중량 kg(개당)."""
    awb: str
    pid: str
    l: float
    w: float
    h: float
    weight: float
    shc: str = ""
    estimated: bool = False      # 치수를 CBM/개수에서 추정했는지
    stackable: bool = True       # False 면 위에 아무것도 못 올림

    @property
    def volume_cm3(self) -> float:
        return self.l * self.w * self.h


@dataclass
class Placement:
    pid: str
    awb: str
    x: float   # cm, 길이 방향 원점
    y: float   # cm, 폭 방향 원점
    z: float   # cm, 바닥 높이
    l: float   # 놓인 방향 기준 치수
    w: float
    h: float
    weight: float
    estimated: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Unplaced:
    piece: Piece
    reason: str   # "fit" | "weight" | "oversize"


@dataclass
class PackResult:
    placements: list[Placement] = field(default_factory=list)
    unplaced: list[Unplaced] = field(default_factory=list)
    placed_volume_cm3: float = 0.0
    envelope_volume_cm3: float = 0.0
    placed_weight_kg: float = 0.0
    max_height_cm: float = 0.0

    @property
    def utilization(self) -> float:
        return self.placed_volume_cm3 / self.envelope_volume_cm3 if self.envelope_volume_cm3 else 0.0
