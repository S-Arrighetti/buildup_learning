# buildup_learning

Booking list와 예비 플랜(팔레트별 AWB 배정)을 받아서, 각 팔레트에 배정된 화물이
실제로 들어가는지(높이 160 컨투어, 팔레트 사이즈, 총중량) 검사하고,
안 들어가면 무엇이 얼마나 넘치는지, 어느 팔레트로 옮기면 되는지 제안하는 도구.

1단계(지금): 높이맵 기반 휴리스틱 패커로 판정.
2단계: 같은 환경 위에서 배치 순서·위치를 학습하는 모델(3D bin packing + RL).
3단계: 3D Build-Up 에 붙여서 결과를 눈으로 확인.

## 사용

```bash
# 설치 없이
set PYTHONPATH=src            # PowerShell: $env:PYTHONPATH="src"
python -m buildup gen --seed 1 --out data/samples        # 합성 샘플 생성
python -m buildup check data/samples/booking.csv data/samples/plan.csv --json out.json

# 또는
pip install -e .
buildup check booking.xlsx plan.xlsx
```

종료 코드: 0 = OK/RISK, 2 = OVER.

## 입력

`booking.csv` (또는 xlsx, 1행 헤더). 치수 그룹마다 한 행. 치수가 없으면 `volume_cbm` 으로 추정.

| 열 | 의미 |
|---|---|
| awb | AWB 번호 |
| qty | 이 행의 개수 |
| length_cm / width_cm / height_cm | 개당 치수 (없으면 비움) |
| weight_kg | 이 행 전체 중량 |
| volume_cbm | 이 행 전체 부피 (치수 없을 때만 사용) |
| shc | 특수화물 코드. FRA/TOP/NST 는 위에 못 쌓는 것으로 처리 |

`plan.csv`

| 열 | 의미 |
|---|---|
| uld | ULD 번호 |
| type | PMC / PAG / PLA / PKC (`src/buildup/data/pallets.json`) |
| contour | LD_160_FLAT / LD_160_CHAMFER / MD_244_FLAT (`contours.json`) |
| awbs | `;` 구분. `AWB:개수` 로 일부만 배정 가능 |

## 출력

```
ULD        TYPE  CONTOUR          AWB       PCS      WEIGHT kg   UTIL   HMAX  STATUS  NOTE
PMC10018   PMC   LD_160_FLAT        4   102/102      1441/6694    57%    138  OK      추정 22 pcs
PMC10083   PMC   LD_160_FLAT        2     20/36      1439/6694    64%    158  OVER    공간 부족 16 pcs
...
제안:
  PMC10083 → PMC10191: 180-72864735 14 pcs 이동 가능
  PMC10120: 180-55177592 18 pcs 는 추가 팔레트 필요
```

- OK / RISK(용적률 85% 이상, 추정 치수 포함 75% 이상, 중량 95% 이상) / OVER(못 넣은 pcs 있음)
- `--json` 으로 저장하면 팔레트별 배치 좌표(x, y, z, l, w, h)와 높이맵이 같이 나옴 → 3D 뷰어 입력용

## 판정 방식

팔레트 바닥을 5cm 격자로 나눈 높이맵. 셀마다 "현재 높이"와 "허용 높이(컨투어)"를 갖는다.
박스는 큰 것부터, 가장 낮게 놓이는 자리 → 뒤쪽 → 왼쪽 순으로 놓는다.
밑면 지지 70% 미만, 컨투어 초과, 팔레트 밖, 중량 초과면 못 놓는다.
휴리스틱이라 실제 숙련자보다 덜 채울 수 있다 → OK 는 믿어도 되고, OVER 는 "빡빡하다"로 읽을 것.

## 구조

```
src/buildup/
  models.py     Piece / Placement / PackResult
  uld.py        팔레트·컨투어 정의 로드, 허용 높이맵 생성
  heightmap.py  격자, 배치 후보 탐색(벡터화), 배치
  packer.py     휴리스틱 패커
  booking.py    booking / plan 읽기, 치수 추정, Piece 펼치기
  checker.py    팔레트별 판정 + 이동/추가 팔레트 제안
  report.py     텍스트 / JSON 출력
  synth.py      합성 데이터 생성
  data/         pallets.json, contours.json (예시값)
tests/          python -m unittest discover tests
data/samples/   합성 샘플
```

모든 샘플 데이터는 합성 데이터이며, 실제 업무 데이터는 저장소에 넣지 않는다 (`.gitignore` 에 xlsx/pdf/data/real 포함).
