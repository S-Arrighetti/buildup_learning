"""python -m unittest discover tests  (또는 pytest)"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from buildup.models import Piece
from buildup.heightmap import HeightMap
from buildup.packer import PackConfig, pack
from buildup.booking import BookingLine, PlanEntry, expand_pieces, estimate_dims, read_booking, read_plan
from buildup.checker import check
from buildup.uld import load_pallets, load_contours, limit_map
from buildup.synth import generate, write
from buildup.report import format_text, to_dict


def box(pid, l, w, h, wt=10.0, awb="A", **kw):
    return Piece(awb=awb, pid=pid, l=l, w=w, h=h, weight=wt, **kw)


class HeightMapTest(unittest.TestCase):
    def test_floor_placement_back_left(self):
        hm = HeightMap(100, 100, 160, cell_cm=5)
        pos = hm.best_position(50, 50, 30)
        self.assertEqual(pos, (0, 0, 0.0))

    def test_stack_needs_support(self):
        hm = HeightMap(100, 100, 160, cell_cm=5)
        hm.place(box("a", 50, 50, 30), 0, 0, 0, 50, 50, 30)
        # 50x50 위에 50x50 → 지지 100%, 바닥이 비어 있으니 바닥 우선 (더 낮음)
        i, j, z = hm.best_position(50, 50, 30)
        self.assertEqual(z, 0.0)
        # 바닥을 다 채우면 위로 올라간다
        hm.place(box("b", 50, 50, 30), 10, 0, 0, 50, 50, 30)
        hm.place(box("c", 100, 50, 30), 0, 10, 0, 100, 50, 30)
        i, j, z = hm.best_position(50, 50, 30)
        self.assertEqual(z, 30.0)

    def test_low_support_rejected(self):
        hm = HeightMap(100, 100, 160, cell_cm=5)
        hm.place(box("a", 20, 100, 30), 0, 0, 0, 20, 100, 30)   # 폭 전체, 길이 20 만 받침
        res = hm.candidates(100, 100, 30, min_support=0.7)
        base, ok = res
        self.assertFalse(ok.any())          # 20% 지지 → 불가

    def test_height_limit(self):
        hm = HeightMap(100, 100, 160, cell_cm=5)
        self.assertIsNone(hm.best_position(50, 50, 161))
        self.assertIsNotNone(hm.best_position(50, 50, 160))

    def test_non_stackable_blocks_top(self):
        hm = HeightMap(50, 50, 160, cell_cm=5)
        env0 = hm.envelope_volume_cm3
        hm.place(box("a", 50, 50, 30, stackable=False), 0, 0, 0, 50, 50, 30)
        self.assertIsNone(hm.best_position(50, 50, 30))
        self.assertEqual(hm.envelope_volume_cm3, env0)   # envelope 은 줄어들지 않는다

    def test_chamfer_limit_map(self):
        pallets, contours = load_pallets(), load_contours()
        lim = limit_map(pallets["PMC"], contours["LD_160_CHAMFER"], 5)
        self.assertEqual(lim.shape, (64 + 4, 49 + 4))   # 10cm 오버행 = 양쪽 2셀씩
        self.assertEqual(lim[:, 0].min(), 160)     # y0 쪽은 풀 높이
        self.assertEqual(lim[:, 50].max(), 100)    # y1 팔레트 가장자리 셀 = 100
        self.assertEqual(lim[:, -1].max(), 80)     # y1 오버행 끝 셀 (-10cm) = 80

    def test_overhang_allowed_with_support(self):
        # 100x100 팔레트, 오버행 10cm 전 면 → 격자 120x120. 110 길이 박스는 오버행으로 들어감 (지지 91%)
        hm = HeightMap(100, 100, 160, cell_cm=5, overhang=10, overhang_slope=0)
        self.assertEqual((hm.nx, hm.ny), (24, 24))
        # 팔레트 밖 바닥은 지지력 없음: 10x10 박스를 오버행 구역에만 놓는 건 불가
        base, ok = hm.candidates(10, 10, 10)
        self.assertFalse(ok[0, 0])                 # 격자 모서리 = 완전히 팔레트 밖
        self.assertTrue(ok[2, 2])                  # 팔레트 모서리
        self.assertIsNone(hm.best_position(125, 50, 30))   # 오버행 합쳐도 120 까지
        pos = hm.best_position(110, 50, 30)
        self.assertIsNotNone(pos)
        i, j, z = pos
        p = hm.place(box("a", 110, 50, 30), i, j, z, 110, 50, 30)
        self.assertLessEqual(p.x, 0)               # 팔레트 모서리 기준 음수 = 튀어나감

    def test_overhang_slope_1_to_1(self):
        # 1:1 경사: 10cm 튀어나오려면 바닥이 10cm 이상 떠 있어야 한다
        hm = HeightMap(100, 100, 160, cell_cm=5, overhang=10, overhang_slope=1.0)
        self.assertIsNone(hm.best_position(110, 50, 30))            # 바닥에서는 오버행 불가
        hm.place(box("base", 100, 100, 10), 2, 2, 0, 100, 100, 10)  # 10cm 받침
        self.assertIsNotNone(hm.best_position(110, 50, 30))         # z=10 → 10cm 오버행 OK
        hm2 = HeightMap(100, 100, 160, cell_cm=5, overhang=10, overhang_slope=1.0)
        hm2.place(box("base", 100, 100, 5), 2, 2, 0, 100, 100, 5)   # 5cm 받침
        self.assertIsNone(hm2.best_position(120, 50, 30))           # 한쪽 최소 10cm 튀어나감 > z=5 → 불가
        self.assertIsNotNone(hm2.best_position(110, 50, 30))        # 양쪽 5cm 씩이면 z=5 로 OK

    def test_overhang_support_threshold(self):
        hm = HeightMap(100, 100, 160, cell_cm=5, overhang=50, overhang_slope=0)
        # 100 길이 박스를 절반 걸치면 지지 50% → 불가, 30% 만 걸치면 70% → 가능
        base, ok = hm.candidates(100, 50, 30, min_support=0.7)
        ox = hm.geom.ox
        self.assertFalse(ok[ox - 10, ox])          # 50cm 튀어나감
        self.assertTrue(ok[ox - 6, ox])            # 30cm 튀어나감

    def test_no_overhang_contour(self):
        pallets, contours = load_pallets(), load_contours()
        c = contours["NO_OVERHANG_160"]
        hm = HeightMap(318, 244, limit_map(pallets["PMC"], c, 5), 5, c.overhang)
        self.assertEqual((hm.nx, hm.ny), (64, 49))
        self.assertIsNone(hm.best_position(320, 50, 30))


class PackerTest(unittest.TestCase):
    def test_fill_one_layer(self):
        hm = HeightMap(100, 100, 160, cell_cm=5)
        pieces = [box(f"p{i}", 50, 50, 40) for i in range(4)]
        res = pack(pieces, hm)
        self.assertEqual(len(res.placements), 4)
        self.assertEqual(res.max_height_cm, 40)
        self.assertEqual(len(res.unplaced), 0)

    def test_second_layer_and_overflow(self):
        hm = HeightMap(100, 100, 100, cell_cm=5)
        pieces = [box(f"p{i}", 50, 50, 40) for i in range(9)]   # 4 + 4 = 8 들어가고 1 남음
        res = pack(pieces, hm)
        self.assertEqual(len(res.placements), 8)
        self.assertEqual(res.unplaced[0].reason, "fit")

    def test_oversize_and_weight(self):
        hm = HeightMap(100, 100, 160, cell_cm=5)
        res = pack([box("big", 120, 50, 50), box("heavy", 10, 10, 10, wt=999)], hm, max_net_kg=500)
        reasons = {u.piece.pid: u.reason for u in res.unplaced}
        self.assertEqual(reasons, {"big": "oversize", "heavy": "weight"})

    def test_rotation_used(self):
        hm = HeightMap(100, 50, 160, cell_cm=5)
        res = pack([box("r", 50, 100, 20)], hm)   # 돌려야만 들어감
        self.assertEqual(len(res.placements), 1)
        self.assertEqual((res.placements[0].l, res.placements[0].w), (100, 50))


class BookingTest(unittest.TestCase):
    def test_estimate_dims_volume_preserved(self):
        l, w, h = estimate_dims(1e6)   # 1 m3
        self.assertAlmostEqual(l * w * h / 1e6, 1.0, places=1)

    def test_expand(self):
        lines = [
            BookingLine("X", 3, 50, 40, 30, 30.0, None, "FRA"),
            BookingLine("Y", 2, None, None, None, 20.0, 0.5, ""),
            BookingLine("Z", 1, None, None, None, 5.0, None, ""),
        ]
        pieces, warns = expand_pieces(lines)
        self.assertEqual(len(pieces), 5)
        self.assertFalse(pieces[0].stackable)
        self.assertTrue(pieces[3].estimated)
        self.assertEqual(len(warns), 1)
        self.assertIn("Z", warns[0])


class CheckerTest(unittest.TestCase):
    def test_over_and_suggestion(self):
        # PMC 하나에 318x244x100 두 장 → 두 번째는 높이 160 초과로 넘침, 빈 PMC 로 이동 제안
        booking = [BookingLine("A", 2, 318, 244, 100, 1000.0, None, "")]
        plan = [PlanEntry("PMC1", "PMC", "LD_160_FLAT", [("A", None)]),
                PlanEntry("PMC2", "PMC", "LD_160_FLAT", [])]
        rep = check(booking, plan, overhang_cm=0)
        self.assertEqual(rep.ulds[0].status, "OVER")
        self.assertEqual(rep.overall, "OVER")
        mv = [s for s in rep.suggestions if s.uld_to == "PMC2"]
        self.assertEqual(len(mv), 1)
        self.assertEqual((mv[0].awb, mv[0].pcs, mv[0].uld_from), ("A", 1, "PMC1"))
        self.assertEqual(rep.extra_needed, [])

    def test_unassigned_and_unknown(self):
        booking = [BookingLine("A", 1, 50, 50, 50, 10.0, None, ""), BookingLine("B", 1, 50, 50, 50, 10.0, None, "")]
        plan = [PlanEntry("PMC1", "PMC", "LD_160_FLAT", [("A", None), ("NOPE", None)])]
        rep = check(booking, plan)
        self.assertEqual(rep.unassigned, {"B": 1})
        self.assertEqual(rep.unknown_awbs, [("PMC1", "NOPE")])
        self.assertEqual(rep.overall, "RISK")

    def test_partial_qty_split(self):
        booking = [BookingLine("A", 4, 100, 100, 100, 40.0, None, "")]
        plan = [PlanEntry("P1", "PMC", "LD_160_FLAT", [("A", 3)]), PlanEntry("P2", "PMC", "LD_160_FLAT", [("A", 1)])]
        rep = check(booking, plan)
        self.assertEqual([u.assigned for u in rep.ulds], [3, 1])
        self.assertEqual(rep.unassigned, {})

    def test_synthetic_roundtrip(self):
        import tempfile
        b, p = generate(seed=3, n_awb=8)
        with tempfile.TemporaryDirectory() as d:
            bp, pp = write(d, b, p)
            rep = check(read_booking(bp), read_plan(pp))
        self.assertTrue(rep.ulds)
        for u in rep.ulds:
            self.assertIn(u.status, ("OK", "RISK", "OVER"))
            self.assertEqual(u.placed + len(u.result.unplaced), u.assigned)
        txt = format_text(rep)
        self.assertIn("전체 판정", txt)
        d = to_dict(rep)
        self.assertEqual(len(d["ulds"]), len(rep.ulds))
        from buildup.viewer import render_html
        html = render_html(d)
        self.assertIn('"overall"', html)
        self.assertNotIn("/*__DATA__*/", html)


if __name__ == "__main__":
    unittest.main()
