import unittest

from vidsum.segment import fmt_range, fmt_ts, plan_segments


class PlanSegmentsTest(unittest.TestCase):
    def test_short_input_is_single_segment(self):
        segs = plan_segments(300, target=600, minimum=240, maximum=1200)
        self.assertEqual([(s.index, s.start, s.end) for s in segs], [(0, 0.0, 300.0)])

    def test_fixed_cut_without_silences(self):
        segs = plan_segments(3000, target=600, minimum=240, maximum=1200)
        self.assertEqual([s.start for s in segs], [0.0, 600.0, 1200.0, 1800.0, 2400.0])
        self.assertEqual(segs[-1].end, 3000.0)

    def test_cut_snaps_to_nearest_silence(self):
        silences = [(595.0, 605.0), (1795.0, 1805.0)]
        segs = plan_segments(3000, target=600, minimum=240, maximum=1200,
                             silences=silences)
        self.assertEqual(segs[0].end, 600.0)
        # 均分后切点为 600/1200/1800/2400，其中 600 与 1800 吸附到静音中点
        self.assertEqual([s.end for s in segs],
                         [600.0, 1200.0, 1800.0, 2400.0, 3000.0])
        self.assertEqual(len(segs), 5)

    def test_silence_outside_window_is_ignored(self):
        # 静音离目标位置太远，不能为了它把段长拉到窗口之外
        silences = [(100.0, 110.0), (2900.0, 2910.0)]
        segs = plan_segments(3000, target=600, minimum=240, maximum=1200,
                             silences=silences)
        self.assertEqual(segs[0].end, 600.0)

    def test_no_segment_exceeds_maximum(self):
        segs = plan_segments(7200, target=600, minimum=240, maximum=1200)
        self.assertTrue(all(s.duration <= 1200 + 1e-6 for s in segs))
        self.assertTrue(all(s.duration >= 240 - 1e-6 for s in segs[:-1]))

    def test_segments_are_contiguous(self):
        segs = plan_segments(5000, target=700, minimum=200, maximum=1400)
        self.assertEqual(segs[0].start, 0.0)
        self.assertEqual(segs[-1].end, 5000.0)
        for prev, nxt in zip(segs, segs[1:]):
            self.assertEqual(prev.end, nxt.start)

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            plan_segments(0)
        with self.assertRaises(ValueError):
            plan_segments(100, minimum=500, maximum=100)

    def test_degenerate_silence_span_ignored(self):
        segs = plan_segments(3000, target=600, minimum=240, maximum=1200,
                             silences=[(10.0, 10.0), (600.0, 600.0)])
        self.assertEqual(segs[0].end, 600.0)


class FormattingTest(unittest.TestCase):
    def test_fmt_ts(self):
        self.assertEqual(fmt_ts(0), "00:00")
        self.assertEqual(fmt_ts(65), "01:05")
        self.assertEqual(fmt_ts(3661), "1:01:01")
        self.assertEqual(fmt_ts(65, always_hours=True), "0:01:05")
        self.assertEqual(fmt_ts(-5), "00:00")

    def test_fmt_range_switches_to_hours(self):
        self.assertEqual(fmt_range(0, 600), "00:00\u201310:00")
        self.assertEqual(fmt_range(3600, 7200), "1:00:00\u20132:00:00")


if __name__ == "__main__":
    unittest.main()
