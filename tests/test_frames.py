"""画面帧的规划、落盘、清单与多模态输入。"""

import json
import unittest
from pathlib import Path

from vidsum import frames
from vidsum.segment import Segment
from tests.support import TempDirCase


def _segments(*spans):
    return [Segment(i, float(start), float(end))
            for i, (start, end) in enumerate(spans)]


class PlanFrameTimesTest(unittest.TestCase):
    def test_even_split_avoids_first_and_last_second(self):
        plan = frames.plan_frame_times(_segments((0.0, 600.0)), 2)
        times = plan[0]
        self.assertEqual(len(times), 2)
        for t in times:
            self.assertGreater(t, 1.0)
            self.assertLess(t, 599.0)
        # 两点关于段中点对称
        self.assertAlmostEqual(sum(times) / 2.0, 300.0, places=3)

    def test_single_frame_lands_on_middle(self):
        plan = frames.plan_frame_times(_segments((0.0, 602.0)), 1)
        self.assertEqual(plan[0], [301.0])

    def test_short_segment_gets_no_frames(self):
        plan = frames.plan_frame_times(_segments((0.0, 1.5), (2.0, 4.0)), 2)
        self.assertEqual(plan, {0: [], 1: []})

    def test_zero_per_segment_disables(self):
        plan = frames.plan_frame_times(_segments((0.0, 600.0)), 0)
        self.assertEqual(plan, {0: []})

    def test_all_segments_planned_together(self):
        plan = frames.plan_frame_times(_segments((0.0, 300.0), (300.0, 600.0)), 3)
        self.assertEqual(sorted(plan), [0, 1])
        self.assertEqual(len(plan[1]), 3)
        for t in plan[1]:
            self.assertGreater(t, 301.0)
            self.assertLess(t, 599.0)


class FramePathTest(unittest.TestCase):
    def test_relpath_encodes_time_in_filename(self):
        rel = frames.frame_relpath(3, 200.5)
        self.assertEqual(rel, "frames/003/0200.500.jpg")

    def test_path_joins_work_dir(self):
        path = frames.frame_path(Path("work"), 3, 200.5)
        self.assertEqual(path.as_posix(), "work/frames/003/0200.500.jpg")

    def test_time_round_trip(self):
        for seconds in (0.0, 5.25, 61.0, 3599.999):
            rel = frames.frame_relpath(0, seconds)
            self.assertEqual(frames.frame_time(rel), seconds)

    def test_unparsable_name_returns_zero(self):
        self.assertEqual(frames.frame_time("frames/000/cover.jpg"), 0.0)


class ManifestTest(TempDirCase):
    def test_round_trip(self):
        work = self.make_dir("frames-manifest-")
        frames.save_manifest(work, {1: ["frames/001/0100.000.jpg"],
                                    0: ["frames/000/0050.000.jpg"]},
                             width=640)
        manifest = frames.load_manifest(work)
        self.assertEqual(manifest["width"], 640)
        self.assertEqual(manifest["frames"],
                         {0: ["frames/000/0050.000.jpg"],
                          1: ["frames/001/0100.000.jpg"]})

    def test_missing_manifest_is_empty(self):
        manifest = frames.load_manifest(self.make_dir("frames-empty-"))
        self.assertEqual(manifest, {"width": 0, "frames": {}})

    def test_corrupt_manifest_is_empty(self):
        work = self.make_dir("frames-corrupt-")
        (work / frames.MANIFEST_FILE).write_text("{ 坏 json", encoding="utf-8")
        self.assertEqual(frames.load_manifest(work), {"width": 0, "frames": {}})


class ImagePartTest(TempDirCase):
    def test_jpeg_becomes_base64_data_uri(self):
        work = self.make_dir("frames-img-")
        path = work / "a.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
        part = frames.image_part(path)
        self.assertEqual(part["type"], "image_url")
        self.assertTrue(part["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
