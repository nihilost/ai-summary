import unittest

from vidsum.asr import ChunkResult, Utterance
from vidsum.render import (render_final, render_progress, render_transcript,
                           render_transcript_partial)
from vidsum.summarize import ChunkNote, SummaryState


def _state():
    state = SummaryState()
    state.running = "- 要点一\n- 要点二"
    state.next_index = 2
    state.notes = [
        ChunkNote(0, 0.0, 600.0, "开场与回顾"),
        ChunkNote(1, 600.0, 1200.0, "核心定义"),
    ]
    return state


class ProgressTest(unittest.TestCase):
    def test_contains_progress_outline_and_summary(self):
        text = render_progress("线性代数", _state(), total=4,
                               source="lecture.mp4")
        self.assertIn("# 线性代数", text)
        self.assertIn("2/4 段已完成", text)
        self.assertIn("已覆盖 20:00", text)
        self.assertIn("## 分段大纲", text)
        self.assertIn("- 00:00\u201310:00 开场与回顾", text)
        self.assertIn("- 10:00\u201320:00 核心定义", text)
        self.assertIn("## 当前滚动摘要", text)
        self.assertIn("- 要点一", text)
        self.assertIn("lecture.mp4", text)

    def test_empty_state_renders_placeholders(self):
        text = render_progress("空", SummaryState(), total=3)
        self.assertIn("0/3 段已完成", text)
        self.assertIn("（尚无已完成的分段）", text)
        self.assertIn("（尚未生成，请稍候）", text)

    def test_hours_are_shown_for_long_videos(self):
        state = SummaryState()
        state.next_index = 1
        state.notes = [ChunkNote(0, 3600.0, 7200.0, "第二小时")]
        text = render_progress("长视频", state, total=2)
        self.assertIn("1:00:00\u20132:00:00", text)


class TranscriptTest(unittest.TestCase):
    def test_timestamps_use_hours(self):
        text = render_transcript("课程", [Utterance(0.0, 2.0, "第一句"),
                                       Utterance(3661.0, 3663.0, "第二句")])
        self.assertIn("**[0:00:00]** 第一句", text)
        self.assertIn("**[1:01:01]** 第二句", text)
        self.assertIn("# 课程 · 完整转写", text)

    def test_empty_transcript(self):
        self.assertIn("（没有转写结果）", render_transcript("课程", []))


class PartialTranscriptTest(unittest.TestCase):
    def _chunk(self, index, start, texts):
        return ChunkResult(index, start, start + 100.0,
                           [Utterance(float(i), float(i + 1), t)
                            for i, t in enumerate(texts)])

    def test_shows_progress_and_offsets(self):
        text = render_transcript_partial(
            "课程", [self._chunk(0, 0.0, ["第一句"]),
                     self._chunk(1, 600.0, ["第二句"])], total=4)
        self.assertIn("> 转写进行中：2/4 段已完成", text)
        self.assertIn("**[0:00:00]** 第一句", text)
        self.assertIn("**[0:10:00]** 第二句", text)

    def test_sorted_by_index_not_completion_order(self):
        text = render_transcript_partial(
            "课程", [self._chunk(1, 600.0, ["第二句"]),
                     self._chunk(0, 0.0, ["第一句"])], total=2)
        self.assertLess(text.index("第一句"), text.index("第二句"))

    def test_empty(self):
        text = render_transcript_partial("课程", [], total=3)
        self.assertIn("（等待转写结果…）", text)


class FinalTest(unittest.TestCase):
    def test_contains_final_md_and_appendix(self):
        text = render_final("课程", _state(), "## 核心结论\n- A\n- B")
        self.assertIn("# 课程", text)
        self.assertIn("## 核心结论", text)
        self.assertIn("## 附录：分段索引", text)
        self.assertIn("- 00:00\u201310:00 开场与回顾", text)

    def test_missing_final_md(self):
        self.assertIn("（未生成终稿）", render_final("课程", _state(), ""))

    def test_frame_links_in_appendix_use_filename_time(self):
        frames = {0: ["frames/000/0125.500.jpg", "frames/000/0374.500.jpg"]}
        text = render_final("课程", _state(), "正文", frames=frames)
        self.assertIn("![第 1 段画面 02:06](frames/000/0125.500.jpg)", text)
        self.assertIn("![第 1 段画面 06:14](frames/000/0374.500.jpg)", text)
        # 第 2 段没有帧，不该冒出别的图链
        self.assertNotIn("frames/001/", text)

    def test_no_frames_keeps_appendix_text_only(self):
        text = render_final("课程", _state(), "正文")
        self.assertIn("- 00:00\u201310:00 开场与回顾", text)
        self.assertNotIn("![", text)


if __name__ == "__main__":
    unittest.main()
