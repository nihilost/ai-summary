import unittest
from unittest import mock

from pathlib import Path

from vidsum import llm as llm_mod
from vidsum.summarize import (SummaryState, compress, derive_heading,
                              finalize, load_state, parse_update, save_state,
                              update_summary)

WELL_FORMED = """\
## 更新后的滚动摘要
- 要点一
- 要点二

## 本段小标题
核心定义回顾"""


from tests.support import TempDirCase


class ParseUpdateTest(unittest.TestCase):
    def test_well_formed(self):
        summary, heading = parse_update(WELL_FORMED)
        self.assertEqual(summary, "- 要点一\n- 要点二")
        self.assertEqual(heading, "核心定义回顾")

    def test_markers_without_hashes(self):
        raw = "更新后的滚动摘要\n正文内容\n\n本段小标题\n某个标题"
        summary, heading = parse_update(raw)
        self.assertEqual(summary, "正文内容")
        self.assertEqual(heading, "某个标题")

    def test_only_summary_marker(self):
        summary, heading = parse_update("## 滚动摘要\n只有摘要没有标题")
        self.assertEqual(summary, "只有摘要没有标题")
        self.assertEqual(heading, "")

    def test_only_heading_marker(self):
        summary, heading = parse_update("前面的正文\n## 小标题\n标题内容")
        self.assertEqual(summary, "前面的正文")
        self.assertEqual(heading, "标题内容")

    def test_no_markers_falls_back_to_whole_text(self):
        summary, heading = parse_update("一段没有任何标记的回复")
        self.assertEqual(summary, "一段没有任何标记的回复")
        self.assertEqual(heading, "")

    def test_empty_input(self):
        self.assertEqual(parse_update(""), ("", ""))
        self.assertEqual(parse_update(None), ("", ""))

    def test_heading_with_trailing_noise(self):
        raw = "## 更新后的滚动摘要\n正文\n\n## 本段小标题\n### 带井号的标题\n多余行"
        _summary, heading = parse_update(raw)
        self.assertEqual(heading, "带井号的标题")


class DeriveHeadingTest(unittest.TestCase):
    def test_uses_first_sentence(self):
        self.assertEqual(derive_heading("今天讲梯度下降法。后面还有很多内容"),
                         "今天讲梯度下降法")

    def test_english_sentence(self):
        self.assertEqual(derive_heading("We start with backprop. Then more."),
                         "We start with backprop")

    def test_truncates_long_heading(self):
        heading = derive_heading("一" * 60)
        self.assertTrue(heading.endswith("…"))
        self.assertLessEqual(len(heading), 25)

    def test_empty_text(self):
        self.assertEqual(derive_heading(""), "")
        self.assertEqual(derive_heading("   "), "")


class StateTest(TempDirCase):
    def test_round_trip(self):
        state = SummaryState()
        state.running = "- 要点"
        state.next_index = 3
        state.notes.append(mock.Mock(index=0, start=0.0, end=10.0, heading="标题"))
        restored = SummaryState.from_dict(state.to_dict())
        self.assertEqual(restored.running, "- 要点")
        self.assertEqual(restored.next_index, 3)
        self.assertEqual(len(restored.notes), 1)

    def test_load_missing_file_returns_empty_state(self):
        self.assertEqual(load_state(self.make_dir()).next_index, 0)

    def test_save_and_load(self):
        work = self.make_dir()
        state = SummaryState()
        state.running = "内容"
        state.next_index = 1
        save_state(work, state)
        self.assertEqual(load_state(Path(work)).running, "内容")
        self.assertTrue((Path(work) / "summary_state.json").is_file())


class UpdateSummaryTest(unittest.TestCase):
    def setUp(self):
        self.state = SummaryState()
        self.calls = []

        def fake_chat(config, messages, **kwargs):
            self.calls.append(messages)
            return WELL_FORMED.replace("核心定义回顾", f"第{len(self.calls)}段标题")

        self.patcher = mock.patch.object(llm_mod, "chat", side_effect=fake_chat)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_state_advances(self):
        update_summary(llm_mod.LLMConfig(api_key="k"), self.state,
                       "新的转写内容", 0, 3, start=0.0, end=600.0)
        self.assertEqual(self.state.next_index, 1)
        self.assertEqual(self.state.running, "- 要点一\n- 要点二")
        self.assertEqual(len(self.state.notes), 1)
        self.assertEqual(self.state.notes[0].heading, "第1段标题")

    def test_long_chunk_is_truncated(self):
        update_summary(llm_mod.LLMConfig(api_key="k"), self.state,
                       "字" * 9000, 0, 1, start=0.0, end=600.0,
                       max_chunk_chars=6000)
        sent = self.calls[0][1]["content"]
        self.assertIn("已截断", sent)
        self.assertLess(len(sent), 7000)

    def test_empty_model_reply_keeps_previous_summary(self):
        with mock.patch.object(llm_mod, "chat", return_value=""):
            update_summary(llm_mod.LLMConfig(api_key="k"), self.state,
                           "内容", 0, 1, start=0.0, end=600.0)
        self.assertEqual(self.state.running, "")
        self.assertEqual(self.state.next_index, 1)
        # 模型没给标题时退回本地推导
        self.assertEqual(self.state.notes[0].heading, "内容")

    def test_oversized_running_summary_is_compressed_first(self):
        self.state.running = "旧" * 5000
        with mock.patch("vidsum.summarize.compress",
                        return_value="压缩后的摘要") as compressed:
            update_summary(llm_mod.LLMConfig(api_key="k"), self.state,
                           "新内容", 0, 1, start=0.0, end=600.0,
                           max_running_chars=4000)
        compressed.assert_called_once()
        self.assertIn("压缩后的摘要", self.calls[-1][1]["content"])


class VisionUpdateTest(TempDirCase):
    """传入关键帧时，user 消息要变成多模态 content 列表。"""

    def setUp(self):
        self.state = SummaryState()
        self.calls = []

        def fake_chat(config, messages, **kwargs):
            self.calls.append(messages)
            return WELL_FORMED

        self.patcher = mock.patch.object(llm_mod, "chat", side_effect=fake_chat)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

        self.frame = self.make_dir("vision-") / "0125.500.jpg"
        self.frame.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    def test_images_become_multimodal_message(self):
        update_summary(llm_mod.LLMConfig(api_key="k"), self.state,
                       "转写内容", 0, 2, start=0.0, end=600.0,
                       images=[self.frame])
        content = self.calls[0][1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("画面截图", content[0]["text"])  # VISION_HINT 进了提示词
        self.assertEqual(len(content), 2)  # 文本 + 1 帧
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"))

    def test_without_images_stays_plain_text(self):
        update_summary(llm_mod.LLMConfig(api_key="k"), self.state,
                       "转写内容", 0, 1, start=0.0, end=600.0)
        self.assertIsInstance(self.calls[0][1]["content"], str)
        self.assertNotIn("画面截图", self.calls[0][1]["content"])


class FinalizeTest(unittest.TestCase):
    def test_finalize_calls_model_and_stores_result(self):
        state = SummaryState()
        state.running = "- 要点一\n- 要点二"
        state.notes.append(mock.Mock(index=0, start=0.0, end=600.0,
                                     heading="第一段"))
        with mock.patch.object(llm_mod, "chat", return_value="## 核心结论\n- A"):
            result = finalize(llm_mod.LLMConfig(api_key="k"), state, "课程名")
        self.assertEqual(result, "## 核心结论\n- A")
        self.assertEqual(state.final, "## 核心结论\n- A")

    def test_finalize_with_empty_summary_skips_model(self):
        with mock.patch.object(llm_mod, "chat") as chat:
            self.assertEqual(finalize(llm_mod.LLMConfig(api_key="k"),
                                      SummaryState(), "课程名"), "")
        chat.assert_not_called()


class CompressTest(unittest.TestCase):
    def test_compress_falls_back_to_input_on_empty_reply(self):
        with mock.patch.object(llm_mod, "chat", return_value="   "):
            self.assertEqual(compress(llm_mod.LLMConfig(api_key="k"),
                                      "原始摘要", limit=100), "原始摘要")


if __name__ == "__main__":
    unittest.main()
