import json
import unittest
from pathlib import Path

from vidsum.asr import (ChunkResult, Utterance, get_backend, looks_duplicated,
                        merge_chunks, normalize, stitch)
from vidsum.asr_api import encode_multipart, parse_transcription


class StitchTest(unittest.TestCase):
    def _chunk(self, index, start, end, texts):
        return ChunkResult(index, start, end,
                           [Utterance(float(i), float(i + 1), t)
                            for i, t in enumerate(texts)])

    def test_timestamps_are_offset_by_chunk_start(self):
        chunks = [self._chunk(0, 0.0, 10.0, ["a"]),
                  self._chunk(1, 600.0, 610.0, ["b"])]
        merged = stitch(chunks)
        self.assertEqual([(u.start, u.text) for u in merged],
                         [(0.0, "a"), (600.0, "b")])

    def test_out_of_order_chunks_are_sorted(self):
        chunks = [self._chunk(1, 600.0, 610.0, ["b"]),
                  self._chunk(0, 0.0, 10.0, ["a"])]
        self.assertEqual([u.text for u in stitch(chunks)], ["a", "b"])

    def test_boundary_duplicate_is_dropped(self):
        chunks = [self._chunk(0, 0.0, 10.0, ["前面的话", "这是一个重复的短语"]),
                  self._chunk(1, 600.0, 610.0, ["这是一个重复的短语", "后面的话"])]
        merged = stitch(chunks)
        self.assertEqual([u.text for u in merged],
                         ["前面的话", "这是一个重复的短语", "后面的话"])

    def test_dedup_can_be_disabled(self):
        chunks = [self._chunk(0, 0.0, 10.0, ["同一个短语"]),
                  self._chunk(1, 600.0, 610.0, ["同一个短语", "新的内容"])]
        merged = stitch(chunks, dedup=False)
        self.assertEqual(len(merged), 3)

    def test_empty_utterances_are_skipped(self):
        chunk = ChunkResult(0, 0.0, 10.0, [Utterance(0, 1, "有内容"),
                                           Utterance(1, 2, "   ")])
        self.assertEqual(len(stitch([chunk])), 1)


class DedupHeuristicTest(unittest.TestCase):
    def test_normalize_strips_punctuation_and_case(self):
        self.assertEqual(normalize("Hello, World!"), "helloworld")

    def test_identical_text_is_duplicate(self):
        self.assertTrue(looks_duplicated("这是一个测试。", "这是一个测试"))

    def test_short_identical_text_is_duplicate(self):
        # 完全一致属于高置信度重复，即使很短也应去掉
        self.assertTrue(looks_duplicated("好", "好"))

    def test_short_different_text_is_not_duplicate(self):
        self.assertFalse(looks_duplicated("好", "行"))

    def test_unrelated_text_is_not_duplicate(self):
        self.assertFalse(looks_duplicated("今天讲梯度下降", "下周考试范围已公布"))


class MergeChunksTest(unittest.TestCase):
    def _chunk(self, index, start, texts):
        return ChunkResult(index, start, start + 100.0,
                           [Utterance(float(i), float(i + 1), t)
                            for i, t in enumerate(texts)])

    def test_returns_per_chunk_kept_utterances(self):
        pairs = merge_chunks([self._chunk(0, 0.0, ["甲", "乙"]),
                              self._chunk(1, 600.0, ["丙"])])
        self.assertEqual([len(utts) for _c, utts in pairs], [2, 1])
        self.assertEqual([u.text for _c, utts in pairs for u in utts],
                         ["甲", "乙", "丙"])

    def test_duplicate_is_dropped_from_the_later_chunk(self):
        pairs = merge_chunks([self._chunk(0, 0.0, ["重复的短语"]),
                              self._chunk(1, 600.0, ["重复的短语", "新内容"])])
        self.assertEqual([u.text for u in pairs[1][1]], ["新内容"])

    def test_offsets_are_absolute(self):
        pairs = merge_chunks([self._chunk(1, 600.0, ["丙"])])
        self.assertEqual(pairs[0][1][0].start, 600.0)

    def test_matches_stitch_output(self):
        chunks = [self._chunk(0, 0.0, ["甲", "重复的短语"]),
                  self._chunk(1, 600.0, ["重复的短语", "乙"])]
        flat = stitch(chunks)
        pairs = merge_chunks(chunks)
        self.assertEqual([u.text for u in flat],
                         [u.text for _c, utts in pairs for u in utts])

    def test_dedup_can_be_disabled(self):
        chunks = [self._chunk(0, 0.0, ["重复的短语"]),
                  self._chunk(1, 600.0, ["重复的短语", "新内容"])]
        self.assertEqual([len(u) for _c, u in merge_chunks(chunks, dedup=False)],
                         [1, 2])


class ChunkResultTest(unittest.TestCase):
    def test_round_trip(self):
        chunk = ChunkResult(2, 10.5, 20.5, [Utterance(1.0, 2.0, "文本")])
        restored = ChunkResult.from_dict(json.loads(json.dumps(chunk.to_dict())))
        self.assertEqual(restored.index, 2)
        self.assertEqual(restored.start, 10.5)
        self.assertEqual(restored.text, "文本")


class MultipartTest(unittest.TestCase):
    def test_body_contains_fields_and_file(self):
        body, ctype = encode_multipart(
            {"model": "whisper-1", "language": "zh"},
            {"file": ("seg.mp3", b"\x00\x01\x02", "audio/mpeg")},
        )
        self.assertIn('name="model"', body.decode())
        self.assertIn("whisper-1", body.decode())
        self.assertIn('filename="seg.mp3"', body.decode())
        self.assertIn(b"\x00\x01\x02", body)
        self.assertIn("multipart/form-data; boundary=", ctype)


class ParseTranscriptionTest(unittest.TestCase):
    def test_verbose_json(self):
        payload = {"text": "完整文本", "duration": 12.5, "segments": [
            {"start": 0.0, "end": 1.5, "text": " 第一句 "},
            {"start": 1.5, "end": 3.0, "text": ""},
            {"start": 3.0, "end": 4.0, "text": "第二句"},
        ]}
        utts = parse_transcription(payload)
        self.assertEqual([(u.start, u.end, u.text) for u in utts],
                         [(0.0, 1.5, "第一句"), (3.0, 4.0, "第二句")])

    def test_text_only_json(self):
        utts = parse_transcription({"text": "只有文本", "duration": 5.0})
        self.assertEqual(len(utts), 1)
        self.assertEqual(utts[0].text, "只有文本")

    def test_plain_text_fallback(self):
        utts = parse_transcription(None, fallback_text="  纯文本返回  ")
        self.assertEqual(utts[0].text, "纯文本返回")

    def test_nothing_parseable(self):
        self.assertEqual(parse_transcription(None, fallback_text=""), [])
        self.assertEqual(parse_transcription({}), [])


class BackendRegistryTest(unittest.TestCase):
    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            get_backend("nope")

    def test_dummy_backend_ignores_extra_kwargs(self):
        backend = get_backend("dummy", model_size="tiny", language="zh")
        utts = backend.transcribe(Path("whatever/007.mp3"))
        self.assertTrue(all(u.text for u in utts))


if __name__ == "__main__":
    unittest.main()
