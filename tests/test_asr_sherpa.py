"""sherpa 后端的单元测试：假 sherpa_onnx + 假音频，不碰真实模型。

fake 的 VoiceActivityDetector 故意在 pop() 时清空 front 的 samples，
复刻真实库里「先读再 pop」这个约束——顺序写反只会拿到空列表。
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from vidsum import asr_sherpa

RATE = 16000


class FakeResult:
    def __init__(self, text):
        self.text = text


class FakeStream:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.waveform = None
        self._text = ""

    def accept_waveform(self, rate, samples):
        self.waveform = (rate, np.asarray(list(samples), dtype=np.float32))

    @property
    def result(self):
        return FakeResult(self._text)


class FakeRecognizer:
    last = None
    last_kwargs = None
    texts = []

    @classmethod
    def from_paraformer(cls, **kwargs):
        cls.last_kwargs = kwargs
        instance = cls()
        cls.last = instance
        return instance

    def __init__(self):
        self.streams = []
        self.index = 0

    def create_stream(self):
        stream = FakeStream(self)
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        if self.index < len(FakeRecognizer.texts):
            stream._text = FakeRecognizer.texts[self.index]
        else:
            stream._text = f"第{self.index}段识别结果"
        self.index += 1


class FakeSegment:
    def __init__(self, start, samples):
        self.start = start
        self.samples = samples


class FakeVad:
    """按固定长度切段；pop() 会把 front 的内部对象清空。

    真实 silero 不会把开头的静音当成语音段吐出来，所以这里也跳过喂入数据
    开头连续的零样本（生产代码垫的预热静音就是零），从第一处非零起才切段，
    seg.start 用喂入流里的绝对样本位置——与真实库口径一致。
    """

    last = None
    last_config = None
    seg_len = RATE          # 默认每段 1 秒
    tail_len = 0            # flush 时补上的一段长度

    def __init__(self, config, buffer_size_in_seconds=0):
        self.config = config
        self.buffer_size_in_seconds = buffer_size_in_seconds
        self.queue = []
        self.fed = 0
        self.pos = 0
        self.lead_end = None   # 开头静音结束的绝对位置，None=还没遇到非零
        self.reset_calls = 0
        self.flush_calls = 0
        FakeVad.last = self
        FakeVad.last_config = config

    def accept_waveform(self, samples):
        samples = list(samples)
        n = len(samples)
        self.fed += n                 # fed 始终按整块累计，保持绝对样本坐标
        if self.lead_end is None:
            z = 0
            while z < n and samples[z] == 0:
                z += 1
            if z == n:               # 整块仍是开头静音，先不切段
                return
            self.lead_end = self.fed - n + z   # 第一处非零的绝对位置
            self.pos = self.lead_end
            samples = samples[z:]   # 静音不参与切段
        while self.pos + FakeVad.seg_len <= self.fed:
            self.queue.append(FakeSegment(self.pos, [0.5] * FakeVad.seg_len))
            self.pos += FakeVad.seg_len

    def empty(self):
        return not self.queue

    @property
    def front(self):
        return self.queue[0]

    def pop(self):
        segment = self.queue.pop(0)
        # 真实库里 pop 会释放内部对象，之后读 samples 只会得到空
        segment.samples = []
        return segment

    def flush(self):
        self.flush_calls += 1
        if self.fed > self.pos:
            length = self.fed - self.pos if not FakeVad.tail_len \
                else FakeVad.tail_len
            self.queue.append(FakeSegment(self.pos, [0.5] * length))

    def reset(self):
        self.reset_calls += 1
        self.queue.clear()
        self.fed = 0
        self.pos = 0
        self.lead_end = None


class FakePunctuator:
    last = None

    def __init__(self, config):
        self.config = config
        self.calls = []
        FakePunctuator.last = self

    def add_punctuation(self, text):
        self.calls.append(text)
        return text + "。"


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeSherpa:
    version = "1.13.8"
    OfflineRecognizer = FakeRecognizer
    VoiceActivityDetector = FakeVad
    OfflinePunctuation = FakePunctuator
    VadModelConfig = _Config
    SileroVadModelConfig = _Config
    OfflinePunctuationConfig = _Config
    OfflinePunctuationModelConfig = _Config


def pcm(seconds, *, value=8000):
    # int16 存小数会截断成 0，假音频必须非零，否则会被当成开头静音整段跳过
    return np.full(int(seconds * RATE), value, dtype=np.int16)


class SherpaTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(".test-tmp") / "asr-sherpa"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))

        self.models = self.tmp / "models"
        (self.models / "asr").mkdir(parents=True)
        (self.models / "vad").mkdir(parents=True)
        (self.models / "punc").mkdir(parents=True)
        (self.models / "asr" / "model.int8.onnx").write_bytes(b"m")
        (self.models / "asr" / "tokens.txt").write_text("a 1\n")
        (self.models / "vad" / "silero_vad.onnx").write_bytes(b"v")
        (self.models / "punc" / "model.onnx").write_bytes(b"p")

        FakeRecognizer.last = None
        FakeRecognizer.last_kwargs = None
        FakeRecognizer.texts = []
        FakeVad.last = None
        FakeVad.last_config = None
        FakeVad.seg_len = RATE
        FakeVad.tail_len = 0
        FakePunctuator.last = None

    def make_backend(self, audio=None, **kwargs):
        patches = [
            mock.patch.dict(sys.modules, {"sherpa_onnx": FakeSherpa}),
            mock.patch.object(asr_sherpa, "decode_pcm",
                              return_value=pcm(10) if audio is None else audio),
            mock.patch.object(asr_sherpa, "log"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        kwargs.setdefault("model_dir", str(self.models))
        return asr_sherpa.SherpaBackend(**kwargs)


class ModelDiscoveryTest(SherpaTestCase):
    def test_explicit_dir_wins(self):
        self.assertEqual(asr_sherpa.find_model_dir(str(self.models)),
                         self.models)

    def test_env_var_is_used(self):
        with mock.patch.dict("os.environ",
                             {"VIDSUM_SHERPA_MODEL_DIR": str(self.models)}):
            self.assertEqual(asr_sherpa.find_model_dir(None), self.models)

    def test_explicit_dir_beats_env_var(self):
        other = self.tmp / "other"
        (other / "asr").mkdir(parents=True)
        (other / "asr" / "model.int8.onnx").write_bytes(b"m")
        (other / "asr" / "tokens.txt").write_text("a 1\n")
        with mock.patch.dict("os.environ",
                             {"VIDSUM_SHERPA_MODEL_DIR": str(self.models)}):
            self.assertEqual(asr_sherpa.find_model_dir(str(other)), other)

    def test_missing_models_lists_searched_paths(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        with mock.patch.dict("os.environ", {}), \
                mock.patch.object(asr_sherpa, "_CANDIDATE_DIRS", (empty,)):
            with self.assertRaises(asr_sherpa.SherpaError) as ctx:
                asr_sherpa.find_model_dir(None)
        message = str(ctx.exception)
        self.assertIn("asr/model.int8.onnx", message)
        self.assertIn(str(empty), message)

    def test_dir_without_asr_model_is_rejected(self):
        partial = self.tmp / "partial"
        (partial / "asr").mkdir(parents=True)
        (partial / "asr" / "tokens.txt").write_text("a 1\n")
        with mock.patch.dict("os.environ", {}), \
                mock.patch.object(asr_sherpa, "_CANDIDATE_DIRS", (partial,)):
            with self.assertRaises(asr_sherpa.SherpaError):
                asr_sherpa.find_model_dir(None)


class ConstructionTest(SherpaTestCase):
    def test_missing_sherpa_package_raises(self):
        with mock.patch.dict(sys.modules, {"sherpa_onnx": None}):
            with self.assertRaises(asr_sherpa.SherpaError) as ctx:
                asr_sherpa.SherpaBackend(model_dir=str(self.models))
        self.assertIn("pip install sherpa-onnx", str(ctx.exception))

    def test_missing_vad_model_raises(self):
        (self.models / "vad" / "silero_vad.onnx").unlink()
        with mock.patch.dict(sys.modules, {"sherpa_onnx": FakeSherpa}):
            with self.assertRaises(asr_sherpa.SherpaError) as ctx:
                asr_sherpa.SherpaBackend(model_dir=str(self.models))
        self.assertIn("VAD", str(ctx.exception))

    def test_recognizer_gets_model_paths_and_threads(self):
        self.make_backend(num_threads=3)
        self.assertEqual(FakeRecognizer.last_kwargs["paraformer"],
                         str(self.models / "asr" / "model.int8.onnx"))
        self.assertEqual(FakeRecognizer.last_kwargs["tokens"],
                         str(self.models / "asr" / "tokens.txt"))
        self.assertEqual(FakeRecognizer.last_kwargs["num_threads"], 3)

    def test_vad_config_carries_thresholds(self):
        self.make_backend(vad_threshold=0.6, min_silence_duration=0.4,
                          min_speech_duration=0.2, max_speech_duration=15.0)
        silero = FakeVad.last_config.silero_vad
        self.assertEqual(silero.model,
                         str(self.models / "vad" / "silero_vad.onnx"))
        self.assertEqual(silero.threshold, 0.6)
        self.assertEqual(silero.min_silence_duration, 0.4)
        self.assertEqual(silero.min_speech_duration, 0.2)
        self.assertEqual(silero.max_speech_duration, 15.0)
        self.assertEqual(FakeVad.last_config.sample_rate, RATE)

    def test_punctuation_model_path_is_passed(self):
        self.make_backend(num_threads=4)
        self.assertEqual(FakePunctuator.last.config.model.ct_transformer,
                         str(self.models / "punc" / "model.onnx"))
        self.assertEqual(FakePunctuator.last.config.model.num_threads, 4)

    def test_punctuation_skipped_when_model_absent(self):
        (self.models / "punc" / "model.onnx").unlink()
        backend = self.make_backend()
        self.assertIsNone(backend._punctuator)

    def test_punctuation_can_be_disabled(self):
        backend = self.make_backend(punctuation=False)
        self.assertIsNone(backend._punctuator)


class TranscribeTest(SherpaTestCase):
    def setUp(self):
        super().setUp()
        self.backend = self.make_backend()

    def test_one_utterance_per_segment_with_offsets(self):
        FakeRecognizer.texts = [f"第{i}句" for i in range(10)]
        result = self.backend.transcribe(Path("x.mp3"))
        self.assertEqual([u.text for u in result],
                         [f"第{i}句。" for i in range(10)])
        self.assertEqual([(round(u.start, 3), round(u.end, 3))
                          for u in result],
                         [(float(i), float(i + 1)) for i in range(10)])

    def test_timestamps_are_relative_to_the_chunk(self):
        FakeVad.seg_len = RATE * 2
        FakeRecognizer.texts = ["整段"] * 5
        result = self.backend.transcribe(Path("x.mp3"))
        self.assertEqual([u.text for u in result], ["整段。"] * 5)
        # 10 秒音频按 2 秒切段，5 段；时间戳从 chunk 起点 0 起算
        self.assertEqual([(u.start, u.end) for u in result],
                         [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0),
                          (6.0, 8.0), (8.0, 10.0)])

    def test_warmup_pad_is_prepended_and_offset_out(self):
        FakeRecognizer.texts = [f"第{i}句" for i in range(10)]
        result = self.backend.transcribe(Path("x.mp3"))
        # VAD 实际收到的样本比音频多出一段预热静音
        self.assertGreater(FakeVad.last.fed, RATE * 10)
        # 但预热被从时间戳里减掉：首句仍从 0 起算，段数也不变
        self.assertEqual(len(result), 10)
        self.assertAlmostEqual(result[0].start, 0.0, places=3)

    def test_warmup_can_be_disabled(self):
        backend = self.make_backend(warmup_seconds=0.0)
        FakeRecognizer.texts = [f"第{i}句" for i in range(10)]
        result = backend.transcribe(Path("x.mp3"))
        self.assertEqual(FakeVad.last.fed, RATE * 10)  # 不垫静音，正好 10 秒
        self.assertEqual(len(result), 10)

    def test_reads_samples_before_pop(self):
        # fake 的 pop() 会清空 front，顺序写反就只能拿到空段
        FakeRecognizer.texts = [f"第{i}句" for i in range(10)]
        result = self.backend.transcribe(Path("x.mp3"))
        self.assertEqual(len(result), 10)
        for stream in FakeRecognizer.last.streams:
            self.assertGreaterEqual(len(stream.waveform[1]), RATE)

    def test_tail_samples_are_not_dropped(self):
        # 音频长度不是喂入块的整数倍时，尾部也要喂进 VAD
        audio = np.full(RATE * 10 + RATE // 2, 8000, dtype=np.int16)
        with mock.patch.object(asr_sherpa, "decode_pcm", return_value=audio):
            FakeRecognizer.texts = [f"第{i}句" for i in range(11)]
            result = self.backend.transcribe(Path("x.mp3"))
        self.assertEqual(len(result), 11)
        self.assertAlmostEqual(result[-1].end, 10.5, places=5)

    def test_short_segments_are_skipped(self):
        FakeVad.seg_len = 800          # 0.05s，低于默认的 0.1s 下限
        FakeRecognizer.texts = ["不该出现"]
        self.assertEqual(self.backend.transcribe(Path("x.mp3")), [])

    def test_min_segment_samples_is_configurable(self):
        FakeVad.seg_len = 800
        backend = self.make_backend(min_segment_samples=400)
        FakeRecognizer.texts = ["短段也留"] * 200
        result = backend.transcribe(Path("x.mp3"))
        # 10 秒 / 0.05 秒 = 200 段
        self.assertEqual([u.text for u in result], ["短段也留。"] * 200)

    def test_empty_text_is_skipped(self):
        FakeRecognizer.texts = [""] * 10
        FakeRecognizer.texts[3] = "第二句"
        result = self.backend.transcribe(Path("x.mp3"))
        self.assertEqual([u.text for u in result], ["第二句。"])

    def test_punctuation_is_applied(self):
        FakeRecognizer.texts = ["你好"] * 10
        result = self.backend.transcribe(Path("x.mp3"))
        self.assertEqual([u.text for u in result], ["你好。"] * 10)
        self.assertEqual(FakePunctuator.last.calls, ["你好"] * 10)

    def test_tail_segment_from_flush_is_used(self):
        # 10.5 秒音频：前 10 秒切成整段，剩下 0.5 秒由 flush 补上
        audio = np.full(RATE * 10 + RATE // 2, 8000, dtype=np.int16)
        with mock.patch.object(asr_sherpa, "decode_pcm", return_value=audio):
            FakeRecognizer.texts = [f"第{i}句" for i in range(11)]
            result = self.backend.transcribe(Path("x.mp3"))
        self.assertEqual(len(result), 11)
        self.assertAlmostEqual(result[-1].end, 10.5, places=5)
        self.assertEqual(result[-1].text, "第10句。")

    def test_empty_audio_returns_empty(self):
        with mock.patch.object(asr_sherpa, "decode_pcm",
                               return_value=np.zeros(0, dtype=np.int16)):
            self.assertEqual(self.backend.transcribe(Path("x.mp3")), [])

    def test_vad_is_reset_between_calls(self):
        FakeRecognizer.texts = [f"第{i}句" for i in range(10)]
        self.backend.transcribe(Path("x.mp3"))
        self.assertEqual(FakeVad.last.reset_calls, 1)
        self.backend.transcribe(Path("x.mp3"))
        self.assertEqual(FakeVad.last.reset_calls, 2)

    def test_decode_uses_16k_float32(self):
        FakeRecognizer.texts = ["第一句"] * 10
        self.backend.transcribe(Path("x.mp3"))
        rate, samples = FakeRecognizer.last.streams[0].waveform
        self.assertEqual(rate, RATE)
        self.assertEqual(samples.dtype, np.float32)
        self.assertTrue(np.all((samples >= -1.0) & (samples <= 1.0)))


class ThreadSafetyTest(SherpaTestCase):
    def test_concurrent_calls_do_not_interleave(self):
        backend = self.make_backend()
        FakeRecognizer.texts = [f"第{i}句" for i in range(40)]
        errors = []

        def work():
            try:
                result = backend.transcribe(Path("x.mp3"))
                if len(result) != 10:
                    errors.append(f"期望 10 句，实得 {len(result)}")
            except Exception as exc:  # pragma: no cover - 仅失败时记录
                errors.append(repr(exc))

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])


class BackendRegistrationTest(SherpaTestCase):
    def patch_sherpa(self):
        patcher = mock.patch.dict(sys.modules, {"sherpa_onnx": FakeSherpa})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_sherpa_is_registered(self):
        from vidsum.asr import get_backend
        self.patch_sherpa()
        backend = get_backend("sherpa", model_dir=str(self.models))
        self.assertIsInstance(backend, asr_sherpa.SherpaBackend)
        self.assertEqual(backend.name, "sherpa")

    def test_unknown_dir_is_reported(self):
        from vidsum.asr import get_backend
        with mock.patch.dict(sys.modules, {"sherpa_onnx": FakeSherpa}), \
                mock.patch.dict("os.environ", {}), \
                mock.patch.object(asr_sherpa, "_CANDIDATE_DIRS", ()):
            with self.assertRaises(asr_sherpa.SherpaError):
                get_backend("sherpa",
                            model_dir=str(self.tmp / "does-not-exist"))

    def test_extra_kwargs_are_filtered_per_backend(self):
        # CLI 会一次性传入所有后端的参数，这里按签名挑出认识的
        from vidsum.asr import get_backend
        self.patch_sherpa()
        backend = get_backend("sherpa", model_dir=str(self.models),
                              model_size="tiny", language="zh",
                              base_url="https://x", api_key="k", model="m")
        self.assertIsInstance(backend, asr_sherpa.SherpaBackend)

    def test_unknown_name_still_raises(self):
        from vidsum.asr import get_backend
        with self.assertRaises(ValueError):
            get_backend("nope")


if __name__ == "__main__":
    unittest.main()
