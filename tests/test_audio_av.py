"""audio_av 的单元测试：用假 av 模块 + 真 numpy，不依赖真实音频文件。

重点覆盖两处容易写错的地方：重采样器必须 resample(None) 排空尾部，
以及切片要靠样本数收口而不是 seek 后的 pts。
"""

import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

from vidsum import audio_av
from vidsum.media import Codec, MediaError

MP3 = Codec("libmp3lame", "mp3", ("-c:a", "libmp3lame", "-q:a", "5"))


class FakeFrame:
    def __init__(self, samples, pts=None):
        self.samples = samples
        self.pts = pts
        self.time_base = Fraction(1, 16000)

    def to_ndarray(self):
        return np.full(self.samples, 7, dtype=np.int16)


class FakeResampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.flushed = False
        self.fed = 0

    def resample(self, frame):
        if frame is None:
            # 没喂过帧就不该有残留，真实 PyAV 也是如此
            if self.flushed or not self.fed:
                return []
            if self.flushed:
                return []
            self.flushed = True
            return [FakeFrame(10)]  # 模拟重采样器内部残留的尾部
        self.fed += 1
        return [FakeFrame(frame.samples)]


class FakeOutStream:
    def __init__(self, name, rate):
        self.name = name
        self.rate = rate
        self.layout = None
        self.bit_rate = None
        self.flush_calls = 0

    def encode(self, frame):
        if frame is None:
            # encode(None) 每个输出流只能调一次，重复调用返回空
            self.flush_calls += 1
            return [("flush", FakeFrame(0))] if self.flush_calls == 1 else []
        return [("packet", frame)]


class FakeOutput:
    def __init__(self, path, *, write_file=True):
        self.path = Path(path)
        self.packets = []
        self.write_file = write_file
        self.streams = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.write_file:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(b"FAKE" * 8)
        return False

    def add_stream(self, name, rate=None):
        stream = FakeOutStream(name, rate)
        self.streams.append(stream)
        return stream

    def mux(self, packet):
        self.packets.append(packet)


class FakeInput:
    def __init__(self, path, frames, duration=None, options=None):
        self.path = Path(path)
        self.frames = list(frames)
        self.duration = duration
        self.options = options
        self.closed = False
        self.seeks = []
        self.outputs = []
        self._audio = [object()]

    @property
    def streams(self):
        outer = self

        class _Streams:
            audio = outer._audio

        return _Streams()

    def decode(self, stream):
        return iter(self.frames)

    def seek(self, offset, *, backward=True):
        # 故意不收 stream：实现一旦传 stream=，offset 的单位就变成了该流的
        # time_base，seek 会跑到完全错误的位置，这里直接让它暴露。
        self.seeks.append((offset, backward))

    def close(self):
        self.closed = True


class FakeCodec:
    # PyAV 13 起 av.codec.codecs_available 是可用编解码器名称的集合
    codecs_available = {"libmp3lame", "aac", "pcm_s16le"}


def make_av(inputs):
    """按路径返回 FakeInput 的假 av 模块；outputs 收集写出过的容器。"""
    outputs = inputs.setdefault("__outputs__", [])

    class _Av:
        # PyAV 17 的 av.time_base 是 int（每秒 tick 数），早期是 Fraction(1, N)
        time_base = 1000000
        AudioResampler = FakeResampler
        codec = FakeCodec

        @staticmethod
        def open(path, mode="r", options=None):
            if "w" in mode:
                out = FakeOutput(path)
                outputs.append(out)
                return out
            container = inputs[str(path)]
            if options:
                container.options = options
            return container

    return _Av


class AvailableTest(unittest.TestCase):
    def test_false_when_av_missing(self):
        with mock.patch.object(audio_av, "_av", None):
            self.assertFalse(audio_av.available())

    def test_true_when_both_present(self):
        with mock.patch.object(audio_av, "_av", object()), \
                mock.patch.object(audio_av, "_np", object()):
            self.assertTrue(audio_av.available())


class DecodePcmTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(".test-tmp") / "audio-av-decode"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))
        self.src = self.tmp / "in.m4a"
        self.src.write_bytes(b"x")

    def _decode(self, frames):
        inputs = {str(self.src): FakeInput(self.src, frames)}
        with mock.patch.object(audio_av, "_av", make_av(inputs)):
            return audio_av.decode_pcm(self.src)

    def test_flushes_resampler_tail(self):
        pcm = self._decode([FakeFrame(100), FakeFrame(50)])
        # 两帧 150 样本 + 重采样器残留 10 个
        self.assertEqual(len(pcm), 160)
        self.assertEqual(pcm.dtype, np.int16)

    def test_empty_input_gives_empty_array(self):
        pcm = self._decode([])
        self.assertEqual(len(pcm), 0)

    def test_resampler_is_16k_mono_s16(self):
        inputs = {str(self.src): FakeInput(self.src, [FakeFrame(4)])}
        seen = {}

        class Recorder(FakeResampler):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                seen.update(kwargs)

        class Av(make_av(inputs)):
            AudioResampler = Recorder

        with mock.patch.object(audio_av, "_av", Av):
            audio_av.decode_pcm(self.src)
        self.assertEqual(seen, {"format": "s16", "layout": "mono", "rate": 16000})

    def test_input_is_closed(self):
        container = FakeInput(self.src, [FakeFrame(4)])
        with mock.patch.object(audio_av, "_av", make_av({str(self.src): container})):
            audio_av.decode_pcm(self.src)
        self.assertTrue(container.closed)


class ProbeDurationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(".test-tmp") / "audio-av-probe"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))
        self.src = self.tmp / "in.m4a"
        self.src.write_bytes(b"x")

    def test_uses_container_duration(self):
        # 2000 秒 = 2e9 微秒
        container = FakeInput(self.src, [], duration=2_000_000_000)
        with mock.patch.object(audio_av, "_av", make_av({str(self.src): container})):
            self.assertAlmostEqual(audio_av.probe_duration(self.src), 2000.0)

    def test_falls_back_to_frame_scan_when_duration_missing(self):
        frames = [FakeFrame(16000, pts=0), FakeFrame(16000, pts=16000)]
        container = FakeInput(self.src, frames, duration=None)
        with mock.patch.object(audio_av, "_av", make_av({str(self.src): container})):
            duration = audio_av.probe_duration(self.src)
        self.assertAlmostEqual(duration, 2.0)

    def test_headers_are_forwarded_to_open(self):
        container = FakeInput(self.src, [], duration=1_000_000)
        with mock.patch.object(audio_av, "_av", make_av({str(self.src): container})):
            audio_av.probe_duration(self.src, headers="Referer: x\r\n")
        self.assertEqual(container.options, {"headers": "Referer: x\r\n"})


class CutSegmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(".test-tmp") / "audio-av-cut"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))
        self.src = self.tmp / "in.m4a"
        self.src.write_bytes(b"x")
        self.dst = self.tmp / "chunks" / "000.mp3"

    def _cut(self, frames, start, end):
        inputs = {str(self.src): FakeInput(self.src, frames)}
        with mock.patch.object(audio_av, "_av", make_av(inputs)):
            audio_av.cut_segment(self.src, self.dst, start, end, codec=MP3)
        return inputs["__outputs__"][-1]

    def test_seeks_before_input(self):
        frames = [FakeFrame(16000, pts=0)] * 5
        inputs = {str(self.src): FakeInput(self.src, frames)}
        with mock.patch.object(audio_av, "_av", make_av(inputs)):
            audio_av.cut_segment(self.src, self.dst, 480.0, 960.0, codec=MP3)
        offset, backward = inputs[str(self.src)].seeks[-1]
        self.assertEqual(offset, int(480.0 * 1000000))
        self.assertTrue(backward)

    def test_seek_handles_legacy_fraction_time_base(self):
        # 早期 PyAV 的 av.time_base 是 Fraction(1, N)，换算必须同样正确
        frames = [FakeFrame(16000, pts=0)] * 3
        inputs = {str(self.src): FakeInput(self.src, frames)}

        class LegacyAv(make_av(inputs)):
            time_base = Fraction(1, 1000000)

        with mock.patch.object(audio_av, "_av", LegacyAv):
            audio_av.cut_segment(self.src, self.dst, 480.0, 960.0, codec=MP3)
        offset, _backward = inputs[str(self.src)].seeks[-1]
        self.assertEqual(offset, int(480.0 * 1000000))

    def test_stops_on_sample_count_not_pts(self):
        # seek 之后 pts 相对起点计时，靠 pts 截断会把整段解到底
        frames = [FakeFrame(16000, pts=16000 * i) for i in range(20)]
        out = self._cut(frames, 0.0, 10.0)
        total = sum(f.samples for _k, f in out.packets)
        # 20 帧全解是 320000 样本；按样本数收口应停在 10 秒附近
        self.assertGreaterEqual(total, 160000)
        self.assertLess(total, 320000)

    def test_flushes_resampler_tail(self):
        frames = [FakeFrame(16000)]
        out = self._cut(frames, 0.0, 10.0)
        # 一帧 16000 已够 10 秒，但重采样器残留的 10 个仍应写出
        self.assertEqual(sum(f.samples for _k, f in out.packets), 16010)

    def test_flushes_output_stream_exactly_once(self):
        # encode(None) 调多次在真实 PyAV 里属于误用，这里守住只调一次
        out = self._cut([FakeFrame(16000)], 0.0, 10.0)
        self.assertEqual(out.streams[0].flush_calls, 1)

    def test_extract_flushes_output_stream_exactly_once(self):
        inputs = {str(self.src): FakeInput(self.src, [FakeFrame(100)])}
        with mock.patch.object(audio_av, "_av", make_av(inputs)):
            audio_av.extract_audio(self.src, self.dst, codec=MP3)
        self.assertEqual(inputs["__outputs__"][-1].streams[0].flush_calls, 1)

    def test_empty_output_raises(self):
        inputs = {str(self.src): FakeInput(self.src, [])}

        class NoFileAv(make_av(inputs)):
            @staticmethod
            def open(path, mode="r", options=None):
                if "w" in mode:
                    return FakeOutput(path, write_file=False)
                return inputs[str(path)]

        with mock.patch.object(audio_av, "_av", NoFileAv):
            with self.assertRaises(MediaError):
                audio_av.cut_segment(self.src, self.dst, 0.0, 1.0, codec=MP3)


class DetectSilencesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(".test-tmp") / "audio-av-silence"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))
        self.src = self.tmp / "in.wav"
        self.src.write_bytes(b"x")

    def _detect(self, pcm, **kwargs):
        with mock.patch.object(audio_av, "decode_pcm", return_value=pcm):
            return audio_av.detect_silences(self.src, **kwargs)

    def test_quiet_and_loud_are_separated(self):
        # 20ms 一帧：前 1s 静音，后 1s 有声
        pcm = np.full(16000 * 2, 8000, dtype=np.int16)
        pcm[:16000] = 0
        spans = self._detect(pcm, noise_db=30, min_silence=0.5)
        self.assertEqual(spans, [(0.0, 1.0)])

    def test_short_quiet_below_min_silence_is_dropped(self):
        pcm = np.full(16000, 8000, dtype=np.int16)
        pcm[:1600] = 0  # 0.1s 静音，短于 0.5s 阈值
        self.assertEqual(self._detect(pcm), [])

    def test_all_quiet_returns_single_span(self):
        pcm = np.zeros(16000 * 3, dtype=np.int16)
        self.assertEqual(self._detect(pcm), [(0.0, 3.0)])

    def test_empty_audio_returns_empty(self):
        self.assertEqual(self._detect(np.zeros(0, dtype=np.int16)), [])


class PickCodecTest(unittest.TestCase):
    def test_prefers_mp3_then_falls_back(self):
        class Av(make_av({})):
            pass

        with mock.patch.object(audio_av, "_av", make_av({})):
            self.assertEqual(audio_av.pick_codec().name, "libmp3lame")

        class NoMp3(FakeCodec):
            codecs_available = {"pcm_s16le"}

        class Av2(make_av({})):
            codec = NoMp3

        with mock.patch.object(audio_av, "_av", Av2):
            self.assertEqual(audio_av.pick_codec().name, "pcm_s16le")

    def test_explicit_preference_is_honoured(self):
        class OnlyOpus(FakeCodec):
            codecs_available = {"libopus"}

        class Av3(make_av({})):
            codec = OnlyOpus

        with mock.patch.object(audio_av, "_av", Av3):
            self.assertEqual(audio_av.pick_codec("libopus").name, "libopus")

    def test_unknown_preference_raises(self):
        with mock.patch.object(audio_av, "_av", make_av({})):
            with self.assertRaises(MediaError):
                audio_av.pick_codec("nope")

    def test_nothing_available_raises(self):
        class Nothing(FakeCodec):
            codecs_available = set()

        class Av4(make_av({})):
            codec = Nothing

        with mock.patch.object(audio_av, "_av", Av4):
            with self.assertRaises(MediaError):
                audio_av.pick_codec()


class MissingDependencyTest(unittest.TestCase):
    def test_error_mentions_pip_install(self):
        with mock.patch.object(audio_av, "_av", None):
            with self.assertRaises(MediaError) as ctx:
                audio_av.probe_duration("x")
        self.assertIn("pip install av", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
