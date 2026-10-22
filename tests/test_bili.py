"""B 站视频页解析：纯逻辑 + 打桩 urlopen 的编排，全程不联网。"""

import unittest
from unittest import mock

import json

from vidsum import bili
from vidsum.media import MediaError

IMG_KEY = "7cd084941338484aae1ad9425b84077c"
SUB_KEY = "3a4b8c1e2f6059a7b3c4d5e6f708192a3b4c5d6e"


class _Resp:
    def __init__(self, data):
        self._data = data if isinstance(data, bytes) else data.encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


PAGE_HTML = (
    "<html><script>window.__INITIAL_STATE__="
    '{"videoData":{"bvid":"BV1mnG6HECt","aid":123,"cid":456,'
    '"pages":[{"cid":456}]},"x":1};(function(){})()</script></html>'
)

VIEW_JSON = json.dumps({"code": 0, "message": "0", "data": {
    "aid": 123, "bvid": "BV1mnG6HECt", "cid": 456,
    "title": "测试稿件", "pages": [{"cid": 456}]}})

NAV_JSON = ('{"code":0,"data":{"wbi_img":{"img_url":'
            '"https://x/' + IMG_KEY + '.png","sub_url":'
            '"https://x/' + SUB_KEY + '.png"}}}')

PLAY_JSON = ('{"code":0,"data":{"dash":{"audio":'
             '[{"bandwidth":80000,"base_url":"https://up/lo.m4s"},'
             '{"bandwidth":320000,"base_url":"https://up/hi.m4s"}]}}}')


class IsBilibiliPageTest(unittest.TestCase):
    def test_positive(self):
        for u in [
            "https://www.bilibili.com/video/BV17mnG6HECt",
            "https://www.bilibili.com/video/BV17mnG6HECt?spm_id_from=1&vd_source=x",
            "https://m.bilibili.com/video/BV1xx411c7Fx",
            "https://www.bilibili.com/video/av170001",
            "http://www.bilibili.com/video/BV1abcdefg",
        ]:
            self.assertTrue(bili.is_bilibili_page(u), u)

    def test_negative(self):
        for u in [
            "https://example.com/course/lec7.mp4",
            "https://cdn.example/x.m3u8",
            "https://www.bilibili.com/",              # 首页，不是视频页
            "https://www.bilibili.com/BV1xx",          # 少了 /video/
            "https://www.youtube.com/watch?v=abc",
            "D:/videos/lec.mp4",
            "",
        ]:
            self.assertFalse(bili.is_bilibili_page(u), u)


class SignTest(unittest.TestCase):
    def test_mixin_key_is_32_and_deterministic(self):
        k = bili.mixin_key(IMG_KEY, SUB_KEY)
        self.assertEqual(len(k), 32)
        self.assertEqual(k, bili.mixin_key(IMG_KEY, SUB_KEY))
        self.assertLessEqual(set(k), set(IMG_KEY + SUB_KEY))
        self.assertNotEqual(k, bili.mixin_key(SUB_KEY, IMG_KEY))

    def test_mixin_key_rejects_short(self):
        with self.assertRaises(bili.BiliError):
            bili.mixin_key("short", "keys")

    def test_wbi_sign_adds_wts_and_rid_sorted(self):
        signed = bili.wbi_sign(
            {"cid": 456, "bvid": "BV1x", "fnval": 4048},
            IMG_KEY, SUB_KEY, wts=1700000000)
        self.assertEqual(int(signed["wts"]), 1700000000)
        self.assertIn("w_rid", signed)
        self.assertEqual(len(signed["w_rid"]), 32)
        # 特殊字符从值里剔除
        signed2 = bili.wbi_sign({"q": "a!'()*b"}, IMG_KEY, SUB_KEY, wts=1)
        self.assertEqual(signed2["q"], "ab")


class ExtractStateTest(unittest.TestCase):
    def test_parses_nested_json_with_braces_in_string(self):
        html = ('<script>window.__INITIAL_STATE__='
                '{"videoData":{"cid":9,"title":"a{b}c"},"n":{"m":{}}};'
                '(function(){})()</script>')
        state = bili.extract_initial_state(html)
        self.assertEqual(state["videoData"]["cid"], 9)
        self.assertEqual(state["videoData"]["title"], "a{b}c")

    def test_missing_marker_returns_none(self):
        self.assertIsNone(bili.extract_initial_state("<html>nope</html>"))


class PickAudioTest(unittest.TestCase):
    def test_picks_highest_bandwidth(self):
        url = bili._pick_audio_url({"data": {"dash": {"audio": [
            {"bandwidth": 80000, "base_url": "https://x/lo.m4s"},
            {"bandwidth": 320000, "base_url": "https://x/hi.m4s"},
        ]}}})
        self.assertEqual(url, "https://x/hi.m4s")

    def test_durl_fallback(self):
        url = bili._pick_audio_url({"data": {"durl": [
            {"base_url": "https://x/whole.mp4"}]}})
        self.assertEqual(url, "https://x/whole.mp4")

    def test_empty_raises(self):
        with self.assertRaises(bili.BiliError):
            bili._pick_audio_url({"data": {}})


class ResolveOrchestrationTest(unittest.TestCase):
    def _fake_urlopen(self, captured):
        def opener(req, timeout=None):
            url = req.full_url
            captured.append(url)
            if "web-interface/view" in url:
                return _Resp(VIEW_JSON)
            if "web-interface/nav" in url:
                return _Resp(NAV_JSON)
            if "/video/BV" in url:
                return _Resp(PAGE_HTML)
            if "player/wbi/v2" in url:
                return _Resp(PLAY_JSON)
            raise AssertionError("unexpected url " + url)
        return opener

    def test_resolve_end_to_end(self):
        captured = []
        with mock.patch("urllib.request.urlopen",
                        side_effect=self._fake_urlopen(captured)):
            audio_url, headers = bili.resolve_audio_url(
                "https://www.bilibili.com/video/BV1mnG6HECt?vd_source=x")
        self.assertEqual(audio_url, "https://up/hi.m4s")
        self.assertEqual(headers["Referer"], "https://www.bilibili.com/")
        self.assertIn("User-Agent", headers)
        # 主路径走 view 接口，不再下载 watch 页
        self.assertTrue(any("web-interface/view?bvid=BV1mnG6HECt" in u
                            for u in captured))
        self.assertFalse(any("/video/BV" in u for u in captured))
        play = next(u for u in captured if "player/wbi/v2" in u)
        self.assertIn("w_rid=", play)
        self.assertIn("wts=", play)
        self.assertIn("bvid=BV1mnG6HECt", play)
        self.assertIn("cid=456", play)
        self.assertIn("fnval=4048", play)

    def test_resolve_rejects_non_bili(self):
        with self.assertRaises(bili.BiliError):
            bili.resolve_audio_url("https://example.com/a.mp4")

    def test_resolve_reports_api_error_code(self):
        def opener(req, timeout=None):
            url = req.full_url
            if "web-interface/view" in url:
                return _Resp(VIEW_JSON)
            if "web-interface/nav" in url:
                return _Resp('{"code":0,"data":{"wbi_img":{"img_url":'
                             '"https://x/' + IMG_KEY + '.png","sub_url":'
                             '"https://x/' + SUB_KEY + '.png"}}}')
            if "/video/BV" in url:
                return _Resp(PAGE_HTML)
            return _Resp('{"code":-400,"message":"bad"}')
        with mock.patch("urllib.request.urlopen", side_effect=opener):
            with self.assertRaises(bili.BiliError):
                bili.resolve_audio_url("https://www.bilibili.com/video/BV1mnG6HECt")

    def test_view_failure_falls_back_to_page(self):
        def opener(req, timeout=None):
            url = req.full_url
            if "web-interface/view" in url:
                return _Resp('{"code":-352,"message":"风控"}')
            if "web-interface/nav" in url:
                return _Resp(NAV_JSON)
            if "/video/BV" in url:
                return _Resp(PAGE_HTML)
            if "player/wbi/v2" in url:
                return _Resp(PLAY_JSON)
            raise AssertionError("unexpected url " + url)
        with mock.patch("urllib.request.urlopen", side_effect=opener):
            audio_url, _ = bili.resolve_audio_url(
                "https://www.bilibili.com/video/BV1mnG6HECt")
        self.assertEqual(audio_url, "https://up/hi.m4s")

    def test_view_and_page_both_fail_reports_both(self):
        def opener(req, timeout=None):
            url = req.full_url
            if "web-interface/view" in url:
                return _Resp('{"code":-352,"message":"风控"}')
            if "/video/BV" in url:
                return _Resp("<html>风控页</html>")
            raise AssertionError("不该继续请求 " + url)
        with mock.patch("urllib.request.urlopen", side_effect=opener):
            with self.assertRaises(bili.BiliError) as ctx:
                bili.resolve_audio_url(
                    "https://www.bilibili.com/video/BV1mnG6HECt")
        msg = str(ctx.exception)
        self.assertIn("__INITIAL_STATE__", msg)
        self.assertIn("view 接口也失败", msg)

    def test_av_id_uses_aid_query(self):
        captured = []
        def opener(req, timeout=None):
            url = req.full_url
            captured.append(url)
            if "web-interface/view" in url:
                # av 号的 view 返回只有 aid、没有 bvid
                return _Resp(json.dumps({"code": 0, "data": {
                    "aid": 170001, "cid": 456, "pages": [{"cid": 456}]}}))
            if "web-interface/nav" in url:
                return _Resp(NAV_JSON)
            if "player/wbi/v2" in url:
                return _Resp(PLAY_JSON)
            raise AssertionError("unexpected url " + url)
        with mock.patch("urllib.request.urlopen", side_effect=opener):
            audio_url, _ = bili.resolve_audio_url(
                "https://www.bilibili.com/video/av170001")
        self.assertTrue(any("web-interface/view?aid=170001" in u
                            for u in captured))
        play = next(u for u in captured if "player/wbi/v2" in u)
        self.assertIn("aid=170001", play)
        self.assertNotIn("bvid=", play)

    def test_bilierror_is_mediaerror(self):
        self.assertTrue(issubclass(bili.BiliError, MediaError))


if __name__ == "__main__":
    unittest.main()
