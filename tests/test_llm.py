import io
import json
import unittest
from unittest import mock

from vidsum import llm


class FakeResponse:
    def __init__(self, payload, status=200):
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status
        self.headers = {"Content-Type": "application/json"}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def completion(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


CONFIG = llm.LLMConfig(base_url="https://example.invalid/v1",
                       api_key="test-key", model="test-model", retries=3)


class ChatTest(unittest.TestCase):
    def test_success_returns_content(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=FakeResponse(completion("你好"))):
            self.assertEqual(llm.chat(CONFIG, [{"role": "user", "content": "hi"}]),
                             "你好")

    def test_authorization_header_is_sent(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["auth"] = request.get_header("Authorization")
            captured["url"] = request.full_url
            return FakeResponse(completion("ok"))

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat(CONFIG, [{"role": "user", "content": "hi"}])
        self.assertEqual(captured["auth"], "Bearer test-key")
        self.assertEqual(captured["url"],
                         "https://example.invalid/v1/chat/completions")

    def test_retries_on_retryable_status(self):
        import urllib.error

        calls = {"n": 0}

        def flaky(request, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.HTTPError(
                    request.full_url, 503, "unavailable", {},
                    io.BytesIO(b"busy"))
            return FakeResponse(completion("恢复成功"))

        with mock.patch("urllib.request.urlopen", side_effect=flaky), \
                mock.patch("vidsum.llm._sleep"):
            self.assertEqual(llm.chat(CONFIG, []), "恢复成功")
        self.assertEqual(calls["n"], 3)

    def test_auth_error_is_not_retried(self):
        import urllib.error

        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.HTTPError(
                            "https://x/v1/chat/completions", 401, "unauthorized",
                            {}, io.BytesIO(b"bad key"))):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.chat(CONFIG, [])
        self.assertIn("401", str(ctx.exception))

    def test_exhausted_retries_raise(self):
        import urllib.error

        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.HTTPError(
                            "https://x", 500, "boom", {},
                            io.BytesIO(b"err"))), \
                mock.patch("vidsum.llm._sleep"):
            with self.assertRaises(llm.LLMError):
                llm.chat(CONFIG, [])


class EmptyContentTest(unittest.TestCase):
    """空正文与截断的处理：推理模型思考吃满预算时要能自救，其它情况要报清楚。"""

    def _response(self, content="", finish_reason="stop", reasoning=None):
        message = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        choice = {"message": message, "finish_reason": finish_reason}
        return {"choices": [choice]}

    def _budget_spy(self, reply):
        """记录每次请求的 max_tokens，并按预算给出回复。"""
        budgets = []

        def fake_urlopen(request, timeout=None):
            budget = json.loads(request.data.decode("utf-8"))["max_tokens"]
            budgets.append(budget)
            return FakeResponse(reply(budget))

        return budgets, fake_urlopen

    def test_reasoning_content_is_not_used_as_answer(self):
        # reasoning_content 装的是思考过程，拿它当终稿只会得到一堆复述
        with mock.patch("urllib.request.urlopen",
                        return_value=FakeResponse(self._response(
                            reasoning="用户要求我整理成总结……"))):
            with self.assertRaises(llm.LLMError):
                llm.chat(CONFIG, [])

    def test_length_truncation_escalates_max_tokens(self):
        budgets, fake = self._budget_spy(
            lambda _b: self._response(finish_reason="length"))
        with mock.patch("urllib.request.urlopen", side_effect=fake):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.chat(CONFIG, [], max_tokens=3072)
        self.assertEqual(budgets, [3072, 6144, 12288])
        self.assertIn("length", str(ctx.exception))
        self.assertIn("--llm-max-tokens", str(ctx.exception))

    def test_truncated_content_escalates_until_complete(self):
        def reply(budget):
            if budget >= 12288:
                return self._response("## 核心结论\n- A", finish_reason="stop")
            return self._response("思考过程漏进了正文……", finish_reason="length")

        budgets, fake = self._budget_spy(reply)
        with mock.patch("urllib.request.urlopen", side_effect=fake):
            self.assertEqual(llm.chat(CONFIG, [], max_tokens=3072),
                             "## 核心结论\n- A")
        self.assertEqual(budgets, [3072, 6144, 12288])

    def test_truncated_at_cap_returns_best_effort(self):
        budgets, fake = self._budget_spy(
            lambda b: self._response(f"截断于 {b}", finish_reason="length"))
        with mock.patch("urllib.request.urlopen", side_effect=fake):
            self.assertEqual(llm.chat(CONFIG, [], max_tokens=3072),
                             "截断于 12288")
        self.assertEqual(budgets, [3072, 6144, 12288])

    def test_escalation_stops_at_cap(self):
        budgets, fake = self._budget_spy(
            lambda _b: self._response(finish_reason="length"))
        with mock.patch("urllib.request.urlopen", side_effect=fake):
            with self.assertRaises(llm.LLMError):
                llm.chat(CONFIG, [], max_tokens=20000)
        self.assertEqual(budgets, [20000, 32768])

    def test_empty_content_without_length_fails_fast(self):
        calls = {"n": 0}

        def fake_urlopen(request, timeout=None):
            calls["n"] += 1
            return FakeResponse(self._response(finish_reason="stop"))

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.chat(CONFIG, [])
        self.assertEqual(calls["n"], 1)
        self.assertIn("stop", str(ctx.exception))

    def test_list_content_is_joined(self):
        payload = {"choices": [{"message": {"role": "assistant", "content": [
            {"type": "text", "text": "第一段"},
            {"type": "text", "text": "第二段"},
        ]}}]}
        with mock.patch("urllib.request.urlopen",
                        return_value=FakeResponse(payload)):
            self.assertEqual(llm.chat(CONFIG, []), "第一段第二段")


class ResolveConfigTest(unittest.TestCase):
    def _args(self, **kwargs):
        defaults = dict(llm_base_url=None, llm_api_key=None, llm_model=None,
                        llm_temperature=0.3, llm_max_tokens=0)
        defaults.update(kwargs)
        return mock.Mock(**defaults)

    def test_flags_win_over_env(self):
        with mock.patch.dict("os.environ", {"VIDSUM_LLM_API_KEY": "env-key",
                                            "VIDSUM_LLM_MODEL": "env-model"}):
            cfg = llm.resolve_config(self._args(llm_api_key="flag-key",
                                                llm_model="flag-model"))
        self.assertEqual((cfg.api_key, cfg.model), ("flag-key", "flag-model"))

    def test_env_used_when_no_flags(self):
        with mock.patch.dict("os.environ", {"VIDSUM_LLM_API_KEY": "env-key",
                                            "VIDSUM_LLM_MODEL": "env-model"}):
            cfg = llm.resolve_config(self._args())
        self.assertEqual((cfg.api_key, cfg.model), ("env-key", "env-model"))
        self.assertTrue(cfg.available)

    def test_openai_env_is_fallback(self):
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-xxx"}):
            cfg = llm.resolve_config(self._args())
        self.assertEqual(cfg.api_key, "sk-xxx")

    def test_missing_key_is_unavailable(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            cfg = llm.resolve_config(self._args())
        self.assertFalse(cfg.available)


if __name__ == "__main__":
    unittest.main()
