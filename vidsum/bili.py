"""B 站视频页 → 直音频流地址解析。

vidsum 的音频层只吃直媒体流（mp4/m3u8）或本地文件；B 站视频页是 HTML，
直接喂给 ffmpeg/PyAV 会以「无法打开音频源」失败。这里把 watch 页解析成
DASH 音频流的直链，并给出拉流要带的请求头（B 站 CDN 会校验 Referer）。

只用标准库，不引第三方依赖。需要外网——在能访问 bilibili.com 的机器上才有效。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .media import MediaError

_DEFAULT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/120.0.0.0 Safari/537.36")
_REFERER = "https://www.bilibili.com/"
_NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
_VIEW_URL = "https://api.bilibili.com/x/web-interface/view"
_PLAYURL_URL = "https://api.bilibili.com/x/player/wbi/v2"

# WBI 签名的重排表（社区公开实现，bilibili-API-collect）
_MIXIN_KEY_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]

_PAGE_RE = re.compile(
    r"^https?://(?:www\.|m\.)?bilibili\.com/video/"
    r"(BV[0-9A-Za-z]+|av\d+)",
    re.IGNORECASE,
)


class BiliError(MediaError):
    """B 站解析失败。继承 MediaError，pipeline 会按同一路径上报。"""


def is_bilibili_page(url: str) -> bool:
    """是否是 B 站视频页链接（BV 号或 av 号）。直链 mp4/m3u8 返回 False。"""
    return bool(_PAGE_RE.match((url or "").strip()))


def _request(url, *, referer, ua, cookie, timeout):
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", ua)
    req.add_header("Referer", referer)
    if cookie:
        req.add_header("Cookie", cookie)
    return urllib.request.urlopen(req, timeout=timeout)


def _get_json(url, *, referer, ua, cookie, timeout):
    try:
        with _request(url, referer=referer, ua=ua, cookie=cookie,
                      timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raise BiliError(f"B 站接口返回 HTTP {exc.code}: {url}") from exc
    except urllib.error.URLError as exc:
        raise BiliError(
            f"连不上 B 站（{exc.reason}）：{url}"
            "；若浏览器能打开 B 站，检查 python 出网：系统代理是否开启、"
            "防火墙是否拦截 python.exe"
        ) from exc
    except ValueError as exc:
        raise BiliError(f"B 站接口返回了非 JSON：{url}") from exc


def _get_html(url, *, referer, ua, cookie, timeout):
    try:
        with _request(url, referer=referer, ua=ua, cookie=cookie,
                      timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raise BiliError(f"打不开视频页（HTTP {exc.code}）：{url}") from exc
    except urllib.error.URLError as exc:
        raise BiliError(
            f"连不上 B 站（{exc.reason}）：{url}"
            "；若浏览器能打开 B 站，检查 python 出网：系统代理是否开启、"
            "防火墙是否拦截 python.exe"
        ) from exc


def extract_initial_state(html: str):
    """从页面里抠出 window.__INITIAL_STATE__ 的 JSON（花括号配平，容忍字符串内的括号）。"""
    marker = "window.__INITIAL_STATE__="
    i = html.find(marker)
    if i < 0:
        return None
    start = html.find("{", i + len(marker))
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(html)):
        ch = html[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(html[start:j + 1])
                except ValueError:
                    return None
    return None


def mixin_key(img_key: str, sub_key: str) -> str:
    """按重排表从 img_key+sub_key 生成 32 位 WBI 签名密钥。"""
    orig = (img_key + sub_key).encode("ascii", "ignore")
    if len(orig) < 64:
        raise BiliError("WBI 密钥不完整，B 站接口结构可能变了")
    return "".join(chr(orig[i]) for i in _MIXIN_KEY_TAB)[:32]


def wbi_sign(params: dict, img_key: str, sub_key: str, *, wts=None) -> dict:
    """给参数加 wts 时间戳与 w_rid 签名（WBI）。"""
    mixin = mixin_key(img_key, sub_key)
    merged = dict(params)
    merged["wts"] = int(wts if wts is not None else time.time())
    cleaned = {
        k: "".join(c for c in str(v) if c not in "!'()*")
        for k, v in sorted(merged.items())
    }
    query = urllib.parse.urlencode(cleaned)
    # 返回的既是签名依据、也是最终拼进请求串的参数：二者必须一致
    cleaned["w_rid"] = hashlib.md5((query + mixin).encode("utf-8")).hexdigest()
    return cleaned


def _wbi_keys(*, ua, cookie, timeout):
    data = _get_json(_NAV_URL, referer=_REFERER, ua=ua, cookie=cookie,
                     timeout=timeout)
    wbi = (data.get("data") or {}).get("wbi_img") or {}

    def key(url):
        name = (url or "").rsplit("/", 1)[-1]
        return name.rsplit(".", 1)[0]

    img_key, sub_key = key(wbi.get("img_url")), key(wbi.get("sub_url"))
    if not img_key or not sub_key:
        raise BiliError("没能从 nav 接口取到 WBI 密钥")
    return img_key, sub_key


def _pick_audio_url(playurl: dict) -> str:
    data = playurl.get("data") or {}
    audios = (data.get("dash") or {}).get("audio") or []
    if audios:
        best = max(audios, key=lambda a: int(a.get("bandwidth") or 0))
        url = best.get("base_url") or best.get("baseUrl")
        if url:
            return url
    durl = data.get("durl") or []
    if durl:
        url = durl[0].get("base_url") or durl[0].get("baseUrl")
        if url:
            return url
    raise BiliError("B 站没返回可用的音频流（可能被风控，或需要登录/大会员）")


def _view_meta(vid: str, *, ua, cookie, timeout) -> dict:
    """web-interface/view 接口拿视频元信息（aid/bvid/cid）。

    不依赖 watch 页的页面结构，比抠 __INITIAL_STATE__ 稳。
    """
    if vid[:2].lower() == "av":
        query = "aid=" + urllib.parse.quote(vid[2:])
    else:
        query = "bvid=" + urllib.parse.quote(vid)
    data = _get_json(f"{_VIEW_URL}?{query}", referer=_REFERER, ua=ua,
                     cookie=cookie, timeout=timeout)
    if data.get("code") != 0:
        raise BiliError(f"view 接口 code={data.get('code')}: "
                        f"{data.get('message') or ''}".strip())
    info = data.get("data") or {}
    if not info.get("cid") and not (info.get("pages") or []):
        raise BiliError("view 接口没返回 cid")
    return info


def resolve_audio_url(page_url: str, *, user_agent: str | None = None,
                      cookie: str | None = None, timeout: float = 20.0,
                      cid: int | None = None):
    """把 B 站视频页解析成 (直音频流 URL, 拉流请求头 dict)。

    返回的请求头至少含 Referer=https://www.bilibili.com/，B 站 CDN 会校验。
    """
    ua = user_agent or _DEFAULT_UA
    page_url = (page_url or "").strip()
    m = _PAGE_RE.match(page_url)
    if not m:
        raise BiliError(f"不是 B 站视频页链接：{page_url}")
    vid = m.group(1)

    # 主路径：view 接口拿 aid/bvid/cid，完全不看 watch 页长什么样
    note = ""
    try:
        video = _view_meta(vid, ua=ua, cookie=cookie, timeout=timeout)
    except BiliError as exc:
        video, note = None, f"（view 接口也失败：{exc}）"

    if video is None:
        # 回退：抠 watch 页的 __INITIAL_STATE__
        html = _get_html(page_url, referer=_REFERER, ua=ua, cookie=cookie,
                         timeout=timeout)
        state = extract_initial_state(html)
        if not state:
            raise BiliError(
                "视频页里找不到 __INITIAL_STATE__（页面结构可能变了）" + note)
        video = state.get("videoData") or state
    bvid = video.get("bvid")
    aid = video.get("aid") or (int(vid[2:]) if vid[:2].lower() == "av" else None)
    part_cid = cid or video.get("cid")
    if not part_cid:
        pages = video.get("pages") or []
        if pages:
            part_cid = pages[0].get("cid")
    if not part_cid:
        raise BiliError("拿不到视频 cid，无法请求播放地址")

    play = {"cid": part_cid, "fnval": 4048, "fnver": 0, "fourk": 1}
    if bvid:
        play["bvid"] = bvid
    elif aid:
        play["aid"] = aid
    else:
        raise BiliError("拿不到 BV/av 号")

    img_key, sub_key = _wbi_keys(ua=ua, cookie=cookie, timeout=timeout)
    signed = wbi_sign(play, img_key, sub_key)
    playurl = _get_json(f"{_PLAYURL_URL}?{urllib.parse.urlencode(signed)}",
                        referer=_REFERER, ua=ua, cookie=cookie, timeout=timeout)
    if playurl.get("code") != 0:
        raise BiliError("播放地址接口返回 code="
                        f"{playurl.get('code')}: {playurl.get('message') or ''}".strip())
    audio_url = _pick_audio_url(playurl)

    headers = {"Referer": _REFERER, "User-Agent": ua}
    if cookie:
        headers["Cookie"] = cookie
    return audio_url, headers
