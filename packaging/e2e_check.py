"""dist\\vidsum.exe 的端到端验证：冻结态子进程 spawn、中文日志编码、真实转写。

用法：先起 `dist\\vidsum\\vidsum.exe --web --port 8799`，再跑本脚本。
脚本会顺手确认 exe 旁边有 models\\whisper（没有就用硬链接镜像过去）。
"""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8799"
PROJECT = Path(__file__).resolve().parent.parent
EXE = PROJECT / "dist" / "vidsum" / "vidsum.exe"
OUT = str(PROJECT / "dist" / "e2e-out")


def ensure_models():
    """保证 exe 旁边有真目录形态的 models\whisper。

    HF 的缓存扫描不认 junction（会当缓存不存在转而去连 Hub），
    且 PyInstaller 每次构建都会清空 dist\vidsum，所以这里用
    packaging\link_models.ps1 以硬链接镜像恢复，秒完成不占空间。
    """
    if (EXE.parent / "models" / "whisper").is_dir():
        return
    script = PROJECT / "packaging" / "link_models.ps1"
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(script)],
        check=True, timeout=600)


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return resp.read()


def run_job(params, timeout):
    reply = post("/api/run", params)
    if not reply.get("ok"):
        print(f"FAIL /api/run 拒绝：{reply}")
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = json.loads(get("/api/status").decode("utf-8"))
        if status.get("state") not in ("running", "idle"):
            return status
        time.sleep(2)
    print("FAIL 任务超时")
    return None


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))
    return cond


def main():
    ensure_models()
    ok = True

    html = get("/").decode("utf-8")
    ok &= check("前端页面从 bundle 里出来", "<html" in html.lower()
                and "vidsum" in html.lower())

    status = run_job({"input": r"E:\nope\不存在的视频.mp4", "asr": "dummy",
                      "output": OUT}, 120)
    if status is None:
        return 1
    logs = "\n".join(status.get("logs") or [])
    ok &= check("冻结态子进程跑通并报错", status["state"] == "error",
                f"state={status['state']}")
    ok &= check("中文日志无乱码", "\ufffd" not in logs and "错误：" in logs,
                logs.strip().splitlines()[-1][:80] if logs.strip() else "")

    status = run_job({"input": r"E:\project2\vidsum-out\audio_p1\audio.mp3",
                      "asr": "faster-whisper", "model_size": "medium",
                      "no_summary": True, "output": OUT}, 600)
    if status is None:
        return 1
    logs = "\n".join(status.get("logs") or [])
    ok &= check("真实转写任务完成", status["state"] == "done",
                f"state={status['state']} err={status.get('error')}")
    ok &= check("GPU 被探测到", "cuda" in logs.lower(),
                str([l for l in logs.splitlines() if "cuda" in l.lower()][:1]))
    ok &= check("模型从 exe 相邻目录加载", "models" in logs.lower(),
                str([l for l in logs.splitlines() if "模型" in l][:1]))
    files = status.get("files") or []
    names = [f.get("name") for f in files]
    ok &= check("transcript.md 生成", "transcript.md" in names, str(names))
    transcript = status.get("transcript") or ""
    ok &= check("转写有实际内容", len(transcript) > 50, f"{len(transcript)} 字符")
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in transcript)
    ok &= check("转写含中文正文", cjk > 20, f"{cjk} 个汉字")
    ok &= check("全链路中文日志无乱码", "\ufffd" not in logs)

    print("E2E", "全部通过" if ok else "存在失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
