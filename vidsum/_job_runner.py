"""子进程任务运行器：读 JSON 参数跑流水线，独立进程便于可靠终止。

Web 控制台把参数序列化成临时 JSON，用 `python -m vidsum._job_runner <json>`
拉起本模块；stdout 逐行流回父进程当实时日志。模型加载卡死时，父进程
terminate() 能直接杀掉整个子进程，不会再出现“停止按钮无效”。
"""

from __future__ import annotations

import json
import sys
import types

from . import pipeline


def _force_utf8_output() -> None:
    """管道场景下把标准输出切成 UTF-8。

    Web 控制台按 UTF-8 读子进程输出。开发态这由 _child_env() 里的
    PYTHONIOENCODING 保证，但 PyInstaller 冻结后该变量被忽略，stdout
    仍是中文 Windows 的 GBK，整条日志流都会变乱码。这里显式 reconfigure，
    源码态和 exe 态行为一致。交互式终端（isatty）保持本地编码——
    那种场合 GBK 才是对的。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and not stream.isatty():
                stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    _force_utf8_output()
    if not argv:
        print("用法：python -m vidsum._job_runner <params.json>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as fh:
        data = json.load(fh)
    args = types.SimpleNamespace(**data)
    try:
        return pipeline.run(args)
    except KeyboardInterrupt:
        print("\n已中断。重跑同一参数可从断点继续。", file=sys.stderr)
        return 130
    except Exception as exc:  # 统一出口：把异常翻成可读的一行
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
