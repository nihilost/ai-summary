"""统一入口：一个可执行体分发到三种模式。

- 无参数或 `--web`        → 本地 Web 控制台（打包后的默认体验）
- `--job-runner <json>`   → 子任务运行器（Web 控制台拉起的干活进程）
- 其余参数                → 原样交给命令行接口

PyInstaller 冻结后子进程没有解释器可用，只能自调用
`vidsum.exe --job-runner ...`，这一层分发让源码态和 exe 态
共用同一套启动逻辑。
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] == "--web":
        from . import webapp
        return webapp.main(argv[1:])
    if argv[0] == "--job-runner":
        from . import _job_runner
        return _job_runner.main(argv[1:])
    from . import cli
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
