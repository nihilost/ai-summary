"""测试共用的辅助基类。"""

from __future__ import annotations

import atexit
import shutil
import unittest
import uuid
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent / ".test-tmp"


@atexit.register
def _cleanup_base() -> None:
    shutil.rmtree(_BASE, ignore_errors=True)


class TempDirCase(unittest.TestCase):
    """为每个测试建一个独立的临时工作目录。

    目录放在项目内的 .test-tmp/ 而非 tempfile.mkdtemp()：后者在某些
    受限环境（容器/沙箱的文件系统层）里建出的目录不可写，会把环境问题
    和断言失败混在一起。进程退出时整体删除 .test-tmp。
    """

    def make_dir(self, prefix: str = "vidsum-test-") -> Path:
        path = _BASE / f"{prefix}{uuid.uuid4().hex[:12]}"
        path.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path
