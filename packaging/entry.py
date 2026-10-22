"""exe 的引导脚本：先进来设环境变量，再进 vidsum 的分发逻辑。

KMP_DUPLICATE_LIB_OK 必须在任何 OpenMP 运行库加载前设好，
引导脚本是整个进程的第一段 Python 代码，放这里最稳。
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from vidsum.entry import main

if __name__ == "__main__":
    import sys

    sys.exit(main())
