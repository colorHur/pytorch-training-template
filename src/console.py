"""控制台输出的零依赖工具。

为什么单独开一个模块
------------------
`force_utf8_stdout()` 必须能被**每个可执行入口**在最早期调用，而
`src/__init__.py` 会连锁导入 torch（实测 `import src` 约 6.4 秒）。
实现若放在包里，`tools/` 下的启动器就只有两个坏选择：白等 6 秒，
或者抄一份副本。所以把实现放进这个只 `import sys` 的模块：

    # 1) 包内使用（src/main.py）—— 正常走包
    from src import force_utf8_stdout

    # 2) 不想触发 src/__init__.py（tools/ 下的启动器）
    sys.path.insert(0, str(ROOT / "src"))   # 挂**目录**，不是包
    from console import force_utf8_stdout

第二条路因为挂的是目录，完全跳过 `src/__init__.py`，import 代价约 0。
"""

from __future__ import annotations

import sys


def force_utf8_stdout() -> None:
    """把标准输出/错误强制切到 UTF-8，避免中文日志在非 UTF-8 环境下崩掉。

    ⚠️ 这个坑在本仓库**踩过两次**，所以它值得一个专门的模块和两条测试。

    - 第一次：`src/main.py` 打印中文训练日志，在 GitHub Actions 的 windows
      runner 上抛 `UnicodeEncodeError: 'charmap' codec ...`，整个训练带
      exit code 1 崩掉。
    - 第二次：新写的 `tools/ddp_launch.py` 又犯了同一个错，只是换了个文件、
      换了个 print 语句。这次是两条 DDP 冒烟测试一起红，而且连累 CI 的排错
      通道 —— 同样的 cp1252 环境把注解 hook 自己也搞崩了，于是现象退化成
      一句毫无信息量的 "Process completed with exit code 1"。

    结论是「下次记得改」靠不住，所以现在有
    `test_all_entrypoints_pin_utf8_stdout` 扫全部入口，漏一个就红。

    ---
    为什么本地永远复现不了：Python 在 **Windows 上输出到管道**时用系统
    locale 编码 —— 英文系统是 cp1252，而中文 locale（cp936）能正常编码中文。
    所以这个坑只在「管道 / 重定向 / CI」场景暴露，本地终端怎么跑都是绿的。

    `errors="replace"`：万一遇到确实无法编码的字符，降级成替代符，
    也不要让日志把训练打断。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
