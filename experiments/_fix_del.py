"""批量把 `del a, b, c` 拆成多行单变量 del —— TorchScript 不支持多操作数 del。

  torch.jit.frontend.NotSupportedError: del with more than one operand is not supported

语义完全等价（del 只是解绑局部名），改动机械、可脚本化、可验证。
用法: python _fix_del.py <file...>   （不传参数则处理 alphastarmini/core/arch/ 下全部）
"""
import re
import sys
import pathlib

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def split_del(path):
    p = pathlib.Path(path)
    src = p.read_text(encoding="utf-8")
    lines = src.split("\n")
    out, changed, skipped = [], 0, []
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)del\s+(.+?)\s*$", line)
        if m and "," in m.group(2):
            names = [x.strip() for x in m.group(2).split(",")]
            if all(IDENT.match(n) for n in names):
                indent = m.group(1)
                for n in names:
                    out.append("%sdel %s" % (indent, n))
                changed += 1
                continue
            skipped.append((i + 1, line.strip()[:70]))
        out.append(line)
    if changed:
        p.write_text("\n".join(out), encoding="utf-8")
    return changed, skipped


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(str(x) for x in pathlib.Path("alphastarmini/core/arch").glob("*.py"))
    total = 0
    for f in files:
        if not pathlib.Path(f).exists():
            continue
        c, sk = split_del(f)
        total += c
        if c:
            print("  %-52s 拆分了 %d 处" % (f, c))
        for ln, txt in sk:
            print("    [跳过] %s:%d  %s" % (f, ln, txt))
    print("合计拆分 %d 处" % total)


if __name__ == "__main__":
    main()
