#!/usr/bin/env python3
"""Last-resort fallback: write the topology of the first parsable input gene tree."""
import re, sys
def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src, encoding="utf-8").read()
    for tree in text.split(";"):
        t = tree.strip()
        if not t.startswith("("):
            continue
        t = re.sub(r"\[[^\]]*\]", "", t)
        t = re.sub(r":[^,();]*", "", t)          # branch lengths
        t = re.sub(r"\)[^,();]+", ")", t)        # internal labels / supports
        t = re.sub(r"\s+", "", t)
        open(dst, "w").write(t + ";\n")
        return
    sys.exit(1)
main()
