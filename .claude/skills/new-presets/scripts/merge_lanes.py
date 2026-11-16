"""Resolve a preset-lane merge: run from the repo root after `git merge` stops.

- crates/ms-engine/tests/presets_e2e.rs: keeps both sides of every conflict
  (ours first) and sets the BUNDLED and LONG_RUN array lengths to the real
  entry counts.
- README.md: keeps both sides, then re-pairs the whole Gallery table two
  presets per row (image row, caption row), so half-empty rows from either
  lane are filled.

Both files keep their line endings and are `git add`ed. Commit afterwards.
Also safe to run without conflicts, just to re-pair the gallery and fix the
array lengths.
"""
import glob
import os
import re
import subprocess
import sys

RS = "crates/ms-engine/tests/presets_e2e.rs"
README = "README.md"


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def union_conflicts(text):
    """Keeps ours then theirs for every conflict block (diff3 base dropped)."""
    out, state = [], None
    for line in text.splitlines(keepends=True):
        if line.startswith("<<<<<<< "):
            state = "ours"
        elif line.startswith("||||||| ") and state == "ours":
            state = "base"
        elif line.rstrip("\r\n") == "=======" and state in ("ours", "base"):
            state = "theirs"
        elif line.startswith(">>>>>>> ") and state == "theirs":
            state = None
        elif state != "base":
            out.append(line)
    if state is not None:
        sys.exit("unterminated conflict block")
    return "".join(out)


def fix_lengths(text):
    def fix(header_re, entry_re, text):
        m = re.search(header_re, text)
        if not m:
            sys.exit(f"no match for {header_re}")
        end = text.index("];", m.end())
        n = len(re.findall(entry_re, text[m.end():end], re.M))
        return text[: m.start(1)] + str(n) + text[m.end(1):], n

    text, b = fix(r"const BUNDLED: \[&str; (\d+)\] = \[", r'^\s*"', text)
    text, l = fix(r"const LONG_RUN: \[\(&str, usize\); (\d+)\] = \[", r'^\s*\("', text)
    print(f"{RS}: BUNDLED {b}, LONG_RUN {l}")
    return text


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def repair_gallery(text):
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    try:
        start = lines.index("## Gallery")
        sep = next(i for i in range(start, len(lines)) if lines[i].startswith("|---"))
    except (ValueError, StopIteration):
        sys.exit("README has no ## Gallery table")
    end = sep + 1
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    rows = lines[sep + 1 : end]
    imgs, caps = [], []
    for i in range(0, len(rows), 2):
        if "![" not in rows[i] or i + 1 >= len(rows):
            sys.exit(f"gallery row {i} is not an image row followed by a caption row: {rows[i][:80]}")
        imgs += [c for c in cells(rows[i]) if c]
        caps += [c for c in cells(rows[i + 1]) if c]
    if len(imgs) != len(caps):
        sys.exit(f"{len(imgs)} images but {len(caps)} captions")
    seen = set()
    for c in imgs:
        if c in seen:
            sys.exit(f"duplicate gallery image {c}")
        seen.add(c)
    new = []
    for i in range(0, len(imgs), 2):
        a, b = imgs[i : i + 2], caps[i : i + 2]
        if len(a) == 1:
            new += [f"| {a[0]} |  |", f"| {b[0]} |  |"]
        else:
            new += [f"| {a[0]} | {a[1]} |", f"| {b[0]} | {b[1]} |"]
    print(f"{README}: {len(imgs)} gallery entries")
    text = nl.join(lines[: sep + 1] + new + lines[end:])
    n = len([p for p in glob.glob("presets/*.splat") if os.path.basename(p) != "default.splat"])
    text, k = re.subn(r"It comes with \d+ presets", f"It comes with {n} presets", text)
    print(f"{README}: intro count {n}" if k else f"{README}: intro preset count not found")
    return text


def main():
    write(RS, fix_lengths(union_conflicts(read(RS))))
    write(README, repair_gallery(union_conflicts(read(README))))
    for path in (RS, README):
        if re.search(r"^(<<<<<<<|>>>>>>>) ", read(path), re.M):
            sys.exit(f"conflict markers left in {path}")
    subprocess.run(["git", "add", RS, README], check=True)
    left = subprocess.run(["git", "diff", "--name-only", "--diff-filter=U"],
                          capture_output=True, text=True).stdout.split()
    if left:
        print("still conflicted (resolve by hand):", *left)


if __name__ == "__main__":
    main()
