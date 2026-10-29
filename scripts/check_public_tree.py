"""Check exactly the tracked publication surface, without reading local caches."""

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {".venv", ".env", "output", "local_data", "node_modules", "__pycache__"}
FORBIDDEN_SUFFIXES = {".pkl", ".parquet", ".db", ".sqlite", ".docx", ".pdf", ".zip", ".log", ".pyc"}
PATTERNS = [
    re.compile(r"gh[pousr]_" + r"[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_" + r"[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA" + r"[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"[A-Za-z]:\\Users\\" + r"[^\s/\\]+"),
]


def main():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    paths = [Path(p) for p in result.stdout.decode("utf-8").split("\0") if p]
    if not paths:
        raise SystemExit("No tracked files: stage the intended public tree first")
    errors = []
    for path in paths:
        if FORBIDDEN_PARTS.intersection(path.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Forbidden published file: {path.as_posix()}")
            continue
        content = (ROOT / path).read_bytes()
        if len(content) > 2_000_000:
            errors.append(f"Oversized publication file: {path.as_posix()}")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            errors.append(f"Non-text publication file: {path.as_posix()}")
            continue
        if any(pattern.search(text) for pattern in PATTERNS):
            errors.append(f"Possible credential or personal path: {path.as_posix()}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Public tree check passed: {len(paths)} text files; no blocked artifacts or pattern matches"
    )


if __name__ == "__main__":
    main()
