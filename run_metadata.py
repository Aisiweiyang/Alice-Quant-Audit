"""Alice 回测目录和运行记录。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = PROJECT_DIR
VERSION_PATH = os.path.join(PROJECT_DIR, "VERSION")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def project_version() -> str:
    with open(VERSION_PATH, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def build_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    stamp = current.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    version_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", project_version()).strip("-.")
    if not version_slug:
        raise ValueError("VERSION must contain at least one safe character")
    return f"{version_slug}_{stamp}"


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id 只能包含字母、数字、点、下划线和短横线，长度不超过 80")
    return run_id


def prepare_run_directory(output_root: str, run_id: str) -> str:
    safe_run_id = validate_run_id(run_id)
    root = os.path.abspath(os.path.expanduser(output_root))
    target = os.path.abspath(os.path.join(root, safe_run_id))
    if os.path.commonpath([root, target]) != root:
        raise ValueError("运行目录越过 output_root")
    if os.path.exists(target):
        raise FileExistsError(f"运行目录已存在，拒绝覆盖: {target}")
    os.makedirs(target)
    return target


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: str) -> str:
    try:
        value = os.path.relpath(path, WORKSPACE_DIR)
    except ValueError:
        value = os.path.basename(path)
    return value.replace(os.sep, "/")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_fingerprint(cache_path: str | None) -> dict[str, Any] | None:
    if not cache_path:
        return None
    path = os.path.abspath(os.path.expanduser(cache_path))
    stat = os.stat(path)
    result = {
        "path": _display_path(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }
    metadata_file = f"{path}.metadata.json"
    if os.path.isfile(metadata_file):
        result["metadata"] = {
            "path": _display_path(metadata_file),
            "size_bytes": os.path.getsize(metadata_file),
            "sha256": _sha256(metadata_file),
        }
    return result


def _run_git(*args: str) -> tuple[str | None, str | None]:
    executable = shutil.which("git")
    if not executable:
        return None, "git executable not found"
    try:
        result = subprocess.run(
            [executable, "-C", WORKSPACE_DIR, *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        return None, str(detail).strip()
    return result.stdout.strip(), None


def repository_state() -> dict[str, Any]:
    commit, error = _run_git("rev-parse", "HEAD")
    if not commit:
        return {
            "available": False,
            "commit": None,
            "branch": None,
            "dirty": None,
            "error": error or "unknown git error",
        }
    branch, branch_error = _run_git("branch", "--show-current")
    status, status_error = _run_git("status", "--porcelain", "--untracked-files=all")
    root, root_error = _run_git("rev-parse", "--show-toplevel")
    errors = [item for item in (branch_error, status_error, root_error) if item]
    return {
        "available": not errors,
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status) if status_error is None else None,
        "root": _display_path(root) if root else None,
        "error": "; ".join(errors) if errors else None,
    }


def git_commit() -> str | None:
    """Backward-compatible accessor used by older callers."""
    return repository_state().get("commit")


def config_snapshot(config: type) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name in sorted(dir(config)):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            continue
        snapshot[name] = value
    return snapshot


def config_fingerprint(config: type) -> str:
    return _json_sha256(config_snapshot(config))


def artifact_inventory(run_dir: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for root, _, files in os.walk(run_dir):
        for filename in sorted(files):
            if filename == "manifest.json":
                continue
            path = os.path.join(root, filename)
            artifacts.append(
                {
                    "path": os.path.relpath(path, run_dir).replace(os.sep, "/"),
                    "size_bytes": os.path.getsize(path),
                    "sha256": _sha256(path),
                }
            )
    return artifacts


def manifest_path(path: str, start: str) -> str:
    absolute_path = os.path.abspath(path)
    try:
        display_path = os.path.relpath(absolute_path, start)
    except ValueError:
        display_path = absolute_path
    return display_path.replace("\\", "/")


def build_manifest(*, args: Any, run_id: str, run_dir: str, config: type) -> dict[str, Any]:
    repository = repository_state()
    strategy_config = config_snapshot(config)
    return {
        "schema_version": 2,
        "run_id": run_id,
        "code_version": project_version(),
        "status": "finished",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": repository.get("commit"),
        "repository": repository,
        "command": sys.argv,
        "period": {"start": args.start, "end": args.end},
        "cache": cache_fingerprint(args.cache),
        "output_dir": manifest_path(run_dir, WORKSPACE_DIR),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "strategy_config": strategy_config,
        "strategy_config_sha256": _json_sha256(strategy_config),
        "artifacts": artifact_inventory(run_dir),
    }


def write_manifest(run_dir: str, manifest: dict[str, Any]) -> str:
    path = os.path.join(run_dir, "manifest.json")
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return path
