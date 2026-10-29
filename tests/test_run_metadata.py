from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from run_metadata import (
    build_manifest,
    build_run_id,
    cache_fingerprint,
    config_fingerprint,
    config_snapshot,
    manifest_path,
    prepare_run_directory,
    project_version,
    repository_state,
    validate_run_id,
    write_manifest,
)


class ExampleConfig:
    alpha = 1
    weights = {"momentum": 0.5}

    @classmethod
    def ignored_callable(cls):
        return None


class RunMetadataTests(unittest.TestCase):
    def test_version_and_generated_run_id_are_stable(self):
        now = datetime(2026, 8, 10, 12, 34, 56, 789000, tzinfo=timezone.utc)
        self.assertEqual(project_version(), "0.1.0")
        self.assertEqual(build_run_id(now), "0.1.0_20260810_123456_789")

    def test_run_id_rejects_paths_and_existing_directories(self):
        with self.assertRaises(ValueError):
            validate_run_id("../overwrite")
        with tempfile.TemporaryDirectory() as root:
            target = prepare_run_directory(root, "experiment-001")
            self.assertTrue(os.path.isdir(target))
            with self.assertRaises(FileExistsError):
                prepare_run_directory(root, "experiment-001")

    def test_cache_fingerprint_records_sha256(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"trusted-cache")
            path = handle.name
        metadata_path = f"{path}.metadata.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump({"source": "test"}, handle)
            result = cache_fingerprint(path)
            self.assertEqual(result["size_bytes"], len(b"trusted-cache"))
            self.assertEqual(result["sha256"], hashlib.sha256(b"trusted-cache").hexdigest())
            self.assertEqual(
                result["metadata"]["path"].split("/")[-1], os.path.basename(metadata_path)
            )
            self.assertEqual(len(result["metadata"]["sha256"]), 64)
        finally:
            os.unlink(path)
            os.unlink(metadata_path)

    def test_config_fingerprint_is_order_stable(self):
        first = config_fingerprint(ExampleConfig)
        second = config_fingerprint(ExampleConfig)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_repository_state_records_branch_and_dirty_flag(self):
        values = [
            ("abc123", None),
            ("codex/alice-v4", None),
            (" M alice_framework_v4/run_metadata.py", None),
            ("C:/repo", None),
        ]
        with patch("run_metadata._run_git", side_effect=values):
            state = repository_state()
        self.assertTrue(state["available"])
        self.assertEqual(state["commit"], "abc123")
        self.assertEqual(state["branch"], "codex/alice-v4")
        self.assertTrue(state["dirty"])

    def test_manifest_links_repository_config_and_hashed_artifacts(self):
        repository = {
            "available": True,
            "commit": "abc123",
            "branch": "codex/alice-v4",
            "dirty": False,
            "root": ".",
            "error": None,
        }
        args = SimpleNamespace(start="2025-01-01", end="2025-01-31", cache=None)
        with tempfile.TemporaryDirectory() as run_dir:
            artifact = os.path.join(run_dir, "metrics.json")
            with open(artifact, "w", encoding="utf-8") as handle:
                json.dump({"sharpe": 1.0}, handle)
            with patch("run_metadata.repository_state", return_value=repository):
                manifest = build_manifest(
                    args=args,
                    run_id="experiment-001",
                    run_dir=run_dir,
                    config=ExampleConfig,
                )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["repository"]["commit"], "abc123")
        self.assertEqual(manifest["strategy_config_sha256"], config_fingerprint(ExampleConfig))
        self.assertEqual(len(manifest["artifacts"][0]["sha256"]), 64)

    def test_manifest_path_falls_back_to_absolute_path_across_drives(self):
        with (
            patch("run_metadata.os.path.abspath", return_value=r"C:\temp\alice-run"),
            patch(
                "run_metadata.os.path.relpath",
                side_effect=ValueError("path is on a different drive"),
            ),
        ):
            path = manifest_path(r"C:\temp\alice-run", r"D:\workspace")
        self.assertEqual(path, "C:/temp/alice-run")

    def test_manifest_is_create_only_and_config_is_serializable(self):
        with tempfile.TemporaryDirectory() as run_dir:
            manifest = {"strategy_config": config_snapshot(ExampleConfig)}
            path = write_manifest(run_dir, manifest)
            with open(path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            self.assertEqual(stored["strategy_config"]["weights"]["momentum"], 0.5)
            self.assertNotIn("ignored_callable", stored["strategy_config"])
            with self.assertRaises(FileExistsError):
                write_manifest(run_dir, manifest)


if __name__ == "__main__":
    unittest.main()
    (build_manifest,)
