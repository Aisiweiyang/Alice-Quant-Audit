import unittest

from scripts.seed_long_history_from_recent import validate_compatibility


class SeedLongHistoryTests(unittest.TestCase):
    def test_compatible_audited_cache_is_accepted(self):
        target = {"source": "baostock", "adjust": "2"}
        recent = {
            "source": "baostock",
            "adjust": "2",
            "quality": {"formal_ready": True},
        }
        validate_compatibility(target, recent)

    def test_non_audited_cache_is_rejected(self):
        target = {"source": "baostock", "adjust": "2"}
        recent = {
            "source": "baostock",
            "adjust": "2",
            "quality": {"formal_ready": False},
        }
        with self.assertRaisesRegex(ValueError, "formal_ready"):
            validate_compatibility(target, recent)

    def test_mixed_sources_are_rejected(self):
        target = {"source": "baostock", "adjust": "2"}
        recent = {
            "source": "akshare-tencent",
            "adjust": "2",
            "quality": {"formal_ready": True},
        }
        with self.assertRaisesRegex(ValueError, "source"):
            validate_compatibility(target, recent)
