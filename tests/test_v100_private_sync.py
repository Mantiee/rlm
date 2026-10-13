import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlm.v100 import private_sync, release_selector
from rlm.v100.managed_bootstrap import refresh_private_sync


class PrivateControlTests(unittest.TestCase):
    def test_private_sync_refreshes_only_after_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "verified-release"
            files = (
                "rlm/v100/private_sync.py",
                "rlm/v100/release_selector.py",
                "tools/audit-synta-hosts.py",
            )
            for name in files:
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"new verified source\n")
            refresh_private_sync(root, folder)
            destination = root / "tools/private-sync"
            self.assertFalse(destination.exists())
            destination.mkdir(parents=True)
            (destination / "private_sync.py").write_bytes(b"old")
            refresh_private_sync(root, folder)
            for name in files:
                self.assertEqual(
                    (destination / Path(name).name).read_bytes(), b"new verified source\n"
                )

    def test_public_repository_rejected_before_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(private_sync, "GitHub") as constructor,
                patch.object(private_sync, "audit") as audit,
            ):
                constructor.return_value.private.side_effect = ValueError("not private")
                with self.assertRaises(ValueError):
                    private_sync.run(Path(directory))
                audit.assert_not_called()

    def test_credentials_are_not_in_uploaded_snapshot(self):
        result = private_sync.clean(
            {
                "secret": "abc",
                "authorization": "Bearer abc",
                "state": "ready",
                "batch_tokens": 256,
                "detail": "failed ghp_abc123ABC123 and github_pat_XYZ123",
            }
        )
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["batch_tokens"], 256)
        self.assertNotIn("secret", result)
        self.assertNotIn("authorization", result)
        self.assertNotIn("ghp_", result["detail"])
        self.assertNotIn("github_pat_", result["detail"])

    def test_only_owned_immutable_release_pins_accepted(self):
        valid = {
            "schema": release_selector.SCHEMA,
            "repository": "Mantiee/rlm",
            "mode": "pinned",
            "revision": "a" * 40,
        }
        self.assertEqual(release_selector.validate(valid)["revision"], "a" * 40)
        for change in (
            {"revision": "main"},
            {"revision": 123},
            {"repository": "other/repo"},
            {"mode": "shell"},
        ):
            with self.assertRaises(ValueError):
                release_selector.validate(valid | change)

    def test_unconfigured_controller_preserves_public_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(release_selector.selection(Path(directory)))

    def test_missing_controller_uses_last_valid_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "research/private-sync/controller-release.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "schema": release_selector.SCHEMA,
                        "repository": "Mantiee/rlm",
                        "mode": "pinned",
                        "revision": "b" * 40,
                    }
                )
            )
            with patch.object(release_selector.urllib.request, "build_opener") as opener:
                opener.return_value.open.side_effect = OSError("peer offline")
                self.assertEqual(release_selector.selection(root, peer=True), "b" * 40)

    def test_unchanged_telemetry_creates_no_commit(self):
        github = object.__new__(private_sync.GitHub)
        value = {"state": "ready"}
        content = base64.b64encode(
            json.dumps(value, indent=2, ensure_ascii=False).encode()
        ).decode()
        with (
            patch.object(github, "content", return_value={"content": content}),
            patch.object(github, "request") as request,
        ):
            github.write("hosts/latest.json", value, "main")
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
