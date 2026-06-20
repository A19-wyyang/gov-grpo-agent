import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from gov_grpo_agent.packaging import build_server_package


class PackagingTests(unittest.TestCase):
    def test_build_server_package_includes_code_and_excludes_local_artifacts(self):
        with TemporaryDirectory() as temp_dir:
            package_path = build_server_package(
                root_dir=Path.cwd(),
                output_path=Path(temp_dir) / "server_bundle.zip",
            )

            self.assertTrue(package_path.exists())
            with ZipFile(package_path) as archive:
                names = set(archive.namelist())

            self.assertIn("gov_grpo_agent/cli.py", names)
            self.assertIn("tests/test_runtime.py", names)
            self.assertIn("README.md", names)
            self.assertIn("pyproject.toml", names)
            self.assertNotIn("artifacts/mvp/summary.json", names)
            self.assertTrue(all(not name.startswith(".git/") for name in names))
            self.assertTrue(all("__pycache__" not in name for name in names))
            self.assertTrue(all(not name.endswith(".pyc") for name in names))
