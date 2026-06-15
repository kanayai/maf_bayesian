from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from src.io.result_selection import existing_netcdf_path


class ExistingNetcdfPathTests(unittest.TestCase):
    def test_returns_resolved_explicit_netcdf_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "posterior.nc"
            result.touch()

            self.assertEqual(existing_netcdf_path(str(result)), result.resolve())

    def test_rejects_missing_result(self) -> None:
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "does not exist"):
            existing_netcdf_path("missing.nc")

    def test_rejects_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(argparse.ArgumentTypeError, "not a file"):
                existing_netcdf_path(directory)

    def test_rejects_non_netcdf_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "posterior.csv"
            result.touch()

            with self.assertRaisesRegex(argparse.ArgumentTypeError, ".nc extension"):
                existing_netcdf_path(str(result))


if __name__ == "__main__":
    unittest.main()
