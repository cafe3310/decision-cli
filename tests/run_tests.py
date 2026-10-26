#!/usr/bin/env python3
"""
Test runner for decision-cli test suite.
"""

import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    tests_dir = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(tests_dir),
        pattern="test_*.py",
        top_level_dir=str(tests_dir.parent),
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
