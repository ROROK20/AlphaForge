"""Build shim. ALL package metadata lives in pyproject.toml.

This file exists only so tooling that still invokes ``setup.py`` keeps working.
It must not declare metadata: setuptools silently ignores it in favour of
pyproject.toml's ``[project]`` table, so a second copy could only drift.
``tests/test_packaging.py`` fails if a keyword argument appears here.
"""

from setuptools import setup

setup()
