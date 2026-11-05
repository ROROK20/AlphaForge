"""Package metadata has ONE source of truth, and every copy of a claim agrees.

setuptools ignores setup.py's metadata when pyproject.toml's ``[project]`` table
declares ``dependencies``, so a second copy could only drift. These tests pin the
contract instead of trusting comments:

* setup.py declares no metadata (it is a bare ``setup()`` shim);
* every dependency floor equals the version requirements.txt pins and the
  suite runs against -- a floor is a claim that that version was tested;
* the supported-Python claim is identical in pyproject, the classifiers, the
  CI matrix, the README badge, CONTRIBUTING and the Dockerfile base image.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
PROJECT = PYPROJECT["project"]


def _pins() -> dict[str, str]:
    """Exact ``name==version`` pins from requirements.txt."""
    pins: dict[str, str] = {}
    for raw in (ROOT / "requirements.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[canonicalize_name(name)] = version.strip()
    return pins


def _floor(req: Requirement) -> str | None:
    floors = [s.version for s in req.specifier if s.operator == ">="]
    return floors[0] if floors else None


def _declared() -> list[Requirement]:
    reqs = [Requirement(r) for r in PROJECT["dependencies"]]
    for extra in ("torch", "dev"):
        reqs += [Requirement(r) for r in PROJECT["optional-dependencies"][extra]]
    return reqs


def _min_python() -> tuple[int, int]:
    m = re.fullmatch(r">=\s*3\.(\d+)", PROJECT["requires-python"].strip())
    assert m, f"requires-python must be a plain floor, got {PROJECT['requires-python']!r}"
    return (3, int(m.group(1)))


def _classifier_pythons() -> list[tuple[int, int]]:
    out = []
    for c in PROJECT.get("classifiers", []):
        m = re.fullmatch(r"Programming Language :: Python :: 3\.(\d+)", c)
        if m:
            out.append((3, int(m.group(1))))
    return sorted(out)


def test_setup_py_declares_no_metadata():
    tree = ast.parse((ROOT / "setup.py").read_text())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "setup"
    ]
    assert len(calls) == 1, "setup.py must contain exactly one setup() call"
    call = calls[0]
    assert not call.args and not call.keywords, (
        "setup.py passes metadata to setup(); pyproject.toml [project] overrides "
        "it silently, so it would be a second, ignored, drifting copy: "
        + ", ".join(k.arg or "**" for k in call.keywords)
    )


@pytest.mark.parametrize("req", _declared(), ids=lambda r: r.name)
def test_every_floor_is_the_tested_pin(req):
    """A floor claims "this version works". Only the pinned set is tested."""
    pins = _pins()
    name = canonicalize_name(req.name)
    floor = _floor(req)
    assert floor is not None, f"{req} has no >= floor"
    if name not in pins:
        # Only declared-but-unpinned deps may skip, and requirements.txt says
        # why for each (jupyter: not installed in the verified environment).
        assert name == "jupyter", f"{req.name} is declared but never pinned/tested"
        return
    assert floor == pins[name], (
        f"{req.name}: floor {floor} != tested pin {pins[name]} in requirements.txt"
    )


def test_every_runtime_dependency_is_pinned():
    pins = _pins()
    missing = [
        r.name for r in (Requirement(d) for d in PROJECT["dependencies"])
        if canonicalize_name(r.name) not in pins
    ]
    assert not missing, f"runtime deps with no tested pin: {missing}"


def test_classifiers_start_at_requires_python():
    pys = _classifier_pythons()
    assert pys, "no 'Programming Language :: Python :: 3.x' classifiers"
    assert pys[0] == _min_python(), (
        f"lowest classifier {pys[0]} != requires-python floor {_min_python()}"
    )


def test_ci_matrix_tests_exactly_the_classified_versions():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    m = re.search(r"python-version:\s*\[([^\]]+)\]", ci)
    assert m, "no python-version matrix in ci.yml"
    matrix = sorted(
        tuple(int(p) for p in v.strip().strip("'\"").split("."))
        for v in m.group(1).split(",")
    )
    assert matrix == _classifier_pythons(), (
        f"CI tests {matrix} but classifiers claim {_classifier_pythons()}"
    )


def test_readme_badge_matches_requires_python():
    readme = (ROOT / "README.md").read_text()
    m = re.search(r"badge/python-3\.(\d+)%2B", readme)
    assert m, "README python badge not found"
    assert (3, int(m.group(1))) == _min_python()


def test_contributing_states_the_same_floor():
    text = (ROOT / "CONTRIBUTING.md").read_text()
    claims = {(3, int(v)) for v in re.findall(r"Python 3\.(\d+)\+", text)}
    assert claims == {_min_python()}, f"CONTRIBUTING claims {sorted(claims)}"


def test_dockerfile_base_and_floors_agree_with_pyproject():
    docker = (ROOT / "Dockerfile").read_text()
    m = re.search(r"^FROM python:3\.(\d+)", docker, re.M)
    assert m and (3, int(m.group(1))) >= _min_python(), "Docker base below requires-python"
    declared = {
        canonicalize_name(r.name): _floor(r)
        for r in (Requirement(d) for d in PROJECT["dependencies"])
    }
    floors = re.findall(r'"([A-Za-z0-9_.\-]+)>=([0-9.]+)"', docker)
    assert floors, "Dockerfile installs no >= floors (pattern changed?)"
    for name, version in floors:
        key = canonicalize_name(name)
        assert key in declared, f"Dockerfile installs undeclared {name}"
        assert version == declared[key], (
            f"Dockerfile {name}>={version} != pyproject floor {declared[key]}"
        )


def test_ci_lint_is_pinned_and_its_rules_are_explicit():
    """CI's lint job must not change meaning when ruff publishes a release.

    Both halves are needed: an exact version pin, and an explicit ``select`` so the
    enforced set is written down in the repo rather than inherited from ruff's
    DEFAULT selection, which changes between releases.
    """
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    installs = re.findall(r"pip install[^\n]*\bruff\b[^\n]*", ci)
    assert installs, "ci.yml must install ruff"
    for line in installs:
        assert re.search(r"\bruff==\d+\.\d+\.\d+\b", line), f"ruff not pinned exactly: {line!r}"
    lint = PYPROJECT.get("tool", {}).get("ruff", {}).get("lint", {})
    assert lint.get("select"), "pyproject.toml must set [tool.ruff.lint] select explicitly"
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    pinned = set(re.findall(r"ruff==(\d+\.\d+\.\d+)", ci))
    assert all(f"ruff=={v}" in contributing for v in pinned), (
        "CONTRIBUTING.md must tell contributors to install the ruff version CI pins"
    )


def test_documented_timings_come_from_the_benchmark_script(capsys, monkeypatch):
    """docs/architecture.md's Performance table must be what
    scripts/benchmark.py measures (same operations), and no doc may carry an
    unmeasured speed multiplier ("100x" with no baseline)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    monkeypatch.setattr(bench, "REPEATS", 1)
    assert bench.main() == 0
    measured = [line.split("|")[1].strip() for line in capsys.readouterr().out.splitlines()
                if line.startswith("| ") and not line.startswith("| Operation")]
    docs = (ROOT / "docs" / "architecture.md").read_text()
    table = docs.split("## Performance", 1)[1]
    documented = [line.split("|")[1].strip() for line in table.splitlines()
                  if line.startswith("| ") and not line.startswith("| Operation")]
    assert documented == measured, "docs table rows must be the script's own rows"
    for path in ("README.md", "docs/architecture.md", "docs/usage.md"):
        text = (ROOT / path).read_text()
        assert not re.search(r"\b\d+x faster|no Python loops", text), path


def test_next_uses_this_repos_patched_postcss():
    """Next 15 asks for postcss 8.4.31 (two high advisories). The override points it
    at the repo's own, and the lockfile must carry no second copy."""
    import json

    package = json.loads((ROOT / "frontend" / "package.json").read_text())
    lock = json.loads((ROOT / "frontend" / "package-lock.json").read_text())["packages"]
    assert package.get("overrides", {}).get("next", {}).get("postcss") == "$postcss"
    assert "postcss" in package["devDependencies"]
    assert "node_modules/next/node_modules/postcss" not in lock
    major, minor, patch = (int(x) for x in lock["node_modules/postcss"]["version"].split("."))
    assert (major, minor, patch) >= (8, 5, 12)  # the first release past the high advisories


def test_every_test_the_readme_names_exists():
    """Every test named in the README's sample output exists."""
    root = Path(__file__).parent.parent
    named = re.findall(r"^(tests/\S+\.py)::(\w+)::(\w+) PASSED$", (root / "README.md").read_text(), re.M)
    assert named
    for path, cls, name in named:
        body = (root / path).read_text()
        assert re.search(rf"^class {cls}\b", body, re.M), f"{path}::{cls}"
        assert re.search(rf"^    def {name}\b", body, re.M), f"{path}::{cls}::{name}"


def test_the_readme_names_the_browser_install_the_smoke_test_needs():
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    assert "npx playwright install chromium" in readme
