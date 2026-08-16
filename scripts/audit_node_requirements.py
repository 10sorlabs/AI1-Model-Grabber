#!/usr/bin/env python3
"""Fail the build when a pinned node pack asks for something with no wheel.

A customer's pod sat at 92% on "Installing custom node ComfyUI-Impact-Pack…" for 46
minutes because the last line of that pack's requirements.txt is
git+https://github.com/facebookresearch/sam2. A VCS requirement has no wheel, so pip
runs a PEP 517 build, and build isolation is --ignore-installed by definition - so pip
downloaded a second complete torch plus the whole nvidia CUDA stack into a temp overlay
to read one package's metadata, at the 87-142 KB/s that pod measured against PyPI.

That specific case is handled at runtime now, with --no-build-isolation. This script is
the guard for the next one: it walks every node pack in the catalog at its pinned sha
and reports every requirement that cannot install as a plain wheel on a pod.

The wheel check goes through packaging, never through substring matching on filenames,
and it tests against an explicit target - CPython 3.12, manylinux x86_64 - rather than
the CI runner's own sys_tags, because the runner is not the pod. Both halves of that
sentence are lessons: doing this audit by hand, I filtered for the literal string
"cp312", concluded opencv-contrib-python had no wheel, and was wrong. Its wheel is
tagged cp37-abi3, and abi3 covers 3.12. That mistake is what this script exists to make
impossible, and there is a test pinning exactly it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "catalog" / "workflows.json"
KNOWN_SOURCE_BUILDS = REPO_ROOT / "docker" / "known-source-builds.txt"

VCS_PREFIXES = ("git+", "hg+", "svn+", "bzr+")

# The pod: python3.12 on a CUDA base image, glibc x86_64. Written out rather than taken
# from sys_tags, because this runs on a GitHub runner and the runner is not the pod.
TARGET_PYTHON = (3, 12)
TARGET_PLATFORMS = (
    [f"manylinux_2_{minor}_x86_64" for minor in range(41, 16, -1)]
    + [
        "manylinux2014_x86_64",
        "manylinux2010_x86_64",
        "manylinux1_x86_64",
        "linux_x86_64",
    ]
)


@dataclass(frozen=True)
class Finding:
    package: str
    node: str
    reason: str

    def __str__(self) -> str:
        return f"{self.package} ({self.node}): {self.reason}"


@lru_cache(maxsize=1)
def pod_tags() -> frozenset:
    """Every wheel tag a pod's pip would accept.

    cpython_tags yields the abi3 tags of older interpreters too - cp37-abi3 among them -
    which is the whole reason this is generated rather than string-matched.
    """
    tags = set(cpython_tags(python_version=TARGET_PYTHON, platforms=TARGET_PLATFORMS))
    tags |= set(
        compatible_tags(python_version=TARGET_PYTHON, platforms=TARGET_PLATFORMS)
    )
    return frozenset(tags)


def wheel_is_installable(filename: str) -> bool:
    try:
        tags = parse_wheel_filename(filename)[3]
    except InvalidWheelFilename:
        return False
    return bool(tags & pod_tags())


def files_for(payload: dict, version: str | None) -> list[dict]:
    """The distribution files pip would choose from for this requirement.

    A pinned == goes to that release; anything else takes the latest, which is what an
    unpinned install resolves to.
    """
    if version:
        return payload.get("releases", {}).get(version, [])
    return payload.get("urls", [])


def pinned_version(requirement: Requirement) -> str | None:
    for specifier in requirement.specifier:
        if specifier.operator in ("==", "==="):
            return specifier.version
    return None


def name_from_url(url: str) -> str:
    """Best-effort package name for a VCS or direct-URL requirement.

    Only used to label the finding and to match the allowlist, never to install
    anything, so "close enough to identify it in a report" is the bar.
    """
    if "#egg=" in url:
        return url.split("#egg=", 1)[1].split("&", 1)[0]
    path = urlsplit(url.split("#", 1)[0]).path.rstrip("/")
    tail = path.rsplit("/", 1)[-1] if path else url
    for suffix in (".git", ".zip", ".tar.gz", ".whl"):
        if tail.endswith(suffix):
            tail = tail[: -len(suffix)]
    return tail or url


def classify_requirement(raw: str, node: str, fetch) -> Finding | None:
    """One requirements.txt line. A Finding means "this builds from source on a pod".

    `fetch` takes a package name and returns its PyPI JSON payload, so the classifier
    itself is pure and the tests never touch the network.
    """
    line = raw.split(" #", 1)[0].split("\t#", 1)[0].strip()
    if line.startswith("#"):
        line = ""
    if not line:
        return None
    # -r nested.txt, --extra-index-url, -e . - pip options rather than requirements.
    if line.startswith("-"):
        return None

    lowered = line.lower()
    if lowered.startswith(VCS_PREFIXES) or " @ git+" in lowered:
        return Finding(
            name_from_url(line.split("@", 1)[-1].strip() if " @ " in line else line),
            node,
            "VCS requirement: pip must build it from a checkout, and build isolation "
            "then re-downloads its whole build-requires set",
        )
    if "://" in line:
        return Finding(
            name_from_url(line.split(" @ ", 1)[-1].strip()),
            node,
            "direct URL requirement: pip cannot choose a wheel for the pod",
        )

    try:
        requirement = Requirement(line)
    except InvalidRequirement:
        return Finding(line, node, "could not be parsed as a requirement")

    try:
        payload = fetch(requirement.name)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal on its own
        return Finding(requirement.name, node, f"could not be checked on PyPI: {exc}")
    if payload is None:
        return Finding(requirement.name, node, "not found on PyPI")

    version = pinned_version(requirement)
    candidates = files_for(payload, version)
    wheels = [
        entry.get("filename", "")
        for entry in candidates
        if entry.get("packagetype") == "bdist_wheel"
    ]
    if any(wheel_is_installable(filename) for filename in wheels):
        return None

    where = f" for {version}" if version else ""
    if wheels:
        return Finding(
            requirement.name,
            node,
            f"has wheels{where} but none for CPython 3.12 on manylinux x86_64: "
            + ", ".join(sorted(wheels)[:4]),
        )
    return Finding(
        requirement.name,
        node,
        f"sdist only{where}: every install is a source build on the pod",
    )


def read_known_source_builds(path: Path) -> dict[str, str]:
    """name  # reason, one per line. The reason is required, and is the point."""
    allowed: dict[str, str] = {}
    if not path.is_file():
        return allowed
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, reason = line.partition("#")
        if name.strip():
            allowed[canonicalize_name(name.strip())] = reason.strip()
    return allowed


def unlisted(findings: list[Finding], allowed: dict[str, str]) -> list[Finding]:
    return [
        finding
        for finding in findings
        if canonicalize_name(finding.package) not in allowed
    ]


def node_packs(catalog: dict) -> list[tuple[str, str, str]]:
    """Unique (name, repo, ref) across every workflow. The same pack is listed twice."""
    seen: dict[tuple[str, str], str] = {}
    for workflow in catalog.get("workflows", []):
        for node in workflow.get("custom_nodes", []):
            repo = str(node.get("repo", "")).strip()
            ref = str(node.get("ref", "")).strip()
            if repo and ref:
                seen.setdefault((repo, ref), str(node.get("name", repo)))
    return [(name, repo, ref) for (repo, ref), name in seen.items()]


def pypi_payload(name: str) -> dict | None:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{name}/json",
        headers={"User-Agent": "10sorlabs-node-requirements-audit"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def requirements_at_ref(repo: str, ref: str, workdir: Path) -> str:
    """Clone blobless, detach at the pinned sha, and read requirements.txt."""
    checkout = workdir / ref[:12]
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--quiet", repo, str(checkout)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--detach", "--quiet", ref],
        check=True,
        capture_output=True,
    )
    requirements = checkout / "requirements.txt"
    return requirements.read_text(encoding="utf-8") if requirements.is_file() else ""


def audit(catalog: dict, fetch=pypi_payload, read=requirements_at_ref) -> list[Finding]:
    findings: list[Finding] = []
    with tempfile.TemporaryDirectory(prefix="node-audit-") as scratch:
        workdir = Path(scratch)
        for name, repo, ref in node_packs(catalog):
            print(f"checking {name} at {ref[:12]}", flush=True)
            text = read(repo, ref, workdir)
            for line in text.splitlines():
                finding = classify_requirement(line, name, fetch)
                if finding is not None:
                    findings.append(finding)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--allowlist", type=Path, default=KNOWN_SOURCE_BUILDS)
    arguments = parser.parse_args()

    catalog = json.loads(arguments.catalog.read_text(encoding="utf-8"))
    findings = audit(catalog)
    allowed = read_known_source_builds(arguments.allowlist)

    for finding in findings:
        key = canonicalize_name(finding.package)
        if key in allowed:
            print(f"known, allowed: {finding}", flush=True)

    offenders = unlisted(findings, allowed)
    if not offenders:
        print("\nEvery node pack requirement installs as a wheel on the pod.")
        return 0

    print("\nThese requirements would build from source on a customer's pod:\n")
    for finding in offenders:
        print(f"  {finding}")
    print(
        f"\nAdd a name and a reason to {arguments.allowlist.name} if a source build is "
        f"genuinely intended - and say there how the pod avoids paying for it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
