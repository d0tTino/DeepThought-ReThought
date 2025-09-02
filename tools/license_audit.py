#!/usr/bin/env python3
"""Audit installed Python packages for approved licenses.

This script inspects the license metadata of all installed distributions and
compares them against a whitelist of approved, commercially friendly licenses.
If any installed dependency does not provide a license or the license is not in
that approved list, the script exits with a non-zero status code.
"""

from __future__ import annotations

import importlib.metadata
import sys

# A set of license identifiers considered acceptable for commercial use.
APPROVED_LICENSES = {
    "MIT",
    "BSD",
    "Apache",
    "Apache-2.0",
    "ISC",
    "MPL-2.0",
    "Python-2.0",
}


def extract_license(dist: importlib.metadata.Distribution) -> str | None:
    """Extract the license string from a distribution's metadata."""
    meta = dist.metadata
    license_str = meta.get("License")
    if not license_str:
        classifiers = meta.get_all("Classifier", [])
        for classifier in classifiers:
            if classifier.startswith("License ::"):
                license_str = classifier.split("::")[-1].strip()
                break
    return license_str


def main() -> int:
    unapproved: list[str] = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", dist.metadata.get("Summary", "unknown"))
        license_str = extract_license(dist)
        if not license_str or not any(
            approved.lower() in license_str.lower() for approved in APPROVED_LICENSES
        ):
            unapproved.append(f"{name}: {license_str or 'UNKNOWN'}")

    if unapproved:
        print("Found dependencies with unapproved or unknown licenses:")
        for item in sorted(unapproved):
            print(f"  - {item}")
        return 1

    print("All dependencies have approved licenses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
