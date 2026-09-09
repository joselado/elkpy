"""spec.py holds version-coupled data; this checks the version itself."""

import re
import struct

from elkpy import config, spec

FIXTURE = config.repo_root() / "tests" / "fixtures" / "h_sc"


def test_elk_version_matches_vendored_source():
    """spec.ELK_VERSION is the vendored release, so an Elk bump fails here
    rather than silently invalidating every task code in this file."""
    source = (config.repo_root() / "vendor" / "elk" / "src" / "modmain.f90").read_text()
    match = re.search(r"version\(3\)\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", source)
    assert match, "could not find `version(3)=[...]` in vendor/elk/src/modmain.f90"
    assert spec.ELK_VERSION == tuple(int(g) for g in match.groups())


def test_elk_version_matches_state_out_first_record():
    """STATE.OUT's first record is `version` (src/writestate.f90), which is
    what a binary-format reader asserts on.  Read it the way such a reader
    would: a gfortran sequential unformatted record is a 4-byte length, the
    payload, then the same 4-byte length again."""
    with open(FIXTURE / "STATE.OUT", "rb") as fh:
        head = fh.read(20)
    length, v0, v1, v2, tail = struct.unpack("<i3ii", head)
    assert length == tail == 12
    assert (v0, v1, v2) == spec.ELK_VERSION
