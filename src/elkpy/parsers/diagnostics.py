"""Parsers for Elk's two developer/diagnostic outputs in this task family:
the RAM-disk report of task 68 (``rdstatus``, src/modramdisk.f90) and the
``TEST_nnn.OUT`` files that ``writetest`` (src/modtest.f90) drops when the
input variable ``test`` is true and that task 500 (``testcheck``) diffs.

Neither is a physical observable, which is why they are parsed here rather
than given a place among the response quantities:

- ``rdstatus`` writes to **standard output**, not to a file, so what is
  parsed is elkpy's captured ``elk.out`` log. Elk's RAM disk holds the
  eigenvector/eigenvalue records (EVECFV, EVECSV, EVALSV, OCCSV, ...) in
  memory instead of on disk when ``ramdisk=.true.`` (the default); the
  report says which records exist and how much memory they occupy, which
  is the practical way to see whether a large run will fit.
- ``writetest`` dumps the single characteristic array of a task -- the
  stress components for task 440, the EFG for 115, the structure factors
  for 195 -- in a fixed, machine-readable form with the tolerance the Elk
  developers consider meaningful for it. That tolerance is genuinely
  useful information: it is Elk's own statement of how reproducible the
  quantity is.
"""

import numpy as np

_TYPE_INTEGER, _TYPE_REAL, _TYPE_COMPLEX = 1, 2, 3


def parse_ramdisk_status(log_path):
    """Parse the ``Info(rdstatus):`` block out of a captured elk log
    (task 68, src/modramdisk.f90's ``rdstatus``).

    Written as::

        write(*,'("Info(rdstatus):")')
        write(*,'(" RAM disk not initialised")')          ! early return
        write(*,'(" Filename : ",A)') file(i)%fname
        write(*,'("  number of records : ",I0)') nr
        write(*,'("  total number of bytes : ",I0)') 4*m
        write(*,'(" Number of files on RAM disk : ",I0)') nf
        write(*,'(" Total number of bytes used by RAM disk : ",I0)') 4*n

    Returns {"initialised": bool, "files": [{"filename", "records",
    "bytes"}], "nfiles": int or None, "bytes": int or None}. An
    uninitialised RAM disk (``ramdisk=.false.``, or nothing cached yet)
    gives ``initialised=False`` and an empty file list rather than raising
    -- that is a legitimate state, not a parse failure.
    """
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        lines = [line.rstrip("\n") for line in fh]
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Info(rdstatus):"):
            start = i
    if start is None:
        raise ValueError(f"no 'Info(rdstatus):' block found in {log_path}")

    result = {"initialised": True, "files": [], "nfiles": None, "bytes": None}
    current = None
    for line in lines[start + 1 :]:
        s = line.strip()
        if s == "RAM disk not initialised":
            result["initialised"] = False
            break
        if s.startswith("Filename :"):
            current = {"filename": s.split(":", 1)[1].strip(), "records": None, "bytes": None}
            result["files"].append(current)
        elif s.startswith("number of records :") and current is not None:
            current["records"] = int(s.split(":", 1)[1])
        elif s.startswith("total number of bytes :") and current is not None:
            current["bytes"] = int(s.split(":", 1)[1])
        elif s.startswith("Number of files on RAM disk :"):
            result["nfiles"] = int(s.split(":", 1)[1])
        elif s.startswith("Total number of bytes used by RAM disk :"):
            result["bytes"] = int(s.split(":", 1)[1])
            break
    return result


def parse_test_file(test_out_path):
    """Parse a ``TEST_nnn.OUT`` file (src/modtest.f90's ``writetest``).

    Layout, with ``vt`` the variable type (1 integer, 2 real, 3 complex)
    and ``nv`` the number of values::

        write(90,'("''",A,"''")') trim(descr)
        write(90,'(2I8)') vt,nv
        write(90,'(G24.14)') tol           ! real/complex only
        write(90,'(2I8)') j,iva(j)                       ! vt = 1
        write(90,'(I8,G24.14)') j,rva(j)                 ! vt = 2
        write(90,'(I8,2G24.14)') j,dble(zva(j)),aimag(zva(j))  ! vt = 3

    Returns {"description": str, "type": int, "tolerance": float or None,
    "values": array (int, float or complex to match ``type``)}.
    """
    with open(test_out_path, encoding="utf-8", errors="replace") as fh:
        rows = [line.strip() for line in fh if line.strip()]
    description = rows[0].strip("'")
    vtype, nvalues = (int(x) for x in rows[1].split())
    if vtype == _TYPE_INTEGER:
        tolerance, data = None, rows[2 : 2 + nvalues]
    else:
        tolerance, data = float(rows[2]), rows[3 : 3 + nvalues]
    if len(data) != nvalues:
        raise ValueError(f"{test_out_path}: expected {nvalues} values, found {len(data)}")

    values = []
    for row in data:
        fields = row.split()
        if vtype == _TYPE_INTEGER:
            values.append(int(fields[1]))
        elif vtype == _TYPE_REAL:
            values.append(float(fields[1]))
        else:
            values.append(complex(float(fields[1]), float(fields[2])))
    dtype = {_TYPE_INTEGER: int, _TYPE_REAL: float, _TYPE_COMPLEX: complex}[vtype]
    return {
        "description": description,
        "type": vtype,
        "tolerance": tolerance,
        "values": np.array(values, dtype=dtype),
    }
