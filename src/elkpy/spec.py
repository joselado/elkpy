"""Version-coupled knowledge about this Elk release: task codes, xctype
codes, and output filenames (docs/design.md #7, docs/roadmap.md Tier 2).

Pure data, not logic -- an Elk version bump should mean editing this one
file. Every entry here has been cross-checked against vendor/elk/src/ (not
just the manual), see the comments citing the source file/subroutine.
"""

XC_CODES = {
    "PZ": 2,       # LDA, Perdew-Zunger/Ceperley-Alder
    "PW": 3,       # LSDA, Perdew-Wang/Ceperley-Alder (Elk's default)
    "Xalpha": 4,   # LDA, X-alpha
    "vBH": 5,      # LSDA, von Barth-Hedin
    "PBE": 20,     # GGA, Perdew-Burke-Ernzerhof
    "RPBE": 21,    # GGA, revised PBE
    "PBEsol": 22,  # GGA, PBEsol
    "WC06": 26,    # GGA, Wu-Cohen
    "AM05": 30,    # GGA, Armiento-Mattsson
}

# task numbers, per manual sec. 5.127 / src/elk.f90's task dispatch
TASKS = {
    "ground_state": 0,
    "ground_state_resume": 1,
    "relax": 2,
    "relax_resume": 3,
    "dos": 10,
    "bands": 20,
    "effective_mass": 25,
    "density_3d": 33,      # src/rhoplot.f90
    "potential_3d": 43,    # src/potplot.f90 (writes both VCL3D.OUT, VXC3D.OUT)
    "elf_3d": 53,          # src/elfplot.f90
    "momentum_matrix": 120,  # src/writepmat.f90 -- writes PMAT.OUT, a
                             # prerequisite of the optics tasks below
                             # (src/dielectric.f90 reads it via getpmat)
    "dielectric": 121,     # src/dielectric.f90 -- dielectric tensor, optical
                           # conductivity and plasma frequency; reads task
                           # 120's PMAT.OUT via getpmat
    "moke": 122,           # src/moke.f90 -- magneto-optic Kerr effect;
                           # calls dielectric internally (which is why it
                           # needs task 120's PMAT.OUT first)
    "phonon_dfpt": 205,    # src/dyntask.f90 -- single-shot, unlike the
                           # classical supercell method (task 200), which
                           # historically needs multiple coordinated runs;
                           # not wrapped here, reachable via run_tasks()
    "phonon_dos": 210,     # src/phdos.f90
    "phonon_dispersion": 220,  # src/phdisp.f90
    "berry_curvature": 9000,  # elkpy extension (not upstream Elk), src/elkpy_berry.f90 --
                              # patches/0002-berry-curvature-wilson-loop.patch; reserved
                              # high task number per docs/design.md #8
    "berry_curvature_path": 9001,  # elkpy extension: small Wilson loop at an arbitrary
                                    # list of k-points (pyqula-style), src/elkpy_berry.f90
    "eigenstate_session": 9002,  # elkpy extension: interactive eigenstate/overlap query
                                  # session (stdin/stdout loop), src/elkpy_eigenstates.f90 --
                                  # patches/0003-eigenstate-session.patch
}

# output filenames, per the `open(unit, file=...)` calls in the cited source
OUTPUT_FILES = {
    "state": "STATE.OUT",                  # src/writestate.f90
    "info": "INFO.OUT",                    # src/gndstate.f90
    "totenergy": "TOTENERGY.OUT",          # src/gndstate.f90
    "band": "BAND.OUT",                    # src/bandstr.f90
    "bandlines": "BANDLINES.OUT",          # src/bandstr.f90
    "tdos": "TDOS.OUT",                    # src/dos.f90
    "geometry_opt": "GEOMETRY_OPT.OUT",    # src/geomopt.f90
    "totenergy_opt": "TOTENERGY_OPT.OUT",  # src/geomopt.f90
    "effmass": "EFFMASS.OUT",              # src/effmass.f90
    "density_3d": "RHO3D.OUT",             # src/rhoplot.f90
    "potential_coulomb_3d": "VCL3D.OUT",   # src/potplot.f90 (task 43 writes
    "potential_xc_3d": "VXC3D.OUT",        # both of these in one run)
    "elf_3d": "ELF3D.OUT",                 # src/elfplot.f90
    "kerr": "KERR.OUT",                    # src/moke.f90
    "eigval": "EIGVAL.OUT",                # src/writeeval.f90
    "efermi": "EFERMI.OUT",                # src/writefermi.f90
    "phdos": "PHDOS.OUT",                  # src/phdos.f90
    "phdisp": "PHDISP.OUT",                # src/phdisp.f90
    "phdlines": "PHDLINES.OUT",            # src/phdisp.f90
    "berry": "ELKPY_BERRY.OUT",            # elkpy extension, src/elkpy_berry.f90
    "berry_path": "ELKPY_BERRY_PATH.OUT",  # elkpy extension, src/elkpy_berry.f90
}

# output filenames that carry indices in the name, as format templates --
# same "pure data" role as OUTPUT_FILES, but one file per requested tensor
# component rather than a single fixed name. Formatted with i=, j= (1, 2, 3
# for x, y, z), matching the `write(fname,'("EPSILON_",2I1,".OUT")') i,j`
# statements in the cited source.
OUTPUT_FILE_TEMPLATES = {
    "epsilon": "EPSILON_{i}{j}.OUT",  # src/dielectric.f90, dielectric tensor
    "sigma": "SIGMA_{i}{j}.OUT",      # src/dielectric.f90, optical conductivity
    "plasma": "PLASMA_{i}{j}.OUT",    # src/dielectric.f90, only when intraband=.true.
}
