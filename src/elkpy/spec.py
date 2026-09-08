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
    "phonon_dfpt": 205,    # src/phonon.f90 -- single-shot, unlike the
                           # classical supercell method (task 200, wrapped as
                           # "phonons_supercell" below), which drives one
                           # displaced-supercell ground state per perturbation
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
    "stm": 162,            # src/wfplot.f90 -- upstream Elk's spin-SUMMED STM
                           # image (occupations replaced by a delta function at
                           # the Fermi energy, then the charge density plotted)
    "spin_stm_2d": 9003,   # elkpy extension: spin-POLARISED STM image on the
                           # plot2d plane, src/elkpy_stm.f90 --
                           # patches/0011-spin-polarized-stm.patch
    "spin_stm_3d": 9004,   # elkpy extension: same on the plot3d parallelepiped
    "transport": 9005,     # elkpy extension: vertical tunnelling transport
                           # (point tip -> substrate plane), the exit-plane
                           # Gram matrices and tip amplitudes,
                           # src/elkpy_transport.f90 --
                           # patches/0012-vertical-transport.patch

    "initial_state": 9006,  # elkpy extension: the ground state as init0/init1/
                           # rhoinit/potks leave it, i.e. at the TOP of Elk's
                           # first SCF iteration, with no STATE.OUT read and no
                           # density update -- the starting point the JAX port
                           # iterates from, src/elkpy_initstate.f90 --
                           # patches/0024-initial-state.patch

    # --- ground state, geometry, mechanical and electric response (docs/design.md #32)
    "hartree_fock": 5,  # src/hartfock.f90 -- Hartree-Fock / hybrid ground state
    "ramdisk_status": 68,  # src/modramdisk.f90 (rdstatus) -- RAM-disk report,
                           # STDOUT only
    "mossbauer": 110,  # src/mossbauer.f90 -- contact density + hyperfine field
    "efg": 115,  # src/writeefg.f90 -- electric field gradient; needs lmaxi >= 2
    "geometry_plot": 190,  # src/geomplot.f90 -- crystal.xsf + crystal.ascii
    "structure_factor_rho": 195,  # src/sfacrho.f90 -- X-ray structure factors
    "structure_factor_mag": 196,  # src/sfacmag.f90 -- magnetic structure
                                  # factors
    "rdmft": 300,  # src/rdmft.f90 -- reduced density matrix functional theory
    "piezoelectric": 380,  # src/piezoelt.f90 -- piezoelectric tensor
                           # dP/d(strain)
    "magnetoelectric": 390,  # src/magnetoelt.f90 -- magnetoelectric tensor
                             # dP_i/dB_j
    "tensor_moments": 400,  # src/writetm.f90 -> src/writetm3.f90 -- DFT+U
                            # tensor moments
    "molecular_dynamics": 420,  # src/moldyn.f90 -- Born-Oppenheimer molecular
                                # dynamics
    "molecular_dynamics_resume": 421,  # src/moldyn.f90 -- restarted from
                                       # TIMESTEP.OUT/ATDVC.OUT
    "strain": 430,  # src/writestrain.f90 -- symmetry-adapted strain-tensor
                    # basis
    "stress": 440,  # src/writestress.f90 -> src/genstress.f90 -- stress as
                    # dE/dt
    "test_check": 500,  # src/testcheck.f90 -- diff TEST_nnn.OUT against
                        # references

    # --- spectra, band-structure variants, Fermi surfaces, real-space plots
    "smearing_functions": 14,  # src/writesf.f90 -- smooth delta and Heaviside
    "lsj": 15,  # src/writelsj.f90 -- total muffin-tin L, S, J
    "lsj_states": 16,  # src/writelsj.f90 -- L, S, J per (k-point, state) in
                       # kstlist
    "band_character_l": 21,  # src/bandstr.f90 -- l-resolved atomic character
    "band_character_lm": 22,  # src/bandstr.f90 -- (l,m)-resolved character
    "band_character_spin": 23,  # src/bandstr.f90 -- spin character; needs
                                # spinpol
    "band_character_moment": 24,  # src/bandstr.f90 -- moment character; needs
                                  # spinpol
    "density_1d": 31,  # src/rhoplot.f90 -- charge density along a line
    "density_2d": 32,  # src/rhoplot.f90 -- charge density on a plane
    "potential_1d": 41,  # src/potplot.f90 -- writes both VCL1D.OUT and
                         # VXC1D.OUT
    "potential_2d": 42,  # src/potplot.f90 -- writes both VCL2D.OUT and
                         # VXC2D.OUT
    "elf_1d": 51,  # src/elfplot.f90 -- ELF along a line
    "elf_2d": 52,  # src/elfplot.f90 -- ELF on a plane
    "wavefunction_1d": 61,  # src/wfplot.f90 -- |psi|^2 of the kstlist state,
                            # line
    "wavefunction_2d": 62,  # src/wfplot.f90 -- |psi|^2 on a plane
    "wavefunction_3d": 63,  # src/wfplot.f90 -- |psi|^2 in a parallelepiped
    "core_wavefunctions": 65,  # src/wfcrplot.f90 -- radial core-state
                               # wavefunctions
    "magnetisation_1d": 71,  # src/vecplot.f90 -- m(r) along a line
    "magnetisation_2d": 72,  # src/vecplot.f90 -- m(r) on a plane, projected
    "magnetisation_3d": 73,  # src/vecplot.f90 -- m(r) in a parallelepiped
    "bxc_1d": 81,  # src/vecplot.f90 -- B_xc(r) along a line
    "bxc_2d": 82,  # src/vecplot.f90 -- B_xc(r) on a plane, projected
    "bxc_3d": 83,  # src/vecplot.f90 -- B_xc(r) in a parallelepiped
    "bxc_divergence_1d": 91,  # src/dbxcplot.f90 -- div B_xc along a line
    "bxc_divergence_2d": 92,  # src/dbxcplot.f90 -- div B_xc on a plane
    "bxc_divergence_3d": 93,  # src/dbxcplot.f90 -- div B_xc in a parallelepiped
    "fermi_surface_product": 100,  # src/fermisurf.f90 -- prod_n (e_n - E_F)
    "fermi_surface_bands": 101,  # src/fermisurf.f90 -- e_n - E_F per crossing
                                 # band
    "fermi_surface_bxsf": 102,  # src/fermisurfbxsf.f90 -- XCrySDen .bxsf band
                                # grid
    "fermi_surface_delta": 103,  # src/fermisurf.f90 -- summed smeared delta at
                                 # E_F
    "fermi_surface_band_deltas": 104,  # src/fermisurf.f90 -- smeared delta per
                                       # band
    "nesting": 105,  # src/nesting.f90 -- Fermi-surface nesting function N(q)
    "elnes": 140,  # src/elnes.f90 -- electron energy loss near-edge structure
    "electric_field_1d": 141,  # src/vecplot.f90 -- E = -grad v_C along a line
    "electric_field_2d": 142,  # src/vecplot.f90 -- E on a plane, projected
    "electric_field_3d": 143,  # src/vecplot.f90 -- E in a parallelepiped
    "atomic_eigenvalues": 150,  # src/writeevsp.f90 -- free-atom Dirac
                                # eigenvalues
    "magnetic_torque_1d": 151,  # src/vecplot.f90 -- m x B_xc along a line;
                                # ndmag == 3
    "magnetic_torque_2d": 152,  # src/vecplot.f90 -- m x B_xc on a plane,
                                # projected
    "magnetic_torque_3d": 153,  # src/vecplot.f90 -- m x B_xc in a
                                # parallelepiped
    "momentum_density": 170,  # src/writeemd.f90 -- EMD on the H+k grid,
                              # unformatted
    "emd_1d": 171,  # src/emdplot.f90 -- the Compton profile
    "emd_2d": 172,  # src/emdplot.f90 -- EMD integrated along the plane normal
    "emd_3d": 173,  # src/emdplot.f90 -- EMD itself, no integration
    "wxc_1d": 341,  # src/wxcplot.f90 -- meta-GGA W_xc, line; needs a libxc
                    # build
    "wxc_2d": 342,  # src/wxcplot.f90 -- meta-GGA W_xc on a plane
    "wxc_3d": 343,  # src/wxcplot.f90 -- meta-GGA W_xc in a parallelepiped
    "paramagnetic_current_1d": 371,  # src/jprplot.f90 -- j_p along a line
    "paramagnetic_current_2d": 372,  # src/jprplot.f90 -- j_p on a plane,
                                     # projected
    "paramagnetic_current_3d": 373,  # src/jprplot.f90 -- j_p in a
                                     # parallelepiped
    "static_density_1d": 471,  # src/rhosplot.f90 -- static (screened) density,
                               # 1D only
    "nonlinear_optics": 125,  # src/nonlinopt.f90 -- chi(-2w;w,w), second
                              # harmonic
    "expmat": 130,  # src/writeexpmat.f90 -- <i,k+q|exp(iq.r)|j,k>

    # --- optical and dielectric response, TDDFT, Bethe-Salpeter
    "wfpw": 135,  # src/writewfpw.f90 -- wavefunctions in a plane-wave basis
    "epsinv": 180,  # src/writeepsinv.f90 -> src/epsinv.f90 -- static RPA eps^-1
    "bse_hamiltonian": 185,  # src/writehmlbse.f90 -- Bethe-Salpeter Hamiltonian
    "bse_eigenvectors": 186,  # src/writeevbse.f90 -- diagonalises HMLBSE.OUT
    "bse_dielectric": 187,  # src/dielectric_bse.f90 -- excitonic dielectric
                            # function
    "anomalous_entropy": 285,  # src/aceplot.f90 -- anomalous correlation
                               # entropy
    "tddft_linear_response": 320,  # src/tddftlr.f90 -- linear-response TDDFT
                                   # with local fields
    "tddft_spin_response": 330,  # src/tddftsplr.f90 -- chi_ij(q,w), i,j = 0..3;
                                 # needs spinpol
    "tddft_spin_response_full": 331,  # src/tddftsplr.f90 -- same, dumping
                                      # chi(G,G')
    "afield": 450,  # src/genafieldt.f90 -- laser vector potential A(t)
    "afield_power": 455,  # src/writeafpdt.f90 -- power density of A(t)
    "efield_fourier": 456,  # src/writeefieldw.f90 -- Fourier transform of E(t)
    "tddft": 460,  # src/tddft.f90 -- real-time TDDFT evolution from t = 0
    "tddft_restart": 461,  # src/tddft.f90 -- restart from the last time step
    "tddft_ehrenfest": 462,  # src/tddft.f90 -- real-time with Ehrenfest nuclear
                             # dynamics
    "tddft_ehrenfest_restart": 463,  # src/tddft.f90 -- Ehrenfest restart
    "dielectric_tdrt": 480,  # src/dielectric_tdrt.f90 -- dielectric tensor from
                             # J(t)
    "dielectric_tdrt_delta": 481,  # src/dielectric_tdrt.f90 -- same, delta-kick
                                   # E(t)
    "phonons_supercell": 200,  # src/phononsc.f90 -- dynamical matrices by
                               # finite displacements
    "phonons_supercell_resume": 201,  # src/phononsc.f90 -- resumes from a
                                      # written STATE file

    # --- phonons, electron-phonon coupling, superconductivity
    "phonons_supercell_dryrun": 202,  # src/phononsc.f90 -- creates empty DYN
                                      # files and stops
    "born_charges": 208,  # src/bornechg.f90 -- static Born effective charges
    "born_charges_dryrun": 209,  # src/bornechg.f90 -- empty BEC files only
    "phonon_modes": 230,  # src/writephn.f90 -- frequencies AND eigenvectors at
                          # any q
    "ephcouple": 240,  # src/ephcouple.f90 -- electron-phonon coupling near E_F
    "ephcouple_write": 241,  # src/ephcouple.f90 -- full window, writes
                             # EPHMAT.OUT
    "phonon_linewidths": 245,  # src/phlwidth.f90 -- phonon linewidths along a
                               # q-path
    "alpha2f": 250,  # src/alpha2f.f90 -- Eliashberg spectral function + Allen-
                     # Dynes T_c
    "eliashberg": 260,  # src/eliashberg.f90 -- isotropic Eliashberg equations
    "gndsteph": 270,  # src/gndsteph.f90 -- coupled electron-phonon Bogoliubov
                      # ground state
    "gndsteph_resume": 271,  # src/gndsteph.f90 -- restart from
                             # EVALUV.OUT/EVECUV.OUT
    "ephdos": 280,  # src/ephdos.f90 -- renormalised DOS of the Bogoliubov
                    # system
    "born_charges_dynamical": 478,  # src/bornecdyn.f90 -- frequency-dependent
                                    # Born charges
    "mae": 28,  # src/mae.f90 -- magnetic anisotropy energy, from atomic
                # densities
    "mae_resume": 29,  # src/mae.f90 -- same, reading STATE.OUT

    # --- magnetism, GW, Wannier90 export, ultra-long-range
    "torque": 160,  # src/torque.f90 -- integral of m(r) x B_xc(r); no output
                    # file
    "spin_spiral_supercell": 350,  # src/spiralsc.f90 -- spin spirals by
                                   # explicit supercell
    "spin_spiral_supercell_resume": 351,  # src/spiralsc.f90 -- reads each q's
                                          # own STATE_Q*.OUT
    "spin_spiral_supercell_dryrun": 352,  # src/spiralsc.f90 -- empty SS_Q*.OUT
                                          # placeholders
    "wannier90": 550,  # src/writew90.f90 -- Wannier90 export; needs libwannier
                       # for .amn/.mmn
    "gw_self_energy": 600,  # src/gwsefm.f90 -- G0W0 self-energy on the
                            # Matsubara axis
    "gw_self_energy_keep_epsinv": 601,  # src/gwsefm.f90 -- same, reusing an
                                        # existing EPSINV.OUT
    "gw_spectral_function": 610,  # src/gwspecf.f90 -- real-axis spectral
                                  # function
    "gw_band_structure": 620,  # src/gwbandstr.f90 -- GW spectral-function band
                               # structure
    "gw_fermi_energy": 630,  # src/writegwefm.f90 -- GW Fermi energy; needs
                             # GWSEFM.OUT
    "gw_density_matrix": 640,  # src/gwdmat.f90 -- GW density matrix; overwrites
                               # EVECSV/OCCSV
    "ulr_ground_state": 700,  # src/gndstulr.f90 -- ultra-long-range ground
                              # state
    "ulr_ground_state_resume": 701,  # src/gndstulr.f90 -- restarting from
                                     # STATE_ULR.OUT
    "ulr_dos": 710,  # src/writedosu.f90 -- ULR total DOS, per unit cell
    "ulr_bands": 720,  # src/bandstrulr.f90 -- ULR band structure at kappa = 0
    "ulr_bands_all_kappa": 725,  # src/bandstrulr.f90 -- ULR bands over every
                                 # kappa, appended
    "ulr_density_1d": 731,  # src/rhouplot.f90 -- 1D ultracell charge density
    "ulr_density_2d": 732,  # src/rhouplot.f90 -- 2D ultracell charge density
    "ulr_density_3d": 733,  # src/rhouplot.f90 -- 3D ultracell charge density
    "ulr_potential_1d": 741,  # src/potuplot.f90 -- 1D ultracell Kohn-Sham
                              # potential
    "ulr_potential_2d": 742,  # src/potuplot.f90 -- 2D ultracell Kohn-Sham
                              # potential
    "ulr_potential_3d": 743,  # src/potuplot.f90 -- 3D ultracell Kohn-Sham
                              # potential
    "ulr_magnetisation_1d": 771,  # src/maguplot.f90 -- 1D ultracell
                                  # magnetisation
    "ulr_magnetisation_2d": 772,  # src/maguplot.f90 -- 2D ultracell
                                  # magnetisation, projected
    "ulr_magnetisation_3d": 773,  # src/maguplot.f90 -- 3D ultracell
                                  # magnetisation
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
    "stm_2d": "STM2D.OUT",                 # src/wfplot.f90, upstream task 162
    "spin_stm_2d": "ELKPY_STM2D.OUT",      # elkpy extension, src/elkpy_stm.f90
    "spin_stm_3d": "ELKPY_STM3D.OUT",      # elkpy extension, src/elkpy_stm.f90
    "spin_stm_dos": "ELKPY_STMDOS.OUT",    # elkpy extension, src/elkpy_stm.f90 --
                                           # cell integrals of the plotted fields
    "transport": "ELKPY_TRANSPORT.OUT",    # elkpy extension,
                                           # src/elkpy_transport.f90
    "fermidos": "FERMIDOS.OUT",            # src/occupy.f90, DOS at the Fermi
                                           # energy per s.c. iteration

    # --- ground state, geometry, mechanical and electric response
    "strain": "STRAIN.OUT",  # src/writestrain.f90
    "stress": "STRESS.OUT",  # src/writestress.f90 (non-ASCII header; open as
                             # UTF-8)
    "piezoelectric": "PIEZOELT.OUT",  # src/piezoelt.f90 (non-ASCII header)
    "magnetoelectric": "MAGNETOELT.OUT",  # src/magnetoelt.f90 (non-ASCII
                                          # header)
    "efg": "EFG.OUT",  # src/writeefg.f90
    "mossbauer": "MOSSBAUER.OUT",  # src/mossbauer.f90
    "structure_factor_rho": "SFACRHO.OUT",  # src/sfacrho.f90
    "tensor_moments": "TENSMOM.OUT",  # src/writetm3.f90 (non-ASCII header for
                                      # tm3type=0)
    "geometry_xsf": "crystal.xsf",  # src/geomplot.f90 -- XCrySDen, ANGSTROM
    "geometry_ascii": "crystal.ascii",  # src/geomplot.f90 -- V_Sim, BOHR
    "geometry_axsf": "crystal.axsf",  # src/writeaxsf.f90 -- animated XSF,
                                      # Angstrom
    "hf_info": "HF_INFO.OUT",  # src/hartfock.f90 -- HF energy decomposition per
                               # iteration
    "band_gap": "GAP.OUT",  # src/hartfock.f90 -- indirect band gap per HF
                            # iteration
    "dtotenergy": "DTOTENERGY.OUT",  # src/hartfock.f90 -- |dE| per iteration
    "moment": "MOMENT.OUT",  # src/hartfock.f90 (spinpol) -- total spin moment
                             # per iteration
    "moment_magnitude": "MOMENTM.OUT",  # src/hartfock.f90 (spinpol) -- its
                                        # magnitude
    "rdm_info": "RDM_INFO.OUT",  # src/rdmft.f90 -- decomposition and occupation
                                 # numbers
    "rdm_energy": "RDM_ENERGY.OUT",  # src/rdmft.f90 -- total energy per outer
                                     # loop
    "md_forcetot": "FORCETOT_TD.OUT",  # src/writetdforces.f90 -- per-step force
                                       # block
    "md_forcemax": "FORCEMAX_TD.OUT",  # src/writetdforces.f90 -- (time,
                                       # forcemax)
    "md_displacement_lattice": "ATDISPL_TD.OUT",  # src/writeatdisp.f90 -- lattice
                                                  # coordinates
    "md_displacement_cartesian": "ATDISPC_TD.OUT",  # src/writeatdisp.f90 -- Cartesian
                                                    # Bohr
    "md_moment_mt": "MOMENTMT_TD.OUT",  # src/writemomtd.f90 -- BARE time
                                        # header, ndmag columns
    "md_moment_ir": "MOMENTIR_TD.OUT",  # src/writemomtd.f90 -- (time,
                                        # momir(1:ndmag))
    "md_timestep": "TIMESTEP.OUT",  # src/writetimes.f90 -- read back by task
                                    # 421
    "md_restart": "ATDVC.OUT",  # src/writeatdvc.f90 -- NOT written on the final
                                # force step

    # --- spectra and the real-space plotting triples
    "density_1d": "RHO1D.OUT",  # src/rhoplot.f90 task 31
    "density_2d": "RHO2D.OUT",  # src/rhoplot.f90 task 32
    "density_lines": "RHOLINES.OUT",  # src/rhoplot.f90 -- plot1d vertex lines
    "potential_coulomb_1d": "VCL1D.OUT",  # src/potplot.f90 task 41
    "potential_xc_1d": "VXC1D.OUT",  # src/potplot.f90 task 41 (same run)
    "potential_coulomb_2d": "VCL2D.OUT",  # src/potplot.f90 task 42
    "potential_xc_2d": "VXC2D.OUT",  # src/potplot.f90 task 42 (same run)
    "potential_lines": "VLINES.OUT",  # src/potplot.f90 -- opened twice, second
                                      # open truncates
    "elf_1d": "ELF1D.OUT",  # src/elfplot.f90 task 51
    "elf_2d": "ELF2D.OUT",  # src/elfplot.f90 task 52
    "elf_lines": "ELFLINES.OUT",  # src/elfplot.f90 task 51
    "wavefunction_1d": "WF1D.OUT",  # src/wfplot.f90 task 61
    "wavefunction_2d": "WF2D.OUT",  # src/wfplot.f90 task 62
    "wavefunction_3d": "WF3D.OUT",  # src/wfplot.f90 task 63
    "wavefunction_lines": "WFLINES.OUT",  # src/wfplot.f90 task 61
    "magnetisation_1d": "MAG1D.OUT",  # src/vecplot.f90 task 71
    "magnetisation_2d": "MAG2D.OUT",  # src/vecplot.f90 task 72
    "magnetisation_3d": "MAG3D.OUT",  # src/vecplot.f90 task 73
    "magnetisation_lines": "MAGLINES.OUT",  # src/vecplot.f90 task 71
    "bxc_1d": "BXC1D.OUT",  # src/vecplot.f90 task 81
    "bxc_2d": "BXC2D.OUT",  # src/vecplot.f90 task 82
    "bxc_3d": "BXC3D.OUT",  # src/vecplot.f90 task 83
    "bxc_lines": "BXCLINES.OUT",  # src/vecplot.f90 task 81
    "electric_field_1d": "EF1D.OUT",  # src/vecplot.f90 task 141
    "electric_field_2d": "EF2D.OUT",  # src/vecplot.f90 task 142
    "electric_field_3d": "EF3D.OUT",  # src/vecplot.f90 task 143
    "electric_field_lines": "EFLINES.OUT",  # src/vecplot.f90 task 141
    "magnetic_torque_1d": "MCBXC1D.OUT",  # src/vecplot.f90 task 151, m(r) x
                                          # B_xc(r)
    "magnetic_torque_2d": "MCBXC2D.OUT",  # src/vecplot.f90 task 152
    "magnetic_torque_3d": "MCBXC3D.OUT",  # src/vecplot.f90 task 153
    "magnetic_torque_lines": "MCBXCLINES.OUT",  # src/vecplot.f90 task 151
    "bxc_divergence_1d": "DBXC1D.OUT",  # src/dbxcplot.f90 task 91
    "bxc_divergence_2d": "DBXC2D.OUT",  # src/dbxcplot.f90 task 92
    "bxc_divergence_3d": "DBXC3D.OUT",  # src/dbxcplot.f90 task 93
    "bxc_divergence_lines": "DBXCLINES.OUT",  # src/dbxcplot.f90 task 91
    "wxc_1d": "WXC1D.OUT",  # src/wxcplot.f90 task 341
    "wxc_2d": "WXC2D.OUT",  # src/wxcplot.f90 task 342
    "wxc_3d": "WXC3D.OUT",  # src/wxcplot.f90 task 343
    "wxc_lines": "WLINES.OUT",  # src/wxcplot.f90 task 341 (WLINES, not
                                # WXCLINES)
    "paramagnetic_current_1d": "JPR1D.OUT",  # src/jprplot.f90 task 371
    "paramagnetic_current_2d": "JPR2D.OUT",  # src/jprplot.f90 task 372
    "paramagnetic_current_3d": "JPR3D.OUT",  # src/jprplot.f90 task 373
    "paramagnetic_current_lines": "JPRLINES.OUT",  # src/jprplot.f90 task 371
    "static_density_1d": "RHOS1D.OUT",  # src/rhosplot.f90 task 471 -- 3 A-field
                                        # directions
    "static_density_lines": "RHOSLINES.OUT",  # src/rhosplot.f90 task 471
    "fermisurf": "FERMISURF.OUT",  # src/fermisurf.f90 tasks 100/101/103/104
    "fermisurf_up": "FERMISURF_UP.OUT",  # src/fermisurf.f90, written when ndmag
                                         # == 1
    "fermisurf_dn": "FERMISURF_DN.OUT",  # src/fermisurf.f90, ditto
    "fermisurf_bxsf": "FERMISURF.bxsf",  # src/fermisurfbxsf.f90 task 102
                                         # (lowercase ext)
    "fermisurf_bxsf_up": "FERMISURF_UP.bxsf",  # src/fermisurfbxsf.f90, ndmag == 1
    "fermisurf_bxsf_dn": "FERMISURF_DN.bxsf",  # src/fermisurfbxsf.f90, ndmag == 1
    "nest_3d": "NEST3D.OUT",  # src/nesting.f90 task 105, N(q) on the q-mesh
    "nesting": "NESTING.OUT",  # src/nesting.f90 task 105, the BZ integral of
                               # N(q)
    "idos": "IDOS.OUT",  # src/dos.f90 task 10 -- the INTERSTITIAL remainder,
                         # not a total
    "elmirep": "ELMIREP.OUT",  # src/writeelmirep.f90, from task 22 or 10 when
                               # lmirep
    "lsj": "LSJ.OUT",  # src/writelsj.f90 task 15
    "lsj_kst": "LSJ_KST.OUT",  # src/writelsj.f90 task 16
    "evalsp": "EVALSP.OUT",  # src/writeevsp.f90 task 150
    "sdelta": "SDELTA.OUT",  # src/writesf.f90 task 14, the smooth Dirac delta
    "stheta": "STHETA.OUT",  # src/writesf.f90 task 14, the smooth Heaviside
                             # step
    "elnes": "ELNES.OUT",  # src/elnes.f90 task 140, double differential cross-
                           # section
    "emd": "EMD.OUT",  # src/writeemd.f90 task 170 -- UNFORMATTED direct-access
    "emd_1d": "EMD1D.OUT",  # src/emdplot1d.f90 task 171
    "emd_2d": "EMD2D.OUT",  # src/emdplot2d.f90 task 172
    "emd_3d": "EMD3D.OUT",  # src/emdplot3d.f90 task 173
    "epsinv": "EPSINV.OUT",  # src/putepsinv.f90 -- UNFORMATTED/DIRECT, one
                             # record per q; written by task 180 or 600, reused
                             # by 601 and the BSE direct term
    "hmlbse": "HMLBSE.OUT",  # src/writehmlbse.f90 task 185 -- opaque
                             # intermediate
    "evbse": "EVBSE.OUT",  # src/writeevbse.f90 task 186 -- opaque intermediate
    "eigval_bse": "EIGVAL_BSE.OUT",  # src/writeevbse.f90 task 186 -- exciton
                                     # energies

    # --- optical and dielectric response, TDDFT, Bethe-Salpeter
    "expiqr": "EXPIQR.OUT",  # src/writeexpmat.f90 task 130 -- UTF-8 header
    "wfpw": "WFPW.OUT",  # src/writewfpw.f90 task 135 -- UNFORMATTED/DIRECT per
                         # k-point
    "chi_transverse": "CHI_T.OUT",  # src/tddftsplr.f90 -- collinear magnets
                                    # only
    "chi0_transverse": "CHI0_T.OUT",  # src/tddftsplr.f90 -- Kohn-Sham,
                                      # collinear only
    "chi_full": "CHI.OUT",  # src/tddftsplr.f90 task 331 -- raw UNFORMATTED
                            # chi(G,G')
    "epsilon_tddft_q": "EPSILON_TDDFT.OUT",  # src/tddftlr.f90 at FINITE q --
                                             # unindexed
    "epsinv_tddft_q": "EPSINV_TDDFT.OUT",  # src/tddftlr.f90 at finite q --
                                           # unindexed
    "faraday": "FARADAY.OUT",  # src/tddftlr.f90 at q=0 -- Faraday rotation,
                               # radians
    "kerr_tddft": "KERR_TDDFT.OUT",  # src/tddftlr.f90 at q=0 -- Kerr angle in
                                     # DEGREES
    "mld": "MLD.OUT",  # src/tddftlr.f90 at q=0 -- magnetic linear dichroism
    "afieldt": "AFIELDT.OUT",  # src/genafieldt.f90 task 450 -- A(t), read back
                               # by 455-481
    "afspt": "AFSPT.OUT",  # src/genafieldt.f90 -- spin-dependent A(t), only
                           # with tafspt
    "td_info": "TD_INFO.OUT",  # src/genafieldt.f90 -- pulse summary, peak
                               # W/cm^2
    "afpdt": "AFPDT.OUT",  # src/writeafpdt.f90 task 455 -- laser power density
                           # vs time
    "afted": "AFTED.OUT",  # src/writeafpdt.f90 task 455 -- A-field energy
                           # density
    "efieldw": "EFIELDW.OUT",  # src/writeefieldw.f90 task 456 -- E(w), three
                               # blocks
    "jtot_td": "JTOT_TD.OUT",  # src/writetddft.f90 -- total current; read by
                               # src/readjtot.f90
    "jtotm_td": "JTOTM_TD.OUT",  # src/writetddft.f90 -- its magnitude
    "totenergy_td": "TOTENERGY_TD.OUT",  # src/writetdengy.f90 -- total energy
                                         # per step; shared by molecular
                                         # dynamics (420/421) and real-time
                                         # TDDFT (460-463)
    "chargeir_td": "CHARGEIR_TD.OUT",  # src/writetddft.f90 -- interstitial
                                       # charge per step
    "chargemt_td": "CHARGEMT_TD.OUT",  # src/writetddft.f90 -- per-atom muffin-
                                       # tin charge
    "moment_td": "MOMENT_TD.OUT",  # src/writemomtd.f90 -- total spin moment,
                                   # ndmag columns; shared by tasks 420/421 and
                                   # 460-463
    "momentm_td": "MOMENTM_TD.OUT",  # src/writemomtd.f90 -- its magnitude; same
                                     # sharing
    "afind_td": "AFIND_TD.OUT",  # src/writetddft.f90 -- induced A-field, only
                                 # with tafindt
    "jtotw": "JTOTW.OUT",  # src/dielectric_tdrt.f90 -- J(w), same shape as
                           # EFIELDW.OUT
    "face_3d": "FACE3D.OUT",  # src/aceplot.f90 task 285 -- fermionic ACE on the
                              # k-mesh
    "bace_3d": "BACE3D.OUT",  # src/aceplot.f90 task 285 -- bosonic ACE on the
                              # q-mesh
    "phonon_modes": "PHONON.OUT",  # src/writephn.f90 -- frequencies and
                                   # eigenvectors per q
    "gammaq": "GAMMAQ.OUT",  # src/writegamma.f90 -- phonon linewidths on the
                             # q-mesh
    "lambdaq": "LAMBDAQ.OUT",  # src/writelambda.f90 -- mode couplings;
                               # GAMMAQ.OUT layout
    "phlwidth": "PHLWIDTH.OUT",  # src/phlwidth.f90 -- shares BAND.OUT's block
                                 # layout

    # --- phonons, electron-phonon coupling, superconductivity
    "phlwlines": "PHLWLINES.OUT",  # src/phlwidth.f90 -- vertex lines for
                                   # PHLWIDTH.OUT
    "alpha2f": "ALPHA2F.OUT",  # src/alpha2f.f90 -- on its own grid, NOT the
                               # wplot range
    "mcmillan": "MCMILLAN.OUT",  # src/alpha2f.f90 -- lambda, w_log, w_rms, mu*,
                                 # T_c (UTF-8)
    "eliashberg_info": "ELIASHBERG.OUT",  # src/eliashberg.f90 -- run log
    "eliashberg_ia": "ELIASHBERG_IA.OUT",  # src/eliashberg.f90 -- ragged block
                                           # per temperature
    "eliashberg_gap_t": "ELIASHBERG_GAP_T.OUT",  # src/eliashberg.f90 -- T (K),
                                                 # Delta, Z
    "eliashberg_gap_ra": "ELIASHBERG_GAP_RA.OUT",  # src/eliashberg.f90 -- Pade of
                                                   # Delta
    "eliashberg_z_ra": "ELIASHBERG_Z_RA.OUT",  # src/eliashberg.f90 -- Pade of Z
    "tdos_eph": "TDOS_EPH.OUT",  # src/ephdos.f90 -- renormalised DOS, TDOS.OUT
                                 # layout
    "faceeh": "FACEEH.OUT",  # src/ephdos.f90 -- ACE vs energy; sparse, below
                             # 1e-4 omitted
    "ephgap": "EPHGAP.OUT",  # src/gndsteph.f90 -- indirect band gap per
                             # iteration
    "eph_info": "EPH_INFO.OUT",  # src/gndsteph.f90 -- self-consistent loop log
    "mae": "MAE.OUT",  # src/mae.f90 -- the anisotropy energy, Hartree
    "mae_per_volume": "MAEPUV.OUT",  # src/mae.f90 -- MAE per unit volume,
                                     # Ha/Bohr^3
    "mae_info": "MAE_INFO.OUT",  # src/mae.f90 -- per-direction record (energies
                                 # at G24.14)
    "gw_self_energy": "GWSEFM.OUT",  # src/putgwsefm.f90 -- UNFORMATTED/DIRECT
                                     # Matsubara Sigma

    # --- magnetism, GW, ultra-long-range
    "gw_total_spectral_function": "GWTSF.OUT",  # src/gwspecf.f90 -- zone-summed
                                                # A(w)
    "gw_band": "GWBAND.OUT",  # src/gwbandstr.f90 -- header is (nwplot, npp1d),
                              # data reshapes (npp1d, nwplot)
    "gw_fermi_energy": "GWEFERMI.OUT",  # src/writegwefm.f90 -- interacting
                                        # Fermi energy
    "ulr_state": "STATE_ULR.OUT",  # src/writestulr.f90 -- ultracell
                                   # density/potential
    "ulr_info": "ULR_INFO.OUT",  # src/gndstulr.f90 -- the ULR loop's own log
    "ulr_rmsdvs": "RMSDVS.OUT",  # src/gndstulr.f90 -- RMS change in the ULR
                                 # potential
    "ulr_tdos": "TDOSULR.OUT",  # src/writedosu.f90 -- per UNIT cell, E_F
                                # subtracted
    "ulr_band": "BANDULR.OUT",  # src/bandstrulr.f90 -- THREE columns (kappa
                                # character)
    "ulr_band_spectral": "BANDSFU.OUT",  # src/bandstrulr.f90 -- header (nkpt0,
                                         # nwplot), data reshapes (nwplot,
                                         # nkpt0)
    "ulr_density_1d": "RHOU1D.OUT",  # src/rhouplot.f90 via src/plotu1d.f90 --
                                     # no header line
    "ulr_density_lines": "RHOULINES.OUT",  # src/plotu1d.f90 -- vertex gridlines
    "ulr_density_2d": "RHOU2D.OUT",  # src/rhouplot.f90 via src/plotu2d.f90 --
                                     # plot2d layout
    "ulr_density_3d": "RHOU3D.OUT",  # src/rhouplot.f90 via src/plotu3d.f90 --
                                     # plot3d layout
    "ulr_potential_1d": "VSU1D.OUT",  # src/potuplot.f90 via src/plotu1d.f90
    "ulr_potential_lines": "VSULINES.OUT",  # src/potuplot.f90 -- vertex
                                            # gridlines
    "ulr_potential_2d": "VSU2D.OUT",  # src/potuplot.f90 via src/plotu2d.f90
    "ulr_potential_3d": "VSU3D.OUT",  # src/potuplot.f90 via src/plotu3d.f90
    "ulr_magnetisation_1d": "MAGU1D.OUT",  # src/maguplot.f90 via
                                           # src/plotu1d.f90 -- ndmag columns
    "ulr_magnetisation_lines": "MAGULINES.OUT",  # src/maguplot.f90 -- vertex
                                                 # gridlines
    "ulr_magnetisation_2d": "MAGU2D.OUT",  # src/maguplot.f90 via
                                           # src/plotu2d.f90 -- projected
    "ulr_magnetisation_3d": "MAGU3D.OUT",  # src/maguplot.f90 via
                                           # src/plotu3d.f90
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

    "structure_factor_mag": "SFACMAG_{j}.OUT",  # src/sfacmag.f90 -- one per
                                                # magnetisation component j =
                                                # 1..ndmag
    "test": "TEST_{id}.OUT",  # src/modtest.f90 -- id zero-padded to 3 digits;
                              # task 500's references carry a trailing
                              # underscore
    "band_character": "BAND_S{species:02d}_A{atom:04d}.OUT",  # src/bandstr.f90 tasks 21-24, one
                                                              # file per atom
    "partial_dos": "PDOS_S{species:02d}_A{atom:04d}.OUT",  # src/dos.f90 task 10, one file per
                                                           # atom
    "core_wavefunction": "WFCORE_S{species:02d}_A{atom:04d}.OUT",  # src/wfcrplot.f90 task 65
    "shg_chi": "CHI_2WWW_{a}{b}{c}.OUT",  # src/nonlinopt.f90 -- THREE Cartesian
                                          # indices, 1-based
    "shg_chi_ii": "CHI_II_2WWW_{a}{b}{c}.OUT",  # src/nonlinopt.f90 -- interband
                                                # contribution
    "shg_eta_ii": "ETA_II_2WWW_{a}{b}{c}.OUT",  # src/nonlinopt.f90 -- intraband
                                                # modulation
    "shg_sigma_ii": "SIGMA_II_2WWW_{a}{b}{c}.OUT",  # src/nonlinopt.f90 -- the three sum
                                                    # to CHI_2WWW
    "epsilon_bse": "EPSILON_BSE_{i}{j}.OUT",  # src/dielectric_bse.f90 task 187,
                                              # 1-based i,j
    "epsilon_tddft": "EPSILON_TDDFT_{i}{j}.OUT",  # src/tddftlr.f90 at q=0, 1-based
                                                  # i,j
    "epsinv_tddft": "EPSINV_TDDFT_{i}{j}.OUT",  # src/tddftlr.f90 at q=0, 1-based
                                                # i,j
    "epsm_tddft": "EPSM_TDDFT_{i}{j}.OUT",  # src/tddftlr.f90 at q=0 --
                                            # MACROSCOPIC part
    "chi_spin": "CHI_{i}{j}.OUT",  # src/tddftsplr.f90 -- i,j run 0..3 (0 =
                                   # charge), unlike every other template here
    "chi0_spin": "CHI0_{i}{j}.OUT",  # src/tddftsplr.f90 -- Kohn-Sham, same 0..3
                                     # indexing
    "epsilon_tdrt": "EPSILON_TDRT_{i}{j}.OUT",  # src/dielectric_tdrt.f90 -- ALL
                                                # NINE always written; optcomp never
                                                # consulted
    "born_charge": "BEC_S{s:02d}_A{a:03d}_P{p:d}.OUT",  # src/becfext.f90 + src/bectask.f90
                                                        # -- task 208 writes 3 scalar lines,
                                                        # task 478 six Re/Im blocks into the
                                                        # SAME name
    "spin_spiral": "SS_Q{m1:02d}{n1:02d}_{m2:02d}{n2:02d}_{m3:02d}{n3:02d}.OUT",  # src/ssfext.f90 + src/sstask.f90 --
                                                                                  # m/n reduced by gcd; a zero
                                                                                  # component gives 0000. Use parsers.
                                                                                  # magnetism.spin_spiral_filename
    "gw_spectral_function": "GWSF_K{ik:06d}.OUT",  # src/writegwsf.f90 -- one per
                                                   # reduced k-point; parse only up to
                                                   # the first blank line
}
