
from typing import Any

from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.flows.phonons import PhononMaker
from jobflow import Flow
from pymatgen.core import Structure

from glassbench.mlip_makers import MLIPMDMaker, MLIPRelaxMaker, MLIPStaticMaker


def make_phonon_mlip(structure: Structure, potential_kwargs: dict | None = None,
                     force_tol: float = 0.02, *, calculator: Any = None) -> Flow:
    relax_kwargs = {"verbose": False, "fmax": force_tol, "final_atoms_object_file": None}
    maker_kw = {"potential_kwargs": potential_kwargs, "calculator_instance": calculator}
    static_energy_maker = MLIPStaticMaker(name="Phonon static", **maker_kw)
    bulk_relax_maker = MLIPRelaxMaker(name="Phonon relax", relax_kwargs=relax_kwargs,
                                      relax_cell=False, **maker_kw)
    phonon_displacement_maker = MLIPStaticMaker(**maker_kw)
    return PhononMaker(
        name=f"Phonon MLIP {structure.composition.reduced_formula}",
        sym_reduce=False,
        symprec=1e-5,
        displacement=0.01,
        min_length=1,
        use_symmetrized_structure=None,
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={
            "kpoint_density_dos": 1,
            "dos_use_tetrahedron_method": False,
            "phonon_dos_sigma": 0.3,  # smearing width in THz, adjust to taste
        },
        bulk_relax_maker=bulk_relax_maker,
        static_energy_maker=static_energy_maker,
        phonon_displacement_maker=phonon_displacement_maker,
        born_maker=None).make(structure)


def make_elastic_mlip(structure: Structure, potential_kwargs: dict | None = None,
                      force_tol: float = 0.02, steps: int = 1000, *,
                      calculator: Any = None) -> Flow:
    relax_kwargs = {"verbose": False, "fmax": force_tol, "final_atoms_object_file": None}
    maker_kw = {"potential_kwargs": potential_kwargs, "calculator_instance": calculator}
    bulk_relax_maker = MLIPRelaxMaker(name="Elastic relax", relax_kwargs=relax_kwargs,
                                      relax_cell=False, **maker_kw)
    fit_elastic_tensor_kwargs = {"fitting_method": "independent"}
    return ElasticMaker(
        name=f"Elastic MLIP {structure.composition.reduced_formula}",
        order=2,
        sym_reduce=False,
        bulk_relax_maker=bulk_relax_maker,
        elastic_relax_maker=bulk_relax_maker,
        fit_elastic_tensor_kwargs=fit_elastic_tensor_kwargs).make(structure)


def make_MD_mlip(structure: Structure, potential_kwargs: dict | None = None,
                 TEBEG: int = 3300, n_steps: int = 1000, *,
                 calculator: Any = None) -> Flow:
    maker_kw = {"potential_kwargs": potential_kwargs, "calculator_instance": calculator}
    md_maker = MLIPMDMaker(
        name=f"MLIP MD {structure.composition.reduced_formula}",
        time_step=2,
        n_steps=n_steps,
        ensemble="nvt",
        temperature=TEBEG,
        dynamics="nose-hoover",
        ase_md_kwargs={},
        calculator_kwargs={},
        ionic_step_data=("structure", "energy", "forces", "stress"),
        tags=None,
        traj_file=None,
        **maker_kw,
    )
    return md_maker.make(structure)
