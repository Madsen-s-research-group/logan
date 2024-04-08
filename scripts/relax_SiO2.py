import ase
from ase import io
from ase.optimize import FIRE
from gpaw import GPAW, PW, mpi

all_atoms = ase.io.iread("mp-546794_models/structures_20.extxyz")
for i, system in enumerate(all_atoms):
    try:
        system.pbc = True
        calc_params = {
            "mode": PW(400),
            "xc": "LDA",
            "kpts": [2, 2, 2],
            "symmetry": {"tolerance": 1.0e-3},
        }
        calc = GPAW(**calc_params)
        system.calc = calc

        relax = FIRE(
            atoms=system, trajectory=f"mp-546794_models/structures_20_{i}.traj"
        )
        relax.run(fmax=0.05, steps=1000)

        ase.io.extxyz.write_extxyz(
            f"mp-546794_models/structures_20_{i}_relaxed.xyz",
            system,
            plain=True,
        )
    except:
        print("Unable to relax structure", i)
