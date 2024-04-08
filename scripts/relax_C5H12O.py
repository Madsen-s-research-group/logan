import ase
from ase import io
from ase.optimize import FIRE
from gpaw import GPAW

all_atoms = ase.io.iread("C5H12O_models/structures_20.extxyz")
for i, system in enumerate(all_atoms):
    try:
        system.center(vacuum=3.0)
        system.pbc = False

        calc = GPAW(mode="lcao")
        system.calc = calc

        relax = FIRE(atoms=system)
        relax.run(fmax=0.05, steps=1000)

        ase.io.extxyz.write_extxyz(
            f"C5H12O_models/structures_20_{i}_relaxed.xyz", system, plain=True
        )
    except:
        print("Unable to relax structure", i)
