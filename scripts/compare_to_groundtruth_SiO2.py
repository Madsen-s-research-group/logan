import ase
from ase import io
from ase.io import Trajectory

test_atoms = ase.io.read("../data/mp-546794_ground_truth.extxyz")
test_ener = test_atoms.get_potential_energy()
print("ground truth", test_ener)
for i in range(20):
    final = Trajectory(f"mp-546794_models/structures_20_{i}.traj")[-1]
    ener = final.get_potential_energy()
    print("structure", i, ener, ener - test_ener)
