import os
import pickle

import ase
from ase import io

from logan.logan import cluster_filter, generate

# Generate structures
model_names = [
    "C5H12O_models/" + name
    for name in os.listdir("C5H12O_models")
    if name.endswith(".pkg")
]

gen_atoms_list = []
for filepath in model_names:
    gen_atoms_list.extend(generate(filepath, 2000))

N = 20
picked = cluster_filter(
    gen_atoms_list,
    bessel=True,
    r_cut=5,
    n_max=4,
    n_neig=20,
    periodic=False,
    n_cluster=N,
)
ase.io.write(f"C5H12O_models/structures_{N}.extxyz", picked)
