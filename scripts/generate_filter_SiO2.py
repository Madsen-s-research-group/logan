import os
import pickle

import ase
from ase import Atoms, io

from logan.logan import cluster_filter, generate

# Generate structures (here only for one structure, one hyperparameter setting)
model_names = [
    "mp-546794_models/" + name
    for name in os.listdir("mp-546794_models")
    if name.endswith(".pkg")
    and name.startswith("model_besselTrue_rcut4_nmax4_neig12")
]

gen_atoms_list = []
for filepath in model_names:
    gen_atoms_list.extend(generate(filepath, 1000))

N = 20
picked = cluster_filter(
    gen_atoms_list,
    bessel=True,
    r_cut=4,
    n_max=4,
    n_neig=12,
    periodic=True,
    n_cluster=N,
)
ase.io.write(f"mp-546794_models/structures_{N}.extxyz", picked)
