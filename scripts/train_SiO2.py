import os

import ase
import jax.numpy as jnp
import numpy as np
from ase import io

from logan.logan import train_logan
from logan.utilities import get_ideal_mult

n_seed = 5
r_cut = 4
n_max = 4
ids = [
    "mp-559550",
    "mp-554151",
    "mp-546794",
    "mp-1188220",
    "mp-555235",
    "mp-546546",
]
for goal_id in ids:
    if not os.path.exists(f"{goal_id}_models"):
        os.makedirs(f"{goal_id}_models")
    train_path = f"../data/{goal_id}_training.extxyz"
    goal_atoms = ase.io.read(f"../data/{goal_id}_ground_truth.extxyz")
    mult = get_ideal_mult(goal_atoms, r_cut)
    anums = goal_atoms.get_atomic_numbers()
    gen_cell = jnp.array(goal_atoms.cell[:, :])
    print(goal_id, mult, anums)

    for bessel in [False, True]:
        for n_neig in [12, 25]:
            for seed in np.random.choice(10**5, n_seed, replace=False):
                save_path = f"{goal_id}_models/model_bessel{bessel}_rcut{r_cut}_nmax{n_max}_neig{n_neig}_seed{seed}.pkg"
                train_logan(
                    train_path=train_path,
                    save_path=save_path,
                    anums=anums,
                    gen_cell=gen_cell,
                    bessel=bessel,
                    r_cut=r_cut,
                    n_max=n_max,
                    n_neig=n_neig,
                    rattle=True,
                    n_rattle=10,
                    rattlesize=0.03,
                    n_epochs=45001,
                    n_step_per_snapshot=4500,
                    n_step_per_validate=100,
                    seed=seed,
                    lr_crit=10**-3.5,
                    lr_gen=10**-3.5,
                    periodic=True,
                    mult=mult,
                    n_type_sample=20,
                )
