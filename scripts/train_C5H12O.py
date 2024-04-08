import os

import ase
import jax.numpy as jnp
import numpy as np
from ase import io

from logan.logan import train_logan

train_path = "../data/C5H12O_training.extxyz"
n_seed = 3
anums = np.array([6] * 5 + [1] * 12 + [8])
gen_cell = jnp.array([[8.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 6.0]])


for r_cut in [5, 6]:
    for n_max in [3, 4]:
        n_neig = 12
        for seed in np.random.choice(10**5, n_seed, replace=False):
            if not os.path.exists("C5H12O_models"):
                os.makedirs("C5H12O_models")
            save_path = f"C5H12O_models/model_bessel_rcut{r_cut}_nmax{n_max}_neig{n_neig}_seed{seed}.pkg"
            train_logan(
                train_path=train_path,
                save_path=save_path,
                anums=anums,
                gen_cell=gen_cell,
                bessel=True,
                r_cut=r_cut,
                n_max=n_max,
                n_neig=n_neig,
                rattle=True,
                n_rattle=20,
                rattlesize=0.05,
                n_epochs=60001,
                n_step_per_snapshot=6000,
                n_step_per_validate=100,
                seed=seed,
                lr_crit=10**-4,
                lr_gen=10**-4,
                periodic=False,
            )
