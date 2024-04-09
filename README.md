# LoGAN

Data-efficient generative models for molecular and crystal structures

This repository accompanies the manuscript [LoGAN: Local generative adversarial network for novel structure prediction](https://doi.org/10.26434/chemrxiv-2024-vf9l1).

## Installation

Clone this repository and change directory:
```
git clone git@github.com:Madsen-s-research-group/logan.git
cd logan
```
To use the LoGAN, you will need a JAX installation. We recommend to install the package inside a conda environment (or any other virtual environment of your choice):

```
conda create -n logan python=3.10
conda activate logan
pip install --upgrade pip setuptools wheel
```
To use JAX with GPU support (recommended):
```
pip install --upgrade "jax[cuda11_local]==0.4.19" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
If you want to use a CPU-version of JAX, instead use
```
pip install --upgrade "jax[cpu]==0.4.19" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
You can now install the LoGAN package.
```
pip install -e .
```

## Reproduce our study

To reproduce the results presented in the manuscript, use the scripts provided in the `scripts` folder. Run
```
python train_C5H12O.py
```
to train 12 models on the QM9 C4H10O and C6H14O data to predict C5H10O isomers. The models will be saved to the folder `C5H12O_models`. On a GPU, each model takes about 15-25 minutes to train. For a quick test, reduce the number of epochs in the script. Then, run
```
python generate_filter_C5H12O.py
```
to produce 2000 structures for each model, cluster them and write the 20 cluster centers to file (`C5H12O_models/structures_20.extxyz`). The number of generated structures can be changed in the script. To run DFT geometry optimizations, we will use GPAW. Install GPAW according to their [documentation](https://wiki.fysik.dtu.dk/gpaw/install.html). Then, run
```
python relax_C5H12O.py
```
to optimize the geometry for each of the 20 predicted structures. Finally,
```
python compare_to_groundtruth_C5H12O.py
```
compares the obtained structures with the ground truth isomers by transforming the coordinate files to SMILES strings.

To reproduce the results for SiO2, run the corresponding scripts for SiO2 in the same order. The first script trains models for all six test systems, and the following scripts evaluate and inspect only one of them (mp-546794). This can be changed in the scripts easily.

## Train on your own data

To train with default hyperparameters on your own data, you will need a file
(readable by ASE) containing all training structures, as well as the unit cell
and type list. If you need periodic boundary conditions, you might need to use
a supercell of your unit cell depending on the chosen hyperparameters
(specifically, the cutoff of the descriptors), which can be specified using
the `mult` variable. For example, to train a model predicting aperiodic
structures of water (atomic numbers [8, 1, 1]) on the data
`my_training_data.extxyz` and save it to `model1.pkg`, the necessary python
code is (you will need to specify a dummy unit cell that is large enough to
contain the molecule):
```
from logan.logan import train_logan
train_logan(train_path="my_training_data.extxyz",
                save_path="model1.pkg",
                anums=np.array([8, 1, 1]),
                gen_cell=np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]),
                n_epochs=1001,
                n_step_per_snapshot=100,
                n_step_per_validate=100,
                periodic=False,
            )
```
In practice, 10-100k epochs are recommended (they are very fast after the
first one). If you would like to predict with periodic boundary conditions, the workflow
is very similar, but you will have to set `period=True`, and set the `mult`
variable to a meaningful supercell, that can fit the chosen descriptor cutoff
radius, e.g. `[2,2,2]` to repeat the cell along each direction.

To generate structures with a trained model for periodic or aperiodic
structures (this info is read from the pickle file anyways), you simply run:
```
from logan.logan import generate
gen_atoms_list = generate("model1.pkg", 2000)
```
If you would like to filter the 2000 generated structures to e.g. 20, you can
simply do (for clustering, you can choose the same or different
descriptor hyperparameters, in the following the defaults were used):
```
from logan.logan import cluster_filter
picked = cluster_filter(gen_atoms_list,
                        periodic=False,
                        n_cluster=20)
```
where for periodic structures, you simply set `periodic=True`.

In general, we recommend to train multiple models with different `seed`,
`r_cut`, `n_max`, `n_neig`, `bessel` (if True, uses Bessel functions, else
Gaussian functions), and generate structures with each model before clustering
and filtering the full list of structures down to the desired number.

## Copyright

Copyright (c) 2024, P. Kovacs, E. Heid, G. K. H. Madsen
