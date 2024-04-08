import pickle
from typing import Callable, List, Sequence

import ase
import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms, io
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .bessel_descriptors import (
    PowerSpectrumGenerator,
    PowerSpectrumGeneratorAtomCenterGauss,
)
from .training import (
    generate_structs,
    make_atomic_number_map,
    make_training_dict,
    train,
)
from .utilities import create_generate_descriptor, expand_atoms


def make_descriptor(
    n_types: int,
    bessel: bool = True,
    r_cut: float = 4,
    n_max: int = 4,
    n_neig: int = 12,
):
    """Creates a descriptor generator method with the given hyperparameters

    Args:
        n_types: The number of unique atom types
        bessel: Boolean whether to use Bessel descriptors (else atom-centered Gaussians)
        r_cut: The cut-off radius for the descriptors
        n_max: The n_max parameter for the descriptors
        n_neig: The maximum number of neighbors to use (even if more are within the cutoff radius)

    Returns:
        A descriptor generator object
    """
    if bessel:
        descriptor_generator = PowerSpectrumGenerator(
            n_max, r_cut, n_types, n_neig
        )
    else:
        descriptor_generator = PowerSpectrumGeneratorAtomCenterGauss(
            n_max, r_cut, n_types, n_neig
        )

    @jax.jit
    def desc_generator_method(allpos, alltype, pos, types, cell):
        desc = descriptor_generator.process_some_data(
            allpos, alltype, pos, cell
        )
        desc = jnp.reshape(desc, (pos.shape[0], -1))
        desc = jnp.append(desc, jax.nn.one_hot(types, n_types), axis=1)

        return desc

    return desc_generator_method


def train_logan(
    train_path: str,
    save_path: str,
    anums: ArrayLike,
    gen_cell: ArrayLike,
    bessel: bool = True,
    r_cut: float = 5,
    n_max: int = 3,
    n_neig: int = 12,
    rattle: bool = True,
    n_rattle: int = 20,
    rattlesize: float = 0.05,
    n_dimensions: int = 3,
    n_latent: int = 20,
    n_scalars: int = 0,
    generator_shape: Sequence[int] = [512, 256, 128, 64, 32],
    critic_shape: Sequence[int] = [256, 128, 64, 32],
    n_epochs: int = 60000,
    n_step_per_snapshot: int = 6000,
    n_step_per_validate: int = 100,
    n_validate_batch: int = 500,
    seed: int = 0,
    n_batch: int = 15,
    n_crit: int = 5,
    n_gp_per_crit: int = 2,
    lr_crit: float = 10**-4,
    lr_gen: float = 10**-4,
    periodic: bool = False,
    mult: ArrayLike = np.array([1, 1, 1]),
    n_type_sample: int = 0,
):
    """Trains a BesselGAN from scratch

    Args:
        train_path: Path to training structures (ASE readable, eg. extxyz format)
        save_path: Path to save model as pickle file
        anums: Numpy array or list of atomic numbers
        gen_cell: Numpy array of unit cell
        bessel: Hyperparameter for descriptors (whether to use Bessel vs Atom-centered Gaussians)
        r_cut: Hyperparameter for descriptors (cutoff radius)
        n_max: Hyperparameter for descriptors (number of basis functions)
        n_neig:  Hyperparameter for descriptors (maximum number of neighbors to consider irrespective of cutoff)
        rattle: Whether to rattle the training configurations
        n_rattle: Number of rattled configuration per training structure
        rattlesize: Magnitude of rattling
        n_dimensions: Dimensionality (usually three-dimensional data)
        n_latent: Number of latent variables (i.e. random numbers) that serve as input to the generator
        n_scalars: Number of extra scalars to predict for postprocessing
        generator_shape: The sequence of hidden layer widths
        critic_shape: The sequence of hidden layer widths
        n_epochs: The number of training epochs
        n_step_per_snapshot: Frequency of saving snapshots of the generator
        n_step_per_validate: Frequency of computing and printing validation statistics
        n_validate_batch: Batch size for validation
        seed: Seed for initializing the model
        n_batch: Batch size for training
        n_crit: Number of critic training steps per generator steps
        n_gp_per_crit: Number of gradient penalty steps per critic steps
        lr_crit: Learning rate of the critic
        lr_gen: Learning rate of the generator
        periodic: Whether to use periodic boundary conditions to create a supercell during pre- and postprocessing
        mult: Supercell size
        n_type_sample: If 0, the training dictionary samples the same number of types as the generated structure, else n_type_sample number of atomic descriptors per type
    """
    # Read atoms, rattle if necessary
    if rattle:
        print("Rattling atoms", n_rattle, rattlesize)
        train_atoms = []
        for atoms in ase.io.iread(train_path):
            if periodic:
                atoms = expand_atoms(atoms, r_cut)
            for i in range(n_rattle):
                tmp = atoms.copy()
                tmp.rattle(rattlesize)
                train_atoms.append(tmp)
    else:
        train_atoms = []
        for atoms in ase.io.iread(train_path):
            if periodic:
                atoms = expand_atoms(atoms, r_cut)
            train_atoms.append(atoms)
    print(
        "Obtained", len(train_atoms), "configurations (rattled if indicated)"
    )

    # Create mapping of atomic numbers to types
    anum_map = make_atomic_number_map(train_atoms)
    n_types = len(anum_map)
    n_points = len(anums)
    reverse_anum_map = {}
    for elem in anum_map:
        reverse_anum_map[anum_map[elem]] = elem
    types = np.array([anum_map[t] for t in anums])
    print("Predicting for", n_points, "atoms:", anums, "with types", types)

    # Create descriptor generator
    desc_generator_method = make_descriptor(
        n_types, bessel=bessel, r_cut=r_cut, n_max=n_max, n_neig=n_neig
    )
    training_dict = make_training_dict(
        train_atoms, desc_generator_method, anum_map, n_type_sample
    )
    print("Made descriptor with params:", bessel, r_cut, n_max, n_neig)

    # Train model
    gen_snapshot = train(
        training_dict=training_dict,
        desc_generator_method=desc_generator_method,
        types=types,
        gen_cell=gen_cell,
        n_points=n_points,
        n_dimensions=n_dimensions,
        n_latent=n_latent,
        n_scalars=n_scalars,
        generator_shape=generator_shape,
        critic_shape=critic_shape,
        n_epochs=n_epochs,
        n_step_per_snapshot=n_step_per_snapshot,
        n_step_per_validate=n_step_per_validate,
        n_validate_batch=n_validate_batch,
        seed=seed,
        n_batch=n_batch,
        n_crit=n_crit,
        n_gp_per_crit=n_gp_per_crit,
        lr_crit=lr_crit,
        lr_gen=lr_gen,
        periodic=periodic,
        mult=mult,
    )

    # Save model
    datapack = {}
    datapack["params_gen"] = gen_snapshot
    datapack["params"] = {
        "generator_shape": generator_shape,
        "n_latent": n_latent,
        "n_scalars": n_scalars,
        "reverse_anum_map": reverse_anum_map,
        "gen_cell": gen_cell,
        "n_dimensions": n_dimensions,
        "mult": mult,
        "n_points": n_points,
        "periodic": periodic,
        "types": types,
    }
    pickle.dump(datapack, open(save_path, "wb"))
    print("Saved model to", save_path)


def generate(filepath: str, n: int):
    """Generates structures using a trained generator

    Args:
        filepath: Path to the trained generator pickle file
        n: Number of generated (unfiltered) structures

    Returns:
        A list of generated structures
    """
    # Load generator
    model_pack = pickle.load(open(filepath, "rb"))
    params_gen = model_pack["params_gen"]
    generator_shape = model_pack["params"]["generator_shape"]
    n_latent = model_pack["params"]["n_latent"]
    n_scalars = model_pack["params"]["n_scalars"]
    reverse_anum_map = model_pack["params"]["reverse_anum_map"]
    gen_cell = model_pack["params"]["gen_cell"]
    n_dimensions = model_pack["params"]["n_dimensions"]
    n_points = model_pack["params"]["n_points"]
    periodic = model_pack["params"]["periodic"]
    types = model_pack["params"]["types"]

    gen_atoms_list = generate_structs(
        n=n,
        params_gen=params_gen,
        generator_shape=generator_shape,
        n_latent=n_latent,
        n_scalars=n_scalars,
        n_dimensions=n_dimensions,
        reverse_anum_map=reverse_anum_map,
        gen_cell=gen_cell,
        n_points=n_points,
        periodic=periodic,
        types=types,
    )

    return gen_atoms_list


def cluster_filter(
    gen_atoms_list: List[Atoms],
    n_cluster: int = 20,
    bessel: bool = True,
    r_cut: float = 5,
    n_max: int = 3,
    n_neig: int = 12,
    periodic: bool = False,
    n_components: int = 2,
):
    """Clusters and filters a list of atoms objects

    Args:
        gen_atoms_list: The sequence of Atoms objects to filter
        n_cluster: Number of clusters, i.e. filtered structures
        bessel: Hyperparameter for descriptors (whether to use Bessel vs Atom-centered Gaussians)
        r_cut: Hyperparameter for descriptors (cutoff radius)
        n_max: Hyperparameter for descriptors (number of basis functions)
        n_neig:  Hyperparameter for descriptors (maximum number of neighbors to consider irrespective of cutoff)
        periodic: Whether to use periodic boundary conditions to create a supercell
        n_components: Number of PCA components

    Returns:
        A list of filtered structures
    """
    anum_map = make_atomic_number_map(gen_atoms_list)
    n_types = len(anum_map.keys())
    desc_generator_method = make_descriptor(
        n_types, bessel=bessel, r_cut=r_cut, n_max=n_max, n_neig=n_neig
    )
    generate_single_desc = create_generate_descriptor(desc_generator_method)

    # Make supercells if necessary:
    gen_atoms_list_supercell = []
    for atoms in gen_atoms_list:
        if periodic:
            atoms = expand_atoms(atoms, r_cut)
        gen_atoms_list_supercell.append(atoms)

    # Make descriptors
    gen_invariant_desc_list = []
    for atoms in gen_atoms_list_supercell:
        tmp_type = np.array(
            [anum_map[num] for num in atoms.get_atomic_numbers()]
        )
        tmp_pos = atoms.positions
        tmp_desc = generate_single_desc(
            tmp_pos, tmp_type, tmp_pos, tmp_type, atoms.cell[:]
        )
        tmp_desc = np.array(tmp_desc.reshape((-1, tmp_desc.shape[-1])))

        invar_desc = np.zeros(shape=(0,))
        for i in range(n_types):
            indices = np.where(tmp_type == i)[0]
            invar_desc = np.append(
                invar_desc, np.mean(tmp_desc[indices, :], axis=0)
            )

        gen_invariant_desc_list.append(invar_desc)

    # Cluster
    pca = PCA(n_components=n_components)
    pca.fit(gen_invariant_desc_list)
    transformed_pca = pca.transform(gen_invariant_desc_list)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto").fit(
        transformed_pca
    )

    # Only use cluster centers
    picked = []
    for i in range(n_cluster):
        center = kmeans.cluster_centers_[i]
        dists = np.sum((transformed_pca - center) ** 2, axis=1)
        ind = np.argmin(dists)
        picked.append(gen_atoms_list[ind])

    return picked
