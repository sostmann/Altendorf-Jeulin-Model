import time
import sys
import os
import copy

import numpy as np
import scipy.stats

import Altendorf_Jeulin_Model.FiberModel as fm
import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.VectorizedForceBiased import run_force_biased_vectorized # vectorized version
from Altendorf_Jeulin_Model.ForceBiased import run_force_biased
from Altendorf_Jeulin_Model.io_utils import (
    print_fiber_positions,
    print_fiber_positions_to_file,
)
from Altendorf_Jeulin_Model.utils import cut_border
from Altendorf_Jeulin_Model.PoissonLines import simulate_poisson_lines

def main_infinite(VV, seed, size, iter: int):
    print(f"This is the Altendorf-Jeulin model for endless fibers - Simulation {iter}")

    image_size = (size, size, size)
    boundary_size = 50
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    ext_image_size = image_size + 2 * boundary_size_vec
    #VV = 0.12
    rng = np.random.RandomState(seed + iter)
    R = 17 / 2.0
    L = np.sqrt(3) / 2 * VV * (image_size[0] + 2 * boundary_size) ** 2 / R**2
    mu = 3 / 4 * np.pi * L * (image_size[0] + 2 * boundary_size) / image_size[0]
    N = int(mu)  # TODO
    A = np.array(
        [[1.697, 0.023, -0.028], [0.023, 0.873, -0.031], [-0.028, -0.031, 0.324]]
    )

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system_endless(
        N,
        R,
        A,
        image_size,
        boundary_size,
        10,
        100,
        has_beta=False,
        seed=seed,
        volume_fraction_should=VV,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # save initial fibers
    folder = "simulations/example"
    os.makedirs(folder, exist_ok=True)
    fs_cut = cut_border(fs, ext_image_size, boundary_size)

    io.save_fibers_as_tif(
        fs_cut,
        scale=1,
        domain=image_size,
        boundary=(0,0,0),
        path=folder + "/aj_model_endless_init.tif",
        is_periodic=False,
    )

    # pack the fibers
    start_time = time.time()
    run_force_biased_vectorized(fs, ext_image_size, is_periodic=False, verbose=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    # save packed fibers
    folder = "simulations/example"
    os.makedirs(folder, exist_ok=True)
    fs_cut = cut_border(fs, ext_image_size, boundary_size)

    io.save_fibers_as_tif(
        fs_cut,
        scale=1,
        domain=image_size,
        boundary=(0,0,0),
        path= folder + "/aj_model_endless.tif",
        is_periodic=False,
    )


def main_finite(VV, seed, size, iter: int):
    print(f"This is the Altendorf-Jeulin model for finite fibers - Simulation {iter}")

    image_size = (size, size, size)
    boundary_size = 50
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    ext_image_size = image_size + 2 * boundary_size_vec
    rng = np.random.RandomState(seed + iter)
    R = 17 / 2.0
    logMu = 4.5621
    logSigma = 0.2936
    L = scipy.stats.lognorm(s=logSigma, scale=np.exp(logMu))
    expected_length = np.exp(logMu + 0.5 * logSigma * logSigma)
    single_fiber_volume = np.pi * R * R * expected_length
    N = int(scipy.stats.poisson.rvs(VV * (200 + 2 * boundary_size) * (200 + 2 * boundary_size) * (200 + 2 * boundary_size) / single_fiber_volume))
    A = np.array(
        [[1.697, 0.023, -0.028], [0.023, 0.873, -0.031], [-0.028, -0.031, 0.324]]
    )

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system(
        N, L, R, A, ext_image_size, 10, 100, is_poisson=True, has_beta=False,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    start_time = time.time()
    run_force_biased_vectorized(fs, ext_image_size, verbose=True, is_periodic=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    # save packed fibers
    folder = "simulations/example"
    os.makedirs(folder, exist_ok=True)
    fs_cut = cut_border(fs, ext_image_size, boundary_size)

    io.save_fibers_as_tif(
        fs_cut,
        scale=1,
        domain=image_size,
        boundary=(0,0,0),
        path= folder + "/aj_model_finite.tif",
        is_periodic=False,
    )

def main_poisson(VV, seed, size):
    image_size = (size, size, size)
    R = 17 / 2.0
    L = np.sqrt(3) / 2 * VV * image_size[0] ** 2 / R**2
    mu = np.pi * L
    A = np.array(
        [[1.697, 0.023, -0.028], [0.023, 0.873, -0.031], [-0.028, -0.031, 0.324]]
    )

    # create a fiber system
    start_time = time.time()
    lines = simulate_poisson_lines(
        mu,
        R,
        A,
        image_size,
        has_beta=False,
        seed=seed,
        volume_fraction_should=VV,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Line simulation - Elapsed time: {elapsed_time:.6f} seconds")

    folder = "simulations/example"
    os.makedirs(folder, exist_ok=True)

    io.save_lines_as_tif(lines, image_size, folder + "/poisson_lines.tif", scale=1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <volume fraction> <seed> <image size>")
        sys.exit(1)

    VV = float(sys.argv[1])
    seed = int(sys.argv[2])
    size = int(sys.argv[3])

    #main_infinite(VV, seed, size, 1)
    #main_finite(VV, seed, size, 1)
    main_poisson(VV, seed, size)