import time
import sys
import os

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
from concurrent.futures import ProcessPoolExecutor

def main(VV, seed, size, iter: int):
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

    # save initialized fibers
    folder = "simulations/AJ_model_endless/size" + str(size) + "_VV" + str(VV) + "_" + str(seed)
    os.makedirs(folder, exist_ok=True)
    filepath = folder + "/aj_model_endless_init_" + str(iter)
    fs_cut_init = cut_border(fs, ext_image_size, boundary_size)
    io.save_fibers_as_small_graph(filepath, fs_cut_init)

    # pack the fibers
    start_time = time.time()
    run_force_biased_vectorized(fs, ext_image_size, is_periodic=False, verbose=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    # save packed fibers
    fs_cut = cut_border(fs, ext_image_size, boundary_size)
    filepath = folder + "/aj_model_endless_" + str(iter)
    #io.write_gad(fs_cut, filepath + ".gad", image_size, 1e-06, is_periodic=False)
    io.save_fibers_as_small_graph(filepath, fs_cut)

    #io.save_fibers_as_tif(
    #    fs,
    #    scale=4,
    #    domain=image_size,
    #    boundary=(boundary_size, boundary_size, boundary_size),
    #    path="examples/outputs/training/size1280_VV" + str(VV) + "_" + str(seed) + "/AJ_model_endless.tif",
    #    is_periodic=False,
    #)
    #io.print_fiber_positions_to_file(fs_cut, "examples/outputs/training/size1280_VV" + str(VV) + "_" + str(seed) +"/nonwoven.txt")

def run_iteration(args):
    VV, seed, size, i = args
    return main(VV, seed, size, i)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <volume fraction> <seed> <image size>")
        sys.exit(1)

    VV = float(sys.argv[1])
    seed = int(sys.argv[2])
    size = int(sys.argv[3])

    args = [
        (VV, seed, size, i)
        for i in range(10)
    ]

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(run_iteration, args))
