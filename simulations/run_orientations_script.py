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


def main(VV, seed, size, beta, iter: int):
    image_size = (size, size, size)
    boundary_size = 50
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    ext_image_size = image_size + 2 * boundary_size_vec

    rng = np.random.RandomState(seed)
    R = 17 / 2.0
    L = np.sqrt(3) / 2 * VV * (image_size[0] + 2 * boundary_size) ** 2 / R ** 2
    mu = 3 / 4 * np.pi * L * (image_size[0] + 2 * boundary_size) / image_size[0]
    N = int(mu)  # TODO

    # sample 0
    # A = np.array(
    #    [[1.697, 0.023, -0.028], [0.023, 0.873, -0.031], [-0.028, -0.031, 0.324]]
    # )
    # sample 2
    # A = np.array(
    #    [[1.483, 0.018, 0.023], [0.018, 1.077, 0.022], [0.023, 0.022, 0.440]]
    # )
    # sample 3
    #A = np.array(
    #    [[1.341, -0.005, 0.158], [-0.005, 1.083, 0.036], [0.158, 0.036, 0.576]]
    #)

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system_endless(
        N,
        R,
        beta,
        image_size,
        boundary_size,
        10,
        100,
        has_beta=True,
        seed=seed,
        volume_fraction_should=VV,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    start_time = time.time()
    run_force_biased_vectorized(fs, ext_image_size, is_periodic=False, verbose=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    # save packed fibers
    folder = "simulations/orientation_analysis/AJ/size" + str(size) + "_VV" + str(VV) + "_A" + str(beta) + "_" + str(
        seed)
    os.makedirs(folder, exist_ok=True)
    fs_cut = cut_border(fs, ext_image_size, boundary_size)
    filepath = folder + "/aj_model_endless_" + str(iter)
    io.save_fibers_as_small_graph(filepath, fs_cut)

def run_iteration(args):
    VV, seed, size, beta, i = args
    return main(VV, seed, size, beta, i)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: python {sys.argv[0]} <volume fraction> <seed> <image size> <beta>")
        sys.exit(1)

    VV = float(sys.argv[1])
    seed = int(sys.argv[2])
    size = int(sys.argv[3])
    beta = float(sys.argv[4])

    args = [
        (VV, seed + i, size, beta, i)
        for i in range(10)
    ]

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(run_iteration, args))
