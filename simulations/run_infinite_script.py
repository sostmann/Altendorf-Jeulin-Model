import time
import sys

import numpy as np
import scipy.stats

import Altendorf_Jeulin_Model.FiberModel as fm
import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.ForceBiased import run_force_biased
from Altendorf_Jeulin_Model.io_utils import (
    print_fiber_positions,
    print_fiber_positions_to_file,
)
from Altendorf_Jeulin_Model.utils import cut_border


def main(VV, seed):
    print("This is the Altendorf-Jeulin model for endless fibers")
    image_size = (1280, 1280, 1280)
    boundary_size = 50
    #VV = 0.12
    rng = np.random.RandomState(seed)
    mean_R = 11
    R = scipy.stats.uniform(loc=mean_R, scale=5)#17 / 2.0
    L = np.sqrt(3) / 2 * VV * (image_size[0] + 2 * boundary_size) ** 2 / mean_R**2
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

    # pack the fibers
    start_time = time.time()
    run_force_biased(fs, image_size, is_periodic=False, verbose=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    io.save_fibers_as_tif(
        fs,
        scale=4,
        domain=image_size,
        boundary=(boundary_size, boundary_size, boundary_size),
        path="examples/outputs/training/size1280_VV" + str(VV) + "_" + str(seed) + "/AJ_model_endless.tif",
        is_periodic=False,
    )
    fs_cut = cut_border(fs, image_size, boundary_size)
    io.print_fiber_positions_to_file(fs_cut, "examples/outputs/training/size1280_VV" + str(VV) + "_" + str(seed) +"/nonwoven.txt")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <a> <b>")
        sys.exit(1)

    a = float(sys.argv[1])
    b = int(sys.argv[2])
    main(a, b)
