import time

import Altendorf_Jeulin_Model.FiberModel as fm
import numpy as np
import scipy.stats
from Altendorf_Jeulin_Model.utils import cut_border
from numpy.f2py.crackfortran import verbose

import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.ForceBiased import run_force_biased
from Altendorf_Jeulin_Model.io_utils import (
    print_fiber_positions_to_file,
)


def main():
    example_AJ_finite()
    #example_AJ_endless()


def example_AJ_finite():
    print("This is the Altendorf-Jeulin model")
    logSigma = 0.2936
    logMu = 4.5621
    L = scipy.stats.lognorm(s=0.2936, scale=np.exp(4.5621))
    R = 3.5
    padding = scipy.stats.lognorm.ppf(0.95, s=0.2936, scale=np.exp(4.5621))
    expected_length = np.exp(logMu + 0.5 * logSigma * logSigma)
    single_fiber_volume = np.pi * R * R * expected_length
    nfibers = int(np.floor(0.5 * (400 + 2 * padding) * (400 + 2 * padding) * (400 + 2 * padding) / single_fiber_volume))
    image_size = np.array([int(400+padding), int(400+padding), int(400+padding)])
    boundary_size = padding
    beta = 0.1

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system(
        nfibers, L, R, beta, image_size, 100, 100, is_poisson=False
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    start_time = time.time()
    run_force_biased(fs, image_size, verbose=True, is_periodic=False, step_size_verbose=10)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    io.fiber_lengths_output("outputs.csv", fs, image_size, boundary_size)

    #io.save_fibers_as_tif(
    #    fs, domain=image_size, path="examples/outputs/AJ_model.tif", is_periodic=True
    #)
    print_fiber_positions_to_file(fs, "examples/outputs/fibers.txt")
    io.write_gad(
        fs,
        "examples/outputs/AJ_model.gad",
        (100,100,100),
        1e-06,
        is_periodic=True,
    )


def example_AJ_endless():
    print("This is the Altendorf-Jeulin model for endless fibers")
    image_size = (400, 400, 400)
    boundary_size = 50
    VV = 0.12
    mean_R = 11
    R = scipy.stats.uniform(loc=mean_R, scale=5)
    L = np.sqrt(3) / 2 * VV * (image_size[0] + 2 * boundary_size) ** 2 / mean_R**2
    mu = 3 / 4 * np.pi * L * (image_size[0] + 2 * boundary_size) / image_size[0]
    A = np.array(
        [[1.697, 0.023, -0.028], [0.023, 0.873, -0.031], [-0.028, -0.031, 0.324]]
    )

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system_endless(
        mu,
        R,
        A,
        image_size,
        boundary_size,
        10,
        100,
        has_beta=False
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    start_time = time.time()
    run_force_biased(fs, image_size, is_periodic=False, verbose=True, output_path="outputs/")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    io.save_fibers_as_tif(
        fs,
        scale=4,
        domain=image_size,
        boundary=(boundary_size, boundary_size, boundary_size),
        path="outputs/AJ_model_endless.tif",
        is_periodic=False,
    )
    fs_cut = cut_border(fs, image_size, boundary_size)
    io.save_fibers_as_small_graph("outputs/nonwoven", fs_cut)
    io.write_gad(fs, "outputs/AJ_model_endless.gad", image_size, 4e-06, is_periodic=False)


if __name__ == "__main__":
    main()
