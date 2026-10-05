import time
import sys
import os

import numpy as np
import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.PoissonLines import simulate_poisson_lines

from concurrent.futures import ProcessPoolExecutor


def main(VV, seed, size, sim_number = 1):
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

    folder = "simulations/poisson-lines/size" + str(size) + "_VV" + str(VV) + "_" + str(seed)
    os.makedirs(folder, exist_ok=True)
    filepath = folder + "/poisson_lines" + str(sim_number)

    #io.save_lines_as_tif(lines, image_size, "examples/outputs/poisson-lines/size800" + str(size) + "_VV" + str(VV) + "_" + str(seed) + "/lines.tif", scale=4)
    io.save_lines_as_graph(filepath, lines)
    #io.write_gad(lines, "simulations/poisson-lines/size" + str(size) + "_VV" + str(VV) + "_" + str(seed) + "/poisson_lines_" + str(i) + ".gad", image_size, 1e-06, is_periodic=False)


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
        (VV, seed + i, size, i)
        for i in range(10)
    ]

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(run_iteration, args))
