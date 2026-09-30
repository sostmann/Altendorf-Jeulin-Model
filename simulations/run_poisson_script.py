import time
import sys

import numpy as np
import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.PoissonLines import simulate_poisson_lines


def main(VV, seed, size, sim_number = 10):
    for i in range(sim_number):
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

        #io.save_lines_as_tif(lines, image_size, "examples/outputs/poisson-lines/size800" + str(size) + "_VV" + str(VV) + "_" + str(seed) + "/lines.tif", scale=4)
        #io.save_lines_as_graph("outputs/poisson-lines/size" + str(size) + "_VV" + str(VV) + "_" + str(seed) + "/nonwoven_sim_" + sim_number, lines)
        io.write_gad(lines, "simulations/poisson-lines/size" + str(size) + "_VV" + str(VV) + "_" + str(seed) + "/poisson_lines_" + str(i) + ".gad", image_size, 1e-06, is_periodic=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <volume fraction> <seed> <image size>")
        sys.exit(1)

    VV = float(sys.argv[1])
    seed = int(sys.argv[2])
    size = int(sys.argv[3])

    main(VV, seed, size, 10)

