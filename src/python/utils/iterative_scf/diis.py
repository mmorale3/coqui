import numpy as np


def diis_solve(res_list):
    space_size = len(res_list)

    B = np.zeros((space_size, space_size), dtype=float)
    for i in range(space_size):
        for j in range(space_size):
            B[i, j] = np.dot(res_list[i], res_list[j]).real

    coeffs = np.linalg.solve(B, np.ones(space_size))
    # renormalization factor
    N = np.sum(coeffs)

    return coeffs / N

