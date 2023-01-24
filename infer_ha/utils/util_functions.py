'''
Contains all the helper functions used in the project infer_ha

'''


import numpy as np
import math

def mat_norm(A):
    ''' This is Euclidean norm '''
    return math.sqrt(np.square(A).sum())


def rel_diff(A, B):
    if (mat_norm(A) + mat_norm(B)) == 0:
        return mat_norm(A - B)    # fixing division by zero error
    else:
        return mat_norm(A - B) / (mat_norm(A) + mat_norm(B))


def matrowex(matr, l):
    """Pick some rows of a matrix to form a new matrix."""
    finalmat = None
    for i in range(0, len(l)):
        if i == 0:
            finalmat = np.mat(matr[l[i]])
        else:
            finalmat = np.r_[finalmat, np.mat(matr[l[i]])]
    return finalmat


