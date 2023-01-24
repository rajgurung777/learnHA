import numpy as np 
import time

def generator_items(m, n):
    """@m : the number of variables
       @n : the order
    """
    if m == 1:
        return np.ones((1,1))*n
    if n == 0:
        return np.zeros((1,m))
    A = np.zeros((1, m))
    A[0][0] = n
    for i in range(1, n+1):
        B = generator_items(m-1,i)
        B_row_number = B.shape[0]
        C = np.ones((B_row_number, 1))*(n-i)
        D = np.c_[C,B]
        A = np.r_[A,D]
    return A

def generate_complete_polynomial(m,n):
    """
    @m: number of variables
    @n: maximum order.

    E.g. output for m = 2, n = 2:
    array([2, 0], [1, 1], [0, 2], [1, 0], [0, 1], [0, 0]).

    """
    A = None
    for i in range(0, n+1):
        g = generator_items(m,n-i)
        if i == 0:
            A = g
        else:
            A = np.r_[A,g]
    return A

if __name__ == "__main__":
    start = time.time()
    A = generate_complete_polynomial(2,2)
    end = time.time()
    print(A)
    print(end-start)
    print("row number: ", A.shape[0])
    print("col number: ", A.shape[1])
