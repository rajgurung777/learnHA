"""
This module computes the derivatives
"""

import numpy as np

from utils import generator as generate # generate_complete_polynomial


def BDF_backward_version(stepM, stepsize, y_points, index):
    """
    Computes an approximate derivatives using backwards differentiation formula (BDF) derived from Linear Multi-step
    Method (LMM) with the step size as M. This function computes the backward version of BDF.
    @param stepM: The step size M of LMM
    @param stepsize: step size between two data points
    @param y_points: the actual data values of the trajectories
    @param index: the position in the trajectory whose derivative is being approximated
    @return:
    """


    if stepM == 2:
        backward_derivative = (3 * y_points[index] - 4 * y_points[index - 1] + 1 * y_points[index - 2]) / (2 * stepsize)

    if stepM == 3:
        backward_derivative = (11 * y_points[index] - 18 * y_points[index - 1] + 9 * y_points[index - 2] - 2 * y_points[index - 3]) / (6 * stepsize)

    if stepM == 4:
        backward_derivative = (25 * y_points[index] - 48 * y_points[index - 1] + 36 * y_points[index - 2] - 16 * y_points[index - 3] + 3 * y_points[index - 4]) / (12 * stepsize)

    if stepM == 5:
        backward_derivative = (137 * y_points[index] - 300 * y_points[index - 1] + 300 * y_points[index - 2] - 200 *
                           y_points[index - 3] + 75 * y_points[index - 4] - 12 * y_points[index - 5]) / (60 * stepsize)

    if stepM == 6:
        backward_derivative = (147 * y_points[index] - 360 * y_points[index - 1] + 450 * y_points[index - 2] - 400 *
            y_points[index - 3] + 225 * y_points[index - 4] - 72 * y_points[index - 5] + 10 * y_points[index - 6]) / (60 * stepsize)

    return backward_derivative


def BDF_forward_version(stepM, stepsize, y_points, index):
    """
    Computes an approximate derivatives using backwards differentiation formula (BDF) derived from Linear Multi-step
    Method (LMM) with the step size as M. This function computes the forward version of BDF.
    @param stepM: The step size M of LMM
    @param stepsize: step size between two data points
    @param y_points: the actual data values of the trajectories
    @param index: the position in the trajectory whose derivative is being approximated
    @return:
    """

    if stepM == 2:
        forward_derivative = (-3 * y_points[index] + 4 * y_points[index + 1] - 1 * y_points[index + 2]) / (2 * stepsize)

    if stepM == 3:
        forward_derivative = (-11 * y_points[index] + 18 * y_points[index + 1] - 9 * y_points[index + 2] + 2 * y_points[index + 3]) / (6 * stepsize)

    if stepM == 4:
        forward_derivative = (-25 * y_points[index] + 48 * y_points[index + 1] - 36 * y_points[index + 2] + 16 * y_points[index + 3] - 3 * y_points[index + 4]) / (12 * stepsize)

    if stepM == 5:
        forward_derivative = (-137 * y_points[index] + 300 * y_points[index + 1] - 300 * y_points[index + 2] + 200 * y_points[index + 3] - 75 * y_points[index + 4] + 12 * y_points[index + 5]) / (60 * stepsize)

    if stepM == 6:
        forward_derivative = (-147 * y_points[index] + 360 * y_points[index + 1] - 450 * y_points[index + 2] + 400 *
                               y_points[index + 3] - 225 * y_points[index + 4] + 72 * y_points[index + 5] - 10 *
                               y_points[index + 6]) / (60 * stepsize)

    return forward_derivative



def diff_method_backandfor(y_list, order, stepsize, stepM):
    """Using multi-step backwards differentiation formula (BDF) to calculate the
    coefficient matrix. We have concatenated all the trajectories into a single list because this helped us discard fewer data than
    considering trajectories as a list of independent trajectories. This is because, for the first M points (M the
    step size of BDF), derivatives can not be computed using the backward BDF. In contrast, for the last M points,
    the derivatives can not be computed using the forward version of BDF. Thus, we discard each trajectory's first and
    last M points (2M points) if we consider independent trajectories. Therefore, considering a concatenation of all
    the trajectories as a single item list will allow us to avoid this discarding of 2M points from each trajectory.
    Moreover, the position of points becomes easy to track concerning a single concatenated list.


    :param
        y_list: a single-item list whose item is a numpy.ndarray containing vector of values (of input and output) as a
            concatenated list of trajectories.
    :param
        order: is the degree of the polynomial of the flow equation (ODE).
    :param
        stepsize: is the sampling time period between two points.
    :param
        stepM: is the step size of Linear Multi-step Method (step M)
    :return:
        The following lists:
        final_A_mat: For every point of a trajectory the coefficients of the monomial terms obtained using the \Phi
         function (or the mapping function) as mention in Jin et al. paper.
        final_b1_mat: the derivatives of each point computed using the backward version of BDF.
        final_b2_mat: the derivatives of each point computed using the forward version of BDF.
        final_y_mat: contains the y_list values for all the points except the first and last M points.
        ytuple: is a list of two tuple with the first item as zero and the second the size of the total points.

    """
    final_A_mat = None
    final_b1_mat = None
    final_b2_mat = None
    final_y_mat = None
    ytuple = []

    # stepM = 5   # the step size of Linear Multi-step Method (step M)

    # print("y_list=", y_list)   #This is the data from simulation except the time column

    L_y = y_list[0].shape[1]
    # print ("L_y=", L_y)  # returns the dimension excluding the time-column
    gene = generate.generate_complete_polynomial(L_y, order)
    L_p = gene.shape[0]
    # print("Value of L_p = ", L_p)  # L_p = total number of terms in the mapping function \Phi as in the paper (depending on the order-size and dimension)
    for k in range(0, len(y_list)):     # this will run only 1 iteration now since we concatenated the trajectories

        y_points = y_list[k]
        L_t = len(y_points)
        # print("value of k =", k)
        D = L_t - stepM  # here M = order5      //Discarding the last M-points
        # print("Value of D = ", D) # D = total-points - 5
        A_matrix = np.zeros((D - stepM, L_p), dtype=np.double)  # stores the mapping function \Phi as in the paper
        b1_matrix = np.zeros((D - stepM, L_y), dtype=np.double)  # stores the backward_BDF using LMM as in the paper
        b2_matrix = np.zeros((D - stepM, L_y), dtype=np.double)  # stores the forward_BDF using LMM  as in the paper
        y_matrix = np.zeros((D - stepM, L_y), dtype=np.double)
        coef_matrix = np.ones((L_t, L_p), dtype=np.double)  # stores the coefficient F as in the paper
        for i in range(0, L_t):
            for j in range(0, L_p):
                for l in range(0, L_y):
                    coef_matrix[i][j] = coef_matrix[i][j] * (y_points[i][l] ** gene[j][l])
        # For all the points i: For each variable, the mapping function \Phi is computed (monomials)

        for i in range(stepM, D):      #//Discarding the first M-points
            # forward
            A_matrix[i - stepM] = coef_matrix[i]
            # b1_matrix[i - 5] = (137 * y_points[i] - 300 * y_points[i - 1] + 300 * y_points[i - 2] -
            #                   200 * y_points[i - 3] + 75 * y_points[i - 4] - 12 * y_points[i - 5]) / (60 * stepsize)
            b1_matrix[i - stepM] = BDF_backward_version(stepM, stepsize, y_points, i)

            # b2_matrix[i - 5] = (-137 * y_points[i] + 300 * y_points[i + 1] - 300 * y_points[i + 2] +
            #           200 * y_points[i + 3] - 75 * y_points[i + 4] + 12 * y_points[i + 5]) / (60 * stepsize)
            b2_matrix[i - stepM] = BDF_forward_version(stepM, stepsize, y_points, i)

            y_matrix[i - stepM] = y_points[i]

        # Finally, A_matrix now contain the monomial terms obtained using \Phi function
        # b1_matrix and b2_matrix contains the forward and backward BDF values using LMM. As in the paper Equation (10)
        # y_matrix contains only the y values for the points for which A_matrix, b1_matrix and b2_matrix are computed
        # print("b1_matrix =", b1_matrix)
        # print("b2_matrix =", b2_matrix)

        if k == 0:  # for our input file else condition never occurs
            ytuple.append((0, A_matrix.shape[0]))       #shape[0] is the rows of A_matrix
            final_A_mat = A_matrix
            final_b1_mat = b1_matrix
            final_b2_mat = b2_matrix
            final_y_mat = y_matrix
        else:
            l1 = final_A_mat.shape[0]
            final_A_mat = np.r_[final_A_mat, A_matrix]
            l2 = final_A_mat.shape[0]
            final_b1_mat = np.r_[final_b1_mat, b1_matrix]
            final_b2_mat = np.r_[final_b2_mat, b2_matrix]
            final_y_mat = np.r_[final_y_mat, y_matrix]
            ytuple.append((l1, l2))

    return final_A_mat, final_b1_mat, final_b2_mat, final_y_mat, ytuple


