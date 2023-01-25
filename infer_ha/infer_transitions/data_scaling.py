import numpy as np

import os


def create_data(output_filename, srcData, destData, L_y, Y):
    """
    Implementation of an equal number of positive and negative data and the size of these data are not very high. It is
    equal to the number of connecting points.

    :param output_filename: file name to store the data values for performing scaling
    :param srcData: takes a list of position of the source mode
    :param destData: takes a list of position of the destination mode
    :param L_y: system dimension
    :param Y: contains the y_list values for all the points except the first and last M points (M is the order in BDF).
    :return: the data values x: the actual values, y: the label positive or negative for the x data values and
            x_gs: the x values in the format suitable for grid search operation
    """
    y = []  # classes
    x = []  # data
    x_p = []
    x_n = []

    # output_filename = "outputs/data_scale"  # This file is also used for SVM scaling

    if os.path.exists(output_filename):
        os.remove(output_filename)
    f_out = open(output_filename, "a")  # Opening file-id for writing output

    x_gs = []  # data for grid search
    for id0 in srcData:  # The class with +1 is stored first
        y.append(1)
        str1 = "+1 "
        x.append({dim + 1: Y[id0, dim] for dim in range(L_y)})
        xx = [Y[id0, dim] for dim in range(L_y)]
        x_gs.append(xx)
        x_p.append({dim + 1: Y[id0, dim] for dim in range(L_y)})

        for dim in range(L_y):
            str1 += str(dim + 1) + ":" + str(Y[id0, dim]) + " "
            # str1 += str(dim + 1) + ":" + "{0:.3f}".format(Y[id0, dim]) + " "
        str1 += "\n"
        f_out.write(str1)
    for id1 in destData:
        y.append(-1)
        str1 = "-1 "
        x.append({dim + 1: Y[id1, dim] for dim in range(L_y)})
        xx = [Y[id1, dim] for dim in range(L_y)]
        x_gs.append(xx)
        x_n.append({dim + 1: Y[id1, dim] for dim in range(L_y)})
        for dim in range(L_y):
            str1 += str(dim + 1) + ":" + str(Y[id1, dim]) + " "
            # str1 += str(dim + 1) + ":" + "{0:.3f}".format(Y[id1, dim]) + " "

        str1 += "\n"
        f_out.write(str1)
    f_out.close()  # File must be closed for being used by svm_read_problem

    return x, y, x_gs

def inverse_scale(guard_coeff, scale_param, L_y):
    """
    Implementation of an equal number of positive and negative data and the size of these data are not very high. It is
    equal to the number of connecting points.

    :param guard_coeff: the coefficients obtained from SVM with the scaled data
    :param scale_param: scaling parameters obtained from SVM
    :param L_y: system dimension
    :return: the coefficients after performing inverse scaling

    """

    # ********* Inverse Scaling Result ************
    '''
    For inverse scaling
    p_offset and p_coeff
    If    a'x + b = 0 is the learned hyperplane with the scaled data.
    Then, the hyperplane after inverse scaling is (a'/p_coeff) x + {(-a'p_offset/p_coeff) + b} = 0
    So, I have to compute: term1 x + term2 = 0, where
      term1 = a'/p_coeff  and       term2 ={-(a' p_offset/p_coeff) + b}
    '''
    # print("scale_param is ", scale_param)
    # print("Scaled guard_coeff is ", guard_coeff)
    p_coeff = scale_param['coef']
    p_offset = scale_param['offset']
    # print("p_offset is ", p_offset)
    # print("p_coeff is ", p_coeff)
    assert (L_y+1, len(guard_coeff))
    a = guard_coeff[:L_y]   # extracting the coefficients
    b = guard_coeff[L_y]    # extracting the intercept term
    # Note that if the data value x has all zeros in its last dimensions/columns then p_coeff and p_offset will discard
    # all these data and will have its dimension reduced by the number of last zero columns (This may be because scaling
    # works with csr_matrix(sparse matrix)).
    # So to fix this reduced size p_coeff and p_offset so that we can perform our inverse-scaling operation. We check
    # two vector's dimensions (p_coeff and our coefficients 'a' and 'b') and append zero/zeros to p_coeff and p_offset
    # to make them compatible for inverse-scaling operations. (For experimental proof check file "AmitSVMTest.py"
    scaledDimension = len(p_coeff)
    trueDataDimension = len(a)
    if scaledDimension != trueDataDimension:
        extraZeros = trueDataDimension - scaledDimension
        for i in range(0, extraZeros):
            p_coeff = np.append(p_coeff, 0.0)
            p_offset = np.append(p_offset, 0.0)

    # print("Modified p_offset is ", p_offset)
    # print("Modified p_coeff is ", p_coeff)

    # term1 = np.divide(a, p_coeff)    # print("term1 is ", term1)
    # term2 = np.divide(p_offset, p_coeff)    # print("term2 is ", term2)
    # term2 = np.dot(a, term2)    # print("term2 is ", term2)
    # term2 = b - term2    # print("term2 is ", term2)

    term1 = np.multiply(a, p_coeff) # Modified in the meeting
    term2 = np.dot(a,p_offset) + b # Modified in the meeting

    coef_size = L_y + 1
    coef = [0] * coef_size
    for i in range(0, L_y):
        coef[i] = term1[i]
    coef[L_y] = term2
    # print("Inverse Scaled guard_coeff is ", coef)
    guard_coeff = coef
    # ********* Inverse Scaling Result ************
    return guard_coeff