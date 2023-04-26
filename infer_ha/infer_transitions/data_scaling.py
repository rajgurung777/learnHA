import numpy as np
import os
from utils import misc_math_functions as myUtil


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


def makeCompatibleCoefficient(p_coeff, boundary_degree):
    """
    Computes new coefficient expression of the form (p_coeff)^boundary_degree.
    Note since SVM consider kernel of the form (gamma.U.V + 1)^boundary_degree, so we have to use the same formula for
    making the new expression compatible with the coefficient-expression computed by SVM.

    @param p_coeff: the coefficient vector obtianed from the scaling function. Here the dimension is the original
                dimension of the system for eg. the variable U or V in SVM mentioned above.
    @param boundary_degree: order for guard polynomial.
    @return: newCoeff: the new computed coefficient to be used for inverse scaling, which will be compatible for inverse
                scale operation.
    """
    # newCoeff = []
    dim_p_coeff = len(p_coeff)
    coeff_expansion = myUtil.multinomial(dim_p_coeff + 1, boundary_degree)  # this coeff_expansion include multinomial coefficients
    # print("coeff_expansion is ", coeff_expansion)

    newCoeff = [0] * int(len(coeff_expansion))
    term_index = 0
    for term in coeff_expansion:
        prod = 1
        for index in range(0, len(term)):
            if index == 0:  # this is for coefficient
                prod = term[index]
            elif index <= len(p_coeff):   # for the rest of the values in term but upto index of p_coeff. <= because we do -1
                if term[index] != 0:
                    # print("term[index] =", term[index])
                    # print("index-1 =", index-1)
                    # print("p_coeff[index-1] =", p_coeff[index-1])
                    prod = prod * pow(p_coeff[index-1], term[index])  # term[index] here is either 1 or 2. Length of p_coeff is one less
                    # print("prod =", prod)
        # print("prod =", prod)
        newCoeff[term_index] = prod
        term_index += 1
    # print("newCoeff =", newCoeff)
    l = len(newCoeff) - 1
    newCoeff = newCoeff[:l] # discarding the last term which is 1 in the kernel expression (gamma.U.V + 1)^2
    # print("newCoeff without last term =", newCoeff)
    return newCoeff

def inverse_scale(guard_coeff, scale_param, L_y, boundary_degree):
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
    Then, the hyperplane after inverse scaling is (a'  p_coeff) x + {(a dot-product p_offset) + b} = 0
    So, I have to compute: term1 x + term2 = 0, where
      term1 = a'  p_coeff  and       term2 ={(a' . p_offset) + b}
    '''
    # print("scale_param is ", scale_param)
    # print("Scaled guard_coeff is ", guard_coeff)
    p_coeff = scale_param['coef']
    p_offset = scale_param['offset']
    # print("p_offset is ", p_offset)
    # print("p_coeff is ", p_coeff)
    # print("L_y+1 is ", L_y+1)
    # print("len(guard_coeff) is ", len(guard_coeff))

    # assert (L_y+1 == len(guard_coeff))  #todo enable later

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
    # print("guard_coeff is ", guard_coeff)

    if boundary_degree == 1:

        term1 = np.multiply(a, p_coeff)
        term2 = np.dot(a,p_offset) + b

        coef_size = L_y + 1
        coef = [0] * coef_size
        for i in range(0, L_y):
            coef[i] = term1[i]
        coef[L_y] = term2
        # print("Inverse Scaled guard_coeff is ", coef)
        guard_coeff = coef
        # ********* Inverse Scaling Result ************
        return guard_coeff


    if boundary_degree >= 2:
        # create new a and b and also create new compatible p_coeff by (p_coeff)^2
        size = len(guard_coeff) - 1
        a = guard_coeff[:size]   # extracting the coefficients
        b = guard_coeff[size]    # extracting the intercept term
        # print("guard_coeff = ", guard_coeff)
        # print("a =", a)
        # print("b =", b)

        new_p_Coeff = makeCompatibleCoefficient(p_coeff, boundary_degree)
        term1 = np.multiply(a, new_p_Coeff)
        new_p_offset = makeCompatibleCoefficient(p_offset, boundary_degree)
        term2 = np.dot(a, new_p_offset) + b

        coef_size = len(term1)
        coef = [0] * (coef_size + 1)
        for i in range(0, coef_size):
            coef[i] = term1[i]
        coef[coef_size] = term2
        # print("Inverse Scaled guard_coeff is ", coef)
        guard_coeff = coef
        # ********* Inverse Scaling Result ************
        return guard_coeff