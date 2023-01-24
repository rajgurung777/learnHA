
from sklearn import preprocessing
import numpy as np

import time
import os

from infer_ha.libsvm.svmutil import *
from infer_ha.utils.util_functions import rel_diff
from infer_ha.clustering.gridSearch_fromSKLearn import gridSearchStart
from utils import misc_math_functions as myUtil


# New implementation when the source and the destination region has equal number of data
# i.e., the positive and negative data are of equal length, so no check or ordering is required
# Moreover, the size of the data is not so large and hence no record are discarded for learning
def getGuard_inequality(srcData, destData, L_y, boundary_order, Y):
    '''
    Implementation for equal number of positive and negative data and the size of these data
    are not very high (it is equal to the number of connecting points.

    :param srcData: Takes a list of position of one mode
    :param destData: Takes a list of position of another mode
    :param L_y: dimension
    :param boundary_order: polynomial degree
    :param Y: contains the actual values arranged based on the position
    :return: guard coefficients

    Note: when we have only single data for each class and if the two data differ by a very small fraction than SVM with
    parameter c=100 fails to return any solution. For example the two data on which SVM fails to find a solution are:
        a1 = [11.3119995521219, 20.0, 2980.72365171949]
        b1 = [11.3123854575116, 20.0, 2980.87930007233]
        train_data = [a1, b1]
        train_labels = [1, -1]
    Fixing: in such case, c=1 seems to work.

    Data Scaling in svm see the format https://github.com/cjlin1/libsvm/tree/master/python  Take a Look at file heart_scale

    '''

    accl = []
    guard_coeff = []
    y = []  # classes
    x = []  # data
    # # Amit: variables for plotting
    x_p = []
    x_n = []

    outputfilename = "outputs/data_scale"    # This file is also used for SVM scaling
    if os.path.exists(outputfilename):
        os.remove(outputfilename)
    f_out = open(outputfilename, "a")  # Opening file-id for writing output

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
    f_out.close()   # File must be closed for being used by svm_read_problem

    data_length = len(srcData)  # or len(destData)
    print("data size for SVM =", data_length)

    # ******* scaling data ************
    # print('Before Storing data in file for conversion to csr_matrix')
    # print(x)
    y, x = svm_read_problem('outputs/data_scale', return_scipy = True)  # y: ndarray, x: csr_matrix
    # print('Before scalling data for One-Versus-One')
    # print(x)
    # print('label y is ', y)
    # pdb.set_trace()
    scale_param = csr_find_scale_param(x, lower=-1, upper=1)
    x = csr_scale(x, scale_param)
    # print("param", scale_param)
    # # ******* scaling data ************

    # ******** Checking Data size and Data similarity ********
    a1 = []
    b1 = []
    relative_difference = 1.0 # assuming for data more than 1 we compute option c==100. Todo: found that even for more data we may have close relative_difference
    # if (len(srcData) == 1):
    #     for dim in range(L_y):
    #         a1.append(Y[srcData, dim])
    #         b1.append(Y[destData, dim])
    #     relative_difference = rel_diff(np.array(a1), np.array(b1))
    #     # print("relative_difference is ", relative_difference)
    # ****** above works only for data size == 1

    skipGridSearch = False
    c_value = 100   # Default value
    c_value_optimal = 100
    count_small_rel_diff = 0
    # print("srcData=", srcData)
    for id1, id2 in zip(srcData, destData):  # iterate both at the same time
        # print("id1 =", id1, "    id2=",id2)
        a1 = [Y[id1, dim] for dim in range(L_y)]
        b1 = [Y[id2, dim] for dim in range(L_y)]
        # print("a1=", a1, "    b1=",b1)
        rel_difference = rel_diff(np.array(a1), np.array(b1))
        # print("a1=", a1, "    b1=",b1, "     relative_diff=", relative_difference)
        if rel_difference < relative_difference:
            relative_difference = rel_difference   # stores the smallest relative difference
        if relative_difference <= 0.0001:
            count_small_rel_diff +=1
    # **********
    print("Total data with small relative difference =", count_small_rel_diff)
    # if relative_difference <= 0.0000001:  #increasing the original analysed value 0.0001
    if relative_difference <= 0.0001:  #increasing the original analysed value 0.0001
        print("********** ********** we found small relative difference =", relative_difference, " ********** **********")
        c_value = 1
        skipGridSearch = True   # Also skip grid search as this will also give problem for grid search

    # ******** Checking Data size and Data similarity ********

    # #********* Grid Search for hyperparameter tuning *************
    # For skiping small relative difference has higher priority than length of data. Therefore, that code appears first

    if (data_length <= 5):    # 5 because GridSearch here is using 5-Fold cross validation
        skipGridSearch = True

    if skipGridSearch == True:
        c_value_optimal = c_value
        # if c_value==1:  # not the default value which indicates small relative_difference encountered
        #     c_value_optimal = 1
        # else:
        #     c_value_optimal = 100
        gamma_value_optimal = float(1/L_y)
        coef_optimal = 1
        # print("Default parameter:- C: 100, gamma:",gamma_value_optimal, ", coef0:", coef_optimal)
    else:
        endTime = time.time()  # endTime: variable creation and start recording but will not be use
        startTime = time.time() # variable creation and started recording the current time
        gamma_value_optimal = float(1 / L_y)
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [0.1, gamma_value_optimal, 0.01],
                      'coef0': [0, 1, 0.1],
                      'kernel': ['poly']}
        # print("x_gs=", x_gs)
        scaler = preprocessing.StandardScaler().fit(x_gs)   # https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
        x_gs_scaled = scaler.transform(x_gs)
        c_value_optimal, gamma_value_optimal, coef_optimal = gridSearchStart(x_gs_scaled, y, param_grid)  # libsvm as backend
        # print("x_gs=", x_gs_scaled)
        # c_value_optimal, gamma_value_optimal, coef_optimal = gridSearchStart(x_gs, y, param_grid)   # using sklean here which take libsvm as backend

        endTime = time.time()   # recording the current time also replaces the previous value
        searchTime = endTime - startTime
        print ("  C=", c_value_optimal, ", Gamma=",gamma_value_optimal, ", coef0=",coef_optimal)
        print ("Search Time (secs): ", searchTime)
    #  ********** End of Grid Search for hyperparameter tuning ************

    prob = svm_problem(y, x)

    c_value = c_value_optimal

    # param = svm_parameter('-t 1 -d %d -c %d -r 1 -b 0 -q' % (boundary_order, c_value))  # -t 1 for Poly and 2 for RBF
    param = svm_parameter('-t 1 -d %d -c %g -r %g -g %g -b 0 -q' % (boundary_order, c_value, coef_optimal, gamma_value_optimal))  #

    # param = svm_parameter('-t 1 -d %d -c %d -r 0 -b 0 -q' % (boundary_order, c_value))  # Try coef0 or r to be 0
    # param = svm_parameter('-t 1 -d %d -c 100 -r 1 -b 0 -q' % boundary_order)  # -t 1 for Poly and 2 for RBF
    # print ("SVM param is ", param)
    # Graphic Interface Observation show that -c 100 gives better hyperplane separation (https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download)
    m = svm_train(prob, param)  # This is the time taking operation. recursion limit exceeded here for large data size

    sv = m.get_SV()
    if (len(sv)==0):    # if error in svm-train with c_value=100 or even with 1. Re-run it with c_value=1 for the second time
        print("SV is empty")
        c_value = 1
        # param = svm_parameter('-t 1 -d %d -c %d -r %d -b 0 -q' % (boundary_order, c_value, coef_optimal))  # -t 1 for Poly and 2 for RBF
        param = svm_parameter('-t 1 -d %d -c %g -r %d -g %g -b 0 -q' % (boundary_order, c_value, coef_optimal, gamma_value_optimal))  #
        # param = svm_parameter('-t 1 -d %d -c %d -r 0 -b 0 -q' % (boundary_order, c_value))  # Try coef0 or r to be 0
        m = svm_train(prob, param)  # running for the 2nd time due to error. Assuming no further error will occur

    svm_save_model('outputs/svm_model_file', m)
    guard_coeff = get_coeffs(L_y, m, gamma_value_optimal, order=boundary_order)  # this gives the hyperplane coefficients

    # print("guard_coeff is ", guard_coeff)
    p_label, p_acc, p_val = svm_predict(y, x, m, '-q')
    print('Accuracy is ', p_acc[0])

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



def get_coeffs(L_y, svm_model, gamma_value_optimal, order=1):
    """
    Author: Amit Gurung


    """
    # Amit: https://stackoverflow.com/questions/51836870/calculate-equation-of-hyperplane-using-libsvm-support-vectors-and-coefficients-i
    nsv = svm_model.get_nr_sv()
    svc = svm_model.get_sv_coef()
    sv = svm_model.get_SV()

    g = -svm_model.rho[0]  # Constant term
    # print("svm_model.rho[0] for rho = ", svm_model.rho[0])
    # print("g after rho = ", g)

    # print ('sv is ', sv)
    # print("svc is ", svc)
    # print("nsv is ", nsv)
    # print (sv[0])
    # L_y = len(sv[0])  # Now I can not use this to get the Number of dimensions
    # print ('Dimension: len(sv[0]) is ', len(sv[0]))
    # print ('sv[0] is ', sv[0])
    # print("svc is ", svc)
    # print("sv is ", sv)
    # print("nsv is ", nsv)

    # gamma = 0.5  # this was used in the original code
    # gamma = 1  # try 3 dimension with gamma==1
    # gamma = float(1/L_y)  # default 1/number_of_features. Observation does not help much but improve the separation line.
    gamma = gamma_value_optimal # computing using grid search which usually is 1/L_y

    # print("gamma optimal=",gamma)

    # Due to the Poly kernel in SVM has formula: (gamma.U'.V + 1)^degree  so in addition to the dimension of V we have + 1 an extra term
    # so the last term is considered as 1 for Eg. in (a+b+c)^degree, the last term c==1. Similar, this is done in run_tests.py to construct Guard equation
    coeff_expansion = myUtil.multinomial(L_y + 1, order)  # this coeff_expansion also include multinomial coefficients

    #Todo: trying to take coef0 = 0 instead of 1 and so we get (gamma.U'.V + 0)^degree
    # coeff_expansion = myUtil.multinomial(L_y, order)  # this coeff_expansion also include multinomial coefficients

    # print("coeff_expansion is ", coeff_expansion)
    list_a = [0] * int(len(coeff_expansion))
    # print("list_a is ", list_a)
    # print("svc[i][0] is ", svc[0][0])
    for i in range(nsv):
        # print("i here is ", i)
        g += svc[i][0]
        # print("g is ", g)
        coef_index = 0
        for term in coeff_expansion:
            term_index = 0
            sv_product = 1
            g_power = 0
            for each_var_power in term:
                flag = term_index in sv[i]
                # print("sv[i]:", sv[i])
                if flag == False:
                    aa = 0.0
                    # print("how????????????")
                else:
                    aa = sv[i][term_index]
                    # print("aa=",aa)

                # if term_index != (len(term) - 1):  # ignoring the last term since gamma is not associated with it
                if (term_index != (len(term) - 1) and term_index != 0): # ignore 1st term which is coefficient term and last term
                    g_power += each_var_power
                # else: # for the last term, which is 1
                #     aa = 1  # so that ^each_var_power will also be 1

                if (term_index == len(term) - 1):
                    aa = 1 # so that ^each_var_power will also be 1

                if term_index == 0:
                    coef_val = each_var_power # only the 1st term is the computed coefficient value
                else:
                    sv_product *= (aa ** each_var_power)  # so now term_index is starting from 1 which is required for SV, as it starts indexing from 1

                term_index += 1

            list_a[coef_index] += svc[i][0] * (gamma ** g_power) * coef_val * sv_product
            coef_index += 1
    # print("g is ", g)
    # print("Before list_a is ", list_a)
    list_a[len(coeff_expansion) - 1] = g  # replacing the last computed value of list_a by this g
    # print("After list_a is ", list_a)
    return list_a


