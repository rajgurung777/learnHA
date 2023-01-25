"""
This module performs operations related to SVM and HA's guard creation.

"""
# from infer_ha.libsvm.svm import svm_problem, svm_parameter  # direct calling created wrong object on svm_problem()
# from infer_ha.libsvm.svmutil import svm_train
from infer_ha.libsvm.svmutil import *

def svm_model_training(x, y, boundary_order, c_value_optimal, coef_optimal, gamma_value_optimal):
    """
    Implementation of SVM training operation with the given hyperparameter(s).

    :param x: the actual data
    :param y: the label on data x
    :param boundary_order: degree of the polynomial kernel
    :param c_value_optimal: the c value one of the SVM hyperparameter.
    :param coef_optimal: optimal value for the SVM hyperparameter coef.
    :param gamma_value_optimal: gamma the SVM hyperparameter.
    :return: the trained SVM model.

    """
    prob = svm_problem(y, x)
    c_value = c_value_optimal
    # param = svm_parameter('-t 1 -d %d -c %d -r 1 -b 0 -q' % (boundary_order, c_value))  # -t 1 for Poly and 2 for RBF
    param = svm_parameter(
        '-t 1 -d %d -c %g -r %g -g %g -b 0 -q' % (boundary_order, c_value, coef_optimal, gamma_value_optimal))  #
    print("prob =", prob)
    # param = svm_parameter('-t 1 -d %d -c %d -r 0 -b 0 -q' % (boundary_order, c_value))  # Try coef0 or r to be 0
    # param = svm_parameter('-t 1 -d %d -c 100 -r 1 -b 0 -q' % boundary_order)  # -t 1 for Poly and 2 for RBF
    # print ("SVM param is ", param)
    # Graphic Interface Observation show that -c 100 gives better hyperplane separation (https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download)
    m = svm_train(prob, param)  # This is the time taking operation. recursion limit exceeded here for large data size

    sv = m.get_SV()
    if (
            len(sv) == 0):  # if error in svm-train with c_value=100 or even with 1. Re-run it with c_value=1 for the second time
        print("SV is empty")
        c_value = 1
        # param = svm_parameter('-t 1 -d %d -c %d -r %d -b 0 -q' % (boundary_order, c_value, coef_optimal))  # -t 1 for Poly and 2 for RBF
        param = svm_parameter(
            '-t 1 -d %d -c %g -r %d -g %g -b 0 -q' % (boundary_order, c_value, coef_optimal, gamma_value_optimal))  #
        # param = svm_parameter('-t 1 -d %d -c %d -r 0 -b 0 -q' % (boundary_order, c_value))  # Try coef0 or r to be 0
        m = svm_train(prob, param)  # running for the 2nd time due to error. Assuming no further error will occur

    return m
