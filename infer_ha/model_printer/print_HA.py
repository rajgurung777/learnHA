'''
This module prints an HA model into a plain .txt file
'''

import os

from utils import generator as generate
from infer_ha.model_printer.print_header import *
from infer_ha.model_printer.print_location import *
from infer_ha.model_printer.print_transition import *


def print_HA(P, G, mode_inv, transitions, learning_parameters, outputfilename):

    maxorder = learning_parameters['ode_degree']
    boundary_order = learning_parameters['guard_degree']
    num_mode = len(P)   # size returned by DTW clustering algorithm.
    total_ode_coeff = G[0].shape[1] # total columns of 1st location's ODE. Size is dimension + constant-intercept term
    system_dim = total_ode_coeff - 1 # minus the intercept term
    gene = generate.generate_complete_polynomial(system_dim, maxorder)
    # print ("Gene matrix here ", gene)
    expression = ""  # list of terms as string
    # for i in range(0, gene.shape[0]):
    #     term =""
    #     for j in range(0, gene.shape[1]):
    #         if (gene[i][j] != 0.):
    #             term = "x" + str(j) + " "
    #             expression+=term
    #     if (i==(gene.shape[0] - 1)):
    #         pass
    #     else:
    #         expression = expression + " + "
    # expression = expression + " 1" #last expression is always 1 here   [0  0]

    for i in range(0, gene.shape[0]):
        term = ""
        number_of_var_per_term = 0
        for j in range(0, gene.shape[1]):
            power = gene[i][j]
            if (power != 0.):
                if (number_of_var_per_term > 0):
                    term = "* "

                if (power > 1):
                    term += "(x" + str(j) + ")^" + str(power) + " "
                else:  # meaning power is 1
                    term += "x" + str(j) + " "
                expression += term
                number_of_var_per_term += 1
        if (i == (gene.shape[0] - 1)):
            pass
        else:
            expression = expression + " + "
    expression = expression + " 1"  # last expression is always 1 here   [0  0]

    Exp = expression.split("+")  # split string of expression into separate term
    # print('Splitted Exp is ',Exp)
    # print("Expression is ", expression)
    # print ("After calling P is ", P)
    # print("After calling G is ", G)


    # outputfilename = '/home/somedir/Documents/python/logs';
    # As file at outputfilename is deleted now, so we should check if file exists or not before deleting them
    if os.path.exists(outputfilename):
        os.remove(outputfilename)

    f_out = open(outputfilename, "a")  # Opening file-id for writing output
    print_header(f_out, num_mode, system_dim, transitions)
    print_location(f_out, P, G, mode_inv, Exp)
    print_transition(f_out, transitions, system_dim, boundary_order)
    f_out.close()

    return
