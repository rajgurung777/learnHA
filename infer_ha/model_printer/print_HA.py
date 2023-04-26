'''
This module prints an HA model into a plain .txt file
'''

import os

from utils import generator as generate
from infer_ha.model_printer.print_header import *
from infer_ha.model_printer.print_location import *
from infer_ha.model_printer.print_transition import *


def print_HA(P_modes, G, mode_inv, transitions, position, learning_parameters, outputfilename):
    """

    :param P_modes: holds a list of modes. Each mode is a list of structures; we call it a segment.
           Thus, P = [mode-1, mode-2, ... , mode-n] where mode-1 = [ segment-1, ... , segment-n] and segments are
           of type ([start_ode, end_ode], [start_exact, end_exact], [p1, ..., p_n]).
           The size of the list P_modes is equal to the number of clusters or modes of the learned hybrid automaton (HA).
    :param G: is a list. Each item of the list G is a list that holds the coefficients (obtained using linear regression)
           of the ODE of a mode of the learned HA.
    :param mode_inv: is a list with items of type [mode-id, invariant-constraints]. Where mode-id is the location number
                  and invariant-constraints holds the bounds (min, max) of each variable in the corresponding mode-id.
    :param transitions: is a list with structure [src_mode, dest_mode, guard_coeff, assignment_coeff, assignment_intercept]
                     where
            src_mode: is the source location number
            dest_mode: is the destination location number
            guard_coeff: is a list containing the coefficient of the guard equation (polynomial)
            assignment_coeff: is a list containing the coefficient of the assignment equations (from linear regression)
            assignment_intercept: is a list containing the intercepts of the assignment equations (linear regression)
    :param position: is a list containing positions of the input list_of_trajectories. This structure is required for printing
            the HA model. Particularly, to get the starting positions of input trajectories for identifying initial mode(s).
    :param learning_parameters: is a dictionary data structure containing all the parameters required for our learning
            algorithm. The arguments of the learning_parameters can also be passed as a command-line arguments. The
            command-line usages can be obtained using the --help command. To find the details of the arguments see the
            file/module "utils/commandline_parser.py"
    :param outputfilename: name of the file where the HA model is printed as an output.
    :return:

    """

    maxorder = learning_parameters['ode_degree']
    boundary_order = learning_parameters['guard_degree']
    num_mode = len(P_modes)   # size returned by DTW clustering algorithm.
    total_ode_coeff = G[0].shape[1] # total columns of 1st location's ODE. Size is dimension + constant-intercept term

    total_ode_rows = G[0].shape[0]  # total row of 1st location's ODE. Size is dimension
    system_dim = total_ode_rows
    # system_dim = total_ode_coeff - 1 # minus the intercept term
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
    print_location(f_out, P_modes, G, mode_inv, Exp, position)
    print_transition(f_out, transitions, system_dim, boundary_order)
    f_out.close()

    return
