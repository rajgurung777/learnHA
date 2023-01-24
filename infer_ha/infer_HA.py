"""
This is the main module for inferring an HA model.
"""


import sys  # This is used for command line arguments

from infer_ha.segmentation.segmentation import two_fold_segmentation, segmented_trajectories
from infer_ha.clustering.clustering import select_clustering
from infer_ha.infer_invariants.invariants import compute_mode_invariant
from infer_ha.libsvm.svmutil import *
from infer_ha.helpers import plotDebug as plotdebug
from infer_ha.segmentation.compute_derivatives import diff_method_backandfor
from utils.trajectories_parser import preprocess_trajectories
from infer_ha.infer_transitions.compute_transitions import compute_transitions

sys.setrecursionlimit(1000000)  # this is the limit


def infer_model(list_of_trajectories, learning_parameters):
    """
    The main module to infer an HA model for the input trajectories.


    :param list_of_trajectories: Each element of the list is a trajectory. A trajectory is a 2-tuple of (time, vector), where
            time: is a list having a single item. The item is the sampling time, stored as a numpy.ndarray having structure
                as (rows, ) where rows is the number of sample points. The dimension cols is empty meaning a single dim array.
            vector: is a list having a single item. The item is a numpy.ndarray with (rows,cols), rows indicates the number of
                points and cols as the system's dimension. The dimension is the total number of variables in the trajectories
                including both input and output variables.
    :param learning_parameters: is a dictionary data structure containing all the parameters required for our learning
                                algorithm. The arguments of the learning_parameters can also be passed as a command-line
                                arguments. The command-line usages can be obtained using the --help command.
                                To find the details of the arguments see the file/module "utils/commandline_parser.py"

    :return:
        P: is a list. Each item of the list P contain list of values which are positions of points of a trajectories.
           The size of the list P is equal to the number of clusters or modes of the learned hybrid automaton (HA).
        G: is a list. Each item of the list G is a list that holds the coefficients (obtained using linear regression)
           of the ODE of a mode of the learned HA.
        mode_inv: is a list with items of type [mode-id, invariant-constraints]. Where mode-id is the location number
                  and invariant-constraints holds the bounds (min, max) of each variable in the corresponding mode-id.
        transitions: is a list with structure [src_mode, dest_mode, guard_coeff, assignment_coeff, assignment_intercept]
                     where
            src_mode: is the source location number
            dest_mode: is the destination location number
            guard_coeff: is a list containing the coefficient of the guard equation (polynomial)
            assignment_coeff: is a list containing the coefficient of the assignment equations (from linear regression)
            assignment_intercept: is a list containing the intercepts of the assignment equations (linear regression)

    """

    stepsize = learning_parameters['stepsize']
    maxorder = learning_parameters['ode_degree']
    boundary_order = learning_parameters['guard_degree']
    ep = learning_parameters['segmentation_error_tol']
    size_of_input_variables = learning_parameters['size_input_variable']
    size_of_output_variables = learning_parameters['size_output_variable']
    variableType_datastruct =  learning_parameters['variableType_datastruct'] # processed and stored in data-struct
    isInvariant = learning_parameters['is_invariant']

    mode_inv = []
    transitions = []

    t_list, y_list, position = preprocess_trajectories(list_of_trajectories)

    # Apply Linear Multistep Method
    A, b1, b2, Y, ytuple = diff_method_backandfor(y_list, maxorder, stepsize)   # compute forward and backward version of BDF
    num_pt = Y.shape[0]
    # print("Initial computation done!")

    # ********* Debugging ***********************
    # if os.path.exists("outputs/amit_backward.txt"):
    #     os.remove("outputs/amit_backward.txt")
    # file_out = open("outputs/amit_backward.txt","a")
    # file_out.write("b1 backward derivatives: ")
    # str1 = ""
    # pos_value = 0
    # relDiff = 0.0
    # for i in range(0, len(b1)):
    #     if (i!=0 and i != len(b1)):
    #         relDiff = rel_diff(b1[i, size_of_input_variables:], b1[(i-1), size_of_input_variables:])  # compute relative difference between current and previous
    #     # str1 = "pos:" + str(pos_value) + " b1[i] = " + str(b1[i]) + "  b1[i,onlyOutVar] = " + str(b1[i, size_of_input_variables:]) + "  relDiff = " + str(relDiff) + " \n"
    #     str1 = "pos: " + str(pos_value) + str(b1[i, size_of_input_variables:]) + "  relDiff = " + str(relDiff) + " \n"
    #     pos_value += 1
    #     file_out.write(str1)
    # file_out.close()

    # if os.path.exists("outputs/amit_forward.txt"):
    #     os.remove("outputs/amit_forward.txt")
    # file_out2 = open("outputs/amit_forward.txt", "a")
    # file_out2.write("b2 forward derivatives: ")
    # pos_value = 0
    # for i in range(0, len(b2)):
    #     if (i!=0 and i != len(b2)):
    #         relDiff = rel_diff(b2[i], b2[i-1])  # compute relative difference between current and previous
    #     str1 = "pos:" + str(pos_value) + "   " +  str(b2[i]) + "  relDiff = " + str(relDiff) + " \n"
    #     pos_value += 1
    #     file_out2.write(str1)
    # file_out2.close()
    # ********* Debugging ***********************

    # print("ytuple is = ", ytuple)
    # Segment and fit
    # res, drop, clfs = segment_and_fit(A, b1, b2, ytuple,ep) #Amit: uses the simple relative-difference between forward and backward BDF presented in the paper, Algorithm-1.
    # res, drop, clfs, res_modified = segment_and_fit_Modified_two(A, b1, b2, ytuple,ep)
    res, drop, clfs, res_modified = two_fold_segmentation(A, b1, b2, ytuple, size_of_input_variables, ep)
    print("Number of segments, len(res)=", len(res))
    res = res_modified  #TODO: We can comment this if this fix is not required. Now guard is closer to the boundary
    segmentedTrajectories, res, clfs = segmented_trajectories(clfs, res, position) # deleted the last segment in each trajectory
    print("Segmentation done!")

    L_y = len(y_list[0][0])  # Number of dimensions

    # ********* Plotting/Visualizing various points for debugging *************************
    # plotdebug.plot_guard_points(segmentedTrajectories, L_y, t_list, Y) # pre-end and end points of each segment
    # plotdebug.plot_reset_points(segmentedTrajectories_modified, L_y, t_list, Y) # plotting Reset or Start points
    # plotdebug.plot_segmentation(res, L_y, t_list, Y) # Trying to verify the segmentation for each segmented points

    number_of_segments_before_cluster = len(res)
    P, G = select_clustering(res, A, b1, clfs, Y, t_list, L_y, learning_parameters) # when len(res) < 2 compute P and G for the single mode
    # print("Fixing Dropped points ...") # I dont need to fix
    # P, Drop = dropclass(P, G, drop, A, b1, Y, ep, stepsize)  # appends the dropped point to a cluster that fits well
    # print("Total dropped points (after fixing) are: ", len(Drop))
    number_of_segments_after_cluster = len(P)

    # *************** Trying to plot points ***********************************
    # plotdebug.plot_dropped_points(t_list, L_y, Y, Drop)
    # plotdebug.plot_after_clustering(t_list, L_y, P, Y)
    mode_inv = compute_mode_invariant(L_y, P, Y, isInvariant)
    # *************** Trying to plot the clustered points ***********************************
    # print("Number of num_mode= ", num_mode)
    # print("Number of Clusters, len(P)=", len(P))
    '''
    TODO: remove taking input 'num_mode' from user
    
    if (len(P) < num_mode):
        print("Number of desired Modes = ", num_mode)
        num_mode = len(P)
        print("Number of Clusters Learned = ", len(P))
    '''
    # num_mode = len(P)

    transitions = compute_transitions(P, position, segmentedTrajectories, L_y, boundary_order, Y,
                                      variableType_datastruct, number_of_segments_before_cluster,
                                      number_of_segments_after_cluster)

    return P, G, mode_inv, transitions


