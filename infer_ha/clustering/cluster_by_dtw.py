"""
This module contains our approach to clustering using the DTW algorithm.

"""

import csv

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw         # https://pypi.org/project/fastdtw/
from sklearn import metrics
from sklearn import linear_model

from ..helpers import plotDebug as plotdebug
from infer_ha.clustering.utils import get_signal_data, compute_correlation
# from infer_ha.clustering.utils import  create_simple_modes_positions
from infer_ha.clustering.utils import create_simple_modes_positions_for_ODE
from infer_ha.utils.util_functions import matrowex



def get_desired_clusters(P_modes, A, b1):
    """
    This function computes the coefficients of the polynomial ODE for each cluster/mode.

    :param P_modes: hols a list of modes. Each mode is a list of structures; we call it a segment.
        Thus, P = [mode-1, mode-2, ... , mode-n] where mode-1 = [ segment-1, ... , segment-n] and segments are
        of type ([start_ode, end_ode], [start_exact, end_exact], [p1, ..., p_n]).
    :param A: For every point of a trajectory the coefficients of the monomial terms obtained using the \Phi
         function (or the mapping function) as mention in Jin et al. paper.
    :param b1: the derivatives of each point computed using the backward version of BDF.
    :return: The computed cluster and the coefficients of the polynomial ODE.
        # P: hols a list of modes. Each mode is a list of structures; we call it a segment.
        # Thus, P = [mode-1, mode-2, ... , mode-n] where mode-1 = [ segment-1, ... , segment-n] and segments are
        # of type ([start_ode, end_ode], [start_exact, end_exact], [p1, ..., p_n]).
        G: is a list containing the list of the coefficients of the polynomial ODE.

    """

    # P = create_simple_modes_positions(P_modes)
    P = create_simple_modes_positions_for_ODE(P_modes)  # for ODE inference we use segment excluding boundary points

    # print("Sort clusters based on Data-size and take the first num_mode clusters")
    length_and_modepts = [(len(P[i]), P[i]) for i in range(0, len(P))]  # create a list of 2-tuple
    # length_and_modepts.sort(reverse=True)  # this data is sorted from highest number of points in the cluster to lowest
    print("DTW: Total clusters = ", len(length_and_modepts))

    #  ***************************************************************
    num_mode = len(length_and_modepts) # Made this change after Paper submission (in the paper engineTiming which was 42
    # reduced to 20 although this will not have effect, since only the first 4 modes were used to generate trajectories
    # now we removed from the argument passing num_mode as user decided argument
    #  ***************************************************************

    mode_pts = []
    # print ("length_and_modepts = ",length_and_modepts) # *** this value can be less than num_mode ***
    # Fixing when the number of segments returned by DTW is less than user's num_mode input
    if len(length_and_modepts) < num_mode:  # This will execute only if we pass num_mode as argument
        num_mode = len(length_and_modepts)

    for i in range(0, num_mode):  # Now since num_mode is assumed to be <= len(length_and_modepts), so first num_mode of data is considered as outputs
        _, mode_ptsi = length_and_modepts[i]
        mode_pts.append(mode_ptsi)
    # Fit each cluster again
    clfs = []
    # print("Computing Linear Regression(ODE) for the combined Cluster")
    for i in range(num_mode):  # For this considered outputs coefficients are computed again
        clf = linear_model.LinearRegression(fit_intercept=False)
        clf.fit(matrowex(A, mode_pts[i]), matrowex(b1, mode_pts[i]))
        clfs.append(clf)

    # P = mode_pts    # we do not want to return simple-segmented-modes
    G = []
    for i in range(len(clfs)):
        G.append(clfs[i].coef_)

    # return P, G
    # return P_modes, G
    return G

def cluster_by_dtw(segmented_traj, A, b1, Y, t_list, L_y, correl_threshold, distance_threshold,
                   size_of_input_variables, maximum_ode_prune_factor=50):
    """
    This function contains our approach to clustering using the DTW algorithm.

    :param segmented_traj: is a list of a custom data structure consisting of segmented trajectories (positions). Each
        item of the list contains tuple of the form ([start_ode, end_ode], [start_exact, end_exact], [p_1, ... , p_n]).
        The Tuple has three items:
            (1) first, a list of two values for recording start and end points for learning ODE
            (2) second, a list of two values for recording start and end points for learning guard and assignment using
            the exact point of a jump
            (3) third, a list of values representing the position of points of the trajectories.
    :param A: For every point of a trajectory the coefficients of the monomial terms obtained using the \Phi
         function (or the mapping function) as mention in Jin et al. paper.
    :param b1: the derivatives of each point computed using the backward version of BDF.
    :param Y: contains the y_list values for all the points except the first and last M points (M is the order in BDF).
    :param t_list: a single-item list whose item is a numpy.ndarray containing time-values as a concatenated list.
    :param L_y: is the dimension (input + output variables) of the system whose trajectory is being parsed.
    :param correl_threshold: threshold value for correlation for DTW comparison of two segmented trajectories.
    :param distance_threshold: threshold value for distance for DTW comparison of two segmented trajectories.
    :param size_of_input_variables: total number of input variables in the given trajectories.
    :param maximum_ode_prune_factor: maximum number of segments to be used for ODE computation per cluster/mode.
    :return: The computed cluster and the coefficients of the polynomial ODE.
        P: holds a list of modes. Each mode is a list of structures; we call it a segment.
        Thus, P = [mode-1, mode-2, ... , mode-n] where mode-1 = [ segment-1, ... , segment-n] and segments are
        of type ([start_ode, end_ode], [start_exact, end_exact], [p1, ..., p_n]).
        G: is a list containing the list of the coefficients of the polynomial ODE.
    """

    P = []  # holds a list of modes and each mode is a list of segments and a mode is a list of segment
            # Thus P = [mode-1, mode-2, ... , mode-n]
            # and mode-1 = [ segment-1, ... , segment-n]
            # and segment-1 = ([start_ode, end_ode], [start_exact, end_exact], [p1, ..., p_n])
    # *******************************************************************************************
    f_ode, t_ode = get_signal_data(segmented_traj, Y, L_y, t_list, size_of_input_variables)  # get the segmented signal from trajectory.
    #Now, f_ode is the derivatives of the segmented data-points
    # print("f_ode is ", f_ode)
    # *******************************************************************************************

    # ******** To estimate the maximum and minimum values during clustering DTW ********
    min_distance = 1e10
    max_distance = 0
    min_correl = 1
    max_correl = -1
    # *********************************************************************************

    # ******************************************************************
    res = segmented_traj  # to keep the OLD implementation's variable
    # ******************************************************************
    inx = 0
    performance_prune_count = 0
    count = len(res)

    # P.append(res[0]) # stores the first segment

    f_ode1 = f_ode
    t_ode1 = t_ode  # t_ode is used only for plotting for debugging
    res1 = res    # makes a copy of the segmented_traj for working
    res2 = res1
    i = 0    # j = 0
    flag = 0
    while (i < count):
        j = i + 1
        mode = [res1[i]]  # to hold list of segments per mode; initialize the first segmented_traj
        delete_position = []
        while (j < count):  #  runs once for each f_ode_[i]
            # print("i=", i, " :f_ode[i] is ", f_ode[i])
            # print(" and j=", j, "  :f_ode[j] is ", f_ode[j])
            dataSize = len(f_ode[i])
            if len(f_ode[i]) > 5:
                dataSize = 5    # setting a small datasize for performance, tradeoff with accuracy
            # half_dataSize = math.ceil(len(f_ode[i])/2)
            # dataSize = half_dataSize     #len(f_ode[i])
            distance1, path = fastdtw(f_ode[i], f_ode[j], radius=dataSize, dist=euclidean)
            distance = distance1 / (len(f_ode[i]) + len(f_ode[j]))
            correlValue = compute_correlation(path, f_ode[i], f_ode[j])
            if distance < min_distance:
                min_distance = distance
            if distance > max_distance:
                max_distance = distance
            if correlValue < min_correl:
                min_correl = correlValue
            if correlValue > max_correl:
                max_correl = correlValue

            # print("i=", i, " and j=",j , " :  distance1 = ", distance1, " :  distance = ", distance, "   and   correlation = ", correlValue)

            # if (i==0 and j>=7 and j<=8):
            # plotdebug.plot_signals(t_ode[i], f_ode[i], t_ode[j], f_ode[j])

            if correlValue >= correl_threshold and distance_threshold == 0:  # distance_threshold is disabled or ignored
                print("******************************************** Found *******************************")
                # print("i=", i, " and j=", j, " : Ignored distance = ", distance, "   and   correlation = ", correlValue)
                performance_prune_count += 1
                if performance_prune_count <= maximum_ode_prune_factor:  # Just this line helps in pruning same segments for performance of ODE computaion
                    # P[inx].extend(res1[j])      # //pushing in the same cluster i for the first 50 segments
                    # print("performance_prune_count=", performance_prune_count)
                    mode.append(res1[j])

                delete_position.append(j)
                flag = 1           # // P[inx] cluster formed

            if correlValue >= correl_threshold and (distance_threshold > 0 and distance < distance_threshold):  # distance is also compared. distance_threshold is threshold value to be supplied wisely
                # print("******************************************** Found *******************************")
                # print("i=", i, " and j=", j, " :  Distance = ", distance, "   and   correlation = ", correlValue)
                performance_prune_count += 1
                if performance_prune_count <= maximum_ode_prune_factor:   # Just this line helps in pruning same segments for performance of ODE computaion
                    # P[inx].extend(res1[j])      # //pushing in the same cluster i for the first 50 segments
                    # print("performance_prune_count=",performance_prune_count)
                    mode.append(res1[j])

                delete_position.append(j)
                flag = 1           # // P[inx] cluster formed

            j = j + 1

        P.append(mode)  # creating the list of modes, with each mode as a list of segments

        # for all delete_position now delete list and update for next iterations
        for val in reversed(delete_position):
            f_ode1.pop(val)
            t_ode1.pop(val)
            res2.pop(val)
        count = len(f_ode1)   # //update the new length of the segments
        f_ode = f_ode1        # //update the new list of ODE data after clustering above
        t_ode = t_ode1
        res1 = res2
        i = i + 1  # reset for next cluster
        performance_prune_count = 0  # reset for next cluster

    # print("len(P) = ", len(P))
    print("CLUSTERING: Distance[min,max] = [", min_distance," , ", max_distance,"]")
    print("CLUSTERING: Correlation[min,max] = [", min_correl, " , ", max_correl, "]")

    # P, G = get_desired_clusters(P, A, b1)
    G = get_desired_clusters(P, A, b1)

    return P, G

