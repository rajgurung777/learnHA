"""
This module is used to parse the list of trajectories structure to construct structures suitable for our algorithm.
"""
import numpy as np

def preprocess_trajectories(list_of_trajectories):
    """
    Converts list of trajectories into a single trajectory.
    We do this conversion in order to avoid discarding 2M data-points from each trajectory during our segmentation
    process (see function diff_method_backandfor() in module compute_derivatives.py).

    :param
        list_of_trajectories: Each element of the list is a trajectory. A trajectory is a 2-tuple of (time, vector),

        where

        time: is a list having a single item. The item is the sampling time, stored as a numpy.ndarray having structure as
           (rows, ) where rows is the number of sample points. The dimension cols is empty meaning a single dim array.
        vector: is a list having a single item. The item is a numpy.ndarray with (rows,cols), rows indicates the number of
           points and cols as the system's dimension. The dimension is the total number of variables in the trajectories
           including both input and output variables.

    :return:
        The lists t_list and y_list containing time and vector as (t_list, y_list) pair and positions.
        Where
        t_list: a single-item list whose item is a numpy.ndarray containing time-values as a concatenated list.
        y_list: a single-item list whose item is a numpy.ndarray containing vector of values (of input and output) as a
            concatenated list of trajectories.
    """

    t_list = []
    y_list = []
    posi = []
    position = []

    if len(list_of_trajectories) == 1:  # when list_of_trajectories is a single trajectory
        t_list, y_list = list_of_trajectories[0]
        posi.append(0)  # start position
        # print("len of t_list =", len(t_list))
        # print("len of t_list[0] =", len(t_list[0]))
        posi.append(len(t_list[0]))
        position.append(posi)
    else:
        t_list, y_list, position = convert_trajectories_to_single_list(list_of_trajectories)

    return t_list, y_list, position


def convert_trajectories_to_single_list(list_of_trajectories):
    '''
    This function performs the actual conversion of the list of trajectories into a single trajectory.

    :param list_of_trajectories: Each element of the list is a trajectory.
                                 A trajectory is a 2-tuple of (time, vector), where
    :time: is a list having a single item. The item is the sampling time, stored as a numpy.ndarray having structure as
           (rows, ) where rows is the number of sample points. The dimension cols is empty meaning a single dim array.
    :vector: is a list having a single item. The item is a numpy.ndarray with (rows,cols), rows indicates the number of
           points and cols as the system's dimension. The dimension is the total number of variables in the trajectories
           including both input and output variables.

    :return: the value pair (time and vector) where
        t_list: a single-item list whose item is a numpy.ndarray containing time-values as a concatenated list
        y_list: a single-item list whose item is a numpy.ndarray containing vector of values (of input and output) as a
                concatenated list of trajectories.
    '''

    t_list = []
    y_list = []
    position = []
    posi = []
    start_posi = 0

    #  ****************************************
    # Create np.array for the first time with the correct dimensions
    trajectory = list_of_trajectories[0]
    t_list_per_traj, y_list_per_traj = trajectory
    temp_t_array_all = t_list_per_traj[0]    # get the time array
    temp_y_array_all = y_list_per_traj[0]  # get the vector array
    total_trajectories = 1
    #  ****************************************

    for traj in list_of_trajectories:
        t_list_per_traj, y_list_per_traj = traj # each traj is a two-tuple containing a list of single item and
                                                # the item is np.array data type
        temp_t_array = t_list_per_traj[0]    # get the time array
        temp_y_array = y_list_per_traj[0]  # get the vector array

        if total_trajectories != 1: # for ==1 we have done outside
            temp_t_array_all = np.concatenate([temp_t_array_all, temp_t_array]) # appending the array in to single dim
            temp_y_array_all = np.vstack([temp_y_array_all, temp_y_array]) # appending the array

        # Computing position needed throughout the algorithm
        posi.append(start_posi)  # the start position of a trajectory
        data_size = len(t_list_per_traj[0])
        end_posi = start_posi + data_size - 1 # minus 1 because of zero-based indexing
        posi.append(end_posi)

        start_posi = end_posi + 1 # +1 to start the next indexing sequence
        position.append(posi)
        posi = []
        total_trajectories = total_trajectories + 1

    t_list.append(temp_t_array_all) # converting the array back to list containing a single item
    y_list.append(temp_y_array_all)
    '''
    print("t_list is ", t_list)
    print("shape of t_list=", t_list[0].shape)
    print("type of t_list=", type(t_list[0]))
    print("y_list is ", y_list)
    print("shape of y_list=", y_list[0].shape)
    print("type of y_list=", type(y_list[0]))
    '''

    print("position = ", position)

    return t_list, y_list, position
