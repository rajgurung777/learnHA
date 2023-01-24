'''
Connecting points for inferring transitions

'''

def create_connecting_points(P, position, segmentedTrajectories):
    '''
    Determine connecting points from the segmented trajectories with respect to clusters
    :return:
    '''
    '''
    # ***************************************** Start of the code *************************************************
    # Amit: Now implementing the idea to search start and end position/point of segments that are on either modes.
    #   we plan to use this points to determine assignment and guard between the modes
    #   We use oneVsOne approach, where every Mode can have a transition to every other Mode (both in forward and backward direction)
    Todo: When n=number_of_modes > 1 but for single mode system this will not work
    for i = 1 to n      //forward transition for each cluster-i to every other cluster-j
        for j= (i+1) to n
        // ******** code below is for forward-transition 
            for t = 1 to len(trajectories)  //loop all trajectories
                for g=1 to len(segments) - 1    //last start-pt is compared with the previous end-pt
                    if end_g \in C_i        //pt end_g present in C_i
                        if start_{g+1} \in C_j       //and pt start_{g+1} present in C_j
                            append(end_g , start_{g+1}) in transition(i,j)
        // ******** code below is for the back-transition 
            for t = 1 to len(trajectories)  //loop all trajectories
                for g=1 to len(segments) - 1    //last start-pt is compared with the previous end-pt
                    if end_g \in C_j        //pt end_g present in C_j
                        if start_{g+1} \in C_i       //and pt start_{g+1} present in C_j
                            append(end_g , start_{g+1}) in transition(j,i)             
    '''

    cluster_len = len(P)
    traj_size = len(position)
    # print("len(position)/traj_size = ", traj_size)
    # Below computes connecting points when the number of clusters > 1. But not for single mode system
    data_points = []  # Structure containing [src, dest, list of connecting-points]
    for i in range(0, cluster_len):
        for j in range(i, cluster_len):  # modified j in range(i, cluster_len) from i+1
            # Code for Forward-Transitions
            data_points_per_trans = []
            for t in range(0, traj_size):  # Loop for all trajectories
                segment_size = len(segmentedTrajectories[t])  # total number of segments in each trajectory
                for g in range(0, segment_size - 1):
                    # last start-point is compared with previous end-point
                    # Now we have pre_end_posi therefore, end_post is index [2] end_posi=segmentedTrajectories[t][g][1]
                    # [1] is the end-pt of the trajectory t and segment g
                    end_posi = segmentedTrajectories[t][g][2]  # [2] is the end-pt of the trajectory t and segment g
                    pre_end_posi = segmentedTrajectories[t][g][1]  # [1] is the pre-end-pt of the trajectory t and segment g
                    if end_posi in P[i]:  # TRUE
                        start_posi = segmentedTrajectories[t][g + 1][
                            0]  # [0] is the start-pt of the trajectory t and segment g+1
                        if start_posi in P[j]:  # if this also returns TRUE
                            # print ("Store. (end,start)=(", end_posi,",",start_posi,") in transition(i,j)=(", i, ",",j,") \n")
                            # data_points.append([i, j, [end_posi, start_posi]])
                            # Now we apppend 3 points, data_points_per_trans.append([end_posi, start_posi])
                            data_points_per_trans.append([pre_end_posi, end_posi, start_posi])
            if len(data_points_per_trans) > 0:
                data_points.append([i, j, data_points_per_trans])
                print("[src, dest, total-points] = [", i, " , ", j, " , ", len(data_points_per_trans), "]")

            # Code for Backward-Transitions
            if i != j:  # loop is to be done only one
                data_points_per_trans = []
                for t in range(0, traj_size):  # Loop for all trajectories
                    segment_size = len(segmentedTrajectories[t])  # Length of each segmented trajectory
                    for g in range(0, segment_size - 1):  # last start-point is compared with previous end-point
                        # Now index of end_posi shifted to [2]: end_posi = segmentedTrajectories[t][g][1]  # [1] is the end-pt of the trajectory t and segment g
                        end_posi = segmentedTrajectories[t][g][2]  # [2] is the end-pt of the trajectory t and segment g
                        pre_end_posi = segmentedTrajectories[t][g][1]  # [1] is the pre-end-pt of the trajectory t and segment g
                        if end_posi in P[j]:  # TRUE
                            start_posi = segmentedTrajectories[t][g + 1][
                                0]  # [1] is the end-pt of the trajectory t and segment g
                            if start_posi in P[i]:  # if this also returns TRUE
                                # print("Store. (end,start)=(", end_posi, ",", start_posi,") in transition(j,i)=(", j, ",", i, ") \n")
                                # data_points.append([j, i, [end_posi, start_posi]])
                                # now we have 3points: data_points_per_trans.append([end_posi, start_posi])
                                data_points_per_trans.append([pre_end_posi, end_posi, start_posi])
                if len(data_points_per_trans) > 0:
                    data_points.append([j, i, data_points_per_trans])
                    print("[src, dest, total-points] = [", j, " , ", i, " , ", len(data_points_per_trans), "]")
    # print("\nLength of data points = ", len(data_points))
    # print("data points are ", data_points)  # forward and backward transition points are all here
    # **************************************** End of the code *****************************************************
    return data_points

