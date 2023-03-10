"""
Contains modules related to the segmentation process

"""

from sklearn import linear_model

from infer_ha.utils.util_functions import rel_diff, matrowex

def two_fold_segmentation(A, b1, b2, ytuple, size_of_input_variables, method, ep=0.01):
    """
    Main idea: (Step-1) We compare backward and forward derivatives at each point of the trajectories. Near the boundary
    of these points, their relative difference will be high. We compute the backward derivatives and compare them with
    their previous backward derivatives to determine the exact point. If they are identical, the point is considered to
    be in the same segment. When the relative difference between the current and previous backward derivatives is
    different, we consider that point as the jump point and record the current point as the last point of the segment.
    (Step-2) After the current point in (Step-1) has been found, we can compute the relative difference of forward
    derivatives of the current point and the next point, and when it is high/dissimilar, we can drop these points. When
    the relative differences are low/similar, we can consider them in the next segment (by marking them as the
    start-point of the next segment).
    //Step-2 can be ignored, in which case only the end-point is closer to the exact boundary, and the start-point will
    still remain far as in the original segmentation algorithm (of Jin et al.).
    Note that we project only the output variables from the trajectory and perform segmentation on this projected data.
    Specifically, we ignore the effect of the input variables during segmentation.

    :param A: stores for every point of a trajectory the coefficients of the monomial terms obtained using the \Phi
         function (or the mapping function) as mention in Jin et al. paper.
    :param b1: the derivatives of each point computed using the backward version of BDF.
    :param b2: the derivatives of each point computed using the forward version of BDF.
    :param ytuple: is a list of two tuple with the first item as zero and the second the size of the total points.
    :param size_of_input_variables: total number of input variables present in the trajectories.
    :param method: clustering method selected by the user (options dtw, dbscan, etc.)
    :param ep: Maximal error toleration value. In the paper, \Epsilon_{FwdBwd}
    :return: The following
        res: is a list of the segmented trajectory (positions). Each item of the list res contains a list of values
        which are positions of points of trajectories. The size of the list res is the total number of segments
        obtained. In the calling function, if 'res' is used, this considers both Step-1 and Step-2. In this segmentation
        process, both a segment's start point and end point are near the jump point (especially the start point).
        drop: list of points/positions that are dropped during segmentation process.
        clfs: is a list. Each item of the list clfs is a list that holds the coefficients (obtained using linear regression)
           of the ODE of each segment of the segmented trajectories.
        res_modified: is a list of the segmented trajectory (positions). Each item of the list res contains a list of values
        which are positions of points of trajectories. The size of the list res is the total number of segments
        obtained.  In the calling function, if 'res_modified' is used, this ignores Step 2. In this segmentation
        process, only the end-point of a segment is near the jump point, but the start-point is far as in the original
        approach of Jin et al.
        Observed that using res in most cases does not help in computing good ODE equations (and also in clustering).

    """


    highDifference = 0.1  # should actually be 0.9
    lowDifference = 0.01   # I can also set 0.1     In the paper, \Epsilon_{Bwd}
    relDiff_backward = 0.0
    relDiff_foward = 0.0
    next_low = 0
    res = []
    res_modified = []
    # print("input size =", size_of_input_variables, "  output size =", size_of_output_variables)
    print("len(ytuple) =", len(ytuple))
    for i in range(0, len(ytuple)):
        (l1, l2) = ytuple[i]
        cur_pos = l1  # start position
        max_id = l2  # end position of the entire data
        low = cur_pos
        start_low = cur_pos
        while True:
            high = cur_pos
            position_diffValue = []
            highest_pos = []
            while high < max_id:
                # print("b1[high]:",b1[high], "   and b1[high,2] =", b1[high,2], "   and b1[high, 1:] =", b1[high, 1:])
                # diff_val = rel_diff(b1[high,2], b2[high,2])  # print("pos:",high,"  diff-val:",diff_val)
                diff_val = rel_diff(b1[high, size_of_input_variables:], b2[high, size_of_input_variables:]) # ignore input-variables note: zero-based indexing
                # print("pos:",high,"  (b1-b2) diff-val:",diff_val)
                if diff_val < ep:
                    high += 1
                else:
                    # print("pos:", high, "  diff-val:", diff_val)
                    break
            # I have here low and high but I want high to take some extra near the actual-boundary
            near_high = high
            good_high = high # this will be improved
            while high < max_id:  # moving high further.
                diff_val = rel_diff(b1[high, size_of_input_variables:], b2[high, size_of_input_variables:])
                relDiff_backward = rel_diff(b1[high, size_of_input_variables:], b1[(high - 1), size_of_input_variables:]) # compute relative difference between current and previous
                # ------------------ Below we also compute relative-difference of forward version --------------------
                if (high < (max_id - 1)):
                    relDiff_foward = rel_diff(b2[high, size_of_input_variables:], b2[(high + 1), size_of_input_variables:])
                else:
                    relDiff_foward = rel_diff(b2[high, size_of_input_variables:], b2[(high - 1), size_of_input_variables:])   # just a fix for the last point
                # print("pos:", high, "   (b1-b2) diff-val:", diff_val, "  relDiff_backward: ", relDiff_backward, "  relDiff_foward: ", relDiff_foward)
                # -----------------------------------------------------------------------------------
                if diff_val >= ep:  # high difference.    This will detect all boundary points
                    position_diffValue.append([high, relDiff_backward, relDiff_foward]) # recording position and diff-value
                    high += 1
                else:  # low difference so same segment
                    break
            if len(position_diffValue) != 0:
                arr_len = len(position_diffValue)
                next_low_search = 0
                for index in range(0, len(position_diffValue)):
                    value_position = position_diffValue[index][0]
                    value_backward = position_diffValue[index][1]
                    value_forward = position_diffValue[index][2]
                    if (next_low_search == 0) and (value_backward >= lowDifference): # found the exact change-point or the boundary point
                        good_high = value_position
                        next_low_search = 1
                        continue  # skipping this position for next_low
                    if (next_low_search == 1) and (value_forward < lowDifference): # found the start points for the next segment
                       next_low = value_position
                       break    # found both good_high and next_low so break the for loop from check forward as they will all satisfy
                # print("good_high=", good_high, "  next_low=", next_low, "  but near_low=", near_high)

            if good_high - low >= 5:
                res.append(list(range(low, near_high)))
                res_modified.append(list(range(low, good_high)))  # ignoring (Step 2)
                # res_modified.append(list(range(start_low, good_high))) # Consider (Step 2): had issue in ODE formation
            if high == max_id:
                break
            cur_pos = high  # the position from where the next iteration should continue as upto high is already done
            low = high  # next_low   ignoring (Step 2)
            # low = next_low  # consider (Step 2)
            start_low = next_low

    all_pts = set()
    for lst in res:
        all_pts = all_pts.union(set(lst))
    drop = list(set(range(max_id)) - all_pts)  # set(range(max_id)): creates set from 0 to max_id. operation - all_pts (all_pts contains the segemented set)

    all_pts_modified = set()
    for lst in res_modified:
        all_pts_modified = all_pts_modified.union(set(lst))
    drop = list(set(range(max_id)) - all_pts_modified)  # set(range(max_id)): creates set from 0 to max_id. operation - all_pts (all_pts contains the segemented set)

    # Fit each segment
    clfs = []

    # cluster_by_DTW = True
    # if cluster_by_DTW == False: # for DTW we do not need clfs computation at this stage but for dbscan/linearpiece we need
    if method != "dtw":  # for DTW we do not need clfs computation at this stage but for dbscan/linearpiece we need
        # print ("len of res", len(res))
        for lst in res:
            # print("List in res is ", lst)
            Ai = matrowex(A, lst)
            Bi = matrowex(b1, lst)
            # print("Bi is ", Bi)
            # print("Ai is ", Ai)
            clf = linear_model.LinearRegression(fit_intercept=False)
            # print ("Testing 1")
            clf.fit(Ai, Bi)
            # print("clf is ", clf.coef_)
            # print("Testing 2")
            clfs.append(clf)


    return res, drop, clfs, res_modified


def segmented_trajectories(clfs, res, position, filter_last_segment=1):
    """
    Perform segmentation of trajectories to create data structure containing segment positions. This process helps in
    keeping track of the connected segments. This information is later used to infer transitions of an HA.
    Note: We delete the last segmented portion of all the trajectories, assuming they are short segments and does not
    help learning. In fact in some cases, they can create problem during clustering (comparing segments for similarity).

    Note: Should not delete the segment if a trajectories has only one segment per trajectory.


    :param
        clfs: is a list. Each item of the list clfs is a list that holds the coefficients (obtained using linear
        regression) of the ODE of each segment of the segmented trajectories.
    :param
        res: is a list of the segmented trajectory (having positions). Each item of the list res contains a list of values
        which are positions of points of trajectories. The size of the list res is the total number of segments
        obtained.
    :param
        position: is a list of position data structure. Each position is a pair (start, end) position of a trajectory.
        For instance, the first item of the list is [0, 100] means that the trajectory has 101 points. The second item
        as [101, 300], meaning the second trajectory has 200 points. Note that all the trajectories are concatenated.
    :param
        filter_last_segment: is a boolean value. 1 to enable the filter condition for removing the last segment and
        0 for not removing the last segment.
    :return: The following data structures
        segmentedTrajectories: the required position data structures
        res: the modified res, after deleting the last segment per trajectory when user enabled the filter option.
        clfs: the modified clfs, after deleting the last segment per trajectory when user enabled the filter option.

    """

    # print("len(position) =", len(position))
    # print("len(res) =", len(res))

    total_trajectories = len(position) # gives the total number of trajectories supplied as input
    total_segments = len(res)  # total segments after segmentation process
    found_single_segment_per_trajecotry = 0 # not found so deletion is required.
    if total_segments == total_trajectories:    # meaning each trajectory has a single segment (continuous single mode system)
        # a simple example is "circle" model or a five-dimensional system in XSpeed or SpaceEx
        found_single_segment_per_trajecotry = 1    # found single-segment-per-trajectory, so do not delete segment

    # **************************** Start of segmentation **************************************************
    # We create a list of segments positions of the form [start, end] for each trajectory to .

    seg = []  # contain start and end position for each segment
    segments_per_traj = []  # contain list of all start and end position for each trajectory. Note each trajectory has the overall start and end position value
    segmentedTrajectories = []  # contain list of all segments of all the trajectories.
    traj_id = 1
    seg = position[traj_id - 1]  # starting trajectory=0
    start_trajectory_pos = seg[0]
    end_trajectory_pos = seg[1]
    traj_segs = []
    del_index = 0 # index pointer for each segment in res
    del_res_indices = []    # store the list of indices of res to be deleted
    for s in res:
        start_segment_pos = s[0]  # start position of the segment
        end_segment_pos = s[len(s) - 1]  # end position of the segment
        pre_end_segment_pos = s[len(s) - 2]  # pre-end position of the segment (2nd last position)
        traj_segs = []
        traj_segs.append(start_segment_pos)
        traj_segs.append(pre_end_segment_pos)
        traj_segs.append(end_segment_pos)
        if (start_segment_pos >= start_trajectory_pos) and (end_segment_pos <= end_trajectory_pos):
            segments_per_traj.append(traj_segs)
        else:  # meaning if any of the above condition fails. I am assuming all segments will be segmented trajectory-wise
            # that is, no segments will overlap.
            del_res_indices.append(del_index - 1)   # stores the previous index for deletion
            # delete when single_segment_per_trajectory not Found and user selected the option filter_last_segment
            if (found_single_segment_per_trajecotry == 0) and (filter_last_segment == 1):
                segments_per_traj = segments_per_traj[ : -1]    # deletes the last segment before creating segmented-trajectories
            segmentedTrajectories.append(segments_per_traj)

            segments_per_traj = []
            traj_id += 1
            seg = position[traj_id - 1]  # starting trajectory=0
            start_trajectory_pos = seg[0]
            end_trajectory_pos = seg[1]
            segments_per_traj.append(traj_segs)  # previously created seg

        del_index += 1

    del_res_indices.append(del_index - 1)  # stores the previous index for deletion
    # delete when single_segment_per_trajectory not Found and user selected the option filter_last_segment
    if (found_single_segment_per_trajecotry == 0) and (filter_last_segment == 1):
        segments_per_traj = segments_per_traj[ : -1]   # deletes the last segment before creating segmented-trajectories
    segmentedTrajectories.append(segments_per_traj)  # the last segmented trajectory
    # ************************************ End of segmentation ******************************************
    # print("segmentedTrajectories is ", segmentedTrajectories)
    # print("list of position to be delted are ", del_res_indices)

    cluster_by_DTW = True   # Todo for running DTW Clustering algorithm for paper

    # delete when single_segment_per_trajectory not Found and user selected the option filter_last_segment
    if (found_single_segment_per_trajecotry == 0) and (filter_last_segment == 1):
        for pos in reversed(del_res_indices):
            res.pop(pos)
            if cluster_by_DTW == False: # for DTW clustering we do not have clfs data. So skipping this line saves time
                clfs.pop(pos)

    return segmentedTrajectories, res, clfs


"""
Testing different approach. Currently we are using two_fold_segmentation.
The list of functions that can removed and currently not in use are: 
    segment_and_fit
    segment_and_fit_Modified_one
    segment_and_fit_Modified_two
    two_fold_segmentation_new: Nearly no dropping of points 
"""
def segment_and_fit(A, b1, b2, ytuple, ep=0.01):
    # Segmentation
    # This function is used to implement the simple segmentation Algorithm-1 in the paper by Jin et al.
    res = []
    for i in range(0, len(ytuple)):
        (l1, l2) = ytuple[i]
        cur_pos = l1    # start position
        max_id = l2    # end position of the entire data
        while True:
            low = cur_pos
            while low < max_id and rel_diff(b1[low], b2[low]) >= ep:
                low += 1
            if low == max_id:
                break
            high = low
            while high < max_id and rel_diff(b1[high], b2[high]) < ep:
                high += 1
            if high - low >= 5:
                res.append(list(range(low, high)))
            cur_pos = high
    all_pts = set()
    for lst in res:
        all_pts = all_pts.union(set(lst))
    drop = list(set(range(max_id)) - all_pts)   # set(range(max_id)): creates set from 0 to max_id. operation - all_pts (all_pts contains the segemented set)

    # Fit each segment
    clfs = []
    # print ("len of res", len(res))
    for lst in res:
        # print("List in res is ", lst)
        Ai = matrowex(A, lst)
        Bi = matrowex(b1, lst)
        # print("Bi is ", Bi)
        # print("Ai is ", Ai)
        clf = linear_model.LinearRegression(fit_intercept=False)
        # print ("Testing 1")
        clf.fit(Ai, Bi)
        # print("clf is ", clf.coef_)
        # print("Testing 2")
        clfs.append(clf)

    return res, drop, clfs


def segment_and_fit_Modified_one(A, b1, b2, ytuple, ep=0.01):
    '''
    # Segmentation
    # Modified the original segmentation to add few more boundary points near the START poistions
    of each segment.
    Observation: Maybe this has affected the ODE and so the clustering (DBSCAN) algo failed to
    properly cluster (compared to its original version).
    Thus, connecting points could not be obtained when many clusters where dropped

    '''

    res = []
    for i in range(0, len(ytuple)):
        (l1, l2) = ytuple[i]
        cur_pos = l1    # start position
        max_id = l2    # end position of the entire data
        while True:
            low = cur_pos
            position_diffValue =[]
            highest_pos = []
            while low < max_id:
                diff_val = rel_diff(b1[low], b2[low])
                # print("pos:",low,"  diff-val:",diff_val)
                if diff_val >= ep:  # high difference
                    position_diffValue.append([low, diff_val])
                    low += 1
                else:       # low difference so same segment
                    break

            if low == max_id:
                break
            high = low  # let high start from where boundary points are discarded according to the old approach
            if len(position_diffValue) != 0:
                highest_pos = max(position_diffValue, key=lambda x: x[1])
                low = highest_pos[0] # Making the highest difference, ie., THE Boundary-point the point of segmentation
            print("Highest position=", highest_pos)

            while high < max_id:
                diff_val = rel_diff(b1[high], b2[high])
                # print("pos:",high,"  diff-val:",diff_val)
                if diff_val < ep:
                    high += 1
                else:
                    break
            # appended when stopped with high/end-point
            if high - low >= 5:
                res.append(list(range(low, high)))
            cur_pos = high
    all_pts = set()
    for lst in res:
        all_pts = all_pts.union(set(lst))
    drop = list(set(range(max_id)) - all_pts)   # set(range(max_id)): creates set from 0 to max_id. operation - all_pts (all_pts contains the segemented set)

    # Fit each segment
    clfs = []
    # print ("len of res", len(res))
    for lst in res:
        # print("List in res is ", lst)
        Ai = matrowex(A, lst)
        Bi = matrowex(b1, lst)
        # print("Bi is ", Bi)
        # print("Ai is ", Ai)
        clf = linear_model.LinearRegression(fit_intercept=False)
        # print ("Testing 1")
        clf.fit(Ai, Bi)
        # print("clf is ", clf.coef_)
        # print("Testing 2")
        clfs.append(clf)

    return res, drop, clfs


def segment_and_fit_Modified_two(A, b1, b2, ytuple, ep=0.01):
    '''
    # Segmentation
    # Amit: Modified the original segmentation to add few more boundary points near the END poistions
    of each segment. The middle point between the first and the last point whose relative-differnce is below ep.
    Observation:

    '''

    res = []
    res_modified = []
    for i in range(0, len(ytuple)):
        (l1, l2) = ytuple[i]
        cur_pos = l1  # start position
        max_id = l2  # end position of the entire data
        low = cur_pos
        while True:
            high = cur_pos
            position_diffValue = []
            highest_pos = []
            while high < max_id:
                diff_val = rel_diff(b1[high], b2[high])  # print("pos:",high,"  diff-val:",diff_val)
                if diff_val < ep:
                    high += 1
                else:
                    break
            # I have here low and high but I want high to take some extra near the actual-boundary
            near_high = high
            good_high = high # this will be improved
            while high < max_id:  # moving high further
                diff_val = rel_diff(b1[high], b2[high])
                print("pos:", high, "  diff-val:", diff_val)
                if diff_val >= ep:  # high difference
                    position_diffValue.append([high, diff_val])
                    high += 1
                else:  # low difference so same segment
                    break
            if len(position_diffValue) != 0:
                arr_len = len(position_diffValue)
                firstPosition_inRange = position_diffValue[0][0]
                lastPosition_inRange = position_diffValue[arr_len-1][0]
                print("arr_len=", arr_len, "  firstPosition_inRange=", firstPosition_inRange, "   lastPosition_inRange=",lastPosition_inRange)
                mid_pos = int((lastPosition_inRange + firstPosition_inRange) / 2)
                print("mid_pos=", mid_pos)
                good_high = mid_pos
                # highest_pos = max(position_diffValue, key=lambda x: x[1])
                # good_high = highest_pos[0]  # Making the highest difference, ie., THE Boundary-point the point of segmentation
            print("Highest position=", highest_pos, "   mid_position=", mid_pos)
            if good_high - low >= 5:
                res.append(list(range(low, near_high)))
                res_modified.append(list(range(low, good_high)))
            if high == max_id:
                break
            cur_pos = high
            low = high # next_low

    all_pts = set()
    for lst in res:
        all_pts = all_pts.union(set(lst))
    drop = list(set(range(max_id)) - all_pts)  # set(range(max_id)): creates set from 0 to max_id. operation - all_pts (all_pts contains the segemented set)

    all_pts_modified = set()
    for lst in res_modified:
        all_pts_modified = all_pts_modified.union(set(lst))
    drop = list(set(range(max_id)) - all_pts_modified)  # set(range(max_id)): creates set from 0 to max_id. operation - all_pts (all_pts contains the segemented set)

    # Fit each segment
    clfs = []
    # print ("len of res", len(res))
    for lst in res:
        # print("List in res is ", lst)
        Ai = matrowex(A, lst)
        Bi = matrowex(b1, lst)
        # print("Bi is ", Bi)
        # print("Ai is ", Ai)
        clf = linear_model.LinearRegression(fit_intercept=False)
        # print ("Testing 1")
        clf.fit(Ai, Bi)
        # print("clf is ", clf.coef_)
        # print("Testing 2")
        clfs.append(clf)

    return res, drop, clfs, res_modified

def two_fold_segmentation_new(A, b1, b2, ytuple, size_of_input_variables, method, ep=0.01):
    """
    Main idea: (Step-1) We compare backward and forward derivatives at each point of the trajectories. Near the boundary
    of these points, their relative difference will be high. We compute the backward derivatives and compare them with
    their previous backward derivatives to determine the exact point. If they are identical, the point is considered to
    be in the same segment. When the relative difference between the current and previous backward derivatives is
    different, we consider that point as the jump point and record the current point as the last point of the segment.
    (Step-2) After the current point in (Step-1) has been found, we can compute the relative difference of forward
    derivatives of the current point and the next point, and when it is high/dissimilar, we can drop these points. When
    the relative differences are low/similar, we can consider them in the next segment (by marking them as the
    start-point of the next segment).
    //Step-2 can be ignored, in which case only the end-point is closer to the exact boundary, and the start-point will
    still remain far as in the original segmentation algorithm (of Jin et al.).
    In this new implementation, we again try to take the exact point of jump and assign the start-point also close.

    Note that we project only the output variables from the trajectory and perform segmentation on this projected data.
    Specifically, we ignore the effect of the input variables during segmentation.

    :param A: stores for every point of a trajectory the coefficients of the monomial terms obtained using the \Phi
         function (or the mapping function) as mention in Jin et al. paper.
    :param b1: the derivatives of each point computed using the backward version of BDF.
    :param b2: the derivatives of each point computed using the forward version of BDF.
    :param ytuple: is a list of two tuple with the first item as zero and the second the size of the total points.
    :param size_of_input_variables: total number of input variables present in the trajectories.
    :param method: clustering method selected by the user (options dtw, dbscan, etc.)
    :param ep: Maximal error toleration value. In the paper, \Epsilon_{FwdBwd}
    :return: The following
        res: is a list of the segmented trajectory (positions). Each item of the list res contains a list of values
        which are positions of points of trajectories. The size of the list res is the total number of segments
        obtained. In the calling function, if 'res' is used, this considers both Step-1 and Step-2. In this segmentation
        process, both a segment's start point and end point are near the jump point (especially the start point).
        drop: list of points/positions that are dropped during segmentation process.
        clfs: is a list. Each item of the list clfs is a list that holds the coefficients (obtained using linear regression)
           of the ODE of each segment of the segmented trajectories.
        res_modified: is a list of the segmented trajectory (positions). Each item of the list res contains a list of values
        which are positions of points of trajectories. The size of the list res is the total number of segments
        obtained.  In the calling function, if 'res_modified' is used, this ignores Step 2. In this segmentation
        process, only the end-point of a segment is near the jump point, but the start-point is far as in the original
        approach of Jin et al.
        Observed that using the true/exact start-point and end-point (instead of boundary points) does not help in
        computing good ODE equations (and also in clustering). We observed that adding the exact start-point makes the
        ODE inference incorrect.

    """



    highDifference = 0.1  # should actually be 0.9
    lowDifference = 0.01   # I can also set 0.1     In the paper, \Epsilon_{Bwd}
    relDiff_backward = 0.0
    relDiff_foward = 0.0
    next_low = 0
    res = []
    res_modified = []
    # print("input size =", size_of_input_variables, "  output size =", size_of_output_variables)
    print("len(ytuple) =", len(ytuple))
    for i in range(0, len(ytuple)):
        (l1, l2) = ytuple[i]
        cur_pos = l1  # start position
        max_id = l2  # end position of the entire data
        low = cur_pos
        # start_low = cur_pos
        while True:
            high = cur_pos
            position_diffValue = []
            highest_pos = []
            while high < max_id:
                # print("b1[high]:",b1[high], "   and b1[high,2] =", b1[high,2], "   and b1[high, 1:] =", b1[high, 1:])
                # diff_val = rel_diff(b1[high,2], b2[high,2])  # print("pos:",high,"  diff-val:",diff_val)
                diff_val = rel_diff(b1[high, size_of_input_variables:], b2[high, size_of_input_variables:]) # ignore input-variables note: zero-based indexing
                # print("pos:",high,"  (b1-b2) diff-val:",diff_val)
                if diff_val < ep:
                    high += 1
                else:
                    # print("pos:", high, "  diff-val:", diff_val)
                    break
            # I have here low and high but I want high to take some extra near the actual-boundary
            near_high = high
            good_high = high # this will be improved
            while high < max_id:  # moving high further.
                diff_val = rel_diff(b1[high, size_of_input_variables:], b2[high, size_of_input_variables:])
                relDiff_backward = rel_diff(b1[high, size_of_input_variables:], b1[(high - 1), size_of_input_variables:]) # compute relative difference between current and previous

                if diff_val >= ep:  # high difference.    This will detect all boundary points
                    position_diffValue.append([high, relDiff_backward]) # recording position and diff-value
                    high += 1
                else:  # low difference so same segment
                    break
            if len(position_diffValue) != 0:
                arr_len = len(position_diffValue)
                next_low_search = 0
                for index in range(0, len(position_diffValue)):
                    value_position = position_diffValue[index][0]
                    value_backward = position_diffValue[index][1]
                    if (value_backward >= lowDifference): # found the exact change-point or the boundary point
                        good_high = value_position  # this is the point where difference found
                        good_high = value_position - 1 # Thus -1 point is the last point in the segment before jump-reset
                        next_low = value_position
                        break    # found both good_high and next_low so break the for loop from check forward as they will all satisfy
                print("good_high=", good_high, "  next_low=", next_low, "  but near_low=", near_high)

            if good_high - low >= 5:
                res.append(list(range(low, near_high)))
                res_modified.append(list(range(low, good_high)))  # ignoring (Step 2)
                # res_modified.append(list(range(start_low, good_high))) # Consider (Step 2): had issue in ODE formation
            if high == max_id:
                break
            cur_pos = high  # the position from where the next iteration should continue as upto high is already done
            # low = high  # next_low   ignoring (Step 2)
            low = next_low  # consider (Step 2). Note irrespective of all possibilities we assume next_low till high all points lie in the next segment

    all_pts = set()
    for lst in res:
        all_pts = all_pts.union(set(lst))
    drop = list(set(range(max_id)) - all_pts)  # set(range(max_id)): creates set from 0 to max_id. operation - all_pts (all_pts contains the segemented set)

    all_pts_modified = set()
    for lst in res_modified:
        all_pts_modified = all_pts_modified.union(set(lst))
    drop = list(set(range(max_id)) - all_pts_modified)  # set(range(max_id)): creates set from 0 to max_id. operation - all_pts (all_pts contains the segemented set)

    # Fit each segment
    clfs = []

    # cluster_by_DTW = True
    # if cluster_by_DTW == False: # for DTW we do not need clfs computation at this stage but for dbscan/linearpiece we need

    if method != "dtw":  # for DTW we do not need clfs computation at this stage but for dbscan/linearpiece we need
        # print ("len of res", len(res))
        for lst in res:
            # print("List in res is ", lst)
            Ai = matrowex(A, lst)
            Bi = matrowex(b1, lst)
            # print("Bi is ", Bi)
            # print("Ai is ", Ai)
            clf = linear_model.LinearRegression(fit_intercept=False)
            # print ("Testing 1")
            clf.fit(Ai, Bi)
            # print("clf is ", clf.coef_)
            # print("Testing 2")
            clfs.append(clf)

    return res, drop, clfs, res_modified
