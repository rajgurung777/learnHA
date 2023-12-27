"""
Contains modules related to the segmentation process

"""

from sklearn import linear_model

from infer_ha.utils.util_functions import rel_diff, matrowex

def two_fold_segmentation(A, b1, b2, ytuple, Y, size_of_input_variables, method, stepM, ep_FwdBwd=0.01, ep_backward=0.1):
    """
    Main idea: (Step-1) We compare backward and forward derivatives at each point of the trajectories. Near the boundary
    of these points, their relative difference will be high. Now, we record these boundary points as the first set of
    start and end points. Next to obtain the exact point of jump, we search further from the first end point to find the
    exact point of jump. To achieve this, we compute backward derivatives of the current point and its previous point
    and compare them. If they are identical, the point is considered to be in the same segment. When the relative
    difference between the current and previous backward derivatives is high/different, we consider that a jump has
    taken and record the previous point as the last point of the segment.
    (Step-2) After the current point in (Step-1) has been found as the jump point. To improve the transition-assignment,
    we do not directly take the next point as the start-point for the next segment, instead we now compute the relative
    difference of forward derivatives between the current and next point. When this relative-difference value is <= ep.
    This current point is recorded as the start point for the next segment.
    Similarly, the next end point is computed as in Step-1.
    In this approach, we now do not drop points (except the first and last M points, where M is the step size in LMM).
    Instead, we record all the points along with two sets of point one for recording the near the boundary points and
    the other representing the exact points where jump happens. The boundary points are used for inferring coefficients
    of the ODE equation using linear regression (and may for DTW distance comparison for clustering).

    Note: We project only the output variables from the trajectory and perform segmentation on this projected data.
    Specifically, we ignore the effect of the input variables during segmentation.

    :param A: stores for every point of a trajectory the coefficients of the monomial terms obtained using the \Phi
         function (or the mapping function) as mention in Jin et al. paper.
    :param b1: the derivatives of each point computed using the backward version of BDF.
    :param b2: the derivatives of each point computed using the forward version of BDF.
    :param ytuple: is a list of two tuple with the first item as zero and the second the size of the total points.
    :param Y: contains the y_list values for all the points except the first and last M points.
    :param size_of_input_variables: total number of input variables present in the trajectories.
    :param method: clustering method selected by the user (options dtw, dbscan, etc.)
    :param stepM: is the step size M in the Linear Multi-step Methods
    :param ep_FwdBwd: Maximal error toleration value. In the paper, \Epsilon_{FwdBwd}
    :return: The following
        segmented_traj: is a list of a custom data structure consisting of segmented trajectories (positions). Each item
        of the list contains tuple of the form ([start_ode, end_ode], [start_exact, end_exact], [p_1, ... , p_n]).
        The Tuple has three items:
            (1) first, a list of two values for recording start and end points for learning ODE
            (2) second, a list of two values for recording start and end points for learning guard and assignment using
            the exact point of a jump
            (3) third, a list of values representing the position of points of the trajectories. Where p_1 and p_n
            are start_exact and end_exact points.
        clfs: is a list. Each item of the list clfs is a list that holds the coefficients (obtained using linear regression)
           of the ODE of each segment of the segmented trajectories.
        drop: list of points/positions that are dropped during segmentation process.
        The size of the list segmented_traj, is the total number of segments obtained.

    Note: Adding the exact points (especially the start-point of a segment) gives incorrect ODE coefficients. This maybe
    due to the use of BDF (backward version) for computing the derivatives which is a component in learning coefficient
    using linear regression. Note the exact points near the boundary using BDF includes some points ( <= M) from
    the previous segment. Thus, either we can drop boundary points for ODE inference or compute a mix of forward BDF
    for boundary points and backward BDF for the rest of the points to infer the coefficients of the ODE correctly.
    """

    # lowDifference = ep_backward #bball=0.9 is good  # rest set 0.01     In the paper, \Epsilon_{Bwd}
    next_low = 0
    segment = tuple()   # a segment to hold ([start_ode, end_ode], [start_exact, end_exact], [p_1, ... , p_n])
    segment_positions = []    # to hold the last item of the segment tuple
    segmented_traj = []
    # print("input size =", size_of_input_variables, "  output size =", size_of_output_variables)
    # print("len(ytuple) =", len(ytuple))
    # print("ytuple =", ytuple)
    max_id = 0  # declaring this variable, so that it can be used to create drop
    for i in range(0, len(ytuple)):
        (l1, l2) = ytuple[i]
        cur_pos = l1  # start position
        max_id = l2 - 1  # end position of the entire data.
        low = cur_pos
        near_low = cur_pos
        good_low = cur_pos
        next_good_low = cur_pos

        while True:
            high = cur_pos
            position_diffValue = []
            while high < max_id:
                # print("b1[high]:",b1[high], "   and b1[high,2] =", b1[high,2], "   and b1[high, 1:] =", b1[high, 1:])
                # diff_val = rel_diff(b1[high,2], b2[high,2])  # print("pos:",high,"  diff-val:",diff_val)
                diff_val = rel_diff(b1[high, size_of_input_variables:], b2[high, size_of_input_variables:]) # ignore input-variables note: zero-based indexing
                # print("pos:",high,"  (b1-b2) diff-val:",diff_val)
                if diff_val < ep_FwdBwd:
                    high += 1
                else:
                    # print("pos:", high, "  diff-val:", diff_val)
                    break
            # I have here low and high but I want high to take some extra near the actual-boundary
            near_high = high - 1   # This is the boundary end-point. upto (high - 1) points lie in the current segment where as high hits the guard condition.
            good_high = high - 1  # this will be improved
            next_good_low = high  # this will be improved
            while high < max_id:  # moving high further to find the end of boundary point i.e., the next start-point.

                diff_val = rel_diff(b1[high, size_of_input_variables:], b2[high, size_of_input_variables:]) # rel diff between backward and forward derivatives

                # relDiff_data_value = rel_diff(Y[high, size_of_input_variables:],
                #                 Y[(high - 1), size_of_input_variables:]) # rel diff: current and previous data-values

                relDiff_backward = rel_diff(b1[high, size_of_input_variables:], b1[(high - 1),
                                size_of_input_variables:])  # compute relative diff: current and previous

                if (high+1) == max_id:  # at the last position we do not want to have index out of bounds error
                    relDiff_forward = rel_diff(b2[high, size_of_input_variables:], b2[high,
                                size_of_input_variables:])  # the last point will not be used, so does not matter
                else:
                    relDiff_forward = rel_diff(b2[high, size_of_input_variables:], b2[(high + 1),
                                size_of_input_variables:])  # rel diff: current and next forward-derivatives

                if diff_val >= ep_FwdBwd:  # high difference.    This will detect and store all boundary points
                    # position_diffValue.append([high, relDiff_backward, relDiff_forward, relDiff_data_value]) # recording position and diff-value
                    position_diffValue.append([high, relDiff_backward, relDiff_forward])  # recording position and diff-value
                    high += 1
                else:  # low difference so same segment
                    break
                # print("high=", high, "  relDiff_Data =", relDiff_data_value, "  relDiff_Bwd=", relDiff_backward, "  relDiff_Fwd =", relDiff_forward)
                # print("pos =", high, "  diff_val =", diff_val,  "  relDiff_Bwd=", relDiff_backward, "  relDiff_Fwd =", relDiff_forward)



            if len(position_diffValue) != 0:
                # find last point and then also check if the next point is fit for the start-pt for next segment, otherwise find next correct point
                found_last_point = 0
                for index in range(0, len(position_diffValue)):
                    value_position = position_diffValue[index][0]
                    value_relDiff_backward_derivative = position_diffValue[index][1]
                    value_relDiff_forward_derivative = position_diffValue[index][2]
                    # value_relDiff_data = position_diffValue[index][3]
                    if (found_last_point == 0) and (value_relDiff_backward_derivative >= ep_backward): # found the exact change-point or the boundary point
                        good_high = value_position - 1  # the previous position is the last/end-point of the previous segment
                        next_good_low = value_position  # the current position is the start-point for the next segment.
                        break   # this will stop searching further

                    #     if (value_relDiff_forward_derivative <= ep_FwdBwd):  # both <=ep_FwdBwd and <=ep_Fwd has same task: to consider data-points are in the same segment
                    #         next_good_low = value_position  # this is a good next segment's start-point
                    #         break  # found both good_high and next_low so break for-loop
                    #     else:
                    #         found_last_point = 1
                    #         continue    # search for next segment's start-point
                    # if (found_last_point == 1) and (value_relDiff_forward_derivative <= ep_FwdBwd): # found the last point but not the next start point
                    #     next_good_low = value_position # this is a good next segment's start-point
                    #     break   # also found the next segment's start-point so break the for-loop
                    #

                # print("good_high=", good_high, "  next_good_low=", next_good_low, "  but near_low=", near_high)
            # else:  # what happens if boundary-point is also the exact point? This block will be executed
            #     print("************ This block will be exectued only once. Since we have both the check  diff_val < ep_FwdBwd and diff_val >= ep_FwdBwd ************")

            # if (good_high - good_low) >= stepM:   this is not safe
            if (near_high - near_low) >= stepM:    # when segment size is >= M points, where M is the step size of LMM
                segment_positions = list(range(good_low, good_high + 1)) # is a list holding the positions of the points. range(x,<y) goes upto < y.
                segment = ([near_low, near_high], [good_low, good_high], segment_positions)
                segmented_traj.append(segment)

            if high == max_id:
                break

            cur_pos = high  # the position from where the next iteration should continue as upto high is already done
            good_low = next_good_low
            near_low = high     # next boundary start-point

    all_pts = set()
    for seg_element in segmented_traj:
        lst_positions = seg_element[2]  # access the third item of the tuple
        all_pts = all_pts.union(set(lst_positions))
    drop = list(set(range(max_id)) - all_pts)  # set(range(max_id)): creates set from 0 to max_id. operation - all_pts (all_pts contains the segemented set)

    # Fit each segment
    clfs = []

    # cluster_by_DTW = True
    # if cluster_by_DTW == False: # for DTW we do not need clfs computation at this stage but for dbscan/linearpiece we need
    if method != "dtw":  # for DTW we do not need clfs computation at this stage, but for dbscan/linearpiece we need
        # print ("len of segmented_traj", len(segmented_traj))
        for seg_element in segmented_traj:
            lst = seg_element[2]  # access the third item of the tuple
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

    return segmented_traj, clfs, drop


def segmented_trajectories(clfs, segmented_traj, position, method, filter_last_segment=1):
    """
    Perform segmentation of trajectories to create data structure containing segment positions. This process helps in
    keeping track of the connected segments. This information is later used to infer transitions of an HA.
    Note: We delete the last segmented portion of all the trajectories, assuming they are short segments and does not
    help learning. In fact in some cases, they can create problem during clustering (comparing segments for similarity).

    Note: Should not delete the segment if a trajectories has only one segment per trajectory.

    :param clfs: is a list. Each item of the list clfs is a list that holds the coefficients (obtained using linear
        regression) of the ODE of each segment of the segmented trajectories.
    :param segmented_traj: is a list of a custom data structure consisting of segmented trajectories (positions). Each item
        of the list contains tuple of the form ([start_ode, end_ode], [start_exact, end_exact], [p_1, ... , p_n]).
        The Tuple has three items:
            (1) first, a list of two values for recording start and end points for learning ODE
            (2) second, a list of two values for recording start and end points for learning guard and assignment using
            the exact point of a jump
            (3) third, a list of values representing the position of points of the trajectories.
    :param position: is a list of position data structure. Each position is a pair (start, end) position of a trajectory.
        For instance, the first item of the list is [0, 100] means that the trajectory has 101 points. The second item
        as [101, 300], meaning the second trajectory has 200 points. Note that all the trajectories are concatenated.
    :param method: clustering method selected by the user (options dtw, dbscan, etc.)
    :param filter_last_segment: is a boolean value. 1 to enable the filter condition for removing the last segment and
        0 for not removing the last segment.
    :return: The following data structures
        segmentedTrajectories: the required position data structures. Is a list, each item is of the
        form (start_segment_pos, pre_end_segment_pos, end_segment_pos). This list structure is used later in learning
        transition's guard and assignment equations.
        segmented_traj: the modified segmented_traj, after deleting the last segment per trajectory when filter option enabled.
        clfs: the modified clfs, after deleting the last segment per trajectory when user enabled the filter option.

    """

    # print("len(position) =", len(position))

    total_trajectories = len(position) # gives the total number of trajectories supplied as input
    total_segments = len(segmented_traj)  # total segments after segmentation process
    found_single_segment_per_trajecotry = 0 # not found so deletion is required.
    if total_segments == total_trajectories:    # meaning each trajectory has a single segment (continuous single mode system)
        # a simple example is "circle" model or a five-dimensional system in XSpeed or SpaceEx
        found_single_segment_per_trajecotry = 1    # found single-segment-per-trajectory, so do not delete segment

    # We create a list of segments positions of the form [start, pre-end, end] for each segment.

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
    for seg_traj_element in segmented_traj:
        s = seg_traj_element[2] # third element of the tuple segmented_traj
        # print("s=",s)
        start_segment_pos = s[0]  # start position of the segment
        pre_end_segment_pos = s[len(s) - 2]  # pre-end position of the segment (2nd last position)
        end_segment_pos = s[len(s) - 1]  # end position of the segment
        # print("start_segment_pos=",start_segment_pos,"   pre_end_segment_pos =",pre_end_segment_pos , "   end_segment_pos=", end_segment_pos)
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
            # print("traj_id =", traj_id)
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

    # cluster_by_DTW = True
    # delete when single_segment_per_trajectory not Found and user selected the option filter_last_segment
    if (found_single_segment_per_trajecotry == 0) and (filter_last_segment == 1):
        for pos in reversed(del_res_indices):
            segmented_traj.pop(pos)
            if method != "dtw":
                clfs.pop(pos)    # for DTW we do not have clfs at this stage, so skipping this line
            # if not cluster_by_DTW:  # for DTW clustering we do not have clfs data. So skipping this line saves time
            #     clfs.pop(pos)

    return segmentedTrajectories, segmented_traj, clfs


"""
Testing different approach. Currently we are using two_fold_segmentation.
The list of functions that can be removed and currently not in use are: 
    segment_and_fit
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
