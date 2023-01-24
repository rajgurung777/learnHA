
from infer_ha.infer_transitions.apply_annotation import apply_annotation
from infer_ha.infer_transitions.connecting_points import create_connecting_points
from infer_ha.infer_transitions.compute_assignments import compute_assignments
from infer_ha.infer_transitions.guards import getGuard_inequality


def compute_transitions(P, position, segmentedTrajectories, L_y, boundary_order, Y,
                        variableType_datastruct, number_of_segments_before_cluster, number_of_segments_after_cluster):

    # print("Computing Connecting points for Transitions ...")
    data_points = create_connecting_points(P, position, segmentedTrajectories)
    print("Computing Connecting points done!")
    print("len(data_points) =",len(data_points))

    transitions = []

    # If a model is a single-mode system for instance, lets say our segmentation identifies a full trajectory as a
    # single mode. And this is true for all input trajectories (say init-size 10 and all 10 trajectories are learned
    # as a single-mode system), then such model do not have/need to learn Transition.
    # So number_of_segments_after_cluster is 1 but number_of_segments_before_cluster will be 10 (initial init-size)
    # Therefore, we need to handle it carefully
    # Fixing separately for single-mode system like Bouncing Ball or single-mode systems without transition.
    # E.g. in the case of bouncing ball all segments are clustered into One.
    if (number_of_segments_after_cluster == 1) and (number_of_segments_before_cluster >= 1):
        tot_input_trajectories = len(position)
        print("Total initial simulation=", tot_input_trajectories)
        if tot_input_trajectories == number_of_segments_before_cluster:
            return transitions  # transitions here is empty for a single mode system without transition.

    '''
    Transition structure: [src, dest, coeffs, guard]
    src and dest are the source and destination ID i.e., the location number
    coeffs: X-coefficients for each mapping of X'. List of coefficients for each X' variable.
    assign_intercept: List of intercepts for each X' variable.
    guard: X-coefficients of the hyperplane equation
    '''
    # transitions = []
    # data_points contains list of connecting points for each Transition
    # Note we are considering possible transition only based on the given trajectory-data.
    # *************** Trying to plot these points. Also creating transition ***********************************
    for imode in range(0, len(data_points)):   # len(data_points) returns the total number of transitions
        list_connection_pt = data_points[imode][2]  # list of connecting point(s) of type [pre_end_pt_position, end_pt_position, start_pt_position]
        src_mode = data_points[imode][0]  # src mode
        dest_mode = data_points[imode][1]  # dest mode

        # Now we only use a few connecting-points [pre_end_point and end_point] to find guard using SVM
        # ******* Step-1: create the source and destination list of positions and Step-2: call getGuardEquation()
        srcData = []
        destData = []
        for connect_pt in list_connection_pt:
            srcData.append(connect_pt[0])  # index [0] is the pre_end_pt_position
            destData.append(connect_pt[1])  # index [1] is the end_pt_position

        guard_coeff = getGuard_inequality(srcData, destData, L_y, boundary_order, Y)

        print("Check guard=", guard_coeff)

        '''
        We will not check any complex condition. We simply apply linear regression to first learn the assignments.
        Then, we check the condition for annotations and whenever annotation information is available we replace
         the computed (learned using linear regression) values using our approach of annotations.
        '''

        assign_coeff, assign_intercept = compute_assignments(list_connection_pt, L_y, Y)
        assignment_coeff = assign_coeff
        assignment_intercept = assign_intercept

        assignment_coeff, assignment_intercept = apply_annotation(Y, variableType_datastruct, list_connection_pt, assignment_coeff, assignment_intercept)

        transitions.append([src_mode, dest_mode, guard_coeff, assignment_coeff, assignment_intercept])
        # print("All Transitions are: ",transitions)

    return transitions