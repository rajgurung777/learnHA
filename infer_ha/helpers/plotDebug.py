import os
import csv
import matplotlib.pyplot as plt

from operator import itemgetter
import itertools

from infer_ha.clustering.utils import create_simple_per_segmented_positions_exact, create_simple_modes_positions_for_ODE
from infer_ha.utils.util_functions import rel_diff


def plot_signals(timeSignal1, Signal1, timeSignal2, Signal2):
    # Note Signal1 and Signal2 only has output variables
    # print("len(t) is ", len(timeSignal1))
    # print("len(Signal1) is ", len(Signal1))
    # x3 = list(map(itemgetter(2), Signal1)) # Two tanks system
    # x3 = list(map(itemgetter(1), Signal1)) # BBall  0:position; 1:velocity
    # x3 = list(map(itemgetter(0), Signal1)) # Excitable Cells  0
    x3 = list(map(itemgetter(0), Signal1)) # Engine timing
    # print("len(x3) is ", len(x3))

    # print("len(t) is ", len(timeSignal2))
    # print("len(Signal2) is ", len(Signal2))
    # y3 = list(map(itemgetter(2), Signal2))  # Two tanks system
    # y3 = list(map(itemgetter(1), Signal2))  # BBall 0:position; 1:velocity
    # y3 = list(map(itemgetter(0), Signal2))  # Excitable Cells  0
    y3 = list(map(itemgetter(0), Signal2))  # Engine timing
    # print("len(y3) is ", len(y3))

    plt.plot(timeSignal1, x3, 'r-', linewidth=2, label='Signal1')
    plt.plot(timeSignal2, y3, 'b--', linewidth=2, label='Signal2')

    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()

def plot_guard_points(segmentedTrajectories_modified, L_y, t_list, Y, stepM):
    x_pts = []
    x_p1 = []
    x_p2 = []
    x_p3 = []
    x_p4 = []
    x_p5 = []
    time_pt = []
    for each_traj_row in range(0, len(segmentedTrajectories_modified)):
        for each_segment in segmentedTrajectories_modified[each_traj_row]:
            pre_end_pt = each_segment[1]
            # end_pt = each_segment[2]
            x_pts.append({dim + 1: Y[pre_end_pt, dim] for dim in range(L_y)})
            time_pt.append(t_list[0][pre_end_pt + stepM])
    # x_p1 = list(map(itemgetter(1), x_pts))
    x_p2 = list(map(itemgetter(2), x_pts))
    # x_p3 = list(map(itemgetter(3), x_pts))
    # x_p4 = list(map(itemgetter(4), x_pts))
    # x_p5 = list(map(itemgetter(5), x_pts))
    plt.figure(1)
    plt.title('Guard points')
    # plt.scatter(time_pt, x_p1)  # padel_angle
    plt.scatter(time_pt, x_p2)  # engine-speed
    # plt.scatter(time_pt, x_p3) #
    # plt.scatter(time_pt, x_p4) # AF the value of our interest
    # plt.scatter(time_pt, x_p5) # controller_mode
    # *******************************************************
    x_pts = []
    x_p1 = []
    x_p2 = []
    x_p3 = []
    x_p4 = []
    x_p5 = []
    time_pt = []
    for each_traj_row in range(0, len(segmentedTrajectories_modified)):
        for each_segment in segmentedTrajectories_modified[each_traj_row]:
            # pre_end_pt = each_segment[1]
            end_pt = each_segment[2]
            x_pts.append({dim + 1: Y[end_pt, dim] for dim in range(L_y)})
            time_pt.append(t_list[0][end_pt + stepM])
    # x_p1 = list(map(itemgetter(1), x_pts))
    x_p2 = list(map(itemgetter(2), x_pts))
    # x_p3 = list(map(itemgetter(3), x_pts))
    # x_p4 = list(map(itemgetter(4), x_pts))
    # x_p5 = list(map(itemgetter(5), x_pts))

    # plt.scatter(time_pt, x_p1)  #
    plt.scatter(time_pt, x_p2)  # engine-speed
    # plt.scatter(time_pt, x_p3) #
    # plt.xlim([0, 3])    #Navigation
    # plt.ylim([0, 3])   #Navigation
    # plt.xlim([0, 10])   #chasing cars      plt.scatter(x_p1)
    # plt.ylim([2000, 2500])   #chasing cars
    plt.xlim([0, 15])  # BBall
    plt.ylim([-1, 1])  # BBall just to see the Guard points closely  plt.scatter(x_p2)
    # plt.xlim([0, 20])  # Oscillator
    # plt.ylim([-1, 1])  # Oscillator
    # plt.xlim([-1, 500])  # Excitable Cell Model      plt.scatter(x_p1)
    # plt.ylim([-80, 60])  # Excitable Cell Model

    # plt.xlim([0, 5])   #time-horizon for Lorenz Attractor
    # plt.ylim([-30, 50])   #height for Lorenz Attractor

    # plt.xlim([-3, 3])   #time-horizon for Ven Der Pol Oscillator
    # plt.ylim([-3, 3])   #height for Ven Der Pol Oscillator

    # plt.xlim([-0.1, 10])  #Two-tanks x1, liquid in tank-1
    # plt.ylim([-1.2, 2.5])  #Two-tanks x1, liquid in tank-1
    # plt.show()

def plot_reset_points(segmentedTrajectories_modified, L_y, t_list, Y, stepM):
    x_pts = []
    x_p1 = []
    x_p2 = []
    x_p3 = []
    x_p4 = []
    x_p5 = []
    time_pt = []
    for each_traj_row in range(0, len(segmentedTrajectories_modified)):
        for each_segment in segmentedTrajectories_modified[each_traj_row]:
            # pre_end_pt = each_segment[1]
            start_pt = each_segment[0]
            x_pts.append({dim + 1: Y[start_pt, dim] for dim in range(L_y)})
            time_pt.append(t_list[0][start_pt + stepM])
    x_p1 = list(map(itemgetter(1), x_pts))
    # x_p2 = list(map(itemgetter(2), x_pts))
    # x_p3 = list(map(itemgetter(3), x_pts))
    # x_p4 = list(map(itemgetter(4), x_pts))
    # x_p5 = list(map(itemgetter(5), x_pts))
    plt.scatter(time_pt, x_p1)
    # plt.scatter(time_pt, x_p2)  # engine-speed
    # plt.scatter(time_pt, x_p3) #
    # plt.xlim([0, 3])    #Navigation
    # plt.ylim([0, 3])   #Navigation
    # plt.xlim([0, 10])  # chasing cars      plt.scatter(x_p1)
    # plt.ylim([0, 2500])  # chasing cars
    # plt.xlim([0, 15])  # BBall
    # plt.ylim([-1, 1])  # BBall just to see the Guard points closely
    # plt.xlim([0, 20])  # Oscillator
    # plt.ylim([-1, 1])  # Oscillator

    plt.xlim([-1, 500])  # Excitable Cell Model      plt.scatter(x_p1)
    plt.ylim([-80, 60])  # Excitable Cell Model

def plot_segmentation(res, L_y, t_list, Y, stepM):
    for imode in range(0, len(res)):
        x_pts = []
        x_p1 = []
        x_p2 = []
        x_p3 = []
        x_p4 = []
        x_p5 = []
        time_pt = []
        for id0 in res[imode]:  # Note: P contains list of clustered segments each segments contains the positions
            x_pts.append({dim + 1: Y[id0, dim] for dim in range(L_y)})
            time_pt.append(t_list[0][id0 + stepM])  # since Y values are after leaving 5 point from start and -5 at the end

        # x_p1 = list(map(itemgetter(1), x_pts))
        # x_p2 = list(map(itemgetter(2), x_pts))
        # x_p3 = list(map(itemgetter(3), x_pts))
        # x_p4 = list(map(itemgetter(4), x_pts))
        x_p5 = list(map(itemgetter(5), x_pts))

        plt.figure(2)
        plt.title('Segmentation')
        # plt.scatter(time_pt, x_p1)  # padel_angle
        # plt.scatter(time_pt, x_p2) # engine-speed
        # plt.scatter(time_pt, x_p3) #
        # plt.scatter(time_pt, x_p4) # AF the value of our interest
        plt.scatter(time_pt, x_p5) # controller_mode for AT speed
        # plt.xlim([0, 3])    #Navigation
        # plt.ylim([0, 3])   #Navigation
        # plt.xlim([0, 10])  # chasing cars      plt.scatter(x_p1)
        # plt.ylim([0, 2500])  # chasing cars

        # plt.xlim([0, 50])   #AFC
        # plt.ylim([14.5, 15.5])   #AFC for x_p4 the AF variable

        # plt.xlim([0, 10])  # Two Tanks      plt.scatter(x_p3)
        # plt.ylim([-1, 2])  # Two Tanks

        # plt.xlim([0, 10])  # Engine Timing System      plt.scatter(x_p3)
        # plt.ylim([1900, 3100])  # Engine Timing System

        # plt.xlim([-1, 500])  # Excitable Cell Model      plt.scatter(x_p1)
        # plt.ylim([-80, 60])  # Excitable Cell Model
        # plt.xlim([0, 15])   #BBall
        # plt.ylim([0, 25])   #BBall
        # plt.xlim([0, 20])  # Oscillator
        # plt.ylim([-1, 1])  # Oscillator
        # plt.ylim([-1.5, 1.5])

        plt.xlim([0, 20])  # AT
        plt.ylim([0, 125])  # AT

        # plt.xlim([0, 5])  # time-horizon for Lorenz Attractor
        # plt.ylim([-30, 50])  # height for Lorenz Attractor

        # plt.xlim([-3, 3])  # time-horizon for Ven Der Pol Oscillator
        # plt.ylim([-3, 3])  # height for Ven Der Pol Oscillator
    plt.show()
        # if imode == 10:
        #     break


def plot_segmentation_new(segmented_traj, L_y, t_list, Y, stepM):
    """

    @param segmented_traj: is a list of a custom data structure consisting of segmented trajectories (positions). Each item
        of the list contains tuple of the form ([start_ode, end_ode], [start_exact, end_exact], [p_1, ... , p_n]).
        The Tuple has three items:
            (1) first, a list of two values for recording start and end points for learning ODE
            (2) second, a list of two values for recording start and end points for learning guard and assignment using
            the exact point of a jump
            (3) third, a list of values representing the position of points of the trajectories.
    @param L_y: is the dimension of the system
    @param t_list: is the list of time step values for each Y values
    @param Y: is the list of values of the system. Y can be n-dimensional
    @return:
    """

    res = create_simple_per_segmented_positions_exact(segmented_traj)

    for imode in range(0, len(res)):
        x_pts = []
        x_p1 = []
        x_p2 = []
        x_p3 = []
        x_p4 = []
        x_p5 = []
        time_pt = []
        for id0 in res[imode]:  # Note: P contains list of clustered segments each segments contains the positions
            x_pts.append({dim + 1: Y[id0, dim] for dim in range(L_y)})
            time_pt.append(t_list[0][id0 + stepM])  # since Y values are after leaving 5 point from start and -5 at the end

        x_p1 = list(map(itemgetter(1), x_pts))
        # x_p2 = list(map(itemgetter(2), x_pts))
        # x_p3 = list(map(itemgetter(3), x_pts))
        # x_p4 = list(map(itemgetter(4), x_pts))
        # x_p5 = list(map(itemgetter(5), x_pts))

        plt.figure(2)
        plt.title('Segmentation')
        plt.scatter(time_pt, x_p1)  # padel_angle    refrigeration-cycle
        # plt.scatter(time_pt, x_p2) # engine-speed
        # plt.scatter(time_pt, x_p3) #engine-speed
        # plt.scatter(time_pt, x_p4) # AF the value of our interest
        # plt.scatter(time_pt, x_p5) # controller_mode for AT speed
        # plt.xlim([0, 3])    #Navigation
        # plt.ylim([0, 3])   #Navigation
        # plt.xlim([0, 10])  # chasing cars      plt.scatter(x_p1)
        # plt.ylim([0, 2500])  # chasing cars

        # plt.xlim([0, 50])   #AFC
        # plt.ylim([14.5, 15.5])   #AFC for x_p4 the AF variable

        # plt.xlim([0, 10])  # Two Tanks      plt.scatter(x_p3)
        # plt.ylim([-1, 2])  # Two Tanks

        plt.xlim([0, 3000])  # refrigeration cycle
        plt.ylim([274, 280])  # refrigeration cycle

        #
        # plt.xlim([0, 10])  # Engine Timing System      plt.scatter(x_p3)
        # plt.ylim([1900, 3100])  # Engine Timing System
        #
        # plt.xlim([0, 10])  # Engine Timing System
        # plt.ylim([10, 100])  # Engine Timing System    plt.scatter(x_p2)


        # plt.xlim([-1, 500])  # Excitable Cell Model      plt.scatter(x_p1)
        # plt.ylim([-80, 60])  # Excitable Cell Model
        # plt.xlim([0, 15])   #BBall
        # plt.ylim([0, 25])   #BBall
        # plt.xlim([0, 20])  # Oscillator
        # plt.ylim([-1, 1])  # Oscillator
        # plt.ylim([-1.5, 1.5])

        # plt.xlim([0, 20])  # AT
        # plt.ylim([0, 125])  # AT

        # plt.xlim([0, 5])  # time-horizon for Lorenz Attractor
        # plt.ylim([-30, 50])  # height for Lorenz Attractor

        # plt.xlim([-3, 3])  # time-horizon for Ven Der Pol Oscillator
        # plt.ylim([-3, 3])  # height for Ven Der Pol Oscillator
    plt.show()
        # if imode == 10:
        #     break

def print_segmented_trajectories(segmented_traj):

    for segs in segmented_traj:
        # segs a tuple of the form ([start_ode, end_ode], [start_exact, end_exact], [p_1, ... , p_n]).
        seg_ode_range = segs[0]
        seg_transition_range = segs[1]
        print("Segment: ODE data point = ", seg_ode_range, "    Transition data point = ", seg_transition_range)


def print_P_modes(P_modes):

    for mode in P_modes:
        for segs in mode:
            # make a simple mode
            points_for_ode = segs[0]
            diff = points_for_ode[1] - points_for_ode[0]
            points_for_jump = segs[1]
            print("Cluster: points_for_ode = ", points_for_ode, "  diff=", diff , "    points_for_jump = ", points_for_jump)


def plot_dropped_points(t_list, L_y, Y, Drop, stepM):

    x_pts = []
    x_p1 = []
    x_p2 = []
    x_p3 = []
    x_p4 = []
    x_p5 = []
    time_pt = []
    for i in Drop:  # Note Drop contains a list of y_list positions that were dropped
        x_pts.append({dim + 1: Y[i, dim] for dim in range(L_y)})
        time_pt.append(t_list[0][i + stepM])  # since Y values are after leaving 5 point from start and -5 at the end

    x_p1 = list(map(itemgetter(1), x_pts))
    # x_p2 = list(map(itemgetter(2), x_pts))
    # x_p3 = list(map(itemgetter(3), x_pts))
    # x_p3 = list(map(itemgetter(3), x_pts))
    # x_p4 = list(map(itemgetter(4), x_pts))
    # x_p5 = list(map(itemgetter(5), x_pts))

    plt.figure(2)

    plt.title('Dropped points during segmentation')
    plt.scatter(time_pt, x_p1) # padel_angle
    # plt.scatter(time_pt, x_p2) # engine-speed
    # plt.scatter(time_pt, x_p4) # AF the value of our interest
    # plt.scatter(time_pt, x_p5) # controller_mode
    # plt.xlim([0, 50])
    plt.show()
    # *************** Trying to plot the dropped points ***********************************

def plot_after_clustering(t_list, L_y, P_modes, Y, stepM):



   # *************** Trying to plot the clustered points CLUSTER-WISE ***********************************
    '''
    Example:
    invariant =[]
    mode_inv = []
    invariant.append([10, 20])
    invariant.append([12, 19])
    invariant.append([900, 990])
    invariant.append([40, 60])
    invariant.append([0, 1])
    mode_inv.append([0, invariant])
    print ("invariant ", invariant)
    print ("Mode invariant = ", mode_inv)
    mode_inv.append([1, invariant])
    mode_inv.append([2, invariant])
    print ("Mode invariant = ", mode_inv)
    '''


    P = create_simple_modes_positions_for_ODE(P_modes)


    NUM_COLORS = len(P)

    # cm = plt.get_cmap('gist_rainbow')
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # ax.set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    #


    for imode in range(0, len(P)):   # This loop runs for each mode. Also, used to obtain Mode invariants
        x_pts = []
        x_p1 = []
        x_p2 = []
        x_p3 = []
        x_p4 = []
        x_p5 = []
        time_pt = []
        invariant = []
        for id0 in P[imode]:  # Note: P contains list of clustered segments each segments contains the positions
            x_pts.append({dim + 1: Y[id0, dim] for dim in range(L_y)})
            time_pt.append(t_list[0][id0 + stepM])   #since Y values are after leaving 5 point from start and -5 at the end

        x_p1 = list(map(itemgetter(1), x_pts))
        # x_p2 = list(map(itemgetter(2), x_pts))
        # x_p3 = list(map(itemgetter(3), x_pts))
        # x_p4 = list(map(itemgetter(4), x_pts))
        # x_p5 = list(map(itemgetter(5), x_pts))
        plt.figure(3)
        plt.title('Segment(s) After Clustering')
        plt.scatter(time_pt, x_p1)  # padel_angle  refrigeration-cycle
        # plt.scatter(time_pt, x_p2) # engine-speed
        # ax.scatter(time_pt, x_p2) # engine-speed
        # plt.scatter(time_pt, x_p3) # for chasing cars
        # ax.scatter(time_pt, x_p3) # engine-speed
        # plt.scatter(time_pt, x_p4) # AF the value of our interest
        # plt.scatter(time_pt, x_p5)  # controller_mode

        # plt.xlim([0, 50])
        # plt.xlim([0, 3])
        # plt.ylim([0, 3])

        # plt.xlim([0, 10])  # chasing cars      plt.scatter(x_p1)
        # plt.ylim([0, 2500])  # chasing cars

        # plt.xlim([0, 50])   #AFC
        # plt.ylim([10, 17])   #AFC for x_p4 the AF variable

        plt.xlim([0, 3000])  # refrigeration cycle
        plt.ylim([274, 280])  # refrigeration cycle

        # plt.xlim([0, 10])  # Engine Timing System      plt.scatter(x_p3)
        # plt.ylim([1900, 3500])  # Engine Timing System

        # plt.xlim([-1, 500])  # Excitable Cell Model      plt.scatter(x_p1)
        # plt.ylim([-80, 60])  # Excitable Cell Model
        # plt.xlim([0, 25])
        # plt.ylim([-1.5, 1.5])
        #
        # plt.xlim([0, 15])   #BBall
        # plt.ylim([0, 25])   #BBall

        # plt.xlim([0, 20])  # Oscillator
        # plt.ylim([-1, 1])  # Oscillator
        # plt.xlim([0, 100])  # chasing cars
        # plt.ylim([-500, 100])  # chasing cars
        # plt.xlim([-0.1, 10])  # Two-tanks x1, liquid in tank-1
        # plt.ylim([-1.2, 2.5])  # Two-tanks x1, liquid in tank-1

        # plt.xlim([0, 10])  # Two Tanks      plt.scatter(x_p3)
        # plt.ylim([-1, 2])  # Two Tanks

        # plt.xlim([0, 5])  # time-horizon for Lorenz Attractor
        # plt.ylim([-30, 50])  # height for Lorenz Attractor

        # plt.xlim([-3, 3])  # time-horizon for Ven Der Pol Oscillator
        # plt.ylim([-3, 3])  # height for Ven Der Pol Oscillator
        # plt.show()

    plt.show()
    # plt.show()
    # print("Mode-invariants =", mode_inv)

def plot_data_values(segmentedTrajectories, Y, L_y):
    """
    Helps in debugging

    @param segmentedTrajectories: the required position data structures. Is a list, each item is of the
        form (start_segment_pos, pre_end_segment_pos, end_segment_pos). This list structure is used later in learning
        transition's guard and assignment equations.
        An example of the values in segmentedTrajectories data structure is
        [[[0, 360, 361], [362, 701, 702], [703, 973, 974], [975, 1191, 1192]], [[1295, 1666, 1667], [1668, 2009, 2010],
         [2011, 2284, 2285], [2286, 2504, 2505]]]
    @return:
    """
    for traj in segmentedTrajectories:
        for seg in traj:
            start = seg[0]
            pre_end = seg[1]
            end = seg[2]
            print("start=",print_data_value(start,Y,L_y))
            print("pre_end=",print_data_value(pre_end,Y,L_y))
            print("end=",print_data_value(end,Y,L_y))

def print_data_value(position, Y, dimension):
    x_pts = []
    x_pts.append({dim + 1: Y[position, dim] for dim in range(dimension)})

    return x_pts

def output_derivatives(b1, b2, Y, size_of_input_variables):
    if os.path.exists("outputs/amit_backward.txt"):
        os.remove("outputs/amit_backward.txt")
    file_out = open("outputs/amit_backward.txt","a")
    file_out.write("b1 backward derivatives: \n")
    str1 = ""
    pos_value = 0
    relDiff = 0.0
    for i in range(0, len(b1)):
        # if (i==0):
        #     print(type(b1[i, size_of_input_variables:]))
        if (i!=0 and i != len(b1)):
            relDiff = rel_diff(b1[i, size_of_input_variables:], b1[(i-1), size_of_input_variables:])  # compute relative difference between current and previous
        # str1 = "pos:" + str(pos_value) + " b1[i] = " + str(b1[i]) + "  b1[i,onlyOutVar] = " + str(b1[i, size_of_input_variables:]) + "  relDiff = " + str(relDiff) + " \n"
        relFwdBwd =  rel_diff(b1[i, size_of_input_variables:], b2[i, size_of_input_variables:])
        str1 = "pos: " + str(i) + " Y: " + str(Y[i, size_of_input_variables:]) + " b1: " + str(b1[i, size_of_input_variables:]) + " relBDF_Diff = " + str(relDiff) + " relDiff_FwdBwd = " + str(relFwdBwd) + "\n"
        # pos_value += 1
        file_out.write(str1)
    file_out.close()

    if os.path.exists("outputs/amit_forward.txt"):
        os.remove("outputs/amit_forward.txt")
    file_out2 = open("outputs/amit_forward.txt", "a")
    file_out2.write("b2 forward derivatives: \n")
    pos_value = 0
    relDiff = 0.0
    for i in range(0, len(b2)):
        if i == (len(b2) - 1):
            relDiff = rel_diff(b2[i], b2[i])  # for the last row
        else:
            relDiff = rel_diff(b2[i], b2[i+1])  # compute relative difference between current and next
        # str1 = "pos:" + str(pos_value) + "   " +  str(b2[i]) + "  relDiff = " + str(relDiff) + " \n"
        relFwdBwd = rel_diff(b1[i, size_of_input_variables:], b2[i, size_of_input_variables:])
        str1 = "pos: " + str(i) + " Y: " + str(Y[i, size_of_input_variables:])  + " b2: " + str(b2[i, size_of_input_variables:]) + " relFwd_Diff = " + str(relDiff) + " relDiff_FwdBwd = " + str(relFwdBwd)  + "\n"
        # pos_value += 1
        file_out2.write(str1)
    file_out2.close()


def analyse_output(segmentedTrajectories, b1, b2, Y, t_list, L_y, size_of_input_variables, stepM, varIndex):
    """

    @param segmentedTrajectories: is a list of a custom data structure consisting of segmented trajectories (positions).
        Each list item contains tuple of the form ([start_ode, end_ode], [start_exact, end_exact], [p_1, ... , p_n]).
        The Tuple has three items:
            (1) first, a list of two values for recording start and end points for learning ODE
            (2) second, a list of two values for recording start and end points for learning guard and assignment using
            the exact point of a jump
            (3) third, a list of values representing the position of points of the trajectories.
    @param b1: Derivatives using backward version of BDF
    @param b2: Derivatives using forward version of BDF
    @param Y:  The actual data of all the input and output variables
    @param t_list: The time steps of the trajectories corresponding to the Y value (but includes all the points unlike
                    the b1,b2 and Y where first and last M points are excluded
    @param L_y: The dimension
    @param size_of_input_variables: total number of input variables
    @return:
    """
    # Note b1,b2 and Y containts data (excluding first and last M points from the trajectories)

    # varIndex = 2  # 0 and 1 are input-variable and 2 is the output variable for engine-timing

    # ##### writing to a csv file for debugging and analysing values #########
    file_csv = open('segmentationFile.csv','w')
    writer = csv.writer(file_csv)   # file pointer created
    rowValue = ["pos", "t_value", "x_"+str(varIndex) , "BDF_backward", "BDF_forward", "Seg-ID"]
    writer.writerow(rowValue)
    count = 0   # count for segment-ID
    for seg in segmentedTrajectories:
        # for seg in traj:
        ode_pos = seg[0]
        exact_pos = seg[1]
        segment_data = seg[2]
        # print("start=",print_data_value(start,Y,L_y))
        # print("pre_end=",print_data_value(pre_end,Y,L_y))
        # print("end=",print_data_value(end,Y,L_y))
        start_pos_ode = ode_pos[0]
        end_pos_ode = ode_pos[1]


        for pos in range(start_pos_ode, end_pos_ode+1):     # since range goes < end_pos_ode
            t_value = t_list[0][pos + stepM]
            variable = Y[pos, varIndex]
            b1_val = b1[pos, varIndex]
            b2_val = b2[pos, varIndex]
            rowValue = [pos, t_value, variable, b1_val, b2_val, count]
            writer.writerow(rowValue)
        count += 1
