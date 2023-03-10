import matplotlib.pyplot as plt

from operator import itemgetter
import itertools

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

def plot_guard_points(segmentedTrajectories_modified, L_y, t_list, Y):
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
            time_pt.append(t_list[0][pre_end_pt + 5])
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
            time_pt.append(t_list[0][end_pt + 5])
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

def plot_reset_points(segmentedTrajectories_modified, L_y, t_list, Y):
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
            time_pt.append(t_list[0][start_pt + 5])
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

def plot_segmentation(res, L_y, t_list, Y):
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
            time_pt.append(t_list[0][id0 + 5])  # since Y values are after leaving 5 point from start and -5 at the end

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

def plot_dropped_points(t_list, L_y, Y, Drop):

    x_pts = []
    x_p1 = []
    x_p2 = []
    x_p3 = []
    x_p4 = []
    x_p5 = []
    time_pt = []
    for i in Drop:  # Note Drop contains a list of y_list positions that were dropped
        x_pts.append({dim + 1: Y[i, dim] for dim in range(L_y)})
        time_pt.append(t_list[0][i + 5])  # since Y values are after leaving 5 point from start and -5 at the end

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

def plot_after_clustering(t_list, L_y, P, Y):

   # *************** Trying to plot the clustered points CLUSTER-WISE ***********************************

    #for each mode i

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
            time_pt.append(t_list[0][id0 + 5])   #since Y values are after leaving 5 point from start and -5 at the end

        # x_p1 = list(map(itemgetter(1), x_pts))
        # x_p2 = list(map(itemgetter(2), x_pts))
        # x_p3 = list(map(itemgetter(3), x_pts))
        x_p4 = list(map(itemgetter(4), x_pts))
        # x_p5 = list(map(itemgetter(5), x_pts))
        plt.figure(3)
        plt.title('Segment(s) After Clustering')
        # plt.scatter(time_pt, x_p1)  # padel_angle
        # plt.scatter(time_pt, x_p2) # engine-speed
        # plt.scatter(time_pt, x_p3) # for chasing cars
        # plt.scatter(time_pt, x_p3)
        plt.scatter(time_pt, x_p4) # AF the value of our interest
        # plt.scatter(time_pt, x_p5)  # controller_mode
        # plt.xlim([0, 50])
        # plt.xlim([0, 3])
        # plt.ylim([0, 3])

        # plt.xlim([0, 10])  # chasing cars      plt.scatter(x_p1)
        # plt.ylim([0, 2500])  # chasing cars

        plt.xlim([0, 50])   #AFC
        plt.ylim([10, 17])   #AFC for x_p4 the AF variable


        # plt.xlim([0, 10])  # Engine Timing System      plt.scatter(x_p3)
        # plt.ylim([500, 3000])  # Engine Timing System
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

        # plt.xlim([0, 5])  # time-horizon for Lorenz Attractor
        # plt.ylim([-30, 50])  # height for Lorenz Attractor

        # plt.xlim([-3, 3])  # time-horizon for Ven Der Pol Oscillator
        # plt.ylim([-3, 3])  # height for Ven Der Pol Oscillator

    plt.show()
    # plt.show()
    # print("Mode-invariants =", mode_inv)
