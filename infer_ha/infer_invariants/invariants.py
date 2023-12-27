"""
This module contains functions for computing mode-invariants.
Currently, a simple over-approximation approach is applied to compute the mode-invariant.
We use a straightforward approach to compute each variable's bound values (Min, Max) in each mode.
A more complex approach can be implemented and tested in this module.
"""

from operator import itemgetter

from infer_ha.clustering.utils import create_simple_modes_positions


def compute_mode_invariant(L_y, P_modes, Y, invariant_enabled):
    """
    This function decides to compute or ignore mode-invariant computation based on the user's choice.

    :param L_y: is the dimension (input + output variables) of the system whose trajectory is being parsed.
    :param P_modes: hols a list of modes. Each mode is a list of structures; we call it a segment.
          Thus, P = [mode-1, mode-2, ... , mode-n] where mode-1 = [ segment-1, ... , segment-n] and segments are
          of type ([start_ode, end_ode], [start_exact, end_exact], [p1, ..., p_n]).
    :param Y: contains the y_list values for all the points except the first and last M points (M is the order in BDF).
    :param invariant_enabled: is the user's choice of computing or ignoring invariant. The value 0 and 1 to compute and
        2 for ignoring.
    :return: A list of values of type [mode-id, invariant-constraints]. mode-id is the location ID and
        invariants-constraints is the list of (min,max) bounds of each variable. The order of the variable is maintained.

    """

    # Todo: think better option.
    # 0 and 1: enabled 2: disabled
    mode_inv = []
    if invariant_enabled == 2:
        print("Computing Mode Invariant IGNORED!")
    else:
        mode_inv = compute_invariant(L_y, P_modes, Y)
        print("Computing Mode Invariant done!")

    # mode_inv = compute_invariant(L_y, P_modes, Y)  # Always compute mode Invariant irrespective of user's choice for BBC
                                                # since it is now needed for automata composition

    return mode_inv


def compute_invariant (L_y, P_modes, Y):
    """
    This function computes the invariant for each mode/cluster.

    :param L_y: is the dimension (input + output variables) of the system whose trajectory is being parsed.
    :param  P_modes: holds a list of modes. Each mode is a list of structures; we call it a segment.
          Thus, P = [mode-1, mode-2, ... , mode-n] where mode-1 = [ segment-1, ... , segment-n] and segments are
          of type ([start_ode, end_ode], [start_exact, end_exact], [p1, ..., p_n]).
    :param Y: contains the y_list values for all the points except the first and last M points (M is the order in BDF).
    :return: A list of values of type [mode-id, invariant-constraints]. mode-id is the location ID and
       invariants-constraints is the list of (min,max) bounds of each variable.
       The order of the variable is maintained in the list.


    """

    P = create_simple_modes_positions(P_modes)

    x_pts = []
    #for each mode i
    invariant = []
    mode_inv = []
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
        invariant = []
        for id0 in P[imode]:  # Note: P contains list of clustered segments each segments contains the positions
            x_pts.append({dim + 1: Y[id0, dim] for dim in range(L_y)})

        for var_dim in range(L_y):  # invariant consists of list of bounds on the variables. The order is maintained
            x_dim_list = list(map(itemgetter(var_dim+1), x_pts))
            upperBound = max(x_dim_list)
            lowerBound = min(x_dim_list)
            invariant.append([lowerBound, upperBound])
        '''            
        upperBound = max(x_p1)
        lowerBound = min(x_p1)
        invariant.append([lowerBound, upperBound])
        upperBound = max(x_p2)
        lowerBound = min(x_p2)
        invariant.append([lowerBound, upperBound])
        upperBound = max(x_p3)
        lowerBound = min(x_p3)
        invariant.append([lowerBound, upperBound])
        upperBound = max(x_p4)
        lowerBound = min(x_p4)
        invariant.append([lowerBound, upperBound])
        upperBound = max(x_p5)
        lowerBound = min(x_p5)
        invariant.append([lowerBound, upperBound])
        '''
        mode_inv.append([imode, invariant])

    # print("Mode-invariants =", mode_inv)
    return mode_inv

