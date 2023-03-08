from infer_ha.model_printer.print_invariant import *
from infer_ha.model_printer.print_flow import *

def print_location(f_out, P, G, mode_inv, Exp, position):
    """
    :param f_out: file pointer where the output is printed.
    :param P: is a list. Each item of the list P contain list of values which are positions of points of a trajectories.
           The size of the list P is equal to the number of clusters or modes of the learned hybrid automaton (HA).
    :param G: is a list. Each item of the list G is a list that holds the coefficients (obtained using linear regression)
           of the ODE of a mode of the learned HA.
    :param mode_inv: is a list with items of type [mode-id, invariant-constraints]. Where mode-id is the location number
                  and invariant-constraints holds the bounds (min, max) of each variable in the corresponding mode-id.
    :param Exp: is the polynomial expression obtained from the mapping \Phi function.
    :param position: is a list containing positions of the input list_of_trajectories. This structure is required for printing
            the HA model. Particularly, to get the starting positions of input trajectories for identifying initial mode(s).

    """
    # ****** Writing the initial mode before so that Automaton gets the initial location ID. ******

    initial_location_list = get_initial_location(P, position)
    # print("initial_location_list = ", initial_location_list)
    # initVal = "Initial-mode " + str(indexVal + 1) + "\n"
    initVal = "Initial-mode " + str(initial_location_list[0] + 1) + "\n"    # Old implementation writing only a single
                            # initial location. The mode in which the 1st/starting trajecotry lies.
    # In the later version we will determine all initial modes and print them here. Accordingly, the interface/syntax
    # of the output model-file will also change. Then, the calling project (BBC4CPS will have to change the model-parser
    # module.)
    f_out.write(initVal)

    # ****** Writing the mode ODE. ******
    for modeID in range(0, len(G)):  # for each mode ODE
        modelabel = "mode " + str(modeID + 1) + "\n"  # printing mode starting from 1, since dReach has issue for mode=0
        f_out.write(modelabel)
        # print(modelabel)
        print_invariant(f_out, mode_inv, modeID)
        print_flow_dynamics(f_out, G, modeID, Exp)

def get_initial_location(P, position):
    """
    At the moment we are printing only the mode-ID where the first/starting trajectory is contained in.
    Finding other initial modes, require searching segmented trajectories and identifying the probable initial positions
    of the trajectories. Probable because we still drop points during segmentation (the start-point of a segmented trajectory).
    This dropping of points (in addition to the first M points, where M is the step in LMM) makes is hard to track the
    position of the initial trajectories using the data-structure position.

    @param P: is a list. Each item of the list P contain list of values which are positions of points of a trajectories.
           The size of the list P is equal to the number of clusters or modes of the learned hybrid automaton (HA).
    @param position: is a list containing positions for the input list-of-trajectories.
    @return: init_locations: contains a list of initial location ID(s), having zero based indexing.

    """
    # print("P = ", P)
    # print("position = ", position)
    init_locations = []
    val = P[0][0]  # first mode
    # print("P[0][0] = val =", val)
    indexVal = 0
    for mods in range(0, len(P)):
        if (P[mods][0] < val):
            val = P[mods][0]
            indexVal = mods

    init_locations.append(indexVal)


    return init_locations

