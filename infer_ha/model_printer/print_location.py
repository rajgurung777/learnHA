from infer_ha.model_printer.print_invariant import *
from infer_ha.model_printer.print_flow import *

def print_location(f_out, P, G, mode_inv, Exp):
    """
    :param f_out: file pointer where the output is printed.
    :param P: is a list. Each item of the list P contain list of values which are positions of points of a trajectories.
           The size of the list P is equal to the number of clusters or modes of the learned hybrid automaton (HA).
    :param G: is a list. Each item of the list G is a list that holds the coefficients (obtained using linear regression)
           of the ODE of a mode of the learned HA.
    :param mode_inv: is a list with items of type [mode-id, invariant-constraints]. Where mode-id is the location number
                  and invariant-constraints holds the bounds (min, max) of each variable in the corresponding mode-id.
    :param Exp: is the polynomial expression obtained from the mapping \Phi function.

    """
    # ****** Writing the initial mode before so that Automaton gets the initial location ID. ******
    val = P[0][0]  # first mode
    indexVal = 0
    for mods in range(0, len(P)):
        if (P[mods][0] < val):
            val = P[mods][0]
            indexVal = mods
    # print("Initial mode-ID = ", indexVal+1) #Since all mode-ID starts from 1
    # print("Initial = ",val)
    initVal = "Initial-mode " + str(indexVal + 1) + "\n"
    f_out.write(initVal)

    # ****** Writing the mode ODE. ******
    for modeID in range(0, len(G)):  # for each mode ODE
        modelabel = "mode " + str(modeID + 1) + "\n"  # printing mode starting from 1, since dReach has issue for mode=0
        f_out.write(modelabel)
        # print(modelabel)
        print_invariant(f_out, mode_inv, modeID)
        print_flow_dynamics(f_out, G, modeID, Exp)
