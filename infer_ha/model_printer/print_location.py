from infer_ha.model_printer.print_invariant import *
from infer_ha.model_printer.print_flow import *

def print_location(f_out, P, G, mode_inv, Exp):
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
