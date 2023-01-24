
def print_flow_dynamics(f_out, G, modeID, Exp):
    f_out.write("ODE: \n")
    for dim in range(0, G[modeID].shape[0]):  # for each dimension forming a row of ODEs
        odes = "x" + str(dim) + "' = "
        for trm in range(0, G[0].shape[1]):  # for each term of the polynomial
            coef = G[modeID][dim][trm]
            # odes += str(coef) + " * " + Exp[trm].strip()
            odes += "{0:.7f}".format(coef) + " * " + Exp[trm].strip()  # Format upto 4 decimal places

            if (trm != (G[0].shape[1] - 1)):
                odes += " + "
        #       print (odes)
        f_out.write(odes)
        f_out.write("\n")

