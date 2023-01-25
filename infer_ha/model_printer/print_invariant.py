
def print_invariant(f_out, mode_inv, modeID):
    """
    :param f_out: file pointer where the output is printed.
    :param mode_inv: is a list with items of type [mode-id, invariant-constraints]. Where mode-id is the location number
                  and invariant-constraints holds the bounds (min, max) of each variable in the corresponding mode-id.
    :param modeID: location ID.

    """

    # Printing mode-invariant here with variable's bound value
    f_out.write("inv:\n")
    if len(mode_inv) == 0:  # not empty
        f_out.write("\n")
    else:
        invs = mode_inv[modeID]
        if modeID == invs[0]:  # this should always be true
            inv_bounds = invs[1]
            inv_str = ""
            for dim in range(0, len(inv_bounds)):  # for each dimension forming a row of ODEs
                var_bounds = inv_bounds[dim]
                low_bound = var_bounds[0]  # access the lower bound
                high_bound = var_bounds[1]  # access the upper bound
                inv_str += "x" + str(dim) + " >= " + str(low_bound) + " & " + "x" + str(dim) + " <= " + str(high_bound)
                if dim != (len(inv_bounds) - 1):
                    inv_str += " & "
            inv_str += "\n"
            f_out.write(inv_str)