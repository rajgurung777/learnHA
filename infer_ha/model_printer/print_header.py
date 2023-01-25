

def print_header(f_out, num_mode, system_dim, transitions):
    """

    :param f_out: file pointer where the output is printed.
    :param num_mode: number of modes.
    :param system_dim: system dimension.
    :param transitions: is a list with structure [src_mode, dest_mode, guard_coeff, assignment_coeff, assignment_intercept]
            where
            src_mode: is the source location number
            dest_mode: is the destination location number
            guard_coeff: is a list containing the coefficient of the guard equation (polynomial)
            assignment_coeff: is a list containing the coefficient of the assignment equations (from linear regression)
            assignment_intercept: is a list containing the intercepts of the assignment equations (linear regression)
    :return:

    """
    # print("Prints header")
    # First line of the file for File-parameter-information. Syntax as below:
    # n_modes   n_dim   n_transitions
    # modes_dim_transitions = str(num_mode) + " " + str(L_y) + " " + str(num_mode - 1) + "\n"
    # oneVersusOne_or_OneVersusRest = 1   #1 for oneVersusOne and 2 for OneVersusRest
    # if (oneVersusOne_or_OneVersusRest == 1):
    #     total_trans = num_mode * (num_mode - 1)/ 2
    # else:
    #     if num_mode == 2:
    #         total_trans = 1
    #     else:
    #         total_trans = num_mode - 1

    total_trans = len(transitions)
    modes_dim_transitions = str(num_mode) + " " + str(system_dim) + " " + str(total_trans) + "\n"
    f_out.write(modes_dim_transitions)
    #    print("\n")
