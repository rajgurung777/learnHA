

def print_header(f_out, num_mode, system_dim, transitions):
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
