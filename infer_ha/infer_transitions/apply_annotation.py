

def apply_annotation(Y, variableType_datastruct, list_connection_pt, assignment_coeff, assignment_intercept):
    """
    Type Annotation function. Type annotation is performed on the assignments based on the variable's type.

    :param Y: contains the y_list values for all the points except the first and last M points (M is the order in BDF).
    :param variableType_datastruct: specific data structure holding user's information about type annotation values.
    :param list_connection_pt: is the connection triplet having (pre-end, end, start) point/position for a connection.
    :param assignment_coeff: contains the values of the coefficients for the assignment equation.
    :param assignment_intercept: contains the values of the intercepts for the assignment equation.
    :return: the coefficients and intercept values of the assignment equations.

    Examples:
    For variable having index i, do the following for the variable of
    type 't1' (continuous)
        1) set coefficient with index i, to 1 and 0 to the rest
        2) set the intercept value to 0 for index i
    type 't2' (from pool values, majority polling: when equal takes the 1st value)
        from the destination connection_pt[1] get the concrete value of variable index i and do the frequency count of each pool-values
        which is supplied by the user and select the highest frequency count pool-value (say pv)
        Now,
        1) set all coefficient with index i, to 0
        2) set the intercept value to pv for index i

    type 't3' (constant assignment or the jump reset assignment)
        1) set all coefficient with index i, to 0
        2) set the intercept value to the constant value for index i


    """

    # print(variableType_datastruct)

    # if no annotation information is provided then below for-loop will not run
    for var_type_detail in variableType_datastruct:  # accessing each variable details
        var_index = var_type_detail[0]
        var_type = var_type_detail[2]
        # print("Annotation Section: var_index=", var_index, "   var_type=", var_type)
        if var_type == "t1":
            for coeff_index in range(len(assignment_coeff[var_index])):
                # print(" yes inside t1, var_index=", var_index, "   coeff_index=", coeff_index)
                if var_index == coeff_index:
                    assignment_coeff[var_index][coeff_index] = 1
                else:
                    assignment_coeff[var_index][coeff_index] = 0
            assignment_intercept[var_index] = 0

        elif var_type == "t2":
            # Take the connecting points and do frequency count of the value
            pool_values = var_type_detail[3]
            frequency_count = [0] * len(pool_values)  # List of zero values of size=number of pool values
            for connection_pt in list_connection_pt:
                dest_loc_position = connection_pt[2]  # Now the start-point of the destination is index [2]
                pv = Y[dest_loc_position, var_index]
                # print("pv=",pv)
                for pool_val_index in range(0, len(pool_values)):
                    if pv == pool_values[pool_val_index]:
                        frequency_count[pool_val_index] += 1
                        break
            # Now check the highest frequency_count value
            max_count_value = max(frequency_count)
            max_count_index = frequency_count.index(max_count_value)
            # print("Majority poll value = ", pool_values[max_count_index])
            for coeff_index in range(0, len(assignment_coeff[var_index])):
                assignment_coeff[var_index][coeff_index] = 0
            assignment_intercept[var_index] = pool_values[max_count_index]

        elif var_type == "t3":
            constant_value = var_type_detail[4]
            for coeff_index in range(len(assignment_coeff[var_index])):
                # print(" yes inside t3, var_index=", var_index, "   coeff_index=", coeff_index)
                assignment_coeff[var_index][coeff_index] = 0
            assignment_intercept[var_index] = constant_value

        elif var_type == "t4":
            print("Suggest to consider type t4 as t2 if values are known. Otherwise linear regression is performed")

        elif var_type == "t5":
            print("we have to use Linear Regression for type t5 variables")

    return assignment_coeff, assignment_intercept
