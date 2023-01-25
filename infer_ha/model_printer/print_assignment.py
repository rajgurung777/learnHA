
def print_assignment(f_out, assign_coeffs, intercepts):
    """

    :param f_out: file pointer where the output is printed.
    :param assign_coeffs: the coefficients obtained from SVM for the guard equation.
    :param intercepts: is the polynomial expression obtained from the function myUtil.multinomial().

    """
    # **** Printing Jump-Reset equations ***** Todo: poly of degree n
    trans_assign = "reset:\n"
    for dim in range(0,
                     len(assign_coeffs)):  # for each dimension. coeffs is a list of values for each dependent variable
        coeffs_per_variable = assign_coeffs[dim]
        intercept_term = intercepts[dim]
        trans_assign_per_variable = "x" + str(dim) + "' = "
        for eachterm in range(0, len(coeffs_per_variable)):
            coeff_term = coeffs_per_variable[eachterm]
            # trans_assign_per_variable += str(coeff_term) + " * " + "x" + str(eachterm)
            trans_assign_per_variable += "{0:.7f}".format(coeff_term) + " * " + "x" + str(eachterm)
            if eachterm != (len(coeffs_per_variable) - 1):
                trans_assign_per_variable += " + "

        trans_assign_per_variable += " + " + "{0:.3f}".format(intercept_term) + "\n"
        trans_assign += trans_assign_per_variable
    f_out.write(trans_assign)
