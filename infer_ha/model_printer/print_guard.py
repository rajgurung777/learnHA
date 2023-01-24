
def print_guard(f_out, coeffs, gExp):
    # The guard/boundary is obtained using SVM (Polynomial Kernel) which is of the form
    # y(w,x)=w * x + b and both the coefficient vector w and x obeys the \Phi mapping function in the paper.
    #
    trans_guard = "guard:\n"
    f_out.write(trans_guard)
    guard = ""
    coef1 = coeffs[0]  # first column coefficient, can be used to divide on both sides to normalize
    for i in range(0, len(coeffs)):  # for each term of the polynomial
        coef_val = coeffs[i]
        if (
                coef1 != 0 and coef1 > 1):  # divide only when first coefficient is >1 to normalize and not spoil/raise values
            coef_val = coef_val / coef1  # Dividing all coefficient by the 1st coefficient, since right hand side of the equation is 0
            # coef_val = coef_val / 100  # guard coefficient further scaled down by 100 for avoiding large numerical calculation issues

        guard += str(coef_val) + " * " + gExp[i].strip()
        # guard += "{0:.4f}".format(coef_val) + " * " + gExp[i].strip()
        # guard += "{0:.10f}".format(coef_val) + " * " + gExp[i]

        if (i != (len(coeffs) - 1)):
            guard += " + "
    guard += " = 0"
    guard = guard + "\n"
    f_out.write(guard)
    # print("guard=", guard)