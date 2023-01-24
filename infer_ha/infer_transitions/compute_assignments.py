
from sklearn import linear_model


def compute_assignments(list_connection_pt, L_y, Y):
    x_pts = []
    y_pts = []
    for connection_pt in list_connection_pt:
        # for id0 in connection_pt:  # Note: connection_pt now contains [pre_end, end, and start positions]
        # SOURCE-POINT: end-pt-position
        id0 = connection_pt[1]  # now index is [1] for end-posi
        x_pts.append([Y[id0, dim] for dim in range(L_y)])

        # DESTINATION-POINT: start-pt-position
        id0 = connection_pt[2]  # Now index is [2] for start_posi
        y_pts.append([Y[id0, dim] for dim in range(L_y)])

    # lin_reg = linear_model.LinearRegression(fit_intercept=False)  # without intercepts
    lin_reg = linear_model.LinearRegression()  # with intercepts
    lin_reg = lin_reg.fit(x_pts, y_pts)
    print("Linear Regression Score = ", lin_reg.score(x_pts, y_pts))
    assign_coeff = lin_reg.coef_
    assign_intercept = lin_reg.intercept_
    # print("Now reg.intercept_")
    # print(lin_reg.intercept_)
    return assign_coeff, assign_intercept
