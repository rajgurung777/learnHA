'''
This module contains the clustering algorithm
 1) DBSCAN and
 2) piecelinear: an approach in the paper by Jin et al.

'''
import numpy as np
import sklearn.cluster as skc
from sklearn import linear_model

import itertools

from infer_ha.utils.util_functions import matrowex


def dbscan_cluster(clfs, res, A, b1, num_mode, dbscan_eps_dist, dbscan_min_samples):
    # Clustering

    num_coeff = clfs[0].coef_.shape[0] * clfs[0].coef_.shape[1]
    # print ("clfs[0].coef_.shape[0]=", clfs[0].coef_.shape[0])   # returns number of dimension or rows 3 for bball
    # print ("clfs[0].coef_.shape[1]=", clfs[0].coef_.shape[1])   # returns number of column i.e, poly coefficients (including intercepts)
    print("len(clfs)=",len(clfs))

    cluster_coefs = [clfs[i].coef_.reshape((num_coeff,)) for i in range(len(clfs))] # 1st each of the coefficient matrix
    # are reshaped (this will make [row by column] coefficient-matrix into a single vector of size num_coeff).
    # This is repeated for all coefficients clfs[i].coef_ , creating an array of features for DBSCAN algo
    # Then, this above data is created into a list [...] which is repeated len(clfs) times. clfs is the segment-size
    # Now, out of len(clfs) list of vectors clustering using DBSCAN is performed giving the optimal number of clusters
    # print(cluster_coefs)  # Is a list of arrays. Each array are values of coefficients of each segments. The coefficients of different rows are reshape to 1-D array

    # We discard/truncate figures with higher decimal point. So that nearly similar coefficients becomes equal
    # and that becomes easy for DBSCAN to identify that as equal/similar. Although DBSCAN may do this internally
    # ***************** Trancate trailing decimal digits *******************
    for x in range(0, len(cluster_coefs)):
        for i, element in enumerate(cluster_coefs[x]):
            # print("element=", element, " formated element=", round(element, 5))
            cluster_coefs[x][i] = round(element, 5)
            # print("Modified element=", element, " formated cluster_coefs[x][i]=", cluster_coefs[x][i])
    # print("After round to 5 deci =", cluster_coefs)  # Is a list of arrays. Each array are values of coefficients of each segments. The coefficients of different rows are reshape to 1-D array
    # **************************************************************************


    # db = skc.DBSCAN(eps=0.2, min_samples=1).fit(cluster_coefs)
    db = skc.DBSCAN(eps=dbscan_eps_dist, min_samples=dbscan_min_samples).fit(cluster_coefs)
    # db = skc.DBSCAN(eps=0.01, min_samples=1).fit(cluster_coefs) #Amit trying
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    mode_pts = []
    for i in range(n_clusters):
        mode_pts.append([])
    for i, lab in enumerate(labels):        # enumerate through all the clusters
        if lab > -1:
            mode_pts[lab].extend(res[i])    # get the positions from res for the same clusters in mode_pts
    length_and_modepts = [(len(mode_pts[i]), mode_pts[i]) for i in range(0, len(mode_pts))] # create a list of 2-tuple
    length_and_modepts.sort(reverse=True)   # this data is sorted from highest number of points in the cluster to lowest
    # print ("length_and_modepts = ",length_and_modepts)
    print ("DBSCAN: Total clusters = ",len(length_and_modepts))
    if len(length_and_modepts)==num_mode:
        print("User decided number of modes EQUALs DBSCAN clustering!!!")

    # In order not to discard any segments in clustering. We consider all the clusters returned by DBSCAN as modes
    # ***************************************
    num_mode = len(length_and_modepts)  # **Note: comment this line to enable user decided mode size.
    # ***************************************
    mode_pts = []
    # print ("length_and_modepts = ",length_and_modepts) # *** this value can be less than num_mode ***
    # Fixing when the number of segments returned by DBSCAN is less than user's num_mode input
    if len(length_and_modepts) < num_mode:  # This will execute when above **Note line is commented
        num_mode = len(length_and_modepts)

    for i in range(0, num_mode):   # Now since num_mode is assumed to be <= len(length_and_modepts), so only the first
        # num_mode of data is considered as outputs.
        _, mode_ptsi = length_and_modepts[i]
        mode_pts.append(mode_ptsi)
    # Fit each cluster again
    clfs = []
    print("Computing Linear Regression(ODE) for the combined Cluster")
    for i in range(num_mode): # For this considered outputs coefficients are computed again
        clf = linear_model.LinearRegression(fit_intercept=False)
        clf.fit(matrowex(A, mode_pts[i]), matrowex(b1, mode_pts[i]))
        clfs.append(clf)

    P = mode_pts
    G = []
    for i in range(len(clfs)):
        G.append(clfs[i].coef_)

    return P, G



def balls_in_boxes3(n, m, badcom, res, A, b1, ep):
    """
    Get n different balls, and m same boxes then compute all the cases.
    Return a list whose entry is a set representing every case.
    Every set contains m tuples representing m boxes.
    Every tuple contains some numbers from 1 to n representing different balls.
    This function implements the PruningSearch algorithm in the paper Jin et al.
    """
    # print("badcom = ", badcom)
    # base case
    if n == m:
        return [{(i,) for i in range(n)}]
    if m == 1:
        bad = 0
        for badlist in badcom:
            if set([i for i in range(n)]) >= set(badlist):
                # print("Set([i for i in range(n)]) = ", set([i for i in range(n)]))
                # print("Set(badlist) = ",set(badlist))
                bad = 1
        if bad == 1:
            return []

        List = []
        for i in range(0, n):
            List = List + res[i]
        A1 = matrowex(A, List)
        B1 = matrowex(b1, List)
        clf = linear_model.LinearRegression(fit_intercept=False)
        clf.fit(A1, B1)
        eps = np.square(clf.predict(A1) - B1).sum()
        if eps > ep:
            return []
        else:
            return [{tuple([i for i in range(n)])}]
    # recursive case
    result_list = []
    # when n th ball goes to a non-empty box
    for case in balls_in_boxes3(n - 1, m, badcom, res, A, b1, ep):
        for box in case:
            # make the n th ball goes to every non-empty box and add the case into result list
            copy_case = case.copy()
            temp_box_list = list(box)
            temp_box_list.append(n - 1)
            bad = 0
            for badlist in badcom:
                if set(temp_box_list) >= set(badlist):
                    bad = 1
            if bad == 0:
                List = []
                for i in temp_box_list:
                    List = List + res[i]
                A1 = matrowex(A, List)
                B1 = matrowex(b1, List)
                clf = linear_model.LinearRegression(fit_intercept=False)
                clf.fit(A1, B1)
                eps = np.square(clf.predict(A1) - B1).sum()
                if eps <= ep:
                    copy_case.remove(box)
                    copy_case.add(tuple(temp_box_list))
                    result_list.append(copy_case)
    # when n th ball goes to an empty box
    for case in balls_in_boxes3(n - 1, m - 1, badcom, res, A, b1, ep):
        # make a box containing n th ball and add it into every case set
        case.add((n - 1,))
        result_list.append(case)
    return result_list


def merge_cluster_tol2(res, A, b1, num_mode, ep):
    print("Before Clustering starts, lets see the segments ordering")
    print(res)

    leng = 0
    for i in range(0, len(res)):
        leng = leng + len(res[i])
    print ("leng the segmented data =", leng)
    print ("Maximum sos value ep * leng =", ep * leng)
    # print ("len(res) Number of segments are =", len(res))
    badcom = []  # This set is the set Pruned
    for i, j in itertools.combinations(range(len(res)), 2):
        A1 = matrowex(A, res[i] + res[j])
        B1 = matrowex(b1, res[i] + res[j])
        clf = linear_model.LinearRegression(fit_intercept=False)
        clf.fit(A1, B1)
        eps = np.square(clf.predict(A1) - B1).sum()
        # print('(',i,',',j,'):',eps)
        # if eps/(len(res[i])+len(res[j])) > ep:
        if eps > ep * leng:
            badcom.append([i, j])
    # print(badcom)
    print("badcom size=", len(badcom))
    print("badcom =", badcom)

    result_list = balls_in_boxes3(len(res), num_mode, badcom, res, A, b1, ep * leng)
    n = len(result_list)
    # print(len(res),'segs')
    print(n,' class')        # if result_list len is 0 meaning PruningSearch algorithm fails to cluster into classes
    # print(num_mode, ' Modes')

    # if (n==0):
    #    result_list = res
    #    print("len of res=",len(res))
    #    n=len(res)

    ListT = []
    for i in range(0, n):
        listT = []
        for j in range(0, num_mode):
            listt = []
            for k in list(list(result_list[i])[j]):
                listt = listt + res[k]
            listT.append(listt)
        ListT.append(listT)
    sq_sum_list = []
    # print ("printing after declaration")
    # print ("value of n=", n)
    # print ("value of num_mode=", num_mode)

    for i in range(0, n):
        sq_sum = 0
        for j in range(0, num_mode):
            A1 = matrowex(A, ListT[i][j])
            B1 = matrowex(b1, ListT[i][j])
            clf = linear_model.LinearRegression(fit_intercept=False)
            clf.fit(A1, B1)
            # sq_sum = sq_sum + rel_diff_mat(clf.predict(A1),B1)
            sq_sum += np.square(clf.predict(A1) - B1).sum()
            # print("Printing sq_sum = ", sq_sum)
        sq_sum_list.append(sq_sum)
    # print ("sq_sum_list = ", sq_sum_list)
    label = sq_sum_list.index(min(sq_sum_list))
    # for i in range(0,len(sq_sum_list)):
    #     print(i,':',sq_sum_list[i],'\n')
    # print(result_list[label])
    P = ListT[label]
    clfs = []
    G = []
    for mode in P:
        A1 = matrowex(A, mode)
        B1 = matrowex(b1, mode)
        clf = linear_model.LinearRegression(fit_intercept=False)
        clf.fit(A1, B1)
        clfs.append(clf)
        G.append(clf.coef_)

    return P, G
