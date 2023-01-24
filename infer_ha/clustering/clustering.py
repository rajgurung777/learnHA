"""
This wrapper module enables the selection of different approaches to the clustering algorithm.
Using this wrapper module, one can implement and test different approaches to the clustering algorithm.
Below we have three approaches:
1) 'piecelinear': the original approach in the paper by Jin et al.
2) 'dbscan': The standard DBSCAN clustering
3) 'dtw': The approach using the DTW algorithm.

The learning algorithm is currently designed by focusing on the DTW algorithm.
"""

from infer_ha.clustering.cluster_by_dtw import cluster_by_dtw
from infer_ha.clustering.cluster_by_others import dbscan_cluster, merge_cluster_tol2

def select_clustering(res, A, b1, clfs, Y, t_list, L_y, learning_parameters):
    """
    A wrapper module that enables the selection of different approaches to the clustering algorithm.

    :param res: is a list of the segmented trajectory (positions). Each item of the list res contains a list of values
        which are positions of points of trajectories. The size of the list res is the total number of segments
        obtained.
    :param A: For every point of a trajectory the coefficients of the monomial terms obtained using the \Phi
         function (or the mapping function) as mention in Jin et al. paper.
    :param b1: the derivatives of each point computed using the backward version of BDF.
    :param clfs: is a list. Each item of the list clfs is a list that holds the coefficients (obtained using linear
        regression) of the ODE of each segment of the segmented trajectories.
    :param Y: contains the y_list values for all the points except the first and last M points (M is the order in BDF).
    :param t_list: a single-item list whose item is a numpy.ndarray containing time-values as a concatenated list.
    :param L_y: is the dimension (input + output variables) of the system whose trajectory is being parsed.
    :param learning_parameters: is a dictionary data structure having the list of commandline arguments passed by the
        user for the learning algorithm.
    :return: The computed cluster and the coefficients of the polynomial ODE.
        P: is a list containing the list of positions of each cluster and
        G: is a list containing the list of the coefficients of the polynomial ODE.
    """

    # print("Clustering segmented points ...")
    maxorder = learning_parameters['ode_degree']
    num_mode = learning_parameters['modes']
    ep = learning_parameters['segmentation_error_tol']
    size_of_input_variables = learning_parameters['size_input_variable']
    method = learning_parameters['methods']
    maximum_ode_prune_factor = learning_parameters['ode_speedup']
    correl_threshold = learning_parameters['threshold_correlation']
    distance_threshold = learning_parameters['threshold_distance']
    dbscan_eps_dist = learning_parameters['dbscan_eps_dist']
    dbscan_min_samples = learning_parameters['dbscan_min_samples']

    P = []
    G = []
    # Choice of Clustering Algorithm
    if len(res) > num_mode:  # clustering is required only if segmentation finds more segments than required modes
        if method == "piecelinear":
            print("Running the Merge-clustering algorithm!!")
            P, G = merge_cluster_tol2(res, A, b1, num_mode, ep)  # This is Algo-2:InferByMerge function in Jin et al.
            # Todo: note this approach does not scale well in clustering high number of segments into low modes.

    if method == "dbscan":
        print("Running DBSCAN clustering algorithm!!")
        P, G = dbscan_cluster(clfs, res, A, b1, num_mode, dbscan_eps_dist, dbscan_min_samples)
    if method == "dtw":
        print("Running clustering using  DTW algorithm!!")
        P, G = cluster_by_dtw(res, A, b1, Y, t_list, L_y, correl_threshold,
                              distance_threshold, size_of_input_variables, maximum_ode_prune_factor) # t_list only used for debugging using plot

    return P, G
