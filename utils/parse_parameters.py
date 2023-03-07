"""
This module is takes a filename as input and parse the file line-by-line and word-by-word to create a data structure of
as a list of trajectories to be passsed as input to the learning algorithm.
"""

import numpy as np

def read_command_line(argv):
    """
    This function parse all the command line arguments and creates a dictionary data type with (key, value) pair
     where key is the name of the parameter and value contains the corresponding value of the parameter.

    :param argv: is the list of command line values obtained using the sys.argv. The order of the arguments
    :return: a python dictionary containing (key, value) pair of all the parameters
    """

    n = len(argv)
    # print("Total number of arguments is ", n)
    if (n < 17):
        print("Missing or Invalid Arguments list!!")
        print("Valid syntax: run_tests.py   InputDataFilename   OutputDataFilename   nModes   clustering-algo   "
              "ODE-order   boundary-order   cluster-error   parameter_threshold1   parameter_threshold2 "
              "simulation-time-horizon     number-of-samples    size-of-inputVar   size-of-outputVar   variable-types "
              " pool-values   maximum-value-ode-speedup    isInvariant")
        # variable_types: will describe the types of variable for annotation  Eg.: "x0=t4, x1=t3, x2=t4, x3=t1, x4=t2"
        # pool_values: values of type=t2, which has pool of values. This is defined here eg, "v4={10, 20, 30, 40}"

        # when clustering-algo==dtw:  parameter_threshold1 is correl_threshold and parameter_threshold2 = distance_threshold
        # when clustering-algo==dbscan:  parameter_threshold1 is dbscan-eps-dist and parameter_threshold2 = dbscan-min-samples
        # DTW clustering algo also has parameter simulation-time-horizon and number-of-samples
        # InputDataFilename has time serise input/output values, with 1st column time, followed by input and output variables in the same order
        # size-of-inputVar   tells us how many columns after time(1st column) are input variables' values
        # size-of-outputVar  tells us how many columns (after time, followed by input) have values of output variables
        # maximum-value-ode-speedup: parameter for improving performance by pruning all segments above this maximum value, for ODE computation
        # isInvariant: user's choice for computing/ignoring invariant computation. 0 and 1:enabled computing; 2: disabled computing
        '''
        Variable Types are defined as follows:
        t1: continuous, of the form x0' := x0 
        t2: assigned from the pool of values, eg., x4' := 0 or x4' = 1 
        t3: random, eg. piecewise constant variable, we do not learn this variable
        t4: unknown constant is assigned. we use linear regression to learn this for eg., x2' := 14.7
        t5: linear, of the form x0' := 0.5 x0 + 1.5 x1. we use linear regression to learn this
        '''
        return

    # print("Total arguments passed:", n)     #total is 16
    # print("\nName of Python script:", sys.argv[0])
    # print("Input Data Filename:", sys.argv[1])
    # print("Output Data Filename:", sys.argv[2])
    # print("Number of Modes:", sys.argv[3])
    # print("Clustering Algorithm:", sys.argv[4])
    # print("ODE Degree:", sys.argv[5])
    # print("Boundary Degree:", sys.argv[6])
    # print("Cluster-error:", sys.argv[7])
    # print("parameter_threshold1:", sys.argv[8])
    # print("parameter_threshold2:", sys.argv[9])
    # print("simulation-time-horizon:", sys.argv[10])
    # print("number-of-samples:", sys.argv[11])
    # print("size-of-inputVar:", sys.argv[12])
    # print("size-of-outputVar:", sys.argv[13])
    # print("variable-types:", sys.argv[14])
    # print("pool-values:", sys.argv[15])
    # print("\n")

    # for i in range(1, n):
    #    print(sys.argv[i], end=" ")
    # filename = "naijun/YLIST1_0_tList.txt" #sys.argv[1]  this is working
    # filename = "naijun/NewFile.txt"  #sys.argv[1]      YLIST5_0_tList.txt is also working
    filename = argv[1]
    list_of_trajectories, position = parse_trajectories(filename)

    t_list, y_list = list_of_trajectories[0] # in our case there is only one tuple
    system_dim = y_list[0].shape[1]  # the system dimension excluding the time-variable
    stepsize = t_list[0][2] - t_list[0][1]  # = 0.1
    print("\nComputed Step-size = ", stepsize)

    # outputfilename = "OKamit.txt" #sys.argv[2]  # filename = "outfilename.txt"
    outputfilename = argv[2]  # filename = "outfilename.txt"
    num_mode = int(argv[3])  # Number of Modes or Locations
    # method= 'piecelinear'    #had bug especially the clustering:PruneSearching process returns []
    # method = 'kmeans'       had tested for a couple of inputdata files works well
    # method = 'dbscan'
    method = argv[4]
    maxorder = int(argv[5])  # maximum degree/order of the polynomial ODE. A value of 1 makes it linear and 2 or more for nonlinear
    # print("Polynomial Degree order: ", maxorder)
    # print("num_mode:", num_mode)
    # print("method:", method)

    # gene1 = generate_complete_polynomial(2, 3)
    # print('gene1 is ', gene1)
    # print('gene1.shape[0] is ', gene1.shape[0]) # number of rows
    # print('gene1.shape[1] is ', gene1.shape[1]) # number of columns
    # return

    boundary_order = int(argv[6])  # This is also important to determine the boundary polynomial
    # Note: implementation missing for boundary_order = 2 for num_mode=2 for dimensions above 2.
    #  and completely missing boundary_order = 2 for num_mode=3
    # The guard/boundary is obtained using SVM (Linear Regression) which is of the form:
    # For boundary_order=1
    # y(w,x)=w0 + w1 x1 + ... + wp xp; where w0 is the intercept and w1...wp are coefficients
    # For boundary_order=2
    # y(w,x)=w0 + w1 x1 + w2 x2 + w3 x1 x2 + w4 x1^2 + w5 x2^2; where w0 is the intercept and x1 and x2 are variables

    # num_mode = 2 #Taken as user input

    ep = float(argv[7])   #0.01   # Maximal sum of square error tolerated in every cluster. Helps in playing with segmentation
    # print("ep input = ",ep)
    # ********** arguments for the clustering algorithm. Both for dbscan and DTW
    # dbscan_eps_dist = float(sys.argv[8])   #0.01   # Maximal sum of square error tolerated in every cluster. Helps in playing with segmentation
    # dbscan_min_samples = float(sys.argv[9])   #0.01   # Maximal sum of square error tolerated in every cluster. Helps in playing with segmentation
    parameter_threshold1 = float(argv[8])  # hold parameter values for dbscan or dtw
    parameter_threshold2 = float(argv[9])  # hold parameter values for dbscan or dtw
    dtw_simu_timeHorizon = float(argv[10])
    dtw_number_of_samples = int(argv[11])

    # ******** Argument for total size of input and output variables in the given trajectories  *****************
    size_of_input_variables = int(argv[12])
    size_of_output_variables = int(argv[13])

    # ******** Parsing the command line argument variable-type and pool-values into a list *****************
    variable_types = argv[14] # Eg.: "x0=t4, x1=t3, x2=t4, x3=t1, x4=t2"
    pool_values = argv[15]   # Eg.:  "x2={10,20,30,40} & x4={14.7,12.5}"
    # ******** ******** ******** ******** ******** ******** ******** ******** ********

    maximum_ode_prune_factor = int(argv[16])
    isInvariant = int(argv[17]) # enabled invariant computing? 0 and 1:enabled; 2:disabled

    variableType_datastruct = [] # structure that holds [var_index, var_name, var_type, pool_values]
    for i in range(0, system_dim):   #create and initialize the datastruct
        variableType_datastruct.append([i, "x" + str(i), "", ""])
    # print ("data = ", variableType_datastruct)
    # print("v_type:",variable_types, "and pool_val:",pool_values,"end")
    # print("Length of v_type:",len(variable_types), "and Length of pool_val:",len(pool_values),"end")

    if len(variable_types) >= 1:    # parse only when values are supplied from command line
        str_var_types = variable_types.split(",")
        # str_var_types = variable_types.split(",")
        # print("Length of str_var_types:")
        # print(len(str_var_types))
        for i in str_var_types:
            str_i_values = i.split("=")
            varName = str_i_values[0].strip()     # trim or remove whitespaces
            varType = str_i_values[1]
            # print("Var Name: ", varName, " var type: ", varType)
            for val in variableType_datastruct:
                if varName in val:
                    index = val[0]
                    variableType_datastruct[index][2] = varType
                    # print("Index=", val[0])
    # print ("data again = ", variableType_datastruct)

    if len(pool_values) >= 1:    # parse only when values are supplied from command line
        str_pool_values = pool_values.split(" & ")  # Eg.:  "x2={10,20,30,40} & x4={14.7,12.5}"
        # print("Pool values:", str_pool_values)
        for i in str_pool_values:  # Eg.:  "x2={10,20,30,40}"
            str_i_values = i.split("=")  # Eg.:  "['x2', '{10,20,30,40}']"
            varName = str_i_values[0]
            varValues = str_i_values[1]  # Eg.:  '{10,20,30,40}'
            size=len(varValues)
            varValues = varValues[1:size-1]
            # print("Var Name: ", varName, " var Values: ", varValues)
            varValues = [float(x) for x in varValues.split(",")]
            for val in variableType_datastruct:
                if varName in val:
                    index = val[0]
                    variableType_datastruct[index][3] = varValues

    # print ("data again = ", variableType_datastruct)

    '''
    See the example output after parsing variable_types ="x0=t4, x1=t3, x2=t2, x3=t1, x4=t2" and pool_values="x2={10,20,30,40} & x4={14.7,12.5}"
    variableType_datastruct =  [[0, 'x0', 't4', ''], [1, 'x1', 't3', ''], [2, 'x2', 't2', [10.0, 20.0, 30.0]], [3, 'x3', 't1', ''], [4, 'x4', 't2', [14.7, 12.5]]]
    '''
    # The structure variableType_datastruct will be empty is no argument is supplied
    # ******** Parsing argument variable-type and pool-values into a list *****************


    # ep = 0.0001     # Trying more precision for shared-gas-burner model
    mergeep = 0.01     # this is not used in the piecelinear Algorithm
    # mergeep = 0.0001     # Trying more precision for shared-gas-burner model
    # mergeep = 0.9   # or can try even higher value then this for AbstractFuelControl





    parameters = (stepsize, maxorder, boundary_order, num_mode, ep, mergeep, parameter_threshold1, parameter_threshold2,
    size_of_input_variables, size_of_output_variables, method, position, variableType_datastruct,
    maximum_ode_prune_factor, isInvariant)

    return filename, outputfilename, list_of_trajectories, parameters



def parse_trajectories(input_filename):
    """
    This is a special case function, because the argument input_filename is a filename which contains all
    trajectories concatenated into a single file.
    This function parse this single file containing concatenated trajectories. Note that there is no symbols or special
    marker to separate trajectories. Each trajectory data are appended one below the other. The only indication is the
    value of the first column (contains the simpling time value). For a new trajectory this column value is always 0.0
    i.e. the start time of a new trajectory/simulation.

    :param input_filename: is the input file name containing trajectories.

    :return:
        list_of_trajectories: Each element of the list is a trajectory. A trajectory is a 2-tuple of (time, vector), where
            time: is a list having a single item. The item is the sampling time, stored as a numpy.ndarray having structure
                as (rows, ) where rows is the number of sample points. The dimension cols is empty meaning a single dim array.
            vector: is a list having a single item. The item is a numpy.ndarray with (rows,cols), rows indicates the number of
                points and cols as the system's dimension. The dimension is the total number of variables in the trajectories
                including both input and output variables.
        stepsize: is the sampling time period between two points.
        system_dimension: is the dimension (input + output variables) of the system whose trajectory is being parsed.

    """


    t_list = []
    y_list = []
    t_tmp = []
    y_tmp = []
    list_of_trajectories = []
    y_list_per_trajectory = []
    y_array_per_trajectory = [] # will be converted to numpy.array
    t_list_per_trajectory = []
    t_array_per_trajectory = [] # will be converted to numpy.array

    seqCount = 0
    with open(input_filename, 'r') as file:
        for line in file:
            colum = 1
            cc = 0
            all_y_pts = []
            for word in line.split():
                if colum == 1:
                    if float(word) == 0.0:  # this check will be enabled only on new trajectory series
                        if seqCount != 0:  # meaning we found the next t=0 and not the starting t=0
                            y_array_per_trajectory = np.array(y_list_per_trajectory) # convert list to array in time complexity O(n)
                            y_list.append(y_array_per_trajectory) # store the vector array into a list

                            t_array_per_trajectory = np.array(t_list_per_trajectory) # convert list to array in time complexity O(n)
                            t_list.append(t_array_per_trajectory) # store the array of time values into a list

                            trajectory = (t_list, y_list)   # create a tuple of (time and vector)
                            list_of_trajectories.append(trajectory) # create a list of trajectories

                            #  reset or re-initialize variables for next iterations
                            y_list = []
                            t_list = []
                            y_list_per_trajectory = []
                            t_list_per_trajectory = []

                    t_list_per_trajectory.append(float(word))
                    colum += 1
                else:
                    all_y_pts.append(float(word))
                    cc += 1

            y_list_per_trajectory.append(all_y_pts)
            seqCount += 1

    file.close()
    # Note: the last trajectory to be added now
    y_array_per_trajectory = np.array(y_list_per_trajectory)  # convert list to array in time complexity O(n)
    y_list.append(y_array_per_trajectory)  # store the vector array into a list

    t_array_per_trajectory = np.array(t_list_per_trajectory)  # convert list to array in time complexity O(n)
    t_list.append(t_array_per_trajectory)  # store the array of time values into a list

    trajectory = (t_list, y_list)  # create a tuple of (time and vector of list of single item, where item is np.array)
    list_of_trajectories.append(trajectory)  # create a list of trajectories

    '''
    print("y_list = ", y_list)
    print("totalPoints = ", totalPoints)
    print("totalPoints in y_list = ", len(y_list))

    print ("Type of t_list = ", type(t_list))
    print("Type of type(t_list[0]) = ", type(t_list[0]))
    print("Type of t_list[0].shape= ", t_list[0].shape)
    print("Type of t_list[0].shape[0]= ", t_list[0].shape[0])
    # print("Type of t_list[0].shape[1]= ", t_list[0].shape[1]) Error coz only one dimension available and no cols
    print ("t_list = ", t_list)

    print("Type of y_list = ", type(y_list))
    print("Type of type(y_list[0]) = ", type(y_list[0]))
    print("Type of y_list[0].shape= ", y_list[0].shape)   # prints the shape of y_list (15015, 5) = (rows, colmns)
    print("Type of y_list[0].shape[1]= ", y_list[0].shape[1])   # shape[0] for rows or records and shape[1] is cols or dimension
    # print("y_list = ", y_list)
    '''

    stepsize = t_list[0][2] - t_list[0][1]  # = 0.1 Computing the step-size from the sampled trajectories
    # print("\nComputed Step-size = ", stepsize)

    system_dimension = y_list[0].shape[1]

    return list_of_trajectories, stepsize, system_dimension


