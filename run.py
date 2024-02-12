import sys
import time
import warnings
warnings.filterwarnings('ignore')   # disables FutureWarning in the use of clf.fit()

from infer_ha import infer_HA as learnHA     #infer_model, svm_classify
from infer_ha.model_printer.print_HA import print_HA
from utils.parse_parameters import parse_trajectories
from utils.commandline_parser import read_commandline_arguments, process_type_annotation_parameters

methods = ['dbscan', 'piecelinear', 'dtw']


def runLearnHA():  # Calling the implementation from project BBC4CPS
    '''
    Hints:
        To analysis the segmentation output: in the file "learnHA/infer_ha/infer_HA.py" uncomment the line 24 having "plot_segmentation_new(segmented_traj, L_y, t_list, Y, stepM)"
        To analysis the clustering output: in the file "learnHA/infer_ha/infer_HA.py" uncomment the line 136 having "plot_after_clustering(t_list, L_y, P_modes, Y, stepM)"
    @return:
    '''
    # input_filename, output_filename, list_of_trajectories, learning_parameters = read_command_line(sys.argv)
    parameters = read_commandline_arguments()   # reads the command line values also can use -h to see help on usages

    num_mode = parameters['modes']
    input_filename = parameters['input_filename']
    output_filename = parameters['output_filename']
    default_user_stepsize = parameters['stepsize']
    list_of_trajectories, stepsize, system_dimension = parse_trajectories(input_filename)
    step_size = default_user_stepsize

    # Giving priority to user selected step-size and not the step-size in the trajectories
    if default_user_stepsize == 0.01:   #default is 0.01
        step_size = stepsize    # obtained step-size from the trajectories
    else:
        step_size = default_user_stepsize  # user provided some step-size

    # print("stepsize = ", step_size)
    # stepsize = 0.01
    # print("list of trajectories is ",list_of_trajectories)
    variableType_datastruct = []  # structure that holds [var_index, var_name, var_type, pool_values]
    if len(parameters['variable_types']) >= 1:  # if user supply annotation arguments
        variableType_datastruct = process_type_annotation_parameters(parameters, system_dimension)

    parameters['stepsize'] = step_size   # we assume trajectories are sampled at fixed size time-step
    parameters['variableType_datastruct'] = variableType_datastruct
    # parameters['position'] = position
    end = time.time()  # creation of variable end

    start = time.time()
    #################################################################################################
    # P, G, mode_inv, transitions = learnHA.infer_model(list_of_trajectories, learning_parameters)
    P_modes, G, mode_inv, transitions, position = learnHA.infer_model(list_of_trajectories, parameters)
    # Note P is the Segmented data. G is the coefficients of the ODE and boundary is the guard conditions
    #################################################################################################
    end = time.time()
    total_learning_time = end - start
    # print("******************* How is this happening *******************")
    # print("inferring_time = ", total_learning_time)
    # print("Number of modes chosen =", num_mode)
    # print("Number of modes learned = ", len(P_modes))

    print_HA(P_modes, G, mode_inv, transitions, position, parameters, output_filename)   # prints an HA model file inside the folder outputs/

    return

# runLearnHA()

if __name__ == '__main__':
    runLearnHA()