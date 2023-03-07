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
    # input_filename, output_filename, list_of_trajectories, learning_parameters = read_command_line(sys.argv)
    parameters = read_commandline_arguments()   # reads the command line values also can use -h to see help on usages
    # Todo: (1) remove position from the parameters and do it in the main infer_HA process

    num_mode = parameters['modes']
    input_filename = parameters['input_filename']
    output_filename = parameters['output_filename']

    list_of_trajectories, stepsize, system_dimension = parse_trajectories(input_filename)
    # print("list of trajectories is ",list_of_trajectories)
    variableType_datastruct = []  # structure that holds [var_index, var_name, var_type, pool_values]
    if len(parameters['variable_types']) >= 1:  # if user supply annotation arguments
        variableType_datastruct = process_type_annotation_parameters(parameters, system_dimension)

    parameters['stepsize'] = stepsize   # we assume trajectories are sampled at fixed size time-step
    parameters['variableType_datastruct'] = variableType_datastruct
    # parameters['position'] = position
    end = time.time()  # creation of variable end

    start = time.time()
    #################################################################################################
    # P, G, mode_inv, transitions = learnHA.infer_model(list_of_trajectories, learning_parameters)
    P, G, mode_inv, transitions = learnHA.infer_model(list_of_trajectories, parameters)
    # Note P is the Segmented data. G is the coefficients of the ODE and boundary is the guard conditions
    #################################################################################################
    end = time.time()
    total_learning_time = end - start
    # print("******************* How is this happening *******************")
    print("inferring_time = ", total_learning_time)
    print("Number of modes chosen =", num_mode)
    print("Number of modes learned = ", len(P))

    print_HA(P, G, mode_inv, transitions, parameters, output_filename)   # prints an HA model file inside the folder outputs/

    return

# runLearnHA()

if __name__ == '__main__':
    runLearnHA()