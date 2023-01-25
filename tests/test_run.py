import unittest

import filecmp
import warnings
warnings.filterwarnings('ignore')   # disables FutureWarning in the use of clf.fit()

from infer_ha.infer_HA import infer_model
from infer_ha.model_printer.print_HA import print_HA
from utils.parse_parameters import parse_trajectories
from utils.commandline_parser import process_type_annotation_parameters


# To execute this test from the project folder "learnHA" type the command
# amit@amit-Alienware-m15-R4:~/MyPythonProjects/learningHA/learnHA$ python -m unittest discover -v


class TestLearnHA(unittest.TestCase):

    def test_runLearnHA_osci_withoutAnnotate(self):

        parameters = {}
        print("Running test runLearnHA module")

        parameters['input_filename'] = "data/simu_oscillator_2.txt"
        parameters['output_filename'] = "oscillator_2_withoutAnnotate_test.txt"

        parameters['clustering_method'] = 1
        parameters['methods'] = "dtw"

        parameters['ode_degree'] = 1
        parameters['modes'] = 4
        parameters['guard_degree'] = 1
        parameters['segmentation_error_tol'] = 0.1
        parameters['threshold_distance'] = 1.0
        parameters['threshold_correlation'] = 0.89
        parameters['dbscan_eps_dist'] = 0.01  # default value
        parameters['dbscan_min_samples'] = 2  # default value
        parameters['size_input_variable'] = 0
        parameters['size_output_variable'] = 2
        parameters['variable_types'] = ''
        parameters['pool_values'] = ''
        parameters['ode_speedup'] = 50
        parameters['is_invariant'] = 0
        parameters['stepsize'] = 0.01

        input_filename = parameters['input_filename']
        output_filename = parameters['output_filename']
        list_of_trajectories, stepsize, system_dimension = parse_trajectories(input_filename)
        # print("list of trajectories is ", list_of_trajectories)
        variableType_datastruct = []  # structure that holds [var_index, var_name, var_type, pool_values]
        if len(parameters['variable_types']) >= 1:  # if user supply annotation arguments
            variableType_datastruct = process_type_annotation_parameters(parameters, system_dimension)

        parameters['stepsize'] = stepsize  # we assume trajectories are sampled at fixed size time-step
        parameters['variableType_datastruct'] = variableType_datastruct
        P, G, mode_inv, transitions = infer_model(list_of_trajectories, parameters)
        # print("Number of modes learned = ", len(P))
        print_HA(P, G, mode_inv, transitions, parameters, output_filename)  # prints an HA model file

        backup_file = "data/test_output/oscillator_2_without_annotation.txt"
        test_generated_file = "oscillator_2_withoutAnnotate_test.txt"

        # shallow mode comparison: where only metadata of the files are compared like the size, date modified, etc.
        # result = filecmp.cmp(backup_file, test_generated_file)
        # print(result)
        # deep mode comparison: where the content of the files are compared.
        result = filecmp.cmp(backup_file, test_generated_file, shallow=False)
        print(result)
        self.assertTrue(result) # Fails if the output generated is not equal to the file stored in the data/test_output

        # pass


    def test_runLearnHA_osci_withAnnotate(self):

        parameters = {}
        print("Running test runLearnHA module with Oscillator model with type annotation")

        parameters['input_filename'] = "data/simu_oscillator_2.txt"
        parameters['output_filename'] = "oscillator_2_withAnnotate_test.txt"

        parameters['clustering_method'] = 1
        parameters['methods'] = "dtw"

        parameters['ode_degree'] = 1
        parameters['modes'] = 4
        parameters['guard_degree'] = 1
        parameters['segmentation_error_tol'] = 0.1
        parameters['threshold_distance'] = 1.0
        parameters['threshold_correlation'] = 0.89
        parameters['dbscan_eps_dist'] = 0.01  # default value
        parameters['dbscan_min_samples'] = 2  # default value
        parameters['size_input_variable'] = 0
        parameters['size_output_variable'] = 2
        parameters['variable_types'] = 'x0=t1,x1=t1'
        parameters['pool_values'] = ''
        parameters['ode_speedup'] = 50
        parameters['is_invariant'] = 0
        parameters['stepsize'] = 0.01

        input_filename = parameters['input_filename']
        output_filename = parameters['output_filename']
        list_of_trajectories, stepsize, system_dimension = parse_trajectories(input_filename)
        # print("list of trajectories is ", list_of_trajectories)
        variableType_datastruct = []  # structure that holds [var_index, var_name, var_type, pool_values]
        if len(parameters['variable_types']) >= 1:  # if user supply annotation arguments
            variableType_datastruct = process_type_annotation_parameters(parameters, system_dimension)

        parameters['stepsize'] = stepsize  # we assume trajectories are sampled at fixed size time-step
        parameters['variableType_datastruct'] = variableType_datastruct
        P, G, mode_inv, transitions = infer_model(list_of_trajectories, parameters)
        # print("Number of modes learned = ", len(P))
        print_HA(P, G, mode_inv, transitions, parameters, output_filename)  # prints an HA model file

        backup_file = "data/test_output/oscillator_2_with_annotation.txt"
        test_generated_file = "oscillator_2_withAnnotate_test.txt"

        # shallow mode comparison: where only metadata of the files are compared like the size, date modified, etc.
        # result = filecmp.cmp(backup_file, test_generated_file)
        # print(result)
        # deep mode comparison: where the content of the files are compared.
        result = filecmp.cmp(backup_file, test_generated_file, shallow=False)
        print(result)
        self.assertTrue(result) # Fails if the output generated is not equal to the file stored in the data/test_output

        # pass


if __name__ == '__main__':
    unittest.main()