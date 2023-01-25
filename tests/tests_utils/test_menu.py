import os
import unittest

# To execute this test from the project folder "learnHA" type the command
# amit@amit-Alienware-m15-R4:~/MyPythonProjects/learningHA/learnHA$ python -m unittest discover -v


class TestLearnHA(unittest.TestCase):

    def test_commandline_parser(self):

        parameters = {}
        print("Running test commandline_parser module")
        os.system("python run.py --help")
        pass


if __name__ == '__main__':
    unittest.main()