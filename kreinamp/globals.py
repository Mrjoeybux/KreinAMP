import os

ROOTPATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
TEST_DATA_PATH = os.path.join(ROOTPATH, "data", "independent_test", "peptides.csv")

"""
To reproduce the results from the publication, use the following settings:

RANDOM_STATE = 0
SHUFFLE_DATA = True
TEST_SIZE = 0.3

"""
RANDOM_STATE = 0
SHUFFLE_DATA = True
TEST_SIZE = 0.3

HPC_PYTHON_PATH = None
"""
If running on HPC, uncomment the line below and set to it to the location of your python interpretter on the HPC
"""
# HPC_PYTHON_PATH = "/gpfs01/home/psxjr4/.conda/envs/krein/bin/python"
