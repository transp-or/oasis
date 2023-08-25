import joblib
import json

from input_data import data_reader
from error_terms import GaussianError, EVError
from simulation import MIP


DATA = joblib.load('data/example_data.joblib')
TT = joblib.load('data/example_tt.joblib')
PARAMS = json.load('data/example_parameters.json')

UTILITY_PARAMS = {
    'error_w': GaussianError(),
    'error_x': GaussianError(),
    'error_d': GaussianError(),
    'error_z': GaussianError(),
    'error_ev': EVError()
    }


N_ITER = 10

def main():

    dataset = data_reader(DATA,PARAMS)
    new_simulation = MIP(dataset, UTILITY_PARAMS,TT)

    results = new_simulation.run(n_iter = N_ITER, verbose = 5)


if __name__ == '__main__':
    main()
