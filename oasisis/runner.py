import joblib
import json

from input_data import data_reader
from error_terms import GaussianError, EVError
from simulation import MIP


DATA = joblib.load('../data/example_data.joblib')
TT = joblib.load('../data/example_tt.joblib')
PARAMS = json.load(open('../data/example_parameters.json','r'))

UTILITY_PARAMS = {
    'error_w': GaussianError(),
    'error_x': GaussianError(),
    'error_d': GaussianError(),
    'error_z': GaussianError(),
    'error_ev': EVError()
    }


N_ITER = 1

def main():
    """Run simulation"""

    dataset = data_reader(DATA,PARAMS)
    new_simulation = MIP(dataset, UTILITY_PARAMS,TT)

    results = new_simulation.run(n_iter = N_ITER, verbose = 25)

    #visualise results
    results.plot(save_fig='png')
    results.plot_distribution(save_fig='png')
    results.compute_statistics(['home', 'work', 'leisure'])


if __name__ == '__main__':
    main()
