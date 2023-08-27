import joblib
import json

from input_data import data_reader
from error_terms import GaussianError, EVError
from simulation import MultidayMIP


DATA = joblib.load('../data/example_data_multiday.joblib')
TT = joblib.load('../data/example_tt_multiday.joblib')
PARAMS = json.load(open('../data/example_parameters_multiday.json', 'r'))

UTILITY_PARAMS = {
    'error_w': GaussianError(),
    'error_x': GaussianError(),
    'error_d': GaussianError(),
    'error_z': GaussianError(),
    'error_ev': EVError()
    }

N_ITER = 5
N_DAYS = 7
DAY_INDEX = [*range(1,8)]
SETTINGS =  {'optimality_target': 3, 'time_limit': 150}


def main():
    """Run multiday simulation"""

    dataset = data_reader(DATA,PARAMS)
    new_simulation = MultidayMIP(dataset, UTILITY_PARAMS,TT, n_days=N_DAYS, day_index=DAY_INDEX, **SETTINGS)

    results = new_simulation.run(N_ITER,verbose = 2)

    #visualise results
    results.plot(plot_iter = 2, save_fig='png') #plot iteration 2
    results.plot_distribution(days = [*range(1,6)], figure_size= [7,4], save_fig= 'png') #time of day distribution for weekdays
    results.plot_distribution(days = [6,7], figure_size= [7,4], save_fig= 'png') #time of day distribution for weekends
    results.compute_statistics(['home', 'work', 'leisure'], days = [*range(1,6)]) #stats for weekdays
    results.compute_statistics(['home', 'work', 'leisure'], days = [6,7]) #stats for weekends



if __name__ == '__main__':
    main()
