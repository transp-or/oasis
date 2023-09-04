
import joblib
import time
import warnings
import math

import numpy as np
import pandas as pd
import metropolis_hastings as mh

from datetime import datetime
from joblib import Parallel, delayed, cpu_count

from metropolis_hastings import random_walk
from helper_func import schedule_to_pandas
from activity import Schedule
from settings import DESIRED_TIMES, DEFAULT_MODES, DEFAULT_MH_PARAMS, DEFAULT_OPERATORS, DEFAULT_VARIABLES, DEFAULT_ACTIVITIES, DEFAULT_P_OPERATORS

from typing import List, Dict, Tuple, Optional, Union


warnings.filterwarnings('once')



class ChoiceSetGenerator():
    """
    This class is used to generate choice sets of Schedule objects for given individuals.

    Attributes:
    -------------------
    - schedules: List of Schedule objects
    - param_file: location of parameters for the target distribution
    - n_alt: number of alternatives in the choice set
    - mh_params: dictionary containing parameters for the random walk
    - activities: list of activities
    - operators: list of operators
    - p_operators: probabilities of operators
    - modes: list of modes
    - locations: list of locations
    - variables: list of variables for target distribution
    - outfile: location of file to save result



    Methods:
    ------------------
    - generate_set: generates choice set for a given individual.
    - run: Run metropolis_hastings algorithm for full dataset.
    - run: Run metropolis_hastings algorithm for full dataset, using parallel processing.
    - compute_sample_correction: Returns the corrective term for the utility function
    - train_test_sets: Creates train and test Dataframes to use in Biogeme.

    """
    def __init__(self, schedules:List, params_file:str,n_alt:int = 10, mh_params:List = DEFAULT_MH_PARAMS,
     activities:Optional[List] = DEFAULT_ACTIVITIES, operators:Optional[List] = DEFAULT_OPERATORS, proba_operators:Optional[List] = None,
     modes: Optional[List] = DEFAULT_MODES, locations:Optional[List] = None, variables: List = DEFAULT_VARIABLES, outfile:str ='choice_set.joblib', seed:int = 42, **kwargs):

        self.schedules = schedules
        self.params_file = params_file
        self.n_alt = n_alt
        self.mh_params = mh_params
        self.activities = activities
        self.operators = operators
        self.p_operators = proba_operators
        self.modes = modes
        self.outfile = outfile

        self.locations = locations

        self.variables = variables

        self.choice_sets = []
        self.accepted_operators = []
        self.acceptance_probas = []

        self.sample_corrections = []



    def generate_set(self, schedule: Schedule)-> Tuple[List, List, List]:
        """
        Generates choice set for a given individual.

        Parameters
        ----------
        - schedule : Schedule object

        Returns
        ----------
        Choice sets, accepted operators and probabilities.

        """
        choice_set = [schedule]
        list_op = []
        probas = []

        n_skip = self.mh_params["n_skip"]
        n_burn = self.mh_params["n_burn"]

        if not self.locations:
            all_locations = schedule.all_locations
        else:
            all_locations = self.locations

        steps = mh.random_walk(init_sched = schedule, operators=self.operators, p_operators = self.p_operators,
        list_act=self.activities, list_loc = all_locations, list_modes = self.modes, param_file = self.params_file, **self.mh_params)

        n = 0
        for state, op, pb in steps:
            if (n>n_burn) and (n%n_skip == 0):
                optype = op.optype
                if optype == "MetaOperator":
                    optype = op.meta_type

                list_op.append(optype)
                choice_set.append(state)
                probas.append(pb)

            n += 1
        return choice_set, list_op, probas

    def run(self) -> None:
        """
        Run metropolis_hastings algorithm for full dataset and saves choice sets, accepted operators and acceptance probabilities to file.
        """
        start = datetime.now()
        for i, schedule in enumerate(self.schedules):
            print(f"Starting generation for individual {i}.\n")

            choice_set, list_op, probas = self.generate_set(schedule)

            self.choice_sets.append(choice_set)
            self.accepted_operators.append(list_op)
            self.acceptance_probas.append(probas)

        end = datetime.now()
        print(f"Total runtime: {end-start}")

        joblib.dump([self.choice_sets,self.accepted_operators, self.acceptance_probas], self.outfile)


    def run_parallel(self, n_cpus:Optional[int] = None, verbose:int = 5)->None:
        """
        Run metropolis_hastings algorithm for full dataset using parallel processing. Saves choice sets, accepted operators and acceptance probabilities to file.

        Parameters
        ----------
        n_cpus: number of CPUs to use for the parallel process.
        verbose: gives frequency of progress ouptuts

        """

        if not n_cpus:
            n_cpus = cpu_count()

        delayed_output = [delayed(self.generate_set)(s) for s in self.schedules]
        results = Parallel(n_jobs=n_cpus, verbose=verbose)(delayed_output)
        joblib.dump(results, self.outfile)

        self.choice_sets, self.accepted_operators,self.acceptance_probas = results


    def compute_sample_correction(self, original_probas: List, unique_probas:List, k: int = 1)->List:
        """
        Returns the corrective term for the utility function, to estimate the model on the sampled choice set passed as input.
        See Ben-Akiva & Lerman (1985) "Discrete choice analysis", p.266

        Parameters
        ---------------
        - original_probas: list of probabilities for alternatives in the choice set (including duplicates)
        - unique_pobas: list of probabilities for unique alternatives in the choice set (excludeing duplicates)
        - k: proportionality constant
        """


        orig_len = [len(x) for x in original_probas]
        new_len = [len(x) for x in unique_probas]

        #Computing sampling proba - see Ben-Akiva and Lerman (1985) p. 266
        diff_len = [i-j+1 for i, j in zip(orig_len, new_len)]
        q = [np.log(k) + sum(unique_probas[i]) + diff_len[i]*np.log(sum(np.exp(unique_probas[i]))) for i in range(len(unique_probas))]

        sample_correction = [i - j for i, j in zip(q, unique_probas)]
        return sample_correction


    def train_test_sets(self, k:int=1, train_ratio:float= 0.7) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        """
        Creates train and test Dataframes to use in Biogeme.

        Parameters:
        --------------
        - k: proportionality constant for sample correction
        - train_ratio: train test split (default: 70% of observations will be used for the train set)

        Returns:
        ---------------
        - Train dataset in wide and long format, Test dataset
        """

        draws_proba = [[1] for x in self.choice_sets]
        for i, proba in enumerate(draws_proba):
            proba.extend(self.acceptance_probas[i])
            draws_proba[i] = proba

        #Check unique alternatives (to compute sample probability)
        unique_id_draws =[[idx for idx, item in enumerate(choice_set) if item not in choice_set[:idx]] for choice_set in self.choice_sets]

        unique_draws = []
        unique_draws_proba = [[1] for x in unique_id_draws]

        n = 0
        for ids, proba in zip(unique_id_draws, unique_draws_proba):
            list_probas = [self.acceptance_probas[n][i-1] for i in ids[1:]]

            proba.extend(list_probas)
            unique_draws_proba[n] = proba
            unique_draws.append([self.choice_sets[n][i-1] for i in ids[1:]])
            n += 1

        self.sample_corrections = self.compute_sample_correction(draws_proba, unique_draws_proba,k)

        #Check how many choice sets have enough unique alternatives for the estimation
        valid_ids = [i for i, elem in enumerate(unique_draws) if len(elem) >= self.n_alt]
        other_ids = [i for i, _ in enumerate(unique_draws) if i not in valid_ids]

        n_train = math.ceil(train_ratio*len(self.choice_sets))

        if len(valid_ids) >= n_train:
            train_ids = np.random.choice(range(len(valid_ids)), n_train, replace = False)
            test_ids = other_ids.extend([i for i in valid_ids if i not in train_ids])
        else:
            train_ids = valid_ids + other_ids
            test_ids = None
            warnings.warn("The train/test ratio could not be satisfied with the requested choice set size. Consider a lower number of alternatives or increasing the number of draws for the random walk. ")


        formatted_train = []
        formatted_test = []

        for t in train_ids:
            cs = [unique_draws[t][0]] #add chosen alternative
            cs.extend(list(np.random.choice(unique_draws[t][1:], self.n_alt-1)))
            formatted_train.append([schedule_to_pandas(sched) for sched in cs])

        if test_ids:
            for t in test_ids:
                cs = unique_draws[t]
                formatted_test.append([schedule_to_pandas(sched) for sched in cs])


        train_probs = [prob for i, prob in enumerate(self.sample_corrections) if i in train_ids]

        #Add an ID to each alternative in the choice set, and creating a single list with all the alternatives for each individual

        for j, list_cs in enumerate(formatted_train):
            for i, cs in enumerate(list_cs):
                cs['alt_id'] = i
                cs['prob_corr'] = train_probs[j][i]

        lng_cs = [pd.concat(formatted) for formatted in formatted_train]

        #Add choice alternative and individual IDs
        for i, cs in enumerate(lng_cs):
            cs.reset_index(drop = True, inplace = True)
            cs['obs_id'] = i
            cs['choice'] = cs['alt_id'].apply(lambda x: 1 if i == 0 else 0)

        #Create choice sets in long format
        db_long = pd.concat(lng_cs).reset_index(drop = True)


        df_activities=db_long.groupby(['obs_id', 'alt_id', 'act_label']).agg(start_time=('start_time','min'),duration=('duration','sum'),choice=('choice','mean'), prob_corr = ('prob_corr', 'mean')).reset_index()

        df_activities['participation'] = 1
        df_long = df_activities[['obs_id','alt_id','choice', 'prob_corr']].drop_duplicates()
        for at in self.activities:
            df_long = df_long.merge(df_activities[df_activities.act_label==at].drop(['choice','act_label', 'prob_corr'],axis=1).rename(columns={v : f'{at}:{v}' for v in self.variables}), how='left', on=['obs_id','alt_id'])
            df_long.fillna(0,inplace=True)

        for at in self.activities:

            if at not in ['home', 'dawn', 'dusk']:
                desired_st = DESIRED_TIMES[at]['desired_start_time']
                desired_dur = DESIRED_TIMES[at]['desired_duration']
            else:
                desired_st = 0
                desired_dur = 0

            st_diff = (df_long[f'{at}:start_time'] - desired_st) * df_long[f'{at}:participation']
            df_long[f'{at}:early'] = ((st_diff>=-12) & (st_diff<=0))*(-st_diff) + ((st_diff>=12) & (st_diff<=24))*(24-st_diff)
            df_long[f'{at}:late'] = ((st_diff>=0)&(st_diff<12))*(st_diff) + ((st_diff>=-24) & (st_diff<-12))*(24+st_diff)

            d_diff = (df_long[f'{at}:duration'] - desired_dur)* df_long[f'{at}:participation']

            df_long[f'{at}:short'] = (d_diff<=0)*(-d_diff) + 0 #Add zero to make sure that every value is strictly positive
            df_long[f'{at}:long'] = (d_diff>=0)*d_diff + 0 #Add zero to make sure that every value is strictly positive

        #Add zero to make sure that every value is strictly positive
        df_long = df_long.apply(lambda x: (x + 0) if (x.dtypes == 'float64') else x, axis=0)

        #Convert from long to wide
        df_wide = pd.DataFrame(df_long['obs_id'].drop_duplicates())
        df_wide['choice'] = 0

        for i in list(df_long.alt_id.unique()):
            tmp = df_long[df_long.alt_id==i].rename(columns={c:f'{c}_{i}' for c in df_long.columns if c not in ['obs_id', 'alt_id', 'choice']})
            df_wide = df_wide.merge(tmp.drop(['alt_id', 'choice'],axis=1), how='left', on=['obs_id'])
            df_wide.choice += i*df_wide.choice[~df_wide.choice.isnull()]

        df_wide.dropna(axis = 0, inplace = True)

        #Delete temporary dataframes from namespace
        del db_long, tmp, df_activities

        return df_wide, df_long, formatted_test
