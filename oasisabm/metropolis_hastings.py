import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from copy import deepcopy

from operators import Operator, OperatorFactory
from helper_func import activity_colors
from activity import Schedule
from typing import Dict, List, Optional, Union, Tuple, Generator


# -------------------------- RANDOM WALK -----------------------------------#

def random_walk(init_sched: Schedule, n_iter: int = 1000, operators: Optional[List] = None,
p_operators: Optional[List] = None, uniform: bool = False, param_file:Optional[str] = None,
rnd_term:float = None, **kwargs)->Generator[Tuple[Schedule, Operator, float], None, None]:
        """
        Random walk generator

        Parameters
        -------------
        - init_sched: Initial Schedule object
        - n_iter: number of iterations
        - p_operators: operators probabilities
        - uniform: if True, defines a constant target distribution
        - param_file: location of file containing the parameters
        - rnd_term: random term to add to the utility function
        - kwargs: additional keyword arguments

        Returns
        -------------
        Generator with current state, operator, and acceptance probability.
        """

        #Initialize schedule and operators
        accepted = 0
        rejected = 0

        init_sched.streamline()

        if not operators:
            operators = ['Block', 'Assign', 'AddAnchor', 'Swap', 'InflateDeflate', 'MetaOperator']
        if not p_operators:
            p_operators = len(operators)*[1/len(operators)]

        current_state = init_sched
        current_discret = current_state.discretization
        op_factory = OperatorFactory()

        #Compute initial weights
        current_weight = target_weight(init_sched, uniform=uniform, param_file=param_file, rnd_term=rnd_term)

        #Start walk
        for n in range(n_iter):
            new_state = deepcopy(current_state)
            op = op_factory.draw(operators, p_operators)

            try:
                new_state = op.apply_change(new_state)
                new_state.streamline()
            except:
                continue

            #Compute change probabilities
            f_proba = forward_probability(current_state, new_state, op)
            b_proba = backward_probability(current_state, new_state, op)

            #Compute new weights and log-ratio (acceptance proba)
            new_weight = target_weight(new_state, uniform=uniform, param_file=param_file, rnd_term=rnd_term)
            ratio = min(np.log(abs(new_weight)) - np.log(abs(current_weight)) + np.log(b_proba) - np.log(f_proba), 0)


            #Accept proposal
            u = np.log(np.random.rand())
            if u <= ratio:
                current_state = new_state
                current_weight = new_weight
                current_state.streamline()

                del new_state, new_weight


                yield current_state, op, ratio

def forward_probability(current_state: Schedule, proposal_state: Schedule, operator: Operator) -> float:
    """Returns forward probability to go from state i (current state) to j (proposed state) within a given iteration.

    Parameters:
    -------------------
    - current_state: current schedule
    - proposal_state: proposal schedule
    - operator: object of class Operator

    Returns:
    ------------------
    - Forward probability
    """

    if not isinstance(operator, Operator):
        raise TypeError("Can't compute forward probability - not a valid operator.")

    proba = operator.proba*operator.compute_change_proba(current_state, proposal_state)
    return proba

def backward_probability(current_state: Schedule, proposal_state: Schedule, operator: Operator) -> float:
    """Returns backward probability to go from state j (proposed state) to i (current state) within a given iteration.

    Parameters:
    -------------------
    - current_state: current schedule
    - proposal_state: proposal schedule
    - operator: object of class Operator

    Returns:
    ------------------
    - Backward probability
    """


    if not isinstance(operator, Operator):
        raise TypeError("Can't compute backward probability - not a valid operator.")


    proba = operator.proba*operator.compute_change_proba(proposal_state, current_state)

    return proba


def target_weight(sched: Schedule, uniform: bool =False, param_file: Optional[str] = None, rnd_term:Optional[float] = None) -> float:
    """
    Computes target weight for MH --> utility of schedule from initial model estimated with simple random sampling

    Parameters
    -----------------
    -sched: Schedule object
    -uniform: if True, returns constant weight
    -param_file: file where to find parameters
    -rnd_term: random term for the utility function

    """

    if uniform or (param_file is None):
        weight = 1

    else:

        parameters = joblib.load(param_file)
        parameters['business_trip:constant'] = 0

        weight = sched.compute_utility(parameters, rnd_term)

    return weight

# -----------------------------  STATISTICS ------------------------#

def between_var(sequences: List) -> float:
    """Computes variance of variable between sequences
    sequences: M x N list containing the N values of the variable of interest for the M sequences
    (N is 1/4 of the length of the sequence at the beginning of the MH process)"""

    M = len(sequences)
    N = len(sequences[0])

    mean_sequences = np.mean([np.mean(x) for x in sequences])

    var = (N / (M - 1)) * sum([(np.mean(s) - mean_sequences) ** 2 for s in sequences])

    return var


def within_var(sequences: List) -> float:
    """Computes variance of variable within sequences
    sequences: M x N list containing the N values of the variable of interest for the M sequences
    (N is 1/4 of the length of the sequence at the beginning of the MH process)"""

    M = len(sequences)
    N = len(sequences[0])

    var = (1 / (M * (N-1))) * sum([sum([(n - np.mean(m)) ** 2 for n in m]) for m in sequences])

    return var


def scale_reduction(sequences: List) -> float:
    """Computes scale reduction of given sequences"""

    N = len(sequences[0])
    B = between_var(sequences)
    W = within_var(sequences)

    return np.sqrt(((N - 1) / N) + B / (N * W))


# ---------------------------- VISUALISATION -------------------------------#



def collect_distributions(accepted_sched : List, activities: Optional[List]=None, exclude_home: bool =True, plot: bool=True, save_fig: Optional[str] = None, return_dict: bool=True):
    """Collects and plots the distributions of the resulting choice set for different metrics
    (start time, duration, activity participation).

    Parameters:
    -------------------
    - accepted sched: list of accepted schedules from the MH algorithm (final choice set)
    - activities: list of activities for the collection of metrics. If none are passed, the metrics are computed
    for all activities
    - exclude_home: if True only collects info on out of home activities
    - plot: If True, returns plots
    - save_fig: export format (png/pdf/svg) as string. if None, the figure is not saved.
    - return_dict: if True, returns dictionary with statistics

    """

    #Initialize dictionaries
    start_times = {}
    daily_duration = {}  # total duration for each activity
    split_duration = {}  # average duration per split
    frequency = {}  # number of splits during the day
    participation = {}  # activity participation

    if activities is None:
        list_act = ["home", "work","education","shopping",
            "errands_services","business_trip","leisure","escort"]
        if exclude_home:
            list_act.remove("home")

    else:
        list_act = activities

    for act in list_act:
        st = []
        d_dur = []
        sp_dur = []
        frq = []
        part = []

        for sched in accepted_sched:
            act_labels = [a.label for a in sched.list_act]

            if act not in act_labels:
                #No participation in activity
                part.append(0)
                continue

            part.append(1)
            act_freq = len([k for k, g in itertools.groupby(act_labels) if k == act])

            frq.append(act_freq)

            st.extend([a.start_time for a in sched.list_act if a.label == act])
            sp_dur.extend([a.duration for a in sched.list_act if a.label == act])
            d_dur.append(sum([a.duration for a in sched.list_act if a.label == act]))

        if sum(part) > 0:  # activity was scheduled in at least one schedule
            start_times[act] = st
            daily_duration[act] = d_dur
            split_duration[act] = sp_dur
            frequency[act] = frq
            participation[act] = part

    if plot:
        #Create dataframes for every statistic
        st_df = pd.DataFrame.from_dict(start_times, orient="index").transpose().melt(var_name="activity", value_name="start_time")
        ddur_df = pd.DataFrame.from_dict(daily_duration, orient="index").transpose().melt(var_name="activity", value_name="duration")
        sdur_df = pd.DataFrame.from_dict(split_duration, orient="index").transpose().melt(var_name="activity", value_name="duration_split")
        fr_df = pd.DataFrame.from_dict(frequency, orient="index").transpose().melt(var_name="activity", value_name="frequency")

        part_df = pd.DataFrame.from_dict(participation, orient="index").transpose().melt(var_name="activity", value_name="participation")
        part_df = part_df.groupby("activity")["participation"].sum().reset_index()
        part_df["percentage"] = part_df["participation"].apply(lambda x: 100 * x / len(list(participation.values())[0]))

        sns.set_style("ticks")
        colors = activity_colors()

        fig, axs = plt.subplots(5, 1, figsize=[10, 20])

        sns.histplot(data=st_df,x="start_time",hue="activity",kde = True,stat="frequency",palette=colors,ax=axs[0])
        sns.histplot(data=ddur_df,x="duration",hue="activity",kde = True,palette=colors,stat="frequency",ax=axs[1])
        sns.histplot(data=sdur_df,x="duration_split",hue="activity",kde = True,palette=colors,stat="frequency",ax=axs[2])
        sns.histplot(data=fr_df,x="frequency",hue="activity",kde = True,palette=colors,stat="frequency",ax=axs[3])
        sns.barplot(data=part_df, x="percentage", y="activity", palette=colors, ax=axs[4])

        axs[0].set_title(f"Distribution of start times", fontweight="bold")
        axs[1].set_title(f"Distribution of total daily duration, per activity", fontweight="bold")
        axs[2].set_title(f"Distribution of duration, per activity split", fontweight="bold")
        axs[3].set_title(f"Distribution of activity frequency, per schedule", fontweight="bold")
        axs[4].set_title(f"Activity participation", fontweight="bold")
        axs[0].set_xlim([0, 24])
        plt.tight_layout()

        if save_fig:
            filename = f'statistics.{save_fig}'
            plt.savefig(filename,format = save_fig)
            print(f'Figure saved at {filename}.')

    if return_dict:
        return [start_times, daily_duration, split_duration, frequency, participation]

    return None
