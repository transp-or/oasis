import numpy as np
import pandas as pd
import seaborn as sns
import math
import os

from activity import Activity, Schedule, ActivityFactory
from typing import List, Dict, Tuple, Union, Optional



def generate_discret_sched(block_size:float=0.25, list_act:Optional[List]=None) -> Dict:
    """Returns a random 24h schedule discretized in blocks of size (duration) n

    Parameters:
    --------------
    - Block_size: expressed in hours
    - list_act: If a list of activities is passed, then all the activities of the list are scheduled. Otherwise,
    schedules are generated randomly from the default list of activities.

    Returns:
    -------------
    - Random schedule as a dictionary
    """

    total_dur = 24 / block_size
    t = 0

    if list_act is None:
        list_act = [
            "home",
            "work",
            "education",
            "shopping",
            "errands_services",
            "business_trip",
            "leisure",
            "escort",
        ]

    # boundary conditions: leave at least one block at home on each side of the schedule
    end = 24 - block_size
    sequence = list(
        np.arange(block_size, 24 - block_size, block_size)
    )  # (label,act_id, start_time, feasible start, feasible end, duration)
    schedule = {
        0: "dawn",
        end: "dusk",
    }  # dict with keys being the block, and value is the activity label

    index = 1

    remaining_dur = total_dur - 2 * block_size  # remove boundary blocks
    n_act = 0
    next_duration = 0
    time_tracker = 0

    while remaining_dur > 0 and list_act:

        next_act = np.random.choice(list_act)  # pick an activity

        if n_act < 1:  # first activity
            next_start = np.random.choice(sequence)  # pick a start time
            first_activity = next_act
        else:
            next_start = time_tracker

        next_duration = np.random.choice(np.arange(remaining_dur, step=block_size))

        # check that the duration is feasible, i.e. the blocks should be unassigned
        # if not --> fill the remaining space

        current_dur = 0  # tracks the actual duration assigned to activity
        for i in np.arange(next_start, next_start + next_duration, block_size):
            if i not in schedule.keys():
                schedule[i] = next_act
                current_dur += block_size
            else:
                break

        # update trackers
        if current_dur > 0:
            remaining_dur -= current_dur
            list_act = list_act.remove(next_act)
            time_tracker = next_start + current_dur
            n_act += 1

    # adjust time at home
    first_activity = list(schedule.keys())[1]  # first out of home
    last_activity = list(schedule.keys())[-2]  # last out of home
    for i in sequence:
        if i not in list(schedule.keys()):
            if i < last_activity:
                schedule[i] = "dawn"
            else:
                schedule[i] = "dusk"
    # sort schedule
    sorted_schedule = {k: schedule[k] for k in sorted(schedule)}

    return sorted_schedule

def discretize_sched(schedule:pd.DataFrame, block_size:float = 0.5)->Dict:
    '''Discretizes given schedule in blocks of size (duration) n

    Parameters:
    -------------
    -schedule: schedule as a pandas dataframe
    -block_size: discretization in hours

    Returns
    --------------
    Schedule as a dictionary, where keys are the chosen discretization.
    '''
    d_sched = {}
    time_slots = range(int(24/block_size))
    sched = schedule.copy()

    mask = (sched.duration >= block_size)
    sched['d_start'] = sched.start_time.apply(lambda x: int(x / block_size))

    stimes = {t : a for t,a in zip(sched[mask].d_start.values, sched[mask].act_label.values)}
    current_act = 'home'


    for n in time_slots:
        if n in stimes.keys():
            current_act = stimes[n]
        d_sched[n] = current_act
    return d_sched


def parse_schedule(schedule: Dict)-> Schedule:
    """ Transforms a dictionary schedule into a Schedule object.

    Parameters:
    -----------
    - schedule: dataframe schedule
    - tt_mat: travel time matrix

    Returns:
    -----------
    Schedule object
    """

    new_schedule = {}
    block_length = list(schedule.keys())[1]-list(schedule.keys())[0]

    for k, v in schedule.iteritems():
        new_schedule[k] = Activity(label = v, start_time = k, duration = block_length)


    return schedule

def parse_df_schedule(schedule:pd.DataFrame, tt_mat:Optional[Dict] = None)-> Schedule:

    """ Transforms a pandas DataFrame schedule into a Schedule object.

    Parameters:
    -----------
    - schedule: dataframe schedule
    - tt_mat: travel time matrix

    Returns:
    -----------
    Schedule object
    """

    list_a = []
    af = ActivityFactory()

    all_act = ["home", "work","education","shopping",
                  "errands_services","business_trip","leisure","escort"]


    for _, row in schedule.iterrows():
        cols = ['start_time', 'end_time', 'duration', 'location']
        params = {x: row[x] for x in cols}
        params['label'] = row.act_label
        params['mode'] = 'driving'
        new_act = af.create(**params)
        list_a.append(new_act)

    schedule = Schedule(list_act = list_a, travel_time_mat = tt_mat)

    return schedule


def round_nearest(x:float, a:int) -> int:
    """
    Rounds x to nearest integer a
    """
    return math.floor(x / a) * a


def lookup_discret() -> pd.DataFrame:
    """Precomputed table to convert one time discretisation to another."""

    lookup_block = {
        round(block, 2): {
            "hour": round_nearest(block, 1),
            "half": round_nearest(block, 0.5),
            "quart": round_nearest(block, 0.25),
        }
        for block in np.linspace(0, 24, num=288, endpoint=False)
    }

    pd_lookup = pd.DataFrame.from_dict(lookup_block, orient="index")
    pd_lookup = pd_lookup.reset_index().rename(columns={"index": "five"})
    return pd_lookup


def sched_from_dict(dict_sched: Dict) -> pd.DataFrame:
    """
    Creates a pandas DataFrame schedule from a dictionary
    """
    df = pd.DataFrame.from_dict(dict_sched, 'index').reset_index()
    df.rename({'index': 'start_time', 0: 'label'}, axis = 1, inplace = True)

    df['shift'] = (df['label'].shift() != df['label']).cumsum() #groupby consecutive values of activities to compute duration
    df = df.groupby(['label','shift'])['start_time'].agg(['first', 'last']) #get first and last occurence of activity
    df = df.sort_values(by = 'shift').reset_index()
    df['duration'] = df['first'].shift(-1).fillna(24) - df['first']
    df['end_time'] = df['first'] + df['duration']

    df = df[['label', 'first', 'end_time','duration']].rename({'first': 'start_time', 'label': 'act_label'},axis = 1)

    label = df.act_label.tolist()
    label[0] = 'dawn'
    label[-1] = 'dusk'
    label = list(map(lambda x: str(x[1]) + str(label[:x[0]].count(x[1]) + 1) if label.count(x[1]) > 1 else str(x[1]), enumerate(label)))
    df['label'] = label

    labels_act = {1: "home",2: "work",3: "education",4: "shopping",
        5: "errands_services",6: "business_trip",8: "leisure", 9: "escort"}

    act_ids = {v : k for k, v in labels_act.items()}

    df['act_id'] = df.act_label.map(act_ids)

    df = df[['act_id', 'act_label', 'label', 'start_time', 'end_time', 'duration']]

    return df


def schedule_to_pandas(schedule: Schedule)->pd.DataFrame:
    '''Transform a Schedule object in a Pandas dataframe.

    Parameters:
    ------------
    -schedule: Schedule object

    Returns:
    ------------
    Schedule as a pandas DataFrame.
    '''

    if not isinstance(schedule, Schedule):
        print("The input is not a valid Schedule object. Please pass a valid Schedule object")
        print(schedule)
        return None

    cols = ['act_id', 'act_label', 'label', 'start_time', 'end_time', 'duration', 'mode', 'location', 'travel_time']

    df = pd.DataFrame(columns = cols)

    labels_act = {1: "home",2: "work",3: "education",4: "shopping",
        5: "errands_services",6: "business_trip",8: "leisure", 9: "escort"}

    act_ids = {v : k for k, v in labels_act.items()}

    df.act_label = pd.Series([x.label for x in schedule.list_act])
    df.start_time = pd.Series([x.start_time for x in schedule.list_act])
    df.end_time = pd.Series([x.end_time for x in schedule.list_act])
    df.duration = pd.Series([x.duration for x in schedule.list_act])
    df.mode = pd.Series([x.mode for x in schedule.list_act])
    df.location = pd.Series([x.location for x in schedule.list_act])
    df.travel_time = pd.Series([schedule.get_travel_time(schedule.list_act[i].location, schedule.list_act[i+1].location, schedule.list_act[i].mode) for i in range(len(schedule.list_act)-1)])

    label = df.act_label.tolist()
    label[0] = 'dawn'
    label[-1] = 'dusk'
    label = list(map(lambda x: str(x[1]) + str(label[:x[0]].count(x[1]) + 1) if label.count(x[1]) > 1 else str(x[1]), enumerate(label)))
    df.label = label


    df.act_id = df.act_label.map(act_ids)

    return df

def activity_colors(list_act: List =None, palette: str ="colorblind")-> List:
    """Match each activity from list to a color from the input palette.
    Useful to keep consistent colors across visualizations

    Parameters:
    ----------------------
    - list_act: list of activity all_labels
    - palette; name of matplotlib/searborn color palette.
    """

    if list_act is None:
        list_act = [
        "home",
        "work",
        "education",
        "shopping",
        "errands_services",
        "business_trip",
        "leisure",
        "escort",
        ]

    colors = {
        a: c for a, c in zip(list_act, sns.color_palette(palette, len(list_act)).as_hex())
        }
    colors["home"] = "gainsboro"

    if "home" in colors.keys():
        colors["dawn"] = colors["home"]
        colors["dusk"] = colors["home"]

    return colors
