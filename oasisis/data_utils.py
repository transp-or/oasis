import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geopy.distance import distance
import seaborn as sns

import joblib

def cplex_to_df(w, x, d, tt, car_avail, mode_travel, keys, act_id, location, minutes = False):
    '''
    Stores a CPLEX solution into a Pandas dataframe.
    '''
    solution_df = pd.DataFrame(columns=['act_id','label', 'start_time', 'end_time', 'duration', 'location', 'mode_travel', 'travel_time'])

    for idx, name in enumerate(keys):
        if w[name].solution_value == 1:
            solution_df.loc[idx, 'label'] = name
            solution_df.loc[idx, 'start_time'] = (x[name].solution_value)
            solution_df.loc[idx, 'duration'] = (d[name].solution_value)
            solution_df.loc[idx, 'act_id'] = act_id[name]
            #solution_df.loc[idx, 'location'] = location[name]
            solution_df.loc[idx, 'mode_travel'] = mode_travel[name]
            solution_df.loc[idx, 'travel_time'] = (tt[name].solution_value)
            solution_df.end_time = solution_df.start_time + solution_df.duration
            solution_df.loc[idx, 'car_avail'] =(car_avail[name].solution_value)
            if minutes:
                solution_df.loc[idx, 'start_time'] = solution_df.loc[idx, 'start_time']/60
                solution_df.loc[idx, 'duration']  = solution_df.loc[idx, 'duration'] /60
                solution_df.loc[idx, 'travel_time'] = solution_df.loc[idx, 'travel_time'] /60

    solution_df['location'] = solution_df['label'].apply(lambda name: location[name])

    solution_df.reset_index(drop=True, inplace= True)
    return solution_df

def plot_schedule(df, axs, colors = 'colorblind'):
    '''
    Plots given schedule.
    df = Pandas dataframe containing schedule. The dataframe must contain the columns 'start_time', 'end_time', 'act_id' and 'label'
    '''

    
    y1 = [0, 0]
    y2 = [1,1]
    axs.fill_between([0, 24], y1, y2, color = 'silver')

    colors = activity_colors(palette = colors)


    for idx, row in df.iterrows():
        label = row['label'].rstrip('0123456789') if row['label'] not in ['dawn', "dusk"] else 'home'
        x = [row['start_time'], row['end_time']]
        axs.fill_between(x, y1, y2, color = colors[label])
        txt_x = np.mean(x)
        txt_y = 1.2
            #plt.text(txt_x, txt_y, '{}, l={}'.format(row['label'], row['loc_id']), horizontalalignment='center',verticalalignment='center', fontsize = 10)
        if 'home' not in row['label']:
            axs.text(txt_x, txt_y, '{}'.format(row['label'].rstrip('0123456789')), horizontalalignment='center',verticalalignment='center', fontsize = 12)#, fontweight = 'bold')


    axs.set_xticks(np.arange(0,25))
    axs.set_yticks([])
    axs.set_xlim([0,24])
    axs.set_ylim([-1,2])
    axs.set_xlabel('Time [h]')
    #plt.show()

    return None
    #if save_png:
        #plt.savefig('schedule{}_{}.png'.format(df.hid.values[0],df.person_n.values[0]))

def plot_mode(df, modes = None):
    '''
    Plots modes used in trip legs of a given schedule (df).
    modes = list of possible modes
    '''

    if modes is None:
        mode_dict = {'driving': 1, 'transit': 2, 'bicycling': 3, 'walking': 4}
    else:
        mode_dict = {m: i for i, m in enumerate(modes)}
    df["mode_id"] = df["mode_travel"].apply(lambda x: mode_dict[x])

    x = df["end_time"].values #time of trip
    y = df["mode_id"].values
    labels = df["label"].values

    fig = plt.figure(figsize = [20, 3])
    plt.plot(x, y, 'o--',color = "silver", mfc = "cadetblue", mec = 'white', linewidth=2, markersize = 10)

    for txt_x, txt_y, txt in zip(x, y, labels):
        plt.text(txt_x, txt_y+0.5, txt, horizontalalignment='center',verticalalignment='center', fontsize = 12)

    plt.xticks(np.arange(24))
    plt.yticks(np.arange(len(mode_dict)) + 1, labels = list(mode_dict.keys()), fontsize = 12)
    plt.xlabel("Trip start time [h]", fontsize = 12,)

    return fig

def create_params(biogeme_pickle, desired_times = None):

    parameters = {}

    results = res.bioResults(pickleFile = biogeme_pickle)
    estim_param = results.getBetaValues()
    param_names = list(estim_param.keys())

    parameters['penalty_early'] = {i.split(":")[0]: estim_param[i] for i in param_names if "early" in i}
    parameters['penalty_late'] = {i.split(":")[0]: estim_param[i] for i in param_names if "late" in i}
    parameters['penalty_short'] = {i.split(":")[0]: estim_param[i] for i in param_names if "short" in i}
    parameters['penalty_long'] = {i.split(":")[0]: estim_param[i] for i in param_names if "long" in i}
    parameters['constants'] = {i.split(":")[0]: estim_param[i] for i in param_names if "constant" in i}

    if desired_times:
        parameters['des_st'] = {i: desired_times[i]['desired_start_time'] for i in list(desired_times.keys())}
        parameters['des_dur'] = {i: desired_times[i]['desired_duration'] for i in list(desired_times.keys())}

    return parameters

def create_pseudo_random_params(N, parameter_file, preference_file, error_var):

    parameters = joblib.load(parameter_file) #penalties from biogeme

    error_w = [np.random.normal(scale = error_var, size = 2) for i in range(N)]
    error_x = [np.random.normal(scale = error_var, size = 4) for i in range(N)] #discretization start time: 4h time blocks
    error_d = [np.random.normal(scale = error_var, size = 6) for i in range(N)]
    error_z = [np.random.normal(scale = error_var, size = 2)for i in range(N)]
    error_EV = [np.random.gumbel() for i in range(N)]


    parameters['errors'] = {'participation': error_w, 'start_time': error_x, 'duration': error_d, 'sequence': error_z, 'EV':error_EV}
    #parameters['preferences'] =[draw_preferences(preference_file) for i in range(N)]

    if 'penalty_travel' not in parameters.keys():
        parameters['penalty_travel'] = -1

    return parameters


def compute_distances_from_tmat(tmat):
    '''Computes a distance matrix using the locations of the provided travel time matrix'''

    distance_matrix = []

    locations = list(tmat.keys())
    try:
        dist_dic = {a : {b: distance(a,b).km for b in locations} for a in locations}
        distance_matrix.append(dist_dic)
    except:
        print(f"Invalid input")
        distance_matrix.append(np.nan)

    return distance_matrix[0]


def bootstrap_mean(data, num_samples, sample_size=None):
    """
    Compute the bootstrap mean of an array.

    Parameters:
    - data: Input array of data.
    - num_samples: Number of bootstrap samples to generate.
    - sample_size: Size of each bootstrap sample. If None, it will be set to the size of the input data.

    Returns:
    - bootstrap_means: Array containing the bootstrap means.
    """
    if sample_size is None:
        sample_size = len(data)

    bootstrap_means = []

    for _ in range(num_samples):
        bootstrap_sample = np.random.choice(data, size=sample_size, replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    ci_low = np.quantile(bootstrap_means, 0.25)
    ci_high = np.quantile(bootstrap_means, 0.75)


    return np.array(bootstrap_means), [ci_low,ci_high]


def print_time_format(time_in_seconds):
    minutes, seconds = divmod(time_in_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d"%(hours,minutes,seconds)


def discretize_sched(schedule, block_size = 0.5):

    '''Returns a 24h schedule discretized in blocks of size (duration) n
    Block size is expressed in hours

    //If no schedule is passed, then a random discretized schedule is generated, otherwise, the passed
    schedule is discretized accordingly (existing activities are broken down into blocks)//

    If a list of activities (list_act) is passed, then all the activities of the list are scheduled. Otherwise,
    schedules are generated randomly from the default list of activities.'''
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

def round_nearest(x, a):
    return (x//a) * a

def activity_colors(list_act = None, palette = 'gist_earth'):
    '''Match each activity from list to a color from the input palette.
    Useful to keep consistent colors across visualizations
    '''

    if list_act is None:
        list_act = ["home", "work","education", "shopping","errands_services",
                    "business_trip","leisure", "escort"]

    colors = {a:c for a, c in zip(list_act, sns.color_palette(palette, len(list_act)).as_hex())}

    if 'home' in colors.keys():
        colors['dawn'] = colors['home']
        colors['dusk'] = colors['home']

    return colors
