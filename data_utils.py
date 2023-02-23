import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#from palettable.wesanderson import *

from geopy.distance import distance
import seaborn as sns
import matplotlib
import pickle
import os
import re

import googlemaps
import json




def cplex_to_df(w, x, d, tt, car_avail, mode, keys, act_id, location, minutes = False):
    '''
    Stores a CPLEX solution into a Pandas dataframe.
    '''
    solution_df = pd.DataFrame(columns=['act_id','label', 'start_time', 'end_time', 'duration', 'location', 'mode', 'travel_time'])

    for idx, name in enumerate(keys):
        if w[name].solution_value == 1:
            solution_df.loc[idx, 'label'] = name
            solution_df.loc[idx, 'start_time'] = (x[name].solution_value)
            solution_df.loc[idx, 'duration'] = (d[name].solution_value)
            solution_df.loc[idx, 'act_id'] = act_id[name]
            #solution_df.loc[idx, 'location'] = location[name]
            solution_df.loc[idx, 'mode'] = mode[name]
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

def plot_schedule(df, colors = 'cb'):
    '''
    Plots given schedule.
    df = Pandas dataframe containing schedule. The dataframe must contain the columns 'start_time', 'end_time', 'act_id' and 'label'
    '''

    if colors == 'cb':
        cmap = matplotlib.colors.ListedColormap(sns.color_palette("colorblind").as_hex())
    else:
        l = (Moonrise5_6.mpl_colors+GrandBudapest3_6.mpl_colors)
        cmap = matplotlib.colors.ListedColormap(l, name = 'WesAnderson', N = 10)
    norm = matplotlib.colors.Normalize(vmin = 1, vmax = 11)

    fig = plt.figure(figsize=[20, 3])
    y1 = [0, 0]
    y2 = [1,1]
    plt.fill_between([0, 24], y1, y2, color = 'silver')

    for idx, row in df.iterrows():
        x = [row['start_time'], row['end_time']]
        plt.fill_between(x, y1, y2, color = cmap(norm(row['act_id'])))
        txt_x = np.mean(x)
        txt_y = 1.2
            #plt.text(txt_x, txt_y, '{}, l={}'.format(row['label'], row['loc_id']), horizontalalignment='center',verticalalignment='center', fontsize = 10)
        if 'home' not in row['label']:
            plt.text(txt_x, txt_y, '{}'.format(row['label']), horizontalalignment='center',verticalalignment='center', fontsize = 12)#, fontweight = 'bold')


    plt.xticks(np.arange(0,25))
    plt.yticks([])
    plt.xlim([0,24])
    plt.ylim([-1,2])
    plt.xlabel('Time [h]')
    #plt.show()

    return fig
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
    df["mode_id"] = df["mode"].apply(lambda x: mode_dict[x])

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

def create_dicts(df, preferences = None, minutes = False):
    '''
    Creates dictionaries out of dataframe containing the schedule, for input in the CPLEX solver.
    '''
    #changing from hours to minutes
    if minutes:
        df['feasible_start'] = (df.feasible_start.round())*60
        df['feasible_end'] = (df.feasible_end.round())*60
        df['start_time'] = (df.start_time.round())*60
        df['duration'] = (df.start_time.round())*60

    location = df.set_index('label')['location'].to_dict()
    #location = df.set_index('label')['loc_id'].to_dict()
    feas_start = df.set_index('label')['feasible_start'].to_dict()
    feas_end = df.set_index('label')['feasible_end'].to_dict()

    flex_early = df.set_index('label')['flex_early'].to_dict()
    flex_late = df.set_index('label')['flex_late'].to_dict()
    flex_short = df.set_index('label')['flex_short'].to_dict()
    flex_long = df.set_index('label')['flex_long'].to_dict()

    if preferences is None:
        des_start = df.set_index('label')['start_time'].to_dict()
        des_duration = df.set_index('label')['duration'].to_dict()
    else:
        st, dur = preferences[0], preferences[1]
        list_labels = df.label.values.tolist()

        des_start = {}
        des_duration = {}

        for l in list_labels:
            ids = df[(df.label == l)].act_id.values
            des_start[l] = st[ids[0]]
            des_duration[l] = dur[ids[0]]

    if 'group' in df.columns:
        group = df.set_index('label')['group'].to_dict()
    else:
        group = None

    if 'mode' in df.columns:
        mode = df.set_index('label')['mode'].to_dict()
    else:
        mode = None


    act_id = df.set_index('label')['act_id'].to_dict()

    keys= df.label.values.tolist()

    return keys, location, feas_start, feas_end, des_start, des_duration, flex_early, flex_late, flex_short, flex_long, group, mode, act_id

def read_schedule(hh, sched_folder = "MTMC_schedules/", tt_folder = "MTMC_ttmatrices/", mode = "driving", compute = False, verbose = False):
    fname_csv = str(hh)
    fname_pickle = str(mode)
    sched_path = os.path.join(sched_folder, f'{hh}.csv')

    if mode == "all":
        tt_path = os.path.join(tt_folder, fname, "_*.pickle")
    else:
        tt_path = os.path.join(tt_folder, f'{hh}_{mode}.pickle')

    #sched_name = re.findall(sched_path, str(sched_path))[0]
    #tt_name =re.findall(tt_path, str(tt_path))[0]

    try:
        sched = pd.read_csv(sched_path)
        sched['location'] = sched.location.apply(lambda x: tuple(map(float,x[1:-1].split(',')))) #need to convert locations back to tuple after import
    #they get converted to str
    except:
        if verbose:
            print(f"Skipped household {hh} (no schedule)")
        sched = None
        travel_times = None

    try:
        travel_times = pickle.load(open(tt_path, "rb"))
    except:
        if compute:
            compute_tmat(sched, hh, modes = mode)
            travel_times = pickle.load(open(tt_path, "rb"))

        else:
            if verbose:
                print(f"Skipped household {hh} (no travel time matrix)")
            sched = None
            travel_times = None

    return sched, travel_times

def get_param_variables(param_dict):

    p_st_e_f, p_st_l_f, p_dur_s_f, p_dur_l_f, p_st_e_m, p_st_l_m, p_dur_s_m, p_dur_l_m, p_st_e_r, p_st_l_r, p_dur_s_r, p_dur_l_r = param_dict['p_st_e_f'], param_dict['p_st_l_f'], param_dict['p_dur_s_f'], param_dict['p_dur_l_f'], param_dict['p_st_e_m'], param_dict['p_st_l_m'], param_dict['p_dur_s_m'], param_dict['p_dur_l_m'], param_dict['p_st_e_r'], param_dict['p_st_l_r'], param_dict['p_dur_s_r'], param_dict['p_dur_l_r']
    p_t, error_w, error_z, error_x, error_d = param_dict['p_t'], param_dict['error_w'], param_dict['error_z'], param_dict['error_x'], param_dict['error_d']

    d_st_h, d_st_w, d_st_edu, d_st_s, d_st_er, d_st_b, d_st_l, d_st_es = param_dict['d_st_h'], param_dict['d_st_w'], param_dict['d_st_edu'], param_dict['d_st_s'], param_dict['d_st_er'], param_dict['d_st_b'], param_dict['d_st_l'], param_dict['d_st_es']

    d_dur_h, d_dur_w, d_dur_edu, d_dur_s, d_dur_er, d_dur_b, d_dur_l, d_dur_es = param_dict['d_dur_h'], param_dict['d_dur_w'], param_dict['d_dur_edu'], param_dict['d_dur_s'], param_dict['d_dur_er'], param_dict['d_dur_b'], param_dict['d_dur_l'], param_dict['d_dur_es']

    #print(p_st_e_f, p_st_l_f, p_dur_s_f, p_dur_l_f, p_st_e_m, p_st_l_m, p_dur_s_m, p_dur_l_m, p_st_e_r, p_st_l_r, p_dur_s_r, p_dur_l_r, p_t, error_w, error_z, error_x, error_d, d_st_h, d_st_w, d_st_edu, d_st_s, d_st_er, d_st_b, d_st_l, d_st_es, d_dur_h, d_dur_w, d_dur_edu, d_dur_s, d_dur_er, d_dur_b, d_dur_l, d_dur_es)
    return p_st_e_f, p_st_l_f, p_dur_s_f, p_dur_l_f, p_st_e_m, p_st_l_m, p_dur_s_m, p_dur_l_m, p_st_e_r, p_st_l_r, p_dur_s_r, p_dur_l_r, p_t, error_w, error_z, error_x, error_d, d_st_h, d_st_w, d_st_edu, d_st_s, d_st_er, d_st_b, d_st_l, d_st_es, d_dur_h, d_dur_w, d_dur_edu, d_dur_s, d_dur_er, d_dur_b, d_dur_l, d_dur_es

def compute_tmat(df_hh, h, modes = 'all'):
    '''
    Computes travel time matrices for specified modes, using the Google Maps API.
    df_hh = dataframe containing the trip diary of the household
    h = id of the household
    '''
    gmaps = googlemaps.Client(key="AIzaSyB9IUU98onWll8R4CRd1bkdNOiMydFfjfg")

    if modes == "all":
        mode = ['walking', 'driving', 'bicycling', 'transit']
    else:
        mode = modes

    for m in mode:
        filename = f'{h}_{m}.pickle'
        try:
            dico = pickle.load(open(f'MTMC_ttmatrices/{filename}', 'rb'))
            print(f'Using cached result for {filename}')
        except:
            print(f'New google request for {filename}')
            try:
                loc = list(df_hh.loc_id.unique())
                loc_dict_rev = {i:j for i,j in zip(df_hh.loc_id, df_hh.location)}
                dico = {}
                for l in loc:
                    o = loc_dict_rev[l]
                    origin = "{},{}".format(o[0], o[1])
                    dic = {}
                    for i in loc:
                        if i == l:
                            dic[i] = 0
                        else:
                            d = loc_dict_rev[i]
                            destination= "{},{}".format(d[0], d[1])
                            dic[i] = gmaps.directions(origin, destination, mode=m)[0]['legs'][0].get('duration', {}).get('value',-1)/3600
                    dico[l] = dic
            #except googlemaps.exceptions as gmapserror:
            except:
                print(f"Unexpected error for HH: {h} for mode: {m} - Change this to better error handling if needed")
        with open(f'MTMC_ttmatrices/{filename}', 'wb') as wf:
            pickle.dump(dico, wf)


def check_valid(schedule):
    '''
    Returns True if the provided schedule has a valid format
    '''

    labels = ['home', 'work', 'education', 'shopping', 'errands_services', 'business_trip', 'leisure', 'escort', 'travel']

    condition1 = schedule is not None

    if condition1:
        condition2 = (schedule['act_id'] <= 11).all()
        condition3 = (schedule['act_label'].isin(labels)).all()
    else:
        condition2 = False
        condition3 = False

    if condition2 and condition3:
        return True
    else:

        return False


def compare_schedules(sched1, sched2):
    '''Check if 2 schedules are equal. i.e. have same sequence of activities (labels) + timings
    sched1 and sched2 should have columns "label", "start time" and "duration"'''

    cond1 = sched1['label'].values.tolist() ==sched2['label'].values.tolist()
    cond2 = sched1['start_time'].values.tolist() ==sched2['start_time'].values.tolist()
    cond3 = sched1['duration'].values.tolist() == sched2['duration'].values.tolist()

    if cond1 and cond2 and cond3:
        return True
    else:
        return False


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
