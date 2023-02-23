import pandas as pd
import numpy as np
from data_utils import cplex_to_df, create_dicts, plot_schedule, plot_mode

import pickle
import googlemaps
import json

from docplex.mp.model import Model
from docplex.mp.conflict_refiner import ConflictRefiner

def optimize_schedule(df = None, travel_times = None, distances = None, n_iter = 1, plot_every = 10, mtmc = True, parameters = None, var = 1, deterministic = False, plot_mode = False):
    '''
    Optimize schedule using CPLEX solver, given timing preferences and travel time matrix.
    Can produce a graphical output if specified (by argument plot_every)
    travel_times = used to be 2d nest Orig X Dest, changed to 3d nest Mode X Orig X Dest --> need to add mode in dictionary
    '''

    period = 24
    modes = ["driving", "bicycling", "transit", "walking"]
    if parameters is None:
        p_st_e = {'F': 0,'M': -0.61,'R': -2.4}#penalties for early arrival
        p_st_l = {'F': 0,'M': -2.4,'R': -9.6}  #penalties for late arrival
        p_dur_s = {'F': -0.61,'M': -2.4,'R': -9.6}#penalties for short duration
        p_dur_l = {'F': -0.61,'M': -2.4,'R': -9.6}#penalties for long duration
        p_t = -1 #penalty for travel time

        #values of time (Weis et al, 2021 - table 6)
        vot = {"driving": 13.2,
        "bicycling": 9.9,
        "transit": 12.3,
        "walking": 6
        }

        #values of leisure and work (Schmid et al 2019):
        vot_act ={"home": 25.2,
            "work" : -20.6,
            "education": -20.6,
            "shopping": 25.2,
            "errands_services": 25.2,
            "business_trip": -20.6,
            "leisure": 25.2,
            "escort": 25.2

        }

        #penalties travel cost
        p_t_cost = {mode : p_t/vot[mode] for mode in modes}

        #penalty activity cost
        p_act_cost = - 2

        #travel costs (BfS 2018)
        costs_travel = {"driving": 0.37,
        "bicycling": 0,
        "transit": 0.03,
        "walking": 0
        }

        #activity costs (derived from Schmid et al 2021, from EVE dataset with household budgets)
        costs_activity = {1: 0, #home
            2 : 0, #work #need to change for wage
            3: 0, #education,
            4: 16.8, #shopping
            5: 36.4, #errands
            6: 0, #business trip
            8: 12, #leisure,
            9: 0} #escort

        error_w = np.random.normal(scale = var, size = 2)
        error_x = np.random.normal(scale = var, size = 4) #discretization start time: 4h time blocks
        error_d = np.random.normal(scale = var, size = 6)
        error_z = np.random.normal(scale = var, size = 2)

        preferences = None
    else:
        p_st_e = {'F': parameters['p_st_e_f'],'M':parameters['p_st_e_m'],'R': parameters['p_st_e_r']}
        p_st_l = {'F': parameters['p_st_l_f'],'M': parameters['p_st_l_m'],'R':parameters['p_st_l_r']}
        p_dur_s = {'F': parameters['p_dur_s_f'],'M': parameters['p_dur_s_m'],'R': parameters['p_dur_s_r']}
        p_dur_l = {'F': parameters['p_dur_l_f'],'M': parameters['p_dur_l_m'],'R': parameters['p_dur_l_r']}

        p_t = parameters['p_t']

        error_w = parameters['error_w']
        error_x = parameters['error_x']
        error_d = parameters['error_d']
        error_z = parameters['error_z']

        pref_st = {1: parameters['d_st_h'],2: parameters['d_st_w'],3: parameters['d_st_edu'],4: parameters['d_st_s'],
        5: parameters['d_st_er'],6: parameters['d_st_b'],8: parameters['d_st_l'], 9: parameters['d_st_es']}

        pref_dur = {1: parameters['d_dur_h'],2: parameters['d_dur_w'],3: parameters['d_dur_edu'],4: parameters['d_dur_s'],
        5: parameters['d_dur_er'],6: parameters['d_dur_b'],8: parameters['d_dur_l'], 9: parameters['d_dur_es']}

        preferences = [pref_st, pref_dur]

    if deterministic:
        EV_error = 0
    else:
        EV_error = np.random.gumbel()


    #dictionaries containing data
    keys, location, feasible_start, feasible_end, des_start, des_duration, flex_early, flex_late, flex_short, flex_long, group, mode, act_id = create_dicts(df, preferences)
    #print(keys, des_start, des_duration, flex_early, flex_late, flex_short, flex_long, mode)

    m = Model()
    m.parameters.optimalitytarget = 3 #global optimum for non-convex models

    #decision variables
    x = m.continuous_var_dict(keys, lb = 0, name = 'x') #start time
    z = m.binary_var_matrix(keys, keys, name = 'z') #activity sequence indicator
    d = m.continuous_var_dict(keys, lb = 0, name = 'd') #duration
    w = m.binary_var_dict(keys, name = 'w') #indicator of  activity choice
    tt = m.continuous_var_dict(keys, lb = 0, name = 'tt') #travel time
    tc = m.continuous_var_dict(keys, lb = 0, name = 'tc') #travel cost
    #md = m.binary_var_matrix(keys, modes, name = 'md') #mode of transportation (availability)
    md_car = m.binary_var_dict(keys, name = 'md') #mode of transportation (availability)
    #z_md = m.binary_var_cube(keys, keys, modes, name = 'z_md') #dummy variable to linearize product of z and md


    #piecewise error variables
    #error_w = m.piecewise(0, [(k,error_participation[k]) for k in [0,1]], 0)
    #error_z = m.piecewise(0, [(k,error_succession[k]) for k in [0,1]], 0)
    #error_x = m.piecewise(0, [(a, error_start[b]) for a,b in zip(np.arange(0, 24, 6), np.arange(4))], 0)
    #error_d = m.piecewise(0, [(a, error_duration[b]) for a,b in zip([0, 1, 3, 8, 12, 16], np.arange(6))], error_duration[-1])
    error_w = m.piecewise(0, [(k,error_w[k]) for k in [0,1]], 0)
    error_z = m.piecewise(0, [(k,error_z[k]) for k in [0,1]], 0)
    error_x = m.piecewise(0, [(a, error_x[b]) for a,b in zip(np.arange(0, 24, 6), np.arange(4))], 0)
    error_d = m.piecewise(0, [(a, error_d[b]) for a,b in zip([0, 1, 3, 8, 12, 16], np.arange(6))], error_d[-1])

    #constraints


    for a in keys:
        ct_sequence = m.add_constraints(z[a,b] + z[b,a] <= 1 for b in keys if b != a)
        ct_sequence_dawn = m.add_constraint(z[a,'dawn'] == 0 )
        ct_sequence_dusk = m.add_constraint(z['dusk',a] == 0 )
        ct_sameact = m.add_constraint(z[a,a] == 0)
        ct_times_inf = m.add_constraints(x[a] + d[a] + tt[a] - x[b] >= (z[a,b]-1)*period for b in keys)
        ct_times_sup = m.add_constraints(x[a] + d[a] + tt[a] - x[b] <= (1-z[a,b])*period for b in keys)
        ct_traveltime = m.add_constraint(tt[a] == m.sum(z[a,b]*travel_times[mode[a]][location[a]][location[b]] for b in keys))
        ct_travelcost = m.add_constraint(tc[a] == m.sum(z[a,b]*costs_travel[mode[a]]*distances[location[a]][location[b]] for b in keys))



        if group[a] in ["home", "dawn", "dusk"]:
            ct_car_home = m.add_constraint(md_car[a] == 1)

        if mode[a] == "driving":
            ct_car_avail = m.add_constraint(w[a] <= md_car[a])

        ct_car_consist_neg = m.add_constraints(md_car[a] >=  md_car[b] + z[a,b] - 1 for b in keys)
        ct_car_consist_pos = m.add_constraints(md_car[b] >=  md_car[a] + z[a,b] - 1 for b in keys)

        ct_nullduration = m.add_constraint(w[a] <= d[a])
        ct_noactivity = m.add_constraint(d[a] <= w[a]*period)
        ct_tw_start = m.add_constraint(x[a] >= feasible_start[a])
        ct_tw_end = m.add_constraint(x[a] + d[a] <= feasible_end[a])

        #if not mtmc: #no duplicates in MTMC !
        ct_duplicates = m.add_constraint(m.sum(w[b] for b in keys if group[b] == group[a])<=1)

        if a != 'dawn':
            ct_predecessor = m.add_constraint(m.sum(z[b,a] for b in keys if b !=a) == w[a])
        if a != 'dusk':
            ct_successor = m.add_constraint(m.sum(z[a,b] for b in keys if b !=a) == w[a] )

    ct_period = m.add_constraint(m.sum(d[a] + tt[a] for a in keys)==period)
    ct_startdawn = m.add_constraint(x['dawn'] == 0)
    ct_enddusk = m.add_constraint(x['dusk']+ d['dusk'] == period)

    #objective function
    m.maximize(m.sum(w[a] * (
    #penalties start time
    (p_st_e[flex_early[a]]) * m.max(des_start[a]-x[a], 0)
    +(p_st_l[flex_late[a]]) * m.max(x[a]-des_start[a], 0)

    #penalties duration
    +(p_dur_s[flex_short[a]]) * m.max(des_duration[a]-d[a], 0)
    +(p_dur_l[flex_long[a]]) * m.max(d[a] - des_duration[a], 0)

    #penalties travel (time and cost)
    +(p_t) * tt[a]
    +(p_t_cost[mode[a]]) * tc[a]

    #activity cost
    + (p_act_cost) * costs_activity[act_id[a]])

    #error terms
    + error_w(w[a])
    + error_x(x[a])
    + error_d(d[a])
    + m.sum(error_z(z[a,b]) for b in keys) for a in keys)+ EV_error)
    #+ error_w*w[a]
    #+ error_x*x[a]
    #+ error_d*d[a]
    #+ m.sum(error_z*z[a,b] for b in keys) for a in keys)+ EV_error)

    solution = m.solve()
    figure = None
    solution_df = None
    mode_figure = None

    try:
        solution_value = solution.get_objective_value()

    except:
        solution_value = None
        print('Could not find a solution - see details')
        print(m.solve_details)
        print('------------------')
        return None

    solution_df = cplex_to_df(w, x, d, tt, md_car, mode, keys, act_id, location) #transform into pandas dataframe

    if n_iter % plot_every == 0:
        figure = plot_schedule(solution_df)
        if plot_mode:
            mode_figure = plot_mode(solution_df)

    return solution_df, figure, solution_value, mode_figure
