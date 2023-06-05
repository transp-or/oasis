import pandas as pd
import numpy as np
from data_utils import cplex_to_df, create_dicts, plot_schedule, plot_mode

import pickle
import json

from docplex.mp.model import Model
from docplex.mp.conflict_refiner import ConflictRefiner

def optimize_schedule(df = None, travel_times = None, distances = None, n_iter = 1, plot_every = 10, mtmc = True, parameters = None, preferences = None, var = 1, deterministic = False, plot_mode = False, verbose = False):
    '''
    Optimize schedule using CPLEX solver, given timing preferences and travel time matrix.
    Can produce a graphical output if specified (by argument plot_every)
    travel_times = used to be 2d nest Orig X Dest, changed to 3d nest Mode X Orig X Dest --> need to add mode in dictionary
    '''

    PERIOD = 24
    MODES = ["driving", "bicycling", "transit", "walking"]

    FLEXIBILITY_LOOKUP = {'education': 'NF',
     'work': 'NF',
     'errands_services': 'F',
     'escort': 'NF',
     'leisure': 'F',
     'shopping': 'F',
     'home': 'F',
     'dawn': 'F',
     'dusk': 'F',
     'business_trip': 'NF'
     }


    activity_specific = False

#---------------------------------- INITIALIZE PARAMETERS ---------------------------------------#

    if parameters is None:
        penalty_early = {'F': 0,'M': -0.61,'NF': -2.4}#penalties for early arrival
        penalty_late = {'F': 0,'M': -2.4,'NF': -9.6}  #penalties for late arrival
        penalty_short = {'F': -0.61,'M': -2.4,'NF': -9.6}#penalties for short duration
        penalty_long = {'F': -0.61,'M': -2.4,'NF': -9.6}#penalties for long duration

        constants = {"home": 0,
        "dawn": 0,
        "dusk": 0,
        "work" : 0,
        "education": 0,
        "shopping": 0,
        "errands_services": 0,
        "business_trip": 0,
        "leisure": 0,
        "escort": 0}

        penalty_travel= -1 #penalty for travel time

        error_w = np.random.normal(scale = var, size = 2)
        error_x = np.random.normal(scale = var, size = 4) #discretization start time: 4h time blocks
        error_d = np.random.normal(scale = var, size = 6)
        error_z = np.random.normal(scale = var, size = 2)

        if deterministic:
            EV_error = 0
        else:
            EV_error = np.random.gumbel()


    else:

        #TO DO: create function to read params from dict/JSON file
        penalty_early = parameters['penalty_early']
        penalty_late =  parameters['penalty_late']
        penalty_short =  parameters['penalty_short']
        penalty_long =  parameters['penalty_long']

        constants = parameters['constants']

        if 'work' in penalty_early.keys():
            activity_specific = True

        penalty_travel = parameters['penalty_travel'] #penalty for travel time

        error_w = parameters['errors']['participation'][n_iter]
        error_x = parameters['errors']['start_time'][n_iter] #discretization start time: 4h time blocks
        error_d = parameters['errors']['duration'][n_iter]
        error_z = parameters['errors']['sequence'][n_iter]

        if deterministic:
            EV_error = 0
        else:
            EV_error = parameters['errors']['EV'][n_iter]

        preferences = parameters['preferences'][n_iter]


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
    "escort": 25.2}

    #penalties travel cost
    #p_t_cost = {mode : p_t/vot[mode] for mode in modes}
    p_t_cost = {mode: 0 for mode in MODES}

    #penalty activity cost
    #p_act_cost = - 2
    p_act_cost = 0

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


    #dictionaries containing data
    keys, location, feasible_start, feasible_end, des_start, des_duration, group, mode, act_id, act_type = create_dicts(df, preferences)

    #define activity specific or generic utility function
    def utility_index(activity):
        if activity_specific:
            return act_type[activity]
        return FLEXIBILITY_LOOKUP[act_type[activity]]


#------------------------------------------- INITIALIZE MODEL --------------------------------------------#

    m = Model()
    m.parameters.optimalitytarget = 3 #global optimum for non-convex models

    #decision variables
    x = m.continuous_var_dict(keys, lb = 0, name = 'x') #start time
    z = m.binary_var_matrix(keys, keys, name = 'z') #activity sequence indicator
    d = m.continuous_var_dict(keys, lb = 0, name = 'd') #duration
    w = m.binary_var_dict(keys, name = 'w') #indicator of  activity choice
    tt = m.continuous_var_dict(keys, lb = 0, name = 'tt') #travel time
    tc = m.continuous_var_dict(keys, lb = 0, name = 'tc') #travel cost

    md_car = m.binary_var_dict(keys, name = 'md') #mode of transportation (availability)
    #z_md = m.binary_var_cube(keys, keys, modes, name = 'z_md') #dummy variable to linearize product of z and md

    error_w = m.piecewise(0, [(k,error_w[k]) for k in [0,1]], 0)
    error_z = m.piecewise(0, [(k,error_z[k]) for k in [0,1]], 0)
    error_x = m.piecewise(0, [(a, error_x[b]) for a,b in zip(np.arange(0, 24, 6), np.arange(4))], 0)
    error_d = m.piecewise(0, [(a, error_d[b]) for a,b in zip([0, 1, 3, 8, 12, 16], np.arange(6))], error_d[-1])

    #------------------------------------------------ CONSTRAINTS ---------------------------------------#


    for a in keys:
        ct_sequence = m.add_constraints(z[a,b] + z[b,a] <= 1 for b in keys if b != a)
        ct_sequence_dawn = m.add_constraint(z[a,'dawn'] == 0 )
        ct_sequence_dusk = m.add_constraint(z['dusk',a] == 0 )
        ct_sameact = m.add_constraint(z[a,a] == 0)
        ct_times_inf = m.add_constraints(x[a] + d[a] + tt[a] - x[b] >= (z[a,b]-1)*PERIOD for b in keys)
        ct_times_sup = m.add_constraints(x[a] + d[a] + tt[a] - x[b] <= (1-z[a,b])*PERIOD for b in keys)
        ct_traveltime = m.add_constraint(tt[a] == m.sum(z[a,b]*travel_times[mode[a]][location[a]][location[b]] for b in keys))
        ct_travelcost = m.add_constraint(tc[a] == m.sum(z[a,b]*costs_travel[mode[a]]*distances[location[a]][location[b]] for b in keys))



        if group[a] in ["home", "dawn", "dusk"]:
            ct_car_home = m.add_constraint(md_car[a] == 1)

        if mode[a] == "driving":
            ct_car_avail = m.add_constraint(w[a] <= md_car[a])

        ct_car_consist_neg = m.add_constraints(md_car[a] >=  md_car[b] + z[a,b] - 1 for b in keys)
        ct_car_consist_pos = m.add_constraints(md_car[b] >=  md_car[a] + z[a,b] - 1 for b in keys)

        ct_nullduration = m.add_constraint(w[a] <= d[a])
        ct_noactivity = m.add_constraint(d[a] <= w[a]*PERIOD)
        ct_tw_start = m.add_constraint(x[a] >= feasible_start[a])
        ct_tw_end = m.add_constraint(x[a] + d[a] <= feasible_end[a])

        ct_duplicates = m.add_constraint(m.sum(w[b] for b in keys if group[b] == group[a])<=1)

        if a != 'dawn':
            ct_predecessor = m.add_constraint(m.sum(z[b,a] for b in keys if b !=a) == w[a])
        if a != 'dusk':
            ct_successor = m.add_constraint(m.sum(z[a,b] for b in keys if b !=a) == w[a] )

    ct_period = m.add_constraint(m.sum(d[a] + tt[a] for a in keys)==PERIOD)
    ct_startdawn = m.add_constraint(x['dawn'] == 0)
    ct_enddusk = m.add_constraint(x['dusk']+ d['dusk'] == PERIOD)

    #----------------------------------- UTILITY FUNCTION ---------------------------------------------#
    m.maximize(

    m.sum(w[a] * (
    #penalties start time
    (penalty_early[utility_index(a)]) * m.max(des_start[act_type[a]]-x[a], 0)
    +(penalty_late[utility_index(a)]) * m.max(x[a]-des_start[act_type[a]], 0)

    #penalties duration
    +(penalty_short[utility_index(a)]) * m.max(des_duration[act_type[a]]-d[a], 0)
    +(penalty_long[utility_index(a)]) * m.max(d[a] - des_duration[act_type[a]], 0)

    #penalties travel (time and cost)
    +(penalty_travel) * tt[a]
    +(p_t_cost[mode[a]]) * tc[a]

    #activity cost
    + (p_act_cost) * costs_activity[act_id[a]])

    #error terms
    + error_w(w[a])
    + error_x(x[a])
    + error_d(d[a])
    + m.sum(error_z(z[a,b]) for b in keys) + constants[act_type[a]] for a in keys) + EV_error

    )


    solution = m.solve()

#--------------------------------------------- OUTPUT ---------------------------------------------------#
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
