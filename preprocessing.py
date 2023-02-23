import pandas as pd
import numpy as np

from cryptography.fernet import Fernet
from sqlalchemy import create_engine

def preprocess_data(source = 'database', key_loc = None, pwd_string = None, user = None, csv_loc = None, travel = False):

    '''
    source = can b
    key_loc = absolute or relative location of Fernet key to access the database
    pwd_string = Fernet password to access the database, see: https://cryptography.io/en/latest/fernet/ to generate key and pwd
    user = username for the database


    Code can be modified with direct access to the database simply by replacing the Fernet part with the actual password and commenting lines 22/23.


    '''
    if source == 'database':
        key = open(key_loc,'rb').read()
        pwd = pwd_string

        engine = create_engine(f'postgresql://{user}:{Fernet(key).decrypt(pwd).decode()}@transporvm1/mtmc')
        query = ("""SELECT "HHNR", "ETNR", f51300, f52900, rdist, ldist, e_dauer, f51100time, f51400time,
        "S_X", "S_Y", "Z_X", "Z_Y", f51700 FROM steps""")

        df = pd.read_sql_query(query, engine)
    elif source == 'csv':
        if not csv_loc:
            print('Please provide a csv data file.')
            return None

        df = pd.read_csv(csv_loc)
        engine = None
    else:
        print('Please provide a valid data file (CSV or SQL database).')
        return None

    #convert HHNR to int
    df.HHNR = df.HHNR.astype('int64')

    #convert time stamps to hours
    stt= (df.f51100time).astype(str).str.split(':')
    i, j, k = stt.str[0], stt.str[1], stt.str[2]
    df['start_time_linear'] = i.astype(int) + (j.astype(float)/60) + (k.astype(float)/3600)

    ent= (df.f51400time).astype(str).str.split(':')
    i, j, k = ent.str[0], ent.str[1], ent.str[2]
    df['end_time_linear'] = i.astype(int) + (j.astype(float)/60) + (k.astype(float)/3600)

    #remove hh who don't finish at home
    hh_list = df.HHNR.unique()
    hh_to_del = []

    for h in hh_list:
        df_hh = df[df.HHNR == h]
        if df_hh.f52900.iloc[-1] != 11:
            hh_to_del.append(h)

    df = df[~df.HHNR.isin(hh_to_del)]
    df.reset_index(drop=True, inplace=True)

    purp_to_del = [1, 13]
    df = df[~df.f52900.isin(purp_to_del)]

    #change activity labels
    id_to_rep = {
    11: 1, #home
    7: 6, #business trips + activity
    10: 9, #both escort activities
    }
    df.f52900.replace(to_replace = id_to_rep, inplace = True)

    #remove NaN values
    df = df[df.f52900 >=0]

    #origin and destinations
    df.f52900.replace(to_replace = id_to_rep, inplace = True)

    return df, engine

def extract_sched_mtmc(df, hh = None, travel = False):
    '''
    Format schedule from MTMC trip diary.
    '''
    if hh is None:
        df_hh = df
        hh = df_hh.HHNR.values.unique()
    else:
        df_hh = df[df.HHNR == hh]

    labels_act = {1: "home",2: "work",3: "education",4: "shopping",
        5: "errands_services",6: "business_trip",8: "leisure", 9: "escort"}
    tw_start = {"home": 0,"work":5,"education":7,"shopping":7,
        "errands_services":7,"business_trip":5,"leisure":0,"escort":0}

    tw_end = {"home": 24,"work":23,"education":23,"shopping":20,
        "errands_services":23,"business_trip":24,"leisure": 24, "escort":24}

    categories = {"home": "discret","work":"mandat","education":"mandat",
        "shopping":"discret","errands_services":"mainten",
        "business_trip":"mandat","leisure": "discret", "escort":"mainten"}

    flex_early = {"mandat": "R","mainten": "M", "discret": "F"}
    flex_late = {"mandat": "R","mainten": "M", "discret": "M"}
    flex_short = {"mandat": "R","mainten": "M", "discret": "F"}
    flex_long = {"mandat": "M","mainten": "F", "discret": "F"}

    origins = list(zip(df_hh.S_Y, df_hh.S_X))
    destinations = list(zip(df_hh.Z_Y, df_hh.Z_X))
    locations = origins
    locations.append(destinations[-1])
    l = list(set(locations))
    loc_dict = {l[i]:i for i in range(len(l))}


    list_act = list(df_hh.f52900.values)
    list_act.insert(0, 1)

    start_times = list(df_hh.end_time_linear.values)
    start_times.insert(0, 0)
    end_times = list(df_hh.start_time_linear.values)
    end_times.append(24)

    schedule = pd.DataFrame()
    schedule['act_id'] =  pd.Series(list_act)
    schedule['act_label'] = schedule.act_id.replace(to_replace = labels_act)

    label = schedule.act_label.tolist()
    label[0] = 'dawn'
    label[-1] = 'dusk'
    label = list(map(lambda x: str(x[1]) + str(label[:x[0]].count(x[1]) + 1) if label.count(x[1]) > 1 else str(x[1]), enumerate(label)))

    ttimes = start_times[1:]-start_times[:-1]
    ttimes.append(0)

    schedule['label'] = label
    schedule['start_time'] = pd.Series(start_times)
    schedule['end_time'] = pd.Series(end_times)
    schedule['duration'] = schedule.end_time - schedule.start_time
    schedule['feasible_start'] = schedule.act_label.replace(to_replace = tw_start)
    schedule['feasible_end'] = schedule.act_label.replace(to_replace = tw_end)
    schedule['location'] = locations
    schedule['loc_id'] = schedule['location'].apply(lambda x: loc_dict[x])
    schedule['categories'] = schedule.act_label.replace(to_replace = categories)
    schedule['flex_early'] = schedule.categories.replace(to_replace = flex_early)
    schedule['flex_late'] = schedule.categories.replace(to_replace = flex_late)
    schedule['flex_short'] = schedule.categories.replace(to_replace = flex_short)
    schedule['flex_long'] = schedule.categories.replace(to_replace = flex_long)
    if travel:
        schedule_b = schedule
        for i in range(0, schedule_b.shape[0]-1):
            t_st = schedule_b.end_time[i]
            t_et = schedule_b.start_time[i+1]
            t_d = t_et - t_st
            schedule = schedule.append({'act_id':11,
                                'label': 'travel',
                                'start_time':t_st,
                                'end_time': t_et,
                                'duration': t_d,
                                'feasible_start':0,
                                'feasible_end':24,
                                'location': None,
                                'loc_id': None}, ignore_index= True
                              )
            schedule.sort_values(by='start_time', inplace = True)
    else:
        schedule['travel_time'] = pd.Series(ttimes)
    filename = f"{int(hh)}.csv"
    schedule.to_csv(f'MTMC_schedules/{filename}',index = False)

    return schedule

def get_geosample(id, type = 'city', user = None, key_loc = None, pwd_string = None, engine = None):
    '''
    Get geographical sample from dataset
    id: ID of the geographic entity
    type: type of geographic entity (options - city, canton, region)
    engine: access to the SQL database
    '''
    df_sampled = None

    if engine is None:
        key = open(key_loc,'rb').read()
        pwd = pwd_string
        engine = create_engine(f'postgresql://{user}:{Fernet(key).decrypt(pwd).decode()}@transporvm1/mtmc')

    query = ("""SELECT "HHNR", "W_AGGLO2000", "W_KANTON", "W_REGION" FROM hh""")
    df_c = pd.read_sql_query(query, engine)

    if type == "city":
        df_sampled = df_c[df_c.W_AGGLO2000 == id]
    elif type == "canton":
        df_sampled = df_c[df_c.W_KANTON == id]
    elif type == "region":
        df_sampled = df_c[df_c.W_REGION == id]
    else:
        print("Sampling can only be done by city, canton or region.")

    hh_sampled = list(df_sampled.HHNR.unique())

    return hh_sampled, df_sampled


def get_socioeco_sample(col, id, city = 5586, user = None, key_loc = None, pwd_string = None, engine = None):
    '''
    Get socio-economic sample from the dataset (MTMC)
    col: column in the dataset
    id: value from col
    city: city code in MTMC. 5586 = Lausanne
    engine: access to the SQL database
    '''
    df_sampled = None

    if engine is None:
        key = open(key_loc,'rb').read()
        pwd = pwd_string
        engine = create_engine(f'postgresql://{user}:{Fernet(key).decrypt(pwd).decode()}@transporvm1/mtmc')


    query = (f"""SELECT hh."HHNR", hh."W_AGGLO2000", target_indiv.\"{col}\" FROM hh
    LEFT JOIN target_indiv ON target_indiv."HHNR" = hh."HHNR" """)
    df_c = pd.read_sql_query(query, engine)

    df_sampled = df_c[(df_c[col].isin(id))]
    df_sampled = df_sampled[df_sampled.W_AGGLO2000 == city]
    hh_sampled = list(df_sampled.HHNR.unique())

    return hh_sampled, df_sampled


def remove_travel(schedule):
    '''
    Preprocessing to remove travel from MTMC preprocessed schedules (!!!) DEVELOPMENT // FOR PARAM ESTIM
    '''

    schedule = schedule[['act_id', 'act_label', 'start_time']]

    et = schedule.start_time.values[1:]
    et = np.append(et, 24)
    durations = et - schedule.start_time.values

    schedule.loc[:, 'end_time'] = et
    schedule.loc[:, 'duration'] = durations
    schedule.loc[:, 'act_id'] = schedule['act_id'].astype('int')

    schedule.rename(columns = {'act_label': 'label'})

    return schedule
