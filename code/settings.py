
DESIRED_TIMES =  {'business_trip': {'desired_duration': 0.25, 'desired_start_time': 10.0},
 'education': {'desired_duration': 7.083333333333335,'desired_start_time': 8.333333333333334},
 'errands_services': {'desired_duration': 0.25,'desired_start_time': 14.083333333333336},
 'escort': {'desired_duration': 0.25, 'desired_start_time': 8.25},
 'leisure': {'desired_duration': 1.0, 'desired_start_time': 12.0},
 'shopping': {'desired_duration': 0.11666666666666714,'desired_start_time': 10.083333333333334},
 'work': {'desired_duration': 8.75, 'desired_start_time': 7.333333333333332}}

DEFAULT_ACTIVITIES = ["home", "work","education","shopping","errands_services","business_trip","leisure","escort"]
DEFAULT_OPERATORS = ['Block', 'Assign', 'AddAnchor', 'Swap', 'InflateDeflate', 'MetaOperator']
DEFAULT_P_OPERATORS = len(DEFAULT_OPERATORS)*[1/len(DEFAULT_OPERATORS)]
DEFAULT_MODES = ["driving", "pt", "cycling"]

DEFAULT_MH_PARAMS = {"n_iter":200000,
"n_burn": 50000,
 "n_skip": 10,
 "uniform": False,
}

DEFAULT_VARIABLES = ['start_time', 'duration', 'participation']
