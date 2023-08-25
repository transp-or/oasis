
import pandas as pd
import json

from typing import List, Dict, Tuple, Union, Optional



def data_reader(df: pd.DataFrame, parameters: Optional[Dict] = None) -> List:
    """
    Transforms data from a dataframe schedule into a list of ActivityData objects

    Parameters
    ---------------
    - df: pandas dataframe
    - parameters: dictionary containing the parameters

    Returns
    ---------------
    List of ActivityData objects
    """
    activities = []
    for _,row in df.iterrows():
        activities.append(ActivityData(data=row, activity_parameters = parameters))
    return activities


class ActivityData():
    """
    This class stores the data related to an activity (type, location, mode, feasible times), and associated parameters (desired times, penalties).

    Attributes:
    ------------
    - label: unique label of the activity
    - group: activity type (does not need to be unique)
    - location: tuple of coordinates (must be an existing key in the travel time dictionary)
    - mode: mode of transportation (must be an existing key in the travel time dictionary)
    - feasible_start: feaible start time in hours
    - feasible_end: feasible end time in hours
    - desired_start: desired start time in hours
    - desired_duration: desired duration in hours
    - act_id: ID of the activity, should either be an integer or a dictionary mapping the activity type to an integer ID.
    - data: structure keeping the data. Can be a dictionary, a dataframe or a valid JSON string.

    Methods:
    ------------
    - read_from_pandas: instantiates class using data from pandas dataframe
    - read_from_dict: instantiates class using data from dictionary
    - add_parameters: add activity-specific parameters

    """
    def __init__(self, label: Optional[str]= None,  group: Optional[str]= None,location: Optional[Tuple]= None, mode: Optional[str]= None,
    activity_parameters: Optional[Dict]= None, feasible_start: Optional[float] = None, feasible_end: Optional[float] = None,
    desired_start: Optional[float] = None, desired_duration: Optional[float] = None, act_id: Union[int, Dict, None] = None, data: Union[Dict, pd.DataFrame, str, None] = None, *args, **kwargs):

        if data is not None:
            if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
                self.read_from_pandas(data, activity_parameters)

            elif isinstance(data,Dict):
                self.read_from_dict(data, activity_parameters)

            elif isinstance(data, str):
                data = json.loads(data)
                self.read_from_dict(data, activity_parameters)

        else:
            self.label = label
            self.group = group
            self.location = location
            self.mode_travel = mode
            self.activity_parameters = activity_parameters
            self.feasible_start = feasible_start if feasible_start else 0
            self.feasible_end = feasible_end if feasible_end else 24
            self.desired_start = desired_start if desired_start else 0
            self.desired_duration = desired_duration if desired_duration else 0

            self.type = self.group if self.group not in ['dawn', 'dusk'] else 'home'

            if act_id and isinstance(act_id, int):
                self.act_id = act_id
            elif act_id:
                self.act_id = act_id[self.type]
            else:
                act_id_default = {"home": 1,"work":2,"education":3,"shopping":4,
                "errands_services":5,"business_trip":6,"leisure":8, "escort":9}
                self.act_id = act_id_default[self.type] if self.type else None

    def read_from_pandas(self, df: pd.DataFrame, params: Optional[Dict]) -> None:
        """
        Instantiates class using data from pandas dataframe

        Parameters
        ---------------
        - df: pandas dataframe
        - params: dictionary containing the parameters
        """
        self.label = df.label
        self.group = df.group
        self.location = df.location
        self.mode_travel = df.mode_travel
        self.feasible_start = df.feasible_start if 'feasible_start' in df else 0
        self.feasible_end = df.feasible_end if 'feasible_end' in df else 24
        self.desired_start = df.desired_start if 'desired_start' in df else 0
        self.desired_duration = df.desired_duration if 'desired_duration' in df else 0
        self.type = self.group if self.group not in ['dawn', 'dusk'] else 'home'
        self.activity_parameters = params[self.type] if params else None
        self.act_id =df.act_id


    def read_from_dict(self, dic: Dict, params: Optional[Dict]) -> None:
        """
        Instantiates class using data from dictionary

        Parameters
        ---------------
        - dic: dictionary
        - params: dictionary containing the parameters
        """
        self.label = dic['label']
        self.group =  dic['group']
        self.location =  dic['location']
        self.mode_travel =  dic['mode']
        self.feasible_start =  dic['feasible_start'] if 'feasible_start' in dic.keys() else 0
        self.feasible_end = dic['feasible_end'] if 'feasible_end' in dic.keys() else 24
        self.desired_start = dic['desired_start'] if 'desired_start' in dic.keys() else 0
        self.desired_duration = dic['desired_duration'] if 'desired_duration' in dic.keys() else 0
        self.type = self.group if self.group not in ['dawn', 'dusk'] else 'home'
        self.activity_parameters = params[self.type] if params else None
        self.act_id =dic['act_id']


    def add_parameters(self, params: Dict) -> None:
        """
        Adds activity-specific parameters

        Parameters
        ---------------
        - params: dictionary containing the parameters
        """
        self.activity_parameters = params
