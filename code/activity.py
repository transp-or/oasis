import numpy as np
import pandas as pd

from itertools import groupby

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
from settings import DESIRED_TIMES



#desired_times = desired_times_stud

# def load_desired_times(pickle_file):
#     '''Loads desired times from input pikle file'''
#     desired_times = pickle.load(open(pickle_file, 'wb'))
#     return desired_times


class Activity:
    """
    This class creates an "activity" unit to be used in the estimation process. A list of activities constitutes a schedule.
    Activity objects can be easily created with the ActivityFactory class.

    Attributes:
    -------------------
    - label: unique label of the activity
    - start_time: discrete start time (int)
    - duration: discrete duration (int)
    - end_time: discrete end time (int)
    - mode : mode of transportation of the associated travel
    - next_act: next activity in the schedule object
    - prev_act: previous activity in the schedule object
    - early: deviation from preferred start time (early)
    - late: deviation from preferred start time (late)
    - short: deviation from preferred duration (short)
    - long: deviation from preferred duration (long)
    - boundary_inf: lower bound for feasible start time
    - boundary_sup: upper bound for feasible end travel_time

    Methods:
    ------------------
    - Getters and setters for protected attributes
    - compute_utility: computed utility function for activity

    """
    def __init__(self,label: str,start_time: int, duration: int, end_time: Optional[int] =None, mode: Optional[str]= None,
    location: Union[Tuple, str, None] = None, prev_act: Optional[str]= None,
    next_act: Optional[str]= None, early: Optional[float] = None, late: Optional[float] = None, short: Optional[float] = None, long: Optional[float] = None):
        self._label = label
        self._start_time = start_time
        self._duration = duration
        self._end_time = end_time
        self._mode = mode
        self._location = location
        self._prev_act = prev_act
        self._next_act = next_act
        self.early = early
        self.late = late
        self.short = short
        self.long = long

        self._boundary_inf = (self.label == "home") and self.start_time == 0
        self._boundary_sup = (self.label == "home") and self.start_time == (24 - self.duration)

        if self._boundary_inf :
            self._prev_act = None
        else:
            self._prev_act = prev_act

        if self._boundary_sup:
            self._next_act = None
        else:
            self._next_act = next_act

        if self._end_time is None:
            self._end_time = self._start_time + self._duration

        if self._label not in ['home', 'dawn', 'dusk']:
            st_diff = self._start_time - DESIRED_TIMES[self._label]['desired_start_time']
            d_diff = self._duration - DESIRED_TIMES[self._label]['desired_duration']
        else:
            st_diff = 0
            d_diff = 0

        if early is None:
            self.early = ((st_diff>=-12) & (st_diff<=0))*(-st_diff) + ((st_diff>=12) & (st_diff<=24))*(24-st_diff)

        if late is None:
            self.late = ((st_diff>=0)&(st_diff<12))*(st_diff) + ((st_diff>=-24) & (st_diff<-12))*(24+st_diff)

        if long is None:
            self.long = (d_diff>=0)*d_diff

        if short is None:
            self.short = (d_diff<=0)*(-d_diff)



    def __eq__(self, othr):
        return (isinstance(othr, type(self))
        and (self.label, self.start_time, self.duration) ==
                        (othr.label, othr.start_time, othr.duration) )

    def __hash__(self):
            return hash((self.label, self.start_time, self.duration))

    def __str__(self):
        return f"{self.label}: start time {self.start_time}, duration {self.duration} h, location {self.location}"


    #Getters and setters for protected attributes
    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, lab):
        self._label = lab

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, dur):
        self._duration = dur

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, st):
        self._start_time = st

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, et):
        self._end_time = et

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, m):
        self._mode = m

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self,loc):
        self._location = loc

    @property
    def prev_act(self):
        return self._prev_act

    @prev_act.setter
    def prev_act(self, act):
        if self._boundary_inf:
            self._prev_act = None
        else:
            self._prev_act = act

    @property
    def next_act(self):
        return self._next_act
    @next_act.setter
    def next_act(self, act):
        if self._boundary_sup:
            self._next_act = None
        else:
            self._next_act = act

    def compute_utility(self, params: Dict, reference_act: List = ['home', 'dawn', 'dusk']):

        """Computes the activity-specific utility function.

        Parameters
        ----------
        params: Dictionary of parameters to be used in the utility function
        reference_act: List of label(s) of the reference activity (default is home)

        Return
        ----------
        V: value of the utility function

        """


        at = self.label

        flexibility_lookup = {'education': 'NF','work': 'NF','errands_services': 'F','escort': 'NF','leisure': 'F','shopping': 'F', 'home': 'F', 'business_trip': 'NF'}

        early = self.early
        late = self.late
        short = self.short
        long = self.long

        V = 0
        if at not in reference_act:

            if at in ["business_trip", "escort", "errands_services"]:
                params[f"{at}:constant"] = 0


            fd = flexibility_lookup[at]
            V += params[f"{at}:constant"]  + params[f'{fd}:early']* early + params[f'{fd}:late']* late + params[f'{fd}:short']* short \
            + params[f'{fd}:long'] * long

        return V


class Travel(Activity):
    """
    This class defines the specific Travel activity.
    """
    def __init__(self):
        super().__init__("Travel", location = None)

    def __str__(self):
        return f"Travel from {self.prev_act.label} (location: {self.prev_act.location}) to {self.next_act.label} (location: {self.next_act.location}), by {self.mode}.\n Travel time: {self.duration}"


class ActivityFactory:
    """
    This class creates and Activity object.

    Methods:
    -------------------
    - create: creates object from Activity class
    """
    def __init__(self):
        pass

    def create(self, label: Optional[str] = None, random: bool = False, list_act: List = ["home","work","education","shopping","errands_services",
    "leisure","escort", "business_trip"],**kwargs) -> Activity:
        """
        Creates an object from the Activity class.

        Parameters:
        ---------------
        -label: label of the activity to create
        -random: if True, creates a random activity
        -list_act: list of possible activity labels
        -**kwargs: other keywords arguments that will be passed to the Activity constructor.
        """


        if (label is None) and (random == False):
            raise ValueError("No activity label was passed.")
        if random:
            label = np.random.choice(list_act)

        new_act = Activity(label, **kwargs)
        return new_act

class Schedule:
    """
    This class stores schedules of activity objects.


    Attributes:
    -----------------
    -list_act: list of Activity objects representing the activities in the schedule
    -total_dur: total schedule duration, default is 24h
    -start: start time of the schedule (default: Oh - midnight)
    -end: end time of the schedule (default: 24h )
    -discretization: schedule discretization in hours, default: 1/60 h
    -feasibility: boolean that indicates if the schedule is feasible
    -travel_time_mat: matrix of travel times
    -anchor_nodes: time points in the schedule where operators changes will be applied. Default: at every hour for empty schedules
    -all_starts: list of the start times of every activity in the schedule
    -all_locations: list of the locations of every activity in the schedule
    -all_labels: list of the labels of every activity in the schedule
    -list_modes: list of the possible modes of transportation


    Methods:
    -----------------

    """
    def __init__(self, list_act: Optional[List]=None, total_dur:int=24, start:int=0, end:int=24, discretization:float=1/60, travel_time_mat:Dict = None)-> None:
        self._list_act = list_act
        self._total_dur = total_dur
        self._start = start
        self._end = end
        self._discretization = discretization
        self._feasibility = False
        self._travel_time_mat = travel_time_mat

        if list_act and len(list_act)>2: #set the anchor nodes to the start times of the existing activities, excluding the first and last one (boundary at home which cannot be changed)
            self._anchor_nodes = [x.start_time for x in list_act[1:-1]]
        else: #if no activity has been passed the anchors are initialized at every hour
            self._anchor_nodes = np.arange(1, 24)

        self.all_starts = [act.start_time for act in self.list_act]
        self.all_ends = [act.end_time for act in self.list_act]
        self.all_locations = [act.location for act in self.list_act]
        self.all_labels = [act.label for act in self.list_act]

        self.list_modes = ["driving","pt","cycling"]



    def __eq__(self, othr):
        return (isinstance(othr, type(self))
                and (self.list_act, self.all_starts) ==
                    (othr.list_act, othr.all_starts))

    def __hash__(self):
        return hash((tuple(self.list_act), tuple(self.all_starts)))

    @property
    def list_act(self):
        return self._list_act

    @list_act.setter
    def list_act(self, l_a: List):
        self._list_act = l_a
        self.update()

    @property
    def total_dur(self):
        return self._total_dur

    @total_dur.setter
    def total_dur(self, dur: int):
        self._total_dur = dur

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self,st:Union[float,int]):
        self._start = st

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self,et:Union[float,int]):
        self._end = et

    @property
    def discretization(self):
        return self._discretization

    @discretization.setter
    def discretization(self, discret: float):
        self._discretization = discret


    @property
    def feasibility(self):
        return self._feasibility

    @feasibility.setter
    def feasibility(self, status: bool):
        self._feasibility = status

    @property
    def travel_time_mat(self):
        return self._travel_time_mat

    @travel_time_mat.setter
    def travel_time_mat(self, tt_mat: Dict):
        self._travel_time_mat = tt_mat

    @property
    def anchor_nodes(self):
        return self._anchor_nodes

    @anchor_nodes.setter
    def anchor_nodes(self,nodes: List):
        self._anchor_nodes = nodes


    def get_travel_time(self, origin: Union[Tuple,str,int], destination: Union[Tuple,str,int], mode: str)->float:
        """
        Extract OD travel time.

        Parameters
        ---------------
        origin: ID of the origin (must be a valid key in the travel time matrix)
        destination: ID of the destination (must be a valid key in the travel time matrix)
        mode: mode of transportation (must be a valid key in the travel time matrix)

        Returns
        --------------
        tt: travel time in hours

        """

        if (None in [origin, destination, mode]) or (self.travel_time_mat is None):
            #!-----change this if you want another behaviour for missing values (here it's setting the TT to 0)---!
            return 0
        try:
            tt = self.travel_time_mat[mode][origin][destination]
        except KeyError:
            #!-----change this if you want another behaviour for missing values (here it's setting the TT to 0)---!
            print('Couldnt compute travel time. Setting to 0.')
            tt = 0
        return tt

    def get_home_location(self):
        """
        Returns the location identified as home.
        """
        return self.list_act[0].location

    def update(self):
        """
        Updates variables when the activities change.
        """
        self.all_starts = [act.start_time for act in self.list_act]
        self.all_ends = [act.end_time for act in self.list_act]
        self.all_locations = [act.location for act in self.list_act]
        self.all_labels = [act.label for act in self.list_act]



    def streamline(self) -> None:
        """This function checks that the schedule is valid in terms of continuity (e.g. no gaps in time or space)"""

        sched_to_update = [x for x in self.list_act if x]

        if not sched_to_update:
            #empty list - set all at home
            self.list_act = [Activity("home", 0, 24, location = self.get_home_location(), mode = 'driving')]
            return None

        #Sort by start time
        sched_to_update.sort(key=lambda x: x.start_time)

        #Check boundary conditions (start and end at home)
        if sched_to_update[0].label not in ['home', 'dawn']:
            current_start = sched_to_update[0].start_time #current start of the schedule
            sched_to_update.insert(0, Activity('home', 0, current_start, location = self.get_home_location(), mode = 'driving'))

        if sched_to_update[-1].label not in ['home', 'dusk']:
            current_end = sched_to_update[-1].end_time#current_end
            sched_to_update.append(Activity('home', current_end, 24-current_end, location = self.get_home_location(), mode = 'driving'))


        for i,a in enumerate(sched_to_update[1:], 1):
            #Check that the start time matches the end time of the previous activity + travel time
            prev_act = sched_to_update[i-1]

            try:
                tt = self.get_travel_time(prev_act.location, a.location, prev_act.mode)
            except KeyError:
                print(f"Couldn't find location. Travel Time matrix: {self.travel_time_mat[prev_act.mode]} \n Locations: {prev_act.location, a.location, prev_act.mode}")
                tt = 0

            if a.start_time != (prev_act.end_time + tt):
                a.start_time = prev_act.end_time + tt

            #Put back updated activity in schedule
            sched_to_update[i] = a

        #Define boundary times
        sched_to_update[0].start_time = self.start
        sched_to_update[-1].end_time = self.end

        #Combine consecutive duplicates of activities
        grouped_labels = groupby(sched_to_update, lambda x: x.label)
        new_list = []

        for _, group in grouped_labels:
            group = list(group)
            new_act = group[0]
            new_act.end_time = group[-1].end_time
            new_list.append(new_act)

        sched_to_update = new_list

        #Fix durations and drop invalid ones (no or negative duration)
        to_drop = []
        for i,a in enumerate(sched_to_update):
            #check durations
            #prev_act = sched_to_update[i-1]
            duration = a.end_time - a.start_time

            if duration > 0:
                a.duration = duration
            else:
                to_drop.append(i)

            #Put back updated activity in schedule
            sched_to_update[i] = a

        #Drop invalid activities
        updated_sched = [x for i,x in enumerate(sched_to_update) if i not in to_drop]


        #Set updated schedule as attribute
        self.list_act = updated_sched

        #Delete variables from namespace
        del sched_to_update, updated_sched, new_list, to_drop


    def compute_utility(self, params:Dict, rnd_term:Optional[float]= None):
        """Computes utility of full schedule, given utility parameters

        Parameters
        ---------------
        params: Dictionary of utility parameters
        rnd_term:random term to add to the utility function

        Returns
        ---------------
        utility: utility function of the schedule

        """

        if not rnd_term:
            #Draw an EV distributed error term
            rnd_term = np.random.gumbel()

        list_act = self.list_act

        #Compute all travel times
        travel_times = 0
        for i in range(len(list_act)-1):
            origin = list_act[i].location
            destination = list_act[i+1].location
            mode = list_act[i].mode
            travel_times += self.get_travel_time(origin, destination, mode)

        utility = sum([a.compute_utility(params) for a in list_act]) + params['travel_time']*travel_times + rnd_term

        return utility

    def which_activity(self, time:float)->Activity:
        """
        Returns activity that is happening at a given time
        """

        start_idx = np.searchsorted(self.all_starts, time, side='right')
        activity = self.list_act[start_idx]

        return activity

    def output(self, plot:bool = False, **kwargs) -> pd.DataFrame:
        """
        This method creates a formatted pandas DataFrame of the schedule.

        Parameters:
        ---------------
        plot: if True, plots schedule
        **kwargs: keyword arguments to be passed to the plotting function

        Returns:
        ---------------
        df: DataFrame
        """
        cols = ['activity', 'start', 'end', 'duration']
        df = pd.DataFrame(columns = cols)

        df.activity = pd.Series([x.label for x in self.list_act])
        df.start = pd.Series([x.start_time for x in self.list_act])
        df.end = pd.Series([x.end_time for x in self.list_act])
        df.duration = pd.Series([x.duration for x in self.list_act])

        if plot:
            pl = self.plot(**kwargs)
            return df, pl

        return df

    def plot(self, list_act: List =None, title: str=None, figure_size: List = [20,3],**kwargs):
        """
        Plots given schedule.

        Parameters:
        ---------------
        -list_act: Default list of activities (for activity colors)
        - title: plot title
        - figure_size: size of the matplotlib figure
        - kwargs: other keyword arguments for matplotlib's functions
        """

        colors = self.activity_colors(list_act)


        fig = plt.figure(figsize=figure_size)
        y1 = [0, 0]
        y2 = [1, 1]
        plt.fill_between([0, 24], y1, y2, color="silver")


        for item in self.list_act:
            x = [item.start_time, item.end_time]
            plt.fill_between(x, y1, y2, color=colors[item.label])


        plt.xticks(np.arange(0, 25))
        plt.yticks([])
        plt.xlim([0, 24])
        plt.ylim([-1, 2])
        plt.xlabel("Time [h]")

        if title:
            plt.title(title, fontsize=12, fontweight="bold")

        leg_labels = [x.label if x not in ["dawn", "dusk"] else "home" for x in self.list_act]
        legend_handles = [mpatches.Patch(color=colors[act], label=act) for act in set(leg_labels)]
        plt.legend(handles=legend_handles)

        return fig


    def activity_colors(self, list_act: List =None, palette: str ="colorblind")-> List:
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
