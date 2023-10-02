import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from activity import Activity, Schedule, ActivityFactory

from typing import List, Dict, Tuple, Optional


class Operator:
    """
        This class creates an "operator" unit to be used in the estimation process. Operators can easily be created with the OperatorFactory class.

        Attributes:
        -------------------
        - optype: label of the operator
        - proba: probability associated with the operator
        - list_operators: the list of currently available operators.

        Methods:
        ------------------
        - describe: prints information on the operator
        - apply_change: applies a change to the given schedule.
        - compute_change_proba: computes probability of change
    """

    def __init__(self, optype:str, proba:float, **kwargs):
        self._optype = optype
        self._proba = proba
        self._list_operators = ["Block", "Assign", "AddAnchor", "InflateDeflate", "Swap"]

    def describe(self):
        print(f"Type: {self._optype} \n Probability: {self._proba}")

    @property
    def proba(self):
        return self._proba

    @proba.setter
    def proba(self, proba: float):
        self._proba = proba

    @property
    def optype(self):
        return self._optype

    @optype.setter
    def optype(self, optype:str):
        self._optype = optype

    @property
    def list_operators(self):
        return self._list_operators

    @list_operators.setter
    def list_operators(self, l_op):
        self._list_operators = l_op


    def apply_change(self, schedule: Schedule) -> Schedule:
        """
        Applies change on given schedule.
        """
        return schedule

    def compute_change_proba(self):
        """Computes forward probability of change."""
        pass


class Block(Operator):
    """
    Block operator: changes the discretization of the schedule.

    Parameters:
    -------------
    proba: probability of the operator
    discret_list: list of possible discretizations in hours
    """
    def __init__(self, proba: float, **kwargs)->None:
        super().__init__("Block", proba)
        self.discret_list = [
            5/60,
            15/60,
            30/60,
            1,
        ]  # expressed in hours (5 min, 15 min, 30 min, 1h)

    def apply_change(self, schedule: Schedule)->Schedule:
        """Changes discretization of given schedule"""

        #if np.log(np.random.rand()) >= np.log(self.proba):
        #    return schedule

        n = np.random.choice(self.discret_list)
        schedule.discretization = n

        #delta = schedule.discretization  # check discretization from input schedule

        #if n != delta:  # Discretization changes
            #act_in_sched = schedule.list_act

            #for a in act_in_sched:

            #    new_time = a.start_time * delta / n
            #    a.start_time = new_time

            #schedule.discretization = n
            #schedule.list_act = act_in_sched



        return schedule


    def compute_change_proba(self, prev_sched:Schedule, new_sched:Schedule, n_discret:int=4, **kwargs)->float:
        """
        Computes change probability.

        Parameters:
        ----------
        -prev_sched, new_sched: Schedule objects
        -n_discret: number of possible discretizations
        """

        if prev_sched.discretization== new_sched.discretization:
            #probability that discretization hasn't changed
            proba = 1 / n_discret
        else:
            #probability that discretization is different
            proba = 1 - (1 / n_discret)
        return proba

class AddAnchor(Operator):
    """
    Anchor operator: changes the anchors of the schedule.

    Parameters:
    -------------
    proba: probability of the operator
    """

    def __init__(self, proba:float, **kwargs):
        super().__init__("Anchor", proba)

    def apply_change(self, schedule:Schedule)-> Schedule:
        """
        Changes anchor nodes of given schedule.
        """

        #if np.log(np.random.rand()) >= np.log(self.proba):
        #    return schedule

        current_anchors = schedule.anchor_nodes
        new_anchor = np.random.uniform(0, 24)

        if new_anchor not in current_anchors:
            current_anchors.append(new_anchor)
            schedule.anchor_nodes = current_anchors

        return schedule


    def compute_change_proba(self, prev_sched:Schedule, new_sched:Schedule, **kwargs) ->float:
        """
        Computes change probability.

        Parameters:
        ----------
        -prev_sched, new_sched: Schedule objects
        """

        previous_anchors = prev_sched.anchor_nodes
        new_anchors = new_sched.anchor_nodes

        if previous_anchors == new_anchors:
            #probability of drawing an anchor that is already present
            proba = len(previous_anchors)/24

        else:
            #probability of drawing an anchor that is not in the list
            proba = 1 - len(previous_anchors)/24

        return proba


class Assign(Operator):
    """
    Assign operator: adds an activity in the schedule

    Parameters:
    -------------
    proba: probability of the operator
    list_act: list of activities to choose from
    p_act: choice probabilities for the activities
    chosen_act_proba: probability of the chosen activity
    """
    def __init__(self, proba, list_act:Optional[List] = None, p_act:Optional[List] = None, **kwargs):
        super().__init__("Assign", proba)


        if list_act:
            self.list_act = list_act
        else:
            self.list_act = ["home","work","education","shopping","errands_services","business_trip","leisure","escort"]

        if p_act:
            self.p_act = p_act
        else:
            self.p_act = [1/len(self.list_act)]*len(self.list_act)

        self.chosen_act_proba = 1/len(self.list_act)

    def apply_change(self, schedule:Schedule)->Schedule:
        """
        Assigns an activity to an existing block or anchor. The boundary conditions
        (first and last block are at home) must be respected.
        """

        #if np.log(np.random.rand()) >= np.log(self.proba):
        #    return schedule

        act_in_sched = [x for x in schedule.list_act if x]

        rnd_node = np.random.choice(schedule.anchor_nodes) #choose a random anchor
        rnd_idx = np.random.choice(range(len(self.list_act)), p = self.p_act)  #select random activity in list
        rnd_act = self.list_act[rnd_idx]

        default_loc = schedule.get_home_location()

        new_act = Activity(rnd_act, rnd_node, duration = schedule.discretization, mode = 'driving', location = default_loc)

        #find where the new activity fits in the schedule
        all_starts = schedule.all_starts

        if rnd_node in all_starts:
            #If another activity starts at the same time, shift by a minute to maintain the order of activities -
            # if the new activity is longer than the older one  it will be deleted during the streamline operation
            #otherwise, the timings will be adjusted accordingly
            idx = all_starts.index(rnd_node)
            up_act = act_in_sched[idx]
            up_act.start_time = act_in_sched[idx].start_time + 1/60

            act_in_sched[idx] = up_act
            new_act.location = up_act.location

        act_in_sched.append(new_act)
        act_in_sched = [x for x in act_in_sched if x] #Remove NaNs

        act_in_sched.sort(key=lambda x: x.start_time) #sort activities by start time
        schedule.list_act = act_in_sched

        return schedule

    def compute_change_proba(self, prev_sched:Schedule, new_sched:Schedule, **kwargs)-> float:

        """
        Computes change probability.

        Parameters:
        ----------
        -prev_sched, new_sched: Schedule objects
        """

        #Probability of choosing an anchor, and that activity is different than the current one

        proba = (1/len(prev_sched.anchor_nodes))*(1-self.chosen_act_proba)
        return proba


class InflateDeflate(Operator):
    """
    Inflate/deflate operator: increases or decreases duration of an activity

    Parameters:
    -------------
    proba: probability of the operator
    """
    def __init__(self, proba, **kwargs):
        super().__init__("InflateDeflate", proba)

    def apply_change(self, schedule:Schedule)->Schedule:
        """Randomly increases duration of one activity, and decrease duration of another by same amount."""

        #if np.log(np.random.rand()) >= np.log(self.proba):
        #    return schedule

        act_in_sched = schedule.list_act
        nodes = schedule.anchor_nodes
        block_size = schedule.discretization

        rnd_act_inf = schedule.which_activity(np.random.choice(nodes))

        direction = np.random.randint(0, 2)  # 0 right (or clockwise) 1 left (or counterclockwise)

        #get position of activities in schedule
        idx_inf = act_in_sched.index(rnd_act_inf.label)

        if direction == 0:
            rnd_act_inf.end_time = rnd_act_inf.end_time + block_size #increase by one unit of time (from current discretization)
            act_in_sched[idx_inf] = rnd_act_inf
        else:
            rnd_act_inf.end_time = rnd_act_inf.end_time - block_size #decrease by one unit of time (from current discretization)
            act_in_sched[idx_inf] = rnd_act_inf

        #update list of activities in schedule
        schedule.list_act = act_in_sched

        return schedule

    def compute_change_proba(self, prev_sched:Schedule, new_sched:Schedule, **kwargs) -> float:
        """
        Computes change probability.

        Parameters:
        ----------
        -prev_sched, new_sched: Schedule objects
        """

        proba = 0.5 * 1/len(prev_sched.anchor_nodes)
        return proba


class Swap(Operator):
    """
    Swap operator: swaps 2 adjacent blocks (start time and duration)

    Parameters:
    -------------
    proba: probability of the operator
    """
    def __init__(self, proba, **kwargs):
        super().__init__("Swap", proba)

    def apply_change(self, schedule:Schedule)->Schedule:

        """
        Swaps 2 adjacent blocks (start time and duration)

        """

        #if np.log(np.random.rand()) >= np.log(self.proba):
        #    return schedule

        act_in_sched = schedule.list_act
        if len(act_in_sched) < 3:
            #nothing to swap
            return schedule

        rnd_act = schedule.which_activity(np.random.choice(schedule.anchor_nodes))

        swap = np.random.randint(0, 2)  # if swap = 0 swap with next act, else swap with previous one

        idx = act_in_sched.index(rnd_act)
        first_act = act_in_sched[idx]
        second_act = act_in_sched[idx + (1 - 2 * swap)]

        try:
            first_act.label = second_act.label
            second_act.label = rnd_act.label


            act_in_sched[idx] = first_act
            act_in_sched[idx + (1 - 2 * swap)] = second_act

            #update list of activities in schedule
            schedule.list_act = act_in_sched


        except ValueError:
            print("Error in swapping activities")

        return schedule



    def compute_change_proba(self, prev_sched:Schedule, new_sched:Schedule, **kwargs):
        """
        Computes change probability.

        Parameters:
        ----------
        -prev_sched, new_sched: Schedule objects
        """
        proba = 0.5 * 1/len(prev_sched.anchor_nodes)
        return proba



class Mode(Operator):
    """
    Mode operator: changes mode of transportation associated with activity

    Parameters:
    -------------
    proba: probability of the operator
    list_modes: list of possible modes to choose from
    p_modes: associated probabilities
    """
    def __init__(self, proba, list_modes = None, p_modes = None, **kwargs):
        super().__init__("Mode", proba)

        if list_modes:
            self.list_modes = list_modes
        else:
            self.list_modes = ["driving","pt","cycling"]

        if p_modes:
            self.p_modes = p_modes
        else:
            self.p_modes= [1/len(self.list_modes)]*len(self.list_modes)


    def apply_change(self, schedule:Schedule)->Schedule:
        """
        This operator randomly changes the travel mode of the selected activty
        """


        #if np.log(np.random.rand()) >= np.log(self.proba):
        #    return schedule


        act_in_sched = [x for x in schedule.list_act if x]
        #acts = schedule.get_list_act()

        rnd_act = schedule.which_activity(np.random.choice(schedule.anchor_nodes))
        rnd_mode = np.random.choice(self.list_modes, p = self.p_modes)  # select random modes in list


        try:
            #modify mode of current activity
            rnd_act.mode = rnd_mode
            idx = act_in_sched.index(rnd_act)
            act_in_sched[idx] = rnd_act

            schedule.list_act = act_in_sched



        except ValueError:
            print("Error in changing the mode of activity")
            print(schedule, rnd_act, rnd_mode)

        return schedule

    def compute_change_proba(self, prev_sched:Schedule, new_sched:Schedule, **kwargs):
        """
        Computes change probability.

        Parameters:
        ----------
        -prev_sched, new_sched: Schedule objects
        """

        prev_modes = [x.mode for x in prev_sched.list_act]
        new_modes = [x.mode for x in new_sched.list_act]

        if prev_modes == new_modes:
            proba = (1/len(prev_sched.anchor_nodes))*(1/len(self.list_modes))

        else:
            proba = (1/len(prev_sched.anchor_nodes))*(1 - 1/len(self.list_modes))

        return proba


class Location(Operator):
    """
    Location operator: changes location associated with activity

    Parameters:
    -------------
    proba: probability of the operator
    list_loc: list of possible locations to choose from
    p_loc: associated probabilities
    """
    def __init__(self, proba, list_loc=None, p_loc=None, **kwargs):
        super().__init__("Location", proba)
        self.list_loc = list_loc
        self.p_loc = p_loc

    def apply_change(self, schedule: Schedule)->Schedule:
        """
        This operator randomly changes the travel mode of the selected activty
        """
        #if np.log(np.random.rand()) >= np.log(self.proba):
        #    return schedule

        if self.list_loc is None:
            self.list_loc = set(schedule.all_locations)
            self.p_loc = [1/len(self.list_loc)]*len(self.list_loc)

        elif self.p_loc is None:
            self.p_loc = [1/len(self.list_loc)]*len(self.list_loc)


        act_in_sched = [x for x in schedule.list_act if x]

        rnd_act = schedule.which_activity(np.random.choice(schedule.anchor_nodes))
        rnd_loc = np.random.choice(self.list_loc, p=self.p_loc)  # select random modes in list


        try:
            #modify location of current activity
            rnd_act.location = rnd_loc
            idx = act_in_sched.index(rnd_act)
            act_in_sched[idx] = rnd_act

            schedule.list_act = act_in_sched

        except ValueError:
            print("Error in changing the location of activity")
            print(schedule, rnd_act, rnd_loc)

        return schedule

    def compute_change_proba(self, prev_sched, new_sched, **kwargs):
        prev_loc = [x.location for x in prev_sched.list_act]
        new_loc = [x.location for x in new_sched.list_act]

        if prev_loc == new_loc:
            proba = (1/len(prev_sched.anchor_nodes))*(1/len(self.list_loc))

        else:
            proba = (1/len(prev_sched.anchor_nodes))*(1 - 1/len(self.list_loc))

        return proba


class MetaOperator(Operator):
    """
    Meta operator: implements a combination of operators

    Parameters:
    -------------
    proba: probability of the operator
    n_op: number of operators to combine
    proba_operators: probabilities of each operato
    operators: list of possible operators

    """
    def __init__(self, proba:float, n_op:int, proba_operators:float, operators:Optional[List]=None, **kwargs):
        super().__init__("MetaOperator", proba)
        self.n_op = n_op
        self.proba_operators = proba_operators
        self.operators = operators
        self._meta_type = self.optype

    @property
    def meta_type(self):
        return self._meta_type

    @meta_type.setter
    def meta_type(self, optype:str):
        self._meta_type = optype

    def set_operators(self) -> None:
        """Creates combination of operators"""

        if (self.n_op > len(self.list_operators)) or (self.n_op <= 1):
            raise ValueError("Invalid number of operators for combination")

        factory = OperatorFactory()
        idx_op = np.random.choice(range(len(self.list_operators)), self.n_op)
        objects = []

        for idx in idx_op:
            operator = self.list_operators[idx]
            obj = factory.create(operator, proba=self.proba_operators[idx])
            objects.append(obj)
            self.meta_type += f"_{operator}"

        self.operators = objects

    def apply_change(self, schedule:Schedule)-> Schedule:

        """Applies combined changes of operators"""

        if self.operators is None:
            self.set_operators()

        for operator in self.operators:
            schedule = operator.apply_change(schedule)
        return schedule

    def compute_change_proba(self, prev_schedule:Schedule, new_schedule:Schedule, time:int=24, n_activities:int=8, n_discret:int=4, **kwargs) -> float:
        params = {"prev_sched": prev_schedule,
        "new_sched": new_schedule,
        "time": time,
        "n_activities": n_activities,
        "n_discret": n_discret}

        proba = 1

        for operator in self.operators:
            proba *= operator.compute_change_proba(**params)

        return proba


class OperatorFactory:
    """
    This class creates an object from the Operator class.
    """
    def __init__(self):
        pass

    def create(self, optype:str, **kwargs) -> Operator:
        """Creates an object from the Operator class

        Parameters
        -----------
        -optype: label of operator
        -kwargs: other keyword arguments that will be passed to the constructor of the Operator class.

        Return
        ----------
        Operator

        """
        object = globals()[optype](**kwargs)
        return object

    def draw(self, list_operators: List, p_operators: Optional[List] =None, **kwargs) -> Operator:
        """
        Randomly creates an object from the Operator class, given a list of possible operators and probabilities.

        Parameters
        -----------
        -list_operators: list of possible operators to choose from
        -p_operators: list of operator probabilities (must be the same length as list_operators and sum up to 1)
        -kwargs: other keyword arguments that will be passed to the constructor of the Operator class.

        Return
        ----------
        Operator
        """

        if not p_operators:
            p_operators = len(list_operators)*[1/len(list_operators)]

        rnd_op = np.random.choice(range(len(list_operators)), p=p_operators)
        n_op = None

        if list_operators[rnd_op] == 'MetaOperator':
            #Choose how many operators to combine
            n_op = np.random.randint(2, len(list_operators))


        operator = self.create(list_operators[rnd_op], proba = p_operators[rnd_op], n_op = n_op, proba_operators = p_operators, **kwargs)

        return operator
