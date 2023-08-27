import numpy as np
import pandas as pd

import time

from docplex.mp.model import Model
from docplex.mp.dvar import Var
from typing import List, Dict, Union, Optional
from data_utils import cplex_to_df, print_time_format

from input_data import ActivityData
from results import Results


class OptimModel():
    """
    This class instanciates an optimisation model.

    Attributes:
    ---------------
    - solver: String, 'MIP' or 'CP'
    - activities: List of unique activities (ActivityData objects) to be scheduled
    - utility_parameters: Dictionary containing non activity-specific parameters to use in the utility function.  The format should be {param: value}.
    - opt_settings: Dictionary containing settings that will be passed to the solver
    - solve_status: Status of the optimisation problem

    Methods:
    ---------------
    - utility_function: defines the activity-specific utility function (overriden by children classes)
    - objective_function: defines the schedule-specific utility function to be maximized (overriden by children classes)
    - initialize: creates the model object, with decision variable and constraints (overriden by children classes)
    - check_input: checks if the input data is corret for the type of simulation selected (overriden by children classes)
    """
    def __init__(self, solver: str,activities: List[ActivityData], utility_parameters: Dict,  optimality_target: int = 3, time_limit:int = 120, *args, **kwargs) -> None:
        """
        Parameters:
        ---------------
        - solver: String, 'MIP' or 'CP'
        - activities: List of unique activities (ActivityData objects) to be scheduled
        - utility_parameters: Dictionary containing non activity-specific parameters to use in the utility function.  The format should be {param: value}.
        - opt_settings: Dictionary containing settings that will be passed to the solver
        """
        self.solver = solver
        self.activities = activities
        self.utility_parameters = utility_parameters
        self.optimality_target = optimality_target
        self.time_limit = time_limit

        self.solve_status = None
        self.solve_details = None


    def utility_function(self) -> None:
        """
        Defines the activity-specific utility function (overriden by children classes)
        """
        print("Please define a utility function for each activity.")

    def objective_function(self) -> None:
        """
        Defines the schedule-specific utility function to be maximized (overriden by children classes)
        """
        print("Please define an objective function for the maximization problem.")

    def initialize(self) -> None:
        """
        Creates the model object, with decision variable and constraints (overriden by children classes)
        """
        return None

    def check_input(self) -> None:
        """Checks if the input data is correct for the type of simulation"""
        return None





class MIP(OptimModel):
    """
    This class instanciates a MIP optimisation model (relies on docplex library).

    Attributes:
    ---------------
    - solver: String, 'MIP'
    - activities: List of unique activities (ActivityData objects) to be scheduled
    - utility_parameters: Dictionary containing non activity-specific parameters to use in the utility function.  The format should be {param: value}.
    - travel_times: Dictionary containing the mode specific travel times. The format should be {mode: {origin: {destination_1: travel time, destination_2...}}}
    - distances: Dictionary containing the mode specific distances. The format is the same as travel_times.
    - period: Time budget in hours. Default is 24h
    - model: model object
    - keys: unique labels of the activities to be scheduled

    Methods:
    ---------------
    - add_constraint: Adds a single constraint to the model object.
    - add_constraints: Adds multiple constraints to the model object, in batch.
    - initialize: Creates the model object, with decision variable and constraints (overrides parent method)
    - utility_function: Defines the activity-specific utility function (overrides parent method)
    - objective_function: Defines the schedule-specific utility function to be maximized (overrides parent method)
    - solve: Solves optimization problem
    - run: Runs the simulation
    - clear: Deletes model object and associated variables/constraints
    - check_input:  checks if the input data is corret for the type of simulation selected (overrides parent method)
    """
    def __init__(self, activities: List[ActivityData], utility_parameters: Dict, travel_times: Dict, distances: Optional[Dict] = None, period: int= 24, *args, **kwargs) -> None:
        super().__init__("MIP", activities, utility_parameters, *args, **kwargs)
        """
        Parameters
        ---------------
        - activities: List of unique activities (ActivityData objects) to be scheduled
        - utility_parameters: Dictionary containing non activity-specific parameters to use in the utility function.  The format should be {param: value}.
        - travel_times: Dictionary containing the mode specific travel times. The format should be {mode: {origin: {destination_1: travel time, destination_2...}}}
        - distances: Dictionary containing the mode specific distances. The format is the same as travel_times.
        - period: Time budget in hours. Default is 24h

        """

        self.travel_times = travel_times
        self.distances = distances
        self.period = period

        self.model = None

        #labels for decision variables
        self.keys = [act.label for act in self.activities]


    def add_constraint(self, constraint) -> None:
        """Calls docplex add_constraint() function. Adds a single constraint to the model object.

        Parameters
        ---------------
        Constraint: mathematical expression."""

        self.model.add_constraint(constraint)

    def add_constraints(self, list_of_constraints: List) -> None:
        """Calls docplex add_constraints() function. Adds a list of constraints to the model object.

        Parameters
        ---------------
        list_of_constraints: list of mathematical expressions."""

        self.model.add_constraints(list_of_constraints)

    def clear(self) -> None:
        """Deletes model object and associated variables and constraints."""
        self.model = None
        self.x = None #start time
        self.z = None #activity sequence indicator
        self.d = None #duration
        self.w = None #indicator of  activity choice
        self.tt = None #travel time
        self.tc = None #travel cost
        self.md_car = None #mode of transportation (availability)

        self.error_w = None
        self.error_z = None
        self.error_x = None
        self.error_d = None



    def initialize(self) -> None:
        """
        Creates the model object, with decision variable and constraints
        """
        self.clear()

        #-----------------------Create a docplex model object------------------------#
        self.model = Model()
        self.model.parameters.optimalitytarget = self.optimality_target
        self.model.parameters.timelimit = self.time_limit

        #-----------------------Add (default) decision variables------------------------#
        self.x = self.model.continuous_var_dict(self.keys, lb = 0, name = 'x') #start time
        self.z = self.model.binary_var_matrix(self.keys, self.keys, name = 'z') #activity sequence indicator
        self.d = self.model.continuous_var_dict(self.keys, lb = 0, name = 'd') #duration
        self.w = self.model.binary_var_dict(self.keys, name = 'w') #indicator of  activity choice
        self.tt = self.model.continuous_var_dict(self.keys, lb = 0, name = 'tt') #travel time
        self.tc = self.model.continuous_var_dict(self.keys, lb = 0, name = 'tc') #travel cost
        self.md_car = self.model.binary_var_dict(self.keys, name = 'md') #mode of transportation (availability)

        #------------------------Add (default) constraints:------------------------------#

        #Budget constraint
        self.add_constraint(self.model.sum(self.d[a] + self.tt[a] for a in self.keys) == self.period)

        #Start at home
        self.add_constraint(self.x['dawn'] == 0)

        #End at home
        self.add_constraint(self.x['dusk']+ self.d['dusk'] == self.period)

        for act in self.activities:
            a = act.label

            #Sequence constraints
            self.add_constraints(self.z[a,b] + self.z[b,a] <= 1 for b in self.keys if b != a)
            self.add_constraint(self.z[a,'dawn'] == 0 )
            self.add_constraint(self.z['dusk',a] == 0 )
            self.add_constraint(self.z[a,a] == 0)

            #Consistency constraints
            self.add_constraints(self.x[a] + self.d[a] + self.tt[a] - self.x[b] >= (self.z[a,b]-1)*self.period for b in self.keys)
            self.add_constraints(self.x[a] + self.d[a] + self.tt[a] - self.x[b] <= (1-self.z[a,b])*self.period for b in self.keys)

            #Travel time constraint
            self.add_constraint(self.tt[a] == self.model.sum(self.z[a,actb.label]*self.travel_times[act.mode_travel][act.location][actb.location] for actb in self.activities))

            #Travel cost constraint
            #self.add_constraint(tc[a] == self.model.sum(z[a,actb.label]*self.costs_travel[act.mode]*self.distances[act.location][actb.location] for actb in self.activities))

            #Car availability at home
            if act.group in ["home", "dawn", "dusk"]:
                self.add_constraint(self.md_car[a] == 1)

            if act.mode_travel == "driving":
                self.add_constraint(self.w[a] <= self.md_car[a])

            #Mode consistency
            self.add_constraints(self.md_car[a] >=  self.md_car[b] + self.z[a,b] - 1 for b in self.keys)
            self.add_constraints(self.md_car[b] >=  self.md_car[a] + self.z[a,b] - 1 for b in self.keys)

            #No duration if activity not performed
            self.add_constraint(self.w[a] <= self.d[a])
            self.add_constraint(self.d[a] <= self.w[a]*self.period)

            #Feasible time windows
            self.add_constraint(self.x[a] >= act.feasible_start)
            self.add_constraint(self.x[a] + self.d[a] <= act.feasible_end)

            #No group duplicates
            self.add_constraint(self.model.sum(self.w[actb.label] for actb in self.activities if actb.group == act.group)<=1)

            if a != 'dawn':
                #predecessor constraint
                self.add_constraint(self.model.sum(self.z[b,a] for b in self.keys if b !=a) == self.w[a])
            if a != 'dusk':
                #successor constraint
                self.add_constraint(self.model.sum(self.z[a,b] for b in self.keys if b !=a) == self.w[a] )

        #------------------------Add (default) error terms:------------------------------#

        #draw new error terms
        error_w = self.utility_parameters['error_w'].draw(2)
        error_z = self.utility_parameters['error_z'].draw(2)
        error_x = self.utility_parameters['error_x'].draw(4)
        error_d = self.utility_parameters['error_d'].draw(6)

        self.error_w = self.model.piecewise(0, [(k,error_w[k]) for k in [0,1]], 0)
        self.error_z = self.model.piecewise(0, [(k,error_z[k]) for k in [0,1]], 0)
        self.error_x = self.model.piecewise(0, [(a, error_x[b]) for a,b in zip(np.arange(0, 24, 6), np.arange(4))], 0)
        self.error_d = self.model.piecewise(0, [(a, error_d[b]) for a,b in zip([0, 1, 3, 8, 12, 16], np.arange(6))], error_d[-1])

        #------------------------Add objective function---------------------------------#
        self.model.maximize(self.objective_function())


    def utility_function(self, activity: ActivityData) -> float:
        """Activity-specific utility function to be defined by the user. Default is linear utility with penalties for schedule deviations (start time and duration)

        Parameters
        ---------------
        activity: ActivityData object

        Returns
        ---------------
        utility: value of activity specific utility for current decision variables
        """
        parameters = activity.activity_parameters
        a = activity.label

        utility = (self.w[a] * (parameters['constant'] +
        #penalties start time
        parameters['early'] * self.model.max(activity.desired_start-self.x[a], 0) +
        parameters['late'] * self.model.max(self.x[a]-activity.desired_start, 0) +

        #penalties duration
        parameters['short'] * self.model.max(activity.desired_duration-self.d[a], 0) +
        parameters['long'] * self.model.max(self.d[a] - activity.desired_duration, 0) +

        #penalties travel (time and cost)
        parameters['travel_time']* self.tt[a]
        #+parameters['travel_cost'] * self.tc[a]

        #activity cost
        #+ parameters['activity_cost'] * costs_activity[act_id[a]])

        #error terms
        + self.error_x(self.x[a])
        + self.error_d(self.d[a])
        + self.model.sum(self.error_z(self.z[a,b]) for b in self.keys)
        ) + self.error_w(self.w[a]))

        return utility

    def objective_function(self)-> float:
        """
        Objective function of the simulation, to be defined by the user.
        default is the sum of utility functions of all activities in schedule.

        Returns
        ---------------
        of: value of objective function for current decision variables
        """
        of = self.model.sum([self.utility_function(a) for a in self.activities]) + self.utility_parameters['error_ev'].draw()
        return of

    def _solve(self) -> (pd.DataFrame, float):
        """
        Solves optimization problem

        Returns
        ---------------
        solution: pandas DataFrame containing the optimal schedule
        runtime: iteration runtime in seconds
        """

        start_time = time.time()
        solution = self.model.solve()
        runtime = time.time() - start_time

        objective_function = None

        if solution:
            mode_travel = {a.label: a.mode_travel for a in self.activities}
            location = {a.label: a.location for a in self.activities}
            act_id = {a.label: a.act_id for a in self.activities}

            solution_df = cplex_to_df(self.w, self.x, self.d, self.tt, self.md_car, mode_travel, self.keys, act_id, location)
            objective_function = solution.get_objective_value()

            self.solve_status = True
            self.solve_details = self.model.solve_details

        return solution_df, runtime, objective_function


    def run(self, n_iter: int = 1, verbose: Union[bool, int] = False) -> Results :
        """
        Runs the simulation.

        Parameters
        ---------------
        - n_iter: number of iterations
        - verbose: if int, prints simulation progress.

        Returns
        ---------------
        Object from Results class.
        """

        # Print to console if verbose and iteration correct multiple
        console_interval = verbose if isinstance(verbose, int) else n_iter+1

        all_runs = []
        all_solutions = []
        all_objectives = []
        print(f"Starting simulation: {n_iter} iterations.")
        print(f"-----------------------------------------")
        start_time = time.time()

        for i in range(n_iter):
            self.initialize()
            if verbose and (i+1) % console_interval == 0:
                print(f"Starting iteration {i+1}/{n_iter}.")
            sol, run, obj = self._solve()
            all_runs.append(run)
            all_solutions.append(sol)
            all_objectives.append(obj)

            if verbose and (i+1) % console_interval == 0:
                print(f"Iteration {i+1} complete. Iteration runtime: {print_time_format(run)}. Time elapsed: {print_time_format(time.time()-start_time)}.")

        print(f"-----------------------------------------")
        print(f"Simulation complete. Total runtime: {print_time_format(time.time()-start_time)}")


        return Results(all_solutions, all_runs, all_objectives)


class MultidayMIP(OptimModel):
    """
    This class instanciates a MIP optimisation model (relies on docplex library) for multiday analyses.

    Attributes:
    ---------------
    - solver: String, 'MultidayMIP'
    - activities: List of unique activities (ActivityData objects) to be scheduled
    - utility_parameters: Dictionary containing non activity-specific parameters to use in the utility function.  The format should be {param: value}.
    - travel_times: Dictionary containing the mode specific travel times. The format should be {mode: {origin: {destination_1: travel time, destination_2...}}}
    - distances: Dictionary containing the mode specific distances. The format is the same as travel_times.
    - period: Time budget in hours. Default is 24h
    - n_days: Number of days. The total time horizon is computed a sn_days*period
    - day_index: List of indices of the days to be scheduled. (E.g., for a full week, [1,..,7] with 1 being Monday and 7 being Sunday)
    - model: model object
    - keys: unique labels of the activities to be scheduled

    Methods:
    ---------------
    - add_constraint: Adds a single constraint to the model object.
    - add_constraints: Adds multiple constraints to the model object, in batch.
    - initialize: Creates the model object, with decision variable and constraints (overrides parent method)
    - utility_function: Defines the activity-specific utility function (overrides parent method)
    - objective_function: Defines the schedule-specific utility function to be maximized (overrides parent method)
    - solve: Solves optimization problem
    - run: Runs the simulation
    - clear: Deletes model object and associated variables/constraints
    """
    def __init__(self, activities: List[ActivityData], utility_parameters: Dict, travel_times: Dict, n_days: int, day_index: Optional[List] = None, distances: Optional[Dict] = None, period: int= 24,  *args, **kwargs) -> None:
        super().__init__("MIP", activities, utility_parameters, *args, **kwargs)
        """
        Parameters
        ---------------
        - activities: List of unique activities (ActivityData objects) to be scheduled
        - utility_parameters: Dictionary containing non activity-specific parameters to use in the utility function.  The format should be {param: value}.
        - travel_times: Dictionary containing the mode specific travel times. The format should be {mode: {origin: {destination_1: travel time, destination_2...}}}
        - n_days: number of days to be scheduled
        - day_index: List containing the indices of the days of the week (1 is Monday and 7 is Sunday)
        - distances: Dictionary containing the mode specific distances. The format is the same as travel_times.
        - period: Time budget in hours. Default is 24h

        """

        self.travel_times = travel_times
        self.distances = distances
        self.period = period
        self.n_days = n_days

        if  (day_index is None) or (len(day_index) != n_days) :
            self.day_index = [x for x in range(1, n_days+1)]
        else:
            self.day_index = day_index


        self.model = None

        #labels for decision variables
        self.keys = [act.label for act in self.activities]

    def add_constraint(self, constraint) -> None:
        """Calls docplex add_constraint() function. Adds a single constraint to the model object.

        Parameters
        ---------------
        Constraint: mathematical expression."""

        self.model.add_constraint(constraint)

    def add_constraints(self, list_of_constraints: List) -> None:
        """Calls docplex add_constraints() function. Adds a list of constraints to the model object.

        Parameters
        ---------------
        list_of_constraints: list of mathematical expressions."""

        self.model.add_constraints(list_of_constraints)

    def clear(self) -> None:
        """Deletes model object and associated variables and constraints."""
        self.model = None
        self.x = None #start time
        self.z = None #activity sequence indicator
        self.d = None #duration
        self.w = None #indicator of  activity choice
        self.tt = None #travel time
        self.tc = None #travel cost
        self.md_car = None #mode of transportation (availability)

        self.error_w = None
        self.error_z = None
        self.error_x = None
        self.error_d = None
        self.error_day = None



    def initialize(self) -> None:
        """
        Creates the model object, with decision variable and constraints
        """
        self.clear()

        #-----------------------Create a docplex model object------------------------#
        self.model = Model()
        self.model.parameters.optimalitytarget = self.optimality_target
        self.model.parameters.timelimit = self.time_limit

        #-----------------------Add (default) decision variables------------------------#
        self.x = [self.model.continuous_var_dict(self.keys, lb = 0, name = f'x{i}') for i in self.day_index] #start time
        self.z = [self.model.binary_var_matrix(self.keys, self.keys, name = f'z{i}') for i in self.day_index]#activity sequence indicator
        self.d = [self.model.continuous_var_dict(self.keys, lb = 0, name = f'd{i}') for i in self.day_index] #duration
        self.w = [self.model.binary_var_dict(self.keys, name = f'w{i}') for i in self.day_index] #indicator of  activity choice
        self.tt = [self.model.continuous_var_dict(self.keys, lb = 0, name = f'tt{i}') for i in self.day_index] #travel time
        self.tc = [self.model.continuous_var_dict(self.keys, lb = 0, name = f'tc{i}') for i in self.day_index] #travel cost
        self.md_car = [self.model.binary_var_dict(self.keys, name = f'md{i}') for i in self.day_index] #mode of transportation (availability)
        self.freq_act = self.model.continuous_var_dict(self.keys, lb = 0, name = f'freq')  #frequency of each activity over entire time budget


        #------------------------Add (default) constraints:------------------------------#

        #Frequency constraint
        self.add_constraints(self.freq_act[a] == self.model.sum(self.w[i][a] for i in range(self.n_days)) for a in self.keys)

        #Daily constraints:
        for i in range(self.n_days):

            #Budget constraint
            self.add_constraint(self.model.sum(self.d[i][a] + self.tt[i][a] for a in self.keys) == self.period)

            #Start at home
            self.add_constraint(self.x[i]['dawn'] == 0)

            #End at home
            self.add_constraint(self.x[i]['dusk']+ self.d[i]['dusk'] == self.period)

            for act in self.activities:
                a = act.label

                #Sequence constraints
                self.add_constraints(self.z[i][a,b] + self.z[i][b,a] <= 1 for b in self.keys if b != a)
                self.add_constraint(self.z[i][a,'dawn'] == 0 )
                self.add_constraint(self.z[i]['dusk',a] == 0 )
                self.add_constraint(self.z[i][a,a] == 0)

                #Consistency constraints
                self.add_constraints(self.x[i][a] + self.d[i][a] + self.tt[i][a] - self.x[i][b] >= (self.z[i][a,b]-1)*self.period for b in self.keys)
                self.add_constraints(self.x[i][a] + self.d[i][a] + self.tt[i][a] - self.x[i][b] <= (1-self.z[i][a,b])*self.period for b in self.keys)

                #Travel time constraint
                self.add_constraint(self.tt[i][a] == self.model.sum(self.z[i][a,actb.label]*self.travel_times[act.mode_travel][act.location][actb.location] for actb in self.activities))

                #Travel cost constraint
                #self.add_constraint(tc[a] == self.model.sum(z[a,actb.label]*self.costs_travel[act.mode]*self.distances[act.location][actb.location] for actb in self.activities))

                #Car availability at home
                if act.group in ["home", "dawn", "dusk"]:
                    self.add_constraint(self.md_car[i][a] == 1)

                if act.mode_travel == "driving":
                    self.add_constraint(self.w[i][a] <= self.md_car[i][a])

                #Mode consistency
                self.add_constraints(self.md_car[i][a] >=  self.md_car[i][b] + self.z[i][a,b] - 1 for b in self.keys)
                self.add_constraints(self.md_car[i][b] >=  self.md_car[i][a] + self.z[i][a,b] - 1 for b in self.keys)

                #No duration if activity not performed
                self.add_constraint(self.w[i][a] <= self.d[i][a])
                self.add_constraint(self.d[i][a] <= self.w[i][a]*self.period)

                #Feasible time windows
                self.add_constraint(self.x[i][a] >= act.feasible_start)
                self.add_constraint(self.x[i][a] + self.d[i][a] <= act.feasible_end)

                #No group duplicates
                self.add_constraint(self.model.sum(self.w[i][actb.label] for actb in self.activities if actb.group == act.group)<=1)

                if a != 'dawn':
                    #predecessor constraint
                    self.add_constraint(self.model.sum(self.z[i][b,a] for b in self.keys if b !=a) == self.w[i][a])
                if a != 'dusk':
                    #successor constraint
                    self.add_constraint(self.model.sum(self.z[i][a,b] for b in self.keys if b !=a) == self.w[i][a] )

        #------------------------Add (default) error terms:------------------------------#

        #draw new error terms -- can change these to make them day-specific
        error_w = self.utility_parameters['error_w'].draw(2)
        error_z = self.utility_parameters['error_z'].draw(2)
        error_x = self.utility_parameters['error_x'].draw(4)
        error_d = self.utility_parameters['error_d'].draw(6)

        self.error_w = [self.model.piecewise(0, [(k,error_w[k]) for k in [0,1]], 0) for i in self.day_index]
        self.error_z = [self.model.piecewise(0, [(k,error_z[k]) for k in [0,1]], 0) for i in self.day_index]
        self.error_x = [self.model.piecewise(0, [(a, error_x[b]) for a,b in zip(np.arange(0, 24, 6), np.arange(4))], 0) for i in self.day_index]
        self.error_d = [self.model.piecewise(0, [(a, error_d[b]) for a,b in zip([0, 1, 3, 8, 12, 16], np.arange(6))], error_d[-1]) for i in self.day_index]
        self.error_day = self.utility_parameters['error_day'].draw(self.n_days)
        #------------------------Add objective function---------------------------------#
        self.model.maximize(self.objective_function())


    def utility_function(self, activity: ActivityData, day: int) -> float:
        """Activity-specific utility function to be defined by the user. Default is linear utility with penalties for schedule deviations (start time and duration)

        Parameters
        ---------------
        activity: ActivityData object
        day: index of day -- 1 is Monday and 7 is Sunday

        Returns
        ---------------
        utility: value of activity specific utility for current decision variables
        """
        parameters = activity.activity_parameters
        a = activity.label

        is_weekend = day in [6,7]
        i = day-1 #0-based index of days

        desired_start = activity.desired_start_weekend if is_weekend else activity.desired_start_weekday
        desired_end = activity.desired_duration_weekend if is_weekend else activity.desired_duration_weekday

        utility = (self.w[i][a] * (parameters['constant'] +
        #penalties start time
        parameters['early'] * self.model.max(activity.desired_start-self.x[i][a], 0) +
        parameters['late'] * self.model.max(self.x[i][a]-activity.desired_start, 0) +

        #penalties duration
        parameters['short'] * self.model.max(activity.desired_duration-self.d[i][a], 0) +
        parameters['long'] * self.model.max(self.d[i][a] - activity.desired_duration, 0) +

        #penalties travel (time and cost)
        parameters['travel_time']* self.tt[i][a]+
        #+parameters['travel_cost'] * self.tc[a]

        #penalty for day preference
        parameters['weekend']*is_weekend

        #activity cost
        #+ parameters['activity_cost'] * costs_activity[act_id[a]])

        #error terms
        + self.error_x[i](self.x[i][a])
        + self.error_d[i](self.d[i][a])
        + self.model.sum(self.error_z[i](self.z[i][a,b]) for b in self.keys)
        ) + self.error_w[i](self.w[i][a])) + self.error_day[i]

        return utility

    def objective_function(self)-> float:
        """
        Objective function of the simulation, to be defined by the user.
        default is the sum of utility functions of all activities in schedule.

        Returns
        ---------------
        of: value of objective function for current decision variables
        """
        daily_utility = [self.model.sum([self.utility_function(a, d) for a in self.activities]) for d in self.day_index]
        frequency_utility = [a.activity_parameters['frequency']*self.model.abs(self.freq_act[a.label]-a.desired_frequency) for a in self.activities]

        of =  self.model.sum(daily_utility) + self.model.sum(frequency_utility) + self.utility_parameters['error_ev'].draw()

        return of

    def _solve(self) -> (pd.DataFrame, float):
        """
        Solves optimization problem

        Returns
        ---------------
        solution: pandas DataFrame containing the optimal schedule
        runtime: iteration runtime in seconds
        """

        multiday_solutions = []

        start_time = time.time()
        solution = self.model.solve()
        runtime = time.time() - start_time

        objective_function = None

        if solution:
            mode_travel = {a.label: a.mode_travel for a in self.activities}
            location = {a.label: a.location for a in self.activities}
            act_id = {a.label: a.act_id for a in self.activities}

            for i in range(self.n_days):
                solution_df = cplex_to_df(self.w[i], self.x[i], self.d[i], self.tt[i], self.md_car[i], mode_travel, self.keys, act_id, location)
                multiday_solutions.append(solution_df)


            objective_function = solution.get_objective_value()


            self.solve_status = True
            self.solve_details = self.model.solve_details

        return multiday_solutions, runtime, objective_function


    def run(self, n_iter: int = 1, verbose: Union[bool, int] = False) -> Results :
        """
        Runs the simulation.

        Parameters
        ---------------
        - n_iter: number of iterations
        - verbose: if int, prints simulation progress.

        Returns
        ---------------
        Object from Results class.
        """

        # Print to console if verbose and iteration correct multiple
        console_interval = verbose if isinstance(verbose, int) else n_iter+1

        all_runs = []
        all_solutions = []
        all_objectives = []
        print(f"Starting simulation: {n_iter} iterations.")
        print(f"-----------------------------------------")
        start_time = time.time()

        for i in range(n_iter):
            self.initialize()
            if verbose and (i+1) % console_interval == 0:
                print(f"Starting iteration {i+1}/{n_iter}.")
            sol, run, obj = self._solve()
            all_runs.append(run)
            all_solutions.append(sol)
            all_objectives.append(obj)

            if verbose and (i+1) % console_interval == 0:
                print(f"Iteration {i+1} complete. Iteration runtime: {print_time_format(run)}. Time elapsed: {print_time_format(time.time()-start_time)}.")

        print(f"-----------------------------------------")
        print(f"Simulation complete. Total runtime: {print_time_format(time.time()-start_time)}")


        return Results(all_solutions, all_runs, all_objectives, multiday = True, day_index = self.day_index)
