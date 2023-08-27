import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from typing import List, Union, Optional
from data_utils import plot_schedule, bootstrap_mean, print_time_format, discretize_sched, activity_colors

from collections import defaultdict

class Results():
    """Class to handle optimization results, plot schedules and compute statistics.

    Attributes:
    ---------------
    - solutions: list of dataframes containing the optimized schedules
    - runtimes: list containing the runtimes of each iteration
    - n_iter: number of simulation iterations.

    Methods:
    ---------------
    - plot: plots the given schedules
    - compute_statistics: compute average duration and frequency for each activity in the optimal schedules
    - plot_distribution:
    - get_solutions: returns list of optimized schedules
    - get_runtimes: returns list of runtimes for each iteration.
    """

    def __init__(self, solutions: Optional[List[pd.DataFrame]]=None, runtimes: Optional[List[float]]=None, objective_values: Optional[List[float]]=None, multiday: bool = False, day_index: Optional[List] = None) -> None:
        """
        Parameters:
        ---------------
        - solutions: list of dataframes containing the optimized schedules
        - runtimes: list containing the runtimes of each iteration
        """
        self.solutions = solutions
        self.runtimes = runtimes
        self.objective_values = objective_values

        self.multiday = multiday
        self.day_index = day_index
        self.n_days = len(self.day_index) if self.day_index else 1
        self.day_names = {i: day_name for i, day_name in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],1)}

        if self.solutions:
            self.n_iter= len(self.solutions)
        else:
            self.n_iter = 0

    def __str__(self) -> str:
        return f'Results object for {self.n_iter} iterations. Total runtime:' + print_time_format(sum(self.runtimes))


    def plot(self, plot_every: int = 1, plot_iter: Optional[int] = None, colors : str = 'colorblind', title : Optional[str] = None, save_fig: Optional[str] = 'png') -> None:
        """
        Plots a given schedule.

        Parameters:
        ---------------
        - plot_every: plotting frequency as a number of iterations
        - plot_iter: index of iteration to plot
        - colors: name of matplotlib/seaborn compatible palette, see options here: https://seaborn.pydata.org/tutorial/color_palettes.html
        - title: plot title as a string
        - save_fig: export format (png/pdf/svg) as string. if None, the figure is not saved.


        Return:
        ---------------
        Matplotlib figure, either printed or saved to an external file if save_fig is not None.
        """

        if self.n_iter == 0:
            print('There is no schedule to plot.')
            return None

        if isinstance(plot_iter,int):
            sol = self.solutions[plot_iter]
            if not sol:
                print('There is no schedule to plot. The optimisation might not have been successful.')
                return None

            if self.multiday:
                fig = self.plot_multiday(sol, colors)
            else:
                fig, ax = plt.subplots(figsize=[20, 3])
                plot_schedule(sol, ax, colors)

            if title:
                plt.title(title, fontsize = 14, fontweight = 'bold')

            if save_fig:
                filename = f'schedule{plot_iter}.{save_fig}'
                plt.savefig(filename,format = save_fig)
                print(f'Figure saved at {filename}.')

            else:
                plt.show()
            return None


        for i,sol in enumerate(self.solutions):
            if i%plot_every == 0:

                if self.multiday:
                    fig = self.plot_multiday(sol, colors)
                else:
                    fig, ax = plt.subplots(figsize=[20, 3])
                    plot_schedule(sol, ax, colors)

                if title:
                    plt.title(title, fontsize = 14, fontweight = 'bold')

                if save_fig:
                    filename = f'schedule{i}.{save_fig}'
                    plt.savefig(filename,format = save_fig)
                    print(f'Figure saved at {filename}.')

                else:
                    plt.show()
        return None

    def plot_multiday(self, multi_schedules: List, colors : str = 'colorblind'):
        '''
        Plots schedules for the multiday case.

        Parameters:
        ---------------
        - multischedules: list of schedules in the multiday time horizon
        - colors: name of matplotlib/seaborn compatible palette, see options here: https://seaborn.pydata.org/tutorial/color_palettes.html

        Returns:
        ---------------
        - Matplotlib figure
        '''

        fig,axs = plt.subplots(self.n_days, 1, figsize=[20, (3*self.n_days +2)])

        for j, day_sched in enumerate(multi_schedules):
            plot_schedule(day_sched, axs[j])
            if self.day_index:
                axs[j].set_title(f'{self.day_names[self.day_index[j]]}', fontweight = 'bold', fontsize = 14)

        plt.tight_layout()
        return fig


    def compute_statistics(self, activities : List = ['education', 'leisure', 'work', 'shopping'], days: Optional[List] = None, bootstrap: int = 100, verbose: bool = True, save: Union[bool, str] = 'out_stats.joblib') -> None:
        """
        Compute aggregate statistics for the optimized schedules.

        Parameters:
        ---------------
        - activities: list of activities of interest for the computations.
        - days: list of days of interest for the computations (only defined in the multiday case)
        - bootstrap: number of bootstrap samples to generate, to compute the 95% confidence intervals.
        - verbose: if True, prints computed statistics.
        - save: if filename is provided, save statistics to file

        Return:
        ---------------
        List of computed statistics, either saved or printed
        """

        all_solutions = []

        if self.multiday:
            if not days:
                days  = self.day_index #aggregate all days to compute distributions

            all_solutions = [day_sol for solution in self.solutions for i, day_sol in enumerate(solution) if self.day_index[i] in days]

        else:
            all_solutions = self.solutions


        for sol in all_solutions:
            sol['act_label'] = sol.label.apply(lambda x: 'home' if x.rstrip('0123456789') in ['dawn', 'dusk'] else x.rstrip('0123456789'))

        sol_ooh = [s for s in all_solutions if len(s.act_label.unique()) > 1] #only out of home solutions

        #------------------------------Proportion of out-of-home schedules----------------------------------------------
        f_ooh = 100 * len(sol_ooh) / len(all_solutions)

        #----------------------------- Average total time out of home (for out of home schedules) ----------------------
        mean_time_ooh = np.mean([d[d.act_label != 'home'].duration.sum() for d in sol_ooh])
        mean_time_ooh_bs, ci_time = bootstrap_mean([d[d.act_label != 'home'].duration.sum() for d in sol_ooh], bootstrap)

        #------------------------- Average number of activities out of home (for out of home schedules)------------------
        mean_act_ooh = np.mean([len(d[d.act_label != 'home'].index) for d in sol_ooh])
        mean_act_ooh_bs, ci_act = bootstrap_mean([len(d[d.act_label != 'home'].index) for d in sol_ooh], bootstrap)

        #---------------------  Average time spent in each activity (for out of home schedules)-------------------------
        mean_time_per_act = [np.mean([d[d.act_label==a].duration.sum() for d in sol_ooh if a in d.act_label.unique()]) for a in activities]
        mean_time_per_act_bs = [bootstrap_mean([d[d.act_label==a].duration.sum() for d in sol_ooh if a in d.act_label.unique()],bootstrap)[0] for a in activities]
        ci_time_act = [bootstrap_mean([d[d.act_label==a].duration.sum() for d in sol_ooh if a in d.act_label.unique()],bootstrap)[1] for a in activities]

        if verbose:
            addition = ' '
            if days:
                if len(days) > 1:
                    con = "and" if len(days) == 2 else "to"
                    addition = f'({self.day_names[min(days)]} {con} {self.day_names[max(days)]})'
                else:
                    addition = f'({self.day_names[min(days)]})'

            print('Summary of collected statistics '+self.multiday*addition)
            print('------------------------------------------------\n')
            print(f'Total number of schedules: {len(all_solutions)}')
            print(f'Proportion of out-of-home schedules: {f_ooh:.2f} %')
            print(f'Average time spent out-of-home: {mean_time_ooh:.2f}, CI: [{ci_time[0]:.3f},{ci_time[1]:.3f}] hours')
            print(f'Average number of out-of-home activities: {mean_act_ooh:.2f}, CI: [{ci_act[0]:.3f}, {ci_act[1]:.3f}]')
            print('------------------------------------------------\n')
            print('Average duration of each activity:')

            for i, act in enumerate(activities):
                print(f'{act.capitalize()}: {mean_time_per_act[i]:.2f}, CI: [{ci_time_act[i][0]:.3f}, {ci_time_act[i][1]:.3f}] hours')
            print('------------------------------------------------\n')

        if save and isinstance(save, str):
            save_dict = {'frequency_ooh': f_ooh,
            'mean_time_ooh': mean_time_ooh,
            'mean_time_ooh_bs':mean_time_ooh_bs,
            'mean_act_ooh': mean_act_ooh,
            'mean_act_ooh_bs': mean_act_ooh_bs,
            'mean_time_per_act':mean_time_per_act,
            ' mean_time_per_act_bs': mean_time_per_act_bs}
            joblib.dump(save_dict, save)

            print(f'Saved statistics to: {save}')

        return None


    def plot_distribution(self, exclude: Optional[List]= ["escort", "business_trip", "errands_services"], block_size: float = 5/60, days: Optional[List] = None, figure_size: List = [7,4], save_fig: Optional[str] = 'png')-> None:
        """
        Plots aggregate time of  day distribution.

        Parameters:
        ---------------
        - exclude: list of activities to exclude from the visualization
        - block_size: size of the discretization in hours. Default: 5/60 hours.
        - days: list of days of interest for the aggregation (only defined in the multiday case)
        - figure_size: size of figure
        - save_fig: xport format (png/pdf/svg) as string. if None, the figure is not saved.

        Return:
        ---------------
        Matplotlib figure, either printed or saved to an external file if save_fig is not None.
        """

        disc_list = []
        all_solutions = []

        if self.multiday:
            if not days:
                days  = self.day_index #aggregate all days to plot the distributions

            all_solutions = [day_sol for solution in self.solutions for i, day_sol in enumerate(solution) if self.day_index[i] in days]

        else:
            all_solutions = self.solutions

        for i,s in enumerate(all_solutions):

            s['act_label'] = s.label.apply(lambda x: x.rstrip('0123456789') if x not in ['dawn', 'dusk'] else 'home')
            list_act = [x for x in s.act_label.unique() if x not in exclude]

            if len(s.act_label.unique()) == 1:
                continue

            s = s[~s.act_label.isin(exclude)]
            discret_s = discretize_sched(s, block_size = block_size)
            disc_list.append(discret_s)


        final_dict = defaultdict(list) #dictionary storing start times distributions for all activities
        time_slots = range(int(24/block_size))

        for t in time_slots:
            for dicts in disc_list:
                if t in dicts.keys():
                    final_dict[t].append(dicts[t])

        disc_df = pd.DataFrame.from_dict(final_dict, orient = 'index').transpose().melt(var_name = 'time', value_name= 'activity')
        disc_df['time'] = disc_df.time.apply(lambda x: round(x * block_size))

        disc_grouped = (disc_df.groupby(['time', 'activity'])['activity'].count()/disc_df.groupby('time')['activity'].count())


        colors = activity_colors(palette = "colorblind")
        colors['home'] = 'gainsboro'

        fig, ax = plt.subplots(figsize = figure_size)
        ax.set_facecolor('gainsboro')

        disc_grouped.unstack().drop('home', axis = 1).plot.bar(stacked = True, color = colors, edgecolor = 'white', width = 1, ax = ax, legend = False, rot = 1)

        # LEGEND
        other_patches = [mpatches.Patch(color = f'{colors[a]}', label=f'{a}') for a in sorted(list_act)]
        plt.legend(handles=other_patches, loc='upper right', fontsize=10)
        ax.set_ylabel("Frequency ", fontsize=12)
        ax.set_xlabel("Time [h]", fontsize=12)
        ax.set_xticks(range(0, 25, 4))
        ax.set_xticklabels(range(0, 25, 4))
        ax.set_xlim([0, 25])
        ax.set_ylim([0, 1])

        title = "Time of day distribution"

        if days:
            if len(days) > 1:
                con = "and" if len(days) == 2 else "to"
                title = f'Time of day distribution, ({self.day_names[min(days)]} {con} {self.day_names[max(days)]})'
            else:
                title = f'Time of day distribution, ({self.day_names[min(days)]})'

        plt.title(title, fontsize = 12)

        if save_fig:
            filename = f'time_of_day_dist.{save_fig}'
            plt.savefig(filename,format = save_fig)
            print(f'Figure saved at {filename}.')

        else:
            plt.show()

        return None


    def get_solutions(self) -> List:
        """Returns list of optimized schedules"""
        return self.solutions

    def get_runtimes(self) -> List:
        """Returns list of runtimes for each iteration."""
        return self.runtimes

    def get_objective_values(self) -> List:
        """Returns list of objective_values for each iteration."""
        return self.objective_values
