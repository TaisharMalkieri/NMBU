# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:51:52 2021

@author: Tormo
"""
import os
import timeit
import pickle
import copy
import json
import pdfkit
import wmi


import psutil as psu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform as sysinf
from statistics import variance, mean


from time import sleep
from algorithms.bubble_sort import bubble_sort
from algorithms.insertion_sort import insertion_sort
from algorithms.quick_sort import quick_sort
from algorithms.merge_sort import merge_sort
from algorithms.threshold_sorts import threshold_sort_quick_to_insertion, threshold_sort_merge_to_insertion
from collections import defaultdict

    

class AlgorithmSpeedTest:
    def __init__(self,seed=0, testset_paradigms=["all_sorted", "all_random", "random_inserts"],):
        """
        Args:
            seed (int, optional): [Seed is a number fed to a rng generator. The seed enables reproducability]. Defaults to 0.
            testset_paradigms (list, optional): [Possible datasets to run the algorithm on.
            Random is a randomly generated dataset based on the seed.
            Sorted is the randomly genereated testset sorted.
            Reverse is the reverse of sorted. 
            ]. Defaults to ["all_sorted", "all_random", "random_inserts"].
        """

        self.hardware_specs = {
            "Machine":sysinf.machine(),
            "Processor":sysinf.processor(),
            "Physical cores":psu.cpu_count(logical=False),
            "Virtual cores":psu.cpu_count(logical=True),
            "GB Ram": (str(round(psu.virtual_memory().total/1000000000)),),
            
            }
        
        self.system_specs = {
            "System": sysinf.platform(),
            "Release": sysinf.release(),
            "Python_version":sysinf.python_version(),
            "Python_build": sysinf.python_build(),
            "Python_compiler": sysinf.python_compiler(),
            "Python_implementation": sysinf.python_implementation(),
            "Numpy_version":np.version.version
            }
        
        self.seed = seed
        #Set an RNG number engine. Changing the seed and resetting the engine will change random number engines behaviour 
        self.rng = np.random.default_rng(seed=self.seed)
        
        self.ibase = 2
        self.ipower = 15
        self.repetiotions = 7
        self.iterations = [self.ibase**x for x in range(1, self.ipower)]
        
        
        self.save_results=True
        self.save_location="save"
        
        #If plots have legends and titles or not
        self.scientific = False
        self.axis_scale  = "log"
        #Default option suggeste for consistent plots featuring t_min, t_avg etc.
        self.y_axis_range = (10**-7, 10**3)
        self.default_plot=True
        
        self.algorithm_lib={
            "bubble_sort":{"algorithm":bubble_sort,
                           "color": "blue"},
            "insertion_sort":{"algorithm":insertion_sort,
                           "color": "green"},
            "quick_sort": {"algorithm":quick_sort,
                           "color": "red"},
            "merge_sort": {"algorithm":merge_sort,
                           "color": "magenta"},
            "python_sort":{"algorithm":sorted,
                           "color": "black"},
            "numpy_sort":{"algorithm":np.sort,
                           "color": "orange"},
            "threshold_quick_to_insertion":{"algorithm":threshold_sort_quick_to_insertion,
                                            "color":"cyan"},
            "threshold_merge_to_insertion":{"algorithm":threshold_sort_merge_to_insertion,
                                            "color":"grey"}
            }
        
        self.testset_lib = {
            "random":{"data": [lambda length: self.rng.integers(low=-10000, high=10000, size=length)],
                      "color":"k"},
            "sorted":{"data": [lambda length: sorted(self.rng.integers(low=-10000, high=10000, size=length))],
                      "color":"b"},
            "reversed":{"data": [lambda length: sorted(self.rng.integers(low=-10000, high=10000, size=length), reverse=True)],
                        "color":"r"},
            "constant": {"data":[lambda length: [5 for x in range(length)]],
                       "color":"g"}
            }
        
    def default_plot_settings(self):
        """
        Sets the plot settings defined as defined by scientific paper standards
        Scaling properties can be changed in self.axis_scale
        """
        plt.rcParams['axes.titlesize'] = 9
        plt.rcParams['axes.labelsize'] = 9
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 8
        plt.xscale(self.axis_scale)
        plt.yscale(self.axis_scale)
        plt.ylim(self.y_axis_range)
        plt.figure(figsize=(84/25.4, 55/25.4))
         
    def run_benchmark(self, algorithm, paradigm):
        """
        Args:
            algorithm (str): The algorithm to run.
            paradigm (str): The testcases from self.testset_lib to test.

        Returns:
            pd.DataFrame: A dataframe containing the calculated results of each iteration.
            The iterations are repeated self.repetion times and the average time taken for each repetition is stored.
            Useful results are then calculated as max, min, avg, variance, variance%, the number of executions done by timer.repeat and the time consumed.
            Headers in the same order.             
        """
        active_algorithm=self.algorithm_lib[algorithm]["algorithm"]
        result = {
            "t_max":[],
            "t_min":[],
            "t_avg":[],
            "t_var":[],
            "t_var_pst":[],
            "n_ar":[],
            "t_ar":[],    
            }
        
        #_visual_plot_test(self.testset_lib[paradigm]["data"][0](100), paradigm)
        for l in self.iterations:
            data = self.testset_lib[paradigm]["data"][0](length=l)
            try:
                if algorithm in ["merge_sort", "quick_sort", "threshold_quick_to_insertion", "threshold_merge_to_insertion"]:
                    p = 1
                    r=l
                    if algorithm in ["quick_sort", "threshold_quick_to_insertion"]:
                        r-=1
                        p=0
                    
                    clock = timeit.Timer(stmt='sort_func(copy(test_set),p,r)',
                    globals={'sort_func':active_algorithm,
                             'copy': copy.copy,
                             'test_set': data,
                             'p': p,
                             'r': r
                             })
                else:
                    clock = timeit.Timer(stmt='sort_func(copy(test_set))',
                    globals={'sort_func':active_algorithm,
                             'copy': copy.copy,
                             'test_set': data,
                             })
                
                """
            
                clock autorange is given a budget of 0.2sek.
                It runs the sorting func repeatedly until it breaks that budget
                autorange continues until the last iteration breaks the budget.
                t_ar tells you how much time was actually spent
                n_ar tells you how many repetitions it took to get to t_ar
                
                """
                
                n_ar , t_ar = clock.autorange()
            
                
                t = clock.repeat(repeat=self.repetiotions, number=n_ar)
                
                t_min = np.min(t)/n_ar
                t_max = np.max(t)/n_ar
                t_avg = np.mean(t)/n_ar
                t_var = variance(t)/n_ar
                t_var_pst = t_var/t_avg*100
                
                
                result["n_ar"].append(n_ar)
                result["t_ar"].append(t_ar)
                result["t_min"].append(t_min)
                result["t_avg"].append(t_avg)
                result["t_max"].append(t_max)
                result["t_var"].append(t_var)
                result["t_var_pst"].append(t_var_pst)
            except Exception as e:
                print("{} failed at iteration: {} \n Exception: {}".format(algorithm, l, e))
        result["length_sorted"]=self.iterations[:len(result["t_min"])]
        return pd.DataFrame.from_dict(data=result)
    
    def benchmark_algorithm(self, algorithm="python_sort", paradigms=["sorted", "random","reversed"], plot_result=True):
        """[Choose an algorithm and testcases to run benchmarks on. Save the results in pickled form in save/data]

        Args:
            algorithm (str, optional): The algorithm to run. Defaults to "python_sort".
            paradigms (list, optional): The testcases from self.testset_lib to test. Defaults to ["sorted", "random","reversed"].
            plot_result (bool, optional): Wether or not to plot the results. Defaults to True.

        Returns:
            dict: A nested dictionary with the first level giving run-specifications such as algorithm and seed.
            benchmar["Results"] is a dict containing the benchmarks generated from the different testcase paradigms
        """
        benchmark = {
            "seed":self.seed,
            "algorithm":algorithm,
            "results":{},
            "plot": None
            }
        for paradigm in paradigms:
            benchmark["results"][paradigm] = self.run_benchmark(algorithm=algorithm, paradigm=paradigm)
    
        if self.save_results:
            path = self.makecheck_save_location(hierarchy=["data"])
            path = os.path.join(path, "{}".format(benchmark["algorithm"]))
            with open(path, "wb") as f:
                pickle.dump(benchmark, f)
            if plot_result:
                self.plot_benchmark(path)
        return benchmark

        
    def benchmark_gauntlet(self):
        print("Running a test gauntlet on: {} algorithms, using {} elements as iteration lengths, repeated {} times".format(len(self.algorithm_lib), self.iterations, self.repetiotions))
        """
        Run testcases on all algorithms in self.algorithm_lib on all testcases in self.testcase_libdatetime.

        Returns:
            [dict]: All results are saved, but returns a dict with the results for convenience
        """
        lib = {}
        for i, algorithm in enumerate(self.algorithm_lib):
            try:
                lib[algorithm]=self.benchmark_algorithm(algorithm=algorithm)
                print("{}: {}) Finished".format(i, algorithm))
            except Exception as e:
                print("{}: {}) Exception raised: {}".format(i, algorithm, e))
        self.save_experiment_specs()
        return lib
    
    def plot_benchmark_variance(self, path):
        """
        Takes the path to a saved dataset and plots the average runtime accompanied by
        a shadowplot showing the variance of the repetitions(dark) and the max and min runtime(lighter color).
        If there are more than one testcase paradigm the process is repeated for all and the results are saved.

        Args:
            path (str): A path directing to a pickled dataset
        """
        df_dict = pd.read_pickle(path)

        algorithm_color = self.algorithm_lib[df_dict["algorithm"]]["color"] 
        for paradigm in df_dict["results"].keys():
            fig, ax = plt.subplots(1,1,figsize=(84/25.4, 55/25.4))
        
            ax.set_xlabel('n')
            ax.set_ylabel('seconds')
            
            lengths = df_dict["results"][paradigm]["length_sorted"]
          
            #avg
            ax.plot(
                lengths,
                df_dict["results"][paradigm]["t_avg"],
                'o-', color=algorithm_color,ms=5)
            
            #possibility range
            ax.fill_between(
                lengths,
                df_dict["results"][paradigm]["t_min"],
                df_dict["results"][paradigm]["t_max"],
                color=algorithm_color,
                alpha=0.2,
                )
            
            #Plausibility range
            ax.fill_between(
                lengths,
                df_dict["results"][paradigm]["t_avg"]+df_dict["results"][paradigm]["t_var"],
                df_dict["results"][paradigm]["t_avg"]-df_dict["results"][paradigm]["t_var"],
                color=algorithm_color,
                alpha=0.8,
                )
                        
            if not self.scientific:
                ax.set_title("{}_{}".format(df_dict['algorithm'], paradigm))
            
            if self.default_plot:
                self.default_plot_settings()
            self.save_plot(df_dict=df_dict, fig=fig, plotname="{}_variance_plot".format(paradigm))

    def plot_benchmark(self, path, result_header="t_min"):
        """
        Args:
            path (str): Input a path to a dataset to plot. 
            result_header (str, optional): Input which header to plot in the dataset. Defaults to "t_min".
        """
        df_dict = pd.read_pickle(path)        
        fig, ax = plt.subplots(1,1,figsize=(84/25.4, 55/25.4))
        
        ax.set_xlabel('n')
        ax.set_ylabel('seconds')
        for paradigm in df_dict["results"].keys():
            #min
            ax.plot(
                df_dict["results"][paradigm]["length_sorted"],
                df_dict["results"][paradigm][result_header],
                'o-', color=self.testset_lib[paradigm]["color"], ms=5)
        if not self.scientific:
            ax.set_title("{} {}".format(df_dict['algorithm'], result_header))
        if self.default_plot:
            self.default_plot_settings()
        self.save_plot(df_dict,fig,"benchmark_plot_{}".format(result_header))
    
    def plot_gauntlet_paradigm_comparison(self, algorithm_list=["merge_sort", "bubble_sort", "python_sort"], result_header="t_min"):
        """[summary]

        Args:
            algorithm_list (list, optional): A list of strings with sorting algorithms as named in self.algorithm_lib. Limit the algorithms to plot. Defaults to ["merge_sort", "bubble_sort", "python_sort"].
            result_header (str, optional): Which store datapoint to use. Defaults to "t_min".
        """
        
        directory = os.path.join(self.save_location, str(self.seed), "data")
        comparison = defaultdict(dict)
        for dumps in os.scandir(directory):
            data = pd.read_pickle(dumps)
            if data["algorithm"] in algorithm_list:
                for paradigm in data["results"]:
                    comparison[paradigm][data["algorithm"]] = data["results"][paradigm][result_header]
            else:
                pass
            
        scale = self.iterations
        for para in comparison:
            fig, ax = plt.subplots(1,1,figsize=(84/25.4, 55/25.4))
            ax.set_xlabel('n')
            ax.set_ylabel('seconds')
            for algorithm in comparison[para]:
                n=len(comparison[para][algorithm])
                ax.plot(scale[:n],comparison[para][algorithm][:n],
                        'o-', color=self.algorithm_lib[algorithm]["color"], ms=5)
                
            if not self.scientific:
                ax.set_title(para)
            path = self.makecheck_save_location(hierarchy=["comparison" ])
            path = os.path.join(path, "{}_{}.pdf".format(para, result_header))
            
            if self.default_plot:
                self.default_plot_settings()
            
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
    
    def algorithm_ranking(self, result_header="t_min", n=None):
        """ A helpful tool to give an absolute ranking at each datapoint

        """
        directory = os.path.join(self.save_location, str(self.seed), "data")
        comparison = defaultdict(dict)
        for dumps in os.scandir(directory):
            data = pd.read_pickle(dumps)
            for paradigm in data["results"]:
                comparison[paradigm][data["algorithm"]] = data["results"][paradigm][result_header]
        ranking = {}
        for paradigm in self.testset_lib:
            ranking[paradigm] = pd.DataFrame(columns=self.iterations[:10])
            for i in range(10):
                ranking[paradigm][self.iterations[i]] = sorted([(algo, comparison[paradigm][algo].iloc[i]) for algo in comparison[paradigm]], key=(lambda x: x[1]))
        
        path = self.makecheck_save_location(hierarchy=["comparison", "ranking"])
        for df in ranking:
            ranking[df].to_csv(os.path.join(path, "{}.csv".format(df)))
        return ranking
            
        
        
    def makecheck_save_location(self, hierarchy=[], seed=None,):
        """
        Args:
            hierarchy (iterable): A list of strings, containing the folder hierarchy of the new save location. By default it adds self.save_location and seed
            seed ([int,str,float], optional): Seed folder used in the folder hierarchy. If None, the self.seed value is used. Defaults to None.

        Returns:
            [type]: [description]
        """
        if not seed:
            seed=self.seed
        seed = str(seed)
        path = os.path.join(self.save_location, seed)
        if not os.path.exists(path):
            os.makedirs(path)
        
        for foldername in hierarchy:
            path = os.path.join(path, foldername)
            if not os.path.exists(path):
                os.makedirs(path)
        
        return path
            
        
    def save_plot(self,df_dict, fig, plotname):
        """
        Args:
            df_dict (dictionary): The dictionary the plot was created from.
            fig (matplotlib.pyplot.Figure): A figure object to save
            plotname (string): A descriptive name for the figure

        Returns:
            [str]: A path to where the plot has been stored
        """
        #Save under directory of the seedvalue
        path = self.makecheck_save_location(hierarchy=["figures", "{}".format(df_dict["algorithm"])])
        #Save under the filename given by the plot function
        path = os.path.join(path, "{}.pdf".format(plotname))
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return path
    
    def save_experiment_specs(self):
        """Collects self.system_specs and hardware specs and write them to a latex file in the save folder"""
        
        filename = os.path.join("save", str(self.seed), "system_specs.tex")
        data = pd.DataFrame.from_dict(data=self.system_specs).iloc[1].transpose()
        data = data.to_latex(header=False)
        with open(filename, "w") as f:
            f.write(data)
            f.close()
        filename = os.path.join("save", str(self.seed), "hardware_specs.tex")
        data = pd.DataFrame.from_dict(data=self.hardware_specs,orient="index")
        data = data.to_latex(header=False)
        with open(filename, "w") as f:
            f.write(data)
            f.close()

    

         
    
def _find_algorithm_threshold(short_range_algorithm_result_path, long_range_algorithm_result_path):
    """
    Useful to find the optimal threshold for swithcing between several algorithms
    Args:
        short_range_algorithm_result_path (str)
        long_range_algorithm_result_path (str)

    Returns:
        str: The iteration where the long range algorithm becomes faster than the short range.  
    """
    a = pd.read_pickle(short_range_algorithm_result_path)
    b = pd.read_pickle(long_range_algorithm_result_path)
    for i, time in enumerate(a["results"]["random"]["t_min"]):
        if time>=b["results"]["random"]["t_min"].iloc[i]:
            print(time, "  +  ", b["results"]["random"]["t_min"].iloc[i])
        else:
            return b["results"]["random"]["length_sorted"].iloc[i]

def export_algorithm_results_to_latex(path, columns=["length_sorted", "t_min", "t_var"]):
    """
    Args:
        path (str): Path to a dataset
        columns (list, optional): Limit the data you want to save. Defaults to ["length_sorted", "t_min", "t_var"].
    """
    data = pd.read_pickle(path)
    seed = str(data["seed"])
    dirname=os.path.join("save", seed, "tables")
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for paradigm in data["results"]:
        if columns:    
            df = data["results"][paradigm][columns].to_latex()
        else:
            df = data["results"][paradigm][columns].to_latex()
        filename = os.path.join("save", seed,"tables","{}_{}.tex".format(data["algorithm"], paradigm))
        with open(filename, "w") as f:
            f.write(df)
            f.close()
            
def _visual_plot_test(iterable, title):
    fig, ax = plt.subplots(1,1)
    ax.plot(iterable)
    ax.set_title(title)
        
        
if __name__ == "__main__":
    insertion = "C:/Users/Tormo/OneDrive/Skrivebord/schule/INF/TermPaper_2021/git/code/save/222/data/insertion_sort"
    bubble = "C:/Users/Tormo/OneDrive/Skrivebord/schule/INF/TermPaper_2021/git/code/save/222/data/bubble_sort"
    bubble = "C:/Users/Tormo/OneDrive/Skrivebord/schule/INF/TermPaper_2021/git/code/save/222/data/bubble_sort"
    
    #quick2insertion = find_algorithm_threshold(testset_location_quick, testset_location_insertion)
    #merge2insertion = find_algorithm_threshold(testset_location_merge, testset_location_insertion)

    a = AlgorithmSpeedTest(seed=222)
    a.scientific = True
    q = a.benchmark_gauntlet()
    
    #a.scientific = False
    #algorithms = a.algorithm_lib
    gauntlet_algorithms = set(a.algorithm_lib)-{"python_sort", "threshold_merge_to_insertion"}
    
    a.plot_gauntlet_paradigm_comparison(gauntlet_algorithms, result_header="t_min")
    for path in os.scandir("C:/Users/Tormo/OneDrive/Skrivebord/schule/INF/TermPaper_2021/git/code/save/222/data"):    
       a.plot_benchmark_variance(path)
       a.plot_benchmark(path, result_header="t_min")
       export_algorithm_results_to_latex(path)
    
    #q = pd.read_pickle("C:/Users/Tormo/OneDrive/Skrivebord/schule/INF/TermPaper_2021/git/code/save/222/data/insertion_sort")
    #a.plot_gauntlet_paradigm_comparison(gauntlet_algorithms, result_header="t_min")
    q = a.algorithm_ranking()
    #a.save_experiment_specs()
    a.y_axis_range=(10**-12, 10**1)
    
    a.plot_gauntlet_paradigm_comparison(gauntlet_algorithms, result_header="t_var")
    #a.plot_gauntlet_paradigm_comparison(gauntlet_algorithms, result_header="t_var_pst")
    for path in os.scandir("C:/Users/Tormo/OneDrive/Skrivebord/schule/INF/TermPaper_2021/git/code/save/222/data"):  
       a.plot_benchmark(path, result_header="t_var")
   
    a.save_experiment_specs()
    
    a.benchmark_algorithm(algorithm="quick_sort")
    a.y_axis_range=(0, 20)
    a.default_plot=False
    #a.plot_gauntlet_paradigm_comparison(gauntlet_algorithms, result_header="t_var_pst")
    
    for path in os.scandir("C:/Users/Tormo/OneDrive/Skrivebord/schule/INF/TermPaper_2021/git/code/save/222/data"):  
        print(pd.read_pickle(path)["algorithm"])
        a.plot_benchmark(path, result_header="t_var_pst", xticks=True)
        