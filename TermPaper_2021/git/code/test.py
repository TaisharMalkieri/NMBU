import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import pickle
import copy
from math import log, factorial
from statistics import variance, mean
from collections import defaultdict
from time import sleep
from collections import Counter

from main import AlgorithmSpeedTest
from algorithms.bubble_sort import bubble_sort
from algorithms.insertion_sort import insertion_sort
from algorithms.quick_sort import quick_sort
from algorithms.merge_sort import merge_sort
from algorithms.threshold_sorts import threshold_sort_quick_to_insertion, threshold_sort_merge_to_insertion

class Test:
    def __init__(self):
        self.seed=100
        self.main_class = AlgorithmSpeedTest(seed=self.seed)
        self.testset_lib = self.main_class.testset_lib
        self.algorithm_lib = self.main_class.algorithm_lib
    

        
    def test_algorithm(self, algorithm):
        result_lib = defaultdict(dict)
        algo = self.algorithm_lib[algorithm]["algorithm"]
        
        for paradigm in self.testset_lib:
            dataset = np.array(self.testset_lib[paradigm]["data"][0:1000].copy().copy())    
            solution = np.sort(dataset.copy().copy())
            if algorithm in ["merge_sort","quick_sort", "threshold_quick_to_insertion", "threshold_merge_to_insertion"]:
                if algorithm in ["quick_sort", "threshold_quick_to_insertion"]:
                     result = algo(dataset, 0, len(dataset)-1)
                else:
                     result = algo(dataset, 1, len(dataset))
                is_sorted = all(result==solution)
                if not is_sorted:
                    result_lib[paradigm] = result
                    
                else:
                    result_lib[paradigm] = all(result==solution)
            else:
                result = algo(dataset)
                is_sorted = all(result==solution)
                if not is_sorted:
                    result_lib[paradigm] = result
                    
                else:
                    result_lib[paradigm] = all(result==solution)
        return result_lib
                
    def test_algorithm_gauntlet(self):
        result_lib = {}
        for algorithm in self.algorithm_lib:
            result_lib[algorithm] = self.test_algorithm(algorithm)
        return result_lib
    
    def evaluate_testsets(self):
        main = self.main_class
        lib = main.testset_lib
        sorted_list = lib["sorted"]["data"][0](1000)
        reverse = lib["reversed"]["data"][0](1000)
        rand = lib["random"]["data"][0](1000)
        number_distribution = Counter(rand).items()
        print(number_distribution)
        
        fig, ax = plt.subplots(4,1)
        ax[0].plot(rand)
        ax[1].plot(sorted_list)
        ax[2].plot(reverse)
        ax[2].bar(number_distribution)
        
        
        
    
    
    def visual_test(self):
        main = self.main_class
        iterations = main.iterations
        base = 10**-5
        cols = ["quadratic", "exponential", "fact", "logarithmic", "linear", "static"]
        quadratic = [base*(x**2) for x in iterations]
        exponential = [base*(2**x) for x in iterations[:9]]
        fact = [factorial(x)*base for x in iterations[:6]]
        logarithmic = [(x*log(x))*base for x in iterations]
        static = [base for x in iterations]
        linear = [x*base for x in iterations]
        
        df = pd.DataFrame([quadratic, exponential, fact, logarithmic, linear, static]
                          ).transpose()
        df.columns = cols
        df.index = iterations
        fig, ax = plt.subplots(1, 1, figsize=(84/25.4, 55/25.4))
        ax.plot(df)
        main.default_plot_settings()
        fig.savefig("save/test/scale_distorition.pdf") 
        ax.legend(df.columns)
        
        fig.savefig("save/test/test.pdf")        

if __name__ == "__main__":
    tst = Test()
    a, b, c = tst.evaluate_testsets()
