# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:42:30 2021

@author: Tormo
"""
from algorithms.merge_sort import merge_sort
from algorithms.quick_sort import quick_sort
from algorithms.bubble_sort import bubble_sort
from algorithms.insertion_sort import insertion_sort



def threshold_sort_merge_to_insertion(A, p, r):
    sort_length = r-p
    threshold = 128
    if sort_length>=threshold:
        A = merge_sort(A, p, r)
    else:
        A = insertion_sort(A)
    return A

def threshold_sort_quick_to_insertion(A, p, r):
    sort_length = r-p
    threshold = 16
    if sort_length>=threshold:
        A = quick_sort(A, p, r)
    else:
        A = insertion_sort(A)
    return A

        
        

if __name__ == "__main__":
    test1 = [1,2,3,4,5,6,7,8]
    test2 = [2,3,4,2,1,4,56,9]
    test3 = test1[::-1]
