#!/usr/bin/env python
# coding: utf-8

# ## Bubble sort algorithm

def bubblesort(A): # From Lecture 4_1
    for j in range(len(A)):
        for i in range(0, len(A)-1):
            if A[i] > A[i+1]:
                temp = A[i]
                A[i] = A[i+1]
                A[i+1] = temp
    return A


def bubble_sort(A): # Based on pseudocode from CRLS.
    for i in range(len(A)-1):
        for j in range(len(A)-1, i, -1):
            if A[j] < A[j-1]:
                A[j], A[j-1] = A[j-1], A[j]
    return A

if __name__ == "__main__":
    test1 = [1,2,3,4,5,6,7,8]
    test2 = [2,3,4,2,1,4,56,9]
    test3 = test1[::-1]
    a = bubble_sort(test1.copy())
    b = bubble_sort(test2.copy())
    c = bubble_sort(test3.copy())



