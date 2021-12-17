#!/usr/bin/env python
# coding: utf-8

# ## Insertion sort algorithm 


def insertion_sort(A):
    for j in range(1, len(A)):
        key = A[j]
        i = j - 1
        while i >= 0 and A[i] > key:
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key
    return(A)


if __name__ == "__main__":
    test1 = [1,2,3,4,5,6,7,8]
    test2 = [2,3,4,2,1,4,56,9]
    test3 = test1[::-1]
    a = insertion_sort(test1.copy())
    b = insertion_sort(test2.copy())
    c = insertion_sort(test3.copy())



