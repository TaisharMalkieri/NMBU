#!/usr/bin/env python
# coding: utf-8

import math

# Merge function
def merge(A, p, q, r):

    L = []
    R = []

    n_1 = (q - p + 1)
    n_2 = (r - q)

    for i in range(n_1):
        L.insert(i, A[p + i - 1])

    for j in range(n_2):
        R.insert(j, A[q + j])

    L.append(math.inf)
    R.append(math.inf)

    i = 0
    j = 0

    for k in range(p - 1, r):
        if L[i] <=  R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1

#Mergesort function
def merge_sort(A, p, r):
    if p < r:
        q = (r + p)//2
        merge_sort(A, p, q)
        merge_sort(A, q + 1, r)
        merge(A, p, q, r)
    return A

if __name__ == "__main__":

    test1 = [1, 2, 3, 4, 5, 6, 7, 8]
    test2 = [2, 3, 4, 2, 1, 4, 56, 9]
    test3 = test1[::-1]

    a = merge_sort(test1.copy().copy(), 1, len(test1))
    print(a)
    b = merge_sort(test2.copy().copy(), 1, len(test2))
    print(b)
    c = merge_sort(test3.copy().copy(), 1, len(test3))
    print(c)
