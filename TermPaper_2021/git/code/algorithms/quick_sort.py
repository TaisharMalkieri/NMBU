#!/usr/bin/env python
# coding: utf-8

# ## Quick sort algorithm 

def quick_sort(A,p,r):
    if p < r:
        q = partition(A,p,r)
        quick_sort(A,p,q-1)
        quick_sort(A,q+1,r)
    return A


def partition(A,p,r):
    x = A[r]
    i = p-1
    for j in range(p, r):
        if A[j] <= x:
            i = i + 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[r] = A[r], A[i+1]
    return i+1
        
if __name__ == "__main__":
    test1 = [1,2,3,4,5,6,7,8]
    test2 = [2,3,4,2,1,4,56,9]
    test3 = test1[::-1]
    a = quick_sort(test1.copy(), 0, len(test1)-1)
    b = quick_sort(test2.copy(), 0, len(test2)-1)
    c = quick_sort(test3.copy(), 0, len(test3)-1)
