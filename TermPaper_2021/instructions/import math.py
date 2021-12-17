import math
q=0

def mergeSort(A, li, ri):

    if len(A)<2:
        return A
    mi = li + (li + ri)//2
    mergeSort(A[li:mi], li, mi)
    mergeSort(A[mi:ri], mi, ri)
    merge(A, li, ri, mi)
    return A

def merge(A,li,ri,mi):
    index1 = ri-li+1
    index2 = mi-ri

    L = []
    R = []
    for x in range(len(index1)):
        L = A[p+x]
    for y in range(len(index2)):
        R = A[q+y]
    
    L.append(math.inf)
    R.append(math.inf)

    i = 1
    j = 1

    for q in range(r-p):
        k=p+q
        if L[i] <= R[j]:
            A[k] = L[i]
            i = i + 1
        else:
            A[k] = R[j]
            j = j + 1
    
    return A
test = [1, 5, 7, 3, 5, 1, 8, 10, 9]
subtestL = [1, 2]
subtestR=[10, 6]


A = subtestL+subtestR
result = mergeSort(A,li=0, ri=len(A) )
A