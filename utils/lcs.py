import numpy as np

def lcs(X, Y):
    m, n = len(X), len(Y)
    L = np.zeros((m + 1, n + 1))

    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i - 1] == Y[j - 1] or (Y[j-1] == "-" and L[i - 1, j - 1]>0):
                L[i, j] = L[i - 1, j - 1] + 1
            else:
                L[i, j] = 0

    i, j = np.unravel_index(L.argmax(), L.shape)
    posX, posY = [], []
    while i > 0 and j > 0:
        if L[i, j] == 0:
            break
        if X[i - 1] == Y[j - 1] or Y[j-1] == "-":
            i -= 1
            j -= 1
            posX.append(i)
            posY.append(j)
            continue
    posX.sort(), posY.sort()
    return len(posX), (min(posX), max(posX), min(posY), max(posY)), L

INDEL, MISS = 3, 10

def lcs_pattern(X, Y):
    m, n = len(X), len(Y)
    L = np.zeros((m + 1, n + 1))

    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i, j] = max(1, L[i - 1, j - 1] + 1)
            else:
                L[i, j] = max(0, L[i - 1, j] - INDEL, L[i, j - 1] - INDEL, L[i - 1, j - 1] - MISS)

    # Following code is used to print LCS

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i, j = np.unravel_index(L.argmax(), L.shape)
    posX, posY = [], []
    while i > 0 and j > 0:
        if L[i, j] == 0:
            break
        if X[i - 1] == Y[j - 1]:
            i -= 1;
            j -= 1
            posX.append(i);
            posY.append(j)
            continue
        insert, delete, miss = L[i - 1, j] - INDEL, L[i, j - 1] - INDEL, L[i - 1, j - 1] - MISS
        if insert > delete and insert > miss:
            i -= 1
        elif delete > miss:
            j -= 1
        else:
            i -= 1;
            j -= 1
    posX.sort(), posY.sort()
    return len(posX), (min(posX), max(posX), min(posY), max(posY)), L
