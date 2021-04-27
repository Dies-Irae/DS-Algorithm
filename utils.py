import numpy as np


def xorSum(indexArray,dataArray):
    """
    accept a array of index and data array(numpy)\n
    do successive xor of npArray[index]
    """
    x = np.zeros(dataArray.shape[1])
    for index in indexArray:
        x = np.logical_xor(x, dataArray[index])
    return x


def weight(x):
    """
    calculate hamming weight
    """
    return np.count_nonzero(x)


def generatorMatrix(n, k, generator_poly):
    """
    input: code length\n
    return: Systematic Generator Matrix and Check Matrix(http://www.rutvijjoshi.co.in/index_files/lecture-26.pdf)
    """
    p = np.zeros((k,n-k))
    for i in range(0,k):
        remainder = np.polydiv(np.eye(n)[i], generator_poly)[1]
        p[i][k-remainder.shape[0]-1:] = remainder
    p = np.remainder(p, 2)
    generatorMatrix = np.block([np.eye(k), p])
    checkMatrix = np.block([p.T, np.eye(n-k)])
    return generatorMatrix, checkMatrix


def errorTable(n, H):
    """
    input: n(code length), H(check matrix)\n
    return: emTable(error table), smTable(syndrome table)
    """
    x, n = H.shape # H.shape is (n-k,n)
    k = n-x
    emTable = np.concatenate((np.identity(k), np.zeros((k,x)) ), axis=1)
    smTable = np.mod(np.matmul(emTable, H.T), 2) #i.e the syndrome table T
    return emTable, smTable


