# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:36:27 2019

@author: Deepika
"""
import numpy as np

"""
Calculate the sum over each column of a numpy array
    [[2,3]
     [1,1]]
sum=[[3,4]] 
"""
a = np.array([[4,4],[5,6]])
col = np.sum(a, axis=0, keepdims=True)
d = np.sum(a, axis=1)
mean = np.mean(a, axis=0, keepdims=True)

print('sum column:',col)
print('sum row:',d)
print('mean col:',mean)

"""
Scikit normalize function
"""
from sklearn.preprocessing import normalize
na = np.array([[1000,8],[100,4],[10,2]])
norm = normalize(na, axis=0)
print('norm:')
print(norm)

"""
Scikit cosine similarity
"""
from sklearn.metrics.pairwise import cosine_similarity
cos = cosine_similarity([[0, 0, 0]], [[1,1, 1]])
print('cos:',cos)