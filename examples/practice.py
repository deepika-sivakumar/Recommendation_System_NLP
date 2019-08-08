# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:36:27 2019

@author: Deepika
"""

import numpy as np
import gensim
import pandas as pd
import os
import ast
from sklearn.metrics.pairwise import cosine_similarity
"""
#Calculate the sum over each column of a numpy array
#    [[2,3]
#     [1,1]]
#sum=[[3,4]] 

a = np.array([[4,4],[5,6]])
col = np.sum(a, axis=0, keepdims=True)
d = np.sum(a, axis=1)
mean = np.mean(a, axis=0, keepdims=True)

print('sum column:',col)
print('sum row:',d)
print('mean col:',mean)


#Scikit normalize function

from sklearn.preprocessing import normalize
na = np.array([[1000,8],[100,4],[10,2]])
norm = normalize(na, axis=0)
print('norm:')
print(norm)


#Scikit cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
cos = cosine_similarity([[0, 0, 0]], [[1,1, 1]])
print('cos:',cos)


#Storing & accessing list from csv

import os
import pandas as pd
reviews_df = pd.read_csv(os.path.join('datasets', 'tourpedia_London_reviews.csv'))
cr = reviews_df.loc[0,'clean_reviews']

print('done')


#word not found in word2vec model

import gensim
from nltk.tokenize import word_tokenize

model = gensim.models.KeyedVectors.load('google_w2vec_trim.model')
sent_tokens = word_tokenize('soane bha')

arr = np.empty((0,300), float)
for word in sent_tokens:
  try:
    wv = np.array(model.wv[word])[np.newaxis]
    arr = np.append(arr, wv, axis=0)
  except KeyError:
    # If word not found, skip adding that vector
    continue
#    wv = np.zeros((1, 300))
  
sum_vec = np.sum(arr, axis=0, keepdims=True)
normed_vec = normalize(sum_vec)
if arr.size == 0:
  avg_vec = np.zeros((1, 300), float)
else:
  avg_vec = np.mean(arr, axis=0, keepdims=True)
print('happy')
"""

#Storing word vectors in csv
model = gensim.models.KeyedVectors.load('google_w2vec_trim.model')
def get_sent_vec_avg(sent_tokens):
    arr = np.empty((0,300), float)
    for word in sent_tokens:
      try:
        wv = np.array(model.wv[word])[np.newaxis]
        arr = np.append(arr, wv, axis=0)
      except KeyError:
        # If word not found, skip adding that vector
        continue
    
    if arr.size == 0:
      avg_vec = np.zeros((1, 300), float)
    else:
      avg_vec = np.mean(arr, axis=0, keepdims=True)
    return avg_vec
  
df1 = pd.read_csv(os.path.join('datasets', 'tourpedia_London_reviews.csv'))
#df1 = df1.head()
df1['clean_reviews'] = df1['clean_reviews'].apply(lambda string: ast.literal_eval(string))
df1['avg_wordvec'] = df1['clean_reviews'].apply(get_sent_vec_avg)
df1.to_csv('test_wv.csv', index = None, header=True)


df2 = pd.read_csv('test_wv.csv')
user_input_vec_avg = get_sent_vec_avg(['awesome','food'])
df2['cosine_similarity_avg'] = df2['avg_wordvec'].apply(
    lambda sent_vec: 
      cosine_similarity(user_input_vec_avg, np.fromstring(sent_vec.strip("[]"), dtype=float, sep=' ')[np.newaxis])[0][0])
print('cos sim:')
print(df2[['clean_reviews','cosine_similarity_avg']])





