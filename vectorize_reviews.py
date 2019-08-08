# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:32:57 2019

@author: Deepika
"""
import os
import gensim
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import ast

class VECTORIZE_TEXT(object):
  
  def __init__(self,city='London', datasets_dir='datasets'):
    self.datasets_dir = datasets_dir
    self.places_file = 'tourpedia_{}_poi.csv'.format(city)
    self.reviews_file = 'tourpedia_{}_reviews.csv'.format(city)
    self.places_df = pd.read_csv(os.path.join(datasets_dir, self.places_file))
    self.reviews_df = pd.read_csv(os.path.join(datasets_dir, self.reviews_file))
    self.model = gensim.models.KeyedVectors.load('google_w2vec_trim.model')
  
  """
    Function to save the pandas dataframe to a CSV file
    df - pandas dataframe to be stored
    file_name - CSV file name
  """
  def save_to_csv(self, df, file_name):
    # Construct the path with directory and file name
    path = os.path.join(self.datasets_dir, file_name)
    # Store the dataframe to a CSV file
    df.to_csv(path, index = None, header=True)

  def save_goog_model(self):
    """
    Function to load the Google's pretrained word2vec model and save a part of it as a custom model
    Call this function only once in the beginning
    """
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('../google_word2vec_model/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)  
    
    model.init_sims(replace=True)
    model.save('google_w2vec_trim.model')
  
  def get_sent_vec_norm(self, sent_tokens):
    arr = np.empty((0,300), float)
    
    for word in sent_tokens:
      try:
        wv = np.array(self.model.wv[word])[np.newaxis]
        arr = np.append(arr, wv, axis=0)
      except KeyError:
        # If word not found, skip adding that vector
        continue
    sum_vec = np.sum(arr, axis=0, keepdims=True)
    normed_vec = normalize(sum_vec)
    return normed_vec
  
  def get_sent_vec_avg(self, sent_tokens):
    arr = np.empty((0,300), float)
    for word in sent_tokens:
      try:
        wv = np.array(self.model.wv[word])[np.newaxis]
        arr = np.append(arr, wv, axis=0)
      except KeyError:
        # If word not found, skip adding that vector
        continue
    
    if arr.size == 0:
      avg_vec = np.zeros((1, 300), float)
    else:
      avg_vec = np.mean(arr, axis=0, keepdims=True)
    return avg_vec
  
  def vectorize_reviews(self):
    # convert string to list
    self.reviews_df['clean_reviews'] = self.reviews_df['clean_reviews'].apply(lambda string: ast.literal_eval(string))
    self.reviews_df['normed_wordvec'] = self.reviews_df['clean_reviews'].apply(self.get_sent_vec_norm)
    self.reviews_df['avg_wordvec'] = self.reviews_df['clean_reviews'].apply(self.get_sent_vec_avg)
#    print('normed_wordvec************************')
#    print('shape:',self.reviews_df['normed_wordvec'].shape)
#    print(self.reviews_df['normed_wordvec'].head())
#    print('avg_wordvec************************')
#    print('shape:',self.reviews_df['avg_wordvec'].shape)
#    print(self.reviews_df['avg_wordvec'].head())
    
    self.save_to_csv(self.reviews_df, self.reviews_file )
    
if __name__ == "__main__":
  sentence_vectors = VECTORIZE_TEXT()
  sentence_vectors.vectorize_reviews()

# self.reviews_df.loc[0,'clean_reviews']
