# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:59:03 2019

@author: Deepika
"""

import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import os
from vectorize_reviews import VECTORIZE_TEXT

class RECOMMENDER(object):
  
  def __init__(self, user_input, city='London', datasets_dir='datasets'):
    self.datasets_dir = datasets_dir
    self.places_file = 'tourpedia_{}_poi.csv'.format(city)
    self.reviews_file = 'tourpedia_{}_reviews.csv'.format(city)
    self.places_df = pd.read_csv(os.path.join(datasets_dir, self.places_file))
    self.reviews_df = pd.read_csv(os.path.join(datasets_dir, self.reviews_file))
  
    self.user_input = user_input
#    self.clean_user_input
    self.user_input_vec_norm = []
    self.user_input_vec_avg = []
    
  def preprocess_user_input(self):
    self.clean_user_input = self.user_input.lower()
    # Tokenization
    # Remove punctuation, special characters, numbers. Keep only alphabets. Convert sentence into tokens
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    self.clean_user_input = tokenizer.tokenize(self.clean_user_input)
    
    # Remove Stopwords
    # Load stop words
    stop_words = stopwords.words('english')
    stop_words.append('u')
    self.clean_user_input = [word for word in self.clean_user_input if word not in stop_words]
  
  def vectorize_user_input(self):
    vectorize_text = VECTORIZE_TEXT()
    self.user_input_vec_norm = vectorize_text.get_sent_vec_norm(self.clean_user_input)
    self.user_input_vec_avg = vectorize_text.get_sent_vec_avg(self.clean_user_input)
    
  def calculate_cos_sim(self):
#    self.reviews_df['cosine_similarity_normed'] = self.reviews_df['normed_wordvec'].apply(lambda sent_vec: cosine_similarity(self.user_input_vec_norm, sent_vec)[0][0])
#    self.reviews_df['cosine_similarity_avg'] = self.reviews_df['avg_wordvec'].apply(lambda sent_vec: cosine_similarity(self.user_input_vec_avg, sent_vec)[0][0])
    self.reviews_df['cosine_similarity_normed'] = self.reviews_df['normed_wordvec'].apply(
        lambda sent_vec: 
          cosine_similarity(self.user_input_vec_avg, np.fromstring(sent_vec.strip("[]"), dtype=float, sep=' ')[np.newaxis])[0][0])

    self.reviews_df['cosine_similarity_avg'] = self.reviews_df['avg_wordvec'].apply(
        lambda sent_vec: 
          cosine_similarity(self.user_input_vec_avg, np.fromstring(sent_vec.strip("[]"), dtype=float, sep=' ')[np.newaxis])[0][0])

  def make_recommendations(self):
    self.preprocess_user_input()
    self.vectorize_user_input()
    self.calculate_cos_sim()
    
    self.df_normed_sorted = self.reviews_df.sort_values(by=['cosine_similarity_normed'], ascending=False)
    self.df_avg_sorted = self.reviews_df.sort_values(by=['cosine_similarity_avg'], ascending=False)
    
    print('Recommendations based on Normalized word2vec (first 5)')
    print(self.df_normed_sorted[['text','cosine_similarity_normed']].head(5))
    
    print('Recommendations based on Average word2vec (first 5)')
    print(self.df_avg_sorted[['text','cosine_similarity_avg']].head(5))

if __name__ == "__main__":
  user_input = 'super food'
  recommender = RECOMMENDER(user_input=user_input, city='London')
  recommender.make_recommendations()