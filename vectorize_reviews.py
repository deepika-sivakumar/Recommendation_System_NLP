# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:32:57 2019

@author: Deepika
"""

import gensim
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

class VECTORIZE_TEXT(object):
  
  def __init__(self):
    self.model = gensim.models.KeyedVectors.load('google_w2vec_trim.model')
  
  def save_goog_model(self):
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('../google_word2vec_model/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)  
    
    model.init_sims(replace=True)
    model.save('google_w2vec_trim.model')
  
  def get_sent_vec(self, sent_tokens):
    arr = np.empty((0,300), float)
    for word in sent_tokens:
      wv = np.array(self.model.wv[word])[np.newaxis]
      arr = np.append(arr, wv, axis=0)
      
    sum_vec = np.sum(arr, axis=0, keepdims=True)
    normed_vec = normalize(sum_vec)
    return normed_vec
  
  def get_sent_vec_avg(self, sent_tokens):
    arr = np.empty((0,300), float)
    for word in sent_tokens:
      wv = np.array(self.model.wv[word])[np.newaxis]
      arr = np.append(arr, wv, axis=0)
      
    avg_vec = np.mean(arr, axis=0, keepdims=True)
    return avg_vec
  
  def convert_sentence2vec(self):
    self.df = pd.DataFrame(columns=['sentence','wordvec'])
    self.df = self.df.append({'sentence': 'I love swimming'}, ignore_index=True)
    self.df = self.df.append({'sentence': 'Come here monkey banana'}, ignore_index=True)

    self.df['sentence'] = self.df['sentence'].apply(lambda sent: word_tokenize(sent))
    self.df['normed_wordvec'] = self.df['sentence'].apply(self.get_sent_vec)
    self.df['avg_wordvec'] = self.df['sentence'].apply(self.get_sent_vec_avg)
    

  def calculate_cos_sim(self, user_input):
    input_vec = self.get_sent_vec(word_tokenize(user_input))
    self.df['cosine_similarity_normed'] = self.df['normed_wordvec'].apply(lambda sent_vec: cosine_similarity(input_vec, sent_vec)[0][0])
    self.df['cosine_similarity_avg'] = self.df['avg_wordvec'].apply(lambda sent_vec: cosine_similarity(input_vec, sent_vec)[0][0])
  
  def recommendations(self):
    df_normed_sorted = self.df.sort_values(by=['cosine_similarity_normed'], ascending=False)
    df_avg_sorted = self.df.sort_values(by=['cosine_similarity_avg'], ascending=False)
    
  def get_cos_similarity_eg(self):
    banana = np.array(self.model.wv['banana'])[np.newaxis]
    fruit = np.array(self.model.wv['fruit'])[np.newaxis]
    gun = np.array(self.model.wv['gun'])[np.newaxis]
    bf = cosine_similarity(banana, fruit)
    bg = cosine_similarity(banana, gun)
    
if __name__ == "__main__":
  sentence_vectors = VECTORIZE_TEXT()
  sentence_vectors.convert_sentence2vec()
#  sentence_vectors.get_cos_sim()
#  sentence_vectors.get_cos_similarity_eg()
  user_input = 'I like sports'#'Run animal fruit'
  sentence_vectors.calculate_cos_sim(user_input)
  sentence_vectors.recommendations()
    
#    df['wordvec'] = df['sentence'].apply(get_word_vec)
#    print('After wordvec')
#    print(df)

"""
model = gensim.models.KeyedVectors.load('google_w2vec_trim.model')
words = 'I like you'.split()
print('words:', words)
sent1 = (model.wv[words[0]] + model.wv[words[1]] + model.wv[words[2]] ) / 3
print('sent1:',sent1)

words2 = 'I love you'.split()
print('words2:', words2)
sent2 = (model.wv[words2[0]] + model.wv[words2[1]] + model.wv[words2[2]] ) / 3
print('sent1:',sent1)

words3 = 'Come here monkey'.split()
print('words3:', words3)
sent3 = (model.wv[words3[0]] + model.wv[words3[1]] + model.wv[words3[2]] ) / 3
print('sent3:',sent3)
"""

