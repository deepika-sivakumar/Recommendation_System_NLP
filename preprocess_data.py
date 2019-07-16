# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:12:01 2019

@author: Deepika
"""
import os
import pandas as pd

class PREPROCESS_DATA(object):
  
  #Constructor 
  def __init__(self, city, datsets_dir='datasets'):
    self.places_file = 'tourpedia_{}_poi.csv'.format(city)
    self.reviews_file = 'tourpedia_{}_reviews.csv'.format(city)
    self.places_df = pd.read_csv(os.path.join(datsets_dir, self.places_file))
    self.reviews_df = pd.read_csv(os.path.join(datsets_dir, self.reviews_file))
  
  """
    Function to clean the data
  """
  def clean_data(self):
    pass
  
    
  
  