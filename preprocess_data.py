# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:12:01 2019

@author: Deepika
"""
import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

class PREPROCESS_TEXT_DATA(object):
  
  #Constructor 
  def __init__(self, city, datasets_dir='datasets'):
    self.datsets_dir = datasets_dir
    self.places_file = 'tourpedia_{}_poi.csv'.format(city)
    self.reviews_file = 'tourpedia_{}_reviews.csv'.format(city)
    self.places_df = pd.read_csv(os.path.join(datasets_dir, self.places_file))
    self.reviews_df = pd.read_csv(os.path.join(datasets_dir, self.reviews_file))
  
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

  def clean_data(self):
    """
    Function to clean the data
    """
    # Keep only English reviews
    self.reviews_df = self.reviews_df[self.reviews_df['language'] == 'en']
    
    # Convert to Lowercase
    self.reviews_df['clean_reviews'] = self.reviews_df['text'].apply(lambda text: text.lower())
  
    # Tokenization
    # Remove punctuation, special characters, numbers. Keep only alphabets. Convert sentence into tokens
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    self.reviews_df['clean_reviews'] = self.reviews_df['clean_reviews'].apply(lambda sent: tokenizer.tokenize(sent))

    # Remove Stopwords
    # Load stop words
    stop_words = stopwords.words('english')
    stop_words.append('u')
    self.reviews_df['clean_reviews'] = self.reviews_df['clean_reviews'].apply(lambda sent: [word for word in sent if word not in stop_words])
    
    # Store it in the CSV file
    self.save_to_csv(self.reviews_df, self.reviews_file)
  
if __name__ == "__main__":
    # Preprocess dataset for London
    preprocess_data = PREPROCESS_TEXT_DATA('London')
    preprocess_data.clean_data()
  
    
  
  