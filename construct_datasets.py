# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:21:15 2019

@author: Deepika
"""
import pandas as pd
import requests
import json
import os

class CONSTRUCT_DATASETS(object):
  
  #Constructor 
  def __init__(self, city, datasets_dir='datasets'):
    self.city = city
    self.datasets_dir = datasets_dir
    # Create the datasets directory if it does not exists
    if not os.path.exists(self.datasets_dir):
      os.makedirs(self.datasets_dir)
    self.places_url = 'http://tour-pedia.org/api/getPlaces?location={}&category=poi'.format(city)
    self.places_filename = 'tourpedia_{}_poi.csv'.format(city)
    self.reviews_filename = 'tourpedia_{}_reviews.csv'.format(city)
    self.places_df = pd.DataFrame()
    self.reviews_df = pd.DataFrame()
  
  """
    Function to execute an API request and return the response as a dataframe
    url - API request URL
    success - API request status
    df - the JSON response as a pandas dataframe
  """
  def get_api_response(self, url):
    # Get the API response
    response = requests.get(url)
    # Check if the API request was successful
    if(response.status_code != 200):
      success = False
      df = pd.DataFrame()
    else:
      success = True
      data = response.text
      # Load the response as json data
      json_data = json.loads(data)
      # Convert the json list to a pandas dataframe
      df = pd.DataFrame(json_data)
    return success, df
  
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

  """
    Function to construct list of places dataset for the city
  """
  def construct_places_dataset(self):
    success, self.places_df = self.get_api_response(self.places_url)
    if(success == False or self.places_df.empty == True):
      print('No Dataset for {} exists!'.format(self.place))
    else:
      # Let us clean the dataset
      # We want to select only relevant columns: id, name, originalId, details, reviews
      self.places_df = self.places_df[['id','originalId', 'name', 'details', 'reviews']]
      # Few rows have misaligned values
      # We want to select the rows that only have unique numeric value for the place id
      self.places_df = self.places_df[self.places_df['id'].astype(str).str.isdigit()]
      # Drop the rows that contain null/nan values
      self.places_df = self.places_df.dropna()
      # Save the Places dataframe to csv file
      self.save_to_csv(self.places_df, self.places_filename)
#      self.places_df.to_csv(self.places_filename, index = None, header=True)
  
  """
    Function to construct reviews dataset for all the places of the city
  """
  def construct_reviews_dataset(self):
    # Iterate through each row and get the reviews of that place
    for i in self.places_df.index:
      success, temp_reviews_df = self.get_api_response(self.places_df.get_value(i,'reviews'))
      # Only if API call is success & not empty construct reviews dataframe for that place
      if(success == True and temp_reviews_df.empty == False):
        temp_reviews_df = temp_reviews_df[['language', 'text', 'time']]
        temp_reviews_df['id'] = self.places_df.get_value(i,'id')
        temp_reviews_df['originalId'] = self.places_df.get_value(i,'originalId')
        temp_reviews_df['details'] = self.places_df.get_value(i,'details')
        # Construct a reviews dataframe of all the places
        self.reviews_df = pd.concat([self.reviews_df,temp_reviews_df])
    # Check if reviews dataframe is empty
    if(self.reviews_df.empty == True):
      print('No Reviews dataset for {}!'.format(self.place))
    else:
      # Save the Reviews dataframe to csv file
      self.save_to_csv(self.reviews_df, self.reviews_filename)
#      self.reviews_df.to_csv(self.reviews_filename, index = None, header=True)

if __name__ == "__main__":
    # Construct dataset for London
    construct_datasets = CONSTRUCT_DATASETS('London')
    construct_datasets.construct_places_dataset()
    construct_datasets.construct_reviews_dataset()
    