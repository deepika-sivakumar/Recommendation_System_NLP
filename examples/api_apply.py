# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:21:15 2019

@author: Deepika
"""
import pandas as pd

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
      # Check if the dataframe is not empty before selecting columns
      if(df.empty == False):
          df = df[['language', 'text', 'time']]
#          print('Review Details Df keys:',df.keys())
    return success, df


