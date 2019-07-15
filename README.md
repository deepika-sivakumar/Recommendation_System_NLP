# A point of interest personalized recommendation system using Natural Language Processing (NLP)

## Project Description:
The user enters a few words to describe their emotions/desires.
<br/>The model analyses other people reviews using and recommends the places that best align with the user's interests.

## Dataset:
List of **Places** and the **User Reviews** for various cities from the Tourpedia dataset accessible via Web API.
<br/>**Tourpedia** contains information about points of interest and attractions of different places in Europe (Amsterdam, Barcelona, Berlin, Dubai, London, Paris, Rome and Tuscany). Data are extracted from four social media: Facebook, Foursquare, Google Places and Booking. 
<br/> http://tour-pedia.org/about/index.html

## Steps:

### Construct Datasets:

* Load the "Places" data as a JSON response making an API request.
* Convert the JSON into pandas dataframe and store it as a CSV file.
* For each place, load the "User Reviews" data making API requests and store it as a CSV file.

### Preprocess Data:

* Select only the relevant columns.
* Remove non-english review rows from the Reviews dataset.
* Remove punctuation, stopwords

### 


