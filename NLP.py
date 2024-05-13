# We can use the same process to creat document term matrix by considering the importance of the words
# Here we will use TfidfVectorizer() from scikit-learn to convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Import the necessary libraries

from google.cloud import storage
from google.cloud import bigquery

# First make connection to BigQuery
# Replace 'your-project-id' with your Google Cloud project ID
# Replace 'your-dataset-name' with your dataset name
project_id = "g4-bdao-ima"
dataset_name = "restaurant_dataset"
table_name = "restaurant_tips" # if you make changes previously, then here you need to make according change

bigquery_client = bigquery.Client(project=project_id)

df = pd.read_csv("E:\\files\study\warwick\BDAO\IMA\df.csv") #load the dataset
df.head()


# set vectorizer - CountVectorizer for word counts
vectorizer = CountVectorizer(stop_words = (["app"]), max_features=10)

# create an array of word counts
doc_vec = vectorizer.fit_transform(df.filtered_text.values.astype('U'))

print(doc_vec.shape)

# convert this to a dataframe
df2 = pd.DataFrame(doc_vec.toarray(), columns=vectorizer.get_feature_names_out())

# set a threshold to drop infrequent words
threshold = 0.1

# drop words based on the threshold
df2 = df2.drop(df2.mean()[df2.mean() < threshold].index.values, axis=1) # Here find out the word with average word count lower than 0.1 and drop them

# join the two datasets together
dtm = df.join(df2, how='left',lsuffix='_left', rsuffix='_right')

df2.mean()

# write the dataframe out to csv and download

import os
os.makedirs('E:\\files\\test', exist_ok=True)
dtm.to_csv('E:\\files\\test\out.csv',index=False)