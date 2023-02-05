import pandas as pd
import numpy as np

df = pd.read_csv('risk_factors_cervical_cancer.csv')
df.head() #show the first 5 rows of the data frame

# replace all "?" with NaN
df = df.replace('?', np.nan)  

# list all "object" columns
df.select_dtypes(include='object').columns

# change "object" values to numeric values 
for col in df.select_dtypes(include='object').columns:
  df[col] = pd.to_numeric(df[col],errors = 'coerce')

# check if there is any missing values
df.isnull().sum()

""" since there're so many missing values in STDs: Time since first diagnosis 
and STDs: Time since last diagnosis, I'm gonna remove these 2 columns 
"""
df = df.drop(columns=['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])

# I choose Schiller method to evaluate the effeciency of the model, so I remove 3 other methods
df = df.drop(columns=['Hinselmann','Citology','Biopsy'])

# now I fill all missing value (NaN) with mean of the same column
df = df.fillna(df.mean())

df.info()

#save the new data frame
df.to_csv('/content/drive/MyDrive/ML Projects/cervical_cancer/new_risk_factor_cervical_cancer.csv')
