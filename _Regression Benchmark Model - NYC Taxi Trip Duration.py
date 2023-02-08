#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# loading the dataset
data = pd.read_csv("C:\\Users\\91858\\Desktop\\Internshala Data Science Specialization\\EDA and ML\\nyc_taxi_trip_duration.csv")
data.head()


# In[4]:


data.shape


# In[5]:


data.isna().sum()


# ## Shuffling and Creating Train and Test Datasets

# In[6]:


# shuffling the dataset 
from sklearn.utils import shuffle

data = shuffle(data, random_state = 42)

div = int(data.shape[0]/4)
train = data.loc[:3*div + 1, :]
test = data.loc[3*div + 1:]


# In[7]:


train.shape, test.shape


# ## Simple Mean or avg trip_duration

# In[8]:


# storing simple mean in a new column in the test dataset
test['simple_mean'] = train['trip_duration'].mean()


# ## As our target variable (trip_duration) is a continuous variable so, the prefeered metrics in this case would be mean_absolute_error i.e., MAE.

# In[9]:


# calculating mean_absolute_error
from sklearn.metrics import mean_absolute_error as MAE

simple_mean_error = MAE(test['trip_duration'], test['simple_mean'])
simple_mean_error


# ## Mean trip_duration wrt passenger_count

# In[10]:


p_count = pd.pivot_table(train, values = 'trip_duration', index = ['passenger_count'], aggfunc = np.mean)
p_count


# In[11]:


# initializing new column to zero 
test['p_count_mean'] = 0

# for every unique entry in passenger_count
for i in train['passenger_count'].unique():
    # assign the mean value corresponding to the unique entry
    test['p_count_mean'][test['passenger_count'] == int(i)] = train['trip_duration'][train['passenger_count'] == int(i)].mean()


# In[12]:


p_count_error  = MAE(test['trip_duration'], test['p_count_mean'])
p_count_error


# ## Mean trip_duration wrt vendor_id

# In[13]:


vendor = pd.pivot_table(train, values = 'trip_duration', index = ['vendor_id'], aggfunc = np.mean)
vendor


# In[14]:


# initializing new column to zero 
test['vendor_mean'] = 0

# for every new entry in vendor_id
for i in train['vendor_id'].unique():
    # assign the mean value corresponding to the new entry 
    test['vendor_mean'][test['vendor_id'] == i] = train['trip_duration'][train['vendor_id'] == i].mean()


# In[15]:


vendor_error = MAE(test['trip_duration'], test['vendor_mean'])
vendor_error


# In[16]:


# converting strings to datetime features
data['pickup_datetime'] = pd.to_datetime(data.pickup_datetime)
data['dropoff_datetime'] = pd.to_datetime(data.dropoff_datetime)

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
test['dropoff_datetime'] = pd.to_datetime(test.dropoff_datetime)

# Extracting day_of_week and hour_of_day
data['day_of_week'] = data['pickup_datetime'].dt.weekday
data['hour_of_day'] = data['pickup_datetime'].dt.hour

train['day_of_week'] = train['pickup_datetime'].dt.weekday
train['hour_of_day'] = train['pickup_datetime'].dt.hour

test['day_of_week'] = test['pickup_datetime'].dt.weekday
test['hour_of_day'] = test['pickup_datetime'].dt.hour


# In[17]:


# checking for the newly added columns
data.head()


# ## Mean trip_duration wrt day_of_week

# In[18]:


day = pd.pivot_table(train, values = 'trip_duration', index = ['day_of_week'], aggfunc = np.mean)
day


# In[19]:


# initializing new column to zero 
test['day_mean'] = 0

# for every new entry in day_of_week
for i in train['day_of_week'].unique():
    # assign the mean value corresponding to the new entry 
    test['day_mean'][test['day_of_week'] == i] = train['trip_duration'][train['day_of_week'] == i].mean()


# In[20]:


day_error = MAE(test['trip_duration'], test['day_mean'])
day_error


# ## Mean trip_duration wrt hour_of_day

# In[21]:


hour = pd.pivot_table(train, values = 'trip_duration', index = ['hour_of_day'], aggfunc = np.mean)
hour


# In[22]:


# initializing new column to zero 
test['hour_mean'] = 0

# for every new entry in hour_of_day
for i in train['hour_of_day'].unique():
    # assign the mean value corresponding to the new entry 
    test['hour_mean'][test['hour_of_day'] == i] = train['trip_duration'][train['hour_of_day'] == i].mean()


# In[23]:


hour_error = MAE(test['trip_duration'], test['hour_mean'])
hour_error


# ## Mean trip_duration wrt day_of_week and hour_of_day

# In[24]:


combo = pd.pivot_table(train, values = 'trip_duration', index = ['day_of_week','hour_of_day'], aggfunc = np.mean)
combo


# In[25]:


# initializing new column to zero 
test['combo_mean'] = 0

# assigning variables to strings to shorten code length
s2 = 'day_of_week'
s1 = 'hour_of_day'

# for every unique value in s1
for i in train[s1].unique():
    # for every unique value in s2
    for j in train[s2].unique():
        # calculate and assign mean to both the corresponding values s1 and s2 simultaneously
        test['combo_mean'][(test[s1] == i) & (test[s2] == j)] = train['trip_duration'][(train[s1] == i) & (train[s2] == j)].mean()


# In[26]:


# calculating absolute mean error 
combo_mean_error = MAE(test['trip_duration'], test['combo_mean'])
combo_mean_error


# In[ ]:




