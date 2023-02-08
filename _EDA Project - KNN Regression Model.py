#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# loading the data 
data = pd.read_csv("C:\\Users\\91858\\Desktop\\Internshala Data Science Specialization\\EDA and ML\\nyc_taxi_trip_duration.csv")
data.head()


# In[4]:


data['trip_duration'].describe()/3600 # trip_duration in hours


# ### Here we have transformed the target variable using natural log

# In[5]:


import seaborn as sns
data['log_trip_duration'] = np.log(data['trip_duration'].values + 1)
sns.distplot(data['log_trip_duration'], kde = False, bins = 200)
plt.show()


# In[6]:


sns.distplot(data['trip_duration'], kde = False, bins = 200)
plt.show()


# ### The distribution of original target variable is not uniform as seen above

# In[7]:


data.shape


# In[8]:


data['passenger_count'].describe()


# ### We can clearly see that the maximum value of passnger_count is 9 whereas the mean is 1.66, therefore, the maximum value is an outlier.

# ## Removal of Outlier using Empirical rule of normal distribution

# In[9]:


# standard deviation factor 
factor = 1


# filtering using standard deviation
data = data[data['passenger_count'] < factor*data['passenger_count'].std()]


# In[10]:


# standard deviation factor 
factor = 1


# filtering using standard deviation
data = data[data['pickup_longitude'] < factor*data['pickup_longitude'].std()]
data = data[data['dropoff_longitude'] < factor*data['dropoff_longitude'].std()]


# In[11]:


# standard deviation factor 
factor = 1


# filtering using standard deviation
data = data[data['trip_duration'] < factor*data['trip_duration'].std()]


# In[12]:


data.shape


# In[13]:


data.head() # new dataset after removal of outliers


# In[14]:


# converting strings to datetime features
data['pickup_datetime'] = pd.to_datetime(data.pickup_datetime)
data['dropoff_datetime'] = pd.to_datetime(data.dropoff_datetime)

# Converting yes/no flag to 1 and 0
data['store_and_fwd_flag'] = 1 * (data.store_and_fwd_flag.values == 'Y')


# In[15]:


data['log_trip_duration'] = np.log(data['trip_duration'].values + 1)


# In[16]:


# Extracting new features from already existing features
data['day_of_week'] = data['pickup_datetime'].dt.weekday
data['hour_of_day'] = data['pickup_datetime'].dt.hour


# In[17]:


data.head()


# In[18]:


cleaned_data = data


# In[19]:


cleaned_data.head()


# In[24]:


# missing values
cleaned_data.isnull().sum()


# In[25]:


cleaned_data = cleaned_data.drop(['id'], axis = 1) # since 'id' has alphanumeric values
cleaned_data.head()


# In[26]:


# converting categorical variables into numbers using get_dummies function in pandas
cleaned_data = pd.get_dummies(cleaned_data, columns = ['vendor_id', 'passenger_count', 'store_and_fwd_flag'])
cleaned_data.head()


# In[27]:


# importing required libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[28]:


# reading the dataset again to check if we are working on the cleaned dataset
cleaned_data.head()


# In[29]:


## Segregating Independent and Dependent Variables
x = cleaned_data.drop(['trip_duration', 'dropoff_datetime', 'log_trip_duration', 'pickup_datetime'], axis = 1) # columns with strings are also dropped
y = cleaned_data['log_trip_duration']


# In[30]:


x.shape, y.shape


# In[31]:


x.head()


# In[32]:


# importing MinMaxScaler from sklearn
from sklearn.preprocessing import MinMaxScaler

# creating an instance of MinMaxScaler
scaler = MinMaxScaler()

# transforming the features in x into the scaled features ranging from 0 to 1
x_scaled = scaler.fit_transform(x)


# In[33]:


# converting scaled features into a dataframe
x = pd.DataFrame(x_scaled, columns = x.columns)


# In[34]:


# importing train_test_split function
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.33, random_state = 42)


# In[35]:


from sklearn.neighbors import KNeighborsRegressor as KNN 
from sklearn.metrics import mean_squared_error as MSE


# In[45]:


# creating an instance of Knn
clf = KNN(n_neighbors = 5)


# In[46]:


# fitting the model 
clf.fit(train_x, train_y)


# In[47]:


# predicting over train set and calculating f1 score
test_predict = clf.predict(train_x)
k = MSE(test_predict, train_y)
print('Test F1 Score ', k)


# ### Elbow for Regression

# In[48]:


def Elbow(k):
    test_error = []
    for i in k:
        clf = KNN(n_neighbors = i)
        clf.fit(train_x, train_y)
        tmp = clf.predict(test_x)
        tmp = MSE(tmp, test_y)
        error = 1 - tmp  
        test_error.append(error)
    return test_error  


# In[49]:


k = range(1,20,1)


# In[ ]:


test = Elbow(k)


# In[44]:


plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('Test Error')
plt.title('Elbow Curve for Test')


# In[ ]:


## minimum error is around  K = 2

