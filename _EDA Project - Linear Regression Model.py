#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# loading the data 
data = pd.read_csv("C:\\Users\\91858\\Desktop\\nyc_taxi_trip_duration.csv")
data.head()


# In[3]:


sns.distplot(data['trip_duration'], kde = False, bins = 200)
plt.show()


# ### Since, the distribution of target variable is not normal, therefore, we need to tranform it. Here I have tranformed it using natural log.  

# In[4]:


data['log_trip_duration'] = np.log(data['trip_duration'].values + 1)
sns.distplot(data['log_trip_duration'], kde = False, bins = 200)
plt.show()


# In[5]:


data.shape


# In[6]:


data['passenger_count'].describe()


# ### We can clearly see that the maximum value of passenger count is 9 whereas the mean is 1.66, hence the maximum value is definitely an oulier.

# ## Removal of Outliers using Empirical rule of normal distribution

# In[7]:


# standard deviation factor 
factor = 1


# filtering using standard deviation
data = data[data['passenger_count'] < factor*data['passenger_count'].std()]


# In[8]:


# standard deviation factor 
factor = 1


# filtering using standard deviation
data = data[data['pickup_longitude'] < factor*data['pickup_longitude'].std()]
data = data[data['dropoff_longitude'] < factor*data['dropoff_longitude'].std()]


# In[9]:


# standard deviation factor 
factor = 1


# filtering using standard deviation
data = data[data['trip_duration'] < factor*data['trip_duration'].std()]


# In[10]:


data.shape


# In[11]:


data.head() # new dataset after removal of outliers


# In[12]:


# converting strings to datetime features
data['pickup_datetime'] = pd.to_datetime(data.pickup_datetime)
data['dropoff_datetime'] = pd.to_datetime(data.dropoff_datetime)

# Converting yes/no flag to 1 and 0
data['store_and_fwd_flag'] = 1 * (data.store_and_fwd_flag.values == 'Y')


# In[13]:


# Extracting new features from already existing features
data['day_of_week'] = data['pickup_datetime'].dt.weekday
data['hour_of_day'] = data['pickup_datetime'].dt.hour


# In[14]:


data = data.drop(['id'], axis = 1) # as 'id' contains alphanumeric strings
data.head()


# In[15]:


# converting categorical variables into numbers using get_dummies function in pandas
data = pd.get_dummies(data, columns = ['vendor_id', 'passenger_count', 'store_and_fwd_flag'])
data.head()


# In[16]:


## Segregating Independent and Dependent Variables
x = data.drop(['trip_duration', 'dropoff_datetime', 'log_trip_duration', 'pickup_datetime'], axis = 1) # columns with strings are also dropped
y = data['log_trip_duration']


# In[17]:


x.shape, y.shape


# ### Importing train_test_split function from sklearn

# In[18]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 56)


# ### Importing LinearRegression and evaluation metric from sklearn

# In[19]:


from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error as mae


# In[20]:


# Creating an instance of LR
lr = LR()

# fitting the model 
lr.fit(train_x, train_y)


# In[21]:


# Predicting over train set and calculating error 
train_predict = lr.predict(train_x)
k = mae(train_predict, train_y)
print('Training Mean Absolute Error', k)


# In[22]:


# Predicting over test set and calculating error 
test_predict = lr.predict(test_x)
k = mae(test_predict, test_y)
print('Testing Mean Absolute Error', k)


# ### Parameters of LR

# In[23]:


lr.coef_


# ### Plotting the Coefficients

# In[24]:


plt.figure(figsize=(8, 6), dpi = 120, facecolor = 'w', edgecolor = 'b')
x = range(len(train_x.columns))
y = lr.coef_
plt.bar(x,y)
plt.xlabel('Variables')
plt.ylabel('Coefficients')
plt.title('Coefficient Plot')


# ## Checking Assumptions of Linear Model

# In[25]:


residuals = pd.DataFrame({'Fitted Values': test_y, 'Predicted Values': test_predict})
residuals['residuals'] = residuals['Fitted Values'] - residuals['Predicted Values']
residuals.head()


# In[26]:


residuals.shape


# ### Plotting the residual curve

# In[30]:


plt.figure(figsize=(10, 6), dpi = 120, facecolor = 'w', edgecolor = 'b')
f = range(0, 128669)
k = [0 for i in range(0, 128669)]
plt.scatter(f, residuals.residuals[:], label = 'residuals')
plt.plot(f, k, color = 'red', label = 'regression line')
plt.xlabel('Fitting points')
plt.ylabel('Residuals')
plt.ylim(-20,20)
plt.title('Residual Plot')
plt.legend()


# ### Checking the distribution of residuals

# In[31]:


plt.figure(figsize=(10, 6), dpi = 120, facecolor = 'w', edgecolor = 'b')
plt.hist(residuals.residuals, bins = 150)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Distribution of Error')
plt.show()


# ### Q-Q Plot

# In[33]:


from statsmodels.graphics.gofplots import qqplot

fig, ax = plt.subplots(figsize = (5, 5), dpi = 120)
qqplot(residuals.residuals, line = 's', ax = ax)
plt.ylabel('Residual Quantiles')
plt.xlabel('Ideal Scaled Quantiles')
plt.ylim(-20,20)
plt.title('Checking Distribution of Residual Errors')
plt.show()


# ### Variance Inflation Factor (Checking for multi-collinearity)

# In[52]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X = add_constant(data.drop(['pickup_datetime', 'dropoff_datetime'], axis = 1))
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# In[54]:


data = data.drop(['trip_duration'], axis = 1)
data.head()


# ## Implementing Linear Regression with Normalization now

# In[55]:


lr = LR(normalize = True)

lr.fit(train_x, train_y)


# In[56]:


# Predicting over train set and calculating error 
train_predict = lr.predict(train_x)
k = mae(train_predict, train_y)
print('Training Mean Absolute Error', k)


# In[57]:


# Predicting over test set and calculating error 
test_predict = lr.predict(test_x)
k = mae(test_predict, test_y)
print('Testing Mean Absolute Error', k)


# ### PLotting the Normalized Coefficient Plot

# In[58]:


plt.figure(figsize=(8, 6), dpi = 120, facecolor = 'w', edgecolor = 'b')
x = range(len(train_x.columns))
y = lr.coef_
plt.bar(x,y)
plt.xlabel('Variables')
plt.ylabel('Coefficients')
plt.title('Normalized Coefficient Plot')


# ### Creating new subsets of data

# In[61]:


x = data.drop(['log_trip_duration', 'dropoff_datetime', 'pickup_datetime'], axis = 1)
y = data['log_trip_duration']
x.shape, y.shape


# ### Arranging Coefficients with features

# In[62]:


Coefficients = pd.DataFrame({'Variables': x.columns, 'coefficients': lr.coef_})
Coefficients.head()


# ### Choosing Variables with significance greater than 1

# In[63]:


sig_var = Coefficients[Coefficients.coefficients > 1]


# ### Extracting significant subset to independent variables

# In[64]:


subset = data[sig_var['Variables'].values]
subset.head()


# In[65]:


subset.shape


# ### Now we will again implement LR Model on that subset

# In[66]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(subset, y, random_state = 56)


# In[67]:


from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error as mae


# In[68]:


# Creating an instance of LR
lr = LR(normalize = True)

# fitting the model 
lr.fit(train_x, train_y)


# In[69]:


# Predicting over train set and calculating error 
train_predict = lr.predict(train_x)
k = mae(train_predict, train_y)
print('Training Mean Absolute Error', k)


# In[70]:


# Predicting over test set and calculating error 
test_predict = lr.predict(test_x)
k = mae(test_predict, test_y)
print('Testing Mean Absolute Error', k)


# In[71]:


plt.figure(figsize=(8, 6), dpi = 120, facecolor = 'w', edgecolor = 'b')
x = range(len(train_x.columns))
y = lr.coef_
plt.bar(x,y)
plt.xlabel('Variables')
plt.ylabel('Coefficients')
plt.title('Normalized Coefficient Plot')


# ### Now the number of significant variables are reduced.

# ## Regularization using Lasso

# In[99]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error  # we will use MSE for evaluation
import matplotlib.pyplot as plt


# ### Below we are defining custom functions for plotting test and train errors Vs lambda values and model evaluation

# In[100]:


def plot_errors(lambdas, train_errors, test_errors, title):
    plt.figure(figsize=(16, 9))
    plt.plot(lambdas, train_errors, label="train")
    plt.plot(lambdas, test_errors, label="test")
    plt.xlabel("$\\lambda$", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=14)
    plt.show()


# In[101]:


def evaluate_model(Model, lambdas):
    training_errors = [] # we will store the error on the training set, for using each different lambda
    testing_errors = [] # and the error on the testing set
    for l in lambdas:
        # in sklearn, they refer to lambda as alpha
        model = Model(alpha=l, max_iter=1000) # we allow max number of iterations until the model converges
        model.fit(train_x, train_y)

        training_preds = model.predict(train_x)
        training_mse = mean_squared_error(train_y, training_preds)
        training_errors.append(training_mse)

        testing_preds = model.predict(test_x)
        testing_mse = mean_squared_error(test_y, testing_preds)
        testing_errors.append(testing_mse)
    return training_errors, testing_errors


# ### Giving a range of lambda values for model evaluation and plotting of errors

# In[103]:


lambdas = np.arange(-10, 10, step=0.1)

lasso_train, lasso_test = evaluate_model(Lasso, lambdas)
plot_errors(lambdas, lasso_train, lasso_test, "Lasso")


# In[ ]:




