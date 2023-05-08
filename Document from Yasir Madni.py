#!/usr/bin/env python
# coding: utf-8

# # Delivered To:

#                  ::: Dr Ayesha Hakim :::

# In[1]:


import numpy as np
import pandas as pd


# # Import the dataset

# In[8]:


df = pd.read_clipboard()


# In[10]:


df.head()


# In[12]:


df.to_csv("Housing.csv")


# In[18]:


df= pd.read_csv('Housing.data', header=None,delim_whitespace=True, names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
df


# # Shape of dataset

# In[21]:


print('Shape of Training dataset:', df.shape)


# # Checking null values for training dataset

# In[22]:


df.isnull().sum()


# # The Target Variable is the last one which is called MEDV.

# ## Here lets change ‘medv’ column name to ‘Price’

# In[38]:


df.rename(columns={'MEDV':'PRICE'}, inplace=True)


# In[39]:


df.head()


# # Exploratory Data Analysis

# ## Information about the dataset features

# In[40]:


df.info()


# ## Describe

# In[41]:


df.describe()


# # Feature Observation

# ## Finding out the correlation between the features

# In[42]:


corr = df.corr()
corr.shape


# ## Plotting the heatmap of correlation between features

# In[45]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[47]:


plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar=False, square= True, fmt='.2%', annot=True, cmap='Greens')


#                                                  ::: HeatMap :::

# # Checking the null values using heatmap

# ## There is any null values are occupyed here

# In[48]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ### Note: There are no null or missing values here.

# In[51]:


sns.set_style('whitegrid')
sns.countplot(x='RAD',data=df)


#                           ::: Counting For RAD Values :::

# In[52]:


sns.set_style('whitegrid')
sns.countplot(x='CHAS',data=df)


#                          ::: Counting For CHAS Feature :::

# In[55]:


sns.set_style('whitegrid')
sns.countplot(x='CHAS',hue='RAD',data=df,palette='RdBu_r')


#                          ::: CHAS DATA :::

# In[57]:


sns.histplot(data=df, x='AGE', color='darkred', bins=40)


#                          ::: HOUSE'S AGE Features Understanding :::

# In[60]:


sns.histplot(df['CRIM'].dropna(), kde=False, color='darkorange', bins=40)


#                          ::: CRIM RATE :::

# In[62]:


sns.histplot(df['RM'].dropna(), color='darkblue', bins=40)


#                          ::: Understanding Number of ROOMS into the HOUSES :::

# # Feature Selection

# ## Lets try to understand which are important feature for this dataset

# In[63]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# ## Independent Columns

# In[64]:


X = df.iloc[:,0:13]


# ## Target Column i.e PRICE range

# In[65]:


y = df.iloc[:,-1]


# In[67]:


y = np.round(df['PRICE'])


# # Apply SelectKBest class to extract top 5 best features

# In[68]:


bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# ## Concat two dataframes for better visualization

# In[69]:


featureScores = pd.concat([dfcolumns,dfscores],axis=1)


# ## Naming the dataframe Columns

# In[70]:


featureScores.columns = ['Specs','Score'] 
featureScores


# # Print 5 best features

# In[72]:


print(featureScores.nlargest(5,'Score'))


# # Feature Importance

# In[73]:


from sklearn.ensemble import ExtraTreesClassifier


# In[74]:


model = ExtraTreesClassifier()
model.fit(X,y)


# ## Use inbuilt class feature_importances of tree based classifiers

# In[75]:


print(model.feature_importances_) 


# ## Plot graph of feature importances for better visualization

# In[77]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


#         ::: Important Features rated by target variable correlation :::

# # Model Fitting
# 
# 
# # Linear Regression

# ### Value Assigning

# In[78]:


x=df.iloc[:,0:13]
y=df.iloc[:,-1]


# In[84]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[85]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[86]:


y_pred=model.predict(x_train)


# In[87]:


print("Training Accuracy:",model.score(x_train,y_train)*100)


# In[90]:


print("Testing Accuracy:",model.score(x_test,y_test)*100)


# In[94]:


from sklearn.metrics import mean_squared_error, r2_score


# In[96]:


print("Model Accuracy:",r2_score(y,model.predict(x))*100)


# In[97]:


plt.scatter(y_train,y_pred)
plt.xlabel('PRICES')
plt.ylabel('PREDICTED PRICES')
plt.title('PRICES VS RPEDICTED PRICES')
plt.show()


#             :::  See! how data points are predicted :::

# # Checking Residuals

# In[98]:


plt.scatter(y_pred,y_train-y_pred)
plt.title('RPEDICTED VS RESIDUALS')
plt.xlabel('PREDICTED')
plt.ylabel('RESIDUALS ')
plt.show()


#                    ::: Predicted Vs Residuals :::
# 

# # Checking Normality of of Errors

# In[99]:


sns.histplot(y_train-y_pred)
plt.title('Histogram of RESIDUALS')
plt.xlabel('RESIDUALS')
plt.ylabel('FREQUENCY ')
plt.show()


#                     ::: Hist Plotting for residuals :::

# # Random Forest Regression

# In[101]:


x=df.iloc[:,[-1,5,10,4,9]]
y=df.iloc[:,[-1]]


# In[102]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[105]:


from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(x_train,y_train)


# In[112]:


y_pred = reg.predict(x_train)


# In[113]:


print("Training Accuracy:",reg.score(x_train,y_train)*100)


# In[114]:


print("Testing Accuracy:",reg.score(x_test,y_test)*100)


# # Visualizing the difference between actual PRICES and PREDICTED values

# In[115]:


plt.scatter(y_train,y_pred)
plt.xlabel('PRICES')
plt.ylabel('PREDICTED PRICES')
plt.title('PRICES VS PREDICTED PRICES')
plt.show()


#                    ::: Linear Regression plotting data points :::

# # Prediction and Final Score:

# ### Finally we made it!!!

# # 1.Linear Regression

# ## Training Accuracy: 77.30135569264233
# 
# ## Testing Accuracy: 58.9222384918251
# 
# ## Model Accuracy: 73.73440319905033

# # 2. Random Forest Regressor
# 
# 

# 
# ## Training Accuracy: 99.99323673544639
# 
# ## Training Accuracy: 99.99323673544639
# 

# # Delivered By:

#                           ::: M Yasir Madni :::
#                           ::: Shoaib Yaseen :::
#                           ::: Muhammad Riyan:::
#                           ::: Hassan Raza   :::
#                           ::: Raza Abbas    :::

# In[ ]:




