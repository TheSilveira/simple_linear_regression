#!/usr/bin/env python
# coding: utf-8

# ## Importing Main Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('insurance.csv',usecols=['age','sex','bmi','children','smoker','charges'])


# ## Exploratory Data Analysis

# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.shape[0]


# In[7]:


le = LabelEncoder()
df[['sex','smoker']] = df[['sex','smoker']].apply(le.fit_transform)


# In[8]:


df.head()


# In[9]:


df.corr().round(4)


# ## Analyzing Y Variable

# In[10]:


ax = sns.boxplot(data=df['charges'], orient='v', width=0.2)
ax.figure.set_size_inches(12,6)
ax.set_title('Charges', fontsize=20)
ax.set_ylabel('US$', fontsize=16)
ax


# In[11]:


sex_legend = {0:'female',1:'male'}
smoker_legend = {0:'no',1:'yes'}


# In[12]:


ax = sns.boxplot(y='charges', x='sex', data=df, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_title('Charges x Sex', fontsize=20)
ax.set_ylabel('US$',fontsize=16)
ax.set_xlabel('Sex', fontsize=16)
ax.set_xticklabels(sex_legend.values(), rotation=0)
ax


# In[13]:


ax = sns.boxplot(y='charges', x='smoker', data=df, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_title('Charges x Smoker', fontsize=20)
ax.set_ylabel('US$',fontsize=16)
ax.set_xlabel('Smoker', fontsize=16)
ax.set_xticklabels(smoker_legend.values(), rotation=0)
ax


# In[14]:


ax = sns.boxplot(y='charges', x='children', data=df, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_title('Charges x Children', fontsize=20)
ax.set_ylabel('US$',fontsize=16)
ax.set_xlabel('Children', fontsize=16)
ax


# In[15]:


ax = sns.distplot(df['charges'])
ax.figure.set_size_inches(12,6)
ax.set_title('Frequency Distribution', fontsize=20)
ax.set_ylabel('US$',fontsize=16)
ax


# In[16]:


ax = sns.pairplot(df, y_vars='charges', x_vars=['sex','smoker','children','bmi','age'], kind='reg')
ax.fig.suptitle('Variables Dispersion', fontsize=20, y=1.1)
ax


# In[33]:


ax = sns.jointplot(x='smoker', y='charges', data=df, kind='reg')
ax.fig.suptitle('Dispersion - Charges x Smoker', fontsize=18, y=1.05)
ax.set_axis_labels('Smoker', 'Charges', fontsize=14)
ax


# In[35]:


df['charges'].min()


# ## Creating model

# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[19]:


y = df['charges']


# In[20]:


X = df[['sex','children','bmi','age']]


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)


# In[22]:


X_train.shape[0]


# In[23]:


X_test.shape[0]


# In[24]:


model = LinearRegression()


# In[25]:


model.fit(X_train, y_train)


# In[26]:


print('R² = {}'.format(model.score(X_train, y_train).round(2)))


# In[27]:


y_pred = model.predict(X_test)


# In[28]:


print('R² = %s' %metrics.r2_score(y_test, y_pred).round(2))


# In[29]:


entry = X_test[0:1]
entry


# In[30]:


model.predict(entry)[0]


# In[31]:


model.intercept_


# In[32]:


model.coef_

