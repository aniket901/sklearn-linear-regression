#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression


# In[3]:


data = pd.read_csv(r'C:\Users\Sahil\Downloads\1.01.+Simple+linear+regression.csv')
data.head()


# In[4]:


x = data['SAT']
y = data['GPA']


# In[5]:


x.shape


# In[6]:


y.shape


# In[7]:


x_matrix = x.values.reshape(-1,1)
x_matrix.shape


# In[8]:


reg = LinearRegression()


# In[9]:


reg.fit(x_matrix,y)


# In[10]:


#rsquared
reg.score(x_matrix,y)


# In[11]:


#coefficient
reg.coef_


# In[12]:


reg.intercept_


# In[13]:


reg.predict([[1730]])


# In[ ]:




