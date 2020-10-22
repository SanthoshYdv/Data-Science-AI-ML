#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[77]:


calories= pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\calories_consumed.csv")
calories


# In[91]:


A= calories.iloc[:,0:1]
A


# In[92]:


B= calories.iloc[:,1:2]
B


# In[106]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
Lin_model= linear_model.LinearRegression()
M= Lin_model.fit(A,B)
M
M.score(A,B)


# In[108]:


B_hat= M.predict(A)
B_hat


# In[110]:


#plot
plt.scatter(A,B_hat)
plt.xlabel("Target Variable")
plt.ylabel("Independent Variable")
plt.show()


# In[111]:


plt.scatter(A,B)
plt.xlabel("Target Variable")
plt.ylabel("Independent Variable")
plt.show()

