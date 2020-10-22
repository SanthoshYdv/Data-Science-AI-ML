#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


company = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Neural_network_py\\50_Startups.csv")
company.head(10)


# In[6]:


company.isnull().sum()


# In[45]:


X = company.drop(["Profit","State"],axis=1)
X
Y = company["Profit"]
Y


# In[46]:


#visualization
plt.hist(Y)


# In[47]:


company.Profit.value_counts()


# In[55]:


#split data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
X_train
Y_train


# In[49]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)


# In[50]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[56]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,20))
X_train = X_train.astype(int)
Y_train = Y_train.astype(int)
mlp.fit(X_train,Y_train)


# In[69]:


#prediction
pred_train = mlp.predict(X_train)
pred_train


# In[70]:


pred_test = mlp.predict(X_test)
pred_test


# In[74]:


from sklearn.metrics import classification_report, confusion_matrix
np.mean(Y_test==pred_test)
np.mean(Y_train==pred_train)

