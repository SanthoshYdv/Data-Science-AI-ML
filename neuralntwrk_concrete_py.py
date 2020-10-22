#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


concrete = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Neural_network_py\\concrete.csv")
concrete
concrete.columns
concrete.isnull().sum()


# In[8]:


concrete.head(10)


# In[10]:


X = concrete.drop(["strength"],axis=1)
X


# In[12]:


Y = concrete["strength"]
Y


# In[13]:


#visualization
plt.hist(Y)


# In[14]:


concrete.strength.value_counts()


# In[49]:


#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)


# In[50]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[51]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30))


# In[58]:


mlp.fit(X_train,y_train)


# In[59]:


prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)


# In[61]:



np.mean(y_test==prediction_test)
np.mean(y_train==prediction_train)

