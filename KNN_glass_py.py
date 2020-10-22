#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


glass = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\KNN_py\\glass.csv")
glass.head(10)


# In[4]:


#split data
from sklearn.model_selection import train_test_split
train,test = train_test_split(glass, test_size = 0.2)


# In[7]:


#knn
from sklearn.neighbors import KNeighborsClassifier as KNC


# In[11]:


#for 3 nearest neighbors
neigh = KNC(n_neighbors = 3)
neigh.fit(train.iloc[:,0:8],train.iloc[:,9])


# In[13]:


#train accuracy
train_acc = np.mean(neigh.predict(train.iloc[:,0:8])==train.iloc[:,9])
train_acc


# In[16]:


#test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:8])==test.iloc[:,9])
test_acc


# In[17]:


#for 5 nearest neighbors
neigh = KNC(n_neighbors = 5)
neigh.fit(train.iloc[:,0:8],train.iloc[:,9])


# In[18]:


#train accuracy
train_acc = np.mean(neigh.predict(train.iloc[:,0:8])==train.iloc[:,9])
train_acc


# In[19]:


#test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:8])==test.iloc[:,9])
test_acc


# In[25]:


#creating empty list variable
acc = []
#running KNN algorithm for 3 to 50 nearerst neighbors
for i in range(3,50,2):
    neigh = KNC(n_neighbors = 5)
    neigh.fit(train.iloc[:,0:8],train.iloc[:,9]) 
    train_acc = np.mean(neigh.predict(train.iloc[:,0:8])==train.iloc[:,9])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:8])==test.iloc[:,9])
    acc.append([train_acc,test_acc])


# In[26]:


# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")


# In[27]:


# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")

