#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


zoo = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\KNN_py\\Zoo.csv")
zoo.head(10)


# In[4]:


#split
from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo, test_size = 0.2)


# In[19]:


from sklearn.neighbors import KNeighborsClassifier as KNC
neigh = KNC(n_neighbors = 3)
neigh.fit(train.iloc[:,1:16],train.iloc[:,17])


# In[35]:


#train_accuracy
train_acc = np.mean(neigh.predict(train.iloc[:,1:16])==train.iloc[:,17])
train_acc


# In[36]:


#test_acc
test_acc = np.mean(neigh.predict(test.iloc[:,1:16])==test.iloc[:,17])
test_acc


# In[37]:


#for 5 nearerst neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC
neigh = KNC(n_neighbors = 5)
neigh.fit(train.iloc[:,1:16],train.iloc[:,17])


# In[38]:


#train_accuracy
train_acc = np.mean(neigh.predict(train.iloc[:,1:16])==train.iloc[:,17])
train_acc


# In[39]:


#test_acc
test_acc = np.mean(neigh.predict(test.iloc[:,1:16])==test.iloc[:,17])
test_acc


# In[44]:


#creating empty list variable
acc =[]
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:16],train.iloc[:,17])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:16])==train.iloc[:,17])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:16])==test.iloc[:,17])
    acc.append([train_acc,test_acc])


# In[51]:


#training_acc plot
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")


# In[53]:


import matplotlib.pyplot as plt
#training_acc plot
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
#test acc plot
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

