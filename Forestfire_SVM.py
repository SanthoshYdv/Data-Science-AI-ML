#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


forest = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Support_Vector_Machines_py\\forestfires.csv")
forest.head()


# In[8]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
forest["size"] = le.fit_transform(forest.size_category)
forest.head()


# In[9]:


forest["size"].value_counts()


# In[10]:


le = LabelEncoder()
forest["mnth"] = le.fit_transform(forest.month)
forest.head()


# In[11]:


forest["mnth"].value_counts()


# In[12]:


le = LabelEncoder()
forest["days"] = le.fit_transform(forest.day)
forest.head()


# In[13]:


forest["days"].value_counts()


# In[14]:


fire = forest.drop(columns = ["month","day","size_category"])
fire.head()


# In[15]:


fire.describe()


# In[16]:


import seaborn as sns
sns.boxplot(x="rain",y="wind",data=fire,palette="hls")


# In[17]:


sns.boxplot(x="days",y="size",data=fire,palette="hls")


# In[18]:


sns.boxplot(x="mnth",y="size",data=fire,palette="hls")


# In[19]:


fire.columns


# In[20]:


fire = fire[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area','dayfri',
             'daymon', 'daysat', 'daysun', 'daythu', 'daytue', 'daywed',
             'monthapr', 'monthaug', 'monthdec', 'monthfeb', 'monthjan', 'monthjul',
            'monthjun', 'monthmar', 'monthmay', 'monthnov', 'monthoct', 'monthsep', 'mnth', 'days', 'size']]
fire.head()


# In[22]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(fire,test_size=0.3)


# In[24]:


train_x = train.iloc[:,0:30]
train_y = train.iloc[:,30]
test_x = test.iloc[:,0:30]
test_y = test.iloc[:,30]


# In[25]:


model_linear = SVC(kernel = "linear")
model_linear.fit(train_x,train_y)


# In[26]:


pred_linear = model_linear.predict(test_x)
np.mean(pred_linear==test_y)


# In[27]:


model_poly = SVC(kernel = "poly")
model_poly.fit(train_x,train_y)
pred_poly = model_poly.predict(test_x)
np.mean(pred_poly==test_y)


# In[28]:


model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_x,train_y)
pred_rbf = model_rbf.predict(test_x)
np.mean(pred_rbf==test_y)

