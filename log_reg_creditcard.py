#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report


# In[7]:


cred= pd.read_csv("C:/Users/santy/Desktop/Python/Python~Assignments/Logistic_reg_py/creditcard.csv")
cred.head(10)


# In[10]:


#visualizations
sb.countplot(x='card', data = cred, palette= "hls")
pd.crosstab(cred.card,cred.income)


# In[12]:


sb.countplot(x='age', data = cred, palette= "hls")
pd.crosstab(cred.card,cred.age)


# In[13]:


sb.countplot(x='dependents', data = cred, palette= "hls")
pd.crosstab(cred.card,cred.dependents)


# In[14]:


sb.countplot(x='reports', data = cred, palette= "hls")
pd.crosstab(cred.card,cred.reports)


# In[15]:


#checking null values
cred.isnull().sum() #no null values present


# In[19]:


#logistic regression model
from sklearn.linear_model import LinearRegression
X= cred.iloc[:,[1,2,3,4,5,8,9,10,11]]
X
Y = cred.iloc[:,0]
Y


# In[24]:


classifier = LogisticRegression()
classifier.fit(X,Y)
classifier.coef_
classifier.predict_proba(X)


# In[28]:


#predictions
y_pred = classifier.predict(X)
y_pred
cred["card_pred"] = y_pred
cred


# In[34]:


#accuracy, confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y,y_pred))


# In[37]:


accuracy = sum(Y==y_pred)/cred.shape[0]
accuracy #0.981
pd.crosstab(y_pred,Y)


# In[39]:


from sklearn.metrics import classification_report
print(classification_report(Y,y_pred))

