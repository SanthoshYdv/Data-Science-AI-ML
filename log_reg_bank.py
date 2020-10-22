#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report


# In[46]:


bank = pd.read_csv("C:/Users/santy/Desktop/Python/Python~Assignments/Logistic_reg_py/bank-full.csv")
bank.head(20)


# In[47]:


#correlation
bank.corr()


# In[19]:


#visualizations
sb.countplot(x='y', data=bank, palette="hls")


# In[26]:


sb.countplot(x='job', data=bank, palette='hls')
pd.crosstab(bank.y,bank.job)


# In[23]:


sb.countplot(x='education', data=bank, palette='hls')
pd.crosstab(bank.y,bank.education)


# In[31]:


sb.countplot(x='job', data=bank, palette='hls')
pd.crosstab(bank.job,bank.education)


# In[48]:


#checking null values
bank.isnull().sum() #no null values found


# In[57]:


from sklearn.linear_model import LogisticRegression
bank.shape
X= bank.iloc[:,[0,5,9,11,12,13,14]]
X
Y= bank.iloc[:,17]
Y


# In[62]:


classifier = LogisticRegression()
classifier.fit(X,Y)
classifier.coef_
classifier.predict_proba(X)


# In[66]:


#predictions
y_pred = classifier.predict(X)
y_pred
bank["y_pred"] = y_pred


# In[71]:


#accuracy
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y,y_pred)


# In[78]:


type(y_pred)
accuracy = (sum(Y==y_pred)/(bank.shape[0]))
accuracy
pd.crosstab(y_pred, Y)


# In[80]:


#classification report
from sklearn.metrics import classification_report
print(classification_report(Y,y_pred))

