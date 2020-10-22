#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report


# In[4]:


bank = pd.read_csv("C:/Users/santy/Desktop/Python/Python~Assignments/Logistic_reg_py/bank-full.csv")
bank.head(10)


# In[5]:


#correlation
bank.corr()


# In[6]:


#visualizations
sb.countplot(x='y', data=bank, palette="hls")


# In[7]:


sb.countplot(x='job', data=bank, palette='hls')
pd.crosstab(bank.y,bank.job)


# In[8]:


sb.countplot(x='education', data=bank, palette='hls')
pd.crosstab(bank.y,bank.education)


# In[9]:


sb.countplot(x='job', data=bank, palette='hls')
pd.crosstab(bank.job,bank.education)


# In[10]:


#checking null values
bank.isnull().sum() #no null values found


# In[11]:


from sklearn.linear_model import LogisticRegression
bank.shape
X= bank.iloc[:,[0,5,9,11,12,13,14]]
X
Y= bank.iloc[:,17]
Y


# In[12]:


classifier = LogisticRegression()
classifier.fit(X,Y)
classifier.coef_
classifier.predict_proba(X)


# In[13]:


#predictions
y_pred = classifier.predict(X)
y_pred
bank["y_pred"] = y_pred


# In[14]:


#accuracy
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y,y_pred)


# In[15]:


type(y_pred)
accuracy = (sum(Y==y_pred)/(bank.shape[0]))
accuracy
pd.crosstab(y_pred, Y)


# In[16]:


#classification report
from sklearn.metrics import classification_report
print(classification_report(Y,y_pred))


# In[20]:


#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
log_roc_auc = roc_auc_score(Y,classifier.predict(X))
fpr, tpr, threshold = roc_curve(Y, classifier.predict_proba(X)[:,1])


# In[24]:


plt.figure()
plt.plot(fpr, tpr, label = "classifier (area = 0.2f)" % log_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Negetive Rate')
plt.title('ROC')
plt.show()

