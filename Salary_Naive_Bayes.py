#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[4]:


salary_train = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Naive_Bayes_py\\SalaryData_Train.csv")
salary_test = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Naive_Bayes_py\\SalaryData_Test.csv")
salary_train.head(10)
salary_test.head(10)


# In[6]:


salary_train1 = salary_train.drop(columns = ["educationno","maritalstatus","relationship","race","native","capitalgain","capitalloss"])
salary_train1.head(10)


# In[8]:


salary_test1 = salary_test.drop(columns = ["educationno","maritalstatus","relationship","race","native","capitalgain","capitalloss"])
salary_test1.head(10)


# In[11]:


salary_train1["Salary"].isna().value_counts()
salary_train1.dropna(subset = ["Salary"], inplace=True)
salary_train1["Salary"] = salary_train["Salary"].fillna(0)


# In[12]:


salary_test1["Salary"].isna().value_counts()
salary_test1["Salary"] = salary_test["Salary"].fillna(0)


# In[13]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salary_train1["Salary"] = le.fit_transform(salary_train1.Salary)
salary_train1["Salary"].value_counts()


# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salary_test1["Salary"] = le.fit_transform(salary_test1.Salary)
salary_test1["Salary"].value_counts()


# In[15]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salary_train1["workclass"] = le.fit_transform(salary_train1.workclass)
salary_train1["workclass"].value_counts()


# In[16]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salary_test1["workclass"] = le.fit_transform(salary_test1.workclass)
salary_test1["workclass"].value_counts()


# In[17]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salary_train1["education"] = le.fit_transform(salary_train1.education)
salary_train1["education"].value_counts()


# In[18]:


le = LabelEncoder()
salary_test1["education"] = le.fit_transform(salary_test1.education)
salary_test1["education"].value_counts()


# In[19]:


le = LabelEncoder()
salary_train1["occupation"] = le.fit_transform(salary_train1.occupation)
salary_train1["occupation"].value_counts()


# In[20]:


le = LabelEncoder()
salary_test1["occupation"] = le.fit_transform(salary_test1.occupation)
salary_test1["occupation"].value_counts()


# In[21]:


le = LabelEncoder()
salary_train1["sex"] = le.fit_transform(salary_train1.sex)
salary_train1["sex"].value_counts()


# In[22]:


le = LabelEncoder()
salary_test1["sex"] = le.fit_transform(salary_test1.sex)
salary_test1["sex"].value_counts()


# In[23]:


salary_train1.head(10)


# In[24]:


salary_test1.head(10)


# In[27]:


xtrain = salary_train1.iloc[:,0:6]
ytrain = salary_train1.iloc[:,6]
xtest = salary_test1.iloc[:,0:6]
ytest = salary_test1.iloc[:,6]


# In[26]:


ignb = GaussianNB()
imnb = MultinomialNB()


# In[29]:


pred_ignb = ignb.fit(xtrain,ytrain).predict(xtest)
pred_ignb


# In[31]:


pred_imnb = imnb.fit(xtrain,ytrain).predict(xtest)
pred_imnb


# In[32]:


#confusion matrix
confusion_matrix(ytest, pred_ignb)


# In[33]:


pd.crosstab(ytest.values.flatten(),pred_ignb)


# In[34]:


np.mean(pred_ignb == ytest.values.flatten())


# In[35]:


confusion_matrix(ytest, pred_imnb)


# In[37]:


pd.crosstab(ytest.values.flatten(),pred_imnb)


# In[38]:


np.mean(pred_imnb == ytest.values.flatten())

