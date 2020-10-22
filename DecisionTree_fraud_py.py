#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[25]:


fraud = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Decision_tree_py\\Fraud_check.csv")
fraud.head(10)


# In[31]:


Income = pd.cut(fraud.Taxable_Income,bins=[0,30000,100000],labels=['Risky','Good'])
Income.head(15)


# In[57]:


fraud.head()


# In[59]:


dummy = pd.get_dummies(fraud['Undergrad'])
dummy


# In[80]:


fraud1 = pd.concat((fraud,dummy),axis = 1)
fraud1


# In[82]:


fraud2 = fraud1.drop(columns=["Undergrad","NO"])
fraud2


# In[88]:


fraud3 = fraud2.rename(columns = {"YES":"Undergrad"})
fraud3.head()


# In[91]:


dummy = pd.get_dummies(fraud3['Marital_Status'])
dummy


# In[95]:


fraud4 = pd.concat((fraud3,dummy), axis=1)
fraud4.head()


# In[101]:


fraud5 = fraud4.drop(columns=["Marital_Status","Divorced","Married"])
fraud5.rename(columns = {"Single":"Marital_Status"})
fraud5.head()


# In[103]:


dummy = pd.get_dummies(fraud5["Urban"])
dummy


# In[109]:


fraud6 = pd.concat((fraud5,dummy), axis=1)
fraud6.head()


# In[113]:


fraud7 = fraud6.drop(columns=["Urban","NO"])
fraud7.head()


# In[119]:


fraud8 = fraud7.rename(columns={"YES":"Urban"})
fraud8.head()
fraud9 = fraud8.drop(columns = ["Taxable_Income"])
fraud9.head()


# In[122]:


column_names = ["City_Population", "Work_Experience", "Undergrad", "Single", "Urban", "Income"]
frauds = fraud9.reindex(columns=column_names)
frauds.head()


# In[125]:


colnames = list(frauds.columns)
colnames


# In[131]:


predictors = colnames[:4]
predictors
target = colnames[5]
target


# In[132]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(frauds, test_size=0.2)


# In[133]:


from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])


# In[135]:


#predictions
pred = model.predict(test[predictors])
pred


# In[137]:


pd.Series(pred).value_counts()


# In[138]:


pd.crosstab(test[target],pred)


# In[139]:


np.mean(pred==test.Income)


# In[ ]:




