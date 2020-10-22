#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


cmp = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Random_forest_py\\Company_Data (1).csv")
cmp.head()


# In[6]:


cmp.describe()


# In[8]:


Sale = pd.cut(cmp.Sales, bins = [0,8,17], labels = ["Bad","Good"])
Sale


# In[14]:


cmp1 = cmp.drop(["Sales"],axis=1)
cmp1
cmp2 = pd.concat((cmp1,Sale), axis=1)
cmp2.head()


# In[17]:


company = pd.get_dummies(cmp2, columns = ["ShelveLoc","Urban","US"], drop_first = True)
company.head()


# In[23]:


column_names = ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education", "ShelveLoc_Good", "ShelveLoc_Medium","Urban_Yes","US_Yes","Sales"]
company_sales = company.reindex(columns=column_names)
company_sales.head()


# colnames = list(company_sales.columns)
# colnames

# In[27]:


predictors = colnames[:11]
predictors


# In[29]:


target = colnames[11]
target


# In[41]:


company_sales.dropna(subset = ["Sales"], inplace=True)
company_sales['Sales'].isnull().sum()


# In[42]:


X = company_sales[predictors]
Y = company_sales[target]


# In[43]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=15,criterion="entropy")


# In[44]:


rf.fit(X,Y) 


# In[45]:


rf.estimators_ 


# In[46]:


rf.classes_


# In[47]:


rf.n_classes_


# In[48]:


rf.n_features_


# In[49]:


rf.n_outputs_ 


# In[50]:


rf.oob_score_ 


# In[51]:


rf.predict(X)


# In[58]:


#storing predicted values
company_sales["rf_pred"] = rf.predict(X)
cols = ['rf_pred','Sales']
cols


# In[59]:


company_sales[cols].head()


# In[60]:


#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(company_sales["Sales"],company_sales["rf_pred"])

