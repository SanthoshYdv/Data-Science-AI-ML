#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


fraud = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Random_forest_py\\Fraud_check.csv")
fraud.head()


# In[12]:


fraud.describe()


# In[14]:


Income = pd.cut(fraud.Taxable_Income, bins = [0,30000,100000], labels = ["Risky","Good"])
Income.head()


# In[19]:


fraud1 = fraud.drop(columns=["Taxable_Income"])
fraud1
fraud2 = pd.concat((fraud1,Income), axis=1)
fraud2.head()


# In[21]:


fraud_data = pd.get_dummies(fraud2, columns=["Undergrad","Marital_Status","Urban"], drop_first=True)
fraud_data.head()


# In[28]:


colnames = list(frauds.columns)
colnames


# In[26]:


column_names = ["City_Population","Work_Experience","Undergrad_YES","Marital_Status_Married","Marital_Status_Single","Urban_YES","Taxable_Income"]
frauds = fraud_data.reindex(columns=column_names)
frauds.head()


# In[30]:


predictors = colnames[:6]
predictors


# In[32]:


target = colnames[6]
target


# In[33]:


X = frauds[predictors]
Y = frauds[target]


# In[34]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=15,criterion="entropy")


# In[35]:


rf.fit(X,Y)


# In[36]:


rf.estimators_


# In[37]:


rf.classes_


# In[38]:


rf.n_classes_


# In[41]:


rf.n_features_


# In[42]:


rf.n_outputs_


# In[43]:


rf.oob_score_


# In[44]:


rf.predict(X)


# In[47]:


frauds["rf_pred"]=rf.predict(X)
frauds.head()


# In[49]:


cols = ["rf_pred","Taxable_Income"]
frauds[cols].head(10)


# In[51]:


from sklearn.metrics import confusion_matrix
confusion_matrix(frauds['Taxable_Income'],frauds['rf_pred'])
np.mean(frauds['Taxable_Income']==frauds['rf_pred'])

