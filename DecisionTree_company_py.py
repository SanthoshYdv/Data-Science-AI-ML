#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[42]:


comp = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Decision_tree_py\\Company_Data.csv")
comp.head()


# In[43]:


comp.describe()


# In[45]:


Sale = pd.cut(comp.Sales, bins = [0,8,17], labels =['Bad','Good'])
Sale.head()


# In[49]:


company = comp.drop(columns = ["Sales"])
company.head()
comp2 = pd.concat((company,Sale), axis = 1)
comp2


# In[52]:


dummy = pd.get_dummies(comp2, columns = ["ShelveLoc","Urban","US"], drop_first = True)
dummy


# In[58]:


column_names = ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education", "ShelveLoc_Good", "ShelveLoc_Medium","Urban_Yes","US_Yes","Sales"]
company_sales = dummy.reindex(columns=column_names)
company_sales.head()


# In[60]:


colnames = list(company_sales.columns)
colnames


# In[62]:


predictors = colnames[:11]
predictors


# In[64]:


target = colnames[11]
target


# In[77]:


company_sales.dropna(subset = ["Sales"], inplace=True)
company_sales['Sales'].isnull().sum()


# In[78]:


import numpy as np
from sklearn.model_selection import train_test_split
train,test = train_test_split(company_sales,test_size = 0.2)


# In[79]:


from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])


# In[81]:


#prediction
pred = model.predict(test[predictors])
pred


# In[83]:


pd.crosstab(test[target],pred)


# In[84]:


#accuracy
np.mean(pred==test.Sales)

