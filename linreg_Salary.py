#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


sal=pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Salary_Data.csv")
sal


# In[4]:


plt.hist(sal.YearsExperience)


# In[5]:


plt.hist(sal.Salary)


# In[7]:


plt.boxplot(sal.YearsExperience)


# In[8]:


plt.boxplot(sal.Salary)


# In[9]:


sal.YearsExperience.corr(sal.Salary)


# In[13]:


#model
import statsmodels.formula.api as smf
model= smf.ols("Salary~YearsExperience", data=sal).fit()
model.summary() #0.957
model.params


# In[17]:


#prediction
pred=model.predict(sal.iloc[:,0])
print(round(pred,2))


# In[20]:


pred.corr(sal.Salary)


# In[19]:


#visualization
plt.scatter(x=sal['YearsExperience'],y=sal['Salary'],color='black');
plt.plot(sal['YearsExperience'],pred,color="red");
plt.xlabel('Experience'); plt.ylabel('Salary')


# In[ ]:




