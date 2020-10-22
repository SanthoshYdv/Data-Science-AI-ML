#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


emp=pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\emp_data.csv")
emp


# In[4]:


emp.Salary_hike.corr(emp.Churn_out_rate)


# In[8]:


plt.hist(emp.Salary_hike)


# In[9]:


plt.hist(emp.Churn_out_rate)


# In[10]:


plt.boxplot(emp.Salary_hike)


# In[11]:


plt.boxplot(emp.Churn_out_rate)


# In[17]:


#model
import statsmodels.formula.api as smf
model=smf.ols("Churn_out_rate~Salary_hike", data=emp).fit()
model.summary() #0.831
model.params


# In[24]:


#prediction
pred= model.predict(emp.iloc[:,0])
pred


# In[27]:


import statsmodels.formula.api as smf
model1=smf.ols("np.log(Churn_out_rate)~Salary_hike", data=emp).fit()
model1.summary()  #0.8754
model1.params


# In[30]:


import statsmodels.formula.api as smf
model2=smf.ols("Churn_out_rate~np.log(Salary_hike)", data=emp).fit()
model2.summary() #0.849
model2.params


# In[32]:


import statsmodels.formula.api as smf
model_final=smf.ols("np.log(Churn_out_rate)~Salary_hike", data=emp).fit()
model_final.summary()  #0.8754
model_final.params


# In[34]:


#prediction
pred_f= model_final.predict(emp.iloc[:,0])
pred


# In[37]:


#visualization
plt.scatter(x=emp['Salary_hike'],y=emp['Churn_out_rate'],color='red');plt.plot(emp['Churn_out_rate'],pred,color='black');plt.xlabel('Salary_hike'),plt.ylabel('Churn_out_rate')


# In[40]:


#residuals
resid=model_final.resid_pearson
resid
plt.plot(model_final.resid_pearson,'o'); plt.axhline(y=0,color='green'); plt.xlabel('observation number');plt.ylabel("standardized residual")

