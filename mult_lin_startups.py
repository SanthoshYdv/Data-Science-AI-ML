#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[30]:


cars=pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Mult_reg_py\\50_Startups.csv")
cars


# In[31]:


#dropping state variable
cars1=cars.drop(["State"], axis=1)
cars1


# In[32]:


#scatter plot
import seaborn as sns
sns.pairplot(cars1)


# In[33]:


#correlation
cars1.corr()


# In[34]:


cars1.columns


# In[40]:


#model based on all variables
import statsmodels.formula.api as smf
m1= smf.ols("Profit~Administration+Marketing_Spend+RnD_Spend", data= cars1).fit()
m1.params
m1.summary() #0.951


# In[41]:


#model considering only RnD Spend
import statsmodels.formula.api as smf
m_r= smf.ols("Profit~RnD_Spend", data= cars1).fit()
m_r.params
m_r.summary() #0.947


# In[44]:


#model considering only Administration
import statsmodels.formula.api as smf
m_a= smf.ols("Profit~Administration", data= cars1).fit()
m_a.params
m_a.summary() #0.040, insignificant


# In[43]:


#model based on only Marketing spend
import statsmodels.formula.api as smf
m_m= smf.ols("Profit~Marketing_Spend", data= cars1).fit()
m_m.params
m_m.summary() #0.559


# In[45]:


#model based on Marketing spend and RnD spend
import statsmodels.formula.api as smf
m_mr= smf.ols("Profit~Marketing_Spend+RnD_Spend", data= cars1).fit()
m_mr.params
m_mr.summary() #0.950


# In[50]:


#influencial index plot
import statsmodels.api as sm
sm.graphics.influence_plot(m1)
#index 48 and 49 have high influence
cars.new=cars1.drop(cars1.index[[48,49]],axis=0)


# In[51]:


cars.new


# In[54]:


#new model
model_new= smf.ols("Profit~Administration+Marketing_Spend+RnD_Spend", data= cars.new).fit()
model_new.params
model_new.summary() #0.963


# In[55]:


#confidence values
print(model_new,conf_int(0.01))


# In[74]:


#prediction
pred_model_new=model_new.predict(cars.new[['Administration','RnD_Spend','Marketing_Spend']])
round(pred_model_new,2)


# In[104]:


#checking vif value for each variable
rsq_ad = smf.ols('Administration ~ Marketing_Spend + RnD_Spend',data=cars.new).fit().rsquared  
vif_ad = 1/(1-rsq_ad)

rsq_mr = smf.ols('Marketing_Spend ~ Administration + RnD_Spend',data=cars.new).fit().rsquared  
vif_mr = 1/(1-rsq_mr)

rsq_rd = smf.ols('RnD_Spend ~ Administration + Marketing_Spend',data=cars.new).fit().rsquared  
vif_rd = 1/(1-rsq_rd)


# In[105]:


#storing all vifs in a data frame
d1= {'variables':['Administration','Marketing','RnD'], 'VIF':[vif_ad,vif_mr,vif_rd]}
vif_frame= pd.DataFrame(d1)
vif_frame


# In[106]:


#added variable plot
sm.graphics.plot_partregress_grid(model_new)


# In[109]:


#final model
model_final = smf.ols("Profit ~ Administration+Marketing_Spend+RnD_Spend", data= cars.new).fit()
model_final.params
model_final.summary() #0.963


# In[112]:


#prediction
pred_final = model_final.predict(cars.new)
print(round(pred_final,2))


# In[113]:


#visualization
plt.scatter(cars.new.Profit, pred_final,c="r"); plt.xlabel("Observed values"); plt.ylabel("Fitted Values")


# In[114]:


#residual vs fitted values
plt.scatter(pred_final, model_final.resid_pearson, c="r"); plt.axhline(y=0, color="blue"); plt.xlabel("fitted values");plt.ylabel("residual values")

