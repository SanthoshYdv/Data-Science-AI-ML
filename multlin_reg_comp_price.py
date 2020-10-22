#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


comp=pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Mult_reg_py\\Computer_Data.csv")
comp


# In[6]:


comp.head(20)


# In[7]:


#correlation
comp.corr()


# In[9]:


#Scatterplot
import seaborn as sns
sns.pairplot(comp)


# In[10]:


comp.columns


# In[16]:


#model considering all variables
import statsmodels.formula.api as smf
m1 = smf.ols("price ~ speed+hd+ram+screen+cd+multi+premium+ads+trend", data=comp).fit()
m1.params
m1.summary() #0.776
#pvalue for all variables are less than 0.05


# In[38]:


#variation inflation of each variable
rsq_price = smf.ols("price ~ speed+hd+ram+screen+cd+multi+premium+ads+trend", data = comp).fit().rsquared
vif_price = 1/(1-rsq_price)

rsq_speed = smf.ols("speed ~price+hd+ram+screen+cd+multi+premium+ads+trend", data = comp).fit().rsquared
vif_speed = 1/(1-rsq_speed)

rsq_hd = smf.ols("hd ~price+speed+ram+screen+cd+multi+premium+ads+trend", data = comp).fit().rsquared
vif_hd = 1/(1-rsq_hd)

rsq_ram = smf.ols("ram ~price+speed+hd+screen+cd+multi+premium+ads+trend", data = comp).fit().rsquared
vif_ram = 1/(1-rsq_ram)

rsq_screen = smf.ols("screen ~price+speed+hd+ram+cd+multi+premium+ads+trend", data = comp).fit().rsquared
vif_screen = 1/(1-rsq_screen)

rsq_ads = smf.ols("ads ~ speed+hd+ram+screen+cd+multi+premium+price+trend", data = comp).fit().rsquared
vif_ads = 1/(1-rsq_ads)

rsq_trend = smf.ols("trend ~ speed+hd+ram+screen+cd+multi+premium+ads+price", data = comp).fit().rsquared
vif_trend = 1/(1-rsq_trend)


# In[41]:


#storing all vif values in a data frame
d1={'variables':['price','speed','hd','ram','screen','ads','trend'], 'vif':[vif_price,vif_speed,vif_hd,vif_ram,vif_screen,vif_ads,vif_trend]}
vif_frame=pd.DataFrame(d1)
vif_frame


# In[42]:


#added variable plot
sm.graphics.plot_partregress_grid(m1)


# In[45]:


#final model
model_final= smf.ols("price ~ speed+hd+ram+screen+cd+multi+premium+ads+trend", data = comp).fit()
model_final.summary() #0.777


# In[48]:


#predictions
pred_final= model_final.predict(comp)
round(pred_final,3)


# In[55]:


#visualization
plt.scatter(comp.price, pred_final, c = "r");plt.xlabel("observed values");plt.ylabel("fitted values")


# In[57]:


#residual vs fitted values
plt.scatter(pred_final, model_final.resid_pearson, c="r");plt.axhline(y=0, color="blue");plt.xlabel("fitted values");plt.ylabel("residual values")

