#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[13]:


corolla=pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Mult_reg_py\\Toyota_Corolla.csv",encoding='windows-1252' )
corolla


# In[28]:


corolla.new=corolla.drop([corolla.index[0]])
corolla.new


# In[34]:


#correlation
corolla.new.corr()


# In[35]:


corolla.new.columns


# In[38]:


#model using all variables
import statsmodels.formula.api as smf
m1= smf.ols("Price ~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data= corolla.new).fit()
m1.params
m1.summary() #0.864
#variables cc and door have p-vcalues higher than 0.05


# In[43]:


#model considering only cc
m2 = smf.ols("Price~cc", data=corolla.new).fit()
m2.summary() #0.016


# In[45]:


#model considering only Doors
m3 = smf.ols("Price~ Doors", data= corolla.new).fit()
m3.summary() #0.035


# In[52]:


#model with both cc and doors
m4= smf.ols("Price~ cc+Doors", data= corolla.new).fit()
m4.summary()


# In[46]:


#Influencial index plot
import statsmodels.api as sm
sm.graphics.influence_plot(m1)


# In[50]:


#index 80 shows high influence
corolla1= corolla.new.drop(corolla.new.index[[80]], axis=0)
corolla1


# In[57]:


#model with altered dataset
m_new= smf.ols("Price ~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = corolla1).fit()
m_new.summary()


# In[70]:


#vif values for all variables
rsq_prc= smf.ols("Price ~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data=corolla1).fit().rsquared
vif_prc= 1/(1-rsq_prc)

rsq_age= smf.ols("Age_08_04~Price+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data=corolla1).fit().rsquared
vif_age= 1/(1-rsq_age)

rsq_km= smf.ols("KM~Price+Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data=corolla1).fit().rsquared
vif_km= 1/(1-rsq_km)

rsq_hp= smf.ols("HP~Age_08_04+KM+Price+cc+Doors+Gears+Quarterly_Tax+Weight", data=corolla1).fit().rsquared
vif_hp= 1/(1-rsq_hp)

rsq_cc= smf.ols("cc~Age_08_04+KM+HP+Price+Doors+Gears+Quarterly_Tax+Weight", data=corolla1).fit().rsquared
vif_cc= 1/(1-rsq_cc)

rsq_doors= smf.ols("Doors~Age_08_04+KM+HP+cc+Price+Gears+Quarterly_Tax+Weight", data=corolla1).fit().rsquared
vif_doors= 1/(1-rsq_doors)

rsq_grs= smf.ols("Gears~Age_08_04+KM+HP+cc+Doors+Price+Quarterly_Tax+Weight", data=corolla1).fit().rsquared
vif_grs= 1/(1-rsq_grs)

rsq_qrtly= smf.ols("Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Price+Weight", data=corolla1).fit().rsquared
vif_qrtly= 1/(1-rsq_qrtly)

rsq_wght= smf.ols("Weight ~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Price", data=corolla1).fit().rsquared
vif_wght= 1/(1-rsq_wght)


# In[74]:


#putting all vif value to a data frame
d1= {'variables':['Price','age','km','hp','cc','doors','gears','tax','weight'],'vif':[vif_prc,vif_age,vif_km,vif_hp,vif_cc,vif_doors,vif_grs,vif_qrtly,vif_wght]}
vif_frame= pd.DataFrame(d1)
vif_frame


# In[75]:


#added variable plot
sm.graphics.plot_partregress_grid(m_new) 


# In[79]:


#since the added variable plot is not showing any significance, dropping doors variable
m_final = smf.ols("Price ~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight", data= corolla1).fit()
m_final.params
m_final.summary()


# In[87]:


#predictions
pred=m_final.predict(corolla1)
round(pred,2)


# In[92]:


#visualization
plt.scatter(corolla1.Price,pred,c="r");plt.xlabel("Observed Values");plt.ylabel("Fitted Values")


# In[93]:


#residual vs fitted values
plt.scatter(pred,m_final.resid_pearson, c="r");plt.axhline(y=0,color="blue");plt.xlabel("Fitted Values");plt.ylabel("residual values")

