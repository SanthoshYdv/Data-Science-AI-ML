#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np


# In[4]:


airlines = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Forecasting\\Airlines_Data.csv")
airlines.head(10)


# In[5]:


airlines.info()


# In[8]:


p = airlines["Month"][0]
p[0:3]
airlines['months']= 0

for i in range(96):
    p = airlines["Month"][i]
    airlines['months'][i]= p[0:3]


# In[9]:


airlines.head(10)


# In[12]:


#creating dummy variables for months
month_dummies = pd.DataFrame(pd.get_dummies(airlines['months']))
airlines1 = pd.concat([airlines,month_dummies], axis=1)
airlines1.head(10)


# In[28]:


airlines1["t"] = np.arange(1,97)
airlines1["t_squared"] = airlines1["t"]*airlines1["t"]
airlines1["log_passenger"] = np.log(airlines["Passengers"])
airlines1.head(5)


# In[35]:


train = airlines1.head(66)
test = airlines1.tail(30)


# In[37]:


###########################################linear model##############################################
import statsmodels.formula.api as smf
linear_model = smf.ols('Passengers~t',data=train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


# In[39]:


#######################exponential##############
Exp = smf.ols('log_passenger~t',data=train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[41]:


#################### Quadratic #########################
Quad = smf.ols('Passengers~t+t_squared',data=train).fit()
pred_Quad = pd.Series(Quad.predict(test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad


# In[43]:


################### Additive seasonality ########################
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[44]:


################## Additive Seasonality Quadratic ############################
add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[46]:


################## Multiplicative Seasonality ##################
Mul_sea = smf.ols('log_passenger~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[47]:


##################Multiplicative Additive Seasonality ###########
Mul_Add_sea = smf.ols('log_passenger~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# In[48]:


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse


# In[50]:


predict_data = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Predict_new.csv")
model_full = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=airlines1).fit()


# In[51]:


pred_new  = pd.Series(add_sea_Quad.predict(predict_data))
pred_new


# In[ ]:




