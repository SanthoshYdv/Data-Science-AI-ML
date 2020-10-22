#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


sales = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Forecasting\\PlasticSales.csv")
sales.head(10)


# In[5]:


sales.info()


# In[7]:


p = sales["Month"][0]
p[0:3]
sales['months']= 0

for i in range(60):
    p = sales["Month"][i]
    sales['months'][i]= p[0:3]
sales.head(10)


# In[8]:


#creating dummy variables for months
month_dummies = pd.DataFrame(pd.get_dummies(sales['months']))
sales1 = pd.concat([sales,month_dummies], axis=1)
sales1.head(10)


# In[12]:


sales1["t"] = np.arange(1,61)
sales1["t_squared"] = sales1["t"]*sales1["t"]
sales1["log_sales"] = np.log(sales1["Sales"])
sales1.head(5)


# In[13]:


train = sales1.head(40)
test = sales1.tail(20)


# In[14]:


###########################################linear model##############################################
import statsmodels.formula.api as smf
linear_model = smf.ols('Sales~t',data=train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# In[15]:


#######################exponential##############
Exp = smf.ols('log_sales~t',data=train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[16]:


#################### Quadratic #########################
Quad = smf.ols('Sales~t+t_squared',data=train).fit()
pred_Quad = pd.Series(Quad.predict(test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


# In[17]:


################### Additive seasonality ########################
add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[18]:


################# Additive Seasonality Quadratic ############################
add_sea_Quad = smf.ols('Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[19]:


################## Multiplicative Seasonality ##################
Mul_sea = smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[20]:


##################Multiplicative Additive Seasonality ###########
Mul_Add_sea = smf.ols('log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# In[21]:


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse


# In[22]:


predict_data = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Predict_new.csv")
model_full = smf.ols('log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()


# In[23]:


pred_new  = pd.Series(add_sea_Quad.predict(predict_data))
pred_new

