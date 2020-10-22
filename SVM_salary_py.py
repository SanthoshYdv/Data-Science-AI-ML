#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


salary_train = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Support_Vector_Machines_py\\SalaryData_Train(1).csv")
salary_test = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Support_Vector_Machines_py\\SalaryData_Test(1).csv")


# In[5]:


salary_train.head()


# In[6]:


salary_test.head()


# In[22]:


salary_train1 = salary_train.drop(columns = ["educationno","maritalstatus","relationship","race","native","capitalgain","capitalloss"])


# In[23]:


salary_test1 = salary_test.drop(columns = ["educationno","maritalstatus","relationship","race","native","capitalgain","capitalloss"])


# In[7]:


salary_train["Salary"].isna().value_counts()
salary_train.dropna(subset = ["Salary"], inplace=True)
salary_train["Salary"] = salary_train["Salary"].fillna(0)


# In[8]:


salary_test["Salary"].isna().value_counts()
salary_test["Salary"] = salary_test["Salary"].fillna(0)


# In[25]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salary_train1["Salary"] = le.fit_transform(salary_train1.Salary)
salary_train1["Salary"].value_counts()


# In[26]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salary_test1["Salary"] = le.fit_transform(salary_test1.Salary)
salary_test1["Salary"].value_counts()


# In[27]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salary_train1["workclass"] = le.fit_transform(salary_train1.workclass)
salary_train1["workclass"].value_counts()


# In[28]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salary_test1["workclass"] = le.fit_transform(salary_test1.workclass)
salary_test1["workclass"].value_counts()


# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
salary_train1["education"] = le.fit_transform(salary_train1.education)
salary_train1["education"].value_counts()


# In[30]:


le = LabelEncoder()
salary_test1["education"] = le.fit_transform(salary_test1.education)
salary_test1["education"].value_counts()


# In[31]:


le = LabelEncoder()
salary_train1["occupation"] = le.fit_transform(salary_train1.occupation)
salary_train1["occupation"].value_counts()


# In[32]:


le = LabelEncoder()
salary_test1["occupation"] = le.fit_transform(salary_test1.occupation)
salary_test1["occupation"].value_counts()


# In[33]:


le = LabelEncoder()
salary_train1["sex"] = le.fit_transform(salary_train1.sex)
salary_train1["sex"].value_counts()


# In[34]:


le = LabelEncoder()
salary_test1["sex"] = le.fit_transform(salary_test1.sex)
salary_test1["sex"].value_counts()


# In[35]:


salary_train1.head()


# In[36]:


salary_test1.head()


# In[38]:


import seaborn as sns
sns.boxplot(x="Salary",y="age", data = salary_train, palette = "hls")


# In[41]:


sns.boxplot(x="Salary",y="education", data = salary_train1, palette = "hls")


# In[43]:


train_x = salary_train1.iloc[:,0:6]
train_y = salary_train1.iloc[:,6]
test_x = salary_test1.iloc[:,0:6]
test_y = salary_test1.iloc[:,6]


# In[44]:


from sklearn.svm import SVC
model_linear = SVC(kernel = "linear")
model_linear.fit(train_x,train_y)


# In[45]:


pred_linear = model_linear.predict(test_x)
np.mean(pred_linear==test_y)


# In[46]:


model_poly = SVC(kernel = "poly")
model_poly.fit(train_x,train_y)


# In[47]:


pred_poly = model_poly.predict(test_x)
np.mean(pred_poly==test_y)


# In[48]:


model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_x,train_y)


# In[49]:


pred_rbf = model_rbf.predict(test_x)
np.mean(pred_rbf==test_y)

