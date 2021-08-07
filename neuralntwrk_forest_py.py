#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[24]:


forest = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Neural_network_py\\forestfires.csv")
forest.head(10)


# In[26]:


forest1 = forest.drop(columns = ['month','day'])
forest1.head(10)


# In[30]:


X = forest1.drop(columns = ['size_category'])
X


# In[32]:


Y = forest1['size_category']
Y


# In[33]:


#visualization
plt.hist(Y)


# In[34]:


#split data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)


# In[35]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)


# In[36]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[37]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30))


# In[38]:


mlp.fit(X_train,Y_train)


# In[40]:


prediction_test = mlp.predict(X_test)



# In[42]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(Y_test,prediction_test))
np.mean(Y_test==prediction_test)


