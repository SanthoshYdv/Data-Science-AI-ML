#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


wine = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\PCA_py\\wine.csv")
wine.head(10)


# In[13]:


wine.describe()


# In[19]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
#selecting data
wine.data = wine.iloc[:,1:]
wine.data.head(10)
WIN = wine.data.values


# In[21]:


#scaling/normalize data
wine_norm = scale(WIN)
wine_norm


# In[47]:


#PCA
pca = PCA(n_components = 3)
pca_values = pca.fit_transform(wine_norm)
pca_values


# In[53]:


var = pca.explained_variance_ratio_
var
var1= np.cumsum(np.round(var,4)*100)
var1


# In[54]:


#visualization
plt.plot(var, color = "red")


# In[67]:


#kmeans clustering using pca values
pca_values
#custom normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
df_norm = norm_func(pca_values)
df_norm


# In[73]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

k = list(range(2,15))
k
TWSS = [] 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = []
    for j in range(i):
        WSS.append(sum(cdist(df_norm[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[74]:


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# In[81]:


# Selecting 6 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=6) 
model.fit(df_norm)

model.labels_ 
md=pd.Series(model.labels_)   
wine['clust']=md  
wine
wine.groupby(wine.clust).mean()


# In[86]:


#clustering using original data
wine.head()
#custom normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
df_norm = norm_func(wine.iloc[:,1:])
df_norm


# In[88]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

k = list(range(2,15))
k
TWSS = []  
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[89]:


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# In[91]:


# Selecting 9 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=9) 
model.fit(df_norm)

model.labels_ 
md=pd.Series(model.labels_)   
wine['clust']=md  
wine
wine.groupby(wine.clust).mean()
#the number of clusters obtained from PCA and original data are different, 6 and 9 respectively.

