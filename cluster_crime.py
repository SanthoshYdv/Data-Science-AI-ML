#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[18]:


crime = pd.read_csv("C:/Users/santy/Desktop/Python/Python~Assignments/Clustering_py/crime_data.csv")
crime.head(10)


# In[19]:


#normalization function
def norm_func(i):
    x = (i-i.min())/(i.max() - i.min())
    return(x)
df_norm = norm_func(crime.iloc[:,1:])
df_norm.head(10)


# In[20]:


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
type(df_norm)


# In[24]:


#creating dendrograms
z = linkage(df_norm, method = "complete", metric = "euclidean")
plt.figure(figsize=(15,5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
z,
leaf_rotation = 0.,
leaf_font_size = 8.,
)
plt.show()


# In[50]:


#clusters
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = 'euclidean').fit(df_norm)
cluster_labels = pd.Series(h_complete.labels_)
cluster_labels.head()


# In[49]:


crime["cluster_complete"] = cluster_labels
crime["cluster_complete"].head()


# In[31]:


crime.head()


# In[51]:


crime.groupby(crime.cluster_complete).mean()


# In[37]:


#dendrogram using single linkage
z1 = linkage(df_norm, method = "single", metric = "euclidean")
plt.figure(figsize=(15,5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
z1,
leaf_rotation = 0.,
leaf_font_size = 8.,
)
plt.show()


# In[43]:


#clusters
from sklearn.cluster import AgglomerativeClustering
h1_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'single', affinity = 'euclidean').fit(df_norm)
cluster1_labels = pd.Series(h1_complete.labels_)
cluster1_labels.head()


# In[52]:


crime["cluster_single"] = cluster_labels
crime["cluster_single"].head()


# In[53]:


crime.groupby(crime.cluster_single).mean()


# In[42]:


#dendrogram using single average
z2 = linkage(df_norm, method = "average", metric = "euclidean")
plt.figure(figsize=(15,5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
z2,
leaf_rotation = 0.,
leaf_font_size = 8.,
)
plt.show()


# In[44]:


#clusters
from sklearn.cluster import AgglomerativeClustering
h2_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'average', affinity = 'euclidean').fit(df_norm)
cluster2_labels = pd.Series(h2_complete.labels_)
cluster2_labels.head()


# In[54]:


crime["cluster_average"] = cluster_labels
crime["cluster_average"].head()


# In[55]:


crime.groupby(crime.cluster_average).mean()


# In[57]:


crime.head(10)

