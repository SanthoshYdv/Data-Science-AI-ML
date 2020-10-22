#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


airline = pd.read_excel("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Clustering_py\\airlines.xlsx")
airline


# In[17]:


#custom normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
def_norm = norm_func(airline.iloc[:,1:11])
def_norm.head(10)


# In[14]:


#dendrogram using complete linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(def_norm, method = "complete", metric = "euclidean")
plt.figure(figsize=(15,5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
z,
leaf_rotation = 0.,
leaf_font_size = 8.,
)
plt.show()


# In[18]:


#clusters
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 5, linkage = "complete", affinity = "euclidean").fit(def_norm)
cluster_labels = pd.Series(h_complete.labels_)
cluster_labels.head


# In[19]:


airline["cluster_complete"]=cluster_labels
airline["cluster_complete"]


# In[21]:


airline.groupby(airline.cluster_complete).mean()


# In[22]:


#dendrogram using average linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z1 = linkage(def_norm, method = "average", metric = "euclidean")
plt.figure(figsize=(15,5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
z1,
leaf_rotation = 0.,
leaf_font_size = 8.,
)
plt.show()


# In[23]:


#clusters
from sklearn.cluster import AgglomerativeClustering
h_average = AgglomerativeClustering(n_clusters = 5, linkage = "average", affinity = "euclidean").fit(def_norm)
cluster_labels = pd.Series(h_complete.labels_)
cluster_labels.head


# In[24]:


airline["cluster_avg"]=cluster_labels
airline["cluster_avg"]


# In[25]:


airline.groupby(airline.cluster_avg).mean()


# In[26]:


#dendrogram using single linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z2 = linkage(def_norm, method = "single", metric = "euclidean")
plt.figure(figsize=(15,5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
z2,
leaf_rotation = 0.,
leaf_font_size = 8.,
)
plt.show()


# In[27]:


#clusters
from sklearn.cluster import AgglomerativeClustering
h_single = AgglomerativeClustering(n_clusters = 5, linkage = "single", affinity = "euclidean").fit(def_norm)
cluster_labels = pd.Series(h_complete.labels_)
cluster_labels.head


# In[28]:


airline["cluster_single"]=cluster_labels
airline["cluster_single"]


# In[29]:


airline.groupby(airline.cluster_single).mean()

