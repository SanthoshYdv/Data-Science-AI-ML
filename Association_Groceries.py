#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[33]:


from mlxtend.frequent_patterns import apriori, association_rules
groceries = []
with open("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Association_rules_py\\groceries.csv") as f:
    groceries = f.read()


# In[34]:


#formatting dataset
groceries = groceries.split("\n")
groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))


# In[39]:


all_groceries_list = [i for item in groceries_list for i in item]
all_groceries_list


# In[40]:


from collections import Counter, OrderedDict
item_frequencies = Counter(all_groceries_list)
item_frequencies


# In[44]:


#sorting
item_frequencies = sorted(item_frequencies, key = lambda x:x[1])
item_frequencies


# In[46]:


frequencies = list(reversed([i[1] for i in item_frequencies]))
frequencies


# In[48]:


items = list(reversed([i[0] for i in item_frequencies]))
items


# In[49]:


#visualization
plt.bar (x=list(range(0,11)), height=frequencies[0:11],color='rgbkymc');
plt.xticks(list(range(0,11)), items[0:11]);
plt.xlabel('Items');
plt.ylabel('Count')


# In[52]:


groceries_series = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:]
groceries_series.columns = ["transactions"]
groceries_series.columns


# In[53]:


x = groceries_series['transactions'].str.join(sep="*").str.get_dummies(sep="*")
frequent_itemsets = apriori(x, min_support = 0.005, max_len = 3, use_colnames = True)
frequent_itemsets


# In[54]:


#sorting
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets.sort_values


# In[56]:


#apply association rules on frequent itemsets
rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 1)
rules.head(10)


# In[57]:


rules.sort_values('lift', ascending = False).head(10)

