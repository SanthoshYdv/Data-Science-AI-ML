#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


from mlxtend.frequent_patterns import apriori, association_rules
books = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Association_rules_py\\book.csv")


# In[9]:


books.head(10)


# In[16]:


frequent_itemsets = apriori(books, min_support = 0.005, max_len = 3, use_colnames = True)
frequent_itemsets


# In[17]:


#sorting
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets.sort_values


# In[18]:


#apply association rules on frequent itemsets
rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 1)
rules.head(10)


# In[19]:


rules.sort_values('lift', ascending = False).head(10)

