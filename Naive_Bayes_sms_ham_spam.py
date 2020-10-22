#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# In[4]:


sms = pd.read_csv("C:\\Users\\santy\\Desktop\\Python\\Python~Assignments\\Naive_Bayes_py\\sms_raw_NB.csv",encoding = "ISO-8859-1")
sms.head()


# In[6]:


# cleaning data 
import re
stop_words = []
with open("C:\\Users\\santy\\Desktop\\Python\\stop.txt") as f:
    stop_words = f.read()
stop_words = stop_words.split("\n")


# In[11]:


def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))
sms.text = sms.text.apply(cleaning_text)


# In[12]:


#remove empty rows
sms.shape
sms = sms.loc[sms.text != " ",:]


# In[13]:


def split_into_words(i):
    return [word for word in i.split(" ")]


# #splitting data 
# from sklearn.model_selection import train_test_split
# sms_train,sms_test = train_test_split(sms,test_size=0.3)

# In[15]:


# Preparing email texts into word count matrix format 
sms_wdm = CountVectorizer(analyzer=split_into_words).fit(sms.text)


# In[17]:


# For all messages
all_sms_matrix = sms_wdm.transform(sms.text)
all_sms_matrix.shape


# In[20]:


# For training messages
train_sms_matrix = sms_wdm.transform(sms_train.text)
train_sms_matrix.shape


# In[22]:


# For testing messages
test_sms_matrix = sms_wdm.transform(sms_test.text)
test_sms_matrix.shape 


# In[23]:


from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB


# In[28]:


# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_sms_matrix,sms_train.type)
train_pred_mul = classifier_mb.predict(train_sms_matrix)
accuracy_train_mul = np.mean(train_pred_m==sms_train.type)
accuracy_train_mul


# In[29]:


test_pred_mul = classifier_mb.predict(test_sms_matrix)
accuracy_test_mul = np.mean(test_pred_m==sms_test.type)
accuracy_test_mul


# In[31]:


# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_sms_matrix.toarray(),sms_train.type.values)
train_pred_gnb = classifier_gb.predict(train_sms_matrix.toarray())
accuracy_train_gnb = np.mean(train_pred_gnb==sms_train.type)
accuracy_train_gnb 


# In[33]:


test_pred_gnb = classifier_gb.predict(test_sms_matrix.toarray())
accuracy_test_gnb = np.mean(test_pred_gnb==sms_test.type)
accuracy_test_gnb

