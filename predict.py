#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import os
import pickle as pkl
import sklearn
import torch
import csv
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
import sys


# In[2]:


def read_files(directory):
    data = list()
    flag = True
    i = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory,filename)
        temp_data = pd.read_csv(filepath,sep='|')
        sick = 0
        if 1 in temp_data['SepsisLabel'].unique():
            sick = 1
            idx = temp_data['SepsisLabel'].idxmax()
            temp_data = temp_data.head(idx)
        temp_data=temp_data.drop(['SepsisLabel'],axis=1)
        temp_data['is_sick'] = sick
        i += 1
        data.append((temp_data,sick,filename[filename.find('_')+1:filename.find('.')]))
    return data


# In[3]:


database = pkl.load(open("database.pkl","rb"))
median_db = database.median()


# In[4]:


test_data = read_files(sys.argv[1])

# In[5]:


# Remove empty tables
empty_test_data = []
for test in test_data:
    if test[0].empty:
        empty_test_data.append((test[2],test[1]))
test_data = [x for x in test_data if not x[0].empty]


# In[6]:


for i in range(len(test_data)):
    test_data[i] = (test_data[i][0].ffill().bfill(),test_data[i][1],test_data[i][2]) # pottentially add limit=number inside the fill function


# In[7]:


### fill the missing test data with medians
for i in range(len(test_data)):
    temp = test_data[i][0]
    for col in test_data[0][0].columns:
        temp[col].fillna(median_db[col],inplace=True)
    test_data[i] = (temp,test_data[i][1],test_data[i][2])


# In[8]:


# Adding columns for test data
for i in range(len(test_data)):
    test_data[i][0]['HRESP'] = test_data[i][0]['HR']*test_data[i][0]['Resp']
    test_data[i][0]['NBTemp'] = test_data[i][0]['Temp']-37
    test_data[i][0]['HRTemp'] = test_data[i][0]['NBTemp']*test_data[i][0]['HR']
    test_data[i][0]['RespTemp'] = test_data[i][0]['NBTemp']*test_data[i][0]['Resp']
    test_data[i][0]['HRGlue'] = test_data[i][0]['HR']*test_data[i][0]['Glucose']
    test_data[i][0]['NBGlue'] = test_data[i][0]['NBTemp']*test_data[i][0]['Glucose']
    test_data[i][0]['RespGlue'] = test_data[i][0]['Resp']*test_data[i][0]['Glucose']


# In[9]:


### Averaging the test data, removing Unit2
avg_test_data = []
classification_test = []
id_test = []
for item in test_data:
    avg_test_data.append(item[0].drop(['is_sick','Unit2'],axis=1,inplace=False).mean().to_numpy())
    classification_test.append(item[1])
    id_test.append(int(item[2]))


# In[13]:


clf = pkl.load(open("model.pkl","rb"))
X = avg_test_data
y_pred = clf.predict(X)
y_pred = list(y_pred)
for item in empty_test_data:
    y_pred.append(1)
    id_test.append(int(item[0]))


# In[15]:


predictions = {"ids":id_test,"SepsisLabel":y_pred}


# In[16]:


# Check ids
pred_df = pd.DataFrame(predictions)


# In[17]:


# Check ids
pred_df.sort_values(["ids"],inplace=True)


# In[19]:


pred_df.to_csv("prediction.csv",index=False,header=False)

