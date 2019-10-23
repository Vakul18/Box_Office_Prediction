#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np


# In[5]:


dataFrameTrain = pd.read_csv('../Data/tmdb-box-office-prediction/train.csv')


# In[6]:


dataFrameTrain.shape


# In[7]:


dataFrameTrain.head(10)


# In[9]:


dataFrameTrain['belongs_to_collection'] = np.where(dataFrameTrain['belongs_to_collection'].isnull(),0,1)


# In[10]:


dataFrameTrain['belongs_to_collection'].head(100)


# In[ ]:




