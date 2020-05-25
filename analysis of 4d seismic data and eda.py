#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load the data and the hystory curves

data=pd.read_hdf("init-init_0000_curves.hdf5")
hist_data=pd.read_hdf("init-init_0000_curves.hdf5").filter(regex='.+H..+')


# In[3]:


#look at the data
data.head()


# In[4]:


#find shape of the data
print(data.info())
data.describe()


# In[5]:


data.columns


# In[6]:


groups=[]
for i in range(0,465):
    groups.append(i)
values=data.values
fig,sub = plt.subplots(155,3)
plt.subplots_adjust(left=0,right=10,bottom=0,top=150,wspace=1, hspace=1)



for ax, i in zip(sub.flatten(),groups):
    ax.plot(values[:,i])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(data.columns[i])


# In[7]:


#so we understand a lot of the features are somehow correlated
plt.figure(figsize=(10,50))
sns.heatmap(data.corr())


# In[8]:


data.corr()


# In[9]:


data['WGOR:I-F-4']


# In[10]:


corr_df=data.corr().abs()
mask=np.triu(np.ones_like(corr_df,dtype=bool))
tri_df=corr_df.mask(mask)
to_drop=[c for c in tri_df.columns if any(tri_df[c]>0.8)]
print(to_drop)


# In[11]:


reduced_df=data.drop(to_drop,axis=1)


# In[12]:


reduced_df.shape


# In[13]:


groups=[]
for i in range(0,163):
    groups.append(i)
values=reduced_df.values
fig,sub = plt.subplots(35,5)
plt.subplots_adjust(left=0,right=10,bottom=0,top=150,wspace=1, hspace=1)



for ax, i in zip(sub.flatten(),groups):
    ax.plot(values[:,i])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(data.columns[i])


# In[14]:


reduced_df


# In[15]:


reduced_df['MSUMERR'].sum()
reduced_df.columns


# In[16]:


for cols in reduced_df.columns:
    if (reduced_df[cols].sum()==0.0):
        reduced_df=reduced_df.drop(cols,axis=1)


# In[17]:


reduced_df.shape


# In[48]:


groups=[]
for i in range(0,44):
    groups.append(i)
values=reduced_df.values
fig,sub = plt.subplots(22,2)
plt.subplots_adjust(left=0,right=12,bottom=0,top=160,wspace=1, hspace=1)
plt.grid(True)


for ax, i in zip(sub.flatten(),groups):
    ax.plot(values[:,i],color='b',alpha=0.8)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.grid()
    ax.set_title(reduced_df.columns[i],fontsize=60)


# In[19]:


reduced_df


# In[20]:


reduced_df=reduced_df.drop('MSUMCOMM',axis=1)


# In[21]:


reduced_df.shape


# In[22]:


reduced_df.corr()


# In[40]:


plt.figure(figsize=(50,50))
sns.heatmap(reduced_df.corr(),annot=True,cmap='coolwarm')


# In[24]:


reduced_df['WBHP:P-F-11B'].sum()


# In[25]:


def test_stationarity(timeseries,col):
    #Determing rolling statistics
    rolmean = timeseries[col].rolling(window=12).mean()
    rolstd = timeseries[col].rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries[col], color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title(col)
    plt.show(block=False)


# In[26]:


for cols in reduced_df.columns:
    test_stationarity(reduced_df,cols)


# In[27]:


reduced_df.columns


# In[28]:


from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose


# In[73]:


for col in reduced_df.columns:
    result=seasonal_decompose(reduced_df[col],model='additive',freq=365)
    result.plot()
    


# In[45]:


groups=[]
for i in range(0,44):
    groups.append(i)
values=data.values
fig,sub = plt.subplots(22,2)
plt.subplots_adjust(left=0,right=10,bottom=0,top=150,wspace=1, hspace=1)



for ax, i in zip(sub.flatten(),groups):
    ax.boxplot(values[:,i])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(reduced_df.columns[i],fontsize=60)


# In[71]:


#visualizing the autocorrelation
from statsmodels.graphics.tsaplots import plot_acf
for i in reduced_df.columns:
    plot_acf(reduced_df[i],title=i)


# In[ ]:





# In[ ]:





# In[ ]:




