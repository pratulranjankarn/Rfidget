#!/usr/bin/env python
# coding: utf-8

# In[1]:
#


# importing required libraries
import obspy as ob
import wget
import math
from itertools import permutations
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates


# In[2]:


url='https://taps.earth.sinica.edu.tw/rfi/seed/RCEC7A/TW-RCEC7A_10-BNX.mseed'
#df = wget.download(url) # downloading the data from the url


# In[3]:


fl = ob.read('TW-RCEC7A_10-BNX.mseed') # reading the mseed file to an obspy stream object


# In[48]:


fl[0].stats # Trace stats

get_ipython().run_line_magic('matplotlib', 'qt # plotting in a matplotlib window')

# plotting the trace
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(fl[0].times("matplotlib"), fl[0].data, "b-")
ax.xaxis_date()
fig.autofmt_xdate()
plt.show()


# In[58]:


# Deciding the data samples to be used for permutation entropy calculation

d_time = int(480) # time spacing in seconds between start of each data sample
l_smpl = int(600) # length of each data sample in seconds
idx = np.arange(0,fl[0].stats.npts-l_smpl*fl[0].stats.sampling_rate,                 fl[0].stats.sampling_rate*d_time,dtype='int64') # indexes for permutation entropy calculation
n_iter = len(idx)
print(len(idx))


# In[63]:


# Function to calculate permutation entropy

def permuentropy(dta,m,L):
# data = dta
# embedding dimension of the time series = m
# time delay used for embedding = L

#If the user wants to find complexity variations in a time series
#it is recommended that m > 3 and m < 8, while the length of data
#should be > 5*m!

    # length of the data vector = dta_len
    dta_len = len(dta)
    # number of vectors in embedding space
    no_vector = dta_len - (m-1)*L
    # estimate factorial of m
    mm = math.factorial(m)
    # list of all permutations
    permlist = list(permutations(np.arange(0,m,1)))
    # creating dictionary of all permutations
    permdict={}
    for i in range(0,mm):
        permdict[str(permlist[i][:])] = i
    #print(permdict)
    # initialize pattern of permutations and other temp vectors to 0
    c = np.zeros((mm,1))
    # loop over number of vectors
    for i in range(0,no_vector):
        #continue
        idx = np.arange(i,i+m*L,L)
        # find coordimates of each point in m-space
        a = dta[idx]
        # sort co-ordinates in ascending order
        iv = np.argsort(a)
        # search dictionary for pattern and get the pattern index
        srchkey = str(tuple(iv))
        if srchkey in permdict:
            jj = permdict[srchkey]
        c[jj] = c[jj] + 1

    # calculate the relative frequency of c
    c = c/no_vector;
    pe = 0;
    # sum up and estimate permutation entropy
    pe = sum(-c[c!=0] * np.log(c[c!=0]))
    # alternate
        #for k in range(0,mm):
        #    if c[k] != 0:
        #        pe = pe - c[k]*(math.log(c[k]))
    
    # normalize permutation entropy by ln(m!)
    pe = pe/(math.log(mm))
    
    return pe,c


# In[64]:


# permutation entropy calculation
strt = datetime.now()
L = 2 # Sample delay used for embedding 
m = 5 # Embedding dimension
tmp = fl[0].data # Original data
pe = np.zeros((n_iter,1)) # Array to store pe values
time_x = np.empty((n_iter,1),dtype='datetime64[s]') # Array to store time values
for i in range(0,n_iter):
    dta = tmp[idx[i]:idx[i]+l_smpl*int(fl[0].stats.sampling_rate)] # Creating a data sample
    pe[i],c = permuentropy(dta,m,L) # calculating permutation entropy for the data sample
    time_x[i] = fl[0].stats.starttime +     (idx[i]+idx[i]+l_smpl*int(fl[0].stats.sampling_rate))     /(int(fl[0].stats.sampling_rate)*2) # Storing time corresponding to each data sample
print(datetime.now()-strt) # displaying running time of pe calculations


# In[62]:


# plotting the pe values with time

get_ipython().run_line_magic('matplotlib', 'qt')

fig, ax = plt.subplots()
ax.plot(time_x,pe)

