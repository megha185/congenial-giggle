#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
from sklearn import linear_model


# In[3]:


df = pd.read_csv(r"C:\Users\Nawed\Desktop\INvideos.csv")


# In[4]:


df.head(3)


# In[7]:


df.info()


# In[8]:


# magic function to get the graph in this webpage only 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("views")
plt.ylabel("likes")
plt.scatter(df.views,df.likes,color="red",marker="^")


# In[9]:


# independent variables can be 1 column or multiple therefore pass it as an array
#here we pass 'views' as one of the parameters to predict 'likes' of the youtube video
x=df[['views']]
y=df.likes

reg = linear_model.LinearRegression()  
reg.fit(x,y)


# In[10]:


# to get the slope->'m' 
print(reg.coef_)

# to get intercept
print(reg.intercept_)


# In[12]:


# predicting the value for 100000 views
print(reg.predict([[100000]]))
#putting the value 100000 in 2d array  and passing it as parameter..otherwise error of 1d ar
print(float(0.02592917*100000+(-414.58581281314764)))


# In[13]:


print(reg.predict([[1000000]]))
print(float(0.02592917*1000000+(-414.58581281314764)))


# In[25]:


# predicting list of values and passing these as a parameter to our model that we created
df1=pd.read_csv(r"C:\Users\Nawed\Desktop\Views.csv")
df1


# In[26]:


q=reg.predict(df1)
q


# In[27]:


df1['Predicted_likes']=q
df1


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("comment_count")
plt.ylabel("likes")
plt.scatter(df.comment_count,df.likes,color="red",marker="^")


# In[30]:


# independent variables can be 1 column or multiple therefore pass it as an array
#here we pass 'comment_count' as one of the parameters to predict 'likes' of the youtube video
x=df[['comment_count']]
y=df.likes

reg1 = linear_model.LinearRegression()  
reg1.fit(x,y)


# In[31]:


# to get the slope->'m' 
print(reg1.coef_)

# to get intercept
print(reg1.intercept_)


# In[35]:


# predicting the value for 1000 comments
print(reg1.predict([[1000]]))
#putting the value 1000 in 2d array  and passing it as parameter..otherwise error of 1d ar
print(float(5.09964472*1000+(13430.981861634144)))


# In[44]:


# predicting list of values and passing these as a parameter to our model that we created
df2=pd.read_csv(r"C:\Users\Nawed\Desktop\Comment_count.csv")
df2


# In[45]:


s=reg.predict(df2)
s


# In[46]:


df2['Predicted_likes']=s
df2


# In[ ]:


## Implementing Mutli Linear Model


# In[48]:


# independent variables here are multiple , therefore we pass it as an array
x=df[['views','comment_count']]
y=df.likes

reg2 = linear_model.LinearRegression()  
reg2.fit(x,y)


# In[49]:


# to get the slope->'m' 
print(reg2.coef_)

# to get intercept
print(reg2.intercept_)


# In[51]:


# predicting the value for 100000 views and 1000 comments

print(reg2.predict([[100000,1000]]))
print(float(0.01811077*100000 +2.48410806*1000 +1226.69567215176))


# In[52]:


# predicting the value for 100000 views and 1000 comments

print(reg2.predict([[200000,1500]]))

print(float(0.01811077*200000 +2.48410806*1500 +1226.69567215176))


# In[53]:


# predicting list of values and passing these as a parameter to our model that we created
df3=pd.read_csv(r"C:\Users\Nawed\Desktop\Comments&views.csv")
df3


# In[55]:


t=reg2.predict(df3)
t


# In[56]:


df3['Predicted_likes']=t

df3

