#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Linear Regression 


# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


df=pd.read_csv('USA_Housing.csv')


# In[37]:


df


# In[38]:


df.head()


# In[39]:


df.isnull()


# In[40]:


len(df)


# In[41]:


df.describe()


# In[42]:


df.columns


# In[43]:


df['Avg. Area Income'].mean()


# In[44]:


df['Avg. Area House Age'].mean()


# In[45]:


df['Avg. Area Number of Rooms'].mean()


# In[46]:


df.plot()


# In[47]:


sns.distplot(df['Avg. Area Income'])


# In[48]:


sns.jointplot(df)


# In[49]:


sns.scatterplot(x='Avg. Area Income',y='Area Population',data=df,sizes=20)


# In[50]:


df.boxplot('Avg. Area Income',)


# In[51]:


sns.pairplot(df)


# In[52]:


df.corr


# In[67]:


df.columns


# In[69]:


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[70]:


X


# In[71]:


y=df['Price']


# In[72]:


y


# In[73]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=10)


# In[74]:


from sklearn.linear_model import LinearRegression


# In[75]:


lm=LinearRegression()


# In[76]:


lm.fit(X_train,y_train)


# In[77]:


print(lm.intercept_)


# In[83]:


lm.coef_


# In[86]:


X_train.columns


# In[87]:


pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])


# In[88]:


df.head()


# In[92]:


predictions=lm.predict(X_test)


# In[93]:


predictions


# In[98]:


plt.scatter(y_test,predictions,)


# In[102]:


sns.distplot((y_test-predictions))


# In[103]:


from sklearn import metrics


# In[110]:


print('MAE:',metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[117]:


sns.jointplot((y_test-predictions))


# In[122]:


sns.jointplot((y_test+predictions))


# In[124]:


df.plot()


# In[129]:


sns.stripplot((y_test-predictions))


# In[ ]:




