#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn


# In[2]:


df=pd.read_csv('C:\\Users\\prudhvija\\OneDrive\\Desktop\\ASA All PGA Raw Data - Tourn Level.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


df.isnull().sum()


# In[9]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[10]:


df.sg_putt.median()


# In[11]:


def impute_nan(df,variable,median):
    df[variable+'_median']=df[variable].fillna(median)


# In[12]:


median=df.sg_putt.median()
median


# In[13]:


impute_nan(df,'sg_putt',median)
df.head()


# In[14]:


df['sg_putt_median'].isnull().sum()


# In[15]:


print(df['sg_putt'].std())
print(df['sg_putt_median'].std())


# In[16]:


fig=plt.figure()
ax=fig.add_subplot(111)
df['sg_putt'].plot(kind='kde',ax=ax)
df['sg_putt_median'].plot(kind='kde',ax=ax,color='blue')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')


# In[17]:


df.pos.median()


# In[18]:


def impute_nan(df,variable,median):
    df[variable+'_median']=df[variable].fillna(median)


# In[19]:


median=df.pos.median()
median


# In[20]:


impute_nan(df,'pos',median)
df.head()


# In[21]:


df['pos_median'].isnull().sum()


# In[22]:


print(df['pos'].std())
print(df['pos'].std())


# In[23]:


fig=plt.figure()
ax=fig.add_subplot(111)
df['pos'].plot(kind='kde',ax=ax)
df['pos_median'].plot(kind='kde',ax=ax,color='blue')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')


# In[24]:


df.sg_arg.median()


# In[25]:


def impute_nan(df,variable,median):
    df[variable+'_median']=df[variable].fillna(median)


# In[26]:


median=df.sg_arg.median()
median


# In[27]:


impute_nan(df,'sg_arg',median)
df.head()


# In[28]:


df['sg_arg_median'].isnull().sum()


# In[29]:


print(df['sg_arg'].std())
print(df['sg_arg_median'].std())


# In[30]:


fig=plt.figure()
ax=fig.add_subplot(111)
df['sg_arg'].plot(kind='kde',ax=ax)
df['sg_arg_median'].plot(kind='kde',ax=ax,color='red')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')


# In[31]:


df.sg_app.median()


# In[32]:


def impute_nan(df,variable,median):
    df[variable+'_median']=df[variable].fillna(median)


# In[33]:


median=df.sg_app.median()
median


# In[34]:


impute_nan(df,'sg_app',median)
df.head()


# In[35]:


df['sg_app_median'].isnull().sum()


# In[36]:


print(df['sg_app'].std())
print(df['sg_app_median'].std())


# In[37]:


fig=plt.figure()
ax=fig.add_subplot(111)
df['sg_arg'].plot(kind='kde',ax=ax)
df['sg_arg_median'].plot(kind='kde',ax=ax,color='pink')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')


# In[38]:


df.sg_ott.median()


# In[39]:


def impute_nan(df,variable,median):
    df[variable+'_median']=df[variable].fillna(median)


# In[40]:


median=df.sg_ott.median()
median


# In[41]:


impute_nan(df,'sg_ott',median)
df.head()


# In[42]:


df['sg_ott_median'].isnull().sum()


# In[43]:


print(df['sg_ott'].std())
print(df['sg_ott_median'].std())


# In[44]:


fig=plt.figure()
ax=fig.add_subplot(111)
df['sg_arg'].plot(kind='kde',ax=ax)
df['sg_arg_median'].plot(kind='kde',ax=ax,color='green')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')


# In[45]:


df.sg_t2g.median()


# In[46]:


def impute_nan(df,variable,median):
    df[variable+'_median']=df[variable].fillna(median)


# In[47]:


median=df.sg_t2g.median()
median


# In[48]:


impute_nan(df,'sg_t2g',median)
df.head()


# In[49]:


df['sg_t2g_median'].isnull().sum()


# In[50]:


print(df['sg_t2g'].std())
print(df['sg_t2g_median'].std())


# In[51]:


fig=plt.figure()
ax=fig.add_subplot(111)
df['sg_arg'].plot(kind='kde',ax=ax)
df['sg_arg_median'].plot(kind='kde',ax=ax,color='yellow')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')


# In[52]:


df.sg_total.median()


# In[53]:


def impute_nan(df,variable,median):
    df[variable+'_median']=df[variable].fillna(median)


# In[54]:


median=df.sg_total.median()
median


# In[55]:


impute_nan(df,'sg_total',median)
df.head()


# In[56]:


df['sg_total_median'].isnull().sum()


# In[57]:


print(df['sg_total'].std())
print(df['sg_total_median'].std())


# In[58]:


fig=plt.figure()
ax=fig.add_subplot(111)
df['sg_arg'].plot(kind='kde',ax=ax)
df['sg_arg_median'].plot(kind='kde',ax=ax,color='purple')
lines,labels=ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')


# In[59]:


df.value_counts()


# In[60]:


course=df.course.value_counts().index


# In[61]:


course_val=df.course.value_counts().values


# In[62]:


plt.pie(course_val[:5],labels=course[:5])     #top 5 courses


# In[63]:


plt.pie(course_val[:5],labels=course[:5],autopct='%1.2f%%')


# observation

# course maximum is Pebble Beach Resort - Pebble Beach, CA,with 21.74
# next is maximum is Torrey Pines North - La Jolla, CA' and 'La Quinta CC - La Quinta, CA' with 19.98%
# Muirfield Village Golf Club - Dublin, OH with 19.19 
# and Sea Island Resort - Sea Island, GA' with 19.11%
# 

# In[64]:


df.date=pd.to_datetime(df.date)


# In[65]:


df.dtypes


# In[66]:


df.drop('pos',axis=1,inplace=True)


# In[67]:


df.drop('Unnamed: 2',axis=1,inplace=True)


# In[68]:


df.drop('Unnamed: 3',axis=1,inplace=True)


# In[69]:


df.drop('Unnamed: 4',axis=1,inplace=True)


# In[70]:


df.drop('sg_putt',axis=1,inplace=True)


# In[71]:


df.drop('sg_arg',axis=1,inplace=True)


# In[72]:


df.drop('sg_app',axis=1,inplace=True)


# In[73]:


df.drop('sg_ott',axis=1,inplace=True)


# In[74]:


df.drop('sg_t2g',axis=1,inplace=True)


# In[75]:


df.drop('sg_total',axis=1,inplace=True)


# In[76]:


df.drop('tournament name',axis=1,inplace=True)


# In[77]:


df.drop('Player_initial_last',axis=1,inplace=True)


# In[78]:


df.drop('player',axis=1,inplace=True)


# In[79]:


df.info()


# In[80]:


df.shape


# In[81]:


df['course'].value_counts().sort_values(ascending=False).head(30)


# In[82]:


top_60=[x for x in df.course.value_counts().sort_values(ascending=False).head(60).index]
top_60


# In[83]:


for label in top_60:
    df[label]=np.where(df['course']==label,1,0)
df[['course']+top_60].head(40)


# In[84]:


sns.distplot(df['sg_putt_median'])


# In[85]:


X=df.iloc[:,:3]
y=df['hole_par']


# In[86]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[87]:


ordered_rank_features=SelectKBest(score_func=chi2,k='all')
ordered_feature=ordered_rank_features.fit(X,y)


# In[88]:


dfscores=pd.DataFrame(ordered_feature.scores_,columns=["score"])
dfcolumns=pd.DataFrame(X.columns)


# In[89]:


features_rank=pd.concat([dfcolumns,dfscores],axis=1)


# In[90]:


features_rank.nlargest(3,'score')


# In[91]:


df.corr()


# In[92]:


corr=df.corr()


# In[93]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# In[94]:


threshold=0.8


# In[95]:


def correlation(df,threshold):
    col_corr=set()
    corr_matrix=df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
               colname=corr_matrix.columns[i]
               col_corr.add(colname)
    return col_corr


# In[96]:


correlation(df.iloc[:,:-1],threshold)


# In[ ]:




