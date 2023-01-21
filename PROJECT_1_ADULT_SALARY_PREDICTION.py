#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing important libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  , OrdinalEncoder , OneHotEncoder , MinMaxScaler , StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib.inline', '')


# In[3]:


# importing the data set
train_data=pd.read_csv('adult_data.csv')


# In[8]:


train_data.head()


# In[9]:


train_data.isnull().sum()


# In[10]:


train_data.nunique()


# In[11]:


train_data.info()


# In[12]:


train_data.describe()


# In[14]:


train_data.shape


# In[19]:


def handle_capital_gain(df):
    df['capital_gain'] = np.where(df['capital_gain'] == 0, np.nan, df['capital_gain'])
    df['capital_gain'] = np.log(df['capital_gain'])
    df['capital_gain'] = df['capital_gain'].replace(np.nan, 0)


# In[20]:


train_data.columns = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
             'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'salary']


# In[21]:


print(train_data)


# In[22]:


handle_capital_gain(train_data)


# In[23]:


print(train_data)


# In[24]:


sns.displot(train_data['capital_gain'])
plt.show()


# # REMOVING OUTLIERS FROM HOURS PER WEEK
# 

# In[26]:


sns.displot(train_data['hours_per_week'])
plt.show()


# In[27]:


sns.boxplot(train_data['hours_per_week'])
plt.show()


# In[37]:


def remove_outlier_from_hours_per_week(train_data):
    IQR =train_data['hours_per_week'].quantile(.75)-train_data['hours_per_week'].quantile(.25)
    
    lower_range=train_data['hours_per_week'].quantile(.25)-(1.5 * IQR)
    upper_range=train_data['hours_per_week'].quantile(.75)+(1.5 * IQR)
    
    train_data.loc[train_data['hours_per_week'] <= lower_range, 'hours_per_week'] = lower_range
    train_data.loc[train_data['hours_per_week'] >= upper_range, 'hours_per_week'] = upper_range


# In[38]:


remove_outlier_from_hours_per_week(train_data)


# In[39]:


sns.boxplot(data=train_data['hours_per_week'])
plt.show()


# # REMOVE OUTLIERS FROM EDUCATION NUM
# 

# In[46]:


sns.distplot(train_data['education_num'])
plt.show()


# In[47]:


sns.boxplot(train_data['education_num'])
plt.show()


# In[49]:


def remove_outlier_education_num(train_data):
    IQR = train_data['education_num'].quantile(0.75) - train_data['education_num'].quantile(0.25)
    
    lower_range = train_data['education_num'].quantile(0.25) - (1.5 * IQR)
    upper_range = train_data['education_num'].quantile(0.75) + (1.5 * IQR)
    
    train_data.loc[train_data['education_num'] <= lower_range, 'education_num'] = lower_range
    train_data.loc[train_data['education_num'] >= upper_range, 'education_num'] = upper_range


# In[50]:


remove_outlier_education_num(train_data)


# In[51]:


sns.boxplot(data=train_data['education_num'])
plt.show()


# # REMOVE OUTLIERS FROM CAPITAL LOSS

# In[53]:


sns.displot(train_data['capital_loss'])
plt.show()


# In[54]:


sns.boxplot(data=train_data['capital_loss'])
plt.show()


# In[57]:


def capital_loss_log(train_data):
    train_data['capital_loss'] = np.where(train_data['capital_loss'] == 0, np.nan, train_data['capital_loss'])
    train_data['capital_loss'] = np.log(train_data['capital_loss'])
    train_data['capital_loss'] = train_data['capital_loss'].replace(np.nan, 0)


# In[58]:


capital_loss_log(train_data)


# In[59]:


sns.boxplot(data=train_data['capital_loss'])
plt.show()


# In[60]:


def remove_outliers_from_capital_loss(train_data):
    IQR = train_data['capital_loss'].quantile(.75)-train_data['capital_loss'].quantile(.25)
    
    lower_range=train_data['capital_loss'].quantile(.25)-(1.5 * IQR)
    upper_range=train_data['capital_loss'].quantile(.75)+(1.5 * IQR)
    
    train_data.loc[train_data['capital_loss']<=lower_range,'capital_loss']=lower_range
    train_data.loc[train_data['capital_loss']>=upper_range,'capital_loss']=upper_range


# In[61]:


remove_outliers_from_capital_loss(train_data)


# In[62]:


sns.boxplot(data=train_data['capital_loss'])
plt.show()


# # TRAIN TEST SPLIT

# In[64]:


X=train_data.iloc[:,:-1]
y=train_data.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42)


# In[65]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[66]:


# splitting the data into categorical and numerical data 
X_train_cat=X_train.select_dtypes(include='object')
X_train_num=X_train.select_dtypes(include=['int32','int64','float32','float64'])
X_test_cat=X_test.select_dtypes(include='object')
X_test_num=X_test.select_dtypes(include=['int32','int64','float32','float64'])


# In[67]:


print(X_train_cat)
print(X_test_cat)
print(X_train_num)
print(X_test_num)


# In[69]:


# preprocessing or encoding for categorical and numerical data
print(X_train.isnull().sum())
print(X_train_cat.isnull().sum())
print(X_train_num.isnull().sum())
print(X_test_cat.isnull().sum())
print(X_test_num.isnull().sum())


# # ORDINAL ENCODER

# In[70]:


# preprocesing ordinal encoder for categorical train and test data

OE=OrdinalEncoder()
OE.fit(X_train_cat)
X_train_cat_enc=OE.transform(X_train_cat)

OE.fit(X_test_cat)
X_test_cat_enc=OE.transform(X_test_cat)


# In[71]:


print(X_train_cat_enc)
print(X_test_cat_enc)


# # STANDARD SCALER

# In[73]:


# preprocessing on numerical test data

SS=StandardScaler()
SS.fit(X_train_num)
X_train_num_enc=SS.transform(X_train_num)

SS.fit(X_test_num)
X_test_num_enc=SS.transform(X_test_num)


# In[74]:


print(X_train_num_enc)
print(X_test_num_enc)


# # LABEL ENCODER

# In[75]:


LE=LabelEncoder()
LE.fit(y_train)
y_train_enc=LE.transform(y_train)

LE.fit(y_test)
y_test_enc=LE.transform(y_test)


# # CONCAT TRAIN AND TEST NUMERICAL AND CATEGORICAL DATA 

# In[77]:


X_train_cat_enc_df=pd.DataFrame(X_train_cat_enc)
X_train_num_enc_df=pd.DataFrame(X_train_num_enc)
X_train_final=pd.concat([X_train_cat_enc_df,X_train_num_enc_df],axis=1)


# In[78]:


X_test_cat_enc_df=pd.DataFrame(X_test_cat_enc)
X_test_num_enc_df=pd.DataFrame(X_test_num_enc)
X_test_final=pd.concat([X_test_cat_enc_df,X_test_num_enc_df],axis=1)


# # MODEL BUILDING AND PREDICTION

# In[82]:


model=LogisticRegression(solver='liblinear')
model.fit(X_train_final,y_train_enc)
y_pred=model.predict(X_test_final)
from sklearn.metrics import classification_report , accuracy_score , recall_score, f1_score , precision_score , confusion_matrix


# In[83]:


print(y_pred)


# # MODEL EVALUATION

# In[84]:


ACCURACY=accuracy_score(y_pred,y_test_enc)
print(ACCURACY)


# In[85]:


RECALL=recall_score(y_pred,y_test_enc)
print(RECALL)


# In[86]:


F1SCORE=f1_score(y_pred,y_test_enc)
print(F1SCORE)


# In[87]:


PRECISION=precision_score(y_pred,y_test_enc)
print(PRECISION)


# In[88]:


CLASSIFICATION=classification_report(y_pred,y_test_enc)
print(CLASSIFICATION)


# In[89]:


CONFUSION=confusion_matrix(y_pred,y_test_enc)
print(CONFUSION)


# In[ ]:




