#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns


# In[3]:


df=pd.read_csv('loan.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.boxplot(column='ApplicantIncome')


# In[8]:


df['ApplicantIncome'].hist (bins=20)


# In[9]:


df['CoapplicantIncome'].hist (bins=20)


# In[10]:


df.boxplot(column="ApplicantIncome", by='Education')


# In[11]:


pd.crosstab(df["Credit_History"], df['Loan_Status'], margins=True)


# In[12]:


df.boxplot(column='LoanAmount')


# In[13]:


df['LoanAmount'].hist(bins=20)


# In[14]:


df['LoanAmountlog']=np.log(df['LoanAmount'])
df['LoanAmountlog'].hist(bins=20)


# In[15]:


df.isnull().sum()


# In[16]:


df["Married"].fillna(df["Married"].mode()[0],inplace=True)
df["Gender"].fillna(df["Gender"].mode()[0],inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0],inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0],inplace=True) 
df["Credit_History"].fillna(df["Credit_History"].mode()[0],inplace=True)


# In[17]:


df.LoanAmount= df.LoanAmount.fillna(df.LoanAmount.mean())
df.LoanAmountlog= df.LoanAmountlog.fillna(df.LoanAmountlog.mean())


# In[18]:


df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace=True)


# In[19]:


df.isnull().sum()


# In[20]:


df["TotalIncome"]= df["ApplicantIncome"]+df["CoapplicantIncome"]
df["TotalIncomelog"]= np.log(df['TotalIncome'])


# In[21]:


df["TotalIncomelog"].hist(bins=20)


# In[22]:


df.head()


# In[23]:


X=df.iloc[:,np.r_[1:5,9:11,13:15]].values
Y=df.iloc[:,12].values


# In[24]:


X


# In[45]:


Y


# In[141]:


from sklearn.model_selection import train_test_split
X_train, X_test , Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)


# In[142]:


print(X_train)


# In[143]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X= LabelEncoder()


# In[144]:


for i in range(0,5):
    X_train[:,i]= labelencoder_X.fit_transform(X_train[:,i])


# In[145]:


X_train[:,7]= labelencoder_X.fit_transform(X_train[:,7])


# In[146]:


X_train


# In[147]:


labelencoder_Y=LabelEncoder() 
Y_train= labelencoder_Y.fit_transform(Y_train)


# In[148]:


Y_train


# In[149]:


for i in range(0,5):
    X_test[:,i]= labelencoder_X.fit_transform(X_test[:,i])


# In[150]:


X_test[:,7]= labelencoder_X.fit_transform(X_test[:,7])


# In[151]:


labelencoder_Y=LabelEncoder()
Y_test= labelencoder_Y.fit_transform(Y_test)


# In[152]:


Y_test


# In[183]:


from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)


# In[184]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(X_train,Y_train)


# In[185]:


Y_pred = DTClassifier.predict(X_test)
Y_pred


# In[186]:


from sklearn import metrics
print("The accuracy of Decision tree is: ",metrics.accuracy_score(Y_pred,Y_test))  


# In[187]:


from sklearn.naive_bayes import GaussianNB
NBClassifier= GaussianNB()
NBClassifier.fit(X_train,Y_train)


# In[188]:


Y_pred= NBClassifier.predict(X_test)


# In[189]:


Y_pred


# In[190]:


print('The Accuracy of Naive Bayes is: ',metrics.accuracy_score(Y_pred,Y_test))


# In[191]:


testing_data= pd.read_csv("testdata.csv")


# In[192]:


testing_data.head()


# In[193]:


testing_data.info()


# In[194]:


testing_data.isnull().sum()


# In[195]:


testing_data["Married"].fillna(df["Married"].mode()[0],inplace=True)
testing_data["Gender"].fillna(df["Gender"].mode()[0],inplace=True)
testing_data["Self_Employed"].fillna(df["Self_Employed"].mode()[0],inplace=True)
testing_data["Dependents"].fillna(df["Dependents"].mode()[0],inplace=True)
testing_data["Credit_History"].fillna(df["Credit_History"].mode()[0],inplace=True)


# In[196]:


testing_data["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace=True)


# In[197]:


testing_data.LoanAmount= testing_data.LoanAmount.fillna(testing_data.LoanAmount.mean())
testing_data['LoanAmountlog']= np.log(testing_data['LoanAmount'])


# In[198]:


testing_data.isnull().sum()


# In[199]:


testing_data.boxplot(column='ApplicantIncome')


# In[200]:


testing_data.boxplot(column='LoanAmount')


# In[201]:


testing_data["TotalIncome"]= testing_data["ApplicantIncome"]+testing_data["CoapplicantIncome"]
testing_data["TotalIncomelog"]= np.log(testing_data['TotalIncome'])


# In[202]:


testing_data.head()


# In[203]:


test = testing_data.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[204]:


for i in range(0,5):
    test[:,i]= labelencoder_X.fit_transform(test[:,i])


# In[205]:


test[:,7]= labelencoder_X.fit_transform(test[:,7])


# In[206]:


test


# In[207]:


test= ss.fit_transform(test)


# In[208]:


pred= NBClassifier.predict(test)


# In[209]:


pred


# In[ ]:




