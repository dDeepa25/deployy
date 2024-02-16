#!/usr/bin/env python
# coding: utf-8

# # Loading important libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Importing the Bankrupt Dataset

# In[2]:


bankrupt = pd.read_csv("/media/gargi/Data/DataSets/bankruptcy-prevention.csv", sep = ';', header = 0)
bankrupt


# In[3]:


bankrupt.describe()


# In[4]:


print(bankrupt.info())
print(bankrupt.shape)


# ## checking is there any missing values are there in data or not

# In[5]:


bankrupt.isnull().sum()


# In[6]:


bankrupt_new = bankrupt.iloc[:,:]
bankrupt_new


# In[7]:


bankrupt_new["class_yn"] = 1
bankrupt_new


# ## Here we are changing  the target variable to bankruptcy = 0, non-bankruptcy = 1

# In[8]:


bankrupt_new.loc[bankrupt[' class'] == 'bankruptcy', 'class_yn'] = 0


# In[9]:


bankrupt_new


# In[10]:


bankrupt_new.drop(' class', inplace = True, axis =1)
bankrupt_new.head()


# # Exploratory Data Analysis (EDA)
# 

# In[11]:


bankrupt_new.corr()


# In[12]:


sns.heatmap(bankrupt_new.corr(), vmin = -1, vmax = 1, annot = True)


# In[13]:


sns.countplot(x = 'class_yn', data = bankrupt_new, palette = 'hls')


# In[14]:


sns.countplot(x = ' financial_flexibility', data = bankrupt_new, palette = 'hls')


# In[15]:


# for visualization 

pd.crosstab(bankrupt.class_yn, bankrupt.industrial_risk).plot(kind='bar')


# In[16]:


bankrupt_new.columns


# In[17]:


pd.crosstab(bankrupt_new[' financial_flexibility'], bankrupt_new['class_yn']).plot(kind = 'bar')


# In[18]:


pd.crosstab(bankrupt_new[' credibility'], bankrupt_new.class_yn).plot(kind = 'bar')


# In[19]:


pd.crosstab(bankrupt_new[' operating_risk'], bankrupt_new.class_yn).plot(kind='bar')


# In[20]:


pd.crosstab(bankrupt_new[' financial_flexibility'], bankrupt_new[' credibility']).plot(kind = 'bar')


# In[21]:


np.shape(bankrupt_new)


# In[22]:


# Input
x = bankrupt_new.iloc[:,:-1]

# Target variable

y = bankrupt_new.iloc[:,-1]


# In[23]:


from sklearn.model_selection import train_test_split # trian and test
from sklearn import metrics
from sklearn import preprocessing 
from sklearn.metrics import classification_report


# ### Sliptting the data into train and test
# 

# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)


# # Preparing models
# ## 1. Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression

logisticlassifier = LogisticRegression() 


# In[26]:


logisticlassifier.fit(x_train, y_train)

logisticlassifier.coef_ # coefficients of features


# ###  After the traing the model then we prediction on test data
# 

# In[27]:


y_pred = logisticlassifier.predict(x_test)
y_pred


# ### let's test the performance of our model - confusion matrix
# 

# In[28]:


from sklearn.metrics import confusion_matrix

confusion_logist = confusion_matrix(y_test, y_pred)

confusion_logist


# ###  Accuracy of a Model

# In[29]:


# Train Accuracy

train_acc_logist = np.mean(logisticlassifier.predict(x_train)== y_train)
train_acc_logist


# In[30]:


# Test Accuracy

test_acc_logist = np.mean(logisticlassifier.predict(x_test)== y_test)
test_acc_logist


# In[31]:


from sklearn.metrics import accuracy_score

logistic_acc = accuracy_score(y_test, y_pred)
logistic_acc


# ### Accuracy of overall model

# In[32]:


logisticlassifier.fit(x, y)

logisticlassifier.coef_ # coefficients of features


# In[33]:


y_pred = logisticlassifier.predict(x)

confusion_matrix = confusion_matrix(y, y_pred)
confusion_matrix


# In[34]:


acc = accuracy_score(y, y_pred)
acc


# In[35]:


logisticlassifier.score(x_test, y_test)


# In[36]:


logisticlassifier.score(x_train, y_train)


# ### From the accuracy we can say that the model is overfitted to avoid overfit problem we use Regularozation method
# #### here we have L1, L2 regularization
# ##### It turns out they have different but equally useful properties. From a practical standpoint, L1 tends to shrink coefficients to zero whereas L2 tends to shrink coefficients evenly. L1 is therefore useful for feature selection, as we can drop any variables associated with coefficients that go to zero.
# #### L1 = lasso regularization

# In[37]:


from sklearn import linear_model

lasso_reg = linear_model.Lasso(alpha = 50, max_iter = 100, tol =0.1)

lasso_reg.fit(x_train, y_train)


# In[38]:


lasso_reg.score(x_test, y_test)


# In[39]:


lasso_reg.score(x_train, y_train)


# #### L2 = Ridge regularization 

# In[40]:


from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha = 50, max_iter = 100, tol = 0.1)

ridge_reg.fit(x_train, y_train)


# In[41]:


ridge_reg.score(x_test, y_test)


# In[42]:


ridge_reg.score(x_train, y_train)


# # 2. KNN model

# In[43]:


from sklearn.neighbors import KNeighborsClassifier as KNC
import warnings
warnings.filterwarnings('ignore')


# ### To choose k value 

# In[44]:


import math
math.sqrt(len(y_test))


# Here we are choosing the k value to be  7 (choosing odd value)
# 
# Define the model KNN and fit model

# In[45]:


KNN_classifier = KNC(n_neighbors =7, p = 2, metric = 'euclidean')


# In[46]:


KNN_classifier.fit(x_train, y_train)


# #### Predict the Test set results

# In[47]:


y_pred = KNN_classifier.predict(x_test)
y_pred


# #### Evaluate model

# In[48]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[49]:


from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred))


# #### Accuracy of KNN model
# 

# In[50]:


from sklearn.metrics import accuracy_score

KNN_acc = accuracy_score(y_test, y_pred)
KNN_acc


# # 3. Naive Bayes Classifier

# In[51]:


from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB


# #### Creating GaussianNB and MultinomialNB functions
# 

# In[52]:


GNB = GaussianNB()
MNB = MultinomialNB()


# #### Building the model with GaussianNB
# 

# In[53]:


Naive_GNB = GNB.fit(x_train ,y_train)

y_pred = Naive_GNB.predict(x_test)
y_pred


# #### Evaluate Model

# In[54]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# #### Accuracy of GNB

# In[55]:


from sklearn.metrics import accuracy_score

GNB_acc = accuracy_score(y_test , y_pred)
GNB_acc


# #### Building the model with MultinomialNB
# 

# In[56]:


Naive_MNB = MNB.fit(x_train ,y_train)

y_pred = Naive_MNB.predict(x_test)
y_pred


# #### Evaluating Model
# 

# In[57]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# #### Accuracy of MNB

# In[58]:


from sklearn.metrics import accuracy_score

MNB_acc = accuracy_score(y_test , y_pred)
MNB_acc


# # 4. Support Vector Machine

# In[59]:


from sklearn.svm import SVC


# #### Kernel = Linear model

# In[60]:


model_linear = SVC(kernel = 'linear')

model_linear.fit(x_train, y_train)

pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test)


# #### Kernel = ploy model

# In[61]:


model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) # Accuracy


# #### Kernel = 'rbf' model --> Radial Basis Function 

# In[62]:


model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) # Accuracy


# from the above kernels in SVM polynomial kernel giving good accuracy
# 

# In[65]:


import pickle
pickle_out = open("model_poly.pkl","wb")
pickle.dump(model_poly, pickle_out)
pickle_out.close()


# In[ ]:




