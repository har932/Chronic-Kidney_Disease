#!/usr/bin/env python
# coding: utf-8

# In[25]:


#Importing Libraries
import pandas as pd
import numpy as np
import pickle


# In[26]:


#Reading dataset
dataset = pd.read_csv("Kidney_data.csv")


# In[27]:


#displaying dataset
dataset


# In[28]:


#dropping column id
dataset = dataset.drop('id', axis=1)


# In[29]:


# Replacing Categorical Values with Numericals
dataset['rbc'] = dataset['rbc'].replace(to_replace = {'normal' : 0, 'abnormal' : 1})
dataset['pc'] = dataset['pc'].replace(to_replace = {'normal' : 0, 'abnormal' : 1})
dataset['pcc'] = dataset['pcc'].replace(to_replace = {'notpresent':0,'present':1})
dataset['ba'] = dataset['ba'].replace(to_replace = {'notpresent':0,'present':1})
dataset['htn'] = dataset['htn'].replace(to_replace = {'yes' : 1, 'no' : 0})

dataset['dm'] = dataset['dm'].replace(to_replace = {'\tyes':'yes', ' yes':'yes', '\tno':'no'})
dataset['dm'] = dataset['dm'].replace(to_replace = {'yes' : 1, 'no' : 0})

dataset['cad'] = dataset['cad'].replace(to_replace = {'\tno':'no'})
dataset['cad'] = dataset['cad'].replace(to_replace = {'yes' : 1, 'no' : 0})

dataset['appet'] = dataset['appet'].replace(to_replace={'good':1,'poor':0,'no':np.nan})
dataset['pe'] = dataset['pe'].replace(to_replace = {'yes' : 1, 'no' : 0})
dataset['ane'] = dataset['ane'].replace(to_replace = {'yes' : 1, 'no' : 0})

dataset['classification'] = dataset['classification'].replace(to_replace={'ckd\t':'ckd'})
dataset["classification"] = [1 if i == "ckd" else 0 for i in dataset["classification"]]


# In[30]:


# Coverting Objective into Numericals:
dataset['pcv'] = pd.to_numeric(dataset['pcv'], errors='coerce')
dataset['wc'] = pd.to_numeric(dataset['wc'], errors='coerce')
dataset['rc'] = pd.to_numeric(dataset['rc'], errors='coerce')


# In[31]:


# Handling Missing Values:
features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
           'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
           'appet', 'pe', 'ane']
for feature in features:
    dataset[feature] = dataset[feature].fillna(dataset[feature].median())


# In[32]:


# Dropping feature (Multicollinearity):
dataset.drop('pcv', axis=1, inplace=True)


# In[33]:


import seaborn as sns 
import matplotlib.pyplot as plt


# In[34]:


# Target feature:
sns.countplot(dataset['classification'])


# In[35]:


# Feature Importance:
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X,y)

plt.figure(figsize=(8,6))
ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(24).plot(kind='barh')
plt.show()


# In[36]:


plt.figure(figsize=(24,14))
sns.heatmap(dataset.corr(), annot=True, cmap='YlGnBu')
plt.show()


# In[37]:


# Independent and Dependent Feature:
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# In[38]:


# After feature importance:
X = dataset[['sg', 'htn', 'hemo', 'dm', 'al', 'appet', 'rc', 'pc']]


# In[39]:


# Train Test Split:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=33)


# In[40]:


# RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)
RandomForest


# In[41]:


y_pred = RandomForest.predict(X_test)
y_pred


# In[42]:


from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test,y_pred))


# In[43]:


filename = 'Kidney.pkl'
pickle.dump(RandomForest, open(filename, 'wb'))


# In[44]:


X


# In[45]:


y


# In[47]:


input_data=  (1.025,0.0,15.8,0.0,0.0,1.0,6.1,0.0)     
# Coverting into data to numpy array so as to avoid reshape error :
input_data_as_numpy_array = np.asarray(input_data)

# Reshapping the array :
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Interpreting the Predicted Result :
result = RandomForest.predict(input_data_reshaped)
result


# In[ ]:




