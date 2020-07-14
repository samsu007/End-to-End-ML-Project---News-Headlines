#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# In[2]:


df = pd.read_csv('Data.csv', encoding="ISO-8859-1")


# In[4]:


df.head()


# In[6]:


df.shape


# In[9]:


df.info()


# In[46]:


train_set = df[df['Date'] < '20150101']
test_set = df[df['Date'] > '20141231']


# In[47]:


# Removing punctuations
data = train_set.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# Renaming column names for ease of access
list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index
data.head()


# In[52]:


# Converting Headlines to lower case
for i in new_Index:
    data[i] = data[i].str.lower()

data.head(1)


# In[58]:


' '.join(str(i) for i in data.iloc[1, 0:25])


# In[60]:


headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(i) for i in data.iloc[row, 0:25]))

headlines[0]


# In[61]:


# In[66]:


# implement Bag of words
countvector = CountVectorizer(ngram_range=(2, 2))
traindataset = countvector.fit_transform(headlines)
traindataset


# In[69]:


# Create Random Forest Classifier Model
randclf = RandomForestClassifier(n_estimators=10, criterion='entropy')
randclf.fit(traindataset, train_set['Label'])


# In[72]:


# Predict the testdata set
test_headlines = []
for row in range(0, len(test_set.index)):
    test_headlines.append(' '.join(str(x) for x in test_set.iloc[row, 2:27]))
testdataset = countvector.transform(test_headlines)
predictions = randclf.predict(testdataset)


# In[73]:


# Check accuracy of Model


# In[75]:


matrix = confusion_matrix(test_set['Label'], predictions)
print(matrix)
score = accuracy_score(test_set['Label'], predictions)
print(score)
report = classification_report(test_set['Label'], predictions)
print(report)


# In[ ]:


#  Above we see the confusion matrix 165 headlines is detected correctly true positive and 149 is true negative
# 27 is detecting false negative and 37 is false negative
