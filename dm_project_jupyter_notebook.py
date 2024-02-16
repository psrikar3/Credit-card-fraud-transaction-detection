#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


# In[16]:


data = pd.read_csv("D:\Data_Mining_Credit_Card\creditcard.csv")


# In[17]:


data.info()


# In[18]:


data.isnull()


# In[19]:


data.isna().sum()


# In[20]:


data.isnull().sum()


# In[21]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[22]:


data.head(10)


# In[ ]:





# In[23]:


data.shape


# In[24]:


data.tail(10)


# In[25]:


data.describe()


# In[31]:


fraud_transactions = data[data['Class'] == 1]
valid_transactions = data[data['Class'] == 0]
outlierFraction_nv = len(fraud_transactions)/float(len(valid_transactions))
print(outlierFraction_nv)
print('Fraud Cases count: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions count: {}'.format(len(data[data['Class'] == 0])))


# In[32]:


sns.set_style('whitegrid')
sns.countplot(x='Class',data=data)


# In[34]:


print("The following are the Amount details of the fraudulent transaction")
fraud_transactions.Amount.describe()


# In[35]:


print("details of valid transaction")
valid_transactions.Amount.describe()


# In[36]:


data['Amount'].hist(bins=30,color='darkred',alpha=0.3, figsize=(20,30))


# In[37]:


# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 12))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# Negative Correlations: V17, V14, V12 and V10 are negatively correlated.The lower these values are, the more likely the end result will be a fraud transaction.
# Positive Correlations: V2, V4, V11, and V19 are positively correlated.The higher these values are, the more likely the end result will be a fraud transaction.
# BoxPlots: We used boxplots to have a better understanding of the distribution of these features in fradulent and non fradulent transactions.

# In[ ]:





# In[40]:


# dividing the X and the Y from the dataset
X = data.drop(['Class'], axis = 1)
Y = data["Class"]
print(X.shape)
print(Y.shape)
# getting just the values for the sake of processing 
# (its a numpy array with no columns)
xData = X.values
yData = Y.values


# In[41]:


# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

# Lets shuffle the data before creating the subsamples

data = data.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = data.loc[data['Class'] == 1]
non_fraud_df = data.loc[data['Class'] == 0][:492]

normal_distributed_df_nv = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df_nv.sample(frac=1, random_state=42)

new_df.head()


# In[42]:


# Make sure we use the subsample in our correlation

#f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))



sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20})
#ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()


# In[43]:


print('Distribution of the Classes in the subsample dataset is as follows')
print(new_df['Class'].value_counts()/len(new_df))



sns.countplot('Class', data=new_df, )
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# In[44]:


f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V17", data=new_df, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=new_df, ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V12", data=new_df, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V10", data=new_df, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

plt.show()


# In[45]:


f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Class", y="V11", data=new_df, ax=axes[0])
axes[0].set_title('V11 attribute vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=new_df, ax=axes[1])
axes[1].set_title('V4 attribute vs Class Positive Correlation')


sns.boxplot(x="Class", y="V2", data=new_df, ax=axes[2])
axes[2].set_title('V2 attribute vs Class Positive Correlation')


sns.boxplot(x="Class", y="V19", data=new_df, ax=axes[3])
axes[3].set_title('V19 attribute vs Class Positive Correlation')

plt.show()


# In[ ]:





# In[46]:


new_df.shape


# In[47]:


# dividing the X and the Y from the dataset
X = new_df.drop(['Class'], axis = 1)
Y = new_df["Class"]
print(X.shape)
print(Y.shape)
# getting just the values for the sake of processing 
# (its a numpy array with no columns)
xData = X.values
yData = Y.values


# In[48]:


# Using Scikit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(
xData, yData, test_size = 0.2, random_state = 42)


# In[49]:


# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
# predictions
yPred = rfc.predict(xTest)


# In[51]:


# Evaluating the classifier
# printing every score of the classifier
# scoring in anything
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

n_outliers_nv = len(fraud_transactions)
n_errors = (yPred != yTest).sum()
print("The model used is Random Forest classifier")

acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc))

prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec))

rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec))

f1 = f1_score(yTest, yPred)
print("The F1-Score is {}".format(f1))

MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is{}".format(MCC))


# In[52]:


from sklearn.linear_model import LogisticRegression


# In[53]:


logmodel=LogisticRegression()


# In[54]:


logmodel.fit(xTrain, yTrain)


# In[55]:


predictions=logmodel.predict(xTest)


# In[56]:


from sklearn.metrics import classification_report


# In[57]:


classification_report(yTest, predictions)


# In[58]:


from sklearn.metrics import accuracy_score


# In[59]:


accuracy_score(yTest, predictions)


# In[242]:


from sklearn import tree


# In[246]:


dt= tree.DecisionTreeClassifier(max_depth=1000)
dt.fit(xTrain, yTrain)


# In[247]:


dt.score(xTest, yTest)


# In[248]:


from sklearn import ensemble
gbc=ensemble.GradientBoostingClassifier()


# In[249]:


gbc.fit(xTrain, yTrain)


# In[250]:


gbc.score(xTest, yTest)


# In[60]:


from sklearn.naive_bayes import GaussianNB


# In[61]:


nbc=GaussianNB()


# In[62]:


nbc.fit(xTrain, yTrain)


# In[66]:


nbc.score(xTest, yTest)


# In[67]:


from sklearn.neighbors import KNeighborsClassifier


# In[92]:


knnc=KNeighborsClassifier(n_neighbors=15)


# In[93]:


knnc.fit(xTrain, yTrain)


# In[94]:


knnc.score(xTest, yTest)


# In[95]:


from sklearn.svm import SVC
svmc=SVC(probability=True, kernel='linear')
svmc.fit(xTrain, yTrain)


# In[96]:


svmc.score(xTest, yTest)


# In[ ]:





# In[97]:


# printing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS,
yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
