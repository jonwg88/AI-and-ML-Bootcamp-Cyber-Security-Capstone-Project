#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import torch.nn as nn
import seaborn as sns
import torch


# In[2]:


#Import the dataset
data = pd.read_csv('capstone dataset.csv')


# In[3]:


#Explore the data using histograma
data.hist(figsize=(20, 15))
plt.tight_layout()
plt.show()


# Features most closely affiliated with phishing URLs are 'URLURL_Length', 'Prefix_Suffix', 'SFH', 'Redirect', and 'Page_Rank'.

# In[4]:


#Explore the data using heatmaps
plt.figure(figsize=(27, 17))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()


# Features most closely correlated with result are 'SSLfinal_State' and 'URL_of_Anchor', which are also closely correlated with each other.

# In[5]:


#View first five rows of the dataset
print(data.head())


# In[6]:


#Determine the number of samples present in the data
print("Number of samples in dataset:", len(data))


# In[7]:


#Determine the number of unique elements in all features
for column in data.columns:
    print(f"Unique elements in {column}: {data[column].unique()}")


# In[8]:


#Check for null values in any features
print(data.isnull().sum())


# In[9]:


#Change 'Result' column to have labels of 1 and 0 instead of 1 and -1 to make binary classification easier
data['Result'] = data['Result'].replace(-1, 0)


# In[10]:


#Remove features that might be correlated with some threshold
#Correlation threshold is 0.9
#'PopUpWidnow' [sic] and 'Favicon' are highly correlated with each other and weakly correlated with 'Result', so one or both should be dropped.
#Drop just 'PopUpWidnow' for now, but if model performance is poor, consider dropping 'Favicon' as well.
def remove_correlated_features(data, threshold):
    correlated_columns = set()
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_columns.add(colname)
    data_mod = data.copy()
    for col in correlated_columns:
        if col in data_mod.columns:
            del data_mod[col]
    return data_mod, correlated_columns

threshold_value = 0.9
mod_data, removed_columns = remove_correlated_features(data, threshold_value)
print("Removed columns:", removed_columns)


# In[85]:


#Split data into training and testing sets
from sklearn.model_selection import train_test_split

X = mod_data.drop('Result', axis=1).values
y = data['Result'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[86]:


#Build classification model using a binary classifier to detect phishing URLs
class URLClassModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return torch.sigmoid(self.linear3(x))
    
    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            return (pred >= 0.5).float()


# In[87]:


#Convert datasets to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# In[88]:


#Create an instance of the model
input_size = X_train.shape[1]
model = URLClassModel(input_size)


# In[89]:


#Establish hyperparameters
lr = 0.0001
epochs = 1000
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[90]:


#Train the URL classification model
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    print("epoch:", i, "loss:", loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[91]:


#Illustrate diagnostic ability of this binary classifier by plotting the ROC curve
from sklearn.metrics import roc_curve, auc

model.eval()
with torch.no_grad():
    y_prob = model(X_test_tensor)
y_prob = y_prob.numpy()
y_test_numpy = y_test_tensor.numpy()

fpr, tpr, thresholds = roc_curve(y_test_numpy, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[92]:


#Validate the accuracy of data by the K-Fold cross-validation technique
from sklearn.model_selection import KFold

X_train_numpy = X_train_tensor.numpy()
y_train_numpy = y_train_tensor.numpy()
accuracies = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train, test in kf.split(X_train_numpy):
    X_val_fold = torch.tensor(X_train_numpy[test], dtype=torch.float32)
    y_val_fold = torch.tensor(y_train_numpy[test], dtype=torch.float32).view(-1, 1)
    
    model.eval()
    with torch.no_grad():
        y_pred = model.predict(X_val_fold)
        accuracy = (y_pred == y_val_fold).float().mean()
        accuracies.append(accuracy.item())
        
avg_accuracy = np.mean(accuracies)
print(f"Average accuracy after 5-fold cross-validation: {avg_accuracy:.4f}")


# In[93]:


from sklearn.metrics import classification_report

model.eval()
with torch.no_grad():
    train_preds = model.predict(X_train_tensor)
train_preds_numpy = train_preds.numpy()

print("Training data classification report:")
print(classification_report(y_train_numpy, train_preds_numpy))

with torch.no_grad():
    test_preds = model.predict(X_test_tensor)
test_preds_numpy = test_preds.numpy()

print("\nTest data classification report:")
print(classification_report(y_test_numpy, test_preds_numpy))


# The model ended up having a training accuracy of 0.91 and a validation accuracy of 0.90. The initial accuracies were very poor, but increasing the number of perceptrons per convolutional layer helped. Increasing the number of epochs during which the model underwent training helped far more. 

# In[ ]:




