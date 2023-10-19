#!/usr/bin/env python
# coding: utf-8

# In the previous session we trained a model for predicting churn and evaluated it. Now let's deploy it

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pickle
import requests


input_file = "model_C=1.0.bin"


C = 1.0
n_splits = 5


df = pd.read_csv("./Customer-churn.csv")

df.columns = df.columns.str.lower().str.replace(" ", "_")

categorical_columns = list(df.dtypes[df.dtypes == "object"].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(" ", "_")

df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == "yes").astype(int)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_test = df_test.churn.values


numerical = ["tenure", "monthlycharges", "totalcharges"]

categorical = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]


# In[5]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print("C=%s %.3f +- %.3f" % (C, np.mean(scores), np.std(scores)))

dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# define the name of the output file
output_file = f"model_C={C}.bin"


# Saving the Model
# open a file with this name, creates it if it does not exist.
# w = write, b - binary
# Below we take the steps to open, save and then close the file. This is manual.
# We can do this easier
f_out = open(output_file, "wb")
pickle.dump((dv, model), f_out)
f_out.close()

# By using the with function, essentially a TryCatch statement, python will do the above and then close our file as well.
with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)


# In[15]:


get_ipython().system("ls -lh *.bin")


# In[19]:
# Loading the file
# pickle.load will load the file and deserialize it
# Notice we change 'wb' to 'rb' because we don't want ot overwrite it, just read it
with open(input_file, "rb") as f_in:
    dv, model = pickle.load(f_in)


# In[20]:


model


# In[42]:


customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
}


# In[43]:


X = dv.transform([customer])


# In[44]:


y_pred = model.predict_proba(X)[0, 1]


# In[45]:


print("input:", customer)
print("output:", y_pred)


# Making requests

# In[46]:


import requests


# In[47]:


url = "http://localhost:9696/predict"


# In[48]:


customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "two_year",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
}


# In[49]:


response = requests.post(url, json=customer).json()


# In[ ]:


response


# In[ ]:


if response["churn"]:
    print("sending email to", "asdx-123d")


# In[ ]:
