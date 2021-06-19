# -*- coding: utf-8 -*-
"""
Created on Tue May 25 18:17:09 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
df = pd.read_csv("Fraud_check.csv")
df.head()


#Converting categorical data into numeric
#As there is categorical columns in the dataset. So we need to convert them into numerical first by using pd.get_dummies.
#Creating dummy variables for Undergrad, MaritalStatus,Urban. Then drop first dummy variable
df = pd.get_dummies(df,columns=["Undergrad","Marital.Status","Urban"], drop_first=True)

#creating a new columns TaxInc and dividing the Taxable.Income on the basis of [10002,30000,99620] for Risky and Good
#splitting the dataset into x(independent features) and y(target)
df["TaxInc"] = pd.cut(df["Taxable.Income"],bins = [10002,30000,99620], labels = ["Risky","Good"])

#Lets assume:Taxable.Income<=30000 is Risky=0 and others are Good=1
#Also after creating a new column TaxInc, made dummies variable of TaxInc, concated right side of df
df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)

#view bottom ten observations
df.tail(10)

#Normalization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
#Normalize the dataframe (considering the numerical part of the data)
df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)


#Declaring the x(independent features) and y(target)
x = df_norm.drop(["TaxInc_Good"], axis=1)
y = df_norm["TaxInc_Good"]

#train test split
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, test_size=0.3, random_state=0)

#using Decision Tree Classifier(model building)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = "entropy")
model1 = model.fit(Xtrain,ytrain)
pred = model1.predict(Xtest)
type(pred)
pd.Series(pred).value_counts()

#evaluating the model
pd.crosstab(ytest,pred)
#or evaluating the model using confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(ytest,pred))

#Trained data
temp = pd.Series(model.predict(Xtrain)).reset_index(drop=True)
#Test data
np.mean(pd.Series(ytrain).reset_index(drop=True) == pd.Series(model.predict(Xtrain)))

#Accuracy test
np.mean(pred==ytest) #67%
#or accuracy test by classification report
print(classification_report(ytest,pred))
#67%

#plot
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(model1,filled=True)
tree.plot_tree





