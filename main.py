# Author: Tony Calarese, John Long, and Lyall Rogers
# Class:  DAT 330-01
# Certification of Authenticity:
# I certify that this is entirely my own work, except where I have given fully documented
# references to the work of others.  I understand the definition and consequences of
# plagiarism and acknowledge that the assessor of this assignment may, for the purpose of
# assessing this assignment reproduce this assignment and provide a copy to another member
# of academic staff and / or communicate a copy of this assignment to a plagiarism checking
# service(which may then retain a copy of this assignment on its database for the purpose
# of future plagiarism checking).

import pandas as pd
import numpy as np
import random
random.seed(30)

# Label Encoder Source
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris


# Lyall
#loans = pd.read_csv('loan_final313.csv')
#print(loans.head())

#loans = loans.drop(['id', 'issue_d', 'final_d', 'home_ownership', 'application_type', 'income_category', 'purpose',
#                    'interest_payments', 'loan_condition', 'grade', 'term'], axis=1)
#loans = loans.replace('munster', 0)
#loans = loans.replace('leinster', 1)
#loans = loans.replace('cannught', 2)
#loans = loans.replace('ulster', 3)
#print(loans['region'])
# Lyall

#Tony Start of Code
class Loans:
    #Initialize with the file name and the column that y will be,
    def __init__(self, file, y_delimiter):
        # Declaring the encoders
        X_encoder = preprocessing.LabelEncoder()
        y_encoder = preprocessing.LabelEncoder()
        data_df = pd.read_csv(file)
        self.y = y_encoder.fit_transform(data_df[y_delimiter])
        self.X = pd.DataFrame()
        self.k = 1 #Default k to 1
        #Loading X and y
        NewData = data_df.loc[:, data_df.columns != y_delimiter]
        for col in NewData.columns: # If the column attributes are objects then append the encoded version of those Otherwise append the data
            self.X[col] = X_encoder.fit_transform(NewData[col]) if NewData[col].dtype == 'O' else NewData[col]

        #Memory Clean up
        del data_df, X_encoder, y_encoder, y_delimiter, NewData
    def setK(self, k):
        self.k = k
    #Running the sklearn KNN
    def sklearnKNN(self):
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        knn.score(X_test, y_test)


    #This will take a list of variables that will be used to filter out of X
    def filterOut(self, columns):
        self.X.drop(columns, axis=1)

    #Function for showing what X and y are
    def showAll(self):
        print(self.X)
        print(self.y)
#Tony End

#Tony Begin
#Main Function
if __name__== "__main__":
    loan1 = Loans(file='loan_final313.csv', y_delimiter='id')
    loan1.filterOut(['issue_d', 'final_d', 'home_ownership', 'application_type',
                     'income_category', 'purpose', 'interest_payments', 'loan_condition',
                     'grade', 'term'])
    loan1.showAll()

#Tony End
