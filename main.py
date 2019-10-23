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
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class Loans:
    #Initialize with the file name and the column that y will be,
    def __init__(self, file, y_delimiter):
        enc = preprocessing.LabelEncoder() #Declaring the Encoder #Tony
        data_df = pd.read_csv(file, error_bad_lines=False) #Tony #Source of reference for errors:https://stackoverflow.com/questions/18039057/python-pandas-error-tokenizing-data

        self.y = enc.fit_transform(data_df[y_delimiter]) #Tony
        self.X = pd.DataFrame() #Tony
        self.k = 3          #Default k to 3 #Tony


        #Loading X and y
        NewData = data_df.loc[:, data_df.columns != y_delimiter] #Tony
        for col in NewData.columns: # If the column attributes are objects then append the encoded version of those Otherwise append the data
            self.X[col] = enc.fit_transform(NewData[col]) if NewData[col].dtype == 'O' else NewData[col] #Tony

        del data_df, enc, y_delimiter, NewData #Memory Clean up #Tony

        #Creating the Train and Test Data
        self.X_train = pd.DataFrame() #Tony
        self.X_test = pd.DataFrame() #Tony
        self.y_train = []  #Tony
        self.y_test = []    #Tony

        #Predictions
        self.y_pred_knn = []    #Tony
        self.y_pred_tree = []

        #Accuracies -- Default to 0 of not run
        self.accuracy_dict = {} #Tony


    #Function to set k, default is 3
    def setK(self, k):
        self.k = k #Tony


    #This will take a list of variables that will be used to filter out of X
    def filterOut(self, columns):
        self.X.drop(columns, inplace=True, axis=1) #Lyall

    def test_train_split(self):
        #Creating the Train and Test Data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, test_size=0.3) #Tony and John
        self.X_train = X_train #Tony
        self.X_test = X_test #Tony
        self.y_train = y_train #Tony
        self.y_test = y_test    #Tony

        del X_train, X_test, y_train, y_test #Memory Clean up for speed #Tony
    #Running the sklearn KNN
    def sklearnKNN(self):
        knn = KNeighborsClassifier(n_neighbors=self.k) #Tony
        knn.fit(self.X_train, self.y_train) #Tony
        self.y_pred_knn = knn.predict(self.X_test) #Tony
        self.accuracy_dict["SKLEARN_KNN"] = metrics.accuracy_score(self.y_test, self.y_pred_knn) #Lyall


    #Running the Descision Tree Variant
    def DescisionTree(self):
        clf = DecisionTreeClassifier(max_depth=4, random_state=0) #Lyall and John
        clf.fit(self.X_train, self.y_train) #Lyall and John
        self.y_pred_tree = clf.predict(self.X_test) #Lyall and John
        self.accuracy_dict["Tree"] = metrics.accuracy_score(self.y_test, self.y_pred_tree) #Lyall

    #Function for showinghow much the predicted data relates to the answer key
    def showAll(self):
        print("Answer Key: ", self.y_test[:20]) #Tony
        print("Y_Pred_KNN", self.y_pred_knn[:20]) #Tony
        print("Y_Pred_Descision_Tree", self.y_pred_tree[:20]) #Tony
        print(self.accuracy_dict.items())


#Tony
#Main Function
if __name__== "__main__":
    loan1 = Loans(file='loan_final313.csv', y_delimiter='loan_condition_cat') #Tony
    #Lyall
    loan1.filterOut(['issue_d', 'final_d', 'home_ownership', 'application_type',
                     'income_category', 'purpose', 'interest_payments', 'grade', 'term', 'id',
                     'loan_condition'])
    loan1.test_train_split() #After Dropping Data Split it up #Lyall and Tony
    loan1.setK(3) #Tony
    loan1.sklearnKNN() #Tony
    loan1.DescisionTree() #John and Lyall
    loan1.showAll() #Tony


