# Authors: Tony Calarese, Lyall Rogers, and John Long
# Class:  DAT 330-01
# Certification of Authenticity:
# I certify that this is entirely my own work, except where I have given fully documented
# references to the work of others.  I understand the definition and consequences of
# plagiarism and acknowledge that the assessor of this assignment may, for the purpose of
# assessing this assignment reproduce this assignment and provide a copy to another member
# of academic staff and / or communicate a copy of this assignment to a plagiarism checking
# service(which may then retain a copy of this assignment on its database for the purpose
# of future plagiarism checking).

#Tony Code Begin

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import dataCollector as data

import pandas as pd
import numpy as np
import math
import statistics #Used with calculating the mode
from statistics import mode
import random
random.seed(30)

#Begin of Tony Edit
if __name__ == "__main__":
    file = "loan_final313_edited_out_categories.csv"
    data_df = pd.read_csv(file, delimiter=",")

      # Declaring the encoders
    X_encoder = preprocessing.LabelEncoder()
    y_encoder = preprocessing.LabelEncoder()

    #Reading and Encoding Data
    # Encoding train data
    # Create new dataset get all columns except chrun
    NewData = data_df.loc[:, data_df.columns != 'Churn']

    # Encoding the Train Data
    y_train = y_encoder.fit_transform(data_df['Churn'])
    X_train = pd.DataFrame()
