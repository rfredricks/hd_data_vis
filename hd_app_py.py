# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 20:23:46 2021

@author: freddy
"""

import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import sys

# load heart_failure_clinical_records_dataset.csv into dataframe
data = pd.read_csv("heart_failure_clinical_records_dataset.csv", header=0)

# y axis (prediction) is death event
y = data.DEATH_EVENT
# x axis (input) is rest of data
x = data.drop('DEATH_EVENT', axis=1)

# split data into training and test set 80/20
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

# create and fit model using random forest classifier and training set
classifier = RandomForestClassifier(n_estimators = 30)
classifier.fit(x_train, y_train)

# make prediction and determine accuracy with test set
y_pred = classifier.predict(x_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

# create user menu
def user_menu():
    while(True):
        print('Make a selection:')
        selection = input('D - dashboard  E - enter data  X - exit  ')
        if(selection == 'D'):
            display_dash()
        elif(selection == 'E'):
            user_input()
        elif(selection == 'X'):
            sys.exit(0)
        else:
            print('Input not recognized. Please make a valid selection.')


# get information to make prediction for new customer
def user_input():
    print('Enter customer information: ')
    age = float(input('Age: '))
    anemia = int(input('Anemia? 0=no, 1=yes: '))
    creatine = int(input('Creatinine phosphokinase: '))
    diabetes = int(input('Diabetes? 0=no, 1=yes: '))
    ef = int(input('Ejection fraction: '))
    hbp = int(input('High blood pressure? 0=no, 1=yes: '))
    platelets = float(input('Platelets: '))
    serum_crea = float(input('Serum creatinine: '))
    serum_sod = int(input('Serum sodium: '))
    sex = int(input('Sex - M=0, F=1: '))
    smoking = int(input('Smoking? 0=no, 1=yes: '))
    sm_time = int(input('Time? '))
    cust = [age, anemia, creatine, diabetes, ef, hbp, platelets, serum_crea, serum_sod, sex, smoking, sm_time]
    cust = np.array(cust)
    cust_predict = classifier.predict(cust.reshape(1,-1))
    print(cust_predict)    
    

# display dashboard
def display_dash():
    pass


user_menu()