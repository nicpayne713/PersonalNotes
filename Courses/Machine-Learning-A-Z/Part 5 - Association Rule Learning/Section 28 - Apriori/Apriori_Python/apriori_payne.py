# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:59:07 2017

@author: paynen3
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range (0,20)])


# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, 
                    min_support = 0.003 , # 3*7/7500 average of 3 purchases per day 
                    min_confidence = 0.2 , # 
                    min_lift = 3 , # 
                    min_length = 2)

results = list(rules)

