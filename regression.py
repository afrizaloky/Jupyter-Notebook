#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import sklearn
import statsmodels.api as sm
sns.set()

# Input Data
data = pd.read_csv('1.01. Simple linear regression.csv')

# Define Variable
y = data['GPA']
x1 = data['SAT']

# Linear Regression
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
const, SAT = results.params

# Visualitation
plt.scatter(x1, y)
yhat = SAT*x1 + const
fig = plt.plot(x1, yhat, lw=4, c='red', label='Regression Linear')
plt.xlabel('SAT', fontsize=15)
plt.ylabel('SAT', fontsize=15)
plt.show()
