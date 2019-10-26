import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes.csv')

x = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
