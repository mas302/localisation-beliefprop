import pandas as pd
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

sys.path.insert(1, '/Users/mariam/Downloads/IIB Project readings/data_cleaned_adapted_features_full')
sys.path.insert(1, '/Users/mariam/Downloads/IIB Project readings/IIB project code/localisation/src')

import sensors

# Load data
df = pd.read_csv('/Users/mariam/Downloads/IIB Project readings/data_cleaned_adapted_features_full/sensoringData_feature_prepared_20_19.0_0.csv')

random.seed(0)
columns = df.columns.values.tolist()
columnsAsStr = ' '.join(columns)
x = re.findall("gps+_[A-Za-z0-9]+_mean", columnsAsStr)
# x += re.findall("acc+_[A-Za-z0-9]+_mean", columnsAsStr)
x += ['activity']

# Extract unique user IDs
userIDs = df['user'].unique().tolist()

parameters = df[x]
encodedParameters = pd.get_dummies(parameters, dtype=float)
encodedParameters.drop(['activity_Walking', 'activity_Driving', 'gps_lat_mean', 'gps_bearing_mean'], axis=1, inplace=True)

rowCount = encodedParameters.shape[0]
midRows = math.ceil((rowCount-1)/2)

# know that a user id and activity defines a session. take haversine distances of time series for consistent behaviour.
encodedParameters['networkLat'] = np.nan
encodedParameters['networkLat'][:midRows] = df['gps_lat_mean'][:midRows]
encodedParameters['networkLat'][midRows:rowCount] = df['gps_lat_mean'][midRows:rowCount] + 1e-3*np.random.randn(rowCount- midRows)

encodedParameters['networkLon'] = np.nan
encodedParameters['networkLon'][:midRows] = df['gps_long_mean'][:midRows]
encodedParameters['networkLon'][midRows:rowCount] = df['gps_long_mean'][midRows:rowCount] + 1e-3*np.random.randn(rowCount- midRows)

encodedParameters['targetVal'] = np.nan
encodedParameters['targetVal'][:midRows] = np.ones((midRows))#df['gps_long_mean'][:500]
encodedParameters['targetVal'][midRows:rowCount] = np.zeros((rowCount- midRows)) #df['gps_long_mean'][500:1000] + np.random.rand(500)

dependentVars = encodedParameters.columns[:-1]
targetVars = encodedParameters.columns[-1]

X_train, X_test, y_train, y_test = train_test_split(encodedParameters[dependentVars], encodedParameters[targetVars], test_size=0.2, random_state=16)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(solver = 'saga', max_iter=10000, penalty='l2', random_state=16, verbose=1)

# fit the model with data
logreg.fit(X_train, y_train)
print(logreg.coef_)

y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred, normalize = 'all')

print(cnf_matrix)

target_names = ['UE unverified', 'UE verified']
print(classification_report(y_test, y_pred, target_names=target_names))