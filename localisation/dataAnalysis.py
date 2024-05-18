import pandas as pd
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re

sys.path.insert(1, '/Users/mariam/Downloads/IIB Project readings/data_cleaned_adapted_features_full')

# Load data
df = pd.read_csv('/Users/mariam/Downloads/IIB Project readings/data_cleaned_adapted_features_full/sensoringData_feature_prepared_20_19.0_0.csv')

# Get columns and find specific GPS features
columns = df.columns.values.tolist()
columnsAsStr = ' '.join(columns)
x = re.findall("gps+_[A-Za-z0-9]+_mean", columnsAsStr)
x += ['user']

# Extract unique user IDs
userIDs = df['user'].unique().tolist()

# Filter for walking activity and sort by user ID
walking_df = df[df['activity'] == 'Inactive'].sort_values('user')

# Group by users and aggregate data
summary_stats = walking_df.groupby('user')[x].agg(['mean']) #, 'std'])
summary_stats.columns = ['{}_{}'.format(col, agg) for col, agg in summary_stats.columns]

# Remove the duplicated 'mean' from column names where necessary
summary_stats.columns = [re.sub(r'_(mean)(?=_mean)', '', col) for col in summary_stats.columns]
summary_stats.drop(['user_mean'], axis=1, inplace=True) #user_std

# Remove columns with zero standard deviation
non_constant_columns = summary_stats.columns[summary_stats.std() != 0]
summary_stats = summary_stats[non_constant_columns]
summary_stats.drop(columns=[col for col in summary_stats.columns if 'user' in col], axis=1, inplace=True)

# Compute the correlation matrix, fill NaN with 0
corr_matrix = summary_stats.corr().fillna(0)

# Replace infinite values with 0, if any
corr_matrix.replace([np.inf, -np.inf], 0, inplace=True)

# Generate and display the cluster map
sns.clustermap(corr_matrix, cmap="coolwarm", standard_scale=1)
plt.show()