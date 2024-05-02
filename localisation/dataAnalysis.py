import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import re

sys.path.insert(1, '/Users/mariam/Downloads/IIB Project readings/data_cleaned_adapted_features_full')

df = pd.read_csv('/Users/mariam/Downloads/IIB Project readings/data_cleaned_adapted_features_full/sensoringData_feature_prepared_20_19.0_0.csv')

columns = df.columns.values.tolist()
columnsAsStr = ' '.join(columns)
x = re.findall("[A-Za-z0-9]+_[A-Za-z0-9]+_mean", columnsAsStr)
x +=['user', 'activity']
rslt_df = df[x]
# print(rslt_df.head)

# correlation_matrix = rslt_df.iloc[:, :-1].corr(method='spearman')
# sns.heatmap(correlation_matrix, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()
sns.pairplot(rslt_df)
plt.show()