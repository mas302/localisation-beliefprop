import pandas as pd
import sys

sys.path.insert(1, '/Users/mariam/Downloads/IIB Project readings/data_cleaned_adapted_features_full')

df = pd.read_csv('/Users/mariam/Downloads/IIB Project readings/data_cleaned_adapted_features_full/sensoringData_feature_prepared_20_19.0_0.csv')
print(df.head())