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
x += ['user', 'id']

# Extract unique user IDs
userIDs = df['user'].unique().tolist()

# Filter for walking activity and sort by user ID
walking_df = df[df['activity'] == 'Walking'].sort_values('user')
driving_df = df[df['activity'] == 'Driving'].sort_values('user')
inactive_df = df[df['activity'] == 'Inactive'].sort_values('user')
active_df = df[df['activity'] == 'Active'].sort_values('user')
# Group by users and aggregate data
summary_stats = walking_df.groupby('user')[x].agg(['mean'])
summary_stats.columns = ['{}_{}'.format(col, agg) for col, agg in summary_stats.columns]

# Remove the duplicated 'mean' from column names where necessary
summary_stats.columns = [re.sub(r'_(mean)(?=_mean)', '', col) for col in summary_stats.columns]
summary_stats.drop(['user_mean'], axis=1, inplace=True) #user_std

walk_by_user = [i for _, i in walking_df.groupby('user')[x]]
drive_by_user = [i for _, i in driving_df.groupby('user')[x]]

##############

# fig, axs = plt.subplots(3, 5, figsize=(15, 10), facecolor='w', edgecolor='k', sharey='all', sharex='all')
# fig.subplots_adjust(hspace = .5, wspace=.5)

# axs = axs.ravel()

# all_handles = []
# all_labels = []

# for i in range(len(walk_by_user)):
#     userData = walk_by_user[i]
#     user_id = userData['user'].iloc[0]
#     stepsUD = len(userData['gps_speed_mean'])
#     plots = []
    
#     scatter = axs[i].scatter(np.linspace(0, stepsUD, stepsUD), userData['gps_speed_mean'], label = 'Mean walking speed', s=1)
#     plots.append(scatter)

#     drivingData = driving_df[driving_df['user'] == user_id]
#     if not drivingData.empty:
#         stepsDD = len(drivingData['gps_speed_mean'])
#         scatter = axs[i].scatter(np.linspace(0, stepsDD, stepsDD), drivingData['gps_speed_mean'], label = 'Mean driving speed', s=1, color = 'r')
#         plots.append(scatter)
    
#     activityData =  active_df[active_df['user'] == user_id]
#     if not activityData.empty:
#         stepsAD = len(activityData['gps_speed_mean'])
#         scatter = axs[i].scatter(np.linspace(0, stepsAD, stepsAD), activityData['gps_speed_mean'], label = 'Mean active speed', s=1, color = '#5F24BA')
#         plots.append(scatter)
    
#     iNactivityData =  inactive_df[inactive_df['user'] == user_id]
#     if not iNactivityData.empty:
#         stepsID = len(iNactivityData['gps_speed_mean'])
#         scatter = axs[i].scatter(np.linspace(0, stepsID, stepsID), iNactivityData['gps_speed_mean'], label = 'Mean inactive speed', s=1, color = '#5893E4')
#         plots.append(scatter)

#     for plot in plots:
#         handle, label = plot.legend_elements()
#         all_handles.extend(handle)
#         all_labels.extend(label)

#     axs[i].set_title("User " + str(i+1))
#     axs[i].set_xscale("log")
#     # axs[i].locator_params(axis='x', nbins=3)

# for ax in axs:
#     ## check if something was plotted 
#     if not bool(ax.has_data()):
#         fig.delaxes(ax)

# unique_labels = dict(zip(all_labels, all_handles))
# fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
# plt.legend()
# plt.show()

#################################

# Create subplots
fig, axs = plt.subplots(3, 5, figsize=(15, 10), facecolor='w', edgecolor='k', sharey=True, sharex=True)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
axs = axs.ravel()

# Collect all handles and labels for legend
all_handles = []
all_labels = []

# Plot data for each user
for i in range(len(walk_by_user)):
    userData = walk_by_user[i]
    user_id = userData['user'].iloc[0]
    stepsUD = len(userData['gps_speed_mean'])

    # Walking data
    scatter = axs[i].scatter(np.linspace(0, stepsUD, stepsUD), userData['gps_speed_mean'], label='Mean walking speed', s=1)
    if scatter.get_label() not in all_labels:
        all_handles.append(scatter)
        all_labels.append(scatter.get_label())

    # Driving data
    drivingData = driving_df[driving_df['user'] == user_id]
    if not drivingData.empty:
        stepsDD = len(drivingData['gps_speed_mean'])
        scatter = axs[i].scatter(np.linspace(0, stepsDD, stepsDD), drivingData['gps_speed_mean'], label='Mean driving speed', s=1, color='r')
        if scatter.get_label() not in all_labels:
            all_handles.append(scatter)
            all_labels.append(scatter.get_label())

    # Active data
    activityData = active_df[active_df['user'] == user_id]
    if not activityData.empty:
        stepsAD = len(activityData['gps_speed_mean'])
        scatter = axs[i].scatter(np.linspace(0, stepsAD, stepsAD), activityData['gps_speed_mean'], label='Mean active speed', s=1, color='#5F24BA')
        if scatter.get_label() not in all_labels:
            all_handles.append(scatter)
            all_labels.append(scatter.get_label())

    # Inactive data
    iNactivityData = inactive_df[inactive_df['user'] == user_id]
    if not iNactivityData.empty:
        stepsID = len(iNactivityData['gps_speed_mean'])
        scatter = axs[i].scatter(np.linspace(0, stepsID, stepsID), iNactivityData['gps_speed_mean'], label='Mean inactive speed', s=1, color='#5893E4')
        if scatter.get_label() not in all_labels:
            all_handles.append(scatter)
            all_labels.append(scatter.get_label())

    axs[i].set_title(f"User {user_id}")
    axs[i].set_xscale("log")

# Remove empty subplots
for ax in axs:
    if not ax.has_data():
        fig.delaxes(ax)

# Create a single legend for the entire figure
fig.legend(all_handles, all_labels, loc = 'lower right')
plt.show()

# print(summary_stats.iloc[:2, :6])

# # Remove columns with zero standard deviation
# non_constant_columns = summary_stats.columns[summary_stats.std() != 0]
# summary_stats = summary_stats[non_constant_columns]
# summary_stats.drop(columns=[col for col in summary_stats.columns if 'user' in col], axis=1, inplace=True)

# covariance = summary_stats[['gps_alt_mean', 'gps_lat_mean', 'gps_long_mean']].cov()
# # print(covariance)
# # Compute the correlation matrix, fill NaN with 0
# corr_matrix = summary_stats.corr().fillna(0)

# # Replace infinite values with 0, if any
# corr_matrix.replace([np.inf, -np.inf], 0, inplace=True)

# # Generate and display the cluster map
# sns.clustermap(corr_matrix, cmap="coolwarm", standard_scale=1)
# plt.show()