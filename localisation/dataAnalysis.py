import pandas as pd
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.legend_handler import HandlerPathCollection

sys.path.insert(1, '/Users/mariam/Downloads/IIB Project readings/data_cleaned_adapted_features_full')

# Load data
df = pd.read_csv('/Users/mariam/Downloads/IIB Project readings/data_cleaned_adapted_features_full/sensoringData_feature_prepared_20_19.0_0.csv')

# Get columns and find specific GPS features
columns = df.columns.values.tolist()
columnsAsStr = ' '.join(columns)
x = re.findall("gps+_[A-Za-z0-9]+_mean", columnsAsStr)
x += re.findall("acc+_[A-Za-z0-9]+_mean", columnsAsStr)
x += ['user', 'id']

# Extract unique user IDs
userIDs = df['user'].unique().tolist()

# Filter for walking activity and sort by user ID
walking_df = df[df['activity'] == 'Walking'].sort_values('user')
driving_df = df[df['activity'] == 'Driving'].sort_values('user')
inactive_df = df[df['activity'] == 'Inactive'].sort_values('user')
active_df = df[df['activity'] == 'Active'].sort_values('user')

# Group by users and aggregate data
summary_stats = active_df.groupby('user')[x].agg(['mean'])
summary_stats.columns = ['{}_{}'.format(col, agg) for col, agg in summary_stats.columns]

# Remove the duplicated 'mean' from column names where necessary
summary_stats.columns = [re.sub(r'_(mean)(?=_mean)', '', col) for col in summary_stats.columns]
summary_stats.drop(['user_mean'], axis=1, inplace=True) #user_std

# walk_by_user = [i for _, i in walking_df.groupby('user')[x]]
# drive_by_user = [i for _, i in driving_df.groupby('user')[x]]
# inactive_by_user = [i for _, i in inactive_df.groupby('user')[x]]
# active_by_user = [i for _, i in active_df.groupby('user')[x]]

# # Create subplots
# fig, axs = plt.subplots(3, 5, figsize=(10, 5), facecolor='w', edgecolor='k', sharey=True)
# axs = axs.ravel()

# # Collect all handles and labels for legend
# all_handles = []
# all_labels = []

# # Initialize lists to track axis limits
# x_limits = [float('inf'), float('-inf')]
# y_limits = [float('inf'), float('-inf')]
# z_limits = [float('inf'), float('-inf')]

# # Plot data for each user
# axes = []
# for i in range(len(walk_by_user)):
#     ax = fig.add_subplot(3, 5, i+1, projection='3d')
#     axes.append(ax)

#     userData = walk_by_user[i]
#     user_id = userData['user'].iloc[0]

#     # Walking data
#     userData = walking_df[walking_df['user'] == user_id]
#     if not userData.empty:
#         scatter = ax.scatter(userData['gps_lat_mean'], userData['gps_long_mean'], userData['gps_alt_mean'], 
#                              label='GPS measurements: walking', s=10)
#         if 'GPS measurements: walking' not in all_labels:
#             all_handles.append(scatter)
#             all_labels.append('GPS measurements: walking')

#         # Update axis limits
#         x_limits = [min(x_limits[0], userData['gps_lat_mean'].min()), max(x_limits[1], userData['gps_lat_mean'].max())]
#         y_limits = [min(y_limits[0], userData['gps_long_mean'].min()), max(y_limits[1], userData['gps_long_mean'].max())]
#         z_limits = [min(z_limits[0], userData['gps_alt_mean'].min()), max(z_limits[1], userData['gps_alt_mean'].max())]

#     # Driving data
#     drivingData = driving_df[driving_df['user'] == user_id]
#     if not drivingData.empty:
#         scatter = ax.scatter(drivingData['gps_lat_mean'], drivingData['gps_long_mean'], drivingData['gps_alt_mean'], 
#                              label='GPS measurements: driving', s=10)
#         if 'GPS measurements: driving' not in all_labels:
#             all_handles.append(scatter)
#             all_labels.append('GPS measurements: driving')

#         # Update axis limits
#         x_limits = [min(x_limits[0], drivingData['gps_lat_mean'].min()), max(x_limits[1], drivingData['gps_lat_mean'].max())]
#         y_limits = [min(y_limits[0], drivingData['gps_long_mean'].min()), max(y_limits[1], drivingData['gps_long_mean'].max())]
#         z_limits = [min(z_limits[0], drivingData['gps_alt_mean'].min()), max(z_limits[1], drivingData['gps_alt_mean'].max())]

    # # Active data
    # activeData = active_df[active_df['user'] == user_id]

    # if activeData.empty:
    #     activeData.dropna
    #     scatter = ax.scatter(activeData['acc_xs_mean'], activeData['acc_ys_mean'], activeData['acc_zs_mean'], 
    #                          label='Mean active acceleration', s=10)
    #     if 'Mean active acceleration' not in all_labels:
    #         all_handles.append(scatter)
    #         all_labels.append('Mean active acceleration')

    #     # Update axis limits
    #     x_limits = [min(x_limits[0], activeData['acc_xs_mean'].min()), max(x_limits[1], activeData['acc_xs_mean'].max())]
    #     y_limits = [min(y_limits[0], activeData['acc_ys_mean'].min()), max(y_limits[1], activeData['acc_ys_mean'].max())]
    #     z_limits = [min(z_limits[0], activeData['acc_zs_mean'].min()), max(z_limits[1], activeData['acc_zs_mean'].max())]
    
    # # Inactive data
    # inActiveData = inactive_df[inactive_df['user'] == user_id]
    # if not inActiveData.empty:
    #     scatter = ax.scatter(inActiveData['acc_xs_mean'], inActiveData['acc_ys_mean'], inActiveData['acc_zs_mean'], 
    #                          label='Mean inactive acceleration', s=10)
    #     if 'Mean inactive acceleration' not in all_labels:
    #         all_handles.append(scatter)
    #         all_labels.append('Mean inactive acceleration')

    #     # Update axis limits
    #     x_limits = [min(x_limits[0], inActiveData['acc_xs_mean'].min()), max(x_limits[1], inActiveData['acc_xs_mean'].max())]
    #     y_limits = [min(y_limits[0], inActiveData['acc_ys_mean'].min()), max(y_limits[1], inActiveData['acc_ys_mean'].max())]
    #     z_limits = [min(z_limits[0], inActiveData['acc_zs_mean'].min()), max(z_limits[1], inActiveData['acc_zs_mean'].max())]

#     ax.set_title(f"User {user_id}")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.grid(True)

#     ax.view_init(elev=30, azim=45)

# # Apply the same axis limits to all subplots
# for ax in axes:
#     ax.set_xlim(x_limits)
#     ax.set_ylim(y_limits)
#     ax.set_zlim(z_limits)
#     ax.tick_params(axis='x', rotation=45)
#     ax.tick_params(axis='y', rotation=45)
#     ax.tick_params(axis='z', rotation=45)

# # Create a single legend for the entire figure with larger markers
# class HandlerSize(HandlerPathCollection):
#     def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
#         markersize = 10  # Change this to adjust legend marker size
#         return super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, markersize, trans)

# fig.legend(all_handles, all_labels, loc='lower right', ncol=1, fontsize='small',
#            handler_map={plt.Line2D: HandlerSize()})

# for ax in axs:
#     if not ax.has_data():
#         fig.delaxes(ax)

# plt.subplots_adjust(hspace=0.2)
# plt.subplots_adjust(wspace=0.5)
# fig.suptitle('Mean accelerometer values for different mobility types')
# fig.tight_layout()
# plt.show()

#################################

# # Plot data for each user
# for i in range(len(walk_by_user)):
#     userData = walk_by_user[i]
#     user_id = userData['user'].iloc[0]
#     stepsUD = len(userData['gps_speed_mean'])

#     # Walking data
#     scatter = axs[i].scatter(userData['gps_lat_mean'], userData['gps_long_mean'], label='Walking', s=2, color= '#581845')

#     if scatter.get_label() not in all_labels:
#         all_handles.append(scatter)
#         all_labels.append(scatter.get_label())

#     # Driving data
#     drivingData = driving_df[driving_df['user'] == user_id]
#     if not drivingData.empty:
#         stepsDD = len(drivingData['gps_speed_mean'])

#         scatter = axs[i].scatter(drivingData['gps_lat_mean'], drivingData['gps_long_mean'], label='Driving', s=2)

#         if scatter.get_label() not in all_labels:
#             all_handles.append(scatter)
#             all_labels.append(scatter.get_label())

#     # Active data
#     activityData = active_df[active_df['user'] == user_id]
#     if not activityData.empty:
#         stepsAD = len(activityData['gps_speed_mean'])
#         scatter = axs[i].scatter(activityData['gps_lat_mean'], activityData['gps_long_mean'], label='Active', s=2, color='#5F24BA')
#         if scatter.get_label() not in all_labels:
#             all_handles.append(scatter)
#             all_labels.append(scatter.get_label())

#     # Inactive data
#     iNactivityData = inactive_df[inactive_df['user'] == user_id]
#     if not iNactivityData.empty:
#         stepsID = len(iNactivityData['gps_speed_mean'])
#         scatter = axs[i].scatter(iNactivityData['gps_lat_mean'], iNactivityData['gps_long_mean'], label='Inactive', s=2, color='#5893E4')
#         if scatter.get_label() not in all_labels:
#             all_handles.append(scatter)
#             all_labels.append(scatter.get_label())

#     axs[i].set_title(f"User {user_id}")
#     # axs[i].set_xscale("log")

# # Remove empty subplots
# for ax in axs:
#     if not ax.has_data():
#         fig.delaxes(ax)

# # Create a single legend for the entire figure
# fig.legend(all_handles, all_labels, loc = 'lower right')
# fig.tight_layout()

# fig.text(0.5, 0.01, 'Change in Latitude', ha='center')
# fig.text(0.01, 0.5, 'Change in Longitude', va='center', rotation='vertical')
# # fig.suptitle('Latitude vs Longitude')
# plt.show()

##########################################

print(summary_stats.iloc[:2, :6])

# Remove columns with zero standard deviation
# non_constant_columns = summary_stats.columns[summary_stats.std() != 0]
# summary_stats = summary_stats[non_constant_columns]
summary_stats.drop(columns=[col for col in summary_stats.columns if 'user' in col], axis=1, inplace=True)
summary_stats.drop(columns=[col for col in summary_stats.columns if 'id' in col], axis=1, inplace=True)

covariance = summary_stats[['gps_alt_mean', 'gps_lat_mean', 'gps_long_mean']].cov()
# print(covariance)
# Compute the correlation matrix, fill NaN with 0
corr_matrix = summary_stats.corr().fillna(0)

# Replace infinite values with 0, if any
corr_matrix.replace([np.inf, -np.inf], 0, inplace=True)

# Generate and display the cluster map
sns.set(font_scale=1.25)
sns.clustermap(corr_matrix, cmap="coolwarm", standard_scale=12)
plt.show()