import numpy as np
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from discretise import *
from pgmpy.inference import BeliefPropagation
import seaborn as sns
import random
import scipy
from scipy.ndimage import uniform_filter1d
import sklearn

plt.rcParams['axes.grid'] = True

random.seed(0)
np.random.seed(0)

G =  FactorGraph()
nodes = ['true_loc', 'est_loc', 'alt', 's', 'b', 'del_lat', 'del_lon', 'acc', 'acc_dash', 'x_acc']
G.add_nodes_from(nodes)

mean1 = [34.785, 0.002608, 0.002865]
covariance1 = [[242.089061, -8.869430e-04, 2.275028e-03], [-0.000887, 6.786957e-07, 5.068323e-07], [0.002275, 5.068323e-07, 5.241057e-07]]
std_lat = 0.006804
std_lon = 0.011514 
std_alt = 38.646093

altitude_ranges1 = np.arange(mean1[0]-2*std_alt, mean1[0]+2*std_alt, 16)
delta_lat_range1 = np.arange(mean1[1]-2*std_lat, mean1[1]+2*std_lat, 0.003)
delta_lon_range1 = np.arange(mean1[2]-2*std_lon, mean1[2]+2*std_lon, 0.005)

prob_table = discretise3vars(mean1, covariance1, delta_lat_range1, delta_lon_range1, altitude_ranges1, 5)

altLatLon =  DiscreteFactor(['del_lat', 'del_lon', 'alt'], [10,10,10], prob_table)
true_loc = 10 + np.random.randn(10000)
est_loc = 10 + np.random.randn(10000)

print("progress check1")

predUELocation = discretise(est_loc, true_loc, 10)
predUETabularCPD = conditionalProbabilityTable(predUELocation)
trueLocationFactor = DiscreteFactor(['true_loc', 'est_loc'], [10, 10], predUETabularCPD)

print("progress check2")

alt = mean1[0] + (std_alt) *np.random.randn(10000)
accuracyEstimation = discretise(est_loc, alt, 10)
accuracyEstimationCPD = conditionalProbabilityTable(accuracyEstimation)
accuracyEstFactor = DiscreteFactor(['est_loc', 'alt'], [10, 10], accuracyEstimationCPD)

mean2 = [34.785, 5.5, 10]
covariance2 = [[1,1,1], [1,2,3], [1,4,5]]
std_speed = 0.5
std_bearing = 2

speed_range = np.arange(mean2[1]-2*std_speed, mean2[1]+2*std_speed, 0.2) 
bearing_range = np.arange(mean2[2]-2*std_bearing, mean2[2]+2*std_bearing, 0.8)

prob_table_sb = discretise3vars(mean2, covariance2, speed_range, bearing_range, altitude_ranges1, 5)
altSpeedBearing = DiscreteFactor(['alt', 's', 'b'], [10, 10, 10], prob_table_sb)

speed = mean2[1]+ np.random.randn(10000)*std_speed
x_acc = 0.001 + np.random.randn(10000)*2
speedXAcc = discretise(speed, x_acc, 10)
speedXAccCPD = conditionalProbabilityTable(speedXAcc)
speedXAccFac = DiscreteFactor(['s', 'x_acc'], [10,10], speedXAccCPD)

print("progress check3")

acc = 100 + 10 * np.random.randn(10000)
lat =  mean1[1] + (std_lat**2) *np.random.randn(10000)
latAccu = discretise(lat, acc, 10)
latAccuCPD = conditionalProbabilityTable(latAccu)
latAcc = DiscreteFactor(['del_lat', 'acc'], [10, 10], latAccuCPD)

print("progress check4")

lon = mean1[2] + (std_lon**2) *np.random.randn(10000)
acc_dash = 50 + 10 * np.random.randn(10000)
lonAccu = discretise(lon, acc_dash, 10)
lonAccuracyCPD = conditionalProbabilityTable(lonAccu)
lonAcc = DiscreteFactor(['del_lon', 'acc_dash'], [10,10], lonAccuracyCPD)

print("progress check5")

factors = [trueLocationFactor, altLatLon, accuracyEstFactor, altSpeedBearing, latAcc, lonAcc, speedXAccFac]
G.add_nodes_from(nodes)
G.add_factors(trueLocationFactor, altLatLon, accuracyEstFactor, altSpeedBearing, latAcc, lonAcc, speedXAccFac)
G.add_edges_from([('s', altSpeedBearing), ('b', altSpeedBearing), ('alt', altSpeedBearing),
                  ('true_loc', trueLocationFactor), ('est_loc', trueLocationFactor),
                  ('est_loc', accuracyEstFactor), ('alt', accuracyEstFactor),
                  ('del_lat', latAcc), ('acc', latAcc),
                  ('del_lon', lonAcc), ('acc_dash', lonAcc),
                  ('alt', altLatLon), ('del_lat', altLatLon), ('del_lon', altLatLon),
                  ('s', speedXAccFac), ('x_acc', speedXAccFac)])

print(G.check_model())

G_spoofed = FactorGraph()

# Spoofed scenario (est_loc biased away from true_loc)
spoofed_est_loc = true_loc + np.random.randn(10000)
spoofedUELocation = discretise(spoofed_est_loc, true_loc, 10)
spoofedUETabularCPD = conditionalProbabilityTable(spoofedUELocation)
spoofedLocationFactor = DiscreteFactor(['spoofed_loc', 'est_loc'], [10, 10], spoofedUELocation)

G_spoofed.add_nodes_from(['spoofed_loc', 'est_loc', 'alt', 's', 'b', 'del_lat', 'del_lon', 'acc', 'acc_dash', 'x_acc'])
G_spoofed.add_factors(spoofedLocationFactor, altLatLon, accuracyEstFactor, altSpeedBearing, latAcc, lonAcc, speedXAccFac)
G_spoofed.add_edges_from([('s', altSpeedBearing), ('b', altSpeedBearing), ('alt', altSpeedBearing),
                  ('spoofed_loc', spoofedLocationFactor), ('est_loc', spoofedLocationFactor),
                  ('est_loc', accuracyEstFactor), ('alt', accuracyEstFactor),
                  ('del_lat', latAcc), ('acc', latAcc),
                  ('del_lon', lonAcc), ('acc_dash', lonAcc),
                  ('alt', altLatLon), ('del_lat', altLatLon), ('del_lon', altLatLon),
                  ('s', speedXAccFac), ('x_acc', speedXAccFac)])

print(G_spoofed.check_model())

bp = BeliefPropagation(G)
bp_spoof = BeliefPropagation(G_spoofed)

bp.calibrate()
bp_spoof.calibrate()

non_spoofed_mapEstimate = bp.map_query(variables=['est_loc'], show_progress=True, evidence={'true_loc': 5})
print(f"Non-spoofed MAP estimate: {non_spoofed_mapEstimate}")

# Perform inference for spoofed scenario
spoofed_mapEstimate = bp_spoof.map_query(variables=['spoofed_loc'], show_progress=True, evidence={'true_loc': 5})
print(f"Spoofed MAP estimate: {spoofed_mapEstimate}")

def map_evidence_to_state(evidence, bins):
    bin_edges = np.linspace(min(evidence), max(evidence), bins)
    discrete_states = np.digitize(evidence, bin_edges) - 1  # Convert to zero-based index
    unique_states=list(set(discrete_states))
    return unique_states


discrete_true_vals = map_evidence_to_state(true_loc, 10)

print("we're here")
non_spoofed_samples = np.array([bp.map_query(variables=['est_loc'], evidence={'true_loc': val})['est_loc'] for val in discrete_true_vals])
print(non_spoofed_samples)
print("we're here")
spoofed_samples = np.array([bp_spoof.map_query(variables=['spoofed_loc'], evidence={'true_loc': val})['spoofed_loc'] for val in discrete_true_vals])
print(spoofed_samples)

# map_states = bp.map_query(variables=['alt', 's', 'b', 'del_lat', 'del_lon'], show_progress=True)
queryEst = bp.query(variables=['est_loc'], show_progress=True)
spoofedQueryEst = bp_spoof.query(variables=['spoofed_loc'], show_progress=True)
# mapEstimate2 = bp.map_query(variables=['est_loc'], show_progress=True, evidence = {'true_loc':19})
# mapEstimate3 = bp.map_query(variables=['est_loc'], show_progress=True, evidence = {'true_loc':10})
print(queryEst)
print(queryEst.values)

states = np.arange(10)
num_samples = 10000
probabilities= queryEst.values
probabilities_spoofed = spoofedQueryEst.values
samples = np.random.choice(states, size=num_samples, p=probabilities)
samplesSpoofed = np.random.choice(states, size=num_samples, p=probabilities_spoofed)

state_min, state_max = states.min(), states.max()
original_values = np.linspace(state_min, state_max, len(states))

# Calculate the original evidence values corresponding to the sampled indices
sampled_values = original_values[samples]
sampled_values_spoof = original_values[samplesSpoofed]

spoof = np.linspace(11, 19, 10)
nonspoof= np.linspace(8, 12, 10)
print(nonspoof[3])

print(np.mean(sampled_values), np.mean(sampled_values_spoof))
# Scale the sampled values to match the known mean and standard deviation
scaled_sampled_values = (sampled_values - sampled_values.mean()) + nonspoof[3]
scaled_sampled_values_spoofed = (sampled_values_spoof - sampled_values_spoof.mean()) + nonspoof[3]

# Compute the mean and standard deviation of the scaled sampled array
# mean = np.mean(scaled_sampled_values)
# std_dev = np.std(scaled_sampled_values)
# meanS = np.mean(scaled_sampled_values_spoofed)
# std_devS = np.std(scaled_sampled_values_spoofed)

print(f"Sampled Values: {scaled_sampled_values[:10]}...")  # Display first 10 samples for verification
# print(f"Mean of the scaled sampled array: {mean}, {meanS}")
# print(f"Standard deviation of the scaled sampled array: {std_dev}, {std_devS}")

# Plot non-spoofed results
fig, axes = plt.subplots(1, 2, figsize= (7,5), sharey = 'all')
x = np.linspace(0,10,10)
# axes[0].plot(x, non_spoofed_samples, label='Non-Spoofed estimate')
# axes[0].plot(x, spoofed_samples,label='Spoofed estimate')
# axes[0].set_ylim(bottom=0)
# axes[0].set_xlabel('States')
# axes[0].set_ylabel('MAP estimate')
print(np.std(sampled_values), np.std(sampled_values_spoof))

axes[0].hist(scaled_sampled_values, alpha=0.5, density=True, label=f'True Location: {round(nonspoof[3], 2)}')
axes[0].hist(scaled_sampled_values_spoofed, alpha=0.5, density=True, label=f'Non-Spoofed Location: {round(nonspoof[3], 2)}')
axes[0].set_ylabel('Marginal Probability')
x2 = np.linspace(0, 10000, 10000).T
print(scipy.special.kl_div(sampled_values, sampled_values_spoof).size)

from scipy.signal import savgol_filter
y= scipy.special.rel_entr(probabilities,probabilities_spoofed)
yhat = savgol_filter(y, 10, 2)

y_smooth = uniform_filter1d(y,size=3)
axes[1].hist(scaled_sampled_values-scaled_sampled_values_spoofed, density=True, label=f'Error', alpha=0.5, color = 'red')
axes[1].set_ylabel('Error (spoofed-true)')
fig.supxlabel('Values')
fig.legend()
plt.show()

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
mapestimates = np.zeros((20,10))

# Generate an array of 10 random numbers from a normal distribution with mean 0 and standard deviation 1
seeds = np.random.randint(1, 100, size=20)

# for i in range(20):
#     for j in range(10):
#         np.random.seed(seeds[i])
#         mapest = bp.map_query(variables=['est_loc'], show_progress=True, evidence = {'alt': i})
#         mapestimates[i][j] = list(mapest.values())[0]
        
# mean = np.mean(mapestimates, axis=1)
# std_dev= np.std(mapestimates, axis=1)
# print(std_dev)
# axes[0].plot(mean)
# # axes[0].fill_between(np.linspace(0,20,20), mean-2*std_dev, mean+2*std_dev, alpha=0.5, color = 'orange')
# axes[0].set_ylim(bottom=0)
# axes[0].set_title('MAP state choice')

# axes[1].hist(mapestimates.reshape(-1), bins='auto', density=True)
# axes[1].set_title('MAP "EST" state histogram')
# fig.supxlabel('"NET" state evidence')
# fig.supylabel('"EST" estimated state')
# plt.legend()
# plt.show()
    
# plt.scatter(np.linspace(0,20,20), mapestimates)

# map_states = bp.map_query(variables=['alt', 's', 'b', 'del_lat', 'del_lon'], show_progress=True)
# queryEst = bp.query(variables=['est_loc', 'true_loc'], show_progress=True, evidence = map_states)
# # mapEstimate2 = bp.map_query(variables=['est_loc'], show_progress=True, evidence = {'true_loc':19})
# # mapEstimate3 = bp.map_query(variables=['est_loc'], show_progress=True, evidence = {'true_loc':10})
# ME_samp = queryEst.sample(100000)
# # ME_samp2 = mapEstimate2.sample(100000)
# # ME_samp3 = mapEstimate3.sample(100000)
# diff2 =  ME_samp.est_loc - ME_samp.true_loc
# # print(np.sum(scipy.special.kl_div(ME_samp.est_loc, ME_samp.true_loc)))

# ME_samp.est_loc.hist(
#     alpha=0.3,
#     label=f"Estimated Location, AVG={ME_samp.est_loc.mean()}",
#     density=True,
# )

# ME_samp.true_loc.hist(
#     alpha=0.3,
#     label=f"True Location, AVG={ME_samp.true_loc.mean()}",
#     density=True,
# )

# diff3.hist(
#     alpha=0.3,
#     label=f"Estimated Location, AVG={diff2.mean()}",
#     density=True,
# )
# plt.ylabel('Marginal Probability')
# plt.xlabel('States')
# plt.legend()
# plt.show()

# y = scipy.special.kl_div(diff, diff2)

# Compare a larger clique against the true marginals.

# # true_loc_marginal = observed_factor_dict[("visitorTeamAbbr",)]
# total_states = len(true_loc)
# axes[0].bar(
#     range(total_states),
#     true_loc.values.flatten(),
# )

# axes[0].set_title("Observed Marginals")
# axes[0].bar(
#     range(20),
#     trueLocationFactor.marginalize(
#         ['true_loc'], inplace=False
#     ).values.flatten(),
# )
# axes[1].bar(
#     range(20),
#     trueLocationFactor.marginalize(
#         ['est_loc'], inplace=False
#     ).values.flatten(),
# )
# axes[2].bar(
#     range(20),
#     altLatLon.marginalize(
#         ['alt', 'del_lat', 'del_lon'], inplace=False
#     ).values.flatten(),
# )
# axes[3].bar(
#     range(20),
#     altSpeedBearing.marginalize(
#         ['s', 'b'], inplace=False
#     ).values.flatten(),
# )
# axes[0].set_title("Marginalised 'true_loc'")
# axes[1].set_title("Marginalised 'est_loc'")
# axes[2].set_title("Marginalised 'alt'")
# axes[3].set_title("Marginalised 'alt', speed, bearing")
# plt.show()



# trueEstimate = bp.query(['true_loc'], show_progress=True)
# print(mapEstimate)

# estimateSampled, _ = np.histogram(mapEstimate.sample(1_000_000))
# trueSampled, _= np.histogram(trueEstimate.sample(1_000_000))
# print(scipy.special.rel_entr(estimateSampled, trueSampled))
# print(estimateSampled)
# print(trueSampled)
# print(scipy.special.rel_entr(estimateSampled, trueSampled))
# plt.hist(estimateSampled,
#     alpha=0.3,
#     label=f"Estimated Location, bins=30 AVG={estimateSampled.mean()}",
#     density=True,
# )
# plt.hist(trueSampled,
#     alpha=0.3,
#     label=f"Estimated Location, bins=30 AVG={trueSampled.mean()}",
#     density=True,
# )

# plt.legend()
# plt.show()

# if __name__ == '__main__':
#     # Print the structure of the Graph to verify its correctness.
#     plt.figure(figsize=(10, 3))
#     top = {team: (i * 2, 0) for i, team in enumerate(nodes)}
#     bottom = {factor: (i, 1) for i, factor in enumerate(factors)}
#     # Draw all the variables & factors with their edges.
#     nx.draw(
#     G,
#     pos={**top, **bottom},
#     edge_color="red",)
    
#     # Draw text labels for the factors above their nodes in the graph.
#     label_dict = {factor: "{" + ",\n".join(factor.scope()) + "}" for factor in G.factors}
#     for node, (x, y) in bottom.items():
#         plt.text(x, y * 1.2, label_dict[node], fontsize=10, ha="center", va="center")
    
#     # Re-draw the variables but with labels this time and colored orange.
#     nx.draw(
#         G.subgraph(nodes),
#         node_color="orange",
#         pos={**top},
#         with_labels=True,)
    
#     plt.show()

