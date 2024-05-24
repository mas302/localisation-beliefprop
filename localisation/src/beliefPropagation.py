import numpy as np
import random
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from discretise import *
from pgmpy.inference import BeliefPropagation
import seaborn as sns
import math
import scipy
from scipy.ndimage.filters import gaussian_filter1d

random.seed(0)

G =  FactorGraph()
nodes = ['true_loc', 'est_loc', 'alt', 's', 'b', 'del_lat', 'del_lon', 'acc', 'acc_dash']
G.add_nodes_from(nodes)

mean1 = [34.785, 0.002608, 0.002865]
covariance1 = [[242.089061, -8.869430e-04, 2.275028e-03], [-0.000887, 6.786957e-07, 5.068323e-07], [0.002275, 5.068323e-07, 5.241057e-07]]
std_lat = 0.006804
std_lon = 0.011514 
std_alt = 38.646093

altitude_ranges1 = np.arange(mean1[0]-2*std_alt, mean1[0]+2*std_alt, 8)
delta_lat_range1 = np.arange(mean1[1]-2*std_lat, mean1[1]+2*std_lat, 0.002) 
delta_lon_range1 = np.arange(mean1[2]-2*std_lon, mean1[2]+2*std_lon, 0.003)

prob_table = discretise3vars(mean1, covariance1, delta_lat_range1, delta_lon_range1, altitude_ranges1, 5)
print(len(altitude_ranges1), len(delta_lon_range1), len(delta_lat_range1))
print(prob_table.shape)

altLatLon =  DiscreteFactor(['del_lat', 'del_lon', 'alt'], [14,16,20], prob_table)
true_loc = 100 + 2.5 * np.random.randn(1000)
est_loc = 12 + 5 * np.random.randn(1000)

print("progress check1")

predUELocation = discretise(est_loc, true_loc, 20)
predUETabularCPD = conditionalProbabilityTable(predUELocation)
trueLocationFactor = DiscreteFactor(['true_loc', 'est_loc'], [20, 20], predUETabularCPD)

print("progress check2")

alt = mean1[0] + (std_alt**2) *np.random.randn(1000)
accuracyEstimation = discretise(est_loc, alt, 20)
accuracyEstimationCPD = conditionalProbabilityTable(accuracyEstimation)
accuracyEstFactor = DiscreteFactor(['est_loc', 'alt'], [20, 20], accuracyEstimationCPD)

mean2 = [34.785, 5.5, 10]
covariance2 = [[1,1,1], [1,2,3], [1,4,5]]
std_speed = 0.5
std_bearing = 2

speed_range = np.arange(mean2[1]-2*std_speed, mean2[1]+2*std_speed, 0.1) 
bearing_range = np.arange(mean2[2]-2*std_bearing, mean2[2]+2*std_bearing, 0.4)

print(len(speed_range), len(bearing_range))

prob_table_sb = discretise3vars(mean2, covariance2, speed_range, bearing_range, altitude_ranges1, 5)
altSpeedBearing = DiscreteFactor(['alt', 's', 'b'], [20, 20, 20], prob_table_sb)

print("progress check3")

acc = 100 + 10 * np.random.randn(1000)
lat =  mean1[1] + (std_lat**2) *np.random.randn(1000)
latAccu = discretise(lat, acc, 14)
latAccuCPD = conditionalProbabilityTable(latAccu)
latAcc = DiscreteFactor(['del_lat', 'acc'], [14, 14], latAccuCPD)

print("progress check4")

lon = mean1[2] + (std_lon**2) *np.random.randn(1000)
acc_dash = 100 + 10 * np.random.randn(1000)
lonAccu = discretise(lon, acc_dash, 16)
lonAccuracyCPD = conditionalProbabilityTable(lonAccu)
lonAcc = DiscreteFactor(['del_lon', 'acc_dash'], [16,16], lonAccuracyCPD)

print("progress check5")

factors = [trueLocationFactor, altLatLon, accuracyEstFactor, altSpeedBearing, latAcc, lonAcc]
G.add_nodes_from(nodes)
G.add_factors(trueLocationFactor, altLatLon, accuracyEstFactor, altSpeedBearing, latAcc, lonAcc)
G.add_edges_from([('s', altSpeedBearing), ('b', altSpeedBearing), ('alt', altSpeedBearing),
                  ('true_loc', trueLocationFactor), ('est_loc', trueLocationFactor),
                  ('est_loc', accuracyEstFactor), ('alt', accuracyEstFactor),
                  ('del_lat', latAcc), ('acc', latAcc),
                  ('del_lon', lonAcc), ('acc_dash', lonAcc),
                  ('alt', altLatLon), ('del_lat', altLatLon), ('del_lon', altLatLon)])

print(G.check_model())

bp = BeliefPropagation(G)
bp.calibrate()
mapEstimate = bp.query(variables=['est_loc', 'true_loc'], show_progress=True, evidence = {'del_lat': 10,
                                                                                'del_lon':10,
                                                                                's':10,
                                                                                'b':10,
                                                                                'alt': 10,
                                                                                'acc':10,
                                                                                'acc_dash':10})

mapEstimate2 = bp.query(variables=['est_loc', 'true_loc'], show_progress=True, evidence = {'del_lat': 5,
                                                                                'del_lon':5,
                                                                                's':5,
                                                                                'b':5,
                                                                                'alt': 5,
                                                                                'acc':5,
                                                                                'acc_dash':5})

probDist, _ = np.histogram(mapEstimate.sample(1000), bins = 100, density=True)
probDist2, _ = np.histogram(mapEstimate2.sample(1000), bins = 100, density=True)
# print(probDist)
y = scipy.special.kl_div(probDist, probDist2)

from scipy.interpolate import interp1d
x=  np.linspace(0, len(y), 100)
xnew= np.linspace(0, len(y), 200)
f_cubic = interp1d(x, y, kind='slinear')
plt.plot(xnew, f_cubic(xnew))
plt.show()

# ME_samp = mapEstimate.sample(1_000_000)
# ME_samp2 = mapEstimate2.sample(1_000_000)
# diff = ME_samp.est_loc - ME_samp.true_loc
# diff2 = ME_samp2.est_loc - ME_samp2.true_loc
# diff.hist(
#     alpha=0.3,
#     label=f"Estimated Location, bins=30 AVG={diff.mean()}",
#     density=True,
# )

# diff2.hist(
#     alpha=0.3,
#     label=f"Estimated Location, bins=30 AVG={diff.mean()}",
#     density=True,
# )
# print(mapEstimate)

# plt.legend()
# plt.show()
# trueEstimate = bp.query(['true_loc'], show_progress=True)
# print(mapEstimate)
# estimate = bp.query(variables=['x', 'y'], evidence={'a':0}, joint=True, show_progress=True)
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

