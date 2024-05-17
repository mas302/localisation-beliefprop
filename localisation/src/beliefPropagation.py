import numpy as np
import random
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from discretise import *
from pgmpy.inference import BeliefPropagation

import numpy as np
from scipy.stats import multivariate_normal, norm

random.seed(0)

mean = [1000, 0, 0]  # Example means for altitude, Δlatitude, Δlongitude
covariance = [[400, 10, 10], [10, 5, 2], [10, 2, 5]]  # Example covariance matrix

delta_lat_range = np.arange(-1, 1.1, 0.2) 
delta_lon_range = np.arange(-1, 1.1, 0.2)

sigma_11 = covariance[0][0]
sigma_12 = np.array(covariance[0][1:3])  # Covariance between x1 and x2, x3 (1x2 array)
sigma_21 = np.array([row[0] for row in covariance[1:3]]).reshape(2, 1)  # Covariance between x2, x3 and x1 (2x1 array)
sigma_22 = np.array([row[1:3] for row in covariance[1:3]])  # 2x2 covariance matrix for x2 and x3
sigma_22_inv = np.linalg.inv(sigma_22)  

sigma_cond = sigma_11 - sigma_12.dot(sigma_22_inv).dot(sigma_21)

altitude_ranges = np.arange(500, 1550, 50)
prob_table = np.zeros((len(delta_lat_range), len(delta_lon_range), len(altitude_ranges)))

for i, delta_lat in enumerate(delta_lat_range):
    for j, delta_lon in enumerate(delta_lon_range):
        cond_mean = mean[0] + sigma_12.dot(sigma_22_inv).dot([delta_lat - mean[1], delta_lon - mean[2]])
        
        for k, alt in enumerate(altitude_ranges):
            lower_bound = alt - 25  
            upper_bound = alt + 25  
            prob = norm.cdf(upper_bound, cond_mean, np.sqrt(sigma_cond)) - norm.cdf(lower_bound, cond_mean, np.sqrt(sigma_cond))
            prob_table[i, j, k] = prob

print(prob_table.size)

center_lat_index = (np.abs(delta_lat_range - 0)).argmax()
center_lon_index = (np.abs(delta_lon_range - 0)).argmax()
print("Conditional Probability Table (sample slice for Δlatitude = 0 and Δlongitude = 0):")
print(sum(prob_table[center_lat_index, center_lon_index]))

G =  FactorGraph()
nodes = ['a', 'y', 'x']
G.add_nodes_from(nodes)

ueFactor = DiscreteFactor(['a', 'x'], [11, 11], np.random.rand(121))
net =  DiscreteFactor(['y'], [21], np.random.rand(21))
pred = DiscreteFactor(['x','y'], [11, 21], np.random.rand(231))

userEquipment = 3 + 2.5 * np.random.randn(1000)
sensorValues= 100 - 5 * np.random.randn(1000)

predUE = discretise(sensorValues, userEquipment, 11)
predUETabularCPD = conditionalProbabilityTable(predUE)
predUETabularCPD.shape
ueDiscretisedFactor =  DiscreteFactor(['a', 'x'], [11, 11], predUETabularCPD)
threeWayFactor = DiscreteFactor(['a', 'x', 'y'], [11, 11, 21], prob_table)

factors = [ueDiscretisedFactor, net, pred, threeWayFactor] 

G.add_nodes_from([ueDiscretisedFactor, net, pred, threeWayFactor])
G.add_factors(ueDiscretisedFactor, net, pred, threeWayFactor)
G.add_edges_from([('a', ueDiscretisedFactor), ('x', ueDiscretisedFactor), 
                  ('x', pred), ('y', pred), 
                  ('y', net),
                  ('a', threeWayFactor), ('x', threeWayFactor), ('y', threeWayFactor)])

print(G.check_model())

bp = BeliefPropagation(G)
bp.calibrate()
print(bp.get_clique_beliefs())
print(bp.map_query(['x', 'y'], show_progress=True, evidence = {'a': 10}))
estimate = bp.query(variables=['x', 'y'], evidence={'a':0}, joint=True, show_progress=True)
estimateSampled = estimate.sample(1_000_000)
estimateSampled.hist(
    alpha=0.3,
    label=f"KC (Home) - SF (Away), bins=30 AVG={estimateSampled.mean()}",
    density=True,
)

plt.title("Arbitrary hist")
plt.legend()
plt.show()

# if __name__ == '__main__':
#     # Print the structure of the Graph to verify its correctness.
#     plt.figure(figsize=(10, 3))
#     top = {team: (i * 2, 0) for i, team in enumerate(sorted(nodes))}
#     bottom = {factor: (i, 1) for i, factor in enumerate(factors)}
#     pos1 = nx.nx_agraph.graphviz_layout(G, prog="dot")
#     # Draw all the variables & factors with their edges.
#     nx.draw(
#     G,
#     pos= pos1, #{**top, **bottom},
#     edge_color="red",)
    
#     # Draw text labels for the factors above their nodes in the graph.
#     label_dict = {factor: "{" + ",\n".join(factor.scope()) + "}" for factor in G.factors}
#     for node, (x, y) in bottom.items():
#         plt.text(x, y * 1.2, label_dict[node], fontsize=10, ha="center", va="center")
    
#     # Re-draw the variables but with labels this time and colored orange.
#     nx.draw(
#         G.subgraph(nodes),
#         node_color="orange",
#         pos=pos1,#{**top},
#         with_labels=True,)
    
#     plt.show()

