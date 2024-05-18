import numpy as np
import random
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from discretise import *
from pgmpy.inference import BeliefPropagation

random.seed(0)

mean1 = [1000, 0, 0]  # Example means for altitude, Δlatitude, Δlongitude
covariance1 = [[400, 10, 10], [10, 5, 2], [10, 2, 5]]  # Example covariance matrix
delta_lat_range1 = np.arange(-1, 1.1, 0.2) 
delta_lon_range1 = np.arange(-1, 1.1, 0.2)
altitude_ranges1 = np.arange(500, 1550, 50)

prob_table = discretise3vars(mean1, covariance1, delta_lat_range1, delta_lon_range1, altitude_ranges1, 50)

# center_lat_index = (np.abs(delta_lat_range - 0)).argmin()
# center_lon_index = (np.abs(delta_lon_range - 0)).argmin()
# print("Conditional Probability Table (sample slice for Δlatitude = 0 and Δlongitude = 0):")
# print(sum(prob_table[center_lat_index, center_lon_index]))

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

# estimate = bp.query(variables=['x', 'y'], evidence={'a':0}, joint=True, show_progress=True)
# estimateSampled = estimate.sample(1_000_000)
# estimateSampled.hist(
#     alpha=0.3,
#     label=f"KC (Home) - SF (Away), bins=30 AVG={estimateSampled.mean()}",
#     density=True,
# )

# plt.title("Arbitrary hist")
# plt.legend()
# plt.show()

if __name__ == '__main__':
    # Print the structure of the Graph to verify its correctness.
    plt.figure(figsize=(10, 3))
    top = {team: (i * 2, 0) for i, team in enumerate(sorted(nodes))}
    bottom = {factor: (i, 1) for i, factor in enumerate(factors)}
    pos1 = nx.nx_agraph.graphviz_layout(G, prog="dot")
    # Draw all the variables & factors with their edges.
    nx.draw(
    G,
    pos= pos1, #{**top, **bottom},
    edge_color="red",)
    
    # Draw text labels for the factors above their nodes in the graph.
    label_dict = {factor: "{" + ",\n".join(factor.scope()) + "}" for factor in G.factors}
    for node, (x, y) in bottom.items():
        plt.text(x, y * 1.2, label_dict[node], fontsize=10, ha="center", va="center")
    
    # Re-draw the variables but with labels this time and colored orange.
    nx.draw(
        G.subgraph(nodes),
        node_color="orange",
        pos=pos1,#{**top},
        with_labels=True,)
    
    plt.show()

