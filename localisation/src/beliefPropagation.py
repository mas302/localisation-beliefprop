import numpy as np
import random
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph
import networkx as nx
import matplotlib.pyplot as plt
from discretise import *
from pgmpy.inference import BeliefPropagation

random.seed(0)

G =  FactorGraph()
nodes = ['a', 'y', 'x']
G.add_nodes_from(nodes)

ueFactor = DiscreteFactor(['a', 'x'], [2, 10], np.random.rand(20))
net =  DiscreteFactor(['y'], [3], np.random.rand(3))
pred = DiscreteFactor(['x','y'], [10, 3], np.random.rand(30))

userEquipment = 3 + 2.5 * np.random.randn(1000)
sensorValues= 100 - 5 * np.random.randn(1000)

predUE = discretise(sensorValues, userEquipment, 10)
predUETabularCPD = conditionalProbabilityTable(predUE)
predUETabularCPD.shape
ueDiscretisedFactor =  DiscreteFactor(['a', 'x'], [10,10], predUE)

factors = [ueDiscretisedFactor, net, pred] 

G.add_nodes_from([ueDiscretisedFactor, net, pred])
G.add_factors(ueDiscretisedFactor, net, pred)
G.add_edges_from([('a', ueDiscretisedFactor), ('x', ueDiscretisedFactor), 
                  ('x', pred), ('y', pred), 
                  ('x', ueDiscretisedFactor), 
                  ('y', net)])

print(G.check_model())

for i in factors:
    print(i.values)

bp = BeliefPropagation(G)
bp.calibrate()
print(bp.get_clique_beliefs())
bp.map_query(['y', 'x'], show_progress=True, evidence = {'a': 0})

if __name__ == '__main__':
    # Print the structure of the Graph to verify its correctness.
    plt.figure(figsize=(10, 3))
    top = {team: (i * 2, 0) for i, team in enumerate(sorted(nodes))}
    bottom = {factor: (i, 1) for i, factor in enumerate(factors)}
    
    # Draw all the variables & factors with their edges.
    nx.draw(
    G,
    pos= {**top, **bottom},
    edge_color="red",)
    
    # Draw text labels for the factors above their nodes in the graph.
    label_dict = {factor: "{" + ",\n".join(factor.scope()) + "}" for factor in G.factors}
    for node, (x, y) in bottom.items():
        plt.text(x, y * 1.2, label_dict[node], fontsize=10, ha="center", va="center")
    
    # Re-draw the variables but with labels this time and colored orange.
    nx.draw(
        G.subgraph(nodes),
        node_color="orange",
        pos={**top},
        with_labels=True,)
    
    plt.show()

