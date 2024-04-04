import numpy as np
import random
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph
import networkx as nx
import matplotlib.pyplot as plt
random.seed(0)

G =  FactorGraph()
nodes = ['a', 'b', 'y', 'x']
G.add_nodes_from(nodes)

ueFactor = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
net =  DiscreteFactor(['y'], [3], np.random.rand(3))
pred = DiscreteFactor(['x','y'], [3,3], np.random.rand(9))

factors = [ueFactor, net, pred] 

G.add_nodes_from([ueFactor, net, pred])
G.add_factors(ueFactor,net,pred)
G.add_edges_from([('a', ueFactor), ('b', ueFactor), 
                  ('x', pred), ('y', pred), 
                  ('x',ueFactor), 
                  ('y', net)])

G.check_model()

# Print the structure of the Graph to verify its correctness.
plt.figure(figsize=(10, 3))
top = {team: (i * 2, 0) for i, team in enumerate(sorted(nodes))}
bottom = {factor: (i, 1) for i, factor in enumerate(factors)}
# Draw all the variables & factors with their edges.
nx.draw(
    G,
    pos= {**top, **bottom},
    edge_color="red",
)

# Draw text labels for the factors above their nodes in the graph.
label_dict = {factor: "{" + ",\n".join(factor.scope()) + "}" for factor in G.factors}
for node, (x, y) in bottom.items():
    plt.text(x, y * 1.2, label_dict[node], fontsize=10, ha="center", va="center")
# Re-draw the variables but with labels this time and colored orange.
nx.draw(
    G.subgraph(nodes),
    node_color="orange",
    pos={**top},
    with_labels=True,
)
plt.show()

