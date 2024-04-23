import numpy as np

def discretise(varOne, varTwo, cardinality):
    weights = np.ones(len(varOne)) / len(varOne)
    h, _, _ = np.histogram2d(varOne, varTwo, cardinality, density=False, weights=weights)
    return h

def conditionalProbabilityTable(hist):
    """ returns p(varTwo|varOne) """
    m,n = hist.shape
    marginalVarOne = np.sum(hist, axis=1)
    cpd = np.zeros((m,n)) 

    for i in range(m):
        cpd[i,:] = hist[i,:]/ marginalVarOne[i]

    return cpd