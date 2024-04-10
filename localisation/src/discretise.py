import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

def integrateOver(mean, cov, x_min, x_max, y_min, y_max):
    mvn = multivariate_normal(mean, cov)
    return dblquad(lambda y, x: mvn.pdf([x, y]), x_min, x_max, lambda x: y_min, lambda x: y_max)

def discretise(limits, cardinalities, mean, cov):

    varOne = np.linspace(limits[0][0], limits[0][1], cardinalities[0])
    varTwo = np.linspace(limits[1][0], limits[1][1], cardinalities[1])

    probabilities = np.zeros((cardinalities[0]-1, cardinalities[1]-1))

    for i in range(cardinalities[0]-1):
        for j in range(cardinalities[1]-1):
            prob, _ = integrateOver(mean, cov, varOne[i], varOne[i+1], varTwo[j], varTwo[j+1])
            probabilities[i, j] = prob
    
    return probabilities

if __name__ == '__main__':
    mean = np.array([0, 0])
    cov = np.array([[1, 0.5], [0.5, 1]])
    probs = discretise([(-5,5),(-5,5)],[10,10], mean, cov)
    print(probs.shape)