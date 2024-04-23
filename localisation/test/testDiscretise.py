import sys

sys.path.insert(1, '/Users/mariam/Downloads/IIB Project readings/IIB project code/localisation/src')

from discretise import *
import random

random.seed(0)

varOne = 3 + 2.5 * np.random.randn(1000)
varTwo = 100 - 5 * np.random.randn(1000)
hist, _, _ = discretise(varOne, varTwo, 10)

def test_discretise():
    assert round(np.sum(hist), 4) == 1 and np.shape(hist) == (10,10)

def test_conditionalProbabilityTable():
    cpd = conditionalProbabilityTable(hist)
    assert np.shape(cpd) == (10,10) and np.all(np.sum(cpd, axis = 1)) == 1

if __name__ == "__main__":
    test_discretise()
    test_conditionalProbabilityTable()
    print("Everything passed.")