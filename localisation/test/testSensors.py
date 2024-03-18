import sys

sys.path.insert(1, '/Users/mariam/Downloads/IIB Project readings/IIB project code/localisation/src')

from sensors import Sensor

def test_distanceHaversine():
    assert Sensor.distanceHaversine(1.1,1.2,1.3,1.4) == 15723.752702701198

if __name__ == "__main__":
    test_distanceHaversine()
    print("Everything passed.")