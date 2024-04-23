import math
import numpy as np

RADIUS = 6371000

class Sensor:

    def __init__(self, latitude, longitude, uncertainty, range):
        self.latitude = latitude
        self.longitude = longitude
        self.sigma = uncertainty
        self.range = range
        self.measuredPosition = [latitude, longitude, uncertainty]

    def distanceHaversine(lat1, lat2, lon1, lon2): # quantifies UE position from itself

        """A function that measures the distance and bearing between a pair of latitude and longitude
        measurements.
        
        Input:
        lat1, lon1: initial latitude and longitude measured by UE GPS measured in degrees
        lat2, lon2: final latitude and longitude measured by UE GPS measures in degrees
        
        Output:
        distance: Haversine distance between the latitudes and longitudes
        initBearing: the initial bearing in degrees. """

        # latitudes in radians
        phi1 = lat1 * math.pi/180 
        phi2 = lat2 * math.pi/180
        phi_diff = phi2 - phi1

        # longitudes in radians
        lon_diff = (lon2-lon1) * math.pi/180

        a = (math.sin(phi_diff/2) **2 + math.cos(phi1) 
              * math.cos(phi2) * math.sin(lon_diff/2) **2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        distance = RADIUS*c
        initBearingRad = math.atan2(math.sin(lon_diff)*math.cos(phi2), 
                             math.cos(phi1)*math.sin(phi2)
                             - math.sin(phi1)*math.cos(phi2)*math.cos(lon_diff))
        initBearing = (initBearing * 180/math.pi + 360) % 360

        return distance, initBearing
    
    def haversine2latlon(haversineDistance, initBearing, lat1, lon1):
        """Function to obtain the latitude and longitude of a UE, given that the Haversine distance
        and the initial bearing have been calculated.
        
        Inputs:
        haversineDistance: the predicted Haversine distance for between the initial and final
        position of the UE.
        initBearing: the predicted initial bearing between the initial and final position of the 
        UE.
        lat1, lon1: the initial latitude and longitude of the UE's initial position.

        Outputs:
        lat2, lon2: the predicted final UE location.
        """

        delta = haversineDistance/RADIUS

        lat2 = math.asin(math.sin(lat1)*math.cos(delta)
                         + math.cos(lat1)*math.sin(delta)*math.cos(initBearing))
        
        lon2 = lon1 + math.atan2(math.sin(initBearing)*math.cos(lat1),
                                 math.cos(delta)- math.sin(lat1)*math.sin(lat2))
        
        return lat2, lon2
    
    def nameLocation(latitude, longitude):
        """Function to appropriately express the latitude and longitude from the sensor output.
        
        Inputs:
        latiude, longitude: Latitude and Longitude of UE location we wish to express in a standard
        form.
        
        Output:
        renamedlatlon: the input latitude and longitude renamed."""
        return None
