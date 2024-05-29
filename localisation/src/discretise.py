import numpy as np
from scipy.stats import multivariate_normal, norm

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
        cpd[i,:] = hist[i,:]/ (marginalVarOne[i]+ 0.000001)

    return cpd

def discretise3vars(mean, covariance, delta_lat_range, delta_lon_range, altitude_ranges, interval_width):

    sigma_11 = covariance[0][0]
    sigma_12 = np.array(covariance[0][1:3])  # Covariance between x1 and x2, x3 (1x2 array)
    sigma_21 = np.array([row[0] for row in covariance[1:3]]).reshape(2, 1)  # Covariance between x2, x3 and x1 (2x1 array)
    sigma_22 = np.array([row[1:3] for row in covariance[1:3]])  # 2x2 covariance matrix for x2 and x3
    sigma_22_inv = np.linalg.inv(sigma_22)  

    sigma_cond = sigma_11 - sigma_12.dot(sigma_22_inv).dot(sigma_21)

    if sigma_cond <= 0:
        # print("Warning: Conditional variance is non-positive, which is invalid for normal distribution.")
        return np.zeros((len(altitude_ranges), len(delta_lat_range), len(delta_lon_range)))

    prob_table = np.zeros((len(delta_lat_range), len(delta_lon_range), len(altitude_ranges)))

    for i, delta_lat in enumerate(delta_lat_range):
        for j, delta_lon in enumerate(delta_lon_range):

            cond_mean = mean[0] + sigma_12.dot(sigma_22_inv).dot([delta_lat - mean[1], delta_lon - mean[2]])
        
            for k, alt in enumerate(altitude_ranges):
                lower_bound = alt - interval_width/2  
                upper_bound = alt + interval_width/2   
                prob = norm.cdf(upper_bound, cond_mean, np.sqrt(sigma_cond)) - norm.cdf(lower_bound, cond_mean, np.sqrt(sigma_cond))
                prob_table[i, j, k] = prob

    # Normalize the probability table to ensure each "row" sums to 1
    for j in range(len(delta_lat_range)):
        for k in range(len(delta_lon_range)):
            prob_sum = np.sum(prob_table[:, j, k])
            if prob_sum > 0:
                prob_table[:, j, k] /= prob_sum
                # Debug: Print the sum of the probabilities to ensure normalization
                # print(f"Normalized sum for slice [:, {j}, {k}]: {np.sum(prob_table[:, j, k])}")
            # else:
            #     print(f"Sum of probabilities for slice [:, {j}, {k}] is: {prob_sum}.")
    
    return prob_table