import numpy as np
#https://github.com/uestc-db/traj-compression/blob/master/online/Dead_Reckoning/Dead_Reckoning.cpp


def cacl_distance(points):
    distance = np.array([])
    for i in range(len(points['lat'])):
        np.apppend(distance, np.sqrt(pow(points['lat'][i] - points['lat'][i - 1], 2) + pow(points['lon'][i] - points['lon'][i - 1], 2)))
    return distance


def cacl_angle(points):
    angles = np.array([])
    for i in range(len(points['lat'])):
        lat_diff = points['lat'][i] - points['lat'][i - 1]
        lon_diff = points['lon'][i] - points['lon'][i - 1]
        np.append(angles, np.arctan2(lon_diff, lat_diff))
    return angles


def dead_reckoning(trajectory, dim_set, eps):
    n = len(trajectory['lat'])
    max_d = 0
    start_idx = 0
    d = cacl_distance(trajectory)
    angles = cacl_angle(trajectory)
    simplifindex = np.array([0])
    for i in range(2, n):
        max_d += abs(d[i - 1] * np.sin(angles[i - 1] - angles[start_idx]))
        if abs(max_d) > eps:
            max_d = 0
            np.append(simplifindex, i - 1)
            start_idx = i - 1

        if simplifindex[simplifindex.size() - 1] != n - 1:
            np.append(simplifindex, n - 1)

    new_trajectory = {}
    for dim in dim_set:
        new_trajectory[dim] = trajectory[dim][simplifindex]

    return new_trajectory
