import numpy as np
# adapt from https://github.com/uestc-db/traj-compression/blob/master/batch/TD-TR/TD-TR.cpp

def calc_SED(pA, pI, pB):

    pA_lat, pA_lon, pA_time = pA
    pI_lat, pI_lon, pI_time = pI
    pB_lat, pB_lon, pB_time = pB


    Di = pI_time - pA_time
    Db = pB_time - pA_time
    if Di == 0:
        time_ratio = 0
    else:
        time_ratio = Db/Di

    lat = pA_lat + (pB_lat - pA_lat) * time_ratio
    lon = pA_lon + (pB_lon - pA_lon) * time_ratio

    lat_diff = lat - pI_lat
    lon_diff = lon - pI_lon
    return np.sqrt((lat_diff * lat_diff) + (lon_diff * lon_diff))

def eps_estimation(trajectory, traj_time):
    st_d = np.array([])
    traj_len = len(trajectory['lat'])
    # time in seconds
    start_location = (trajectory['lat'][0], trajectory['lon'][0], traj_time[0])
    final_location = (trajectory['lat'][-1], trajectory['lon'][-1], traj_time[-1])
    for i in range(traj_len):
        middle = (trajectory['lat'][i], trajectory['lon'][i], traj_time[i])
        d = calc_SED(start_location, middle, final_location)
        st_d = np.append(st_d,d)

    return st_d.mean(), st_d.std()


def TR_compression(trajectory, dim_set, traj_time, epsilon, max_epsilon):
    new_trajectory = {}
    for dim in dim_set:
        new_trajectory[dim] = np.array([])
    dmax = 0
    idx = 0
    traj_len = len(trajectory['lat'])

    if traj_len > 3:
        # time in seconds
        start_location = (trajectory['lat'][0], trajectory['lon'][0], traj_time[0])
        final_location = (trajectory['lat'][-1], trajectory['lon'][-1], traj_time[-1])
        for i in range(traj_len):
            middle = (trajectory['lat'][i], trajectory['lon'][i], traj_time[i])
            d = calc_SED(start_location, middle, final_location)
            if d > dmax and d < max_epsilon:
                idx = i
                dmax = d

        print(f'dmax: {dmax}, index: {idx}, trajlen: {traj_len}')
        if dmax > epsilon:
            traj1 = {}
            traj2 = {}
            for dim in dim_set:
                traj1[dim] = trajectory[dim][0:idx]
                traj2[dim] = trajectory[dim][idx:]
            recResults1 = TR_compression(traj1, dim_set, traj_time[0:idx], epsilon, max_epsilon)
            recResults2 = TR_compression(traj2, dim_set, traj_time[idx:], epsilon, max_epsilon)

            for dim in dim_set:
                new_trajectory[dim] = np.append(new_trajectory[dim], recResults1[dim])
                new_trajectory[dim] = np.append(new_trajectory[dim], recResults2[dim])

        else:
            trajectory['time'] = trajectory['time'].astype(str)
            for dim in dim_set:
                if traj_len == 1:
                    new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][0])
                else:
                    new_trajectory[dim] = np.append(new_trajectory[dim], [trajectory[dim][0], trajectory[dim][-1]])
    else:
        trajectory['time'] = trajectory['time'].astype(str)
        for dim in dim_set:
            if traj_len == 1:
                new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][0])
            else:
                new_trajectory[dim] = np.append(new_trajectory[dim], [trajectory[dim][0], trajectory[dim][-1]])

    return new_trajectory


def compression(dataset, dim_set=None, verbose=True):
    _ids = list(dataset.keys())
    new_dataset = {}
    for id_a in range(len(_ids)):
        new_dataset[_ids[id_a]] = {}

    if dim_set == None:
        dim_set = dataset[_ids[0]].keys()

    id_a = 0
    # for id_a in range(len(self._ids)):
    while id_a < len(_ids):
        if verbose:
            print(f"Compressing {id_a} of {len(_ids)}")
        # trajectory a
        curr_traj = dataset[_ids[id_a]]
        traj_time = curr_traj['time'].astype('datetime64[s]')
        traj_time = np.hstack((0, np.diff(traj_time).cumsum().astype('float')))
        traj_time = traj_time / traj_time.max()
        epsilon, max_epsilon = eps_estimation(curr_traj, traj_time)
        max_epsilon = epsilon + max_epsilon
        compress_traj = TR_compression(curr_traj, dim_set, traj_time, epsilon, max_epsilon)
        new_dataset[_ids[id_a]] = compress_traj
        id_a = id_a + 1

    return new_dataset