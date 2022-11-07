import os
import pandas as pd
import numpy as np
from src.compression import compression
import pickle

def dict_to_pandas(dataset):
    """
    It converts the dict dataset into pandas format.
    :return: dataset in a pandas format.
    """
    new_dataset = pd.DataFrame()
    ids = dataset.keys()
    for i in ids:
        curr_traj = pd.DataFrame.from_dict(dataset[i])
        new_dataset = pd.concat([new_dataset, curr_traj], axis=0)
    return new_dataset


def pandas_to_dict(dataset):
    """
    It converts the pandas dataset into dict format.
    :return: dataset in a pandas format.
    """
    new_dataset = {}
    ids = dataset['trips'].unique()

    for id in ids:
        # getting one trajectory
        trajectory = dataset[dataset['trips'] == id]
        trajectory.set_index(['trips'])

        # converting trajectory to dict
        new_dataset[id] = {}
        for col in trajectory.columns:
            new_dataset[id][col] = np.array(trajectory[col])

    return new_dataset


def get_raw_dataset(dataset_path):
    """
    It compress the trajectories and provide compress rate and processing time.
    :param dataset_path: path of dataset with trajectories
    :param compress: compress algorithm: 'DP', 'TR', 'SP', 'DP_TR','DP_SP','TR_DP','TR_SP','SP_TR','SP_DP' (Default: 'DP')
    :param alpha: compression threshold
    """
    dataset = pd.read_csv(dataset_path, parse_dates=['time'], low_memory=False)
    dataset['time'] = dataset['time'].astype('datetime64[ns]')
    if not 'trips' in dataset.columns:
        dataset = dataset.rename(columns={'trajectory': 'trips'})
    dataset = dataset.sort_values(by=['trips', "time"])
    dataset_dict = pandas_to_dict(dataset)

    return dataset_dict


def compress_trips(dataset_path, compress='DP', alpha=1, **args):
    """
    It compress the trajectories and provide compress rate and processing time.
    :param dataset_path: path of dataset with trajectories
    :param compress: compress algorithm: 'DP', 'TR', 'SP', 'DP_TR','DP_SP','TR_DP','TR_SP','SP_TR','SP_DP' (Default: 'DP')
    :param alpha: compression threshold
    """
    dataset_dict = get_raw_dataset(dataset_path)

    file_name = os.path.basename(dataset_path)
    file_name = os.path.splitext(file_name)[0]

    compress_path = f"./data/compressed/{file_name}_{compress}_{alpha}.csv"
    compress_rate_path = f"./data/compressed/{file_name}_{compress}_{alpha}_compress_rate.p"
    time_rate_path = f"./data/compressed/{file_name}_{compress}_{alpha}_compress_time.p"

    if not os.path.exists(compress_path):
        if not os.path.exists(f"./data/preprocessed/compressed/"):
            os.makedirs(f"./data/preprocessed/compressed/")
        compress_dataset, compression_rate, processing_time = compression(dataset=dataset_dict, metric=compress, alpha=alpha)
        dataset = dict_to_pandas(compress_dataset)
        dataset = dataset.drop_duplicates()
        dataset.to_csv(compress_path, index=False)
        pickle.dump(compression_rate, open(compress_rate_path, 'wb'))
        pickle.dump(processing_time, open(time_rate_path, 'wb'))
    else:
        # print('\tCompression already computed')
        dataset = pd.read_csv(compress_path, parse_dates=['time'])
        compression_rate = pickle.load(open(compress_rate_path, 'rb'))
        processing_time = pickle.load(open(time_rate_path, 'rb'))
        dataset['time'] = dataset['time'].astype('datetime64[ns]')
        dataset = dataset.sort_values(by=['trips', "time"])
        compress_dataset = {}
        ids = dataset['trips'].unique()
        for id in ids:
            # getting one trajectory
            trajectory = dataset[dataset['trips'] == id]
            trajectory.set_index(['trips'])
            # converting trajectory to dict
            compress_dataset[id] = {}
            for col in trajectory.columns:
                compress_dataset[id][col] = np.array(trajectory[col])

    return compress_dataset, compression_rate, processing_time
