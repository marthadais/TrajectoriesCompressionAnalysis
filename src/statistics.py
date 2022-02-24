import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


def barplot(data, xlabel='Dates', ylabel='Frequency', file='./results_data/frequency_bar.png', **args):
    """
    It generates the bar plot showing the outliers.
    :param data: the dataset
    :param xlabel: the label of axis x
    :param ylabel: the label of axis y
    :param file: the path to svae the plot
    """
    data = data.sort_values(data.columns[0])

    if 'more' in args.keys():
        data2 = args['more']
        data2 = data2.sort_values(data2.columns[0])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.bar(data.iloc[:, 0], data.iloc[:, 1], width=0.3, color='royalblue', align='center')
    if 'more' in args.keys():
        ax.bar((data2.iloc[:, 0]+0.3), data2.iloc[:, 1], width=0.3, color='seagreen', align='center')

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=90)
    plt.xticks(data.iloc[:, 0])
    plt.tight_layout()
    plt.savefig(file, bbox_inches='tight')
    plt.close()


def sc_bar_plt(data, folder='./results_data/'):
    """
    It generates the silhouette score bar graph.
    :param data: the dataset
    :param folder: the path were is the clustering results to save the statistics
    """
    lower_limit = data.loc[0, 'threshold_std']
    avg_limit = data['silhouette'].mean()
    data = data.sort_values([data.columns[1], data.columns[0]])
    data = data.reset_index()[[data.columns[0], data.columns[1]]]

    color_order = ['red', 'orange', 'blue', 'green', 'yellow', 'pink', 'violet', 'maroon', 'wheat', 'yellowgreen',
                   'lime', 'indigo', 'azure', 'olive', 'cyan', 'beige', 'skyblue', 'lavender', 'gold', 'fuchsia',
                   'purple']

    pad = 0
    i = 1
    n_clusters = len(np.unique(data.iloc[:, 1]))
    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(111)
    for c in range(n_clusters):
        if np.unique(data.iloc[:, 1])[c] == -1:
            i = 0
        if n_clusters <= 20:
            curr_c = color_order[i]
        else:
            curr_c = cm.tab20(float(c) / n_clusters)
        sample = data[data.iloc[:, 1] == np.unique(data.iloc[:, 1])[c]]
        ax.barh(range(pad, len(sample.iloc[:, 0])+pad), sample.iloc[:, 0], label=f'{np.unique(data.iloc[:, 1])[c]}', color=curr_c)
        pad = pad + len(sample.iloc[:, 0]) + 2
        i = i+1

    ax.plot([lower_limit, lower_limit], [0, pad-1], "--", label=f'lower_limit', color='red')
    ax.plot([avg_limit, avg_limit], [0, pad-1], "--", label=f'average', color='black')
    ax.set_ylabel('Instances', fontsize=25)
    ax.set_xlabel('Silhouette Score', fontsize=25)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    # ax.set_title(f'Individual Silhouette Score for {n_clusters} groups')
    if n_clusters < 20:
        plt.legend(fontsize=25)
    # plt.tight_layout()
    plt.savefig(f'{folder}/silhoutte.png', bbox_inches='tight')
    plt.close()


def statistics(dataset, col='trajectory', folder='./results_data/'):
    """
    It computes the statistics based on one attribute.
    :param dataset: the dataset
    :param col: the attribute to compute statistics (DEfault: 'trajectory')
    :param folder: the path were is the clustering results to save the statistics
    :return: average, standard deviation and the statistics for each id in col
    """
    vt_trajectory = dataset[col].unique()
    avg = dataset[['lat', 'lon']].mean(axis=0)
    std = dataset[['lat', 'lon']].std(axis=0)
    id_statistcs = pd.DataFrame()
    for i in vt_trajectory:
        sample = dataset[dataset[col] == i]
        row = [i, sample['mmsi'].iloc[0], sample['Clusters'].iloc[0], sample.shape[0],
               len(np.unique(sample['vessel_type'])), np.unique(sample['vessel_type']), len(np.unique(sample['flag'].astype(str))), np.unique(sample['flag'].astype(str)),
               sample['silhouette'].iloc[0], sample['threshold_std'].iloc[0], sample['scores-3std'].iloc[0],
               sample['lat'].mean(), sample['lat'].std(), sample['lon'].mean(), sample['lon'].std()]
        row = pd.DataFrame([row], columns=[col, 'mmsi', 'cluster', 'n_observations', 'n_vessel_type','vessel_type',
                                            'n_flags', 'flags',
                                           'silhouette', 'threshold', 'scores',
                                           'lat_avg', 'lat_std', 'lon_avg', 'lon_std'])
        id_statistcs = pd.concat([id_statistcs, row], ignore_index=True)

    id_statistcs.to_csv(f'{folder}/trajectory_statistcs_measure.csv')
    return avg, std, id_statistcs


def statistics_clusters(dataset, col='Clusters', folder='./results_data/'):
    """
    It computes the statistics based on one attribute.
    :param dataset: the dataset
    :param col: the attribute to compute statistics (DEfault: 'Clusters')
    :param folder: the path were is the clustering results to save the statistics
    :return: average, standard deviation and the statistics for each id in col
    """
    N = dataset[col].unique()

    trajectory_df = dataset.loc[:, ['trajectory', 'scores-3std', 'Clusters', 'silhouette']]
    trajectory_df.drop_duplicates('trajectory', inplace=True)

    avg = dataset[['lat', 'lon']].mean(axis=0)
    std = dataset[['lat', 'lon']].std(axis=0)
    id_statistcs = pd.DataFrame()
    for i in N:
        sample = dataset[dataset[col] == i]
        sample_trajectory = trajectory_df[trajectory_df[col] == i]
        row = [i, sample['Cl_Silhouette'].unique()[0], sample_trajectory['silhouette'].mean(), sample['silhouette'].std(), sample['threshold_std'].iloc[0], (sample_trajectory['scores-3std'] == 1).sum(), (sample_trajectory['scores-3std'] == -1).sum(),
               sample['lat'].mean(), sample['lat'].std(), sample['lon'].mean(), sample['lon'].std(), len(sample_trajectory), len(sample['mmsi'].unique()),
               len(np.unique(sample['vessel_type'])), np.unique(sample['vessel_type']),
               len(sample['flag'].unique()), list(np.unique(sample['flag'].astype(str)))]
        row = pd.DataFrame([row], columns=[col, 'cl_silhouette', 'sc_mean', 'sc_std', 'threshold_std', 'non-outliers', 'outliers',
                                           'lat_avg', 'lat_std', 'lon_avg', 'lon_std', 'n_trajectory', 'n_mmsi', 'n_vessel_types',
                                           'vessel_types', 'n_flags', 'flags'])
        id_statistcs = pd.concat([id_statistcs, row], ignore_index=True)

    id_statistcs.to_csv(f'{folder}/statistcs_clusters_measure.csv')
    barplot(id_statistcs[[col, 'non-outliers']], more=id_statistcs[[col, 'outliers']], xlabel='Clusters',
            ylabel='Number of trajectorys', file=f'{folder}/cluster_count_non-outliers.png')

    return avg, std, id_statistcs


def file_statistics(file, directory):
    """
    It receives the path with the clustering results and the folder path to compute the statistics and plot images.
    :param file: the path were is the dataset
    :param directory: the path were is the clustering results to save the statistics
    """
    dataset = pd.read_csv(file)
    dataset['time'] = dataset['time'].astype('datetime64[ns]')

    trajectory_df = dataset.loc[:, ['trajectory', 'silhouette', 'Clusters', 'threshold_std']]
    trajectory_df.drop_duplicates(inplace=True)
    sc_bar_plt(trajectory_df[['silhouette', 'Clusters', 'threshold_std']], folder=directory)

    print('\t summarizing trajectories info')
    traj_info = statistics(dataset, folder=directory)
    traj_info = traj_info[2]['cluster']

    new_directory = '/'.join(directory.split('/')[0:-1])

    print('\t summarizing clusters info')
    statistics_clusters(dataset, folder=directory)
