from preprocessing.clean_trajectories import Trajectories
from src.extract_distances import DistanceMatrix
from datetime import datetime
import os

print('Starting')
### Reading and cleaning dataset
# Number of vessels
n_samples = 30
# Fishing type
vessel_type = [30, 1001, 1002]
# Time period
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 30)
# Attributes
dim_set = ['lat', 'lon']
# polygon region
region_limits = [30, 38, -92, -70]

# Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day), region=region_limits)

#### Computing Distances
metric = 'dtw'
# metric = 'md'

folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{metric}/'
# folder = f'./results/{dataset_name}/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{metric}_fast/'
if region_limits is not None:
    folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{n_samples}_{region_limits}/{metric}_fast/'

dim_set = ['lat', 'lon']

features_path = f'{folder}/features_distance.p'
if not os.path.exists(features_path):
    dataset_dict = dataset.pandas_to_dict()
    features = DistanceMatrix(dataset=dataset_dict, features_opt=metric, dim_set=dim_set, folder=folder)