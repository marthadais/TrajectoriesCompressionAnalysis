from preprocessing.clean_trajectories import Trajectories
from datetime import datetime
import src.analysis as analysis
import os

print('Starting')
### Reading and cleaning dataset
# Number of vessels
n_samples = None
# Fishing type
vessel_type = [30, 1001, 1002]
# Time period
start_day = datetime(2020, 3, 1)
end_day = datetime(2020, 5, 31)
# Attributes
dim_set = ['lat', 'lon']
# polygon region
# region_limits = [46, 51, -128, -119]
# long beach
region_limits = [33, 34, -119, -117.5]

# Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day),
                       region=region_limits)

folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/dtw/'
# folder = f'./results/{dataset_name}/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{metric}_fast/'
if region_limits is not None:
    folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{n_samples}_{region_limits}/dtw_fast/'

if not os.path.exists(folder):
    os.makedirs(folder)
# dataset_dict = dataset.get_dataset()

rates1, times1 = analysis.factor_analysis(dataset, 'DP', folder)
measure = analysis.factor_dist_analysis(dataset, 'DP', folder)
measure_purity = analysis.factor_cluster_analysis(dataset, 'DP', folder)

rates2, times2 = analysis.factor_analysis(dataset, 'TR', folder)
measure_rt = analysis.factor_dist_analysis(dataset, 'TR', folder)
measure_purity_rt = analysis.factor_cluster_analysis(dataset, 'TR', folder)

rates3, times3 = analysis.factor_analysis(dataset, 'SP', folder)
measure_sp = analysis.factor_dist_analysis(dataset, 'SP', folder)
measure_purity_SP = analysis.factor_cluster_analysis(dataset, 'SP', folder)

rates4, times4 = analysis.factor_analysis(dataset, 'TR_SP', folder)
measure_tr_sp = analysis.factor_dist_analysis(dataset, 'TR_SP', folder)
measure_purity_tr_sp = analysis.factor_cluster_analysis(dataset, 'TR_SP', folder)

rates5, times5 = analysis.factor_analysis(dataset, 'SP_TR', folder)
measure_sp_tr = analysis.factor_dist_analysis(dataset, 'SP_TR', folder)
measure_purity_sp_tr = analysis.factor_cluster_analysis(dataset, 'SP_TR', folder)

analysis.lines_compression(folder)

