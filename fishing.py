from preprocessing.clean_trajectories import Trajectories
from datetime import datetime
import src.analysis as analysis
import os

print('Starting')
### Reading and cleaning dataset
# Number of vessels
n_samples = None
# https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
#FISHING
vessel_type = [30, 1001, 1002]

#My Combination
# vessel_type = [30, 1001, 1002, 21, 22, 31, 32, 52, 1023, 1025, 36, 37, 1019]
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 6, 30)
# Attributes
dim_set = ['lat', 'lon']
# Francisco Bay
region_limits = [37.6, 39, -122.9, -122.2]

# Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day),
                       region=region_limits)
metric = 'dtw'

folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{metric}/'
if region_limits is not None:
    folder = f'./results/DCAIS/vt_{vessel_type}/date_{start_day.date()}_to_{end_day.date()}/{region_limits}/{metric}/'

if not os.path.exists(folder):
    os.makedirs(folder)

print(folder)

# Compression
rates1, times1 = analysis.factor_analysis(dataset, 'DP', folder)
rates2, times2 = analysis.factor_analysis(dataset, 'TR', folder)
rates3, times3 = analysis.factor_analysis(dataset, 'SP', folder)
rates4, times4 = analysis.factor_analysis(dataset, 'TR_SP', folder)
rates5, times5 = analysis.factor_analysis(dataset, 'SP_TR', folder)
rates6, times6 = analysis.factor_analysis(dataset, 'SP_DP', folder)
rates7, times7 = analysis.factor_analysis(dataset, 'DP_SP', folder)
rates8, times8 = analysis.factor_analysis(dataset, 'TR_DP', folder)
rates9, times9 = analysis.factor_analysis(dataset, 'DP_TR', folder)

# Distance matrices
measure = analysis.factor_dist_analysis(dataset, 'DP', folder, metric=metric)
measure_rt = analysis.factor_dist_analysis(dataset, 'TR', folder, metric=metric)
measure_sp = analysis.factor_dist_analysis(dataset, 'SP', folder, metric=metric)
measure_tr_sp = analysis.factor_dist_analysis(dataset, 'TR_SP', folder, metric=metric)
measure_sp_tr = analysis.factor_dist_analysis(dataset, 'SP_TR', folder, metric=metric)
measure_dp_sp = analysis.factor_dist_analysis(dataset, 'DP_SP', folder, metric=metric)
measure_sp_dp = analysis.factor_dist_analysis(dataset, 'SP_DP', folder, metric=metric)
measure_tr_dp = analysis.factor_dist_analysis(dataset, 'TR_DP', folder, metric=metric)
measure_dp_tr = analysis.factor_dist_analysis(dataset, 'DP_TR', folder, metric=metric)


# Clustering
measure_purity = analysis.factor_cluster_analysis(dataset, 'DP', folder, metric=metric)
measure_purity_tr = analysis.factor_cluster_analysis(dataset, 'TR', folder, metric=metric)
measure_purity_sp = analysis.factor_cluster_analysis(dataset, 'SP', folder, metric=metric)
measure_purity_tr_sp = analysis.factor_cluster_analysis(dataset, 'TR_SP', folder, metric=metric)
measure_purity_sp_tr = analysis.factor_cluster_analysis(dataset, 'SP_TR', folder, metric=metric)
measure_purity_dp_sp = analysis.factor_cluster_analysis(dataset, 'DP_SP', folder, metric=metric)
measure_purity_sp_dp = analysis.factor_cluster_analysis(dataset, 'SP_DP', folder, metric=metric)
measure_purity_tr_dp = analysis.factor_cluster_analysis(dataset, 'TR_DP', folder, metric=metric)
measure_purity_dp_tr = analysis.factor_cluster_analysis(dataset, 'DP_TR', folder, metric=metric)

analysis.lines_compression(folder, metric=metric)

