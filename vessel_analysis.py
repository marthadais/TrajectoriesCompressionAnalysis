import src.analysis as analysis
import os

# Attributes
dim_set = ['lat', 'lon']

# data_path = './data/non-crop/DCAIS_[30, 1001, 1002]_region_[37.6, 39, -122.9, -122.2]_01-04_to_30-06_trips.csv'
# data_path = './data/non-crop/DCAIS_[80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1017, 1024]_region_[47.5, 49.3, -125.5, -122.5]_01-04_to_30-06_trips.csv'

# fishing vessels
data_path = './data/crop/DCAIS_[30, 1001, 1002]_region_[37.6, 39, -122.9, -122.2]_01-04_to_30-06_trips.csv'

# tanker vessels
# data_path = './data/crop/DCAIS_[80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1017, 1024]_region_[47.5, 49.3, -125.5, -122.5]_01-04_to_30-06_trips.csv'

file_name = os.path.basename(data_path)
file_name = os.path.splitext(file_name)[0]

# metric = 'frechat'
metric = 'md'
# metric = 'dtw'

folder = f'./results/crop/{file_name}/{metric}/'
if not os.path.exists(folder):
    os.makedirs(folder)

print(folder)

# Compression
rates1, times1 = analysis.factor_analysis(data_path, 'DP', folder)
# rates2, times2 = analysis.factor_analysis(data_path, 'TR', folder)
# rates3, times3 = analysis.factor_analysis(data_path, 'SP', folder)
# rates4, times4 = analysis.factor_analysis(data_path, 'TR_SP', folder)
# rates5, times5 = analysis.factor_analysis(data_path, 'SP_TR', folder)
# rates6, times6 = analysis.factor_analysis(data_path, 'SP_DP', folder)
# rates7, times7 = analysis.factor_analysis(data_path, 'DP_SP', folder)
# rates8, times8 = analysis.factor_analysis(data_path, 'TR_DP', folder)
# rates9, times9 = analysis.factor_analysis(data_path, 'DP_TR', folder)

# Distance matrices
# measure = analysis.factor_dist_analysis(data_path, 'DP', folder, metric=metric)
# measure_rt = analysis.factor_dist_analysis(data_path, 'TR', folder, metric=metric)
# measure_sp = analysis.factor_dist_analysis(data_path, 'SP', folder, metric=metric)
# measure_tr_sp = analysis.factor_dist_analysis(data_path, 'TR_SP', folder, metric=metric)
# measure_sp_tr = analysis.factor_dist_analysis(data_path, 'SP_TR', folder, metric=metric)
# measure_dp_sp = analysis.factor_dist_analysis(data_path, 'DP_SP', folder, metric=metric)
# measure_sp_dp = analysis.factor_dist_analysis(data_path, 'SP_DP', folder, metric=metric)
# measure_tr_dp = analysis.factor_dist_analysis(data_path, 'TR_DP', folder, metric=metric)
# measure_dp_tr = analysis.factor_dist_analysis(data_path, 'DP_TR', folder, metric=metric)


# Clustering
measure_purity = analysis.factor_cluster_analysis(data_path, 'DP', folder, metric=metric)
measure_purity_tr = analysis.factor_cluster_analysis(data_path, 'TR', folder, metric=metric)
measure_purity_sp = analysis.factor_cluster_analysis(data_path, 'SP', folder, metric=metric)
measure_purity_tr_sp = analysis.factor_cluster_analysis(data_path, 'TR_SP', folder, metric=metric)
measure_purity_sp_tr = analysis.factor_cluster_analysis(data_path, 'SP_TR', folder, metric=metric)
measure_purity_dp_sp = analysis.factor_cluster_analysis(data_path, 'DP_SP', folder, metric=metric)
measure_purity_sp_dp = analysis.factor_cluster_analysis(data_path, 'SP_DP', folder, metric=metric)
measure_purity_tr_dp = analysis.factor_cluster_analysis(data_path, 'TR_DP', folder, metric=metric)
measure_purity_dp_tr = analysis.factor_cluster_analysis(data_path, 'DP_TR', folder, metric=metric)

analysis.lines_compression(folder, metric=metric)

