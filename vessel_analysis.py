import src.analysis as analysis
import os

# fishing vessels
data_path = './data/crop/DCAIS_[30, 1001, 1002]_region_[37.6, 39, -122.9, -122.2]_01-04_to_30-06_trips.csv'
# tanker vessels
# data_path = './data/crop/DCAIS_[80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1017, 1024]_region_[47.5, 49.3, -125.5, -122.5]_01-04_to_30-06_trips.csv'

# Measures: 'dtw', 'md', 'frechet'
metric = 'frechet'

# fishing = 2
# tanker = 3
mcs = 2

# path and folder for results
file_name = os.path.basename(data_path)
file_name = os.path.splitext(file_name)[0]
folder = f'./results/crop/{file_name}/{metric}/'
if not os.path.exists(folder):
    os.makedirs(folder)

print(folder)

# Compression
# rates_dp, times_dp = analysis.factor_analysis(data_path, 'DP', folder)
# rates_tr, times_tr = analysis.factor_analysis(data_path, 'TR', folder)
# rates_sp, times_sp = analysis.factor_analysis(data_path, 'SP', folder)
# rates_tr_sp, times_tr_sp = analysis.factor_analysis(data_path, 'TR_SP', folder)
# rates5_sp_tr, times_sp_tr = analysis.factor_analysis(data_path, 'SP_TR', folder)
# rates6_sp_dp, times_sp_dp = analysis.factor_analysis(data_path, 'SP_DP', folder)
# rates_dp_sp, times_dp_sp = analysis.factor_analysis(data_path, 'DP_SP', folder)
# rates_tr_dp, times_tr_dp = analysis.factor_analysis(data_path, 'TR_DP', folder)
# rates_dp_tr, times_dp_tr = analysis.factor_analysis(data_path, 'DP_TR', folder)

# Distance matrices
# measure_dp = analysis.factor_dist_analysis(data_path, 'DP', folder, metric=metric)
# measure_rt = analysis.factor_dist_analysis(data_path, 'TR', folder, metric=metric)
# measure_sp = analysis.factor_dist_analysis(data_path, 'SP', folder, metric=metric)
# measure_tr_sp = analysis.factor_dist_analysis(data_path, 'TR_SP', folder, metric=metric)
# measure_sp_tr = analysis.factor_dist_analysis(data_path, 'SP_TR', folder, metric=metric)
# measure_dp_sp = analysis.factor_dist_analysis(data_path, 'DP_SP', folder, metric=metric)
# measure_sp_dp = analysis.factor_dist_analysis(data_path, 'SP_DP', folder, metric=metric)
# measure_tr_dp = analysis.factor_dist_analysis(data_path, 'TR_DP', folder, metric=metric)
# measure_dp_tr = analysis.factor_dist_analysis(data_path, 'DP_TR', folder, metric=metric)


# Clustering
measure_nmi = analysis.factor_cluster_analysis(data_path, 'DP', folder, metric=metric, mcs=mcs)
measure_nmi_tr = analysis.factor_cluster_analysis(data_path, 'TR', folder, metric=metric, mcs=mcs)
measure_nmi_sp = analysis.factor_cluster_analysis(data_path, 'SP', folder, metric=metric, mcs=mcs)
measure_nmi_tr_sp = analysis.factor_cluster_analysis(data_path, 'TR_SP', folder, metric=metric, mcs=mcs)
measure_nmi_sp_tr = analysis.factor_cluster_analysis(data_path, 'SP_TR', folder, metric=metric, mcs=mcs)
measure_nmi_dp_sp = analysis.factor_cluster_analysis(data_path, 'DP_SP', folder, metric=metric, mcs=mcs)
measure_nmi_sp_dp = analysis.factor_cluster_analysis(data_path, 'SP_DP', folder, metric=metric, mcs=mcs)
measure_nmi_tr_dp = analysis.factor_cluster_analysis(data_path, 'TR_DP', folder, metric=metric, mcs=mcs)
measure_nmi_dp_tr = analysis.factor_cluster_analysis(data_path, 'DP_TR', folder, metric=metric, mcs=mcs)

analysis.lines_compression(folder, metric=metric)

