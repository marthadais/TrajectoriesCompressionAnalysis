# TrajectoriesCompressionAnalysis

The proposed methodology assess how compression algorithms influence the clustering analysis with respect to anomaly detection of vessel trajectories. Results shows that a suitable compression algorithm for a particular scenario can reduce the overall processing time with a low impact on the clustering outcome.

## Files Description
1. vessel_analysis.py
   - Code to execute the analysis on vessels
   - Select the vessel type:
     1. this line refers to fishing vessel
     ```
     # fishing vessels
     data_path = './data/crop/DCAIS_[30, 1001, 1002]_region_[37.6, 39, -122.9, -122.2]_01-04_to_30-06_trips.csv'
     ```
     2. this line refers to tankers vessel
     ```
     # tanker vessels
     data_path = './data/crop/DCAIS_[80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1017, 1024]_region_[47.5, 49.3, -125.5, -122.5]_01-04_to_30-06_trips.csv'
     ```
   - Select the distance measure
     1. Define the metric as desired
     - 'dtw': dynamic time warping
     - 'hd': hausdorff distance
     - 'dfd': discrete fréchet distance
     - 'md': merge distance
     ```
     metric = 'dtw'
     ```
   - Select the minimum size of a cluster measure
     1. Define minimum size of a cluster measure
     - Fishing vessels: 2
     - Tanker vessels: 3
     ```
     msc = 2
     ```
 
2. source folder (src)
   - analysis.py: contains functions to analyze the different factors and plot images
     1. compression analysis
     2. distances analysis
     3. clustering analysis
   - clustering.py: contains clustering class to compute the clustering 
   - compression.py: contains function to compute the compression
   - distance.py: contains functions to compute distance between trajectories

2. preprocessing folder
   - compress_trajectories.py: contains functions to read dataset and compute compression of all trajectories in the dataset
   
## Requirements
The python version used was Python 3.9.5.
The requirements to execute the code is in the file requirements.txt.
