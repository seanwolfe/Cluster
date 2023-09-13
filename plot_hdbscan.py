# -*- coding: utf-8 -*-
"""
====================================
Demo of HDBSCAN clustering algorithm
====================================
.. currentmodule:: sklearn

In this demo we will take a look at :class:`cluster.HDBSCAN` from the
perspective of generalizing the :class:`cluster.DBSCAN` algorithm.
We'll compare both algorithms on specific datasets. Finally we'll evaluate
HDBSCAN's sensitivity to certain hyperparameters.

We first define a couple utility functions for convenience.
"""
# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.datasets import make_blobs
from MmCluster import preprocessing_main
from sklearn import preprocessing
import pandas as pd
import os


def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor=tuple(col),
                markersize=1 if k == -1 else 0.5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.tight_layout()


# While standardizing data (e.g. using
# :class:`sklearn.preprocessing.StandardScaler`) helps mitigate this problem,
# great care must be taken to select the appropriate value for `eps`.
#
# HDBSCAN is much more robust in this sense: HDBSCAN can be seen as
# clustering over all possible values of `eps` and extracting the best
# clusters from all possible clusters (see :ref:`User Guide <HDBSCAN>`).
# One immediate advantage is that HDBSCAN is scale-invariant.
# %%
# Multi-Scale Clustering
# ----------------------
# HDBSCAN is much more than scale invariant though -- it is capable of
# multi-scale clustering, which accounts for clusters with varying density.
# Traditional DBSCAN assumes that any potential clusters are homogenous in
# density. HDBSCAN is free from such constraints. To demonstrate this we
# consider the following dataset

# %%
# This dataset is more difficult for DBSCAN due to the varying densities and
# spatial separation:
#
# - If `eps` is too large then we risk falsely clustering the two dense
#   clusters as one since their mutual reachability will extend
#   clusters.
# - If `eps` is too small, then we risk fragmenting the sparser clusters
#   into many false clusters.
#
# Not to mention this requires manually tuning choices of `eps` until we
# find a tradeoff that we are comfortable with.
# To properly cluster the two dense clusters, we would need a smaller value of
# epsilon, however at `eps=0.3` we are already fragmenting the sparse clusters,
# which would only become more severe as we decrease epsilon. Indeed it seems
# that DBSCAN is incapable of simultaneously separating the two dense clusters
# while preventing the sparse clusters from fragmenting. Let's compare with
# HDBSCAN.
# %%
# HDBSCAN is able to adapt to the multi-scale structure of the dataset without
# requiring parameter tuning. While any sufficiently interesting dataset will
# require tuning, this case demonstrates that HDBSCAN can yield qualitatively
# better classes of clusterings without users' intervention which are
# inaccessible via DBSCAN.

# %%
# Hyperparameter Robustness
# -------------------------
# Ultimately tuning will be an important step in any real world application, so
# let's take a look at some of the most important hyperparameters for HDBSCAN.
# While HDBSCAN is free from the `eps` parameter of DBSCAN, it does still have
# some hyperparameters like `min_cluster_size` and `min_samples` which tune its
# results regarding density. We will however see that HDBSCAN is relatively robust
# to various real world examples thanks to those parameters whose clear meaning
# helps tuning them.
#
# `min_cluster_size`
# ^^^^^^^^^^^^^^^^^^
# `min_cluster_size` is the minimum number of samples in a group for that
# group to be considered a cluster.
#
# Clusters smaller than the ones of this size will be left as noise.
# The default value is 5. This parameter is generally tuned to
# larger values as needed. Smaller values will likely to lead to results with
# fewer points labeled as noise. However values which too small will lead to
# false sub-clusters being picked up and preferred. Larger values tend to be
# more robust with respect to noisy datasets, e.g. high-variance clusters with
# significant overlap.

# %%
# `min_samples`
# ^^^^^^^^^^^^^
# `min_samples` is the number of samples in a neighborhood for a point to
# be considered as a core point, including the point itself.
# `min_samples` defaults to `min_cluster_size`.
# Similarly to `min_cluster_size`, larger values for `min_samples` increase
# the model's robustness to noise, but risks ignoring or discarding
# potentially valid but small clusters.
# `min_samples` better be tuned after finding a good value for `min_cluster_size`.

# Different types:
# train: original non-preprocessed data (around min-cluster-size = 100 is nice)
# transform: transformed to make gaussian (around min-cluster-size = 40-60 is interesting, so is 150, 200)
# train_scaled: using StandardScalar from sklearn (around min-cluster-size = 90 is interesting)
# train_normed: normalizing the data taking the L2 norm (not really nice)

# for min samples tunning
# train: (100, 350) (100, 100)
# transform: (50, 10-60), (150, 60-90), (200, 30-60)
# train_scaled: (90, 100)
train_transformed, train_scaled, train_normalized, train, test = preprocessing_main()

types = [train, train_transformed, train_scaled, train_normalized]
PARAMS = [({"min_cluster_size": 100, "min_samples": 350}, {"min_cluster_size": 100, "min_samples": 100}),
          ({"min_cluster_size": 50, "min_samples": 50}, {"min_cluster_size": 150, "min_samples": 80},
           {"min_cluster_size": 200, "min_samples": 50}), ({"min_cluster_size": 90, "min_samples": 100},), ()]

PARAMS = [({"min_cluster_size": 100, "min_samples": 100},)]
# np.set_printoptions(threshold=np.inf)
for i, params in enumerate(PARAMS):
    for j, param in enumerate(params):
        if param:
            print(param)
            hdb = HDBSCAN(**param).fit(types[i])
            labels = hdb.labels_
            cluster_ids = []
            for idx, id in enumerate(set(labels)):
                cluster_id = [i for i, value in enumerate(labels) if value == id]

                cluster_ids.append(cluster_id)

train_vals = []
for idx, cluster in enumerate(cluster_ids):
    train_val = train[cluster, :]
    train_vals.append(train_val)

master_data = pd.read_csv('minimoon_master_final.csv', sep=" ", header=0, names=['Object id', 'H', 'D', 'Capture Date',
                                                             'Helio x at Capture', 'Helio y at Capture',
                                                             'Helio z at Capture', 'Helio vx at Capture',
                                                             'Helio vy at Capture', 'Helio vz at Capture',
                                                             'Helio q at Capture', 'Helio e at Capture',
                                                             'Helio i at Capture', 'Helio Omega at Capture',
                                                             'Helio omega at Capture', 'Helio M at Capture',
                                                             'Geo x at Capture', 'Geo y at Capture',
                                                             'Geo z at Capture', 'Geo vx at Capture',
                                                             'Geo vy at Capture', 'Geo vz at Capture',
                                                             'Geo q at Capture', 'Geo e at Capture',
                                                             'Geo i at Capture', 'Geo Omega at Capture',
                                                             'Geo omega at Capture', 'Geo M at Capture',
                                                             'Moon (Helio) x at Capture',
                                                             'Moon (Helio) y at Capture',
                                                             'Moon (Helio) z at Capture',
                                                             'Moon (Helio) vx at Capture',
                                                             'Moon (Helio) vy at Capture',
                                                             'Moon (Helio) vz at Capture',
                                                             'Capture Duration', 'Spec. En. Duration',
                                                             '3 Hill Duration', 'Number of Rev',
                                                             '1 Hill Duration', 'Min. Distance',
                                                             'Release Date', 'Helio x at Release',
                                                             'Helio y at Release', 'Helio z at Release',
                                                             'Helio vx at Release', 'Helio vy at Release',
                                                             'Helio vz at Release', 'Helio q at Release',
                                                             'Helio e at Release', 'Helio i at Release',
                                                             'Helio Omega at Release',
                                                             'Helio omega at Release',
                                                             'Helio M at Release', 'Geo x at Release',
                                                             'Geo y at Release', 'Geo z at Release',
                                                             'Geo vx at Release', 'Geo vy at Release',
                                                             'Geo vz at Release', 'Geo q at Release',
                                                             'Geo e at Release', 'Geo i at Release',
                                                             'Geo Omega at Release',
                                                             'Geo omega at Release', 'Geo M at Release',
                                                             'Moon (Helio) x at Release',
                                                             'Moon (Helio) y at Release',
                                                             'Moon (Helio) z at Release',
                                                             'Moon (Helio) vx at Release',
                                                             'Moon (Helio) vy at Release',
                                                             'Moon (Helio) vz at Release', 'Retrograde',
                                                             'Became Minimoon', 'Max. Distance', 'Capture Index',
                                                             'Release Index', 'X at Earth Hill', 'Y at Earth Hill',
                                                             'Z at Earth Hill', 'Taxonomy', 'STC', "EMS Duration",
                                                             "Periapsides in EMS", "Periapsides in 3 Hill",
                                                             "Periapsides in 2 Hill", "Periapsides in 1 Hill",
                                                             "STC Start", "STC Start Index", "STC End", "STC End Index",
                                                             "Helio x at EMS", "Helio y at EMS", "Helio z at EMS",
                                                             "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS",
                                                             "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
                                                             "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)",
                                                             "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
                                                             "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                                             "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
                                                             "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)",
                                                             'Entry Date to EMS', 'Entry to EMS Index',
                                                             'Exit Date to EMS', 'Exit Index to EMS',
                                                             "Dimensional Jacobi", "Non-Dimensional Jacobi", 'Alpha_I',
                                                             'Beta_I', 'Theta_M'])
final_clusters = []
columns_to_compare = ['1 Hill Duration', 'Min. Distance']
df_to_compare = master_data[columns_to_compare].to_numpy()

# go through all the files of test particles
population_dir = os.path.join('~/Documents/sean/minimoon_integrations', 'minimoon_files_oorb')
# variables
one_hill = 0.01
seconds_in_day = 86400
km_in_au = 149597870700 / 1000
mu_e = 3.986e5 * seconds_in_day ** 2 / np.power(km_in_au, 3)  # km^3/s^2

for j, cluster_vals in enumerate(train_vals):

    matching_indices = np.where(np.all(df_to_compare[:, None, :] == cluster_vals, axis=-1))

    ############################################################
    # for adding characteristics that don't exist
    ############################################################
    for i, cluster_item in enumerate(matching_indices[0]):
        print(i)
        if i > 3:
            break
        master_i = master_data.iloc[cluster_item]
        name = str(master_i['Object id']) + ".csv"
        data_i = pd.read_csv(population_dir + '/' + name,  sep=" ", header=0, names=["Object id", "Julian Date", "Distance", "Helio q",
        "Helio e", "Helio i", "Helio Omega", "Helio omega", "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx",
        "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z", "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i",
        "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)",
        "Earth vy (Helio)", "Earth vz (Helio)", "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)",
        "Moon vy (Helio)", "Moon vz (Helio)", "Synodic x", "Synodic y", "Synodic z", "Eclip Long"])

        # indices of trajectory when insides one hill
        in_1hill_idxs = [index for index, value in enumerate(data_i['Distance']) if value <= one_hill]
        # Get 1 Hill entrance and last exit indices

        # Get index of minimum distance and min dist.
        min_dist = min(data_i['Distance'])
        min_dist_index = data_i.index[data_i['Distance'] == min_dist]

        # Get min specific energy
        spec_energy_in_one_hill_temp = [(np.linalg.norm([data_i['Geo vx'].iloc[i], data_i['Geo vy'].iloc[i],
                                        data_i['Geo vz'].iloc[i]]) ** 2 / 2 - mu_e / data_i['Distance'].iloc[i]) * km_in_au ** 2 / seconds_in_day ** 2
                                        for i, value in data_i.iterrows()]
        if in_1hill_idxs:
            one_hill_start_idx = in_1hill_idxs[0]
            one_hill_end_idx = in_1hill_idxs[-1]

            spec_energy_in_one_hill = spec_energy_in_one_hill_temp[one_hill_start_idx:one_hill_end_idx]
            min_spec_energy = min(spec_energy_in_one_hill)
            min_spec_energy_ind = pd.Series(spec_energy_in_one_hill).idxmin()

        fig = plt.figure()
        ax1 = fig.add_subplot()
        ax1.plot(data_i['Julian Date'] - data_i['Julian Date'].iloc[0], data_i['Distance'], label='Geocentric Distance', color='grey', linewidth=1)
        ax1.scatter(data_i['Julian Date'].iloc[min_dist_index] - data_i['Julian Date'].iloc[0], data_i['Distance'].iloc[min_dist_index], color='black', label='Min. Distance')
        ax1.scatter(master_i['Capture Date'] - data_i['Julian Date'].iloc[0], data_i['Distance'].iloc[master_i['Capture Index']], color='yellow', label='Capture Start')
        ax1.set_ylabel('Distance to Earth (AU)')
        ax1.set_xlabel('Time (days)')
        ax1.set_title(str(master_i['Object id']))

        ax2 = ax1.twinx()
        ax2.plot(data_i['Julian Date'] - data_i['Julian Date'].iloc[0], spec_energy_in_one_hill_temp, color='tab:purple')

        if in_1hill_idxs:
            ax1.scatter(data_i['Julian Date'].iloc[one_hill_start_idx] - data_i['Julian Date'].iloc[0], data_i['Distance'].iloc[one_hill_start_idx], color='red', label='One Hill Start')
            ax1.scatter(data_i['Julian Date'].iloc[one_hill_end_idx] - data_i['Julian Date'].iloc[0] , data_i['Distance'].iloc[one_hill_end_idx],
                        color='blue', label='One Hill End')
            ax1.plot(data_i['Julian Date'].iloc[one_hill_start_idx:one_hill_end_idx] - data_i['Julian Date'].iloc[0],
                     data_i['Distance'].iloc[one_hill_start_idx:one_hill_end_idx],
                     label='Days in 1 Hill: ' + str(round(master_i['1 Hill Duration'], 2)), color='green', linewidth=3)

            ax2.scatter(data_i['Julian Date'].iloc[min_spec_energy_ind + one_hill_start_idx] - data_i['Julian Date'].iloc[0], min_spec_energy, color='pink', label='Min. Spec Energy')
        ax2.set_ylabel('Spec. Energy ($km^2/s^2$)', color='tab:purple')
        ax2.tick_params(axis='y', labelcolor='tab:purple')
        ax1.legend(loc='upper right')
        ax1.set_ylim([0, 0.03])
        ax2.legend(loc='upper left')
        ax2.set_ylim([-1, 1])
        plt.show()

# %%
# `dbscan_clustering`
# ^^^^^^^^^^^^^^^^^^^
# During `fit`, `HDBSCAN` builds a single-linkage tree which encodes the
# clustering of all points across all values of :class:`~cluster.DBSCAN`'s
# `eps` parameter.
# We can thus plot and evaluate these clusterings efficiently without fully
# recomputing intermediate values such as core-distances, mutual-reachability,
# and the minimum spanning tree. All we need to do is specify the `cut_distance`
# (equivalent to `eps`) we want to cluster with.