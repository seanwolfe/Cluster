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

for i, params in enumerate(PARAMS):
    for j, param in enumerate(params):
        if param:
            print(param)
            hdb = HDBSCAN(**param).fit(types[i])
            labels = hdb.labels_
            fig, axes = plt.subplots(figsize=(7, 5))
            plot(train[:, :2], labels, hdb.probabilities_, param, ax=axes)
            fig, axes = plt.subplots(figsize=(7, 5))
            plot(types[i][:, :2], labels, hdb.probabilities_, param, ax=axes)
            fig, axes = plt.subplots(figsize=(7, 5))
            plot(train[:, 1:], labels, hdb.probabilities_, param, ax=axes)
            fig, axes = plt.subplots(figsize=(7, 5))
            plot(types[i][:, 1:], labels, hdb.probabilities_, param, ax=axes)
            fig, axes = plt.subplots(figsize=(7, 5))
            plot(train[:, [0, 2]], labels, hdb.probabilities_, param, ax=axes)
            fig, axes = plt.subplots(figsize=(7, 5))
            plot(types[i][:, [0, 2]], labels, hdb.probabilities_, param, ax=axes)
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