import pandas as pd
import numpy as np
from GeneticAlgorithm.genetic_selector import GeneticSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn import preprocessing
import matplotlib.gridspec as gridspec


rx_dict = {
    'Best chromosome': re.compile(r"Best chromosome: \[(?P<bchrome>(\d+ ){17}\d)"),
    'Best score': re.compile(r'Best score: (?P<bscore>-?\d+.\d+)'),
    'Best epoch': re.compile(r'Best epoch: (?P<bepoch>\d+)'),
    'Test scores': re.compile(r'Test scores: \[(?P<tscore>(-?\d+.\d+, ){30}-?\d+.\d+)'),
    'Train scores': re.compile(r'Train scores: \[(?P<trscore>(-?\d+.\d+, ){30}-?\d+.\d+)'),
    'Chromosomes history': re.compile(r'Chromosomes history: \[(?P<chist>(array\(\[(\d, ){17}\d\]\), ){30}array\(\[(\d, ){17}\d\]\)\])')
}

def _parse_line(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None

def parser(filepath):
    data = []  # create an empty list to collect the data
    # open the file and read through it line by line
    bchromes = []
    bscores = []
    bepochs = []
    tscores = []
    trscores = []
    chists = []
    with open(filepath, 'r') as file_object:
        line = file_object.readline()

        while line:

            # at each line check for a match with a regex
            key, match = _parse_line(line)

            # extract school name
            if key == 'Best chromosome':
                bchrome = match.group('bchrome')
                bchromes.append([int(value) for idx, value in enumerate(bchrome.split(" "))])
            elif key == 'Best score':
                bscore = match.group('bscore')
                bscores.append(float(bscore))
            elif key == 'Best epoch':
                bepoch = match.group('bepoch')
                bepochs.append(int(bepoch))
            elif key == 'Test scores':
                tscore = match.group('tscore')
                tscores.append([float(value) for idx, value in enumerate(tscore.split(", "))])
            elif key == 'Train scores':
                trscore = match.group('trscore')
                trscores.append([float(value) for idx, value in enumerate(trscore.split(", "))])
            elif key == 'Chromosomes history':
                chist = match.group('chist')
                c = []
                for idx, value in enumerate(chist.split('array(')):
                    if idx == 0:
                        pass
                    else:
                        # print(value.strip('[').strip(']), '))
                        value_i = value.strip('[')
                        value_ii = value_i.strip(']), ')
                        c.append([int(value_iii) for idxi, value_iii in enumerate(value_ii.split(','))])
                chists.append(np.array(c))

            line = file_object.readline()

    return bchromes, bscores, bepochs, tscores, trscores, chists

@staticmethod
def parse_main():

    bchromes_tot = []
    bscores_tot = []
    bepochs_tot = []
    tscores_tot = []
    trscores_tot = []
    chists_tot = []
    file_paths = ['trees-100-tco-moon-earth', 'trees-150-tco-moon-earth',
                  'trees-200-tco-moon-earth', 'trees-250-tco-moon-earth',
                  'trees-300-tco-moon-earth', 'trees-350-tco-moon-earth',
                  'trees-400-tco-moon-earth', 'trees-450-tco-moon-earth']

    for ix, file_path in enumerate(file_paths):
        bchromes, bscores, bepochs, tscores, trscores, chists = parser(file_path)
        bchromes_tot.append(bchromes)
        bscores_tot.append(bscores)
        tscores_tot.append(tscores)
        bepochs_tot.append(bepochs)
        trscores_tot.append(trscores)
    #
    fig = plt.figure()
    print(bscores_tot)
    print(trscores_tot)
    num_est = [100, 150, 200, 250, 300, 350, 400, 450]
    plt.plot(num_est, [-bscore[0] for idx, bscore in enumerate(bscores_tot)], color='blue', label='Testing Score')
    plt.plot(num_est, [-max(trscores[0]) for idx, trscores in enumerate(trscores_tot)], color='red', label='Training Score')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Root Mean Square Error (days)')
    plt.legend()
    plt.show()

    return

@staticmethod
def feature_selection_main():
    file_path = 'cluster_df_pruned.csv'
    cluster_data = pd.read_csv(file_path, sep=' ', header=0, names=["1 Hill Duration", "Min. Distance",
                                                                    "Helio x at Capture",
                                                                    "Helio y at Capture", "Helio z at Capture",
                                                                    "Helio vx at Capture", "Helio vy at Capture",
                                                                    "Helio vz at Capture", "Moon (Helio) x at Capture",
                                                                    "Moon (Helio) y at Capture",
                                                                    "Moon (Helio) z at Capture",
                                                                    "Moon (Helio) vx at Capture",
                                                                    "Moon (Helio) vy at Capture",
                                                                    "Moon (Helio) vz at Capture",
                                                                    "Capture Date", "Earth (Helio) x at Capture",
                                                                    "Earth (Helio) y at Capture",
                                                                    "Earth (Helio) z at Capture",
                                                                    "Earth (Helio) vx at Capture",
                                                                    "Earth (Helio) vy at Capture",
                                                                    "Earth (Helio) vz at Capture"])

    targets = cluster_data.loc[:, ("1 Hill Duration", 'Min. Distance')]
    features = [cluster_data.loc[:, ("Helio x at Capture", "Helio y at Capture", "Helio z at Capture",
                                     "Helio vx at Capture", "Helio vy at Capture", "Helio vz at Capture",
                                     "Moon (Helio) x at Capture", "Moon (Helio) y at Capture",
                                     "Moon (Helio) z at Capture",
                                     "Moon (Helio) vx at Capture", "Moon (Helio) vy at Capture",
                                     "Moon (Helio) vz at Capture", "Earth (Helio) x at Capture",
                                     "Earth (Helio) y at Capture", "Earth (Helio) z at Capture",
                                     "Earth (Helio) vx at Capture", "Earth (Helio) vy at Capture",
                                     "Earth (Helio) vz at Capture")],
                cluster_data.loc[:, ("Helio x at Capture", "Helio y at Capture", "Helio z at Capture",
                                     "Helio vx at Capture", "Helio vy at Capture", "Helio vz at Capture", "Capture Date"
                                     )]]

    # Set random state
    random_state = 42

    # Define estimator
    rf_reg = RandomForestRegressor(n_estimators=250, random_state=random_state)

    # Load example dataset from Scikit-learn
    one = 0
    X = pd.DataFrame(data=features[one])
    y = pd.Series(data=targets.loc[:, '1 Hill Duration'])

    # Split into train and test
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=random_state)

    # Set a initial best chromosome for first population
    best_chromosome = np.array([1, 0, 0, 0, 0, 0, 0]) if one == 1 else np.array(
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0])

    # Create GeneticSelector instance
    # You should not set the number of cores (n_jobs) in the Scikit-learn
    # model to avoid UserWarning. The genetic selector is already parallelizable.
    genetic_selector = GeneticSelector(
        estimator=rf_reg, scoring='neg_root_mean_squared_error', cv=5, n_gen=3, population_size=10,
        crossover_rate=0.8, mutation_rate=0.15, tournament_k=2,
        calc_train_score=True, initial_best_chromosome=best_chromosome,
        n_jobs=-1, random_state=random_state, verbose=0)

    # Fit features
    genetic_selector.fit(train_X, train_y)

    # Show the results
    support = genetic_selector.support()
    best_chromosome = support[0][0]
    score = support[0][1]
    best_epoch = support[0][2]
    print(f'Best chromosome: {best_chromosome} -> (Selected Features IDs: {np.where(best_chromosome)[0]})')
    print(f'Best score: {score}')
    print(f'Best epoch: {best_epoch}')

    test_scores = support[1]
    train_scores = support[2]
    chromosomes_history = support[3]
    print(f'Test scores: {test_scores}')
    print(f'Train scores: {train_scores}')
    print(f'Chromosomes history: {chromosomes_history}')

    return


def fuzzy_c_main(data):

    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

    # determining the optimal number of clusters
    fpcs = []
    for ncenters in range(2, 10):
        print(ncenters)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data.T, ncenters, 2, error=0.005, maxiter=5000, init=None)

        fig1, axes1 = plt.subplots(figsize=(5, 5))
        xpts = data[:, 0]
        ypts = data[:, 1]

        # highest cluster membership
        cluster_membership = np.argmax(u, axis=0)

        for j in range(ncenters):
            axes1.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', markersize=0.5, color=colors[j])

        # Mark the center of each fuzzy cluster
        for pt in cntr:
            axes1.plot(pt[0], pt[1], 'rs')

        axes1.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
        fig1.tight_layout()

        # Store fpc values for later
        fpcs.append(fpc)
    print(fpcs)
    plt.show()

    # for prediction
    # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    #     cluster_data.loc[:, ('Helio vz at Capture', 'Min. Distance', "1 Hill Duration",)].transpose(), 5, 2, error=0.00005, maxiter=5000, init=None)

    # new prediction data - test data
    # cluster_data_test = cluster_data_ini.loc[:2000, ('Helio vz at Capture', 'Min. Distance')]  # data you use to predict
    # validation_data = cluster_data_ini['1 Hill Duration'].iloc[:2000]  # data you want to predict
    # print(validation_data)

    # do prediction
    # u_pred, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    #     cluster_data_test.T, cntr[:, :2], 2, error=0.005, maxiter=1000)
    #
    # cluster_targets = cntr[:, 2]  # the estimates of the target to predict from fuzzy c-means

    # for idx, point in enumerate(u_pred.T):
    #
    #     print(point)
    #     highest cluster membership
    #     cluster_membership = np.argmax(point)

        # print(cluster_targets)

        # print("Fuzzy c-means predicts this object belongs to cluster: " + str(cluster_membership))
        # print("This cluster has duration in 1 Hill: " + str(cluster_targets[cluster_membership]))
        # print("This object has duration in 1 Hill: " + str(validation_data.iloc[idx]))
        # print("Error: " + str(abs(cluster_targets[cluster_membership]
        #                       - validation_data.iloc[idx])))
        # print("This cluster has minimum distance: " + str(cluster_targets[cluster_membership, 1]))
        # print("This object has minimum distance: " + str(validation_data['Min. Distance'].iloc[idx]))
        # print("Error: " + str(abs(cluster_targets[cluster_membership, 1]
        #                       - validation_data['Min. Distance'].iloc[idx])))


    #
    # fig2, axes2 = plt.subplots(figsize=(8, 8))
    # xpts = cluster_data['Helio x at Capture']
    # ypts = cluster_data['1 Hill Duration']
    # for j in range(ncenters):
    #     axes2.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', color=colors[j])
    #
    # # Mark the center of each fuzzy cluster
    # for pt in cntr:
    #     axes2.plot(pt[0], pt[18], 'rs')
    #
    # axes2.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    # fig2.tight_layout()
    #
    # fig3, axes3 = plt.subplots(figsize=(8, 8))
    # xpts = cluster_data['Helio y at Capture']
    # ypts = cluster_data['1 Hill Duration']
    # for j in range(ncenters):
    #     axes3.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', color=colors[j])
    #
    # # Mark the center of each fuzzy cluster
    # for pt in cntr:
    #     axes3.plot(pt[1], pt[18], 'rs')
    #
    # axes3.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    # fig3.tight_layout()

    # fig4, axes4 = plt.subplots()
    # xpts = cluster_data['Helio y at Capture']
    # ypts = cluster_data['1 Hill Duration']
    # for j in range(ncenters):
    #     axes4.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    # for pt in cntr:
    #     axes4.plot(pt[0], pt[10], 'rs')
    #
    # axes4.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    # fig4.tight_layout()

    # fig2, ax2 = plt.subplots()
    # ax2.plot(np.r_[2:11], fpcs)
    # ax2.set_xlabel("Number of centers")
    # ax2.set_ylabel("Fuzzy partition coefficient")
    # plt.show()

@staticmethod
def preprocessing_main():

    file_path = 'cluster_df_pruned.csv'
    cluster_data_ini = pd.read_csv(file_path, sep=' ', header=0, names=["1 Hill Duration", "Min. Distance",
                                                                        "Helio x at Capture", "Helio y at Capture",
                                                                        "Helio z at Capture", "Helio vx at Capture",
                                                                        "Helio vy at Capture", "Helio vz at Capture",
                                                                        "Moon (Helio) x at Capture",
                                                                        "Moon (Helio) y at Capture",
                                                                        "Moon (Helio) z at Capture",
                                                                        "Moon (Helio) vx at Capture",
                                                                        "Moon (Helio) vy at Capture",
                                                                        "Moon (Helio) vz at Capture",
                                                                        "Capture Date", "Earth (Helio) x at Capture",
                                                                        "Earth (Helio) y at Capture",
                                                                        "Earth (Helio) z at Capture",
                                                                        "Earth (Helio) vx at Capture",
                                                                        "Earth (Helio) vy at Capture",
                                                                        "Earth (Helio) vz at Capture"])

    cluster_data_train, cluster_data_test = train_test_split(cluster_data_ini.loc[:,
                                                             ('Helio vz at Capture', '1 Hill Duration', 'Min. Distance')], test_size=0.1,
                                                             random_state=0)

    # fig = plt.figure()
    # plt.hist(cluster_data['Helio vz at Capture'], 1000)
    # fig = plt.figure()
    # plt.hist(cluster_data_train['1 Hill Duration'], 1000)
    # fig = plt.figure()
    # plt.hist(cluster_data_train['Min. Distance'], 1000)

    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
    cluster_data_trans = quantile_transformer.fit_transform(cluster_data_train)
    scalar = preprocessing.StandardScaler().fit(cluster_data_train.to_numpy())
    train_scaled = scalar.transform(cluster_data_train.to_numpy())
    train_normed = preprocessing.normalize(cluster_data_train.to_numpy(), norm='l2')

    # fig = plt.figure()
    # plt.hist(cluster_data_trans[:, 0], 1000)
    # fig = plt.figure()
    # plt.hist(cluster_data_trans[:, 1], 1000)
    # fig = plt.figure()
    # plt.hist(cluster_data_trans[:, 2], 1000)
    # plt.show()

    return cluster_data_trans, train_scaled, train_normed, cluster_data_train.to_numpy(), cluster_data_test.to_numpy()

def OPTICS_main(cluster_data_train, cluster_data_trans):

    clust = OPTICS(min_samples=50, xi=0.0048, min_cluster_size=0.05)

    # Run the fit
    clust.fit(cluster_data_train)

    labels_050 = cluster_optics_dbscan(
        reachability=clust.reachability_,
        core_distances=clust.core_distances_,
        ordering=clust.ordering_,
        eps=0.5,
    )
    labels_200 = cluster_optics_dbscan(
        reachability=clust.reachability_,
        core_distances=clust.core_distances_,
        ordering=clust.ordering_,
        eps=2,
    )

    space = np.arange(len(cluster_data_trans))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])

    # Reachability plot
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
    ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
    ax1.set_ylabel("Reachability (epsilon distance)")
    ax1.set_title("Reachability Plot")

    # OPTICS
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in zip(range(0, 5), colors):
        Xk = cluster_data_trans[clust.labels_ == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax2.plot(cluster_data_trans[clust.labels_ == -1, 0], cluster_data_trans[clust.labels_ == -1, 1], "k+", alpha=0.1)
    ax2.set_title("Automatic Clustering\nOPTICS")

    # DBSCAN at 0.5
    colors = ["g.", "r.", "b.", "c."]
    for klass, color in zip(range(0, 4), colors):
        print(labels_050)
        cluster_data_train['labels 50'] = labels_050
        Xk = cluster_data_train[cluster_data_train['labels 50'] == klass]
        print(len(Xk))
        print(Xk)
        ax3.plot(Xk['1 Hill Duration'], Xk['Min. Distance'], color, alpha=0.3)
    outliers = cluster_data_train[cluster_data_train['labels 50'] == -1]
    ax3.plot(outliers['1 Hill Duration'], outliers['Min. Distance'], "k+", alpha=0.1)
    ax3.set_title("Clustering at 0.5 epsilon cut\nDBSCAN")

    # DBSCAN at 2.
    colors = ["g.", "m.", "y.", "c."]
    for klass, color in zip(range(0, 4), colors):
        Xk = cluster_data_trans[labels_200 == klass]
        ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax4.plot(cluster_data_trans[labels_200 == -1, 0], cluster_data_trans[labels_200 == -1, 1], "k+", alpha=0.1)
    ax4.set_title("Clustering at 2.0 epsilon cut\nDBSCAN")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # parse_main()
    # feature_selection_main()

    trans, scaled, normed, train, test = preprocessing_main()
    fuzzy_c_main(train)
