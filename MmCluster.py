import pandas as pd
import numpy as np
import sklearn.metrics

from GeneticAlgorithm.genetic_selector import GeneticSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn import preprocessing
import matplotlib.gridspec as gridspec
from sklearn import svm
from sklearn.pipeline import make_pipeline
import multiprocessing
import sklearn as sk
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestClassifier

rx_dict = {
    'Best chromosome': re.compile(r"Best chromosome: \[(?P<bchrome>(\d+ ){17}\d)"),
    'Best score': re.compile(r'Best score: (?P<bscore>-?\d+.\d+)'),
    'Best epoch': re.compile(r'Best epoch: (?P<bepoch>\d+)'),
    'Test scores': re.compile(r'Test scores: \[(?P<tscore>(-?\d+.\d+, ){30}-?\d+.\d+)'),
    'Train scores': re.compile(r'Train scores: \[(?P<trscore>(-?\d+.\d+, ){30}-?\d+.\d+)'),
    'Chromosomes history': re.compile(
        r'Chromosomes history: \[(?P<chist>(array\(\[(\d, ){17}\d\]\), ){30}array\(\[(\d, ){17}\d\]\)\])')
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
    plt.plot(num_est, [-max(trscores[0]) for idx, trscores in enumerate(trscores_tot)], color='red',
             label='Training Score')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Root Mean Square Error (days)')
    plt.legend()
    plt.show()

    return


@staticmethod
def feature_selection_main():
    file_path = 'cluster_df.csv'
    cluster_data_ini = pd.read_csv(file_path, sep=' ', header=0,
                                   names=["Object id", "1 Hill Duration", "Min. Distance", "EMS Duration", 'Retrograde',
                                          'STC', "Became Minimoon", "3 Hill Duration", "Helio x at Capture",
                                          "Helio y at Capture", "Helio z at Capture", "Helio vx at Capture",
                                          "Helio vy at Capture", "Helio vz at Capture", "Moon (Helio) x at Capture",
                                          "Moon (Helio) y at Capture", "Moon (Helio) z at Capture",
                                          "Moon (Helio) vx at Capture", "Moon (Helio) vy at Capture",
                                          "Moon (Helio) vz at Capture", "Capture Date", "Helio x at EMS",
                                          "Helio y at EMS", "Helio z at EMS", "Helio vx at EMS", "Helio vy at EMS",
                                          "Helio vz at EMS", "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
                                          "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)",
                                          "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)", "Moon x at EMS (Helio)",
                                          "Moon y at EMS (Helio)", "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
                                          "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", "Entry Date to EMS",
                                          "Earth (Helio) x at Capture", "Earth (Helio) y at Capture",
                                          "Earth (Helio) z at Capture", "Earth (Helio) vx at Capture",
                                          "Earth (Helio) vy at Capture", "Earth (Helio) vz at Capture"])

    targets = cluster_data_ini.loc[:, ("Retrograde", 'Min. Distance')]
    features = [cluster_data_ini.loc[:, ("Helio x at Capture", "Helio y at Capture", "Helio z at Capture",
                                         "Helio vx at Capture", "Helio vy at Capture", "Helio vz at Capture",
                                         "Moon (Helio) x at Capture", "Moon (Helio) y at Capture",
                                         "Moon (Helio) z at Capture",
                                         "Moon (Helio) vx at Capture", "Moon (Helio) vy at Capture",
                                         "Moon (Helio) vz at Capture", "Earth (Helio) x at Capture",
                                         "Earth (Helio) y at Capture", "Earth (Helio) z at Capture",
                                         "Earth (Helio) vx at Capture", "Earth (Helio) vy at Capture",
                                         "Earth (Helio) vz at Capture")],
                cluster_data_ini.loc[:, ("Helio x at Capture", "Helio y at Capture", "Helio z at Capture",
                                         "Helio vx at Capture", "Helio vy at Capture", "Helio vz at Capture",
                                         "Capture Date"
                                         )],
                cluster_data_ini.loc[:, ("Helio x at Capture", "Helio y at Capture", "Helio z at Capture",
                                         "Helio vx at Capture", "Helio vy at Capture", "Helio vz at Capture",
                                         "Moon (Helio) x at Capture", "Moon (Helio) y at Capture",
                                         "Moon (Helio) z at Capture",
                                         "Moon (Helio) vx at Capture", "Moon (Helio) vy at Capture",
                                         "Moon (Helio) vz at Capture", "Earth (Helio) x at Capture",
                                         "Earth (Helio) y at Capture", "Earth (Helio) z at Capture",
                                         "Earth (Helio) vx at Capture", "Earth (Helio) vy at Capture",
                                         "Earth (Helio) vz at Capture", "Capture Date")]]

    # Set random state
    random_state = 42

    # Define estimator
    rf_reg = RandomForestClassifier(n_estimators=250, random_state=random_state)

    # Load example dataset from Scikit-learn
    one = 0  # one for 7-element epoch features or 0 for 18-elements state vector based features
    X = pd.DataFrame(data=features[one])
    y = pd.Series(data=targets.loc[:, 'Retrograde'])

    # Split into train and test
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=random_state)

    # Set a initial best chromosome for first population

    # optimal is: [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0] for tco-moon-earth,
    # generic is: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for tco-moon-earth
    if one == 0:
        best_chromosome = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif one == 1:
        best_chromosome = np.array([1, 0, 0, 0, 0, 0, 0])
    else:
        best_chromosome = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Create GeneticSelector instance
    # You should not set the number of cores (n_jobs) in the Scikit-learn
    # model to avoid UserWarning. The genetic selector is already parallelizable.
    genetic_selector = GeneticSelector(estimator=rf_reg, cv=5, n_gen=10,
                                       population_size=10,
                                       crossover_rate=0.8, mutation_rate=0.15, tournament_k=2,
                                       calc_train_score=True, initial_best_chromosome=best_chromosome, n_jobs=-1,
                                       random_state=random_state, verbose=0)

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


@staticmethod
def make_set(master, variable, axislabel, ylim):
    fig8 = plt.figure()
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(master['Helio x at Capture'], variable, s=0.1)
    ax1.set_xlabel('Helio x at Capture (AU)')
    ax1.set_ylabel(axislabel)
    ax1.set_ylim(ylim)
    # plt.savefig("figures/helx_3hill.svg", format="svg")

    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(master['Helio y at Capture'], variable, s=0.1)
    ax2.set_xlabel('Helio y at Capture (AU)')
    ax2.set_ylabel(axislabel)
    ax2.set_ylim(ylim)
    # plt.savefig("figures/hely_3hill.svg", format="svg")

    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(master['Helio z at Capture'], variable, s=0.1)
    ax3.set_xlabel('Helio z at Capture (AU)')
    ax3.set_ylabel(axislabel)
    ax3.set_ylim(ylim)
    # plt.savefig("figures/helz_3hill.svg", format="svg")

    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(master['Helio vx at Capture'], variable, s=0.1)
    ax4.set_xlabel('Helio vx at Capture')
    ax4.set_ylabel(axislabel)
    ax4.set_ylim(ylim)
    # plt.savefig("figures/helvx_3hill.svg", format="svg")

    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(master['Helio vy at Capture'], variable, s=0.1)
    ax5.set_xlabel('Helio vy at Capture')
    ax5.set_ylabel(axislabel)
    ax5.set_ylim(ylim)
    # plt.savefig("figures/helvy_3hill.svg", format="svg")

    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(master['Helio vz at Capture'], variable, s=0.1)
    ax6.set_xlabel('Helio vz at Capture')
    ax6.set_ylabel(axislabel)
    ax6.set_ylim(ylim)
    # plt.savefig("figures/helvz_3hill.svg", format="svg")

    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(master['Capture Date'], variable, s=0.1)
    ax7.set_xlabel('Capture Date (JD)')
    ax7.set_ylabel(axislabel)
    ax7.set_ylim(ylim)

    fig8 = plt.figure()
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(master['Moon (Helio) x at Capture'], variable, s=0.1)
    ax1.set_xlabel('Moon (Helio) x at Capture (AU)')
    ax1.set_ylabel(axislabel)
    ax1.set_ylim(ylim)
    # plt.savefig("figures/helx_3hill.svg", format="svg")

    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(master['Moon (Helio) y at Capture'], variable, s=0.1)
    ax2.set_xlabel('Moon (Helio) y at Capture (AU)')
    ax2.set_ylabel(axislabel)
    ax2.set_ylim(ylim)
    # plt.savefig("figures/hely_3hill.svg", format="svg")

    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(master['Moon (Helio) z at Capture'], variable, s=0.1)
    ax3.set_xlabel('Moon (Helio) z at Capture (AU)')
    ax3.set_ylabel(axislabel)
    ax3.set_ylim(ylim)
    # plt.savefig("figures/helz_3hill.svg", format="svg")

    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(master['Moon (Helio) vx at Capture'], variable, s=0.1)
    ax4.set_xlabel('Moon (Helio) vx at Capture')
    ax4.set_ylabel(axislabel)
    ax4.set_ylim(ylim)
    # plt.savefig("figures/helvx_3hill.svg", format="svg")

    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(master['Moon (Helio) vy at Capture'], variable, s=0.1)
    ax5.set_xlabel('Moon (Helio) vy at Capture')
    ax5.set_ylabel(axislabel)
    ax5.set_ylim(ylim)
    # plt.savefig("figures/helvy_3hill.svg", format="svg")

    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(master['Moon (Helio) vz at Capture'], variable, s=0.1)
    ax6.set_xlabel('Moon (Helio) vz at Capture')
    ax6.set_ylabel(axislabel)
    ax6.set_ylim(ylim)
    # plt.savefig("figures/helvz_3hill.svg", format="svg")

    fig8 = plt.figure()
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(master['Earth (Helio) x at Capture'], variable, s=0.1)
    ax1.set_xlabel('Earth (Helio) x at Capture (AU)')
    ax1.set_ylabel(axislabel)
    ax1.set_ylim(ylim)
    # plt.savefig("figures/helx_3hill.svg", format="svg")

    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(master['Earth (Helio) y at Capture'], variable, s=0.1)
    ax2.set_xlabel('Earth (Helio) y at Capture (AU)')
    ax2.set_ylabel(axislabel)
    ax2.set_ylim(ylim)
    # plt.savefig("figures/hely_3hill.svg", format="svg")

    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(master['Earth (Helio) z at Capture'], variable, s=0.1)
    ax3.set_xlabel('Earth (Helio) z at Capture (AU)')
    ax3.set_ylabel(axislabel)
    ax3.set_ylim(ylim)
    # plt.savefig("figures/helz_3hill.svg", format="svg")

    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(master['Earth (Helio) vx at Capture'], variable, s=0.1)
    ax4.set_xlabel('Earth (Helio) vx at Capture')
    ax4.set_ylabel(axislabel)
    ax4.set_ylim(ylim)
    # plt.savefig("figures/helvx_3hill.svg", format="svg")

    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(master['Earth (Helio) vy at Capture'], variable, s=0.1)
    ax5.set_xlabel('Earth (Helio) vy at Capture')
    ax5.set_ylabel(axislabel)
    ax5.set_ylim(ylim)
    # plt.savefig("figures/helvy_3hill.svg", format="svg")

    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(master['Earth (Helio) vz at Capture'], variable, s=0.1)
    ax6.set_xlabel('Earth (Helio) vz at Capture')
    ax6.set_ylabel(axislabel)
    ax6.set_ylim(ylim)
    # plt.savefig("figures/helvz_3hill.svg", format="svg")


@staticmethod
def make_set2(master):
    c1_data = master[master['C1 Membership'] > 0.8]
    c2_data = master[master['C2 Membership'] > 0.8]
    c3_data = master[master['C3 Membership'] > 0.8]
    c4_data = master[master['C4 Membership'] > 0.8]
    c5_data = master[master['C5 Membership'] > 0.8]
    c6_data = master[master['C6 Membership'] > 0.8]
    c7_data = master[master['C7 Membership'] > 0.8]
    c8_data = master[master['C8 Membership'] > 0.8]
    ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen', 'grey', 'b', 'orange', 'g', 'r', 'c',
     'm', 'y', 'k', 'Brown', 'ForestGreen', 'grey']
    fig8 = plt.figure()
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(master['Helio x at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax1.scatter(c1_data['Helio x at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax1.scatter(c2_data['Helio x at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax1.scatter(c3_data['Helio x at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax1.scatter(c4_data['Helio x at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax1.scatter(c5_data['Helio x at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax1.scatter(c6_data['Helio x at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax1.scatter(c7_data['Helio x at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax1.scatter(c8_data['Helio x at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax1.set_xlabel('Helio x at Capture (AU)')
    ax1.set_ylabel('1 Hill Duration (days)')
    ax1.set_ylim([0, 2500])
    # plt.savefig("figures/helx_3hill.svg", format="svg")

    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(master['Helio y at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax2.scatter(c1_data['Helio y at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax2.scatter(c2_data['Helio y at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax2.scatter(c3_data['Helio y at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax2.scatter(c4_data['Helio y at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax2.scatter(c5_data['Helio y at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax2.scatter(c6_data['Helio y at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax2.scatter(c7_data['Helio y at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax2.scatter(c8_data['Helio y at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax2.set_xlabel('Helio y at Capture (AU)')
    ax2.set_ylabel('1 Hill Duration (days)')
    ax2.set_ylim([0, 2500])
    # plt.savefig("figures/hely_3hill.svg", format="svg")

    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(master['Helio z at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax3.scatter(c1_data['Helio z at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax3.scatter(c2_data['Helio z at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax3.scatter(c3_data['Helio z at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax3.scatter(c4_data['Helio z at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax3.scatter(c5_data['Helio z at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax3.scatter(c6_data['Helio z at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax3.scatter(c7_data['Helio z at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax3.scatter(c8_data['Helio z at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax3.set_xlabel('Helio z at Capture (AU)')
    ax3.set_ylabel('1 Hill Duration (days)')
    ax3.set_ylim([0, 2500])
    # plt.savefig("figures/helz_3hill.svg", format="svg")

    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(master['Helio vx at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax4.scatter(c1_data['Helio vx at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax4.scatter(c2_data['Helio vx at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax4.scatter(c3_data['Helio vx at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax4.scatter(c4_data['Helio vx at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax4.scatter(c5_data['Helio vx at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax4.scatter(c6_data['Helio vx at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax4.scatter(c7_data['Helio vx at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax4.scatter(c8_data['Helio vx at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax4.set_xlabel('Helio vx at Capture (AU/day)')
    ax4.set_ylabel('1 Hill Duration (days)')
    ax4.set_ylim([0, 2500])
    # plt.savefig("figures/helvx_3hill.svg", format="svg")

    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(master['Helio vy at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax5.scatter(c1_data['Helio vy at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax5.scatter(c2_data['Helio vy at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax5.scatter(c3_data['Helio vy at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax5.scatter(c4_data['Helio vy at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax5.scatter(c5_data['Helio vy at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax5.scatter(c6_data['Helio vy at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax5.scatter(c7_data['Helio vy at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax5.scatter(c8_data['Helio vy at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax5.set_xlabel('Helio vy at Capture (AU/day)')
    ax5.set_ylabel('1 Hill Duration (days)')
    ax5.set_ylim([0, 2500])
    # plt.savefig("figures/helvy_3hill.svg", format="svg")

    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(master['Helio vz at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax6.scatter(c1_data['Helio vz at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax6.scatter(c2_data['Helio vz at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax6.scatter(c3_data['Helio vz at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax6.scatter(c4_data['Helio vz at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax6.scatter(c5_data['Helio vz at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax6.scatter(c6_data['Helio vz at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax6.scatter(c7_data['Helio vz at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax6.scatter(c8_data['Helio vz at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax6.set_xlabel('Helio vz at Capture (AU/day)')
    ax6.set_ylabel('1 Hill Duration (days)')
    ax6.set_ylim([0, 2500])
    # plt.savefig("figures/helvz_3hill.svg", format="svg")

    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(master['Capture Date'], master['1 Hill Duration'], s=0.1, color='b')
    ax7.scatter(c1_data['Capture Date'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax7.scatter(c2_data['Capture Date'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax7.scatter(c3_data['Capture Date'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax7.scatter(c4_data['Capture Date'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax7.scatter(c5_data['Capture Date'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax7.scatter(c6_data['Capture Date'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax7.scatter(c7_data['Capture Date'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax7.scatter(c8_data['Capture Date'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax7.set_xlabel('Capture Date (JD)')
    ax7.set_ylabel('1 Hill Duration (days)')
    ax7.set_ylim([0, 2500])

    fig8 = plt.figure()
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(master['Moon (Helio) x at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax1.scatter(c1_data['Moon (Helio) x at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax1.scatter(c2_data['Moon (Helio) x at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax1.scatter(c3_data['Moon (Helio) x at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax1.scatter(c4_data['Moon (Helio) x at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax1.scatter(c5_data['Moon (Helio) x at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax1.scatter(c6_data['Moon (Helio) x at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax1.scatter(c7_data['Moon (Helio) x at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax1.scatter(c8_data['Moon (Helio) x at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax1.set_xlabel('Moon (Helio) x at Capture (AU)')
    ax1.set_ylabel('1 Hill Duration (days)')
    ax1.set_ylim([0, 2500])
    # plt.savefig("figures/helx_3hill.svg", format="svg")

    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(master['Moon (Helio) y at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax2.scatter(c1_data['Moon (Helio) y at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax2.scatter(c2_data['Moon (Helio) y at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax2.scatter(c3_data['Moon (Helio) y at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax2.scatter(c4_data['Moon (Helio) y at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax2.scatter(c5_data['Moon (Helio) y at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax2.scatter(c6_data['Moon (Helio) y at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax2.scatter(c7_data['Moon (Helio) y at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax2.scatter(c8_data['Moon (Helio) y at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax2.set_xlabel('Moon (Helio) y at Capture (AU)')
    ax2.set_ylabel('1 Hill Duration (days)')
    ax2.set_ylim([0, 2500])
    # plt.savefig("figures/hely_3hill.svg", format="svg")

    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(master['Moon (Helio) z at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax3.scatter(c1_data['Moon (Helio) z at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax3.scatter(c2_data['Moon (Helio) z at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax3.scatter(c3_data['Moon (Helio) z at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax3.scatter(c4_data['Moon (Helio) z at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax3.scatter(c5_data['Moon (Helio) z at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax3.scatter(c6_data['Moon (Helio) z at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax3.scatter(c7_data['Moon (Helio) z at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax3.scatter(c8_data['Moon (Helio) z at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax3.set_xlabel('Moon (Helio) z at Capture (AU)')
    ax3.set_ylabel('1 Hill Duration (days)')
    ax3.set_ylim([0, 2500])
    # plt.savefig("figures/helz_3hill.svg", format="svg")

    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(master['Moon (Helio) vx at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax4.scatter(c1_data['Moon (Helio) vx at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax4.scatter(c2_data['Moon (Helio) vx at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax4.scatter(c3_data['Moon (Helio) vx at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax4.scatter(c4_data['Moon (Helio) vx at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax4.scatter(c5_data['Moon (Helio) vx at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax4.scatter(c6_data['Moon (Helio) vx at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax4.scatter(c7_data['Moon (Helio) vx at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax4.scatter(c8_data['Moon (Helio) vx at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax4.set_xlabel('Moon (Helio) vx at Capture (AU/day)')
    ax4.set_ylabel('1 Hill Duration (days)')
    ax4.set_ylim([0, 2500])
    # plt.savefig("figures/helvx_3hill.svg", format="svg")

    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(master['Moon (Helio) vy at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax5.scatter(c1_data['Moon (Helio) vy at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax5.scatter(c2_data['Moon (Helio) vy at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax5.scatter(c3_data['Moon (Helio) vy at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax5.scatter(c4_data['Moon (Helio) vy at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax5.scatter(c5_data['Moon (Helio) vy at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax5.scatter(c6_data['Moon (Helio) vy at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax5.scatter(c7_data['Moon (Helio) vy at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax5.scatter(c8_data['Moon (Helio) vy at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax5.set_xlabel('Moon (Helio) vy at Capture (AU/day)')
    ax5.set_ylabel('1 Hill Duration (days)')
    ax5.set_ylim([0, 2500])
    # plt.savefig("figures/helvy_3hill.svg", format="svg")

    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(master['Moon (Helio) vz at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax6.scatter(c1_data['Moon (Helio) vz at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax6.scatter(c2_data['Moon (Helio) vz at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax6.scatter(c3_data['Moon (Helio) vz at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax6.scatter(c4_data['Moon (Helio) vz at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax6.scatter(c5_data['Moon (Helio) vz at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax6.scatter(c6_data['Moon (Helio) vz at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax6.scatter(c7_data['Moon (Helio) vz at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax6.scatter(c8_data['Moon (Helio) vz at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax6.set_xlabel('Moon (Helio) vz at Capture (AU/day)')
    ax6.set_ylabel('1 Hill Duration (days)')
    ax6.set_ylim([0, 2500])
    # plt.savefig("figures/helvz_3hill.svg", format="svg")

    fig8 = plt.figure()
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(master['Earth (Helio) x at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax1.scatter(c1_data['Earth (Helio) x at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax1.scatter(c2_data['Earth (Helio) x at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax1.scatter(c3_data['Earth (Helio) x at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax1.scatter(c4_data['Earth (Helio) x at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax1.scatter(c5_data['Earth (Helio) x at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax1.scatter(c6_data['Earth (Helio) x at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax1.scatter(c7_data['Earth (Helio) x at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax1.scatter(c8_data['Earth (Helio) x at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax1.set_xlabel('Earth (Helio) x at Capture (AU)')
    ax1.set_ylabel('1 Hill Duration (days)')
    ax1.set_ylim([0, 2500])
    # plt.savefig("figures/helx_3hill.svg", format="svg")

    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(master['Earth (Helio) y at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax2.scatter(c1_data['Earth (Helio) y at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax2.scatter(c2_data['Earth (Helio) y at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax2.scatter(c3_data['Earth (Helio) y at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax2.scatter(c4_data['Earth (Helio) y at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax2.scatter(c5_data['Earth (Helio) y at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax2.scatter(c6_data['Earth (Helio) y at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax2.scatter(c7_data['Earth (Helio) y at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax2.scatter(c8_data['Earth (Helio) y at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax2.set_xlabel('Earth (Helio) y at Capture (AU)')
    ax2.set_ylabel('1 Hill Duration (days)')
    ax2.set_ylim([0, 2500])
    # plt.savefig("figures/hely_3hill.svg", format="svg")

    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(master['Earth (Helio) z at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax3.scatter(c1_data['Earth (Helio) z at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax3.scatter(c2_data['Earth (Helio) z at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax3.scatter(c3_data['Earth (Helio) z at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax3.scatter(c4_data['Earth (Helio) z at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax3.scatter(c5_data['Earth (Helio) z at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax3.scatter(c6_data['Earth (Helio) z at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax3.scatter(c7_data['Earth (Helio) z at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax3.scatter(c8_data['Earth (Helio) z at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax3.set_xlabel('Earth (Helio) z at Capture (AU)')
    ax3.set_ylabel('1 Hill Duration (days)')
    ax3.set_ylim([0, 2500])
    # plt.savefig("figures/helz_3hill.svg", format="svg")

    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(master['Earth (Helio) vx at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax4.scatter(c1_data['Earth (Helio) vx at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax4.scatter(c2_data['Earth (Helio) vx at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax4.scatter(c3_data['Earth (Helio) vx at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax4.scatter(c4_data['Earth (Helio) vx at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax4.scatter(c5_data['Earth (Helio) vx at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax4.scatter(c6_data['Earth (Helio) vx at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax4.scatter(c7_data['Earth (Helio) vx at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax4.scatter(c8_data['Earth (Helio) vx at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax4.set_xlabel('Earth (Helio) vx at Capture (AU/day)')
    ax4.set_ylabel('1 Hill Duration (days)')
    ax4.set_ylim([0, 2500])
    # plt.savefig("figures/helvx_3hill.svg", format="svg")

    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(master['Earth (Helio) vy at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax5.scatter(c1_data['Earth (Helio) vy at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax5.scatter(c2_data['Earth (Helio) vy at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax5.scatter(c3_data['Earth (Helio) vy at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax5.scatter(c4_data['Earth (Helio) vy at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax5.scatter(c5_data['Earth (Helio) vy at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax5.scatter(c6_data['Earth (Helio) vy at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax5.scatter(c7_data['Earth (Helio) vy at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax5.scatter(c8_data['Earth (Helio) vy at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax5.set_xlabel('Earth (Helio) vy at Capture (AU/day)')
    ax5.set_ylabel('1 Hill Duration (days)')
    ax5.set_ylim([0, 2500])
    # plt.savefig("figures/helvy_3hill.svg", format="svg")

    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(master['Earth (Helio) vz at Capture'], master['1 Hill Duration'], s=0.1, color='b')
    ax6.scatter(c1_data['Earth (Helio) vz at Capture'], c1_data['1 Hill Duration'], s=0.5, color='orange')
    ax6.scatter(c2_data['Earth (Helio) vz at Capture'], c2_data['1 Hill Duration'], s=0.5, color='g')
    ax6.scatter(c3_data['Earth (Helio) vz at Capture'], c3_data['1 Hill Duration'], s=0.5, color='r')
    ax6.scatter(c4_data['Earth (Helio) vz at Capture'], c4_data['1 Hill Duration'], s=0.5, color='Brown')
    ax6.scatter(c5_data['Earth (Helio) vz at Capture'], c5_data['1 Hill Duration'], s=0.5, color='m')
    ax6.scatter(c6_data['Earth (Helio) vz at Capture'], c6_data['1 Hill Duration'], s=0.5, color='y')
    ax6.scatter(c7_data['Earth (Helio) vz at Capture'], c7_data['1 Hill Duration'], s=0.5, color='k')
    ax6.scatter(c8_data['Earth (Helio) vz at Capture'], c8_data['1 Hill Duration'], s=0.5, color='grey')
    ax6.set_xlabel('Earth (Helio) vz at Capture (AU/day)')
    ax6.set_ylabel('1 Hill Duration (days)')
    ax6.set_ylim([0, 2500])
    plt.show()
    # plt.savefig("figures/helvz_3hill.svg", format="svg")
    return


@staticmethod
def make_set3(master):
    retrograde_pop = master[master['Retrograde'] == 1]
    prograde_pop = master[master['Retrograde'] == 0]

    fig8 = plt.figure()
    bin_size = 0.01
    retro_pts = retrograde_pop['Helio x at Capture']
    pro_pts = prograde_pop['Helio x at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.01
    retro_pts = retrograde_pop['Helio y at Capture']
    pro_pts = prograde_pop['Helio y at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.0001
    retro_pts = retrograde_pop['Helio z at Capture']
    pro_pts = prograde_pop['Helio z at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.0001
    retro_pts = retrograde_pop['Helio vx at Capture']
    pro_pts = prograde_pop['Helio vx at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.0001
    retro_pts = retrograde_pop['Helio vy at Capture']
    pro_pts = prograde_pop['Helio vy at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.000001
    retro_pts = retrograde_pop['Helio vz at Capture']
    pro_pts = prograde_pop['Helio vz at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.01
    retro_pts = retrograde_pop['Moon (Helio) x at Capture']
    pro_pts = prograde_pop['Moon (Helio) x at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)
    fig8 = plt.figure()
    bin_size = 0.01
    retro_pts = retrograde_pop['Moon (Helio) y at Capture']
    pro_pts = prograde_pop['Moon (Helio) y at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)
    fig8 = plt.figure()
    bin_size = 0.00001
    retro_pts = retrograde_pop['Moon (Helio) z at Capture']
    pro_pts = prograde_pop['Moon (Helio) z at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.00001
    retro_pts = retrograde_pop['Moon (Helio) vx at Capture']
    pro_pts = prograde_pop['Moon (Helio) vx at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.00001
    retro_pts = retrograde_pop['Moon (Helio) vy at Capture']
    pro_pts = prograde_pop['Moon (Helio) vy at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.0000001
    retro_pts = retrograde_pop['Moon (Helio) vz at Capture']
    pro_pts = prograde_pop['Moon (Helio) vz at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.01
    retro_pts = retrograde_pop['Earth (Helio) x at Capture']
    pro_pts = prograde_pop['Earth (Helio) x at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)
    fig8 = plt.figure()
    bin_size = 0.01
    retro_pts = retrograde_pop['Earth (Helio) y at Capture']
    pro_pts = prograde_pop['Earth (Helio) y at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)
    fig8 = plt.figure()
    bin_size = 0.00001
    retro_pts = retrograde_pop['Earth (Helio) z at Capture']
    pro_pts = prograde_pop['Earth (Helio) z at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.00001
    retro_pts = retrograde_pop['Earth (Helio) vx at Capture']
    pro_pts = prograde_pop['Earth (Helio) vx at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.00001
    retro_pts = retrograde_pop['Earth (Helio) vy at Capture']
    pro_pts = prograde_pop['Earth (Helio) vy at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)

    fig8 = plt.figure()
    bin_size = 0.0000001
    retro_pts = retrograde_pop['Earth (Helio) vz at Capture']
    pro_pts = prograde_pop['Earth (Helio) vz at Capture']
    # Calculate the number of bins based on data range and bin size
    retro_data_range = max(retro_pts) - min(retro_pts)
    pro_data_range = max(pro_pts) - min(pro_pts)
    retro_num_bins = int(retro_data_range / bin_size)
    pro_num_bins = int(pro_data_range / bin_size)
    plt.hist(retro_pts, bins=retro_num_bins)
    plt.hist(pro_pts, bins=pro_num_bins)
    plt.show()
    return


def make_set4(master, variable, axislabel, ylim):
    fig8 = plt.figure()
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(master['Synodic x at Capture'], variable, s=0.1)
    ax1.set_xlabel('Synodic x at Capture (AU)')
    ax1.set_ylabel(axislabel)
    ax1.set_ylim(ylim)
    # plt.savefig("figures/helx_3hill.svg", format="svg")

    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(master['Synodic y at Capture'], variable, s=0.1)
    ax2.set_xlabel('Synodic y at Capture (AU)')
    ax2.set_ylabel(axislabel)
    ax2.set_ylim(ylim)
    # plt.savefig("figures/hely_3hill.svg", format="svg")

    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(master['Synodic z at Capture'], variable, s=0.1)
    ax3.set_xlabel('Synodic z at Capture (AU)')
    ax3.set_ylabel(axislabel)
    ax3.set_ylim(ylim)
    # plt.savefig("figures/helz_3hill.svg", format="svg")

    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(master['Synodic vx at Capture'], variable, s=0.1)
    ax4.set_xlabel('Synodic vx at Capture')
    ax4.set_ylabel(axislabel)
    ax4.set_ylim(ylim)
    # plt.savefig("figures/helvx_3hill.svg", format="svg")

    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(master['Synodic vy at Capture'], variable, s=0.1)
    ax5.set_xlabel('Synodic vy at Capture')
    ax5.set_ylabel(axislabel)
    ax5.set_ylim(ylim)
    # plt.savefig("figures/helvy_3hill.svg", format="svg")

    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(master['Synodic vz at Capture'], variable, s=0.1)
    ax6.set_xlabel('Synodic vz at Capture')
    ax6.set_ylabel(axislabel)
    ax6.set_ylim(ylim)
    # plt.savefig("figures/helvz_3hill.svg", format="svg")

    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(master['Capture Date'], variable, s=0.1)
    ax7.set_xlabel('Capture Date (JD)')
    ax7.set_ylabel(axislabel)
    ax7.set_ylim(ylim)

    fig8 = plt.figure()
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(master['Moon (Synodic) x at Capture'], variable, s=0.1)
    ax1.set_xlabel('Moon (Synodic) x at Capture (AU)')
    ax1.set_ylabel(axislabel)
    ax1.set_ylim(ylim)
    # plt.savefig("figures/helx_3hill.svg", format="svg")

    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(master['Moon (Synodic) y at Capture'], variable, s=0.1)
    ax2.set_xlabel('Moon (Synodic) y at Capture (AU)')
    ax2.set_ylabel(axislabel)
    ax2.set_ylim(ylim)
    # plt.savefig("figures/hely_3hill.svg", format="svg")

    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(master['Moon (Synodic) z at Capture'], variable, s=0.1)
    ax3.set_xlabel('Moon (Synodic) z at Capture (AU)')
    ax3.set_ylabel(axislabel)
    ax3.set_ylim(ylim)
    # plt.savefig("figures/helz_3hill.svg", format="svg")

    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(master['Moon (Synodic) vx at Capture'], variable, s=0.1)
    ax4.set_xlabel('Moon (Synodic) vx at Capture')
    ax4.set_ylabel(axislabel)
    ax4.set_ylim(ylim)
    # plt.savefig("figures/helvx_3hill.svg", format="svg")

    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(master['Moon (Synodic) vy at Capture'], variable, s=0.1)
    ax5.set_xlabel('Moon (Synodic) vy at Capture')
    ax5.set_ylabel(axislabel)
    ax5.set_ylim(ylim)
    # plt.savefig("figures/helvy_3hill.svg", format="svg")

    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(master['Moon (Synodic) vz at Capture'], variable, s=0.1)
    ax6.set_xlabel('Moon (Synodic) vz at Capture')
    ax6.set_ylabel(axislabel)
    ax6.set_ylim(ylim)
    # plt.savefig("figures/helvz_3hill.svg", format="svg")

    plt.show()
    return


def eci_ecliptic_to_sunearth_synodic(object_id):
    """
    This function transforms coordinates from the ECI ecliptic plane to an Earth-centered Sun-Earth co-rotating frame,
    also referred to as a synodic frame
    :param sun_eph: the ephemeris x, y, z of the sun with respect to the ECI ecliptic frame 3 x n
    :param obj_xyz: the position of the object in the ECI ecliptic frame 3 x n
    :return: The transformed x, y, z coordinates 3 x n
    """

    file_path = 'cluster_df.csv'
    master = pd.read_csv(file_path, sep=' ', header=0,
                         names=['Object id', '1 Hill Duration', 'Min. Distance', 'EMS Duration', 'Retrograde',
                                'STC', 'Became Minimoon', 'Taxonomy',
                                '3 Hill Duration', 'Helio x at Capture', 'Helio y at Capture',
                                'Helio z at Capture', 'Helio vx at Capture',
                                'Helio vy at Capture', 'Helio vz at Capture',
                                'Moon (Helio) x at Capture',
                                'Moon (Helio) y at Capture',
                                'Moon (Helio) z at Capture',
                                'Moon (Helio) vx at Capture',
                                'Moon (Helio) vy at Capture',
                                'Moon (Helio) vz at Capture', 'Capture Date', "Helio x at EMS",
                                "Helio y at EMS",
                                "Helio z at EMS",
                                "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS",
                                "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
                                "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)",
                                "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
                                "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
                                "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", "Entry Date to EMS",
                                'Earth (Helio) x at Capture', 'Earth (Helio) y at Capture',
                                'Earth (Helio) z at Capture',
                                'Earth (Helio) vx at Capture', 'Earth (Helio) vy at Capture',
                                'Earth (Helio) vz at Capture'])

    obj_xyz = master[master['Object id'] == object_id].loc[:, ['Helio x at Capture', 'Helio y at Capture',
                                                               'Helio z at Capture']].to_numpy().reshape(3, )
    obj_vxyz = master[master['Object id'] == object_id].loc[:, ['Helio vx at Capture', 'Helio vy at Capture',
                                                                'Helio vz at Capture']].to_numpy().reshape(3, )
    moon_xyz = master[master['Object id'] == object_id].loc[:,
               ['Moon (Helio) x at Capture', 'Moon (Helio) y at Capture',
                'Moon (Helio) z at Capture']].to_numpy().reshape(3, )
    moon_vxyz = master[master['Object id'] == object_id].loc[:, ['Moon (Helio) vx at Capture',
                                                                 'Moon (Helio) vy at Capture',
                                                                 'Moon (Helio) vz at Capture']].to_numpy().reshape(3, )
    earth_xyz = master[master['Object id'] == object_id].loc[:, ['Earth (Helio) x at Capture',
                                                                 'Earth (Helio) y at Capture',
                                                                 'Earth (Helio) z at Capture']].to_numpy().reshape(3, )
    earth_vxyz = master[master['Object id'] == object_id].loc[:, ['Earth (Helio) vx at Capture',
                                                                  'Earth (Helio) vy at Capture',
                                                                  'Earth (Helio) vz at Capture']].to_numpy().reshape(
        3, )

    u_s = (- earth_xyz / np.linalg.norm(earth_xyz))
    theta = np.arctan2(u_s[1], u_s[0])

    # Rotation matrix - about z
    Rz_theta = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])

    omega = np.cross(earth_xyz, earth_vxyz) / np.linalg.norm(earth_xyz) ** 2

    t_obj_xyz = Rz_theta.T @ (obj_xyz - earth_xyz)
    t_moon_xyz = Rz_theta.T @ (moon_xyz - earth_xyz)
    v_obj_xyz = Rz_theta.T @ (obj_vxyz - earth_vxyz) - np.cross(Rz_theta.T @ omega, t_obj_xyz)
    v_moon_xyz = Rz_theta.T @ (moon_vxyz - earth_vxyz) - np.cross(Rz_theta.T @ omega, t_moon_xyz)
    ans = np.reshape([t_obj_xyz, t_moon_xyz, v_obj_xyz, v_moon_xyz], (12,))
    # print(ans)
    return ans


def fuzzy_c_main():
    file_path = 'cluster_df.csv'
    cluster_data_ini = pd.read_csv(file_path, sep=' ', header=0,
                                   names=["Object id", "1 Hill Duration", "Min. Distance", "EMS Duration", 'Retrograde',
                                          'STC', "Became Minimoon", "3 Hill Duration", "Helio x at Capture",
                                          "Helio y at Capture", "Helio z at Capture", "Helio vx at Capture",
                                          "Helio vy at Capture", "Helio vz at Capture", "Moon (Helio) x at Capture",
                                          "Moon (Helio) y at Capture", "Moon (Helio) z at Capture",
                                          "Moon (Helio) vx at Capture", "Moon (Helio) vy at Capture",
                                          "Moon (Helio) vz at Capture", "Capture Date", "Helio x at EMS",
                                          "Helio y at EMS", "Helio z at EMS", "Helio vx at EMS", "Helio vy at EMS",
                                          "Helio vz at EMS", "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
                                          "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)",
                                          "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)", "Moon x at EMS (Helio)",
                                          "Moon y at EMS (Helio)", "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
                                          "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", "Entry Date to EMS",
                                          "Earth (Helio) x at Capture", "Earth (Helio) y at Capture",
                                          "Earth (Helio) z at Capture", "Earth (Helio) vx at Capture",
                                          "Earth (Helio) vy at Capture", "Earth (Helio) vz at Capture"])

    # make_set(cluster_data_ini, cluster_data_ini['1 Hill Duration'], '1 Hill Duration', [0, 400])
    # make_set(cluster_data_ini, cluster_data_ini['Min. Distance'], 'Min. Distance', [0, 0.01])

    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen', 'grey', 'b', 'orange', 'g', 'r', 'c',
              'm', 'y', 'k', 'Brown', 'ForestGreen', 'grey']

    ncenters = 8
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np.array([cluster_data_ini['1 Hill Duration'].to_numpy()]),
                                                     ncenters, 2, error=0.005, maxiter=5000, init=None)

    cluster_data_ini['C1 Membership'] = u[0, :]
    cluster_data_ini['C2 Membership'] = u[1, :]
    cluster_data_ini['C3 Membership'] = u[2, :]
    cluster_data_ini['C4 Membership'] = u[3, :]
    cluster_data_ini['C5 Membership'] = u[4, :]
    cluster_data_ini['C6 Membership'] = u[5, :]
    cluster_data_ini['C7 Membership'] = u[6, :]
    cluster_data_ini['C8 Membership'] = u[7, :]

    make_set3(master=cluster_data_ini)
    make_set2(master=cluster_data_ini)

    # determining the optimal number of clusters
    # fpcs = []
    # for ncenters in range(2, 15):
    #     print(ncenters)
    #     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    #         np.array([cluster_data_ini['1 Hill Duration'].to_numpy()]), ncenters, 2, error=0.005, maxiter=5000,
    #         init=None)

    # highest cluster membership
    # cluster_membership = np.argmax(u, axis=0)

    # fig = plt.figure()
    # for j in range(ncenters):
    #     plt.scatter(cluster_data_ini['1 Hill Duration'], u[j, :], color=colors[j], s=1, label='Cluster: ' + str(j))
    #     plt.xlabel('1 Hill Duration')
    #     plt.ylabel('Memebership')
    #     plt.legend()

    # fig, ax = plt.subplots()
    # for j in range(ncenters):
    # Set the bin size
    # bin_size = 5
    # pts = cluster_data_ini['1 Hill Duration'].iloc[cluster_membership == j]
    # Calculate the number of bins based on data range and bin size
    # data_range = max(pts) - min(pts)
    # num_bins = int(data_range / bin_size)
    # plt.hist(pts, color=colors[j], bins=num_bins)
    # plt.xlim([0, 2500])
    # plt.title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    # plt.xlabel('1 Hill Duration (days)')
    # plt.ylabel('Count')

    # fig1, axes1 = plt.subplots(figsize=(5, 5))
    # xpts = data[:, 0]
    # ypts = data[:, 1]

    # for j in range(ncenters):
    #     axes1.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', markersize=0.5, color=colors[j])

    # Mark the center of each fuzzy cluster
    # for pt in cntr:
    #     axes1.plot(pt[0], pt[1], 'rs')
    #
    # axes1.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    # fig1.tight_layout()

    # Store fpc values for later
    # fpcs.append(fpc)
    # plt.show()
    # print(fpcs)
    # fig = plt.figure()
    # plt.plot(np.linspace(2, 14, len(fpcs)), fpcs)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Fuzzy Partition Index')
    # plt.show()

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
def preprocessing_main(X, y):
    # fig = plt.figure()
    # plt.hist(cluster_data['Helio vz at Capture'], 1000)
    # fig = plt.figure()
    # plt.hist(cluster_data_train['1 Hill Duration'], 1000)
    # fig = plt.figure()
    # plt.hist(cluster_data_train['Min. Distance'], 1000)
    random_state = 42
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=random_state)
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
    cluster_data_trans = quantile_transformer.fit_transform(test_X)
    scalar = preprocessing.StandardScaler().fit(train_X.to_numpy())
    train_scaled = scalar.transform(train_X.to_numpy())
    train_normed = preprocessing.normalize(train_X.to_numpy(), norm='l2')

    # fig = plt.figure()
    # plt.hist(cluster_data_trans[:, 0], 1000)
    # fig = plt.figure()
    # plt.hist(cluster_data_trans[:, 1], 1000)
    # fig = plt.figure()
    # plt.hist(cluster_data_trans[:, 2], 1000)
    # plt.show()

    return (cluster_data_trans, train_scaled, train_normed, train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(),
            test_y.to_numpy())


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


def estimator_tests():
    file_path = 'cluster_df.csv'
    cluster_data_ini = pd.read_csv(file_path, sep=' ', header=0,
                                   names=["Object id", "1 Hill Duration", "Min. Distance", "EMS Duration", 'Retrograde',
                                          'STC', "Became Minimoon", "3 Hill Duration", "Helio x at Capture",
                                          "Helio y at Capture", "Helio z at Capture", "Helio vx at Capture",
                                          "Helio vy at Capture", "Helio vz at Capture", "Moon (Helio) x at Capture",
                                          "Moon (Helio) y at Capture", "Moon (Helio) z at Capture",
                                          "Moon (Helio) vx at Capture", "Moon (Helio) vy at Capture",
                                          "Moon (Helio) vz at Capture", "Capture Date", "Helio x at EMS",
                                          "Helio y at EMS", "Helio z at EMS", "Helio vx at EMS", "Helio vy at EMS",
                                          "Helio vz at EMS", "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
                                          "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)",
                                          "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)", "Moon x at EMS (Helio)",
                                          "Moon y at EMS (Helio)", "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
                                          "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", "Entry Date to EMS",
                                          "Earth (Helio) x at Capture", "Earth (Helio) y at Capture",
                                          "Earth (Helio) z at Capture", "Earth (Helio) vx at Capture",
                                          "Earth (Helio) vy at Capture", "Earth (Helio) vz at Capture"])

    transed, scaled, normed, train_X, train_y, test_X, test_y = preprocessing_main(
        cluster_data_ini.loc[:, ["Helio x at Capture",
                                 "Helio y at Capture",
                                 "Helio z at Capture",
                                 "Helio vx at Capture",
                                 "Helio vy at Capture",
                                 "Helio vz at Capture",
                                 "Moon (Helio) x at Capture",
                                 "Moon (Helio) y at Capture",
                                 "Moon (Helio) z at Capture",
                                 "Moon (Helio) vx at Capture",
                                 "Moon (Helio) vy at Capture",
                                 "Moon (Helio) vz at Capture",
                                 "Earth (Helio) x at Capture",
                                 "Earth (Helio) y at Capture",
                                 "Earth (Helio) z at Capture",
                                 "Earth (Helio) vx at Capture",
                                 "Earth (Helio) vy at Capture",
                                 "Earth (Helio) vz at Capture"]],
        cluster_data_ini['Retrograde'])

    pipe = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier())
    pipe.fit(train_X, train_y)
    print(pipe.score(test_X, test_y))

    pipe = make_pipeline(RandomForestClassifier())
    pipe.fit(train_X, train_y)
    print(pipe.score(test_X, test_y))

    pipe4 = make_pipeline(preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
                          RandomForestClassifier())
    pipe4.fit(train_X, train_y)
    print(pipe4.score(test_X, test_y))

    # pipe = make_pipeline(preprocessing.StandardScaler(), MLPRegressor())
    # pipe.fit(train_X, train_y)
    # print(pipe.score(test_X, test_y))
    #
    # pipe2 = make_pipeline(MLPRegressor())
    # pipe2.fit(preprocessing.normalize(train_X), train_y)
    # print(pipe2.score(preprocessing.normalize(test_X), test_y))
    #
    # pipe3 = make_pipeline(MLPRegressor())
    # pipe3.fit(train_X, train_y)
    # print(pipe3.score(test_X, test_y))
    #
    # pipe4 = make_pipeline(preprocessing.QuantileTransformer(output_distribution='normal', random_state=0), MLPRegressor())
    # pipe4.fit(train_X, train_y)
    # print(pipe4.score(test_X, test_y))
    #
    # pipe = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor())
    # pipe.fit(train_X, train_y)
    # print(pipe.score(test_X, test_y))
    #
    # pipe2 = make_pipeline(RandomForestRegressor())
    # pipe2.fit(preprocessing.normalize(train_X), train_y)
    # print(pipe2.score(preprocessing.normalize(test_X), test_y))
    #
    # pipe3 = make_pipeline(RandomForestRegressor())
    # pipe3.fit(train_X, train_y)
    # print(pipe3.score(test_X, test_y))
    #
    # pipe4 = make_pipeline(preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
    #                       RandomForestRegressor())
    # pipe4.fit(train_X, train_y)
    # print(pipe4.score(test_X, test_y))

    # np.set_printoptions(threshold=np.inf)
    # print(clf.predict(test_X)[:100])
    # print(test_y[:100])
    # print(abs(test_y - clf.predict(test_X)/test_y * 100)[:100])
    # print(sum(abs(test_y - clf.predict(test_X))) / len(test_y))


@staticmethod
def synodic_main():
    file_path = 'cluster_df.csv'
    cluster_data_ini = pd.read_csv(file_path, sep=' ', header=0,
                                   names=['Object id', '1 Hill Duration', 'Min. Distance', 'EMS Duration', 'Retrograde',
                                          'STC', 'Became Minimoon', 'Taxonomy',
                                          '3 Hill Duration', 'Helio x at Capture', 'Helio y at Capture',
                                          'Helio z at Capture', 'Helio vx at Capture',
                                          'Helio vy at Capture', 'Helio vz at Capture',
                                          'Moon (Helio) x at Capture',
                                          'Moon (Helio) y at Capture',
                                          'Moon (Helio) z at Capture',
                                          'Moon (Helio) vx at Capture',
                                          'Moon (Helio) vy at Capture',
                                          'Moon (Helio) vz at Capture', 'Capture Date', "Helio x at EMS",
                                          "Helio y at EMS",
                                          "Helio z at EMS",
                                          "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS",
                                          "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
                                          "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)",
                                          "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
                                          "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                          "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
                                          "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", "Entry Date to EMS",
                                          'Earth (Helio) x at Capture', 'Earth (Helio) y at Capture',
                                          'Earth (Helio) z at Capture',
                                          'Earth (Helio) vx at Capture', 'Earth (Helio) vy at Capture',
                                          'Earth (Helio) vz at Capture'])

    print(cluster_data_ini)

    pool = multiprocessing.Pool()
    res = pool.map(eci_ecliptic_to_sunearth_synodic, cluster_data_ini['Object id'])  # input your function
    pool.close()
    res2 = np.array(res)
    cluster_data_ini['Synodic x at Capture'] = res2[:, 0]
    cluster_data_ini['Synodic y at Capture'] = res2[:, 1]
    cluster_data_ini['Synodic z at Capture'] = res2[:, 2]
    cluster_data_ini['Synodic vx at Capture'] = res2[:, 6]
    cluster_data_ini['Synodic vy at Capture'] = res2[:, 7]
    cluster_data_ini['Synodic vz at Capture'] = res2[:, 8]
    cluster_data_ini['Moon (Synodic) x at Capture'] = res2[:, 3]
    cluster_data_ini['Moon (Synodic) y at Capture'] = res2[:, 4]
    cluster_data_ini['Moon (Synodic) z at Capture'] = res2[:, 5]
    cluster_data_ini['Moon (Synodic) vx at Capture'] = res2[:, 9]
    cluster_data_ini['Moon (Synodic) vy at Capture'] = res2[:, 10]
    cluster_data_ini['Moon (Synodic) vz at Capture'] = res2[:, 11]

    cluster_data_ini.to_csv('cluster_df_synodic.csv', sep=' ', header=True, index=False)

    plt.scatter(cluster_data_ini['Synodic x at Capture'], cluster_data_ini['Synodic y at Capture'], s=0.1)
    plt.show()


if __name__ == '__main__':
    file_path = 'cluster_df_synodic.csv'
    master = pd.read_csv(file_path, sep=' ', header=0,
                         names=["Object id", "1 Hill Duration", "Min. Distance", "EMS Duration", 'Retrograde',
                                'STC', "Became Minimoon", "3 Hill Duration", "Helio x at Capture",
                                "Helio y at Capture", "Helio z at Capture", "Helio vx at Capture",
                                "Helio vy at Capture", "Helio vz at Capture", "Moon (Helio) x at Capture",
                                "Moon (Helio) y at Capture", "Moon (Helio) z at Capture",
                                "Moon (Helio) vx at Capture", "Moon (Helio) vy at Capture",
                                "Moon (Helio) vz at Capture", "Capture Date", "Helio x at EMS",
                                "Helio y at EMS", "Helio z at EMS", "Helio vx at EMS", "Helio vy at EMS",
                                "Helio vz at EMS", "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
                                "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)",
                                "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)", "Moon x at EMS (Helio)",
                                "Moon y at EMS (Helio)", "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
                                "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", "Entry Date to EMS",
                                "Earth (Helio) x at Capture", "Earth (Helio) y at Capture",
                                "Earth (Helio) z at Capture", "Earth (Helio) vx at Capture",
                                "Earth (Helio) vy at Capture", "Earth (Helio) vz at Capture", "Synodic x at Capture",
                                "Synodic y at Capture", "Synodic z at Capture", "Synodic vx at Capture",
                                "Synodic vy at Capture", "Synodic vz at Capture", "Moon (Synodic) x at Capture",
                                "Moon (Synodic) y at Capture", "Moon (Synodic) z at Capture",
                                "Moon (Synodic) vx at Capture", "Moon (Synodic) vy at Capture",
                                "Moon (Synodic) vz at Capture"])

    # make_set4(master, master['1 Hill Duration'], '1 Hill Duration', [0,10000])

    synodic_main()
    # parse_main()
    # feature_selection_main()

    # trans, scaled, normed, train, test = preprocessing_main()
    # fuzzy_c_main()
    # estimator_tests()
