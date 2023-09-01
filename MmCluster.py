import pandas as pd
import numpy as np
from GeneticAlgorithm.genetic_selector import GeneticSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    file_path = 'cluster_df.csv'
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
                                      "Moon (Helio) x at Capture", "Moon (Helio) y at Capture", "Moon (Helio) z at Capture",
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
    rf_reg = RandomForestRegressor(n_estimators=300, random_state=random_state)

    # Load example dataset from Scikit-learn
    X = pd.DataFrame(data=features[1])
    y = pd.Series(data=targets.loc[:, '1 Hill Duration'])

    # Split into train and test
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=random_state)

    # Set a initial best chromosome for first population
    best_chromosome = np.array([1, 1, 0, 0, 0, 0, 0])

    # Create GeneticSelector instance
    # You should not set the number of cores (n_jobs) in the Scikit-learn
    # model to avoid UserWarning. The genetic selector is already parallelizable.
    genetic_selector = GeneticSelector(
        estimator=rf_reg, cv=5, n_gen=30, population_size=10,
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
