#!/usr/bin/env python

# This script generates all visualisations and runs all of the model training and evaluation
# NB: it is quite long running, particularly the parts with iterative_feature_addition and polynomial features

# This was converted from the .ipynb file with some cleaning up for legibility
# If you want to execute select parts of the code then I would recommend doing so using Jupyter Lab

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, cross_val_score
from sklearn.feature_selection import RFECV

CRITICAL_TEMP = 'critical_temp'


# Setup matplotlib

sns.set()


# Define functions

def print_metrics(y, y_hat):
    rmse = mean_squared_error(y, y_hat, squared=False)
    r2 = r2_score(y, y_hat)
    print(f'RMSE: {rmse:.4f}\nR^2: {r2:.4f}')

def print_cross_validate_scores(scores):
    for scoring, score in scores.items():
        if not scoring.startswith("test_"):
            continue

        scoring = scoring[len("test_"):]

        # don't display negative scores
        if scoring.startswith("neg"):
            scoring = scoring[len("neg_"):]
            score = -score

        # show root mean squared error instead of mean squared error
        if scoring == "mean_squared_error":
            scoring = "RMSE"
            score = np.sqrt(score)

        print(f'{scoring}: {score.mean():.2f} (+/- {(score.std() * 2):.2f})')

def plot_predictions(estimator, X, y, filename):
    predicted = cross_val_predict(estimator, X, y)

    fig, ax = plt.subplots()
    ax.scatter(y, predicted, s=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.savefig(filename)
    plt.show()

def run_linear_regression(*args, **kwargs):
    return run_regression(*args, **kwargs, regression=LinearRegression())

def run_regression(X, y, filename, regression, should_print_metrics=True, plot=False):
    scores = cross_validate(regression, X, y,
                            scoring=['neg_mean_squared_error', 'r2', 'neg_median_absolute_error'],
                           return_estimator=True)

    if should_print_metrics:
        print_cross_validate_scores(scores)

    if plot:
        plot_predictions(regression, X, y, filename)

    return scores['estimator']

def recursive_feature_elim(estimator, X, y, plot=False):
    linreg = LinearRegression()
    rfecv = RFECV(estimator=linreg, scoring="neg_mean_squared_error")
    rfecv.fit(X, y)

    print(f'Optimal number of features: {rfecv.n_features_}')

    if plot:
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Negative mean squared error")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

    # Inspect dropped features from RFE
    dropped_features = []
    for i, isSelected in enumerate(rfecv.support_):
        if not isSelected:
            dropped_features.append(list(X)[i])

    print(f'Dropped features: {", ".join(dropped_features)}')

    return rfecv, dropped_features

def iterative_feature_addition(scoring, num_folds, regression, print_progress=False, plot=True):
    # num_folds: increase for greater precision but worse runtime

    # At each step, test adding each element feature. Stick with the best one from each step, according to `scoring`
    # Keep track of the best set of features of any length, and return it

    X_grow = X.copy()
    all_time_best_score = -float('inf')  # beaten by whatever the first score is
    results = []

    while len(list(X_grow)) < len(list(X_incl_elements)):
        best_score = -float('inf')

        for element in list(all_elements_X):
            if element in list(X_grow):
                continue

            X_tmp = X_grow.copy()
            X_tmp[element] = X_elements[element].to_numpy()

            score = cross_val_score(regression, X_tmp, y, cv=num_folds, scoring=scoring).mean()
            if score > best_score:
                best_X = X_tmp
                best_score = score

        X_grow = best_X

        # optionally print progress because this is long running
        if print_progress:
            print(f'best score with {len(list(X_grow))} features: {best_score}')

        # store results for plotting or further analysis
        results.append({
            'num_features': len(list(X_grow)),
            'added_feature': list(best_X)[-1],
            'best_score': best_score
        })

        if best_score > all_time_best_score:
            all_time_best_X = best_X
            all_time_best_score = best_score

    if plot:
        plot_iterative_feature_addition_results(results)

    return all_time_best_X, all_time_best_score, results

def plot_iterative_feature_addition_results(results):
    x_points = [r['num_features'] for r in results]
    y_points = [r['best_score'] for r in results]
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel(f"Score")
    plt.plot(x_points, y_points)
    plt.show()

def get_poly_no_interaction(X, degree):
    # sklearn's PolynomialFeatures usefully adds polynomial terms, but also adds interacting terms (product of each feature)
    # we don't want the interacting terms, as this increases the amount of correlation between features

    poly = PolynomialFeatures(degree=degree)
    X_poly_interaction = poly.fit_transform(X).T
    X_poly_no_interaction = []

    names = poly.get_feature_names()
    for i in range(len(names)):
        if " " not in names[i]:
            X_poly_no_interaction.append(X_poly_interaction[i])

    return np.array(X_poly_no_interaction).T


# Load data and perform train-test splitting before doing any analysis

data_df = pd.read_csv('train.csv')
elements_df = pd.read_csv('unique_m.csv')

# check that the two tables' entries are in the same order
assert(elements_df[CRITICAL_TEMP].equals(data_df[CRITICAL_TEMP]))

# need to combine the two datasets before performing train-test split to make sure the right rows are in each split
all_data_X = data_df.loc[:, :'wtd_std_Valence']
all_elements_X = elements_df.loc[:, :'Rn']
all_X = pd.concat([all_data_X, all_elements_X], axis=1)

all_y = data_df[CRITICAL_TEMP]

# test size 0.33 to match paper for comparability
# random_state set to some constant int so that the results are reproduceable
# and so that rerunning the script does not constitute peeking at the test data
X_incl_elements, X_incl_elements_test, y, y_test = train_test_split(all_X, all_y, test_size=0.33, random_state=42)

# now create input matrices without the element data
X = X_incl_elements.loc[:, list(all_data_X)]
X_test = X_incl_elements_test.loc[:, list(all_data_X)]

X_elements = X_incl_elements.loc[:, list(all_elements_X)]
X_elements_test = X_incl_elements_test.loc[:, list(all_elements_X)]


# Calculate mean correlation between each feature and the critical temperature
aggregations = "mean wtd gmean entropy range std".split()

X_and_y = X.copy()
X_and_y.loc[:, CRITICAL_TEMP] = y

mean_corr = X.apply(lambda x: x.corr(y)).abs().mean()
elements_mean_corr = X_elements.apply(lambda x: x.corr(y)).abs().mean()
print(f'Mean correlation between general inputs and critical temp is {mean_corr}')
print(f'Mean correlation between elements inputs and critical temp is {elements_mean_corr}')


# Do some visualisation of the data to analyse the features

# First, univariate visualisation of the means

def get_categories(supercategory):
    return [cat for cat in list(X) if supercategory in cat and not any([a in cat for a in aggregations if a != supercategory])]

raw_categories = [CRITICAL_TEMP, "number_of_elements"] + get_categories("mean")

X_and_y.loc[:, raw_categories].hist(bins=10, color='steelblue', linewidth=1.0,
              xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(0, 0, 4, 4))


# Visualise relation between each of our standard categories, with line of best fit

raw_minus_crit = raw_categories.copy()
raw_minus_crit.remove(CRITICAL_TEMP)
raw_minus_crit.remove("number_of_elements")
for feature in raw_minus_crit:
    sns.jointplot(feature, CRITICAL_TEMP, data=X_and_y, kind="kde")

plt.tight_layout(rect=(0, 0, 4, 4))


# Visualise correlation between all features in a heatmap

f, ax = plt.subplots(figsize=(20, 16))
corr = X_and_y.corr()
heatmap = sns.heatmap(round(corr,2), ax=ax, cmap="coolwarm", square=True, xticklabels=True, yticklabels=True, linewidth=0.02)
f.subplots_adjust(top=1.2)
f.suptitle('Correlation between all features', fontsize=14);


# Basic linear regression
reg = run_linear_regression(X, y, "basic_regression", plot=True)


# Try standardising the input
X_scaled = preprocessing.scale(X)
run_linear_regression(X_scaled, y, "standardised_regression", plot=True);


# Perform recursive feature elimination

rfecv, dropped_features = recursive_feature_elim(LinearRegression(), X, y, plot=True)

# Run linear regression without the eliminated features, showing that model performance is not reduced
X_reduced = X.drop(dropped_features, axis=1)
run_linear_regression(X_reduced, y, "RFE dropped features", plot=True)

# And let's inspect the coefficients of the dropped features in a linear regression on all basic features
models = run_linear_regression(X, y, "basic_regression")
for dropped_feature in dropped_features:
    index = list(X).index(dropped_feature)
    coeffs = [model.coef_[index] for model in models]

    print(f'Mean coefficient for {dropped_feature}: {sum(coeffs) / float(len(coeffs))}')


# Try also including data on which elements are present

run_linear_regression(X_incl_elements, y, "X_incl_elements", plot=True)


# Try to determine the best subset of *all* features using RFE
rfecv, dropped_features = recursive_feature_elim(LinearRegression(), X_incl_elements, y, plot=True)


# Run linear regression without the eliminated features, demonstrating performance is not great with the reduced feature set
X_reduced = X_incl_elements.drop(dropped_features, axis=1)
run_linear_regression(X_reduced, y, "elements_numfeatures", plot=True)


# Because the above method seemed to perform very poorly, try a more careful approach
# Rather than removing features, add element features to the standard data features gradually, and keep track of the best performing set

all_time_best_X, all_time_best_score, results = iterative_feature_addition("neg_mean_squared_error", 3, LinearRegression(), print_progress=True, plot=False)
print(f'Optimal number of features {len(list(all_time_best_X))}')
plot_iterative_feature_addition_results(results[0:-2])
plot_iterative_feature_addition_results(results)

# Inspect those really bad last elements added
last_two_elements_added = [results[i]['added_feature'] for i in range(-2, 0)]  # it's ['Cd', 'Pr']
print(f'Poor performing last elements: {last_two_elements_added}')


run_linear_regression(all_time_best_X, y, "featureadditionresult", plot=True)


for i in range(1, 4):
    print("degree " + str(i))
    X_poly_higher_degree = get_poly_no_interaction(X, i)
    run_linear_regression(X_poly_higher_degree, y, f"polynomial_degree_{i}", plot=True)


for i in range(1, 4):
    print("expanded - degree " + str(i))
    X_poly_higher_degree = get_poly_no_interaction(all_time_best_X, i)
    run_linear_regression(X_poly_higher_degree, y, f"polynomial_expanded_degree_{i}", plot=True)


# Try regularising input

run_regression(X, y, "ridge", RidgeCV(), plot=True)

# Ridge does not make much difference with the normal feature set or the full best feature set
X_poly_degree_3 = get_poly_no_interaction(X, 3)
run_regression(X_poly_degree_3, y, "ridge_poly", RidgeCV(), plot=True)


# Finally, run the best model found so far on the test set, for the final evaluation

# Train
model = LinearRegression().fit(X_poly_degree_3, y)

# Test
X_test_poly_3 = get_poly_no_interaction(X_test, 3)
predictions = model.predict(X_test_poly_3)

print('Test set performance')
metrics = {'RMSE': (lambda x, y: math.sqrt(mean_squared_error(x, y))), 'r2': r2_score, 'Median absolute error': median_absolute_error}
for metric, func in metrics.items():
    print(f'{metric}: {func(predictions, y_test)}')

fig, ax = plt.subplots()
ax.scatter(y_test, predictions, s=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.savefig("test_set_predictions")
plt.show()
