import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Setup cross-validation for X and Y
X_Train, X_Test, y_Train, y_Test = train_test_split(diabetes_X, diabetes_y, test_size=0.5, random_state=420)


def lin_reg():
    # Initialize Linear Regression Model
    lin_reg = linear_model.LinearRegression()

    # Train the model using the training sets and perform prediction, print outputs
    lin_reg.fit(X_Train, y_Train)
    lin_y_pred = lin_reg.predict(X_Test)
    print("Mean squared error for Linear Regression: %.2f" % mean_squared_error(y_Test, lin_y_pred))
    print('Variance score for Linear Regression: %.2f' % r2_score(y_Test, lin_y_pred))
    print('\n')


def ridge_reg():
    # Initialize Linear Regression Model
    ridge_reg = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 2, 5, 10], cv=10)

    # Train the model using the training sets and return prediction
    ridge_reg.fit(X_Train, y_Train)
    print('Ridge Regression Score: ', ridge_reg.score(X_Train, y_Train))
    print('Ridge Regression Alpha', ridge_reg.alpha_)
    ridge_y_pred = ridge_reg.predict(X_Test)
    print("Mean squared error for Ridge Regression: %.2f" % mean_squared_error(y_Test, ridge_y_pred))
    print('Variance score for Ridge Regression: %.2f' % r2_score(y_Test, ridge_y_pred))
    print("\n")


def bayesian_ridge_reg():
    # Initialize Linear Regression Model
    bayesian_ridge_reg = linear_model.BayesianRidge()
    cv_results = cross_validate(bayesian_ridge_reg, X_Train, y_Train, cv=3)
    print(cv_results)

    # # Train the model using the training sets and return prediction
    # bayesian_ridge_reg.fit(X_Train, y_Train)
    # bayes_ridge_y_pred = bayesian_ridge_reg.predict(X_Test)
    # print("Mean squared error for Bayesian Ridge Regression: %.2f" % mean_squared_error(y_Test, bayes_ridge_y_pred))
    # print('Variance score for Bayesian Ridge Regression: %.2f' % r2_score(y_Test, bayes_ridge_y_pred))
    # print("\n")


lin_reg()
ridge_reg()
bayesian_ridge_reg()
