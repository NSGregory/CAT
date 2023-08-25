
'''
This module is meant to fascilitate/automate the interpretation of machine
learning modules using several common libraries:
    - SHAP
    - ELI5
    - Yellowbrick

It is used as part of the CAT pipeline for assessing catwalk data
'''

#imports
import pandas as pd
import shap
import os
import sys
import sklearn

def shap_linear_example():
    X, y = shap.datasets.california(n_points=1000)
    X100 = shap.utils.sample(X, 100)

    #a simple linear model
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    for i in range(X.shape[1]):
        print(X.columns[i], '=', model.coef_[i].round(4))

    #explain the model's predictions using SHAP values
    explainer = shap.Explainer(model.predict, X100)
    shap_values = explainer(X100)
    sample_ind = 20
    shap.partial_dependence_plot("MedInc", model.predict, X100, ice=False,
                                 model_expected_value=True, feature_expected_value=True,
                                 shap_values=shap_values[sample_ind:sample_ind+1,:])
    return 1

def shap_classifier_example():

#main function -- the action starts here
def main():
    '''
    put the main function here
    '''
    #for now, pass
    pass
    return


if __name__ == '__main__':
    main()


