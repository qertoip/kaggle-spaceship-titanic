import pandas as pd

from sklearn.feature_selection import chi2 as chi2_and_pvalue
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from kaggle_space_titanic.ml_utils import print_hr


def feature_importance(x, y, new_model):
    chi2_importance(x, y)
    pvalue_importance(x, y)
    perm_importance(x, y, new_model)
    model_importance(x, y, new_model)


def chi2_importance(x, y):
    print_hr()
    print('Chi-squared feature importance:\n')
    importance = chi2_and_pvalue(x, y)
    chi2_scores = pd.Series(importance[0])
    chi2_scores.name = 'Chi-squared'
    chi2_scores.index = x.columns
    chi2_scores.sort_values(inplace=True, ascending=False)
    print(chi2_scores)


def pvalue_importance(x, y):
    print_hr()
    print('P-value feature importance:\n')
    importance = chi2_and_pvalue(x, y)
    pvalue_scores = pd.Series(importance[1])
    pvalue_scores.name = 'P-value'
    pvalue_scores.index = x.columns
    pvalue_scores.sort_values(inplace=True, ascending=True)
    print(pvalue_scores)


def perm_importance(x, y, new_model):
    print_hr()
    print('Permutation feature importance:\n')

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1013)

    model = new_model()
    model.fit(x_train, y_train)

    scoring = ['accuracy']
    multi = permutation_importance(model, x_test, y_test, n_repeats=50, random_state=1013, scoring=scoring)

    for metric in multi:
        print(f'Metric: {metric}')
        r = multi[metric]
        worst_cols = []
        for i in r.importances_mean.argsort()[::-1]:
            #if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            worst_cols.append(x.columns[i])
            print(f'    {x.columns[i]:<26}'
                  f'{r.importances_mean[i]:.5f}'
                  f' +/- {r.importances_std[i]:.5f}')
        # print("', '".join(reversed(worst_cols)))


def model_importance(x, y, new_model):
    print_hr()
    print('Model subjective feature importance:\n')
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1013)
    model = new_model()
    model.fit(x_train, y_train)
    imp = model.get_feature_importance(prettified=True)
    print(imp)
