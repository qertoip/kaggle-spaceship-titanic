import os, sys, re, math, statistics as stat
from typing import TextIO
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from numpy import NaN
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.svm import SVC, SVR

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from xgboost import XGBClassifier

from matplotlib import pyplot as plt
import seaborn as sns


def main(train_io: TextIO, test_io: TextIO, out_io: TextIO):
    # Read raw data
    train_df, test_df = read_input(train_io, test_io)

    # Extract features
    x, y, x_test = extract_features(train_df, test_df)

    # Score
    score_model(x, y)

    # Feature selection
    feature_importance(x, y)

    # Learn
    model = new_model()
    train_model(model, x, y)

    # Use
    predictions = predict(model, x_test)

    # Write predictions
    write_output(out_io, x_test, predictions)
    print("Done!")


def read_input(train_io: TextIO, test_io: TextIO) -> (DataFrame, DataFrame):
    print("Reading input...")
    train_df = pd.read_csv(train_io, index_col='PassengerId')
    test_df = pd.read_csv(test_io, index_col='PassengerId')
    return train_df, test_df


def new_model():
    return CatBoostClassifier(
        random_state=1013, silent=True
    )


def extract_features(train_df: DataFrame, test_df: DataFrame):
    print("Extracting features...")
    all_df = pd.concat([train_df, test_df])
    all_df.sort_index(inplace=True)
    all_df = extract_features_df(all_df)

    train_xy_df = all_df[all_df['Transported'].notnull()]
    train_xy_df.sort_index(inplace=True)

    test_x_df = all_df[all_df['Transported'].isnull()]
    test_x_df.sort_index(inplace=True)
    test_x_df = test_x_df.drop(columns=['Transported'])

    features_correlation = train_xy_df.corr()
    features_correlation_abs = features_correlation.abs()

    train_x_df = train_xy_df.drop(columns='Transported')
    train_y_sr = train_xy_df['Transported']

    # for label in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    #     boxplot(label, train_xy_df)

    return train_x_df, train_y_sr, test_x_df


def extract_features_df(df: DataFrame) -> DataFrame:
    #             HomePlanet CryoSleep  Cabin    Destination   Age    VIP  RoomService  FoodCourt  ShoppingMall     Spa  VRDeck                Name Transported
    # PassengerId
    # 0001_01         Europa     False  B/0/P    TRAPPIST-1e  39.0  False          0.0        0.0           0.0     0.0     0.0     Maham Ofracculy       False
    # 0002_01          Earth     False  F/0/S    TRAPPIST-1e  24.0  False        109.0        9.0          25.0   549.0    44.0        Juanna Vines        True
    # 0003_01         Europa     False  A/0/S    TRAPPIST-1e  58.0   True         43.0     3576.0           0.0  6715.0    49.0       Altark Susent       False
    # 0003_02         Europa     False  A/0/S    TRAPPIST-1e  33.0  False          0.0     1283.0         371.0  3329.0   193.0        Solam Susent       False
    # 0004_01          Earth     False  F/1/S    TRAPPIST-1e  16.0  False        303.0       70.0         151.0   565.0     2.0   Willy Santantines        True
    # 0005_01          Earth     False  F/0/P  PSO J318.5-22  44.0  False          0.0      483.0           0.0   291.0     0.0   Sandie Hinetthews        True
    # 0006_01          Earth     False  F/2/S    TRAPPIST-1e  26.0  False         42.0     1539.0           3.0     0.0     0.0  Billex Jacostaffey        True
    # 0006_02          Earth      True  G/0/S    TRAPPIST-1e  28.0  False          0.0        0.0           0.0     0.0     NaN  Candra Jacostaffey        True
    # 0007_01          Earth     False  F/3/S    TRAPPIST-1e  35.0  False          0.0      785.0          17.0   216.0     0.0       Andona Beston        True
    # 0008_01         Europa      True  B/1/P    55 Cancri e  14.0  False          0.0        0.0           0.0     0.0     0.0      Erraiam Flatic        True

    # Drop not useful
    df.drop(columns=['Name'], inplace=True)

    # Extract from: PassengerIdGroupSize: int, GroupTransported: float64
    df = df.join(extract_group_size_and_group_transported(df))

    # Bool to int
    bin_cols = ['CryoSleep', 'VIP']
    df.replace({True: 1, False: 0}, inplace=True)
    df[bin_cols] = df[bin_cols].fillna(value=0).astype(int)

    # Float NaN-s
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Age'] = df['Age'].astype(int)

    # TotalExpenses
    # room_service = df['RoomService'].fillna(0)
    # food_court = df['FoodCourt'].fillna(0)
    # shopping_mall = df['ShoppingMall'].fillna(0)
    # spa = df['Spa'].fillna(0)
    # vrdeck = df['VRDeck'].fillna(0)

    df['FoodCourt'].fillna(0, inplace=True)
    df['ShoppingMall'].fillna(0, inplace=True)
    df['RoomService'].fillna(0, inplace=True)
    df['Spa'].fillna(0, inplace=True)
    df['VRDeck'].fillna(0, inplace=True)

    df['ExpensesTransported'] = df['FoodCourt'] + df['ShoppingMall']
    df['ExpensesNotTransported'] = df['VRDeck'] + df['Spa'] + df['RoomService']
    df['TotalExpenses'] = df['FoodCourt'] + df['ShoppingMall'] + df['VRDeck'] + df['Spa'] + df['RoomService']

    # Expenses to log10
    for label in ['FoodCourt', 'ShoppingMall', 'RoomService', 'Spa', 'VRDeck', 'ExpensesTransported', 'ExpensesNotTransported', 'TotalExpenses']:
        df[label] = np.log10(df[label] + 1)

    # df.drop(columns=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], inplace=True)

    hot_encoding = pd.get_dummies(df['HomePlanet'], prefix='HomePlanet')
    df = df.join(hot_encoding)
    df.drop(columns='HomePlanet', inplace=True)
    #df['HomePlanet'].fillna('N/A', inplace=True)

    hot_encoding = pd.get_dummies(df['Destination'], prefix='Destination')
    df = df.join(hot_encoding)
    df.drop(columns='Destination', inplace=True)
    # df['Destination'].fillna('N/A', inplace=True)

    df = df.join(cabin_parts(df['Cabin']))
    df.drop(columns='Cabin', inplace=True)

    #df.drop(columns=['HomePlanet_Europa', 'HomePlanet_Earth', 'Destination_PSO J318.5-22', 'Cabin_Deck_G', 'Destination_55 Cancri e', 'Cabin_Deck_T', 'Cabin_Deck_A', 'PassengerId_GroupSize', 'PassengerId_Postfix', 'VIP', 'Destination_TRAPPIST-1e', 'Cabin_Deck_F', 'Cabin_Deck_D'], inplace=True)
    
    #df['PreviousPassengerTransported'] = df['Transported'].shift(1)
    #df['PreviousPassengerTransported'].ffill(inplace=True)

    #df['NextPassengerTransported'] = df['Transported'].shift(-1)
    #df['NextPassengerTransported'].bfill(inplace=True)

    return df


def extract_group_size_and_group_transported(df: DataFrame) -> DataFrame:
    # 0001_01    -> group size 1
    # 0002_01    -> group size 1
    # 0003_01    |
    # 0003_02    | -> group size 2
    split_df = df.index.str.split('_', expand=True).to_frame()
    prefixes = sorted(map(lambda s: int(s), split_df[0].values))
    postfixes = list(map(lambda s: int(s), split_df[1].values))
    group_sizes = Counter(prefixes)
    group_transported = defaultdict(int)
    for i, row in df.iterrows():
        if 'Transported' in row and isinstance(row['Transported'], bool):
            group = int(i.split('_')[0])
            increase = int(row['Transported'])
            group_transported[group] += increase  # 0 or 1

    new_df = DataFrame(index=df.index)
    new_df['PassengerId_Prefix'] = prefixes
    new_df['PassengerId_Postfix'] = postfixes
    new_df['PassengerId_GroupSize'] = new_df['PassengerId_Prefix'].map(lambda group: group_sizes[group])
    new_df['PassengerId_Alone'] = (new_df['PassengerId_GroupSize'] == 1).astype(int)
    #new_df['GroupTransportedRatio'] = new_df['PassengerId_Prefix'].map(lambda group: group_transported[group] / group_sizes[group])
    new_df.drop(columns=['PassengerId_Prefix', 'PassengerId_Postfix', 'PassengerId_GroupSize'], inplace=True)
    return new_df


def cabin_parts(cabin: Series) -> DataFrame:
    sr = cabin.map(lambda value: value.split('/'), na_action='ignore')

    df = DataFrame(index=cabin.index)
    df.index.name = 'PassengerId'

    # Deck is categorical
    categorical = sr.map(lambda triple: triple[0], na_action='ignore')
    hot_encoding = pd.get_dummies(categorical, prefix='Cabin_Deck')
    df = df.join(hot_encoding)
    # df['CabinDeck'] = sr.map(lambda triple: 'BCGZAFDET'.index(triple[0]), na_action='ignore')  #  => <0, 8>
    # categorical = sr.map(lambda triple: triple[0], na_action='ignore')
    # df['Cabin_Deck'] = categorical
    # df['Cabin_Deck'].fillna('N/A', inplace=True)

    # Num is integer
    df['Cabin_Num'] = sr.map(lambda triple: int(triple[1]), na_action='ignore')
    df['Cabin_Num'].fillna(df['Cabin_Num'].mean(), inplace=True)
    df['Cabin_Num_Group300'] = df['Cabin_Num'].floordiv(300).astype(int)
    # hot_encoding = pd.get_dummies(df['Cabin_Num_Group300'], prefix='Cabin_Num_Group300')
    # df = df.join(hot_encoding)
    # df.drop(columns='Cabin_Num_Group300', inplace=True)
    df.drop(columns='Cabin_Num', inplace=True)

    # Side is binary
    df['Cabin_Side'] = sr.map(lambda triple: triple[2], na_action='ignore')
    df.replace({'P': 1, 'S': 0}, inplace=True)
    df['Cabin_Side'] = df['Cabin_Side'].fillna(0.5)

    return df


def score_model(x: DataFrame, y):
    print("CV-scoring model...")
    model = new_model()
    result = cross_val_score(model, x, y, cv=5)
    print()
    print(f'  Mean CV score: {result.mean()*100:.2f}%')
    print(f'Stdev CV scores: {result.std()*100:.2f}%')
    print(f'  All CV scores: {[str(round(r*100, 2)) + "%" for r in result]}')
    print(f'Lowest CV score: {min(result*100):.2f}%')


def feature_importance(x, y):
    # importance = chi2(x, y)
    # chi2_scores = pd.Series(importance[0])
    # chi2_scores.name = 'Chi2'
    # chi2_scores.index = x.columns
    # chi2_scores.sort_values(inplace=True, ascending=False)
    # print(chi2_scores)
    #
    # print()
    #
    # pvalue_scores = pd.Series(importance[1])
    # pvalue_scores.name = 'P-value'
    # pvalue_scores.index = x.columns
    # pvalue_scores.sort_values(inplace=True, ascending=True)
    # print('P-value:\n', pvalue_scores)

    model = new_model()

    print("Feature importance...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1013)

    print("  - training...")
    model.fit(x_train, y_train)

    scoring = ['accuracy']
    print("  - running permutations...")
    multi = permutation_importance(model, x_test, y_test, n_repeats=100, random_state=0, scoring=scoring)

    for metric in multi:
        print(f'Metric: {metric}')
        print('----------------------------------------------------------------')
        r = multi[metric]
        worst_cols = []
        for i in r.importances_mean.argsort()[::-1]:
            #if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            worst_cols.append(x.columns[i])
            print(f'    {x.columns[i]:<26}'
                  f'{r.importances_mean[i]:.5f}'
                  f' +/- {r.importances_std[i]:.5f}')
        print("', '".join(reversed(worst_cols)))


def train_model(model, x, y):
    print("Training...")
    model.fit(x, y)


def predict(model: Pipeline, x_challenges):
    print("Predicting outputs...")
    return model.predict(x_challenges)


def write_output(file: TextIO, x_test: DataFrame, predictions: list[float]):
    print("Saving outputs...")
    file.write('PassengerId,Transported\n')
    for passenger_id, prediction in zip(x_test.index, predictions):
        file.write(f'{passenger_id},{bool(prediction)}\n')


def boxplot(label, df):
    chart = sns.boxplot(x=df[label])
    chart.set(
        title=f'BoxPlot for {label}',
    )
    chart.get_figure().savefig(f'boxplot-{label}.png')
    chart.clear()
    plt.clf()
    plt.cla()


def nice_seaborn():
    sns.set(rc={'figure.dpi': 300, 'savefig.dpi': 300})
    sns.set_theme(
        context='paper',  # small fonts
        style='darkgrid',  # light grey background with grid lines (the default)
        palette='deep',  # low luminance, low saturation (the default)
    )


def nice_pandas():
    pd.options.display.float_format = '{:.8f}'.format


if __name__ == '__main__':
    nice_seaborn()
    nice_pandas()
    with open('../input/train.csv') as train, \
         open('../input/test.csv') as test,   \
         open('../output.csv', 'w') as output:
        main(train, test, output)
