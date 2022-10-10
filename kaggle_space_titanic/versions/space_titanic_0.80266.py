import os, sys, re, math, statistics as stat
from typing import TextIO
from collections import Counter

import numpy as np
import pandas as pd
from numpy import NaN
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.svm import SVC, SVR

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from xgboost import XGBClassifier


def main(train_io: TextIO, test_io: TextIO, out_io: TextIO):
    # Read raw data
    train_df, test_df = read_input(train_io, test_io)

    # Extract features
    x, y, x_test = extract_features(train_df, test_df)

    model = make_pipeline(
        XGBClassifier(gamma=7, reg_lambda=25, subsample=0.5, random_state=1013)
    )

    # Score
    score_model(model, x, y)

    # Learn
    train_model(model, x, y)

    # Use
    predictions = predict(model, x_test)

    # Write predictions
    write_output(out_io, x_test, predictions)
    print("Done!")


def train_model(model, x, y):
    print("Training...")
    model.fit(x, y)


def col_names(base_str, start_ix, end_inclusive_ix):
    return [base_str + str(i) for i in range(start_ix, end_inclusive_ix + 1)]


def read_input(train_io: TextIO, test_io: TextIO) -> (DataFrame, DataFrame):
    print("Reading input...")
    train_df = pd.read_csv(train_io, index_col='PassengerId')
    test_df = pd.read_csv(test_io, index_col='PassengerId')
    # print(train_df.shape)
    # train_df.dropna(inplace=True)
    # print(train_df.shape)
    return train_df, test_df


def extract_features(train_df: DataFrame, test_df: DataFrame):
    print("Extracting features...")
    all_df = pd.concat([train_df, test_df])
    all_df.sort_index(inplace=True)
    all_df = extract_features_df(all_df)

    #train_xy_df = extract_features_df(train_df)
    #test_x_df = extract_features_df(test_df)
    #combined_x_df.sort_index(inplace=True)

    train_xy_df = all_df[all_df['Transported'].notnull()]
    train_xy_df.sort_index(inplace=True)

    test_x_df = all_df[all_df['Transported'].isnull()]
    test_x_df = test_x_df.drop(columns=['Transported'])
    test_x_df.sort_index(inplace=True)

    features_correlation = train_xy_df.corr()

    train_x_df = train_xy_df.drop(columns='Transported')
    train_y_sr = train_xy_df['Transported']

    return train_x_df, train_y_sr, test_x_df


def series_of_passenger_id_group_size(index: pd.Index) -> Series:
    # 0001_01    -> group size 1
    # 0002_01    -> group size 1
    # 0003_01    |
    # 0003_02    | -> group size 2

    prefixes = [item.split('_')[0] for item in sorted(index.values)]
    group_sizes = Counter(prefixes)
    sr = Series(index=index, name='PassengerIdGroupSize', dtype=int)
    for i in index:
        group = i.split('_')[0]
        sr._set_value(i, group_sizes[group])
    return sr


def extract_features_df(df: DataFrame) -> DataFrame:
    # PassengerId, HomePlanet, CryoSleep, Cabin, Destination, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Name, Transported

    # Drop categorical
    df.drop(columns=['Name'], inplace=True)

    # Extract group size from PassengerId
    df = df.join(series_of_passenger_id_group_size(df.index))

    # Bool to int
    bin_cols = ['CryoSleep', 'VIP']
    df.replace({True: 1, False: 0}, inplace=True)
    df[bin_cols] = df[bin_cols].fillna(value=0)

    # Float NaN-s
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['RoomService'] = df['RoomService'].fillna(0)
    df['FoodCourt'] = df['FoodCourt'].fillna(0)
    df['ShoppingMall'] = df['ShoppingMall'].fillna(0)
    df['Spa'] = df['Spa'].fillna(0)
    df['VRDeck'] = df['VRDeck'].fillna(0)

    home_planet_he_df = pd.get_dummies(df['HomePlanet'], prefix='HomePlanet')
    df = df.join(home_planet_he_df)
    df.drop(columns='HomePlanet', inplace=True)

    hot_encoding = pd.get_dummies(df['Destination'], prefix='Destination')
    df = df.join(hot_encoding)
    df.drop(columns='Destination', inplace=True)

    df = df.join(cabin_parts(df['Cabin']))
    df.drop(columns='Cabin', inplace=True)

    # df['PreviousPassengerTransported'] = df['Transported'].shift(1)
    # df['PreviousPassengerTransported'].ffill(inplace=True)
    #
    # df['NextPassengerTransported'] = df['Transported'].shift(-1)
    # df['NextPassengerTransported'].bfill(inplace=True)

    # relevant_columns = ['CryoSleep', 'RoomService', 'Spa', 'VRDeck', 'Cabin_P1_B', 'Cabin_P1_C', 'Cabin_P3', 'Cabin_P1_E', 'Cabin_P1_F', 'Age']
    # if 'Transported' in df.columns:
    #     relevant_columns.append('Transported')
    # df = df[relevant_columns]

    return df


def cabin_parts(cabin: Series) -> DataFrame:
    sr = cabin.map(lambda value: value.split('/'), na_action='ignore')

    df = DataFrame(index=cabin.index)
    df.index.name = 'PassengerId'

    # P1 is categorical
    categorical = sr.map(lambda triple: triple[0], na_action='ignore')
    hot_encoding = pd.get_dummies(categorical, prefix='Cabin_P1')
    df = df.join(hot_encoding)

    # P2 is integer
    df['Cabin_P2'] = sr.map(lambda triple: int(triple[1]), na_action='ignore')
    df['Cabin_P2'] = df['Cabin_P2'].fillna(df['Cabin_P2'].mean())

    # P3 is binary
    df['Cabin_P3'] = sr.map(lambda triple: triple[2], na_action='ignore')
    df.replace({'P': 1, 'S': 0}, inplace=True)
    df['Cabin_P3'] = df['Cabin_P3'].fillna(0.5)

    return df


def score_model(model, x, y):
    print("CV-scoring model...")
    result = cross_val_score(model, x, y)
    print()
    print(f'  Mean CV score: {result.mean()}')
    print(f'Stdev CV scores: {result.std()}')
    print(f'  All CV scores: {result}')
    print(f'Lowest CV score: {min(result)}')


def predict(model: Pipeline, x_challenges):
    print("Predicting outputs...")
    return model.predict(x_challenges)


def write_output(file: TextIO, x_test: DataFrame, predictions: list[float]):
    print("Saving outputs...")
    file.write('PassengerId,Transported\n')
    for passenger_id, prediction in zip(x_test.index, predictions):
        file.write(f'{passenger_id},{bool(prediction)}\n')


if __name__ == '__main__':
    with open('../input/train.csv') as train, \
         open('../input/test.csv') as test,   \
         open('../output.csv', 'w') as output:
        main(train, test, output)
