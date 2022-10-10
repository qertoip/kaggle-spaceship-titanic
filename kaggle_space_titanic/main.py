from collections import Counter, defaultdict
from typing import TextIO

import numpy as np
import pandas as pd

from pandas import DataFrame, Series

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from matplotlib import pyplot as plt
import seaborn as sns

from kaggle_space_titanic.feature_importance import feature_importance
from kaggle_space_titanic.ml_utils import *
from kaggle_space_titanic.charts import analyze_raw_data, analyze_extracted_data


def main(train_io: TextIO, test_io: TextIO, out_io: TextIO):
    # Read raw data
    train_df, test_df = read_input(train_io, test_io)

    # Analyze raw data
    analyze_raw_data(train_df, test_df)

    # Extract features
    train_df, test_df = extract_features(train_df, test_df)

    # Analyze extracted data
    analyze_extracted_data(train_df, test_df)

    # Drop features
    drop = ['Cabin_Side', 'Cabin_Deck', 'Cabin_Num']
    train_df.drop(columns=drop, inplace=True)
    test_df.drop(columns=drop, inplace=True)

    # Train X | Y
    x = train_df.drop(columns='Transported')
    y = train_df['Transported']

    # Score
    score_model(x, y)

    # Feature selection (report only)
    # feature_importance(x, y, new_model)

    # Train model on the full training set
    model = new_model()
    train_model(model, x, y)

    # Use
    predictions = predict(model, test_df)
    # predictions = expense_based_override(train_df, test_df, predictions)   # didn't help

    # Write predictions
    write_output(out_io, test_df, predictions)
    print("Done!")


def read_input(train_io: TextIO, test_io: TextIO) -> (DataFrame, DataFrame):
    print("Reading input...")
    train_df = pd.read_csv(train_io, index_col='PassengerId')
    test_df = pd.read_csv(test_io, index_col='PassengerId')
    return train_df, test_df


def new_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        random_state=1013,
        silent=True
    )


def extract_features(train_df: DataFrame, test_df: DataFrame) -> (DataFrame, DataFrame):
    print("Extracting features...")

    # Combine labaled training data and unlabeled test data.
    # We will leverage unlabeled test data as in semi-supervised learning.
    # This is fine because Kaggle Spaceship Titanic challenge is a closed problem.
    # We maximize performance on the unlabeled test set by any means - there will be no other test set.
    all_df = pd.concat([train_df, test_df])
    all_df.sort_index(inplace=True)
    all_df = extract_features_df(all_df)

    # Extract the labeled train set
    train_xy_df = all_df[all_df['Transported'].notnull()]
    train_xy_df.sort_index(inplace=True)

    # Extract the unlabeled test set
    test_x_df = all_df[all_df['Transported'].isnull()]
    test_x_df.sort_index(inplace=True)
    test_x_df = test_x_df.drop(columns=['Transported'])

    features_correlation = train_xy_df.corr()
    features_correlation_abs = features_correlation.abs()

    return train_xy_df, test_x_df


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

    df['FoodCourt'].fillna(0, inplace=True)
    df['ShoppingMall'].fillna(0, inplace=True)
    df['RoomService'].fillna(0, inplace=True)
    df['Spa'].fillna(0, inplace=True)
    df['VRDeck'].fillna(0, inplace=True)

    df['ExpensesTransported'] = df['FoodCourt'] + df['ShoppingMall']
    df['ExpensesNotTransported'] = df['VRDeck'] + df['Spa'] + df['RoomService']
    df['ExpensesTotal'] = df['FoodCourt'] + df['ShoppingMall'] + df['VRDeck'] + df['Spa'] + df['RoomService']

    # Expenses to log10
    expense_columns = ['FoodCourt', 'ShoppingMall', 'RoomService', 'Spa', 'VRDeck', 'ExpensesTransported', 'ExpensesNotTransported', 'ExpensesTotal']
    for expense in expense_columns:
         df[expense] = np.log10(df[expense] + 1)
    df.rename(lambda c: c + '_Log10' if c in expense_columns else c, inplace=True, axis='columns')

    # Combining source with destination didn't help
    #
    # df['HomePlanet_Destination'] = df['HomePlanet'].str.cat(df['Destination'], sep='_To_')
    # hot_encoding = pd.get_dummies(df['HomePlanet_Destination'], prefix=None)
    # df = df.join(hot_encoding)
    # df.drop(columns=['HomePlanet', 'Destination'], inplace=True)

    hot_encoding = pd.get_dummies(df['HomePlanet'], prefix='HomePlanet')
    df = df.join(hot_encoding)
    df.drop(columns='HomePlanet', inplace=True)
    # df['HomePlanet'].fillna('N/A', inplace=True)

    hot_encoding = pd.get_dummies(df['Destination'], prefix='Destination')
    df = df.join(hot_encoding)
    df.drop(columns='Destination', inplace=True)
    # df['Destination'].fillna('N/A', inplace=True)

    df = df.join(extract_cabin_parts(df['Cabin']))
    df.drop(columns='Cabin', inplace=True)

    # df['PreviousPassengerTransported'] = df['Transported'].shift(1)
    # df['PreviousPassengerTransported'].ffill(inplace=True)

    # df['NextPassengerTransported'] = df['Transported'].shift(-1)
    # df['NextPassengerTransported'].bfill(inplace=True)

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


def extract_cabin_parts(cabins: Series) -> DataFrame:
    df = DataFrame(index=cabins.index)
    df.index.name = 'PassengerId'

    cabins = cabins.map(lambda value: value.split('/'), na_action='ignore')

    # Deck is categorical (A, B, C, etc)
    categorical = cabins.map(lambda cabin: cabin[0], na_action='ignore')
    hot_encoding = pd.get_dummies(categorical, prefix='Cabin_Deck')
    df = df.join(hot_encoding)
    df['Cabin_Deck'] = cabins.map(lambda cabin: cabin[0], na_action='ignore')

    # Num is integer
    df['Cabin_Num'] = cabins.map(lambda cabin: int(cabin[1]), na_action='ignore')
    df['Cabin_Num'].fillna(df['Cabin_Num'].mean(), inplace=True)
    df['Cabin_Num_Div_300'] = df['Cabin_Num'].floordiv(300).astype(int)

    # Side is binary with NaN, hence OHE
    df['Cabin_Side'] = cabins.map(lambda cabin: cabin[2], na_action='ignore')
    hot_encoding = pd.get_dummies(df['Cabin_Side'], prefix='Cabin_Side')
    df = df.join(hot_encoding)

    return df


def score_model(x: DataFrame, y):
    print_hr()
    print("CV-scoring model...")
    model = new_model()
    result = cross_val_score(model, x, y, cv=5)
    print()
    print(f'  Mean CV score: {result.mean()*100:.2f}%')
    print(f'Stdev CV scores: {result.std()*100:.2f}%')
    print(f'  All CV scores: {[str(round(r*100, 2)) + "%" for r in result]}')
    print(f'Lowest CV score: {min(result*100):.2f}%')


def train_model(model, x, y):
    print("Training...")
    model.fit(x, y)


def predict(model: Pipeline, x_challenges):
    print("Predicting outputs...")
    return model.predict(x_challenges)


def expense_based_override(train_df, test_df: DataFrame, predictions):
    max_transported_vrdeck = train_df.loc[train_df['Transported'] == 1.0]['VRDeck'].max()
    max_transported_spa = train_df.loc[train_df['Transported'] == 1.0]['Spa'].max()
    max_transported_roomservice = train_df.loc[train_df['Transported'] == 1.0]['RoomService'].max()
    max_not_transported_shoppingmall = sorted(train_df.loc[train_df['Transported'] == 0]['ShoppingMall'])[-3]  # the last two are outliers
    max_not_transported_foodcourt = sorted(train_df.loc[train_df['Transported'] == 0]['FoodCourt'])[-3]  # the last two are outliers

    for i, (index, row) in enumerate(test_df.iterrows()):
        if \
                row['VRDeck'] > max_transported_vrdeck or \
                row['Spa'] > max_transported_spa or \
                row['RoomService'] > max_transported_roomservice:
            if predictions[i] == float(1):
                print(f'Overriding 1 -> 0 for index={index}')
                predictions[i] = 0.0
        if \
                row['ShoppingMall'] > max_not_transported_shoppingmall or \
                row['FoodCourt'] > max_not_transported_foodcourt:
            if predictions[i] == float(0):
                print(f'Overriding 0 -> 1 for index={index}')
                predictions[i] = 1.0

    return predictions


def write_output(file: TextIO, x_test: DataFrame, predictions: list[float]):
    print("Saving outputs...")
    file.write('PassengerId,Transported\n')
    for passenger_id, prediction in zip(x_test.index, predictions):
        file.write(f'{passenger_id},{bool(prediction)}\n')


if __name__ == '__main__':
    nice_seaborn()
    nice_pandas()
    with open('input/train.csv') as train, \
         open('input/test.csv') as test,   \
         open('output.csv', 'w') as output:
        main(train, test, output)
