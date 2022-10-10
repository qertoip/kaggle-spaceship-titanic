from pathlib import Path

from pandas import DataFrame, Series
from matplotlib import pyplot as plt
import seaborn as sns


def analyze_raw_data(train_df: DataFrame, test_df: DataFrame):
    print('Charting raw data...')
    transported_by_age(train_df)
    transported_by_cryosleep(train_df)
    transported_by_vip(train_df)
    transported_by_homeplanet(train_df)
    transported_by_destination(train_df)

    transported_by_vrdeck(train_df)
    transported_by_spa(train_df)
    transported_by_roomservice(train_df)
    transported_by_foodcourt(train_df)
    transported_by_shoppingmall(train_df)

    expenses_boxplots(train_df)

    # from autoviz.AutoViz_Class import AutoViz_Class
    # AV = AutoViz_Class()
    # AV.AutoViz(
    #     filename='input/train.csv',
    #     depVar="Transported",
    #     dfte=train_df,
    #     chart_format="html",
    #     save_plot_dir='charts_raw_data'
    # )


def analyze_extracted_data(train_df: DataFrame, test_df: DataFrame):
    print('Charting extracted data...')
    transported_by_vrdeck_log10(train_df)
    transported_by_spa_log10(train_df)
    transported_by_roomservice_log10(train_df)
    transported_by_foodcourt_log10(train_df)
    transported_by_shoppingmall_log10(train_df)

    transported_by_cabin_deck(train_df)
    transported_by_cabin_num(train_df)
    transported_by_cabin_side(train_df)
    # transported_by_home_dest(train_df)
    transported_by_passenger_alone(train_df)
    transported_by_group_div_300(train_df)


def transported_by_age(train_df):
    chart = sns.histplot(data=train_df, x='Age', binwidth=1, hue='Transported')
    chart.set(title=f'Transported by Age')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_cryosleep(train_df):
    chart = sns.countplot(data=train_df, x='CryoSleep', hue='Transported')
    chart.set(title=f'Transported by CryoSleep count')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_vip(train_df):
    chart = sns.countplot(data=train_df, x='VIP', hue='Transported')
    chart.set(title=f'Transported by VIP count')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_homeplanet(train_df):
    chart = sns.countplot(data=train_df, x='HomePlanet', hue='Transported')
    chart.set(title=f'Transported by HomePlanet count')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_destination(train_df):
    chart = sns.countplot(data=train_df, x='Destination', hue='Transported')
    chart.set(title=f'Transported by Destination count')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_vrdeck(train_df):
    chart = sns.scatterplot(data=train_df, x='VRDeck', y='Transported')
    chart.set(title=f'Transported by VRDeck scatter')
    save_and_clear(chart, 'charts_raw_data')

    chart = sns.histplot(data=train_df, x='VRDeck', bins=50, hue='Transported')
    chart.set(title=f'Transported by VRDeck distribution')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_spa(train_df):
    chart = sns.scatterplot(data=train_df, x='Spa', y='Transported')
    chart.set(title=f'Transported by Spa scatter')
    save_and_clear(chart, 'charts_raw_data')

    chart = sns.histplot(data=train_df, x='Spa', bins=50, hue='Transported')
    chart.set(title=f'Transported by Spa distribution')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_roomservice(train_df):
    chart = sns.scatterplot(data=train_df, x='RoomService', y='Transported')
    chart.set(title=f'Transported by RoomService scatter')
    save_and_clear(chart, 'charts_raw_data')

    chart = sns.histplot(data=train_df, x='RoomService', bins=50, hue='Transported')
    chart.set(title=f'Transported by RoomService distribution')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_foodcourt(train_df):
    chart = sns.scatterplot(data=train_df, x='FoodCourt', y='Transported')
    chart.set(title=f'Transported by FoodCourt scatter')
    save_and_clear(chart, 'charts_raw_data')

    chart = sns.histplot(data=train_df, x='FoodCourt', bins=50, hue='Transported')
    chart.set(title=f'Transported by FoodCourt distribution')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_shoppingmall(train_df):
    chart = sns.scatterplot(data=train_df, x='ShoppingMall', y='Transported')
    chart.set(title=f'Transported by ShoppingMall scatter')
    save_and_clear(chart, 'charts_raw_data')

    chart = sns.histplot(data=train_df, x='ShoppingMall', bins=50, hue='Transported')
    chart.set(title=f'Transported by ShoppingMall distribution')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_passenger_alone(train_df):
    chart = sns.countplot(data=train_df, x='PassengerId_Alone', hue='Transported')
    chart.set(title=f'Transported by PassengerId_Alone count')
    save_and_clear(chart, 'charts_raw_data')


def transported_by_group_div_300(train_df):
    chart = sns.countplot(data=train_df, x='Cabin_Num_Div_300', hue='Transported')
    chart.set(title=f'Transported by Cabin_Num_Div_300 count')
    save_and_clear(chart, 'charts_raw_data')


def expenses_boxplots(train_df):
    for label in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        boxplot(label, train_df)


def transported_by_vrdeck_log10(train_df):
    chart = sns.scatterplot(data=train_df, x='VRDeck_Log10', y='Transported')
    chart.set(title=f'Transported by VRDeck_Log10 scatter')
    save_and_clear(chart, 'charts_extracted_data')

    chart = sns.histplot(data=train_df, x='VRDeck_Log10', bins=50, hue='Transported')
    chart.set(title=f'Transported by VRDeck_Log10 distribution')
    save_and_clear(chart, 'charts_extracted_data')


def transported_by_spa_log10(train_df):
    chart = sns.scatterplot(data=train_df, x='Spa_Log10', y='Transported')
    chart.set(title=f'Transported by Spa_Log10 scatter')
    save_and_clear(chart, 'charts_extracted_data')

    chart = sns.histplot(data=train_df, x='Spa_Log10', bins=50, hue='Transported')
    chart.set(title=f'Transported by Spa_Log10 distribution')
    save_and_clear(chart, 'charts_extracted_data')


def transported_by_roomservice_log10(train_df):
    chart = sns.scatterplot(data=train_df, x='RoomService_Log10', y='Transported')
    chart.set(title=f'Transported by RoomService_Log10 scatter')
    save_and_clear(chart, 'charts_extracted_data')

    chart = sns.histplot(data=train_df, x='RoomService_Log10', bins=50, hue='Transported')
    chart.set(title=f'Transported by RoomService_Log10 distribution')
    save_and_clear(chart, 'charts_extracted_data')


def transported_by_foodcourt_log10(train_df):
    chart = sns.scatterplot(data=train_df, x='FoodCourt_Log10', y='Transported')
    chart.set(title=f'Transported by FoodCourt_Log10 scatter')
    save_and_clear(chart, 'charts_extracted_data')

    chart = sns.histplot(data=train_df, x='FoodCourt_Log10', bins=50, hue='Transported')
    chart.set(title=f'Transported by FoodCourt_Log10 distribution')
    save_and_clear(chart, 'charts_extracted_data')


def transported_by_shoppingmall_log10(train_df):
    chart = sns.scatterplot(data=train_df, x='ShoppingMall_Log10', y='Transported')
    chart.set(title=f'Transported by ShoppingMall_Log10 scatter')
    save_and_clear(chart, 'charts_extracted_data')

    chart = sns.histplot(data=train_df, x='ShoppingMall_Log10', bins=50, hue='Transported')
    chart.set(title=f'Transported by ShoppingMall_Log10 distribution')
    save_and_clear(chart, 'charts_extracted_data')


def transported_by_cabin_deck(train_df):
    chart = sns.countplot(data=train_df, x='Cabin_Deck', hue='Transported')
    chart.set(title=f'Transported by Cabin_Deck count')
    save_and_clear(chart, 'charts_extracted_data')


def transported_by_cabin_num(train_df):
    chart = sns.scatterplot(data=train_df, x='Cabin_Num', y='Transported')
    chart.set(title=f'Transported by Cabin_Num scatter')
    save_and_clear(chart, 'charts_extracted_data')

    chart = sns.histplot(data=train_df, x='Cabin_Num', binwidth=25, hue='Transported')
    chart.set(title=f'Transported by Cabin_Num distribution')
    save_and_clear(chart, 'charts_extracted_data')


def transported_by_cabin_side(train_df):
    chart = sns.countplot(data=train_df, x='Cabin_Side', hue='Transported')
    chart.set(title=f'Transported by Cabin_Side count')
    save_and_clear(chart, 'charts_extracted_data')


def transported_by_home_dest(train_df):
    chart = sns.countplot(data=train_df, x='HomePlanet_Destination', hue='Transported')
    chart.set(title=f'Transported by HomePlanet_Destination count')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    save_and_clear(chart, 'charts_extracted_data')


def boxplot(label, df):
    chart = sns.boxplot(x=df[label])
    chart.set(title=f'BoxPlot for {label}')
    save_and_clear(chart, 'charts_raw_data')


def save_and_clear(chart, dirpath):
    filename = chart.get_title().lower().replace(' ', '_') + '.png'
    filepath = Path(dirpath) / filename
    chart.get_figure().savefig(filepath)
    chart.clear()
    plt.cla()
    plt.clf()
