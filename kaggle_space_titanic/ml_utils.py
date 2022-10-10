import pandas as pd
import seaborn as sns


def nice_seaborn():
    sns.set(rc={'figure.dpi': 300, 'savefig.dpi': 300})
    sns.set_theme(
        context='paper',  # small fonts
        style='darkgrid',  # light grey background with grid lines (the default)
        palette='deep',  # low luminance, low saturation (the default)
    )
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})


def nice_pandas():
    pd.options.display.float_format = '{:.8f}'.format


def print_hr():
    print('\n\n' + '='*80 + '\n\n')