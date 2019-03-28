import numpy as np
import pandas as pd
from itertools import permutations
import copy
from matplotlib import pyplot as plt
import seaborn as sns






if __name__ == '__main__':
    """
    Plots the max_deformations for three different lego parts over all 77 experiments.
    """

    data_path = "../data/lego_data.csv"
    data_dict = {}
    parameters = np.array(['holding_pressure',
                           'holding_pressure_time',
                           'melt_temp',
                           'mold_temp',
                           'cooling_time',
                           'volume_flow'])
    data = pd.read_csv(data_path, sep=';')

    sns.set()
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.set_xlabel('Experiments', fontsize=35)
    ax.set_ylabel('Deformation', fontsize=35)
    colors = ['steelblue', 'g', 'darkorange']
    i = 0
    for part, df_part in data.groupby('lego'):
        if part in ['3x1_Lego', '4x2_Lego', '6x1_Lego']:
            x_axis = len(df_part['max_deformation'])
            x_axis = np.arange(0, x_axis)
            print(df_part)
            df_part_sorted = df_part.sort_values(by=list(parameters))
            df_part_sorted.reset_index()
            plt.plot(x_axis, df_part_sorted['max_deformation'], label=part, linewidth=5, color=colors[i])
            i += 1
    plt.legend(fontsize=25)
    plt.show()
