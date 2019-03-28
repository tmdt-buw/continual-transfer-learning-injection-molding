from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns


if __name__ == '__main__':
    """
    Loads and plots the evaluation results of continual transfer learning in comparison with baseline performances. 
    2 Plots: final model performances for the lego parts and required data fractions
    """
    fracs = []
    scores = []

    for filename in os.listdir('../results'):
        if filename[0:26] == 'finetune_retune_fracs_perm':
            fracs.append(np.loadtxt('../results/{}'.format(filename)))
        elif filename[0:33] == 'finetune_retune_scores_final_perm':
            values = np.loadtxt('../results/{}'.format(filename))
            scores.append(values)

    # Results from baseline experiments
    baseline2_scores_mean = np.array([0.95659248, 0.92964662, 0.85767489, 0.79422912, 0.82774922 ,0.78185565])
    baseline2_scores_std = np.array([0.04333446, 0.0595248, 0.09236332, 0.12773539, 0.10335722, 0.11119036])
    control_fracs_mean = np.array([0.71, 0.65, 0.76, 0.74, 0.67, 0.6900000000000001])
    control_fracs_std = np.array([0.16248809496813374, 0.1202271554554524, 0.10396078054371142, 0.09966629547095767, 0.1830194339616981, 0.13])


    sns.set()
    fig = plt.figure(figsize=(20,8))

    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)
    ax = fig.add_subplot(121)
    x_axis = ['Part 1', 'Part 2', 'Part 3', 'Part 4', 'Part 5', 'Part 6']

    ax.set_ylabel('Performance ($R^2$ Score)', fontsize=30)
    ax.yaxis.set_tick_params(labelsize=28)
    ax.xaxis.set_tick_params(labelsize=28)
    ax.plot(x_axis, mean_scores, marker='o', linewidth=4, markersize=15, color='darkorange', label='Continual Transfer')
    ax.fill_between(x_axis, mean_scores + std_scores, mean_scores - std_scores, alpha=0.2, color='darkorange')

    ax.plot(x_axis, baseline2_scores_mean, marker='o', linewidth=4, markersize=15, color='g', label='Baseline')
    ax.fill_between(x_axis, baseline2_scores_mean + baseline2_scores_std, baseline2_scores_mean - baseline2_scores_std, alpha=0.2, color='g')
    ax.legend(fontsize=20, loc=3)



    mean_fracs = np.mean(fracs, axis=0)
    std_fracs = np.std(fracs, axis=0)
    ax2 = fig.add_subplot(122)
    x_axis = ['Part 1', 'Part 2', 'Part 3', 'Part 4', 'Part 5', 'Part 6']

    ax2.set_ylabel('Proportion', fontsize=30)
    ax2.yaxis.set_tick_params(labelsize=28)
    ax2.xaxis.set_tick_params(labelsize=28)
    ax2.plot(x_axis, mean_fracs, marker='o', markersize=15, linewidth=4, color='darkorange', label='Continual Transfer ')
    ax2.fill_between(x_axis, mean_fracs + std_fracs / 2, mean_fracs - std_fracs / 2, alpha=0.2, color='darkorange')

    ax2.plot(x_axis, control_fracs_mean, marker='o', markersize=15, linewidth=4, color='steelblue', label='Control')
    ax2.fill_between(x_axis, control_fracs_mean + control_fracs_std / 2, control_fracs_mean - control_fracs_std / 2, alpha=0.2, color='steelblue')
    ax2.legend(fontsize=20)

    plt.tight_layout()
    plt.show()

