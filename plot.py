"""
This module plot the regret of different algorithms
"""
# http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
import matplotlib as mpl
mpl.use("pdf")

import numpy as np
import os
import errno

# https://gist.github.com/alexrudy/a7982903a2fb2ab0dde3
# http://blog.olgabotvinnik.com/blog/2012/11/15/2012-11-15-how-to-set-helvetica-as-the-default-sans-serif-font-in/
pgf_with_rc_fonts = {
    'text.usetex': False,  # Change to True to use latex
    "font.family": "sans-serif",
    "pgf.texsystem": "lualatex",
    "font.serif": [],  # use latex default serif font
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],  # use a specific sans-serif font
}
mpl.rcParams.update(pgf_with_rc_fonts)

import matplotlib.pyplot as plt

agents_style = [
    'dotted',
    'solid',
    'dashed',
    'dashdot',
]

agents_marker = [
    '8',
    '>',
    'd',
    '*',
    's',
    '1']

agents_linewidth = [
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    6.0]

agents_color = [
    'red',
    'blue',
    'green',
    'magenta',
    'aqua',
    'c',
    'blueviolet'

]

# http://tableaufriction.blogspot.se/2012/11/finally-you-can-use-tableau-data-colors.html
# http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

tableau10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
             (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

for i in range(len(tableau10)):
    r, g, b = tableau10[i]
    tableau10[i] = (r / 255., g / 255., b / 255.)


def get_style(agent_id):
    if agent_id < len(agents_style):
        return [agents_style[agent_id], 'None', agents_linewidth[agent_id], tableau10[agent_id]]
    else:
        return ['-', agents_marker[agent_id - len(agents_style)], agents_linewidth[agent_id], tableau10[agent_id]]


def mkdir_p(path):
    """
    Create a directory if not exists
    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def plot_regret(regrets, names, deviations, figure_name):
    """
    Plot the regret
    :param regrets:
    :param names:
    :param deviations:
    :param figure_name:
    :return:
    """
    num_figures = len(regrets)

    horizon = len(regrets[0])
    x = np.arange(1, horizon + 1)

    plt.figure(1)
    ax = plt.gca()
    ax.set_ylabel(r'Individual Rational Regret', fontsize=16)
    plt.xlabel(r'Number of rounds', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, linestyle='dotted')

    # https://stackoverflow.com/questions/48213884/transparent-error-bars-without-affecting-markers
    for i in range(num_figures):
        markers, caps, bars = plt.errorbar(x, regrets[i], yerr=deviations[i], label=names[i], linestyle=get_style(i)[0],
                                           marker=get_style(i)[1], color=get_style(i)[3], antialiased=True,
                                           errorevery=int(np.sqrt(horizon)))
        # loop through bars and caps and set the alpha value
        [(bar.set_alpha(0.1), bar.set_aa(True)) for bar in bars]
        [(cap.set_alpha(0.1), cap.set_aa(True)) for cap in caps]

    plt.figure(1)
    legend = plt.legend(loc='best', fancybox=True, framealpha=0.5)
    # Remove the box for legend and add background
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    plt.axis('tight')
    ax = plt.gca()
    plt.xticks(list(plt.xticks()[0]) + [1])
    ax.set_xlim(left=1, right=horizon)

    mkdir_p('figures')
    plt.savefig(os.path.join('figures', str(figure_name) + '.png'))
    plt.close()
