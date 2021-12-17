# Example code for creating a figure of suitable size
# for inclusion in the term paper.

import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['text.usetex'] = True

def new_figure(height=55):
    "Return figure with width 84mm and given height in mm."

    return plt.figure(figsize=(84/25.4, 55/25.4))

x = range(10)
y = [v**2 - 1 for v in x]

fig = new_figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, 'o-')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
fig.savefig('sample_plot.pdf', bbox_inches='tight')
