"""Printing functions for slps"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from time import gmtime
import datetime

X_train = np.load('X_train_surge_new.npz')
X_test = np.load('X_test_surge_new.npz')

fps = 10
slp_train = X_train['slp']
slp_test = X_test['slp']

pressure_average = 1013
pressure_range = 30
vmin = pressure_average - pressure_range
vmax = pressure_average + pressure_range
cmap = mpl.cm.coolwarm

def time_formatted(t):
    t_dt = datetime.datetime.fromtimestamp(t)
    return t_dt.strftime('%Y-%m-%d, %H:%M:%S')

def movie_slp(i):
    slp_i = None
    name = 'slp_evolution_'+str(i)+'.mov'
    L = len(slp_train)
    date = ''
    if i <= L:
        slp_i = slp_train[i-1]
        date = time_formatted(X_train['t_slp'][i-1][0])
    else:
        slp_i = slp_test[i-1-L]
        date = time_formatted(X_test['t_slp'][i-1-L][0])
    slp_i = slp_i/100
    fig, ax = plt.subplots()
    ax.set_title('Time evolution of sea-level pressure (hPa) n°'+str(i)+'\n starting '+date)
    img = ax.imshow(slp_i[0], vmin=vmin, vmax=vmax, cmap=cmap)
    hc = plt.colorbar(img)
    ims = [[ax.imshow(slp_i[j], vmin=vmin, vmax=vmax, cmap=cmap, animated=True)] for j in range(1, len(slp_i))]
    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)

    ani.save(name)

movie_slp(5606)
