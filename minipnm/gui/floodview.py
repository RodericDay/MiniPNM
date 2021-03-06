import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


water_cmap = matplotlib.cm.get_cmap('Blues')
water_cmap.set_over('b')
water_cmap.set_under(alpha=0) # evaporated water is transparent
water_cmap.set_bad('w') # frozen water (np.nan) is white


def floodview(water_history, state_history, topology=None):
    '''
    create two views to see the evolution of some data
    representing a state simultaneously with water evolution

    water array should be [0, 1, NaN] for g, l, s
    the state array can be anything
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1)

    if topology is None:
        topology = np.zeros_like(water_history[0])
    # draw a nice background to see water channeling
    ax1.imshow(topology, interpolation='nearest', cmap='copper')
    # draw water on top
    water_layer = ax1.imshow(water_history[0],
        vmin=0.5, vmax=0.51 ,cmap=water_cmap)
    # consider that linear solutions of isolated clusters
    # default to zero.
    # separate and make optional the setting of vmin and vmax
    state_layer = ax2.imshow(state_history[0],
        vmin=np.min(state_history),
        vmax=np.max(state_history),
        cmap='coolwarm')
    fig.colorbar(state_layer)

    slider = Slider(
        ax=fig.add_axes([0.2,0.01,0.6,0.03]),
        label='array_index',
        valmin=0,
        valmax=len(water_history)-1)

    def update(f):
        i = int(round(f))
        water_layer.set_data( water_history[i] )
        state_layer.set_data( state_history[i] )
        fig.canvas.draw()

    slider.on_changed(update)

    plt.show()


if __name__ == '__main__':
    # a topology may be a 2D array
    G = np.random.rand(30,30)
    # histories and states would be 3D stacks
    W = np.random.rand(100,30,30)
    W[W<0.1] = np.nan
    S = np.random.rand(100,30,30) + np.linspace(0,2,100).reshape(100,1,1)
    floodview(W, S, G)
