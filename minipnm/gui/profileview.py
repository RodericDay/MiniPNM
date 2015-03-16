import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

'''
this is an interactive tool to explore the profile views
of a given model
'''

def insert_point(point, line):
    '''
    this method takes in a line and a point, and
    updates the line so that the point is organized
    along with the rest

    call ax.relim() and ax.autoscale_view() to ensure
    that additions are visible

    warning: modifies line data!
    '''
    x, y = point
    xs, ys = [list(arr) for arr in line.get_data()]
    xs.append(x)
    ys.append(y)
    xs, ys = zip( *((xs[i], ys[i]) for i in np.argsort(xs)) )
    line.set_data(xs, ys)

def update_ax(ax, x, y):
    '''
    this method averages out points according to x
    '''
    unique_x = np.unique(x)
    average_y = [y[x==xi].mean() for xi in unique_x]
    ax.plot(unique_x, average_y, 'k')

def profileview(model):
    plt.rc('font', size=8)

    fig, axs = plt.subplots(3, sharex=True)
    axs[0].set_title('protonic potential (V)')
    axs[1].set_title('oxygen molar fraction')
    axs[2].set_title('current density (A/m2)')
    axs[2].set_xlabel('distance from membrane (um)')
    fig.subplots_adjust(right=0.45)

    # the mask is assumed to account for boundary values
    mask = ~(model.gdl | model.membrane)
    # we denote the distance from the membrane wall x
    x = model.distance_from_membrane[mask] * 1E6
    # the polarization curve is permanent and unique
    pcax = fig.add_axes([.5, .2, .4, .7])
    pcax.set_title('polarization curve')
    pcax.yaxis.tick_right()
    pcax.yaxis.set_label_position("right")
    pcax.set_xlabel('geometric current density (A/cm**2)')
    pcax.set_ylabel('voltage at gdl (V)')
    polcurve, = pcax.plot([], [], 'ko-')

    def update(V):
        # try networks generate l==1
        model.resolve(V, 263, flood=False)

        i = model.current_history[-1][mask]
        I = i.sum() / model.face_area / 1000
        reading = (I, V)

        insert_point(reading, polcurve)
        pcax.relim()
        pcax.autoscale_view()

        for ax, yh in zip(axs,
            [   model.proton_history,
                model.oxygen_history,
                model.current_history,
            ]):
            y = yh[-1]
            update_ax(ax, x, y)
        fig.canvas.draw()

    slider = Slider(label='V',
        ax=fig.add_axes([.5, .05, .4, .03]),
        valmin=0,
        valmax=1.2,
    )

    slider.on_changed(update)
    # generate an undersampled polcurve
    for V in np.linspace(0.05, 1.0, 5):
        slider.set_val(V)
    slider.set_val(0.65)

    # some extra interactivity
    fig.canvas.mpl_connect('button_press_event',
        lambda event: slider.set_val(event.ydata) if pcax is event.inaxes else None)
    plt.show()


if __name__ == '__main__':
    import minipnm as mini
    G = np.random.uniform(400E-9, 900E-9, [20, 10])
    model = mini.models.ArrayModel(G, 2 * 1000E-9)
    profileview(model)
