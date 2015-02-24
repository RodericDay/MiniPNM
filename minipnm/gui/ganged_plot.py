import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=8)

class GangedPlot(dict):
    '''
    fundamentally a dictionary, whose entries are the
    corresponding profile plot axes
    '''
    def __init__(self, model):
        # link data
        self.model = model

        # matpotlib init
        self.fig = plt.figure()
        self.fig.subplots_adjust(hspace=0.001)


        # profile plots
        n = len(model.properties)-1
        for i, ylabel in enumerate(model.properties[1:], 1):
            ax = self[ylabel] = self.fig.add_subplot(n, 2, 2*i-1)
            plt.setp( ax.get_xticklabels(), visible=False )
            ax.set_ylabel(ylabel, rotation=0)
            ax.yaxis.set_label_coords(0.5, 0.76)
            ax.set_yticks([])
        ax.set_xlabel(model.properties[0])
        plt.setp( ax.get_xticklabels(), visible=True )

        # polarization curve
        self.pax = self.fig.add_subplot(1, 2, 2)
        self.pax.yaxis.tick_right()
        self.pax.yaxis.set_label_position("right")
        self.pax.set_xlabel('geometric current density (A/cm**2)')
        self.pax.set_ylabel('voltage at gdl (V)')
        self.fig.canvas.mpl_connect('button_press_event',
            lambda event: self.click(event) if self.pax is event.inaxes else self.reframe(event.inaxes))
        self.fig.canvas.mpl_connect('motion_notify_event',
            lambda event: self.click(event) if self.pax is event.inaxes and event.button is 1 else None)
        self.pax.plot(*model.polarization_curve())
        plt.show()

    def reframe(self, ax):
        if ax:
            print( 'autoscaled {}'.format(ax) )

    def update(self, state, c='k'):
        '''
        state must be a dictionary object containing relevant data
        '''
        first_key = next(iter(state))
        x = state[first_key]
        for ylabel, ax in self.items():
            data = state[ylabel]
            ax.lines = [] # clears plot, let gc delete objects
            ax.set_xlim(x.min(), x.max())
            ax.plot(x, data, c)

            if data.ptp():
                ymin, ymax = data.min(), data.max()
            else:
                ymean = data.mean()
                ymin, ymax = ymean-0.5, ymean+0.5

            # yminold, ymaxold = ax.get_ylim()
            # ymin = min(yminold, ymin)
            # ymax = max(ymaxold, ymax)

            ax.set_ylim(ymin, ymax)

            # fix yticks
            yticks = np.linspace(ymin, ymax, 7)[1:-1]
            ax.set_yticks(yticks)

    def click(self, event):
        variable = event.ydata
        state = self.model.resolve(variable)
        self.update(state)
        self.fig.canvas.draw()

        # for ax in self.values():
        #     ax.lines = []
        # try:
        # except Exception as error:
        #     # plot all error states
        #     try:
        #         for state in error:
        #             self.update(state, 'r')
        #     except TypeError:
        #         raise error

if __name__ == '__main__':
    import minipnm as mini

    model = mini.models.Model()
    model.protonic_conductivity *= 1E-5
    plot = GangedPlot(model)
