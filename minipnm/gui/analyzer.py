import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
plt.rc('font', size=8)

class GUI():

    def __init__(self, properties):
        # profiles
        n = len(properties)
        self.fig, self.axs = plt.subplots(n, figsize=(10, 10), sharex=True)
        for ax, label in zip(self.axs.flat, properties):
            ax.set_title(label)
        tax = self.fig.add_axes([0.15,0.05,0.2,0.01])
        sld = Slider(ax=tax, label='time (s)', valmin=0, valmax=10, valinit=5)
        sld.on_changed(print)
        plt.subplots_adjust(hspace=0.3, top=0.5, right=0.4)

        # polcurve
        self.pax = self.fig.add_axes([0.3,0.55,0.4,0.4])
        self.pax.axhline(0, color='r')

        # cross-section
        self.cax = self.fig.add_axes([0.5, 0.1, 0.4, 0.4])
        im = self.cax.imshow(np.random.rand(20,20))

        zax = self.fig.add_axes([0.55,0.05,0.3,0.01])
        sld = Slider(ax=zax, label='z', valmin=0, valmax=1, valinit=0)
        sld.on_changed(print)

        # some connections
        self.fig.canvas.mpl_connect('key_press_event',
            lambda event: exit() if event.key=='q' else None)
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('motion_notify_event',
            lambda event: self.click(event) if event.button==1 else None)

        # console?
        self.console = self.fig.text(0.01, 0.01, '')

    def click(self, event):
        if event.inaxes is self.pax:
            marker_line = self.pax.lines[0]
            marker_line.set_ydata([event.ydata, event.ydata])
            # sample!
        elif event.inaxes in self.axs:
            self.cax.set_title( event.inaxes.get_title() )
        self.log( event.inaxes )

    def log(self, thing):
        self.console.set_text(repr(thing))
        self.fig.canvas.draw()

    def show(self):
        plt.show()


properties = ['a', 'b', 'c']
GUI(properties).show()
