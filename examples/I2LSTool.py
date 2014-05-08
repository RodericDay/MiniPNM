from __future__ import print_function
import os
import numpy as np
import minipnm as mini

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
plt.rcParams['image.cmap'] = 'coolwarm'

class I2LSTool(object):
    fig, ax = plt.subplots(1)
    annotation = ax.annotate("LOL", (10,10))
    source = plt.Rectangle((0,0),10,10, color='red', alpha=0.8)
    ax.add_patch(source)
    sink = plt.Rectangle((30,3),10,10, color='blue', alpha=0.8)
    ax.add_patch(sink)
    index = 0
    sol = None

    def __init__(self):
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.reset)
        self.mid = self.fig.canvas.mpl_connect('motion_notify_event', self.resize)
        self.vid = self.fig.canvas.mpl_connect('motion_notify_event', self.measure)
        self.rid = self.fig.canvas.mpl_connect('key_press_event', self.handlekey)

    def load_file(self, filepath):
        im = mini.imread(filepath, dmax=400)
        filename = os.path.basename(filepath)
        network = mini.Cubic(im, im.shape)
        network = network - (im > im.mean())
        self.network = network

    def load_dir(self, dirpath):
        R = mini.gaussian_noise([100,100,10])
        network = mini.Cubic(R, R.shape)
        self.network = network - (R>R.mean())
        
        # add slider
        self.fig.subplots_adjust(bottom=0.2)
        self.axsld = self.fig.add_axes([0.2,0.05,0.6,0.05])
        self.slider = Slider(self.axsld, 'Depth', valmin=0, valinit=0, closedmax=False,
                             valmax=self.network.resolution[2], valfmt='%i')
        def handle_slider_change(value):
            self.index = int(value)
            self.update()
        self.slider.on_changed(handle_slider_change)

        self.wid = self.fig.canvas.mpl_connect('scroll_event',
            lambda event: self.slider.set_val(np.clip(self.slider.val+event.step,0,9)))

    def update(self, *garbage, **more_garbage):
        img = self.network.asarray(self.sol)[:,:,self.index].T
        self.axim.set_data(img) 
        self.fig.canvas.draw()

    def run(self):
        try:
            self.axim = self.ax.matshow(self.network.asarray()[:,:,0].T)
            plt.show()
        except AttributeError:
            print( "Load a file/dir!" )

    def autotarget(fn):

        def wrapped(self, event):
            try:
                if event.inaxes is self.axsld:
                    return
            except AttributeError as err:
                pass

            if event.button == 1:
                target = self.source
            elif event.button == 3:
                target = self.sink
            else:
                return
            fn(self, event, target)

        return wrapped

    def handlekey(self, event):
        {
            'enter':self.simulate,
            'escape':self.unsolve
        }.get(event.key, lambda event: '')(event)

    @autotarget
    def reset(self, event, target):
        x,y = self.ax.transData.inverted().transform((event.x, event.y))
        target.set_bounds(x,y,0,0)
        self.fig.canvas.draw()

    @autotarget
    def resize(self, event, target):
        x0,y0 = target.get_xy()
        x,y = self.ax.transData.inverted().transform((event.x, event.y))
        w,h = x-x0, y-y0
        target.set_bounds(x0,y0,w,h)
        self.fig.canvas.draw()

    def simulate(self, event):
        x,y,z = self.network.coords
        center = lambda bbox: np.array([bbox.xmin+bbox.width/2, bbox.ymin+bbox.height/2])
        masker = lambda bbox: (x>bbox.xmin)&(x<bbox.xmax)&(y>bbox.ymin)&(y<bbox.ymax)
        source_bbox = self.source.get_bbox()
        sink_bbox = self.sink.get_bbox()
        print("Apply DBC differential of 2u")
        ics = 3*masker(source_bbox) + 1*masker(sink_bbox)
        self.axim.set_clim(vmin=0, vmax=3)
        sol = mini.linear_solve(self.network, ics)
        flux_source = self.network.flux(sol, masker(source_bbox)).sum()
        flux_sink = self.network.flux(sol, masker(sink_bbox)).sum()
        distance = np.linalg.norm(center(source_bbox)-center(sink_bbox))
        try:
            assert np.allclose(flux_source, flux_sink)
            print("Flux at Source/Sink: ", flux_source)
        except AssertionError:
            print("WARNING: Flux divergence!")
            print("Flux at Source: ", flux_source)
            print("Flux at Sink: ", flux_sink)
        finally:
            print("Distance between centroids: ", distance)
            print("Multi", flux_source * distance)

        self.sol = sol
        self.update()

    def unsolve(self, event):
        self.sol = None
        self.axim.set_clim(vmin=0, vmax=self.network.asarray().max())
        self.update()

    def measure(self, event):
        if self.sol is None:
            return

        xm, ym = self.ax.transData.inverted().transform((event.x, event.y))
        x,y,z = self.network.coords
        nearest_vertex = np.abs([x-xm, y-ym]).sum(axis=0).argmin()
        self.annotation.set_text("{:.5f}V".format(self.sol[nearest_vertex]))
        self.annotation.xytext = (xm,ym)
        self.update


if __name__ == '__main__':
    sample = 'X 1050-C'
    dirpath = os.path.abspath(sample)
    photos = list(sorted(fn for fn in os.listdir(sample) if '.tif' in fn))
    filepath = os.path.join(dirpath, photos[0])

    tool = I2LSTool()
    # tool.load_file(filepath)
    tool.load_dir(dirpath)
    tool.run()