import sys
from PyQt4 import QtCore, QtGui
import pyqtgraph as QtGraph

import minipnm as mini
import MKS
MKS.define(globals())

class Walker(QtGui.QMainWindow):\

    def __init__(self):
        super(Walker, self).__init__()
        self.tb = self.addToolBar("yo")
        self.tb.addAction('>', self.march).setShortcut('Return')
        self.tb.addAction('x', self.close).setShortcut('Q')

        self.gl = QtGraph.GraphicsLayoutWidget()
        self.setCentralWidget(self.gl)

        self.initCatalyst()
        self.initPlots()

    def initCatalyst(self):
        shape = [100, 1, 1]
        t = 0.725 * um # thickness
        P = 1.5E5 * Pa

        self.cl = mini.models.SimpleLatticedCatalystLayer(shape, t, 0.422)
        self.cl.generate_agglomerate(200 * m**2/m**2, 10 * nm)
        self.cl.generate_systems(P)
        self.cl.solve_systems( 0 * A/m**2 )
        self.x = self.cl.depth(um)

    def initPlots(self):
        self.gl.addPlot(row=0, col=0, pen='b').setYRange(-0.1, 2)
        self.gl.addPlot(row=1, col=0, pen='g').setYRange(0, 1)
        self.gl.addPlot(row=2, col=0, pen='r').setYRange(200, 500)
        self.gl.addPlot(row=3, col=0, pen='c')
        
        self.polcurve = self.gl.addPlot(row=0, col=1, rowspan=4)
        self.polcurve.setYRange(0, 1.2)
        self.polcurve.setMouseEnabled(False, False)

        # Rob's data
        i0 = [0.0001, 0.0302, 0.0502, 0.0601, 0.0996, 0.2002, 0.5994, 0.9989, 1.1990, 1.4985, 1.6978]
        v0 = [0.9478, 0.8704, 0.8551, 0.8502, 0.8315, 0.7865, 0.7418, 0.7116, 0.7007, 0.6839, 0.6550]
        self.polcurve.plot(i0, v0)

        self.variableLine = self.polcurve.addLine(y=0.75, movable=True)
        self.measureLine = self.polcurve.addLine(x=0, movable=False)
        self.variableLine.sigPositionChangeFinished.connect(self.reset)

    def reset(self, line):
        self.cl.electron_transport.bvals[:] = line.value()

    def march(self):
        last_i = self.cl.orr
        new_i = self.cl.solve_systems(last_i)
        self.setPlot(0, self.cl.electronic_potential(V),
                        self.cl.protonic_potential(V),
                        self.cl.overpotential(V), )
        self.setPlot(1, self.cl.oxygen_molar_fraction )
        self.setPlot(2, self.cl.temperature(K) )
        self.setPlot(3, last_i(A/m**2),
                        new_i(A/m**2), )

        self.measureLine.setValue( self.cl.measured_current_density(new_i)(A/cm**2) )

    def setPlot(self, r, *ys):
        plot = self.gl.getItem(r, 0)
        plot.hideAxis('left')
        for i, y in enumerate(ys):
            try:
                target = plot.listDataItems()[i]
            except IndexError:
                target = plot.plot()
            finally:
                target.setData(self.x, y)

app = QtGui.QApplication([])

walker = Walker()
walker.show()

app.exec_()
