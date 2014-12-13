from PyQt4 import QtCore, QtGui
import pyqtgraph as QtGraph


class ProfilePlot(QtGraph.PlotWidget):

    def __init__(self):
        super(ProfilePlot, self).__init__()
        self.plotItem.setRange(xRange=(0, 2))
        self.plotItem.setLabels(bottom='x (m)')
        self.plotItem.hideAxis('left')
        self.dummy = []

    def addSeries(self, data, penColor):
        newCurve = QtGraph.PlotCurveItem(data, pen=penColor)
        newPlot = QtGraph.ViewBox()
        newPlot.addItem(newCurve)
        newPlot.setXLink(self.plotItem)
        newAx = QtGraph.AxisItem('right')
        newAx.setLabel('whoa', color=penColor)
        newAx.linkToView(newPlot)
        newAx.setZValue(-10000)
        self.plotItem.scene().addItem(newPlot)
        self.plotItem.layout.addItem(newAx, 2, len(self.dummy)+3)
        self.dummy.append(newPlot)
        self.updateViews()
        return newCurve

    def updateViews(self):
        viewBox = self.plotItem.vb
        viewRect = viewBox.sceneBoundingRect()
        for plot in self.dummy:
            plot.setGeometry(viewRect)
            plot.linkedViewChanged(viewBox, plot.XAxis)

def userInput(line):
    voltageAtBoundary = line.value()



if __name__ == '__main__':
    app = QtGui.QApplication([])

    t = [0, 1, 2]
    p = [0, 0.05, 0.06]
    o = [1.22, 1.27, 1.32]
    k = [100, 200, 300]
    x = [0.1, 0.11, 0.2]
    i = [10, 20, 15]

    widget = QtGui.QWidget()
    quitAction = QtGui.QAction("Quit", widget)
    quitAction.setShortcut("q")
    quitAction.triggered.connect(exit)
    layout = QtGui.QHBoxLayout(widget)
    widget.addAction(quitAction)
    profilePlot = ProfilePlot()
    protonicPotentialCurve = profilePlot.addSeries(p, 'b')
    reactionRateCurve = profilePlot.addSeries(k, 'r')
    oxygenFractionCurve = profilePlot.addSeries(x, 'g')
    currentDensityCurve = profilePlot.addSeries(i, 'y')
    layout.addWidget(profilePlot)


    polarizationPlot = QtGraph.PlotWidget()
    polarizationPlot.setMouseEnabled(x=False, y=False)
    polarizationPlot.setLimits(xMin=0, yMin=0, minXRange=1, minYRange=1)
    # polarizationPlot.setRange(xRange=(0,2), yRange=(0,2))
    polarizationPlot.setLabels(left='V', bottom='A/cm2')

    i0 = [0.0001, 0.0302, 0.0502, 0.0601, 0.0996, 0.2002, 0.5994, 0.9989, 1.1990, 1.4985, 1.6978]
    v0 = [0.9478, 0.8704, 0.8551, 0.8502, 0.8315, 0.7865, 0.7418, 0.7116, 0.7007, 0.6839, 0.6550]

    polarizationPlot.plot(i0, v0)

    userLine = polarizationPlot.addLine(y=0.8, movable=True)
    userLine.sigPositionChanged.connect(userInput)
    userLine.setBounds([0, 2])
    layout.addWidget(polarizationPlot)
    

    widget.show()

    app.exec_()
