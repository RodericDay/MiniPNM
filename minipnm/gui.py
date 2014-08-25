import io, bisect

import numpy as np
from scipy import misc

try:
    from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from .graphics import Scene

    from PyQt4 import QtGui, QtCore
    import pyqtgraph as QtGraph
except ImportError:
    QtGui = type("vtk, PyQt, or pyqtgraph missing. functionality unavailable!",
                 (), {"QWidget": object, "QMainWindow": object})


class Canvas(QtGui.QWidget):

    def __init__(self):
        super(Canvas, self).__init__()
        self.vbox = QtGui.QVBoxLayout(self)
        self.toolBar = QtGui.QToolBar(self)
        self.vbox.addWidget(self.toolBar)
        self.toolBar.addAction("2NumPy", self.toArray)
        self.scene = QtGui.QGraphicsScene()
        self.view = QtGui.QGraphicsView(self.scene, self)
        self.vbox.addWidget(self.view)

    def toArray(self):
        pixmap = QtGui.QPixmap.grabWidget(self.view)
        byteArray = QtCore.QByteArray()
        _buffer = QtCore.QBuffer(byteArray)
        _buffer.open(QtCore.QIODevice.WriteOnly)
        pixmap.save(_buffer, 'PNG')
        _file = io.BytesIO(byteArray.data())
        array = misc.imread(_file)
        return array


class GUI(QtGui.QMainWindow):

    def __init__(self, parent=None):
        self.app = QtGui.QApplication([])
        QtGui.QMainWindow.__init__(self, parent)

        self.initToolBar()
        self.initVtk() # the show command is called here, before vtk initialize
        self.initPlot()
        # self.initTree()
        self.statusBar().showMessage("Ready")

    def initToolBar(self):
        self.toolBar = QtGui.QToolBar(self)
        self.addToolBar(self.toolBar)

    def initVtk(self):
        self.frame = QtGui.QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(parent=self.frame)

        self.vl = QtGui.QVBoxLayout()
        self.vl.addWidget(self.vtkWidget)

        self.scene = Scene(self.vtkWidget)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

    def initPlot(self):
        self.plotWidget = QtGraph.PlotWidget(self)
        self.plotWidget.setMouseEnabled(False, False)
        self.plotDockWidget = QtGui.QDockWidget("Plot", self)
        self.plotDockWidget.setFeatures(QtGui.QDockWidget.DockWidgetClosable)
        self.plotDockWidget.setWidget(self.plotWidget)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.plotDockWidget)
        self.toolBar.addAction(self.plotDockWidget.toggleViewAction())

    def initTree(self):
        self.treeView = QtGui.QTreeView(self)
        self.treeDockWidget = QtGui.QDockWidget("Tree", self)
        self.treeDockWidget.setFeatures(QtGui.QDockWidget.DockWidgetClosable)
        self.treeDockWidget.setWidget(self.treeView)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.treeDockWidget)
        self.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.LeftDockWidgetArea)
        self.toolBar.addAction(self.treeDockWidget.toggleViewAction())

    def plotXY(self, x, y):
        self.plotWidget.clear()
        self.plotWidget.plot(x,y, symbol='o')
        self.timeLine = QtGraph.InfiniteLine(angle=90, movable=True)
        self.timeLine.setBounds((min(x), max(x)))
        self.timeLine.sigPositionChanged.connect(self.updateRenderWindow)
        self.plotWidget.addItem(self.timeLine)

    def updateRenderWindow(self):
        x = self.plotWidget.plotItem.listDataItems()[0].xData
        t = bisect.bisect_left(x, self.timeLine.value())
        self.scene.update_all(t=t)

    def run(self):
        self.show()
        self.scene.iren.Initialize()
        self.app.exec_()
