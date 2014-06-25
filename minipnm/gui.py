from itertools import count
import numpy as np
from PyQt4 import QtGui, QtCore
import pyqtgraph as QtGraph
import vtk
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class Sphere(vtk.vtkActor):
    _ids = count(1)

    def __init__(self, center=(0,0,0), radius=1):
        self.id = self._ids.next()

        # Create source
        source = vtk.vtkSphereSource()
        source.SetCenter(*center)
        source.SetRadius(radius)

        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        self.SetMapper(mapper)

    def update(self, t):
        self.GetProperty().SetOpacity(t)

    def __str__(self):
        return "{} #{:0>3}".format(self.__class__.__name__, self.id)

class MainWindow(QtGui.QMainWindow):

    def __init__(self, parent=None):
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

        self.vl = QtGui.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.show()
        self.iren.Initialize()

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

    def addActor(self, actor):
        self.ren.AddActor(actor)
        self.ren.ResetCamera()

    def plotXY(self, x, y):
        self.plotWidget.clear()
        self.plotWidget.plot(x,y)
        self.timeLine = QtGraph.InfiniteLine(angle=90, movable=True)
        self.timeLine.setBounds((min(x), max(x)))
        self.timeLine.sigPositionChanged.connect(self.updateRenderWindow)
        self.plotWidget.addItem(self.timeLine)

    def updateRenderWindow(self):
        t = self.timeLine.value()
        for idx in range(self.ren.VisibleActorCount()):
            actor = self.ren.GetActors().GetItemAsObject(idx)
            actor.update(t)
        self.iren.Render()


def debug():
    import sys
    import minipnm as mini

    pdf = (np.random.weibull(3) for _ in count())
    network = mini.Bridson(pdf, [50,50,5])
    network.pairs = mini.Delaunay.edges_from_points(network.points)

    x,y,z = network.coords
    source = x==x.min()
    history = mini.invasion(network, source, 1./network['radii'])

    wires = mini.Wires(network.points, network.pairs)
    spheres = mini.Spheres(network.points, network['radii'] * history)

    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.addActor(wires)
    window.addActor(spheres)

    x = np.arange(len(history))
    y = (network['radii'] * history).sum(axis=1)
    window.plotXY(x,y)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    debug()
