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

class RenderedItemsModel(QtCore.QAbstractItemModel):

    def __init__(self, renderer, parent=None):
        super(RenderedItemsModel, self).__init__(parent)
        self.ren = renderer

    def index(self, row, column, parent):
        return self.createIndex(row, column)

    def parent(self, index):
        return self.createIndex(1, 0)

    def rowCount(self, parent):
        return self.ren.GetActors().GetReferenceCount()

    def columnCount(self, parent):
        return 1

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole and index.column() == 0:
            return str(self.ren.GetActors().GetItemAsObject(index.row()))
        return None

class MainWindow(QtGui.QMainWindow):

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.initVtk() # the show command is called here, before vtk initialize
        self.initPlot()
        self.initTree()
        self.initToolBar()
        self.statusBar().showMessage("Ready")

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

    def initTree(self):
        self.treeModel = RenderedItemsModel(self.ren, self)
        self.treeView = QtGui.QTreeView(self)
        self.treeView.setModel(self.treeModel)
        self.treeDockWidget = QtGui.QDockWidget("Tree", self)
        self.treeDockWidget.setFeatures(QtGui.QDockWidget.DockWidgetClosable)
        self.treeDockWidget.setWidget(self.treeView)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.treeDockWidget)
        self.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.LeftDockWidgetArea)

    def initToolBar(self):
        self.toolBar = QtGui.QToolBar(self)
        self.addToolBar(self.toolBar)
        self.toolBar.addAction(self.plotDockWidget.toggleViewAction())
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


def create_2d_data():
    x = np.linspace(0,1,1000)
    y = np.sin(x)
    return x, y

def debug():
    window.treeDockWidget.hide()
    window.addActor(Sphere())
    window.addActor(Sphere((1,2,3)))
    window.plotXY(*create_2d_data())

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    debug()
    sys.exit(app.exec_())
