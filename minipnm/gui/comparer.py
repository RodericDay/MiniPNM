from PyQt4 import QtCore, QtGui
import pyqtgraph as QtGraph
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk


class SceneWidget(QtGui.QFrame):

    def __init__(self):
        super(SceneWidget, self).__init__()
        self.renderer = vtk.vtkRenderer()
        self.interactor = QVTKRenderWindowInteractor(parent=self)
        self.interactor.GetRenderWindow().AddRenderer(self.renderer)
        
        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.interactor)

        self.configure()

    def configure(self):
        # changes camera to a more sensible default
        camera = vtk.vtkInteractorStyleTrackballCamera()
        camera.SetCurrentRenderer(self.renderer)
        self.interactor.SetInteractorStyle(camera)

        # set up picker to handle item selections
        self.picker = vtk.vtkCellPicker()
        self.interactor.SetPicker(self.picker)
        self.picker.AddObserver("EndPickEvent", self.handlePick)

    def addActors(self, actorList):
        for actor in actorList:
            self.renderer.AddActor(actor)

    def handlePick(self, obj, event):
        actor = self.picker.GetActor()
        try:
            glyph3D = actor.glyph3D
        except:
            return
        pointIds = glyph3D.GetOutput().GetPointData().GetArray("InputPointIds")
        selectedId = int(pointIds.GetTuple1(self.picker.GetPointId()))

        actor.callable(selectedId)
        actor.polydata.Modified()

        currents = model.polarization_curve(voltages)
        line.setData(x=currents(A/m**2), y=line.yData)
        
        self.interactor.GetRenderWindow().Render()


class Main(QtGui.QMainWindow):

    def __init__(self):
        super(Main, self).__init__()
        self.toolBar = self.addToolBar("")
        self.toolBar.addAction("X", app.quit).setShortcut("Q")
        self.widget = QtGui.QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QtGui.QVBoxLayout(self.widget)


if __name__ == '__main__':
    import numpy as np
    import minipnm as mini
    import MKS
    MKS.define(globals())

    app = QtGui.QApplication([])
    
    model = mini.models.SimpleLatticedCatalystLayer.test()

    sceneFrame = SceneWidget()
    sceneFrame.addActors(model.geometry.actors())

    plotWidget = QtGraph.PlotWidget()

    voltages = np.arange(0, 1.5, 0.1) * V
    currents = model.polarization_curve(voltages)
    plotWidget.plot( currents(A/m**2), voltages(V), pen='g' )
    line = plotWidget.plot( currents(A/m**2), voltages(V), pen='r' )

    main = Main()
    main.layout.addWidget(sceneFrame)
    main.layout.addWidget(plotWidget)
    main.show()
    sceneFrame.interactor.Initialize()

    app.exec_()
