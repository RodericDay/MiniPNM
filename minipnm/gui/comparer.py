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
        actor.polydata.Modified()

        model.water_transport[selectedId] = ~model.water_transport[selectedId]
        history.script[:] = model.geometry.spheres.radii*2 * model.water_transport[:] # update graphx
        history.update()

        # actor.callable(selectedId)
        currents = model.polarization_curve(voltages)
        polarizationCurve.setData(x=currents(A/cm**2), y=polarizationCurve.yData) #autoupdates :)
        
        self.updateLight()

    def updateLight(self):
        t = model.topology.coords[0]
        y = model.reach_steady_state(variableLine.value()*V)
        y[0] = 0 # stupid way of stopping auto-range from messing things
        attributeString = str(comboBox.currentText())
        if attributeString != 'oxygen_molar_fraction':
            y = getattr(model, attributeString).quantity
        transversalCurve.setData(t, y)

        wires.script[:] = y * 10
        wires.update()

        self.interactor.GetRenderWindow().Render()     


class Main(QtGui.QMainWindow):

    def __init__(self):
        super(Main, self).__init__()
        self.toolBar = self.addToolBar("")
        self.toolBar.addAction("X", app.quit).setShortcut("Q")
        self.widget = QtGui.QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QtGui.QGridLayout(self.widget)


if __name__ == '__main__':
    import numpy as np
    import minipnm as mini
    import MKS
    MKS.define(globals())

    app = QtGui.QApplication([])
    
    model = mini.models.SimpleLatticedCatalystLayer.test()
    print model.npores
    spheres, tubes, history = model.geometry.actors(model.water_transport)
    wires, = model.topology.actors(offset=model.topology.bbox*[0,-1.1,0])
    voltages = np.arange(0, 1.5, 0.1) * V
    currents = model.polarization_curve(voltages)

    sceneFrame = SceneWidget()
    sceneFrame.addActors([spheres, tubes, history])
    sceneFrame.addActors([wires])

    polarizationCurveWidget = QtGraph.PlotWidget()
    polarizationCurveWidget.setMouseEnabled(False, False)
    polarizationCurveWidget.plot( currents(A/cm**2), voltages(V), pen='g' )
    polarizationCurve = polarizationCurveWidget.plot( currents(A/cm**2), voltages(V), pen='r' )
    variableLine = polarizationCurveWidget.addLine(y=0.6, movable=True)
    variableLine.sigPositionChanged.connect(sceneFrame.updateLight)

    transversalCurveWidget = QtGraph.PlotWidget()
    transversalCurveWidget.setMouseEnabled(False, False)
    transversalCurve = transversalCurveWidget.plot( [0,1], [1,0], pen=None, symbol='o' )

    comboBox = QtGui.QComboBox()
    comboBox.addItems(['oxygen_molar_fraction','protonic_potential',
                        'electronic_potential','overpotential','local_current'])
    comboBox.currentIndexChanged.connect(sceneFrame.updateLight)

    main = Main()
    main.toolBar.addWidget(comboBox)
    main.layout.addWidget(sceneFrame, 0, 0)
    main.layout.addWidget(transversalCurveWidget, 0, 1)
    main.layout.addWidget(polarizationCurveWidget, 1, 0, 1, 2)
    main.show()
    sceneFrame.interactor.Initialize()
    sceneFrame.updateLight()

    app.exec_()
