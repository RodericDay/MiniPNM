from PyQt4 import QtCore, QtGui
import pyqtgraph as QtGraph
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk


class SceneWidget(QtGui.QFrame):
    nodeSelected = QtCore.pyqtSignal(int)

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
        pickPoint = self.picker.GetPointId()
        try:
            glyph3D = actor.glyph3D
        except:
            return
        pointIds = glyph3D.GetOutput().GetPointData().GetArray("InputPointIds")
        selectedId = int(pointIds.GetTuple1(pickPoint))
        actor.polydata.Modified()

        self.nodeSelected.emit(selectedId)

    def showEvent(self, event):
        '''
        call Initialize on the interactor *after* the window is first shown
        '''
        super(SceneWidget, self).showEvent(event)
        self.interactor.Initialize()


class DualWidget(QtGui.QMainWindow):
    '''
    generic settings for widgets which contain both a render window
    and a plot widget
    '''
    def __init__(self):
        super(DualWidget, self).__init__()
        self.sceneWidget = SceneWidget()
        self.plotWidget = QtGraph.PlotWidget()
        self.plotWidget.setMouseEnabled(False, False)

        self.container = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout(self.container)
        self.layout.addWidget(self.sceneWidget)
        self.layout.addWidget(self.plotWidget)
        self.setCentralWidget(self.container)


class GeometryWidget(DualWidget):
    
    def __init__(self, geometry):
        super(GeometryWidget, self).__init__()

        shells, tubes, self.balloons = geometry.actors(0)
        self.sceneWidget.addActors([shells, tubes, self.balloons])
        self.sceneWidget.nodeSelected.connect(self.swapFloodState)
        r = geometry.spheres.radii / 1E-9
        values, bin_edges = np.histogram(r)
        curve = self.plotWidget.plot(bin_edges, values, stepMode=True, fillLevel=0, brush=(0, 255, 0, 80))
        self.plotWidget.setLabels(bottom='r (nm)', left='N')

    def swapFloodState(self, nodeId):
        model = self.parent().parent().model
        model.water_transport[nodeId] = ~model.water_transport[nodeId]
        self.balloons.script[:] = model.geometry.spheres.radii*2 * model.water_transport[:]
        self.balloons.update()

        self.parent().parent().plotLastCase()
        self.parent().parent().boundaryConditionChanged()


class DataWidget(DualWidget):
    
    def __init__(self, topology):
        super(DataWidget, self).__init__()

        self.displaySelector = QtGui.QComboBox()
        self.displaySelector.addItems([
            # 'oxygen_molar_fraction',
            'local_current_density',
            'protonic_potential',
            'electronic_potential',
            'overpotential',
        ])
        self.toolBar = self.addToolBar("Tools")
        self.toolBar.addWidget(self.displaySelector)
        self.displaySelector.currentIndexChanged.connect(self.updateContents)

        self.wires, = mini.Network.load(topology).actors()
        self.sceneWidget.addActors([self.wires])
        x,y,z = topology.coords / 1E-6
        r = np.random.rand(x.size)
        self.curve = self.plotWidget.plot(x, r, symbol='o', pen=None)
        self.plotWidget.setLabels(bottom='x (um)')

    def updateContents(self):
        attribute = str(self.displaySelector.currentText())
        model = self.parent().parent().model
        data = getattr(model, attribute)
        try: data=data.quantity
        except: pass
        self.curve.setData(self.curve.xData, data)

        self.wires.script[:] = data
        self.wires.update()
        self.sceneWidget.interactor.GetRenderWindow().Render()    


class ModelExplorer(QtGui.QMainWindow):

    def __init__(self, model):
        super(ModelExplorer, self).__init__()
        self.toolBar = self.addToolBar("Main")
        self.toolBar.addAction("Exit", exit).setShortcut("q")

        self.centralWidget = QtGui.QWidget()
        self.layout = QtGui.QHBoxLayout(self.centralWidget)
        self.setCentralWidget(self.centralWidget)

        self.model = model

        self.geometryWidget = GeometryWidget(model.geometry)
        self.layout.addWidget(self.geometryWidget)

        self.dataWidget = DataWidget(model.topology)
        self.layout.addWidget(self.dataWidget)

        self.polarizationCurveWidget = QtGraph.PlotWidget()
        self.polarizationCurveWidget.setMouseEnabled(False, False)
        self.polarizationCurveWidget.setLabels(bottom='Current Density (A/cm2)', left='Voltage (V)')

        self.userInputLine = self.polarizationCurveWidget.addLine(y=0.6, movable=True)
        self.userInputLine.sigPositionChanged.connect(self.boundaryConditionChanged)
        self.polarizationCurve = self.polarizationCurveWidget.plot([], [])
        self.layout.addWidget(self.polarizationCurveWidget)

        self.boundaryConditionChanged()
        self.show()

    def boundaryConditionChanged(self):
        voltage = self.userInputLine.value() * V
        self.model.reach_steady_state(voltage)
        self.dataWidget.updateContents()

    def plotBaseCase(self, voltages):
        currents = self.model.polarization_curve(voltages)
        self.polarizationCurveWidget.plot( currents(A/cm**2), voltages(V), pen='b')

    def plotLastCase(self):
        voltages = np.linspace(0.65, 1, 10) * V
        currents = self.model.polarization_curve(voltages)
        self.polarizationCurve.setData(currents(A/cm**2), voltages(V), pen='r')

    def plotRobBenchmarkData(self):
        i0 = np.array([0.0001, 0.0302, 0.0502, 0.0601, 0.0996, 0.2002, 0.5994, 0.9989, 1.1990, 1.4985, 1.6978])
        v0 = np.array([0.9478, 0.8704, 0.8551, 0.8502, 0.8315, 0.7865, 0.7418, 0.7116, 0.7007, 0.6839, 0.6550])
        self.polarizationCurveWidget.plot(i0, v0, pen='g')


if __name__ == '__main__':
    import numpy as np
    import minipnm as mini
    import MKS
    MKS.define(globals())

    model = mini.models.SimpleLatticedCatalystLayer.test()
    print model.npores

    app = QtGui.QApplication([])
    modelExplorer = ModelExplorer(model)
    voltages = np.linspace(0.5, 1, 10) * V
    modelExplorer.plotBaseCase(voltages)
    modelExplorer.plotRobBenchmarkData()
    app.exec_()
