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
    
    sceneFrame.updateLight()

    app.exec_()
