from __future__ import print_function
import os
from PyQt4 import QtCore, QtGui
import minipnm as mini

def save(button_pressed):
    pixmap = QtGui.QPixmap.grabWidget(view)
    pixmap.save('Screenshot.png')

def process(button_pressed):
    pixmap = QtGui.QPixmap.grabWidget(view)
    pixmap.save('temp.png')
    m.close()

    # departure zone
    try:
        im = mini.imread('temp.png', zoom=0.15)
        im = im[1:-1,1:-1] # get rid of stupid borders
        centers, radii = mini.extract_spheres(~im[:,::-1])
        network = mini.Delaunay(centers)
        network['radii'] = radii
        
        scene = mini.Scene()
        scene.add_spheres(network.points, network['radii'])
        try:
            scene.add_tubes(network.points, network.pairs)
        except IndexError:
            pass
        scene.play()
    finally:
        os.system("rm temp.png")

def mouseMoveEvent(event):
    event.accept()
    scenePos = event.scenePos()
    x, y = scenePos.x(), scenePos.y()
    brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
    w = h = penSize.value()
    scene.addEllipse(x-w//2, y-h//2, w, h, brush=brush)

def wheelEvent(event):
    event.accept()
    delta = event.delta() # 120
    penSize.setValue( penSize.value() + 20*delta//120 )

app = QtGui.QApplication([])

scene = QtGui.QGraphicsScene()
penSize = QtGui.QSpinBox()
penSize.setValue(50)
penSize.setRange(10,500)
scene.mousePressEvent = mouseMoveEvent
scene.mouseMoveEvent = mouseMoveEvent
scene.wheelEvent = wheelEvent
    
scene.setSceneRect(0,0,800,600)
view = QtGui.QGraphicsView(scene)

m = QtGui.QMainWindow()
m.setGeometry(200,200,1000,800)

t = QtGui.QToolBar()
for a in [
    QtGui.QAction("Save", m, triggered=save, shortcut=QtGui.QKeySequence.Save),
    QtGui.QAction("Clear", m, triggered=lambda event: scene.clear()),
    QtGui.QAction("Process", m, triggered=process)
    ]:
    t.addAction(a)
t.addWidget(penSize)
m.addToolBar(t)
m.setCentralWidget(view)
m.show()

app.exec_()