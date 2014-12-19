import io
from scipy import misc

class CanvasWidget(QtGui.QWidget):
    finalized = QtCore.pyqtSignal(object)

    def __init__(self):
        super(CanvasWidget, self).__init__()
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
        self.finalized.emit(array)

if __name__ == '__main__':
    app = QtGui.QApplication([])

    canvas = CanvasWidget()
    canvas.finalized.connect(exit)
    canvas.show()

    app.exec_()
