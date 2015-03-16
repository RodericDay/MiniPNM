try:
    from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from PyQt4 import QtGui, QtCore
    import pyqtgraph as QtGraph
except ImportError:
    QtGui = type("vtk, PyQt, or pyqtgraph missing. functionality unavailable!",
                 (), {"QWidget": object, "QMainWindow": object})

from .floodview import floodview
from .profileview import profileview
