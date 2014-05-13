from __future__ import print_function
from itertools import cycle
import numpy as np
from matplotlib import cm
import vtk

from .misc import normalize

def vscalars(data):
    dataArray = vtk.vtkDoubleArray()
    for i in data:
        dataArray.InsertNextValue(i)
    return dataArray

def vpoints(coords):
    points = vtk.vtkPoints()
    for x,y,z in coords:
        points.InsertNextPoint(x, y, z)
    return points

def vpolys(pairs):
    polys = vtk.vtkCellArray()
    for hi, ti in pairs:
        vil = vtk.vtkIdList()
        vil.InsertNextId(hi)
        vil.InsertNextId(ti)
        polys.InsertNextCell(vil)
    return polys

def vcolors(values, cmap='coolwarm'):
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colormap = cm.get_cmap(cmap)
    mapped = colormap(values)
    for r,g,b,a in 255*mapped:
        colors.InsertNextTuple3(r,g,b)
    return colors


class Scene(object):

    def __init__(self):
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetSize(800,600)
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.iren.Initialize()
        self.iren.AddObserver('TimerEvent', lambda obj, ev: self.renWin.Render())

    def add_wires(self, points, pairs, point_weights=None, alpha=1):
        '''
        consists of points and lines
        '''
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vpoints(points))
        polydata.SetLines(vpolys(pairs))
        if point_weights is not None:
            looper = cycle(np.atleast_2d(normalize(point_weights)))
            def animation(obj, event):
                polydata.GetPointData().SetScalars(vcolors(next(looper)))
            self.iren.AddObserver('TimerEvent', animation)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(alpha)
        self.ren.AddActor(actor)

    def add_spheres(self, points, radii, alpha=1, color=(1,1,1)):
        '''
        consists of centers and radii
        '''
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vpoints(points))
        looper = cycle(np.atleast_2d(radii))
        def animation(obj, event):
            polydata.GetPointData().SetScalars(vscalars(next(looper)))
        self.iren.AddObserver('TimerEvent', animation)

        def Glyph():
            pid = glypher.GetPointId()
            sphere.SetCenter(glypher.GetPoint())
            sphere.SetRadius(glypher.GetPointData().GetScalars().GetValue(pid))
            glyphActor.GetProperty().SetColor(*color)

        sphere = vtk.vtkSphereSource()
        glypher = vtk.vtkProgrammableGlyphFilter()
        glypher.SetInput(polydata)
        glypher.SetSource(sphere.GetOutput())
        glypher.SetGlyphMethod(Glyph)
        glyphMapper = vtk.vtkPolyDataMapper()
        glyphMapper.SetInputConnection(glypher.GetOutputPort())
        glyphActor = vtk.vtkActor()
        glyphActor.SetMapper(glyphMapper)
        glyphActor.GetProperty().SetOpacity(alpha)
        glyphActor.GetProperty().SetColor(*color)
        self.ren.AddActor(glyphActor)

    def play(self, rate=1):
        if rate:
            timer = self.iren.CreateRepeatingTimer(rate)
        self.iren.Start()
