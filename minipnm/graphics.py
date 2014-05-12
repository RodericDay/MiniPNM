from __future__ import print_function
from itertools import cycle
import numpy as np
from matplotlib import cm
import vtk

from .misc import normalize

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
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.iren.Initialize()
        self.iren.AddObserver('TimerEvent', lambda obj, ev: self.renWin.Render())
        timer = self.iren.CreateRepeatingTimer(1)

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

    def add_spheres(self, points, radii, fill_history=None, alpha=0.3):
        '''
        consists of centers and radii
        '''
        for i, (point, radius) in enumerate(zip(points, radii)):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*point)
            sphere.SetRadius(radius)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(alpha)
            self.ren.AddActor(actor)

    def play(self):
        self.iren.Start()
