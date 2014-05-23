from __future__ import print_function
import os
from itertools import cycle
import numpy as np
from matplotlib import cm
import vtk

from .misc import normalize

def vscalars(data):
    dataArray = vtk.vtkFloatArray()
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

    def __init__(self, size=(800,600)):
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetSize(*size)
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)
        self.iren.Initialize()
        self.iren.AddObserver('TimerEvent', self.timeout)
        self.animation_batch = []

    def map_and_append(fn):
        '''
        this decorator handles some boilerplate code regarding actor addition
        '''
        def handler(self, *args, **kwargs):
            alpha = kwargs.pop('alpha', 1)
            color = kwargs.pop('color', None)

            polydata = fn(self, *args, **kwargs)
            mapper = vtk.vtkPolyDataMapper()
            try:
                mapper.SetInput(polydata)
            except TypeError:
                mapper.SetInputConnection(polydata.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(alpha)
            if color is not None:
                mapper.SetScalarVisibility(False)
                actor.GetProperty().SetColor(*color)
            self.ren.AddActor(actor)

        return handler

    @map_and_append
    def add_wires(self, points, pairs, history=None, cmap='coolwarm'):
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vpoints(points))
        polydata.SetLines(vpolys(pairs))

        if history is not None:
            looper = cycle(np.atleast_2d(normalize(history)))
            self.animation_batch.append(
                lambda: polydata.GetPointData().SetScalars(
                        vcolors(next(looper), cmap))
            )

        return polydata

    @map_and_append
    def add_tubes(self, points, pairs, history=None, radius=1, detail=5,
                  cmap='coolwarm'):
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vpoints(points))
        polydata.SetLines(vpolys(pairs))

        if history is not None:
            looper = cycle(np.atleast_2d(normalize(history)))
            self.animation_batch.append(
                lambda: polydata.GetPointData().SetScalars(
                        vcolors(next(looper), cmap))
            )

        tubeFilter = vtk.vtkTubeFilter()
        tubeFilter.SetInput(polydata)
        tubeFilter.SetRadius(radius)
        tubeFilter.SetNumberOfSides(detail)
        return tubeFilter

    @map_and_append
    def add_spheres(self, points, radii):
        '''
        consists of centers and radii
        '''
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vpoints(points))

        looper = cycle(np.atleast_2d(radii))
        self.animation_batch.append(
            lambda: polydata.GetPointData().SetScalars(
                    vscalars(next(looper)))
        )

        def Glyph():
            pid = glypher.GetPointId()
            sphere.SetCenter(glypher.GetPoint())
            try:
                value = glypher.GetPointData().GetScalars().GetValue(pid)
                sphere.SetRadius(value)
            except AttributeError:
                pass

        sphere = vtk.vtkSphereSource()
        glypher = vtk.vtkProgrammableGlyphFilter()
        glypher.SetInput(polydata)
        glypher.SetSource(sphere.GetOutput())
        glypher.SetGlyphMethod(Glyph)

        return glypher

    def timeout(self, object=None, event=None):
        for fn in self.animation_batch:
            fn()
        self.ren.ResetCameraClippingRange()
        self.renWin.Render()

    def play(self, rate=1):
        if rate:
            timer = self.iren.CreateRepeatingTimer(rate)
        self.iren.Start()

    def save(self, size=(400,300), frames=1):
        self.renWin.SetSize(*size)
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.renWin)
         
        writer = vtk.vtkPNGWriter()
        try:
            os.system("mkdir tmp")

            for i in range(frames):
                self.timeout()
                w2if.Modified()
                writer.SetFileName("tmp/{:0>3}.png".format(i))
                writer.SetInput(w2if.GetOutput())
                writer.Write()

            os.system("convert -delay 20 -loop 0 ./tmp/*.png ~/animated.gif")
        finally:
            os.system("rm -r tmp")
