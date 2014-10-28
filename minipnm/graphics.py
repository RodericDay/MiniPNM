from __future__ import print_function
from tempfile import NamedTemporaryFile
from subprocess import call
import itertools as it
import numpy as np

import matplotlib as mpl
from matplotlib import cm

try:
    import vtk
except ImportError:
    vtk = type("vtk module missing. functionality unavailable!",
               (), {'vtkActor': object})


class Actor(vtk.vtkActor):
    callable = print

    def __init__(self):
        raise NotImplementedError()

    def set_points(self, points):
        pointArray = vtk.vtkPoints()
        for x,y,z in points:
            pointArray.InsertNextPoint(x, y, z)
        self.polydata.SetPoints(pointArray)

    def set_lines(self, lines):
        cellArray = vtk.vtkCellArray()
        for ids in lines:
            idList = vtk.vtkIdList()
            for i in ids:
                idList.InsertNextId(i)
            cellArray.InsertNextCell(idList)
        self.polydata.SetLines(cellArray)

    def set_scalars(self, values):
        floats = vtk.vtkFloatArray()
        for v in values:
            floats.InsertNextValue(v)
        self.polydata.GetPointData().SetScalars(floats)

    def update(self, t=0):
        i = t % len(self.script)
        self.set_scalars(self.script[i])


class Wires(Actor):
    '''
    Points and lines. The script determines the color of the lines.
    '''
    def __init__(self, points, pairs, values=None, cmap=None, alpha=1, vmin=None, vmax=None):
        self.polydata = vtk.vtkPolyData()
        self.set_points(points)
        self.set_lines(pairs)
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInput(self.polydata)
        self.SetMapper(self.mapper)
        self.GetProperty().SetOpacity(alpha)

        if values is None:
            values = np.ones(len(points))*0.5
            vmin, vmax = 0, 1
        self.script = np.atleast_2d(values)
        cmap = cm.get_cmap(cmap if cmap is not None else 'coolwarm')
        vmin = vmin if vmin is not None else self.script.min()
        vmax = vmax if vmax is not None else self.script.max()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        self.update()

    def set_scalars(self, values):
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)

        for r,g,b,a in 255*self.cmapper.to_rgba(values):
            colors.InsertNextTuple3(r,g,b)

        self.polydata.GetPointData().SetScalars(colors)


class Spheres(Actor):

    def __init__(self, centers, radii=1, alpha=1, color=(1,1,1)):
        self.polydata = vtk.vtkPolyData()
        self.set_points(centers)

        self.source = vtk.vtkSphereSource()
        self.glyph3D = vtk.vtkGlyph3D()
        self.glyph3D.SetSourceConnection(self.source.GetOutputPort())
        self.glyph3D.SetInput(self.polydata)
        self.glyph3D.GeneratePointIdsOn()
        self.glyph3D.Update()

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.glyph3D.GetOutputPort())
        self.SetMapper(self.mapper)

        self.script = 2*np.atleast_2d(np.ones(len(centers)))*radii

        self.GetProperty().SetOpacity(alpha)
        r,g,b = color
        self.mapper.ScalarVisibilityOff()
        self.GetProperty().SetColor(r,g,b)

        self.update()


class Tubes(Actor):

    def __init__(self, centers, vectors, radii, alpha=1, cmap=None):
        tails = centers - np.divide(vectors, 2.)
        heads = centers + np.divide(vectors, 2.)
        points = np.vstack(zip(tails, heads))
        pairs = np.arange(len(centers)*2).reshape(-1, 2)
        radii = np.repeat(radii, 2)

        assert (points.size/3. == pairs.size)
        assert (pairs.size == radii.size)

        self.polydata = vtk.vtkPolyData()
        self.set_points(points)
        self.set_lines(pairs)
        self.set_scalars(radii)

        self.tubeFilter = vtk.vtkTubeFilter()
        self.tubeFilter.SetInput(self.polydata)
        self.tubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        self.tubeFilter.SetNumberOfSides(10)
        # self.tubeFilter.CappingOn()

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.tubeFilter.GetOutputPort())
        self.mapper.ScalarVisibilityOff()
        self.SetMapper(self.mapper)

        self.GetProperty().SetOpacity(alpha)

        self.script = [0]

    def update(self, t=0):
        pass


class Scene(object):
    ticks = it.count(0)

    def __init__(self, parent=None, fix_camera=True):
        '''
        fix_camera : more sensible default
        '''
        if parent is not None:
            self.renWin = parent.GetRenderWindow()
            self.iren = self.renWin.GetInteractor()
        else:
            self.renWin = vtk.vtkRenderWindow()
            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.SetRenderWindow(self.renWin)

        self.ren = vtk.vtkRenderer()
        self.renWin.AddRenderer(self.ren)

        if fix_camera:
            camera = vtk.vtkInteractorStyleTrackballCamera()
            camera.SetCurrentRenderer(self.ren)
            self.iren.SetInteractorStyle(camera)

        self.picker = vtk.vtkCellPicker()
        self.iren.SetPicker(self.picker)
        self.picker.AddObserver("EndPickEvent", self.handle_pick)

    def __iter__(self):
        for aid in range(self.ren.VisibleActorCount()):
            actor = self.ren.GetActors().GetItemAsObject(aid)
            if hasattr(actor, 'script'):
                yield actor

    @property
    def count(self):
        return len([actor for actor in self])
        
    def __len__(self):
        if self.count==0:
            return 0
        return max(len(actor.script) for actor in self)

    def handle_pick(self, obj, event):
        actor = self.picker.GetActor()
        glyph3D = actor.glyph3D
        pointIds = glyph3D.GetOutput().GetPointData().GetArray("InputPointIds")
        selectedId = int(pointIds.GetTuple1(self.picker.GetPointId()))
        actor.callable(selectedId)
        actor.polydata.Modified()
        self.renWin.Render()

    def update_all(self, obj=None, event=None, t=None):
        if t is None:   t = next(self.ticks)
        for actor in self:
            actor.update(t)
        self.renWin.Render()

    def save(self, frames, outfile='animated.gif'):
        '''
        takes a snapshot of the frames at given t, and returns the paths
        '''
        windowToImage = vtk.vtkWindowToImageFilter()
        windowToImage.SetInput(self.renWin)
        writer = vtk.vtkPNGWriter()
        writer.SetInput(windowToImage.GetOutput())

        slide_paths = []
        for t in frames:
            f = NamedTemporaryFile(suffix='.png', delete=False)
            self.update_all(t=t)
            windowToImage.Modified()
            writer.SetFileName(f.name)
            writer.Write()
            slide_paths.append( f.name )

        call(["convert"] + slide_paths + [outfile])
        call(["rm"] + slide_paths)

    def play(self, timeout=1):
        self.iren.Initialize()
        if timeout is not None:
            self.iren.AddObserver('TimerEvent', self.update_all)
            self.timer = self.iren.CreateRepeatingTimer(timeout)
        self.update_all()
        self.iren.Start()

    def add_actors(self, list_of_actors, label=False):
        for actor in list_of_actors:
            self.ren.AddActor(actor)
            if label:
                labelMapper = vtk.vtkLabeledDataMapper()
                labelMapper.SetInput(actor.polydata)
                labelActor = vtk.vtkActor2D()
                labelActor.SetMapper(labelMapper)
                self.ren.AddActor(labelActor)
