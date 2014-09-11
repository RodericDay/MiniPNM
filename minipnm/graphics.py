from tempfile import NamedTemporaryFile
from subprocess import call
import itertools as it
import numpy as np
from matplotlib import cm

try:
    import vtk
except ImportError:
    vtk = type("vtk module missing. functionality unavailable!",
               (), {'vtkActor': object})

def bind(array):
    array = np.atleast_2d(array).astype(float)
    array = np.subtract(array, array.min())
    array = np.divide(array, array.max()) if array.max() != 0 else array
    return array

class Actor(vtk.vtkActor):

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

    def connect(self, fun):
        self.callable = fun

    def update(self, t=0):
        i = t % len(self.script)
        self.set_scalars(self.script[i])


class Wires(Actor):
    '''
    Points and lines. The script determines the color of the lines.
    '''
    def __init__(self, points, pairs, values=None, alpha=1, cmap=None):
        self.polydata = vtk.vtkPolyData()
        self.set_points(points)
        self.set_lines(pairs)
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInput(self.polydata)
        self.SetMapper(self.mapper)

        if values is None:
            values = [0.5 for _ in points]
        self.script = 255*bind(values)
        self.cmap = cmap
        self.update()

        self.GetProperty().SetOpacity(alpha)

    def set_scalars(self, values):
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        if self.cmap is None:
            self.cmap = 'coolwarm'
        colormap = cm.get_cmap(self.cmap)
        mapped = colormap(values.astype(int))
        for r,g,b,a in 255*mapped:
            colors.InsertNextTuple3(r,g,b)
        self.polydata.GetPointData().SetScalars(colors)


class Spheres(Actor):

    def __init__(self, centers, radii, alpha=1, color=(1,1,1)):
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

        self.script = 2*np.atleast_2d(radii)

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
        radii = np.hstack([radii, radii])

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
        self.tubeFilter.CappingOn()

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.tubeFilter.GetOutputPort())
        self.mapper.ScalarVisibilityOff()
        self.SetMapper(self.mapper)

        self.GetProperty().SetOpacity(alpha)

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
        return max(len(actor.script) for actor in self)

    def handle_pick(self, obj, event):
        try:
            actor = self.picker.GetActor()
            glyph3D = actor.glyph3D
            pointIds = glyph3D.GetOutput().GetPointData().GetArray("InputPointIds")
            selectedId = int(pointIds.GetTuple1(self.picker.GetPointId()))
            actor.callable(selectedId)
            actor.polydata.Modified()
            self.renWin.Render()
        except Exception as e:
            print e

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
