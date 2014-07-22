from itertools import count
import numpy as np
from matplotlib import cm
import vtk


class Actor(vtk.vtkActor):

    def __init__(self):
        raise NotImplementedError()

    def pointArray(self, vertex_coords):
        points = vtk.vtkPoints()
        for x,y,z in vertex_coords:
            points.InsertNextPoint(x, y, z)
        return points

    def lineArray(self, edge_pairs):
        lines = vtk.vtkCellArray()
        for t,h in edge_pairs:
            l = vtk.vtkIdList()
            l.InsertNextId(t)
            l.InsertNextId(h)
            lines.InsertNextCell(l)
        return lines

    def floatArray(self, array):
        floats = vtk.vtkFloatArray()
        for n in array:
            floats.InsertNextValue(n)
        return floats

    def colorArray(self, array, cmap=None):
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        if cmap is None:
            cmap = 'coolwarm'
        colormap = cm.get_cmap(cmap)
        mapped = colormap(array)
        for r,g,b,a in 255*mapped:
            colors.InsertNextTuple3(r,g,b)
        return colors

    def update(self, t=0):
        pass


class Wires(Actor):

    def __init__(self, vertex_coords, edge_pairs, vertex_weights=None, alpha=1, cmap=None):
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.pointArray(vertex_coords))
        self.polydata.SetLines(self.lineArray(edge_pairs))
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInput(self.polydata)
        self.SetMapper(self.mapper)

        if vertex_weights is None:
            weights = np.atleast_2d([128 for _ in vertex_coords])
        else:
            weights = np.atleast_2d(vertex_weights)
            weights = np.subtract(weights, weights.min())
            weights = np.true_divide(weights, weights.max())
        self.weights = weights
        self.cmap = cmap
        self.update()
        
        self.GetProperty().SetOpacity(alpha)

    def update(self, t=0):
        i = t % len(self.weights)
        self.polydata.GetPointData().SetScalars(self.colorArray(self.weights[i], self.cmap))


class Tubes(Actor):

    def __init__(self, centers, vectors, radii, alpha=1, cmap=None):
        tails = centers - vectors/2.
        heads = centers + vectors/2.
        points = np.vstack(zip(tails, heads))
        pairs = np.arange(len(centers)*2).reshape(-1, 2)
        radii = radii.repeat(2)

        assert (points.size/3. == pairs.size)
        assert (pairs.size == radii.size)

        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.pointArray(points))
        self.polydata.SetLines(self.lineArray(pairs))
        self.polydata.GetPointData().SetScalars(self.floatArray(radii))

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


class Spheres(Actor):

    def __init__(self, centers, radii, alpha=1, color=(1,1,1)):
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.pointArray(centers))
        self.radii = np.atleast_2d(radii)
        self.update()

        self.sphere_source = vtk.vtkSphereSource()
        self.glypher = vtk.vtkProgrammableGlyphFilter()
        self.glypher.SetInput(self.polydata)
        self.glypher.SetSource(self.sphere_source.GetOutput())
        self.glypher.SetGlyphMethod(self.glyph_method)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.glypher.GetOutputPort())
        self.SetMapper(self.mapper)

        self.GetProperty().SetOpacity(alpha)
        r,g,b = color
        self.mapper.SetScalarVisibility(False)
        self.GetProperty().SetColor(r,g,b)

    def glyph_method(self):
        pid = self.glypher.GetPointId()
        self.sphere_source.SetCenter(self.glypher.GetPoint())
        radius = self.glypher.GetPointData().GetScalars().GetValue(pid)
        self.sphere_source.SetRadius(radius)

    def update(self, t=0):
        i = t % len(self.radii)
        self.polydata.GetPointData().SetScalars(self.floatArray(self.radii[i]))


class Scene(object):
    ticks = count(0)

    def __init__(self, fix_camera=True):
        '''
        fix_camera : more sensible default
        '''
        # create
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()

        # hookups
        self.renWin.AddRenderer(self.ren)
        self.iren.SetRenderWindow(self.renWin)
        
        if fix_camera:
            camera = vtk.vtkInteractorStyleTrackballCamera()
            self.iren.SetInteractorStyle(camera)

    def update_all(self, object=None, event=None):
        for aid in range(self.ren.VisibleActorCount()):
            actor = self.ren.GetActors().GetItemAsObject(aid)
            actor.update(next(self.ticks))
        self.renWin.Render()

    def play(self, timeout=1):
        self.iren.Initialize()
        if timeout is not None:
            self.iren.AddObserver('TimerEvent', self.update_all)
            self.timer = self.iren.CreateRepeatingTimer(timeout)
        self.update_all()
        self.iren.Start()
        
    def add_actor(self, actor):
        self.ren.AddActor(actor)

    # legacy functions
    def add_wires(self, points, pairs, weights=None, alpha=1, cmap=None):
        wires = Wires(points, pairs, weights, alpha, cmap)
        self.add_actor(wires)

    def add_tubes(self, centers, vectors, radii, alpha=1, cmap=None):
        tubes = Tubes(centers, vectors, radii, alpha, cmap)
        self.add_actor(tubes)

    def add_spheres(self, points, radii, alpha=1, color=(1,1,1)):
        spheres = Spheres(points, radii, alpha, color)
        self.add_actor(spheres)

#~~
def save_gif(self, size=(400,300), frames=1):
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