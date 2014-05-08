from collections import Counter
import numpy as np
from matplotlib import cm
import vtk

def vpoints(network):
    points = vtk.vtkPoints()
    for x,y,z in network.points:
        points.InsertNextPoint(x, y, z)
    return points

def vpolys(network):
    polys = vtk.vtkCellArray()
    for hi, ti in network.pairs:
        vil = vtk.vtkIdList()
        vil.InsertNextId(hi)
        vil.InsertNextId(ti)
        polys.InsertNextCell(vil)
    return polys

def vcolors(values, cmap='coolwarm'):
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colormap = cm.get_cmap(cmap)
    mapped = colormap(values.astype(float))
    for r,g,b,a in 255*mapped:
        colors.InsertNextTuple3(r,g,b)
    return colors

def wires(network, alpha=0.5):
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vpoints(network))
    polydata.SetLines(vpolys(network))

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(polydata)

    actor = vtk.vtkActor()
    actor.GetProperty().SetOpacity(alpha)
    actor.SetMapper(mapper)

    return actor

def sphere(point, volume, alpha=None, color=None):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(*point)
    sphere.SetRadius(volume)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    actor = vtk.vtkActor()
    if alpha:
        actor.GetProperty().SetOpacity(alpha)
    if color:
        actor.GetProperty().SetColor(color)
    actor.SetMapper(mapper)

    return actor

def animate(iren, timeout, rate=1):
    iren.AddObserver('TimerEvent', timeout)
    timer = iren.CreateRepeatingTimer(rate)

def render(network, values=None, rate=1):
    ren = vtk.vtkRenderer()
    wires_actor = wires(network)
    ren.AddActor(wires_actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()

    view2d = np.atleast_2d(values) # cuts down on a lot of error-checking
    counter = {'value': 0} # find a better way to get around namespace issues
    
    def timeout(object, event):
        counter['value'] = (counter['value']+1)%len(view2d)
        wires_actor.GetMapper().GetInput().GetPointData().SetScalars(vcolors(view2d[counter['value']]))
        renWin.Render()
    
    if values is not None and len(view2d) > 0:
        animate(iren, timeout, rate)
    
    return iren