import numpy as np
import vtk
import minipnm as mini

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

def wires(network, alpha=0.2):
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

def animate(iren, ani_actors, volumes, fill_array, rate):

    def timeout(object, event, fills):
        for actor, volume in zip(ani_actors, volumes*fills):
            actor.GetMapper().GetInputConnection(0,0).GetProducer().SetRadius(volume)
        iren.GetRenderWindow().Render()

        c['n'] = (c['n']+1)%len(fill_array)
    
    c = {'n':0} # stupid hack because I'm bad at namespaces
    iren.AddObserver('TimerEvent', lambda o, e: timeout(o, e, fill_array[c['n']]))
    timer = iren.CreateRepeatingTimer(rate)

def render(network, volumes, fill_array, rate=1):
    ren = vtk.vtkRenderer()
    ren.AddActor(wires(network))

    for point, volume in zip(network.points, volumes):
        ren.AddActor(sphere(point, volume, alpha=0.2))

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()

    if fill_array.ndim==2:      
        ani_actors = []  
        for point, volume, fill in zip(network.points, volumes, fill_array[0]):
            actor = sphere(point, volume*fill, alpha=1, color=(0.2,0.4,0.5))
            ani_actors.append(actor)
            ren.AddActor(actor)
        animate(iren, ani_actors, volumes, fill_array, rate)
    
    iren.Start()

network = mini.Delaunay.random(1000)
network = network - ~network.boundary()
print network
x,y,z = network.coords
base = np.zeros_like(x)
gradients = np.random.rand(x.size)/100
N = int(1/gradients.mean())
fill_array = np.vstack([n*gradients for n in range(N)]).clip(0,1)

render(network, x/20+0.1, fill_array)