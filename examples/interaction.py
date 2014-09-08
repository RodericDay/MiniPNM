import random
from vtk import *
import minipnm as mini

# some source stuff
g = (1 for i in xrange(100000))
c, r = mini.algorithms.poisson_disk_sampling(r=g, bbox=[10,10,10])

network = mini.Radial(c, r)
adj = network.adjacency_matrix
shells, tubes = network.actors()
contents = mini.graphics.Spheres(c, [0 for _ in r], color=(0,0,1))
glyph3D = contents.glyph3D
polydata = contents.polydata
scalars = polydata.GetPointData().GetScalars()

ren = vtkRenderer()
ren.AddActor(shells)
ren.AddActor(tubes)
ren.AddActor(contents)

renWin = vtkRenderWindow()
renWin.AddRenderer(ren)

def add_bucket(i, amt=0.1):
    neighbors = adj.row[adj.col==i]
    f = [scalars.GetValue(j)/2. for j in range(len(r))]
    f[i] += amt
    surplus = max(0, f[i] - r[i])
    f[i] = min(r[i], f[i])
    [scalars.SetValue(j, f[j]*2.) for j in range(len(r))]
    polydata.Modified()

    if all(a==b for a,b in zip(r,f)):
        print 'full!'
        return

    if surplus:
        add_bucket(random.choice(neighbors), surplus)

def handlePick(obj, event):
    ''' get the id of the source point '''
    if not picker.GetActor() is shells:
        return
    
    pointIds = glyph3D.GetOutput().GetPointData().GetArray("InputPointIds")
    selectedId = int(pointIds.GetTuple1(picker.GetPointId()))

    add_bucket(selectedId)
    
    renWin.Render()

picker = vtkCellPicker()
picker.AddObserver("EndPickEvent", handlePick)

iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.GetInteractorStyle().SetCurrentRenderer(ren)
iren.SetPicker(picker)
iren.Start()
