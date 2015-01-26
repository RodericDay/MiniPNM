import numpy as np
import minipnm as mini

def test_scene(N=10):
    try:
        import vtk
    except ImportError:
        return
    scene = mini.Scene()
    network = mini.Cubic([10,10])
    # draw a simple wired cubic going from red to white to blue
    script = [network.diagonals.data[0]*i for i in range(N)]
    wires = mini.graphics.Wires(network.points, network.pairs, script)
    scene.add_actors([wires])
    # draw some random green popping bubbles
    network['x'] += 11
    base = np.random.rand(network.order)/4+0.25
    radii = [(base+0.5/N*i)%0.5 for i in range(N)]
    spheres = mini.graphics.Spheres(network.points, radii, color=(0,1,0), alpha=0.5)
    scene.add_actors([spheres])
    # draw a tube cross
    network['x'] -= 11
    network['y'] -= 11
    tubes = mini.graphics.Tubes([network.centroid]*2,[(10,0,0),(0,10,0)],[1,1])    
    scene.add_actors([tubes])
    return scene

if __name__ == '__main__':
    scene = test_scene(30)
    scene.play()
