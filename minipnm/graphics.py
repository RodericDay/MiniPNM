from __future__ import division
import numpy as np
import warnings

def render(network, values=None):
    import vtk

    points = vtk.vtkPoints()
    for x,y,z in network.points:
        points.InsertNextPoint(x, y, z)

    polys = vtk.vtkCellArray()
    for hi, ti in network.pairs:
        vil = vtk.vtkIdList()
        vil.InsertNextId(hi)
        vil.InsertNextId(ti)
        polys.InsertNextCell(vil)

    # process value array
    if values is not None:
        values = np.array(values).flatten()
        values = np.subtract(values, values.min())
        values = np.true_divide(values, values.max())
    else:
        values = []

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    for v in values:
        r = 255*(v)
        g = 100
        b = 255*(1-v)
        colors.InsertNextTuple3(r,g,b)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(polys)
    if colors.GetNumberOfTuples() == network.size[0]:
        polydata.GetPointData().SetScalars(colors)
    elif colors.GetNumberOfTuples() > 0:
        raise Exception("Mismatch: {} points, {} scalars".format(
                        network.size[0], colors.GetNumberOfTuples()))

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    iren.Initialize()
    renWin.Render()
    iren.Start()

from xml.etree import ElementTree as ET

TEMPLATE = '''
<?xml version="1.0" ?>
<VTKFile byte_order="LittleEndian" type="PolyData" version="0.1">
    <PolyData>
        <Piece NumberOfLines="0" NumberOfPoints="0">
            <Points>
            </Points>
            <Lines>
            </Lines>
            <PointData>
            </PointData>
        </Piece>
    </PolyData>
</VTKFile>
'''.strip()

def _append_array(parent, name, array, n=1):
    dtype_map = {
        'int8'   : 'Int8',
        'int16'  : 'Int16',
        'int32'  : 'Int32',
        'int64'  : 'Int64',
        'uint8'  : 'UInt8',
        'uint16' : 'UInt16',
        'uint32' : 'UInt32',
        'uint64' : 'UInt64',
        'float32': 'Float32',
        'float64': 'Float64',
        'str'    : 'String',
    }
    element = ET.SubElement(parent, 'DataArray')
    element.set("Name", name)
    element.set("NumberOfComponents", str(n))
    element.set("type", dtype_map[str(array.dtype)])
    element.text = '\t'.join(map(str,array.ravel()))

def _element_to_array(element, n=1):
    string = element.text
    dtype = element.get("type")
    array = np.fromstring(string, sep='\t')
    array = array.astype(dtype)
    if n is not 1:
        array = array.reshape(array.size//n, n)
    return array

def save_vtp(network, filename):
    ''' takes in a network
    '''
    root = ET.fromstring(TEMPLATE)

    network = network.copy() # ensure we do not affect the original

    num_points = len(network.points)
    num_throats = len(network.pairs)
    
    piece_node = root.find('PolyData').find('Piece')
    piece_node.set("NumberOfPoints", str(num_points))
    piece_node.set("NumberOfLines", str(num_throats))

    points_node = piece_node.find('Points')
    _append_array(points_node, "coords", network.coords.ravel('F'), n=3)

    lines_node = piece_node.find('Lines')
    _append_array(lines_node, "connectivity", network.pairs)
    # insert a break?
    _append_array(lines_node, "offsets", 2*np.arange(len(network.pairs))+2)

    point_data_node = piece_node.find('PointData')
    for key, array in sorted(network.items()):
        if array.size != network.size[0]: continue
        _append_array(point_data_node, key, array)

    tree = ET.ElementTree(root)
    tree.write(filename)

def load_vtp(filename):
    network = {}
    tree = ET.parse(filename)
    piece_node = tree.find('PolyData').find('Piece')

    # extract connectivity
    conn_element = piece_node.find('Lines').find('DataArray')
    array = _element_to_array(conn_element, 2)
    network['heads'], network['tails'] = array.T

    for element in piece_node.find('PointData').iter('DataArray'):

        key = element.get('Name')
        array = _element_to_array(element)
        network[key] = array

    return network