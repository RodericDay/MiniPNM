import vtk
import minipnm as mini

'''
Keypress j / Keypress t: toggle between joystick (position sensitive) and trackball (motion sensitive) styles. In joystick style, motion occurs continuously as long as a mouse button is pressed. In trackball style, motion occurs when the mouse button is pressed and the mouse pointer moves.
Keypress c / Keypress a: toggle between camera and actor modes. In camera mode, mouse events affect the camera position and focal point. In actor mode, mouse events affect the actor that is under the mouse pointer.
Button 1: rotate the camera around its focal point (if camera mode) or rotate the actor around its origin (if actor mode). The rotation is in the direction defined from the center of the renderer's viewport towards the mouse position. In joystick mode, the magnitude of the rotation is determined by the distance the mouse is from the center of the render window.
Button 2: pan the camera (if camera mode) or translate the actor (if actor mode). In joystick mode, the direction of pan or translation is from the center of the viewport towards the mouse position. In trackball mode, the direction of motion is the direction the mouse moves. (Note: with 2-button mice, pan is defined as <Shift>-Button 1.)
Button 3: zoom the camera (if camera mode) or scale the actor (if actor mode). Zoom in/increase scale if the mouse position is in the top half of the viewport; zoom out/decrease scale if the mouse position is in the bottom half. In joystick mode, the amount of zoom is controlled by the distance of the mouse pointer from the horizontal centerline of the window.
Keypress 3: toggle the render window into and out of stereo mode. By default, red-blue stereo pairs are created. Some systems support Crystal Eyes LCD stereo glasses; you have to invoke SetStereoTypeToCrystalEyes() on the rendering window.
Keypress e: exit the application.
Keypress f: fly to the picked point
Keypress p: perform a pick operation. The render window interactor has an internal instance of vtkCellPicker that it uses to pick.
Keypress r: reset the camera view along the current view direction. Centers the actors and moves the camera so that all actors are visible.
Keypress s: modify the representation of all actors so that they are surfaces.
Keypress u: invoke the user-defined function. Typically, this keypress will bring up an interactor that you can type commands in. Typing u calls UserCallBack() on the vtkRenderWindowInteractor, which invokes a vtkCommand::UserEvent. In other words, to define a user-defined callback, just add an observer to the vtkCommand::UserEvent on the vtkRenderWindowInteractor object.
Keypress w: modify the representation of all actors so that they are wireframe.
'''

network = mini.Radial(*mini.algorithms.poisson_disk_sampling(limit=3))
spheres, tubes = network.actors()

def handleIt(obj, event):
    ''' i need the id '''
    actor = picker.GetActor()
    if actor is None:
        exit()
    # print dir(picker), '\n'
    print actor.mapper, '\n'
    print str(picker).replace('\n',' ||')
    print picker.GetSelectionPoint(), repr(actor)

scene = mini.Scene()
picker = vtk.vtkCellPicker()

scene.iren.SetPicker(picker)
scene.add_actors([spheres, tubes])
scene.ren.ResetCamera()
scene.iren.Initialize()

labelMapper = vtk.vtkLabeledDataMapper()
labelMapper.SetInput(spheres.polydata)
labelActor = vtk.vtkActor2D()
labelActor.SetMapper(labelMapper)
scene.add_actors([labelActor])

picker.AddObserver("EndPickEvent", handleIt)
picker.Pick(141.0, 132.0, 0.0, scene.ren)

scene.play()
