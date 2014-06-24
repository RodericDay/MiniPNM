'''
hypothetical script
'''
# define tons of physical constants
# or.... we have a database and defaults
#~~

gdl = mini.GDL() # mini.Bridson(big_pore_pdf, dims)
acl = mini.ACL()
ml = mini.ML()
ccl = mini.CCL()

mini.merge_rule = not_delaunay # default: delaunay
fuel_cell = gdl + acl + ml + ccl + gdl # or | ?
fuel_cell.add_boundaries('lr')
x,y,z = fuel_cell.coords
proton_source = (x==x.min()) & (y>np.percentile(y, 50))
air_source = (x==x.max()) & (y>np.percentile(y,50))

# simulation
def simulation(fuel_cell):
    state = {'bunch_of_arrays_describing_state'}
    steady_state = False
    while not steady_state:
        # these could be defined right above
        state = mini.resolve_chemical_reactions(fuel_cell, state)
        state = mini.resolve_fluid_interactions(fuel_cell, state)
    return state_histories_over_time

# visualization
scene = mini.Scene()
for element in fuel_cell:
    scene.add_proper_feature(element)
scene.play()

'''
maybe eventually build multi-script programs ie:
- one generator that saves vtp file, run only if file not existent
- one simulator that loads file, saves another file with more output
- one visualizer that reads the last one and properly animates everything