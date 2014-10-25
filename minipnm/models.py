import numpy as np
import simulations

'''
Houses comprehensive multi-layer models
'''

def BV(i):
    ''' fake butler-volmer. takes current returns voltage. per point. '''
    return 1.223 - 0.45 - 0.03 * np.log(i) # V

def takesarray(method):
    def wrapped(self, array):
        out = []
        for inputs in array:
            print "Calling {} @ {:0<6}".format(method.__name__, inputs)
            out.append( method(self, inputs) )
        return np.array(out).T
    return wrapped


class CatalystLayer(object):

    def __init__(self, network):
        ''' create all the layers of the model and the ways
        they will interface with each other '''
        self.network = network
        adj = network.adjacency_matrix
        volumes = self.network.volumes
        # layers
        self.current_layer = simulations.Simulation(adj)
        self.voltage_layer = simulations.Simulation(adj)
        self.water_layer = simulations.Invasion(adj, volumes)
        self.oxygen_layer = simulations.Diffusion(adj)

    def __getattr__(self, string):
        ''' ugly hack. if attribute not found, check network for it
            to improve, maybe network subclass in the first place?
            maybe metaclass?
        '''
        try:
            return super(CatalystLayer, self).__getattr__(string)
        except AttributeError:
            return getattr(self.network, string)

    def reset(self, current=None, voltage=None, saturation=None):
        self.current_layer.reset(current)
        self.voltage_layer.reset(voltage)
        self.water_layer.reset(saturation)

    @property
    def voltage(self):
        return self.voltage_layer.state.mean()

    @property
    def saturation(self):
        return 0

    @takesarray
    def ICC(self, current):
        '''
        takes current density demand
        returns steady-state supply voltage
        '''
        self.reset(current=current)
        t = 0
        while t < 1:
            t += 1
            print "@", t
        return BV(current*0.9), self.saturation

    @takesarray
    def ICV(self, voltage):
        self.reset(voltage=voltage)
        t = 0
        while t < 1:
            pass
        return voltage, self.saturation

    def render(self, scene):
        offset = self.bbox*[0,1,0]*1.5
        self.current_layer.render(self.points+offset, scene=scene)
        self.network.render(scene=scene)
        self.water_layer.render(self.points, scene=scene)
