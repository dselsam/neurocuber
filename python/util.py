import numpy as np

def npsoftmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def set_pyro_config():
    import Pyro4
    Pyro4.config.SERVERTYPE            = "multiplex"
    Pyro4.config.SERIALIZERS_ACCEPTED  = ["marshal", "json", "serpent", "pickle"]
    Pyro4.config.SERIALIZER            = "pickle"
    Pyro4.config.COMPRESSION           = True
    Pyro4.config.DETAILED_TRACEBACK    = True
