import numpy as np


def SphericalToCartesian(R,theta,phi):
    X = R*np.cos(theta)
    Y = R*np.sin(theta)*np.cos(phi)
    Z = R*np.sin(theta)*np.sin(phi)
   
    return X,Y,Z

def listify(arg):
    if none_iterable(arg):
        return [arg]
    else:
        return arg
    
def none_iterable(*args):
    """
    return true if none of the arguments are either lists or tuples
    """
    return all([not isinstance(arg, list) and not isinstance(arg, tuple) and  not isinstance(arg, np.ndarray) for arg in args])   
