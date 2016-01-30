import numpy as np

def line(x,params):
    return params['a'].value*x + params['b'].value

def quadratic(x,params):
    return (params['a'].value)*x**2 + (params['b'].value)*x + params['c'].value

def sine(x,params):
    return (params['A'].value)*np.sin(params['P'].value*x + params['phi'].value)

def sigmoid(x,params):
    """
    This function is a generalized sigmoid, of the form:

               a + (1/(1 + exp(-b(x-x0))))*c

    With:

        a :     Controls the base level of the sigmoid (minimum value of sigmoid 
                is equal to a).

        c :     Controls the upper level of the sigmoid (maximum value of sigmoid 
                is equal to a+c)

        x0:     Controls where the change is made between the minimum and maximum
                value of the sigmoid.

        b:      Controls how smooth and slow the change between minimum and maximum 
                is made. Large values imply sharp changes, small values small changes.

    """
    return params['a'].value + (1./(1. + np.exp(-(x-params['x0'].value)*params['b'].value)))*params['c'].value
