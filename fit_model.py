#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import lmfit
import models

def fit(x, y, model, params):
    def residuals(params, x, y):
        return y - model(x,params)

    # Initialize lmfit:
    prms = lmfit.Parameters()
    for p in params.keys():
        prms.add(p, value = params[p]['guess'], vary = True)

    # Run the fit:    
    result = lmfit.minimize(residuals,prms,args=(x,y))

    # Return results:
    if result.success:
        return result.params,model(x,result.params)
    else:
        print 'Fit failed. Returning initial parameters.'
        return params,y

# Read data:
x,y = np.loadtxt('datasets/example_sigmoid.dat',unpack=True)

# Define parameters:
params = {}
params['c'] = {}
params['c']['guess'] = 200.0
params['a'] = {}
params['a']['guess'] = 50.0
params['x0'] = {}
params['x0']['guess'] = 4.0
params['b'] = {}
params['b']['guess'] = 10.0

# Fit:
out_params, y_pred = fit(x, y, models.sigmoid, params)

# Print results:
print '\t #### Results ####'
for p in out_params.keys():
        print '\t > Parameter '+p+' final value: ',out_params[p].value
# Plot fit:
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.plot(x,y,'.')
plt.plot(x,y_pred,'-')
plt.show()
