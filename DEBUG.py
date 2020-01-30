#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Libraries
import yulesim as ys
import matplotlib.pyplot as plt

# Import Data
prices = ys.import_prices('MCD',5)
rt = ys.log_returns(prices['Close'])

# Init State
chain = ys.markovchain(rt)



plt.figure()
plt.plot(chain.xt)
plt.show()



"""
fig,ax =  plt.subplots(2,1)
ax[0].plot(prices['Close'])
ax[0].set_title('Closing Prices')
ax[0].grid(linestyle='--')
ax[1].plot(rt)
ax[1].set_title('Log Returns')
ax[1].grid(linestyle='--')
plt.show()
"""


"""
# Verify gamma pdf
import numpy as np
import scipy as sp

alpha = 2
beta = 1/10.0

rv = np.random.gamma(alpha,1/beta,10000)

plt.figure()
n = plt.hist(rv,bins=100,density=True)
pdf = beta**alpha / sp.special.gamma(alpha) * np.power(n[1],alpha-1) * np.exp(-beta * n[1])
plt.plot(n[1],pdf)
plt.grid(linestyle='--')
plt.show()

"""



