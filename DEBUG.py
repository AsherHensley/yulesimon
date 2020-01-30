#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from yulesimon import yulesimon
import matplotlib.pyplot as plt

# Setup
test = yulesimon()
prices = test.import_prices('MCD',5)
rt = test.log_returns(prices['Close'])

# Init
test.init_markov_chain(rt,rho=1)


plt.figure()
plt.plot(test.chain.xt)
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



