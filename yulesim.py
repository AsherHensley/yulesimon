#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Dependencies
import pandas_datareader.data as web
import datetime
import numpy as np
import scipy as sp

class markovchain:
    
    def __init__(self,data,a=2,b=2,c=5,d=5,rho=2):
        
        # Init Markov Chain
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.rho = rho
        self.xt = np.zeros(np.size(data))
        self.lambdas = np.array(sample_gamma_posterior(data[0],0,self.c,self.d))
        
        # Sample Yule-Simon Partitions
        self.__init_partitions(data)
            
    def __init_partitions(self,data):
        
        # Setup
        nsamp = np.size(data)
        self.xt[0] = 1
        counter = 1.0
        state = 1.0
        
        # Run
        for kk in range(1,nsamp):
            
            # Transition Weights
            w = self.__forward_weights(data[kk],counter)
            
            # Sample Regime Change
            u = np.random.uniform()
            regime_change = u < w[1]
            
            # Update State
            if regime_change==True:
                state = state + 1
                new_lambda = sample_gamma_posterior(data[kk],0,self.c,self.d)
                self.lambdas = np.append(self.lambdas,new_lambda)
                counter = 1.0
            else:
                counter = counter + 1  
            
            # Update Partition
            self.xt[kk] = state
    
    
    def __forward_weights(self,data,counter):
    
        # Prior
        p = np.zeros(2)
        p[0] = counter / (counter + self.rho)
        p[1] = self.rho / (counter + self.rho)
        
        # Likelihoods
        L = np.zeros(2)
        L[0] = gaussian_pdf(data,0,self.lambdas[-1])
        L[1] = student_pdf(data,0,self.c / self.d,2 * self.c)
        
        # Compute Weights
        w = L * p
        return w / np.sum(w)
            
          
def import_prices(symbol,n_years):
    
    # Setup Historical Window
    end = datetime.datetime.today()
    start = datetime.date(end.year-n_years,1,1)
    
    # Attempt to Fetch Price Data
    prices = []
    try:
        prices = web.DataReader(symbol, 'yahoo', start, end)
    except:
        print("No information for ticker # and symbol: " + symbol)
              
    return prices
     

def log_returns(closing_prices):
    return np.diff(np.log(closing_prices))
     
       
def gaussian_pdf(data,mu,precision):        
    return np.sqrt(precision / (2.0 * np.pi)) * np.exp(-0.5 * precision * (data - mu)**2)


def student_pdf(data,mu,precision,dof):   
    Z = sp.special.gamma(dof / 2.0 + 0.5)
    Z = Z / sp.special.gamma(dof / 2.0)
    Z = Z * np.sqrt(precision / (np.pi * dof));        
    return Z * (1 + precision / dof * (data - mu)**2)**(-dof/2.0 - 0.5);

    
def sample_gamma_posterior(data,mu,c0,d0):    
    N = np.size(data)
    cN = c0 + 0.5 * N
    dN = d0 + 0.5 * np.sum(np.square(data - mu))    
    return np.random.gamma(cN,1/dN,1)
        
        
        
        
        
        
        
        
        
        
        
        
        