#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Dependencies
import pandas_datareader.data as web
import datetime
import numpy as np
import scipy as sp

class history:
    
    def __init__(self):
        
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.rho = 0
        self.xt = []
        self.lambdas = []


class markovChain(history):
    
    def __init__(self):
        
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.rho = 0
        self.xt = []
        self.lambdas = []
        self.history = history()

        
class yulesimon(markovChain):
    
    def __init__(self):

        self.chain = markovChain()
            
    
    def import_prices(self,symbol,n_years):
        
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
         
    
    def log_returns(self,closing_prices):
        
        return np.diff(np.log(closing_prices))
                  
    
    def init_markov_chain(self,rt,a=2,b=2,c=5,d=5,rho=2):
        
        # Init Hyperparamerers
        self.chain.a = a
        self.chain.b = b
        self.chain.c = c
        self.chain.d = d
        self.chain.rho = rho
        
        # Init Yule-Simon State Variable
        nsamp = np.size(rt)
        self.chain.xt = np.zeros(nsamp)
        self.chain.xt[0] = 1
        
        # Init Lambdas
        self.chain.lambdas = np.array(self.sample_gamma_posterior(rt[0],0))
        
        # Sample Yule-Simon Partitions
        counter = 1.0
        state = 1.0
        for kk in range(1,nsamp):
            
            # Transition Weights
            w = self.__forward_weights(rt[kk],counter)
            
            # Sample Regime Change
            u = np.random.uniform()
            regime_change = u < w[1]
            if regime_change==True:
                state = state + 1
                new_lambda = self.sample_gamma_posterior(rt[kk],0)
                self.chain.lambdas = np.append(self.chain.lambdas,new_lambda)
                counter = 1.0
            else:
                counter = counter + 1  
            
            # Update State
            self.chain.xt[kk] = state
                
        
    def __forward_weights(self,rt,counter):
        
        # Prior
        p = np.zeros(2)
        p[0] = counter / (counter + self.chain.rho)
        p[1] = self.chain.rho / (counter + self.chain.rho)
        
        # Likelihoods
        L = np.zeros(2)
        L[0] = self.gaussian_pdf(rt,0,self.chain.lambdas[-1])
        L[1] = self.student_pdf(rt,0,self.chain.c / self.chain.d,2 * self.chain.c)
        
        # Return Weights
        w = L * p
        return w / np.sum(w)
        
    
    def gaussian_pdf(self,data,mu,precision):
        
        return np.sqrt(precision / (2.0 * np.pi)) * np.exp(-0.5 * precision * (data - mu)**2)


    def student_pdf(self,data,mu,precision,dof):
        
        Z = sp.special.gamma(dof / 2.0 + 0.5)
        Z = Z / sp.special.gamma(dof / 2.0)
        Z = Z * np.sqrt(precision / (np.pi * dof));
        
        return Z * (1 + precision / dof * (data - mu)**2)**(-dof/2.0 - 0.5);
    
    
    def sample_gamma_posterior(self,data,mu):
        
        N = np.size(data)
        cN = self.chain.c + 0.5 * N
        dN = self.chain.d + 0.5 * np.sum(np.square(data - mu))
        
        return np.random.gamma(cN,1/dN,1)
        
        
        
        
        
        
        
        
        
        
        
        
        