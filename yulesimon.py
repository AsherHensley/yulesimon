#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma, beta
from scipy.stats import normaltest, multivariate_normal
import pandas_datareader.data as web
import datetime

#-----------------------------------------------------------------------------
# GetYahooFeed
#-----------------------------------------------------------------------------
def GetYahooFeed(symbol,n_years):
    
    # Setup Historical Window
    end = datetime.datetime.today()
    start = datetime.date(end.year-n_years, end.month, end.day)
    
    # Attempt to Fetch Price Data
    log_returns = []
    try:
        df = web.DataReader(symbol, 'yahoo', start, end)
        closing_prices = df.Close.to_numpy()
        log_returns = np.diff(np.log(closing_prices))
        dates = df.index.copy()
           
    except:
        print("No information for ticker # and symbol: " + symbol)
              
    return closing_prices, log_returns, dates
     
#-----------------------------------------------------------------------------
# Simulate
#-----------------------------------------------------------------------------
def Simulate(N=500, alpha=1.0, seed=13, a=1.0, b=1.0):
    
    # Setup
    x = np.zeros(N)
    y = np.zeros(N)
    counter = 1
    np.random.seed(seed)
    
    # Sample Partitions
    for idx in range(1,N):
        
        u = np.random.uniform()
        if u < (counter / (counter + alpha)):
            counter += 1
            x[idx] = x[idx-1]
        else:
            counter = 1
            x[idx] = x[idx-1]+1
          
    # Sample Precisions
    lambdas = np.random.gamma(a, 1/b, int(x[idx]+1)) 
    
    # Sample Observations
    for state in range(len(lambdas)):
        mask = x==state
        y[mask] = np.random.normal(0.0, 1/np.sqrt(lambdas[state]), sum(mask))
      
    return x, y, lambdas

#-----------------------------------------------------------------------------
# Gaussian
#-----------------------------------------------------------------------------
def Gaussian(data,mu,precision):        
    return np.sqrt(precision / (2.0 * np.pi)) * np.exp(-0.5 * precision * (data - mu)**2)

#-----------------------------------------------------------------------------
# Student
#-----------------------------------------------------------------------------
def Student(data,mu,precision,dof):   
    Z = gamma(dof / 2.0 + 0.5)
    Z = Z / gamma(dof / 2.0)
    Z = Z * np.sqrt(precision / (np.pi * dof));        
    return Z * (1 + precision / dof * (data - mu)**2)**(-dof/2.0 - 0.5);

#-----------------------------------------------------------------------------
# ExpectedValue
#-----------------------------------------------------------------------------
def ExpectedValue(data,burnin,downsample): 
    return np.mean(data[:,int(burnin):int(downsample):-1],axis=1)

#-----------------------------------------------------------------------------
# TimeSeries
#-----------------------------------------------------------------------------
class TimeSeries():

    #-------------------------------------------------------------------------
    # __init__
    #-------------------------------------------------------------------------
    def __init__(self, data, alpha=5.0, a0=1.0, b0=1.0, Q = 1e-9, init='uniform', init_segments=50):
        self.data = data
        self.nsamp = np.size(self.data)
        self.alpha = alpha
        self.a0 = a0
        self.b0 = b0 # * np.var(data)
        self.lambdas = np.array(self.__gamma_posterior(data[0]))
        self.x = np.zeros(data.shape)
        self.mu = np.zeros(data.shape)
        self.Q = Q

        if (init=='uniform'):
            self.__init_partitions_uniform(init_segments)
        elif (init=='prior'):
            self.__init_partitions()
        else:
            raise ValueError('Unknown Initialization Type: ' + init)
            
        self.__kalman_filter()
        
    #-------------------------------------------------------------------------
    # __init_partitions_uniform
    #-------------------------------------------------------------------------
    def __init_partitions_uniform(self,nsegments):
        
        # Setup
        segment_size = round(self.nsamp/nsegments)
        self.x[0] = 0
        counter = 1.0
        state = 0
        
        # Run
        for kk in range(1,self.nsamp):
            
            if counter >= segment_size:
                counter = 1
                state += 1
                new_lambda = self.__gamma_posterior(self.data[kk])
                self.lambdas = np.append(self.lambdas, new_lambda)
                
            self.x[kk] = state
            counter += 1
        
    #-------------------------------------------------------------------------
    # __init_partitions
    #-------------------------------------------------------------------------
    def __init_partitions(self):
    
        # Setup
        self.x[0] = 0
        counter = 1.0
        state = 0
        
        # Run
        for kk in range(1,self.nsamp):
            
            # Transition Weights
            w = self.__forward_weights(self.data[kk],counter)
            
            # Sample Regime Change
            u = np.random.uniform()
            regime_change = u > w[0]
            
            # Update State
            if regime_change==True:
                state = state + 1
                new_lambda = self.__gamma_posterior(self.data[kk])
                self.lambdas = np.append(self.lambdas, new_lambda)
                counter = 1.0
            else:
                counter += 1  
            
            # Update Partition
            self.x[kk] = state
     
    #-------------------------------------------------------------------------
    # __forward_weights
    #-------------------------------------------------------------------------
    def __forward_weights(self,yt,counter):
    
        # Prior
        p = np.zeros(2)
        p[0] = counter / (counter + self.alpha)
        p[1] = self.alpha / (counter + self.alpha)
        
        # Likelihoods
        L = np.zeros(2)
        L[0] = Gaussian(yt,0,self.lambdas[-1])
        L[1] = Student(yt,0,self.a0 / self.b0,2 * self.a0)
        
        # Compute Weights
        w = L * p
        return w / np.sum(w)
    
    #-------------------------------------------------------------------------
    # __gamma_posterior
    #-------------------------------------------------------------------------
    def __gamma_posterior(self, data, a0=-1, b0=-1):   
       
        mu = 0.0
        N = np.size(data)
        
        if a0==-1:
            a0 = self.a0

        if b0==-1:
            b0 = self.b0
        
        aN = a0 + 0.5 * N
        bN = b0 + 0.5 * np.sum(np.square(data - mu)) 
        
        return np.random.gamma(aN,1/bN,1)
    
    #-------------------------------------------------------------------------
    # step
    #-------------------------------------------------------------------------
    def step(self, N=100):
        
        history = self.__init_history(N)

        for step in range(N):
            self.__sample_partitions()
            self.__sample_lambdas()
            self.__sample_alpha()
            self.__sample_gamma_hyperparameters()
            self.__kalman_filter() 
            self.__update_history(history, step+1)
            
            if (step % round(N/100)) == 0:
                print(".",end='')
                
        return history
    
    #-------------------------------------------------------------------------
    # __sample_gamma_hyperparameters
    #-------------------------------------------------------------------------
    def __sample_gamma_hyperparameters(self):
        
        # Initial Condition
        X = np.array([self.a0,self.b0])
        
        # Proposal Distribution Sigma
        sigma2_a = (2*0.50) ** 2
        sigma2_b = (2*0.25) ** 2
        
        # Proposal Distribution
        Q = lambda z,mu,Sig: multivariate_normal.pdf(z,mean=mu,cov=Sig) / multivariate_normal.cdf([0,0],mean=-mu,cov=Sig)
            
        # Target Distribution
        logP = lambda z,x: np.sum(np.log(x**(z[0]-1) * np.exp(-z[1]*x) / (gamma(z[0]) * z[1]**(-z[0]))))
        
        # Step
        step_fail = True
        for ii in range(10):
            a0_prop = np.random.normal(self.a0,sigma2_a)
            b0_prop = np.random.normal(self.b0,sigma2_b)
            if (a0_prop >= 0) & (b0_prop >= 0):
                Y = np.array([a0_prop,b0_prop])
                step_fail = False
                break
        
        if step_fail == False:
            Sigma = np.array([[sigma2_a,0],[0,sigma2_b]])
            ratio = np.exp(logP(Y,self.lambdas)-logP(X,self.lambdas)) * Q(X,Y,Sigma) / Q(Y,X,Sigma)
            A = min(1,ratio)
            if np.random.uniform() <= A:
                self.a0 = a0_prop
                self.b0 = b0_prop
    
    #-------------------------------------------------------------------------
    # __kalman_filter
    #-------------------------------------------------------------------------
    def __kalman_filter(self):

        R = 1/self.lambdas[self.x.astype('int')]
        V = np.zeros(np.size(R))
        P = np.zeros(np.size(R))
        mu = np.zeros(np.size(R))
        
        # Initial Value
        V0 = 1/(10*self.lambdas[0])
        K = V0 / (V0 + R[0])
        mu[0] = 0.0 
        V[0] = (1-K) * V0
        P[0] = V[0] + self.Q
        
        # Forward Recursion
        for ii in range(1,self.nsamp):
            K = P[ii-1] / (P[ii-1] + R[ii])
            mu[ii] = mu[ii-1] + K * (self.data[ii] - mu[ii-1])
            V[ii] = (1-K) * P[ii-1]
            P[ii] = V[ii] + self.Q
            
        # Backward Recursion
        self.mu[-1] = mu[-1]
        for ii in range(self.nsamp-2,-1,-1):
            J = V[ii]/P[ii]
            self.mu[ii] = mu[ii] + J * (self.mu[ii+1]-mu[ii])
            
    #-------------------------------------------------------------------------
    # __sample_alpha
    #-------------------------------------------------------------------------  
    def __sample_alpha(self):
        
        n = self.__get_partitions_counts()
        N = len(n)
        w = -np.log(np.random.beta(self.alpha+1, n))
        self.alpha = np.random.gamma(1+N,1/(1+sum(w)) )

    #-------------------------------------------------------------------------
    # __init_history
    #-------------------------------------------------------------------------  
    def __init_history(self, N):

        class struct():
            pass    
        
        history = struct()
        history.log_likelihood = np.zeros(N+1)
        history.std_deviation = np.zeros((self.nsamp, N+1))
        history.boundaries = np.zeros((self.nsamp, N+1))
        history.mean = np.zeros((self.nsamp, N+1))
        history.alpha = np.zeros(N+1)
        history.pvalue = np.zeros(N+1)
        history.process_noise = np.zeros(N+1)
        history.hyperparameter_a0 = np.zeros(N+1)
        history.hyperparameter_b0 = np.zeros(N+1)
        
        self.__update_history(history, 0)

        return history
    
    #-------------------------------------------------------------------------
    # __update_history
    #-------------------------------------------------------------------------  
    def __update_history(self, history, idx):
        
        history.log_likelihood[idx] = self.__log_likelihood()
        history.std_deviation[:,idx] = 1/np.sqrt(self.lambdas[self.x.astype('int')])
        history.boundaries[:,idx] = np.append(0,np.diff(self.x))
        history.mean[:,idx] = self.mu
        history.alpha[idx] = self.alpha
        history.process_noise [idx] = self.Q
        history.hyperparameter_a0[idx] = self.a0
        history.hyperparameter_b0[idx] = self.b0
        
        # Goodness of fit
        h,p = normaltest((self.data - self.mu)*np.sqrt(self.lambdas[self.x.astype('int')]))
        history.pvalue[idx] = p  
            
    #-------------------------------------------------------------------------
    # __log_likelihood
    #-------------------------------------------------------------------------      
    def __log_likelihood(self):
        
        lambdas = self.lambdas[self.x.astype('int')]
        L = np.sum(np.log(Gaussian(self.data-self.mu, 0, lambdas)))
        
        n = self.__get_partitions_counts()
        L += np.sum(np.log(self.alpha * beta(n, self.alpha+1)))
        
        return L
            
    #-------------------------------------------------------------------------
    # __sample_partitions
    #-------------------------------------------------------------------------      
    def __sample_partitions(self):
    
        for kk in range(self.nsamp):
            boundary = self.__get_boundary_type(kk)
            
            if boundary != "None":
                self.__update_markov_chain(kk, boundary)
                
    #-------------------------------------------------------------------------
    # __sample_lambdas
    #-------------------------------------------------------------------------      
    def __sample_lambdas(self):
        
        N = int(max(self.x)+1)
        for ii in range(N):
            yt = self.data[self.x==ii] - self.mu[self.x==ii]
            self.lambdas[ii] = self.__gamma_posterior(yt)
                 
    #-------------------------------------------------------------------------
    # __get_boundary_type
    #------------------------------------------------------------------------- 
    def __get_boundary_type(self, idx):
        
        xt = self.x[idx]
        
        if (idx==0):
            if self.x[0]==self.x[1]:
                boundary = "FirstOpen"  
            else:    
                boundary = "FirstClosed"
            
        elif (idx==(len(self.data)-1)):
            if xt==self.x[idx-1]:
                boundary = "LastOpen"  
            else:    
                boundary = "LastClosed"
        
        elif (self.x[idx-1]!=xt) & (self.x[idx+1]==xt):
            boundary = "Left"
            
        elif (self.x[idx-1]==xt) & (self.x[idx+1]!=xt): 
            boundary = "Right"
            
        elif (self.x[idx-1]!=xt) & (self.x[idx+1]!=xt): 
            boundary = "Double"
            
        else:
            boundary = "None"
            
        return boundary 
    
    #-------------------------------------------------------------------------
    # __update_markov_chain
    #------------------------------------------------------------------------- 
    def __update_markov_chain(self, idx, boundary):
        
        n = self.__get_partitions_counts()
        yt = self.data[idx] - self.mu[idx]
        xt = int(self.x[idx])
        wnew = self.alpha / (1+self.alpha) * Student(yt,0,self.a0/self.b0,2*self.b0) 
        
        if boundary=="FirstOpen":
            w0 = (n[0]-1) / (n[0]+self.alpha) * Gaussian(yt,0,self.lambdas[0])
            w = np.array([w0,wnew])
            self.__sample_first_open(w, yt)
            
        elif boundary=="FirstClosed":  
            w0 = n[1] / (n[1]+self.alpha+1) * Gaussian(yt,0,self.lambdas[1])
            w = np.array([w0,wnew])
            self.__sample_first_closed(w, yt)
              
        elif boundary=="LastOpen":
            w0 = (n[-1]-1) / (n[-1]+self.alpha) * Gaussian(yt,0,self.lambdas[-1])
            w = np.array([w0,wnew])
            self.__sample_last_open(w, yt)
            
        elif boundary=="LastClosed":  
            w0 = n[xt-1] / (n[xt-1]+self.alpha+1) * Gaussian(yt,0,self.lambdas[-2])
            w = np.array([w0,wnew])
            self.__sample_last_closed(w, yt)
        
        elif boundary=="Left":
            w0 = (n[xt]-1) / (n[xt]+self.alpha) * Gaussian(yt,0,self.lambdas[xt])
            w1 = n[xt-1] / (n[xt-1]+self.alpha+1) * Gaussian(yt,0,self.lambdas[xt-1])
            w = np.array([w0,w1,wnew])
            self.__sample_left_boundary(w, idx)
  
        elif boundary=="Right":
            w0 = (n[xt]-1) / (n[xt]+self.alpha) * Gaussian(yt,0,self.lambdas[xt])
            w1 = n[xt+1] / (n[xt+1]+self.alpha+1) * Gaussian(yt,0,self.lambdas[xt+1])
            w = np.array([w0,w1,wnew])
            self.__sample_right_boundary(w, idx)

        elif boundary=="Double":
            w0 = n[xt-1] / (n[xt-1]+self.alpha+1) * Gaussian(yt,0,self.lambdas[xt-1])
            w1 = n[xt+1] / (n[xt+1]+self.alpha+1) * Gaussian(yt,0,self.lambdas[xt+1])
            w = np.array([w0,w1,wnew])
            self.__sample_double_boundary(w, idx)

        else:
            raise ValueError('Unknown Boundary Type: ' + boundary)
        
        return w
    
    #-------------------------------------------------------------------------
    # __sample_first_open
    #------------------------------------------------------------------------- 
    def __sample_first_open(self, w, yt):
        
        u = self.__sample_discrete(w)
        if u==0: 
            # No Change
            self.x[0] = 0
            
        else:
            # Add New Partition
            self.x = self.x + 1
            self.x[0] = 0
            new_lambda = self.__gamma_posterior(yt)
            self.lambdas = np.append(new_lambda, self.lambdas)
     
    #-------------------------------------------------------------------------
    # __sample_first_closed
    #------------------------------------------------------------------------- 
    def __sample_first_closed(self, w, yt):
        
        u = self.__sample_discrete(w)
        if u==0:
            # Merge to Right Partitions
            self.x[0] = 1
            self.x = self.x - 1
            self.lambdas = self.lambdas[1:]
            
        else:
            # Add New Partition
            self.x[0] = 0
            new_lambda = self.__gamma_posterior(yt)
            self.lambdas[0] = new_lambda
            
    #-------------------------------------------------------------------------
    # __sample_last_open
    #------------------------------------------------------------------------- 
    def __sample_last_open(self, w, yt):
        
        u = self.__sample_discrete(w)
        if u==0:
            # No Change
            self.x[-1] = self.x[-1]
               
        else:
            # Add New Partition
            self.x[-1] = self.x[-1]+1
            new_lambda = self.__gamma_posterior(yt)
            self.lambdas = np.append(self.lambdas, new_lambda)
        
    #-------------------------------------------------------------------------
    # __sample_last_closed
    #------------------------------------------------------------------------- 
    def __sample_last_closed(self, w, yt):
        
        u = self.__sample_discrete(w)
        if u==0:
            # Merge to Left Partition
            self.x[-1] = self.x[-1] - 1
            self.lambdas = self.lambdas[:-1]
            
        else:
            # Add New Partition 
            self.x[-1] = self.x[-1]
            self.lambdas[-1] = self.__gamma_posterior(yt)
        
    #-------------------------------------------------------------------------
    # __sample_left_boundary
    #------------------------------------------------------------------------- 
    def __sample_left_boundary(self, w, idx):
    
        u = self.__sample_discrete(w)
        yt = self.data[idx] - self.mu[idx]
        xt = int(self.x[idx])
        
        if u==0:
            # No Change
            self.x[idx] = self.x[idx]
        
        elif u==1:
            # Merge to Left Partition
            self.x[idx] = self.x[idx] - 1
            
        else:
            # Add New Partition
            self.x[(idx+1):] = self.x[(idx+1):] + 1
            new_lambda = self.__gamma_posterior(yt)
            tmp = np.append(self.lambdas[:xt], new_lambda)
            self.lambdas = np.append(tmp, self.lambdas[xt:])
            
    #-------------------------------------------------------------------------
    # __sample_right_boundary
    #------------------------------------------------------------------------- 
    def __sample_right_boundary(self, w, idx):
        
        u = self.__sample_discrete(w)
        yt = self.data[idx] - self.mu[idx]
        xt = int(self.x[idx])
        
        if u==0:
            # No Change
            self.x[idx] = self.x[idx]
        
        elif u==1:
            # Merge to Right Partition
            self.x[idx] = self.x[idx] + 1 
            
        else:
            # Add New Partition
            self.x[idx] = self.x[idx] + 1
            self.x[(idx+1):] = self.x[(idx+1):] + 1
            new_lambda = self.__gamma_posterior(yt)
            tmp = np.append(self.lambdas[:(xt+1)], new_lambda)
            self.lambdas = np.append(tmp, self.lambdas[(xt+1):])
            
    #-------------------------------------------------------------------------
    # __sample_double_boundary
    #------------------------------------------------------------------------- 
    def __sample_double_boundary(self, w, idx):
        
        u = self.__sample_discrete(w)
        yt = self.data[idx] - self.mu[idx]
        xt = int(self.x[idx])
        
        if u==0:
            # Merge to Left Partition
            self.x[idx:] = self.x[idx:] - 1
            self.lambdas = np.delete(self.lambdas,xt)
            
        elif u==1:
            # Merge to Right Partition
            self.x[(idx+1):] = self.x[(idx+1):] - 1
            self.lambdas = np.delete(self.lambdas,xt)
            
        else:
            # Add New Partition 
            self.lambdas[xt] = self.__gamma_posterior(yt)
    
    #-------------------------------------------------------------------------
    # __get_partitions_counts
    #------------------------------------------------------------------------- 
    def __get_partitions_counts(self):
        
        tmp = np.diff(self.x).astype('bool')
        tmp = np.append(True,tmp)
        tmp = np.append(tmp,True)
        n = np.diff(np.where(tmp))
        
        return n[0]
    
    #-------------------------------------------------------------------------
    # sample_discrete
    #------------------------------------------------------------------------- 
    def __sample_discrete(self, w):
        
        w = w / np.sum(w)
        cdf = np.append(0,np.cumsum(w))
        u = np.random.uniform()
        
        for kk in range(1,len(cdf)):
            if (u>=cdf[kk-1]) & (u<=cdf[kk]):
                idx = kk-1
                break
            
        return idx
        
