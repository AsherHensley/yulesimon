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
def GetYahooFeed(symbol,start_date,end_date):
    
    # Attempt to Fetch Price Data
    log_returns = []
    try:
        df = web.DataReader(symbol, 'yahoo', start_date, end_date)
        closing_prices = df.Close.to_numpy()
        log_returns = np.diff(np.log(closing_prices))
        dates = df.index.copy()
           
    except:
        print("No information for ticker # and symbol: " + symbol)
              
    return closing_prices, log_returns, dates
     
#-----------------------------------------------------------------------------
# GaussianNoiseProcess
#-----------------------------------------------------------------------------
def GaussianNoiseProcess(N=500, alpha=1.0, a=1.0, b=1.0, Q=1e-4, seed=13):
    
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
    
    # Sample Means
    mu = np.zeros(N)
    for idx in range(1,N):
        mu[idx] = np.random.normal(mu[idx-1],Q)
        
    # Sample Observations
    for state in range(len(lambdas)):
        mask = x==state
        y[mask] = np.random.normal(0.0, 1/np.sqrt(lambdas[state]), sum(mask))
    y += mu
      
    return y, x, lambdas, mu


#-----------------------------------------------------------------------------
# PoissonNoiseProcess
#-----------------------------------------------------------------------------
def PoissonNoiseProcess(N=500, alpha=1.0, a=1.0, b=1.0, seed=13):
    
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
        y[mask] = np.random.exponential(1.0/lambdas[state], sum(mask))
      
    return y, x, lambdas

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
# Exponential
#-----------------------------------------------------------------------------
def Exponential(data,LAMBDA):        
    return LAMBDA * np.exp(-LAMBDA * data)

#-----------------------------------------------------------------------------
# Exponential-Gamma Marginal
#-----------------------------------------------------------------------------
def ExpGammaMarginal(data,a,b):        
    return a * (b ** a) / ((b + data) ** (a + 1.0))

#-----------------------------------------------------------------------------
# ExpectedValue
#-----------------------------------------------------------------------------
def ExpectedValue(data,burnin,downsample,mask=[]): 
    tmp = data[:,int(burnin)::int(downsample)]
    if len(mask)>0:
        mask = mask[int(burnin)::int(downsample)]
        tmp = tmp[:,mask]   
    shape = tmp.shape

    return np.mean(tmp,axis=1), shape[1]

#-----------------------------------------------------------------------------
# MixtureModel
#-----------------------------------------------------------------------------
def GaussianMixtureModel(z,mu_t,sigma_t):
    PDF = z*0
    N = len(mu_t)
    for ii in range(N):
        PDF = PDF + Gaussian(z, mu_t[ii], 1 / (sigma_t[ii]**2))
    return PDF/N

#-----------------------------------------------------------------------------
# TimeSeries
#-----------------------------------------------------------------------------
class TimeSeries():

    #-------------------------------------------------------------------------
    # __init__
    #-------------------------------------------------------------------------
    def __init__(self, data, alpha=5.0, a0=1.0, b0=1.0, Q = 1e-9, init='uniform',
                 init_segments=50, mean_removal=False, sample_ab=False, 
                 prop_scale=3, likelihood='Gaussian',use_memory=False,rho=1.0):
        self.data = data
        self.nsamp = np.size(self.data)
        self.alpha = alpha
        self.prop_scale = prop_scale
        self.a0 = a0
        self.b0 = b0 
        self.sample_ab = sample_ab
        self.likelihood = likelihood
        self.lambdas = np.array(self.__gamma_posterior(data[0]))
        self.x = np.zeros(data.shape)
        self.mu = np.zeros(data.shape)
        self.Q = Q
        self.mean_removal = mean_removal
        
        self.use_memory = use_memory
        self.rho = rho
        self.z = np.array([0])
        self.ulambdas = self.lambdas.copy()
        
        if ((init=='uniform') & (use_memory==False)):
            self.__init_partitions_uniform(init_segments)
        elif ((init=='prior') | (use_memory==True)):
            self.__init_partitions()
        else:
            raise ValueError('Unknown Initialization Type: ' + init)
         
        if self.mean_removal==True:
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
                if self.use_memory==True:
                    new_lambda,zk = self.__init_memory(self.data[kk]) 
                    self.z = np.append(self.z, zk)
                else:
                    new_lambda = self.__gamma_posterior(self.data[kk])
                self.lambdas = np.append(self.lambdas, new_lambda)
                counter = 1.0
            else:
                counter += 1  
            
            # Update Partition
            self.x[kk] = state
            
    #-------------------------------------------------------------------------
    # __init_memory
    #-------------------------------------------------------------------------
    def __init_memory(self,yt): 
        
        # Setup
        tables,counts = np.unique(self.z,return_counts=True)
        num_tables = counts.size
        p = np.zeros(num_tables+1)
        
        # Existing Table Probabilities
        for kk in range(num_tables):
            p[kk] = counts[kk] * self.__measure_model(yt, 0.0, self.ulambdas[kk])
        
        # New Table Probability
        if (self.likelihood=='Gaussian'):
            L = Student(yt, 0.0, self.a0 / self.b0, 2 * self.a0)
        elif (self.likelihood=='Exponential'):
            L = ExpGammaMarginal(yt, self.a0, self.b0)
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)
        p[num_tables] = self.rho * L
        
        # Select Table
        zk = self.__sample_discrete(p)
        
        if (zk==num_tables):
            new_lambda = self.__gamma_posterior(yt)
            self.ulambdas = np.append(self.ulambdas, new_lambda)
        else:
            new_lambda = self.ulambdas[zk]
        
        return new_lambda, zk        
        
    #-------------------------------------------------------------------------
    # __measure_model
    #-------------------------------------------------------------------------
    def __measure_model(self,yt,parm1,parm2):
        
        if (self.likelihood=='Gaussian'):
            val = Gaussian(yt,parm1,parm2)
        elif (self.likelihood=='Exponential'):
            val = Exponential(yt,parm2)
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)
        
        return val
     
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
        L[0] = self.__measure_model(yt, 0.0, self.lambdas[-1])
        if (self.likelihood=='Gaussian'):
            L[1] = Student(yt, 0.0, self.a0 / self.b0, 2 * self.a0)
        elif (self.likelihood=='Exponential'):
            L[1] = ExpGammaMarginal(yt, self.a0, self.b0)
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)
        
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
            
        if (self.likelihood=='Gaussian'):
            aN = a0 + 0.5 * N
            bN = b0 + 0.5 * np.sum(np.square(data - mu))  
        elif (self.likelihood=='Exponential'):
            aN = a0 + N
            bN = b0 + np.sum(data) 
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)

        return np.random.gamma(aN,1/bN,1)
    
    #-------------------------------------------------------------------------
    # step
    #-------------------------------------------------------------------------
    def step(self, N=100):
        
        history = self.__init_history(N)
        reject = np.zeros(N)
        ratio = np.zeros(N)

        for step in range(N):
            
            self.__sample_partitions()
            
            if self.use_memory==True:
                self.__sample_lambdas_from_memory()
            else:
                self.__sample_lambdas()
                    
            self.__sample_alpha()
            
            # Sample Gamma Hyperparameters
            if self.sample_ab==True:
                reject[step],ratio[step] = self.__sample_gamma_hyperparameters()
                
            if self.mean_removal==True:
                self.__kalman_filter() 
                self.__sample_process_noise()
                
            self.__update_history(history, step+1)
            
            if (step % round(N/100)) == 0:
                print(".",end='')
          
        print("\n")
        print("Done.\n")
        
        # Print Reject Rate        
        if self.sample_ab==True:
            print("Metropolis-Hastings Rejection Rate: " + str(100*np.mean(reject)) + "%")
                
        return history
    
    #-------------------------------------------------------------------------
    # __sample_process_noise
    #-------------------------------------------------------------------------
    def __sample_process_noise(self):
        self.Q  = 1/self.__gamma_posterior(np.diff(self.mu),1,1)
    
    #-------------------------------------------------------------------------
    # __sample_gamma_hyperparameters
    #-------------------------------------------------------------------------
    def __sample_gamma_hyperparameters(self):
        
        # Initial Condition
        X = np.array([self.a0,self.b0])
        
        # Proposal Distribution Sigma
        sigma2_a = (self.prop_scale) ** 2
        sigma2_b = (self.prop_scale) ** 2
        
        # Proposal Distribution
        logQ = lambda z,mu,Sig: np.log(multivariate_normal.pdf(z,mean=mu,cov=Sig)) - np.log(multivariate_normal.cdf([0,0],mean=-mu,cov=Sig))
            
        # Target Distribution
        logP = lambda z,x: (z[0]-1)*np.sum(np.log(x)) - z[1]*np.sum(x) - x.size * (np.log(gamma(z[0])) - z[0]*np.log(z[1]))
        
        # Step
        step_fail = True
        reject = True
        for ii in range(10):
            a0_prop = np.random.normal(self.a0,sigma2_a)
            b0_prop = np.random.normal(self.b0,sigma2_b)
            if (a0_prop >= 0) & (b0_prop >= 0):
                Y = np.array([a0_prop,b0_prop])
                step_fail = False
                break
        
        if step_fail == False:
            Sigma = np.array([[sigma2_a,0],[0,sigma2_b]])
            ratio = np.exp(logP(Y,self.lambdas) - logP(X,self.lambdas) + logQ(X,Y,Sigma) - logQ(Y,X,Sigma))
            A = min(1,ratio)
            if np.random.uniform() <= A:
                self.a0 = a0_prop
                self.b0 = b0_prop
                reject = False
                
        return reject,ratio
    
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
        
        if self.use_memory==True:
            history.state = np.zeros((self.nsamp, N+1))
        
        self.__update_history(history, 0)

        return history
    
    #-------------------------------------------------------------------------
    # __update_history
    #-------------------------------------------------------------------------  
    def __update_history(self, history, idx):
        
        history.log_likelihood[idx] = self.__log_likelihood()
        
        if (self.likelihood=='Gaussian'):
            history.std_deviation[:,idx] = 1/np.sqrt(self.lambdas[self.x.astype('int')])
        elif (self.likelihood=='Exponential'):
            history.std_deviation[:,idx] = 1/self.lambdas[self.x.astype('int')]
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)
        
        history.boundaries[:,idx] = np.append(0,np.diff(self.x))
        history.mean[:,idx] = self.mu
        history.alpha[idx] = self.alpha
        history.process_noise [idx] = self.Q
        history.hyperparameter_a0[idx] = self.a0
        history.hyperparameter_b0[idx] = self.b0
        
        if self.use_memory==True:
            history.state[:,idx] = self.z[self.x.astype('int')]
        
        # Goodness of fit
        if (self.likelihood=='Gaussian'):
            h,p = normaltest((self.data - self.mu)*np.sqrt(self.lambdas[self.x.astype('int')]))
        else:
            p = -1.0
        history.pvalue[idx] = p  
            
    #-------------------------------------------------------------------------
    # __log_likelihood
    #-------------------------------------------------------------------------      
    def __log_likelihood(self):
        
        lambdas = self.lambdas[self.x.astype('int')]
        if (self.likelihood=='Gaussian'):
            L = np.sum(np.log(Gaussian(self.data-self.mu, 0, lambdas)))
        elif (self.likelihood=='Exponential'):
            L = np.sum(np.log(Exponential(self.data, lambdas)))
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)
        
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
    # __sample_new_state
    #-------------------------------------------------------------------------  
    def __sample_new_state(self,yt,zk_init):
        
        # Get Current Customer Arrangement
        tables,counts = np.unique(self.z,return_counts=True)
        num_tables = self.ulambdas.size
        p = np.zeros(num_tables+1)
        
        for kk in range(len(tables)):
            
            # Get Customer Count for Table kk
            num_customers = counts[kk]
            idx = tables[kk]
            if num_customers==0:
                continue
            
            # Remove Current Customer
            if zk_init==idx:
                num_customers -= 1
                
            # Get Likelihoods
            L = self.__measure_model(yt, 0.0, self.ulambdas[idx])
            p[idx] = num_customers * np.prod(L)
            
        # New Table Probability
        #if zk_init>=0:
        if (self.likelihood=='Gaussian'):
            L = Student(yt, 0.0, self.a0 / self.b0, 2 * self.a0)
        elif (self.likelihood=='Exponential'):
            L = ExpGammaMarginal(yt, self.a0, self.b0)
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)
        p[num_tables] = self.rho * np.prod(L)
        
        # Assign New Table
        zk = self.__sample_discrete(p)
        
        # Assign New Lambda
        if (zk==num_tables):
            new_lambda = self.__gamma_posterior(yt)
            self.ulambdas = np.append(self.ulambdas, new_lambda)
        else:
            new_lambda = self.ulambdas[zk]
            
        return new_lambda, zk
            
    #-------------------------------------------------------------------------
    # __sample_lambdas_from_memory
    #-------------------------------------------------------------------------      
    def __sample_lambdas_from_memory(self):
        
        # Update Customer Assignments
        N = int(max(self.x)+1)
        for ii in range(N):
            
            # Get Measurements
            if (self.likelihood=='Gaussian'):
                yt = self.data[self.x==ii] - self.mu[self.x==ii]
            elif (self.likelihood=='Exponential'):
                yt = self.data[self.x==ii]
            else:
                raise ValueError('Unknown Likelihood: ' + self.likelihood)
                
            # Sample New State
            new_lambda, zk = self.__sample_new_state(yt,self.z[ii])
            self.lambdas[ii] = new_lambda
            self.z[ii] = zk
              
        # Update Table Dishes
        M = np.max(self.z)
        for kk in range(M):
            
            # Get Measurements
            mask = self.z[self.x.astype('int')]==kk
            
            if (self.likelihood=='Gaussian'):
                yt = self.data[mask] - self.mu[mask]
            elif (self.likelihood=='Exponential'):
                yt = self.data[mask]
            else:
                raise ValueError('Unknown Likelihood: ' + self.likelihood)
                
            self.ulambdas[kk] = self.__gamma_posterior(yt)
            self.lambdas[self.z==kk] = self.ulambdas[kk]
                
    #-------------------------------------------------------------------------
    # __sample_lambdas
    #-------------------------------------------------------------------------      
    def __sample_lambdas(self):
        
        N = int(max(self.x)+1)
        for ii in range(N):
            if (self.likelihood=='Gaussian'):
                yt = self.data[self.x==ii] - self.mu[self.x==ii]
            elif (self.likelihood=='Exponential'):
                yt = self.data[self.x==ii]
            else:
                raise ValueError('Unknown Likelihood: ' + self.likelihood)
                
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
        wnew = self.alpha / (1+self.alpha) 
        
        if (self.likelihood=='Gaussian'):
            wnew *= Student(yt, 0.0, self.a0 / self.b0, 2 * self.a0)
        elif (self.likelihood=='Exponential'):
            wnew *= ExpGammaMarginal(yt, self.a0, self.b0)
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)
        
        if boundary=="FirstOpen":
            w0 = (n[0]-1) / (n[0]+self.alpha) * self.__measure_model(yt,0,self.lambdas[0])
            w = np.array([w0,wnew])
            self.__sample_first_open(w, yt)
            
        elif boundary=="FirstClosed":  
            w0 = n[1] / (n[1]+self.alpha+1) * self.__measure_model(yt,0,self.lambdas[1])
            w = np.array([w0,wnew])
            self.__sample_first_closed(w, yt)
              
        elif boundary=="LastOpen":
            w0 = (n[-1]-1) / (n[-1]+self.alpha) * self.__measure_model(yt,0,self.lambdas[-1])
            w = np.array([w0,wnew])
            self.__sample_last_open(w, yt)
            
        elif boundary=="LastClosed":  
            w0 = n[xt-1] / (n[xt-1]+self.alpha+1) * self.__measure_model(yt,0,self.lambdas[-2])
            w = np.array([w0,wnew])
            self.__sample_last_closed(w, yt)
        
        elif boundary=="Left":
            w0 = (n[xt]-1) / (n[xt]+self.alpha) * self.__measure_model(yt,0,self.lambdas[xt])
            w1 = n[xt-1] / (n[xt-1]+self.alpha+1) * self.__measure_model(yt,0,self.lambdas[xt-1])
            w = np.array([w0,w1,wnew])
            self.__sample_left_boundary(w, idx)
  
        elif boundary=="Right":
            w0 = (n[xt]-1) / (n[xt]+self.alpha) * self.__measure_model(yt,0,self.lambdas[xt])
            w1 = n[xt+1] / (n[xt+1]+self.alpha+1) * self.__measure_model(yt,0,self.lambdas[xt+1])
            w = np.array([w0,w1,wnew])
            self.__sample_right_boundary(w, idx)

        elif boundary=="Double":
            w0 = n[xt-1] / (n[xt-1]+self.alpha+1) * self.__measure_model(yt,0,self.lambdas[xt-1])
            w1 = n[xt+1] / (n[xt+1]+self.alpha+1) * self.__measure_model(yt,0,self.lambdas[xt+1])
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
            if self.use_memory==True:
                new_lambda, zk = self.__sample_new_state(yt,-1)
                self.z = np.append(zk, self.z)
            else:
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
            if self.use_memory==True:
                self.z = self.z[1:]
            
        else:
            # Add New Partition
            self.x[0] = 0
            if self.use_memory==True:
                new_lambda, zk = self.__sample_new_state(yt,-1)
                self.z[0] = zk
            else:
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
            if self.use_memory==True:
                new_lambda, zk = self.__sample_new_state(yt,-1)
                self.z = np.append(self.z, zk)
            else:
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
            if self.use_memory==True:
                self.z = self.z[:-1]
            
        else:
            # Replace Partition 
            self.x[-1] = self.x[-1]
            if self.use_memory==True:
                new_lambda, zk = self.__sample_new_state(yt,-1)
                self.z[-1] = zk
            else:
                new_lambda = self.__gamma_posterior(yt)
            self.lambdas[-1] = new_lambda
        
    #-------------------------------------------------------------------------
    # __sample_left_boundary
    #------------------------------------------------------------------------- 
    def __sample_left_boundary(self, w, idx):
    
        u = self.__sample_discrete(w)
            
        if (self.likelihood=='Gaussian'):
            yt = self.data[idx] - self.mu[idx]
        elif (self.likelihood=='Exponential'):
            yt = self.data[idx]
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)           
            
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
            if self.use_memory==True:
                new_lambda, zk = self.__sample_new_state(yt,-1)
                tmp = np.append(self.z[:xt], zk)
                self.z = np.append(tmp, self.z[xt:])
            else:
                new_lambda = self.__gamma_posterior(yt)
            tmp = np.append(self.lambdas[:xt], new_lambda)
            self.lambdas = np.append(tmp, self.lambdas[xt:])
            
    #-------------------------------------------------------------------------
    # __sample_right_boundary
    #------------------------------------------------------------------------- 
    def __sample_right_boundary(self, w, idx):
        
        u = self.__sample_discrete(w)
        
        if (self.likelihood=='Gaussian'):
            yt = self.data[idx] - self.mu[idx]
        elif (self.likelihood=='Exponential'):
            yt = self.data[idx]
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)  
            
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
            if self.use_memory==True:
                new_lambda, zk = self.__sample_new_state(yt,-1)
                tmp = np.append(self.z[:(xt+1)], zk)
                self.z = np.append(tmp, self.z[(xt+1):])
            else:
                new_lambda = self.__gamma_posterior(yt)
            tmp = np.append(self.lambdas[:(xt+1)], new_lambda)
            self.lambdas = np.append(tmp, self.lambdas[(xt+1):])
            
    #-------------------------------------------------------------------------
    # __sample_double_boundary
    #------------------------------------------------------------------------- 
    def __sample_double_boundary(self, w, idx):
        
        u = self.__sample_discrete(w)
        
        if (self.likelihood=='Gaussian'):
            yt = self.data[idx] - self.mu[idx]
        elif (self.likelihood=='Exponential'):
            yt = self.data[idx]
        else:
            raise ValueError('Unknown Likelihood: ' + self.likelihood)  
            
        xt = int(self.x[idx])
        
        if u==0:
            # Merge to Left Partition
            self.x[idx:] = self.x[idx:] - 1
            self.lambdas = np.delete(self.lambdas,xt)
            if self.use_memory==True:
                self.z = np.delete(self.z,xt)
            
        elif u==1:
            # Merge to Right Partition
            self.x[(idx+1):] = self.x[(idx+1):] - 1
            self.lambdas = np.delete(self.lambdas,xt)
            if self.use_memory==True:
                self.z = np.delete(self.z,xt)
            
        else:
            # Replace Current State
            if self.use_memory==True:
                new_lambda, zk = self.__sample_new_state(yt,-1)
                self.z[xt] = zk
            else:
                new_lambda = self.__gamma_posterior(yt)
            self.lambdas[xt] = new_lambda
    
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
        
        idx = -1
        
        for kk in range(1,len(cdf)):
            if (u>=cdf[kk-1]) & (u<=cdf[kk]):
                idx = kk-1
                break
            
        if idx==-1:
            print("WARNING: Discrete Sampler Failure -- check for NaNs in the input data")     
            
        return idx
        
