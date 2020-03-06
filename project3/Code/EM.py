import numpy as np
import matplotlib.pyplot as plt
import math


np.random.seed(0)
X = np.linspace(-5,5)   #Default Sample Size = 50
X0 = X*np.random.rand(len(X)) + 20
X1 = X*np.random.rand(len(X)) - 20
X2 = X*np.random.rand(len(X)) 

X_stack = np.stack((X0,X1,X2)).flatten()

class EM:
    
    def __init__(self,X,iterations):
        self.X = X
        self.mean = None
        self.sigma = None
        self.weight = None
        self.iterations = iterations
        
        
    def Gaussian1D(self,x,mu,sigma):    
        A = 1/(sigma*np.sqrt(2*math.pi))
        Z = A*np.exp(-(x-mu)**2/(2.*sigma**2))

        return Z
    
        
    def start(self):
    
        self.weights = [1/3,1/3,1/3]
        self.sigma = [3,1,2]
        self.mean = [-4,2,3]
        
        
        #Working on the Expectation-Step:
        
        for iterations in range(self.iterations):
        
            r = np.zeros((len(X_stack),3))
            
            for i,g,w in zip(range(3),[self.Gaussian1D(X_stack,self.mean[0],self.sigma[0]),self.Gaussian1D(X_stack,self.mean[1],self.sigma[1]),self.Gaussian1D(X_stack,self.mean[2],self.sigma[2])],self.weights):
                
                r[:,i] = w*g
                
            for i in range(len(r)):
                r[i] = r[i]/(np.sum(self.weights)*np.sum(r,axis=1)[i])
            
            if iterations ==0:
                for i in range(len(r)):
                    plt.scatter(self.X[i],0,c=np.array([r[i][0],r[i][1],r[i][2]]))
                
                
                X_new = np.linspace(-28,28,num=150)     
            
                for g,i in zip([self.Gaussian1D(X_new,self.mean[0],self.sigma[0]),self.Gaussian1D(X_new,self.mean[1],self.sigma[1]),self.Gaussian1D(X_new,self.mean[2],self.sigma[2])],['r','g','b']):
                    plt.plot((X_new),g,c=i)
                    
                                
                                
           #Working on the Maximization-Step:
           
            m_c = []
           
            for j in range(len(r[0])):
           
                m = np.sum(r[:,j])
                m_c.append(m)
           
            for c in range(len(m_c)):
           
                self.weights[c] = (m_c[c]/np.sum(m_c))
           
            self.mean = np.sum(self.X.reshape(len(self.X),1)*r,axis =0)/m_c    
            sigma_c = []
           
            
            for i in range(len(r[0])):
                form = np.dot(((np.array(r[:,i]).reshape(150,1))*(self.X.reshape(len(self.X),1)-self.mean[i])).T,(self.X.reshape(len(self.X),1)-self.mean[i]))
           
                sigma_c.append((1/m_c[i])*form)
               
            flattened = []
            for sublist in sigma_c:
                for val in sublist:
                    flattened.append(val)
                   
            if iterations == 6:
                
                for i in range(len(r)):
                    plt.scatter(self.X[i],0,c=np.array([r[i][0],r[i][1],r[i][2]]))
               
                for g,i in zip([self.Gaussian1D(X_new,self.mean[0],flattened[0]),
                                self.Gaussian1D(X_new,self.mean[1],flattened[1]),
                                self.Gaussian1D(X_new,self.mean[2],flattened[2])],['r','g','b']):    
                    
                    plt.plot(X_new,g,c=i)
           
            plt.show()        
                                                  
                                                  
                                                  
EM = EM(X_stack,7)
EM.start()                                           

