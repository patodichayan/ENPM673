import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from scipy.stats import norm

###  Read the image frames  ###
i = 1

while (i <= 28):
    GreenI = cv2.imread('../Data/Frames/Green/Training/Cropped/Cropped_Image{}.png'.format(i))
    GChannel = (GreenI[:, :, 1])

    GChannel = np.concatenate(GChannel)
    i = i + 4

j = 1

while (j <= 121):
    RedI = cv2.imread('../Data/Frames/Red/Training/Cropped/Cropped_Image{}.png'.format(j))
    RChannel = (RedI[:, :, 2])
    RChannel = np.concatenate(RChannel)

    YellowI = cv2.imread('../Data/Frames/Yellow/Training/Cropped/Cropped_Image{}.png'.format(j))
    YChannel = (YellowI[:, :, 1] + YellowI[:, :, 2])

    YChannel = np.concatenate(YChannel)
    X = sorted(RChannel)

    j = j + 20

   ##############################

np.random.seed(0)

X_stack = np.asarray(X)

print (X_stack)


class GM1D:
    def __init__(self,X,iterations):
        self.iterations = iterations
        self.X = X
        self.mu = None
        self.pi = None
        self.sigma = None
    
    

    def Gaussian1D(self,x,mu,sigma):

        A = 1/(sigma*np.sqrt(2*math.pi))
        Z = A*np.exp(-(x-mu)**2/(2.*sigma**2))

        return Z

    
    def run(self):
       
        """
        Instantiate the random mu, pi and sigma
        """
        self.mu = [-15,11,30,22]
        self.pi = [1/25,1/25,1/25,1/25]
        self.sigma = [5,24,59,17]

        """
        E-Step
        """
       ####1D-3Gaussian start##################################################################################################################################
        for iter in range(self.iterations):
            """Create the array r with dimensionality nxK"""
            r = np.zeros((len(X_stack),4))
 
            """
            Probability for each datapoint x_i to belong to gaussian g
            """
            for c,g,p in zip(range(4),[self.Gaussian1D(X_stack,self.mu[0],self.sigma[0]),
                                       self.Gaussian1D(X_stack,self.mu[1],self.sigma[1]),self.Gaussian1D(X_stack,self.mu[2],self.sigma[2]),
                                       self.Gaussian1D(X_stack,self.mu[3],self.sigma[3])],self.pi):
                r[:,c] = p*g # Write the probability that x belongs to gaussian c in column c.
                                      # Therewith we get a 60x3 array filled with the probability that each x_i belongs to one of the gaussians

            """
            Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to
            cluster c
            """
            for i in range(len(r)):
                r[i] = r[i]/(np.sum(self.pi)*np.sum(r,axis=1)[i])
            """Plot the data"""

            """Plot the gaussians"""
            g_sum = np.zeros(3900)
            for g,c in zip([self.Gaussian1D((np.linspace(-25, 350, num=3900)),self.mu[0],self.sigma[0]),
                            self.Gaussian1D((np.linspace(-25, 350, num=3900)),self.mu[1],self.sigma[1]),self.Gaussian1D((np.linspace(-25, 350, num=3900)),self.mu[2],self.sigma[2]),
                            self.Gaussian1D((np.linspace(-25, 350, num=3900)),self.mu[3],self.sigma[3])],['r','g','b']):
                g_sum = g_sum + g
            if iter==9:
                plt.plot(np.linspace(-25, 350, num=3900), g_sum, c='b')
                y=np.zeros(len(X_stack))
                plt.scatter(X_stack,y)
                plt.xlim(10,300)
                plt.title("Red Channel Red Buoy_1D-4Gaussian")


            """M-Step"""
   
            """calculate m_c"""
            m_c = []
            for c in range(len(r[0])):
                m = np.sum(r[:,c])
                m_c.append(m) # For each cluster c, calculate the m_c and add it to the list m_c
            """calculate pi_c"""
            for k in range(len(m_c)):
                self.pi[k] = (m_c[k]/np.sum(m_c)) # For each cluster c, calculate the fraction of points pi_c which belongs to cluster c
            """calculate mu_c"""
            X = np.reshape(X_stack, (len(X_stack), 1))

            self.mu = np.sum(X*r,axis=0)/m_c
            print('mu',self.mu)
            """calculate sigma_c"""
            sigma_c = []
            for c in range(len(r[0])):
                sigma_c.append((1/m_c[c])*np.dot(((np.array(r[:,c]).reshape((len(X_stack), 1)))*(X-self.mu[c])).T,(X-self.mu[c])))
            #print('sigma_c',sigma_c)
            
            flat_list = []
            for sublist in sigma_c:
                for item in sublist:
                    for sub_item in item:
                        flat_list.append(sub_item)
            print(flat_list)


            plt.show()


GM = GM1D(X_stack,10)
GM.run()
