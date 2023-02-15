import numpy as np 
import util as ut
import os
import time

class ShallowWaterFlowSolver:
    def __init__(self, dem_file):
        print("\n .....loading DEM file using CA-ffé.....")
        print("\n", time.ctime(), "\n")

        self.dem_file = dem_file
        self.z, self.ClosedBC, self.bounds = ut.DEMRead(dem_file)
        self.dem_shape = self.z.shape
        self.cell_length = 1

        self.d = np.zeros_like(self.z, dtype=np.double)
        self.u = np.zeros_like(self.z, dtype=np.double)
        self.v = np.zeros_like(self.z, dtype=np.double)
        self.n0= np.zeros_like(self.z, dtype=np.double)

        self.theta = np.array([1, 1, -1, -1])

        self.t = 0
        self.delta_t = 0
        self.g = 9.81
        self.CFL = 0.2

        # it is the delta in the paper
        self.min_WD= 0.01
        # it is epsilon in the paper
        self.min_Head =0.01

    def SetSimulationTime(self, t, delta_t):
        self.t = t
        self.delta_t = delta_t

    def CFL_deltat(self, d, u, v):
       return self.CFL * np.min(self.cell_length/(np.sqrt(u**2 + v**2) + np.sqrt(self.g*d)))

    def ComputeBernoulliHead(self, z, d, u, v):
        # Calculate the Bernoulli head
        return z + d + (u**2 + v**2)/(2*self.g)
    
    def ComputeFlowDirectionH(self, H):
        # Check if (H0-Hi)>=epsilon for Normal Flow Condition 
        tmp = H[0]-H[1:5]
        for i in tmp:
            if i>=self.min_Head:
                i=1
            else:
                i=0
        return tmp

    def ComputeFlowDirectionQ(self, d, u, v):
        # Check if (Q-theta)>=0 for Normal Flow Condition
        Q_i = [(u[0]*d[0]-u[1]*d[1]),(v[0]*d[0]-v[2]*d[2]),(u[0]*d[0]-u[3]*d[3]),(v[0]*d[0]-v[4]*d[4])]
        Q_i = Q_i * self.theta
        for i in Q_i:
            if i>=0:
                i=1
            else:
                i=0
        return Q_i

    def ComputeMassFlux(self, n0, H, d, z):
        d_bar_i = (d[0] + d[1:5]) / 2
        Q_mannings = (self.cell_length / n0) * d_bar_i**(5/3) * np.sqrt((H[0]-H) / self.cell_length)

        z_bar_i = np.maximum(z[0], z[1:5])
        h0_i = H[0] - z_bar_i
        h_i = np.maximum(0, H[1:5] - z_bar_i)
        psi_i = (1 - (h_i / h0_i)**1.5)**0.385
        Q_weirs = (2/3) * self.cell_length * np.sqrt(2*self.g) * psi_i * h0_i**1.5
        return np.minimum(Q_mannings, Q_weirs)

    def RunSimulation(self):
        current_time = 0
        while current_time < self.t:
            for i in range(1, self.dem_shape[0] - 1):
                for j in range(1, self.dem_shape[1] - 1):
                    # check if the cell is dry or wet (d0>=delta)
                    if self.d[i, j]>=self.min_WD:
                        # For simplicity, the central cell is indexed as 0, and its neighbor cells 
                        # at the east, north, west, and south sides are indexed as 1, 2, 3, and 4, respectively
                        z = np.array([self.z[i, j], self.z[i+1, j], self.z[i, j+1], self.z[i-1, j], self.z[i, j-1]])
                        d = np.array([self.d[i, j], self.d[i+1, j], self.d[i, j+1], self.d[i-1, j], self.d[i, j-1]])
                        u = np.array([self.u[i, j], self.u[i+1, j], self.u[i, j+1], self.u[i-1, j], self.u[i, j-1]])
                        v = np.array([self.v[i, j], self.v[i+1, j], self.v[i, j+1], self.v[i-1, j], self.v[i, j-1]])
                        
                        H = self.ComputeBernoulliHead(z, d, u, v)

                        FD = self.ComputeFlowDirectionH(H)
                        Q_vel = self.ComputeFlowDirectionH(d, u, v)

                        # chek if the normal flow condition is valid
                        if Q_vel*FD>0:
                            Q = self.ComputeMassFlux(self.n0[i, j], H, d, z)
                            d0_new = d[0] + self.delta_t / delta_t**2 * -np.sum(Q)
                    
            delta_t = min(self.delta_t, self.CFL_deltat(self.d, self.u, self.v))
            current_time += delta_t