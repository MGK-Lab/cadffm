import numpy as np
import util as ut
import os
import time
import math
import sys
import copy
import pandas as pd

#Quadratic Equation Solver
def quad_Eq(a, b, c, sign):
    if (math.isnan(a)+math.isnan(b)+ math.isnan(c))==0:
        discriminant = b**2 - 4*a*c
        if discriminant > 0:
            # two real solutions
            x1 = (-b + math.sqrt(discriminant)) / (2*a)
            x2 = (-b - math.sqrt(discriminant)) / (2*a)
        
            solution1 = sign*x1 if sign*x1 > 0 else 0
            solution2 = sign*x2 if sign*x2 > 0 else 0
            # take the minimum of absolute values of two solutions
            if (solution1 == 0 and solution2 == 0):
                solution = 0
            if (solution1 * solution2 == 0):
                solution = solution1 if solution1 > 0 else solution2
                solution *= sign
            if (solution1 * solution2 > 0):
                solution = min(solution1, solution2)
                solution *= sign
        elif discriminant == 0:
            # one real solution
            x = -b / (2*a)
            solution = x if sign*x > 0 else 0
        else:
            solution = 0
    else:
        solution = 0

    return solution

def delta(a):
    if a>0:
        return 1
    else:
        return 0

def non_zero_min(a, b):
    if a * b != 0:
        return np.minimum(a, b)
    else:
        return (a + b)
    
# global constants to reduce computational time
c4_3 = 4/3
c5_3 = 5/3
c2_3 = 2/3

# A Cellular Automata Dynamic Fast Flood Model (main class)
class CADFFM:
    def __init__(self, dem_file, n, CFL, min_wd, min_head, g):
        print("\n .....loading DEM file using CADFFM.....")
        self.begining = time.time()
        print("\n", time.ctime(), "\n")

        self.dem_file = dem_file
        self.z, self.ClosedBC, self.bounds, self.cell_length = ut.RasterToArray(dem_file)
        self.dem_shape = self.z.shape
        self.ClosedBC[:,-1] = False
        self._initialize_arrays()                           # initialize arrays
        self.n0[:] = n                                      # Manning's n
        self.t = 0
        self.delta_t = 0
        # how fast approach the calculated timestep from adaptive reduced timestep
        self.delta_t_bias = 0
        self.g = g                                          # acceleration due to gravity
        self.CFL = CFL                                      # Courant–Friedrichs–Lewy condition
        self.BCtol= 1.0e6                                   # set as a boundary cell elevation
        self.min_WD = min_wd                                # minimum water depth to consider the cell as wet
        self.min_Head = min_head                            # minimum head difference to consider the flow direction         
        self.OpenBC[:,-1] = True                            # open BC at the right edge     
        self.remaining_volume = 0                           # total remaining volume of water
        self.initial_volume = 0                           # total remaining volume of water
        self.cell_length_2 = self.cell_length**2
        
    # initialize arrays   
    def _initialize_arrays(self):
        self.u = np.zeros_like(self.z, dtype=np.double)
        self.v = np.zeros_like(self.z, dtype=np.double)
        self.d = np.zeros_like(self.z, dtype=np.double)
        self.WL = np.zeros_like(self.z, dtype=np.double)
        self.n0 = np.zeros_like(self.z, dtype=np.double)
        self.OpenBC = np.zeros_like(self.z, dtype=np.bool_)
        self.theta = np.array([1, 1, -1, -1])
    
    def set_simulation_time(self, t, delta_t, delta_t_bias=0):
        self.t = t
        self.delta_t = delta_t
        self.delta_t_bias = delta_t_bias
    
    def set_output_path(self, output_path):
        self.DEM_path = "./"
        name = self.dem_file.split('/')
        name = name[-1].split('.')
        self.outputs_name = name[0] + "_out"
        self.DEM_path = os.path.join(output_path, 'DEM/')
        self.csv_merg_path = os.path.join(output_path, 'csv_merged/')
        self.WL_output = os.path.join(output_path, 'WL/')
        self.outputs_TND = os.path.join(output_path, 'TND/')
        # create a file if the file path is not created for above paths 
        for path in [self.DEM_path, self.csv_merg_path, 
                     self.WL_output, self.outputs_TND]:
            if not os.path.exists(path):
                os.makedirs(path)
        
    def set_BCs(self):
        # it modified the defined boundary cells' elevation
        self.z[self.ClosedBC] += self.BCtol 
        self.d[self.OpenBC] = 0       

    def reset_BCs(self):
        # it restore boundary cells' elevation to normal
        self.z[self.ClosedBC] -= self.BCtol
        # self.z[self.OpenBC] += self.BCtol
    
    # set the depth of the reservoir as input   
    def set_reservoir(self, depth, length):
        dam_length = round(length/self.cell_length)+1
        self.d[1:-1, 1:dam_length] = depth

    # calculate max timestep based on Courant–Friedrichs–Lewy condition
    def CFL_delta_t(self, d, u, v):
        velocity_magnitude = np.sqrt(u**2 + v**2) + np.sqrt(self.g * d)
        velocity_magnitude = np.maximum(velocity_magnitude, 1e-10)
        return self.CFL * np.min(self.cell_length / velocity_magnitude)

    # Calculate the Bernoulli head
    def compute_Bernoulli_head(self, z, d, u, v):
        return z + d + (u**2 + v**2)/(2*self.g)
    
    # Check if (H0-Hi)>=epsilon for Normal Flow Condition
    def normal_flow_direction_H(self, H):
        H_diff = H[0]-H[1:5]
        H_i_dir = np.zeros(4, dtype=np.int64)
        H_i_dir[H_diff >= self.min_Head] = 1

        return H_i_dir
    
    # Check if (Hi-H0)>epsilon for Special Flow Condition
    def special_flow_direction_H(self, H):
        H_diff = H[1:5]-H[0]
        H_i_dir = np.zeros(4, dtype=np.int64)
        H_i_dir[H_diff >= self.min_Head] = 1

        return H_i_dir
    
    # Check if (Q*theta)>=0 for Normal Flow Condition
    def normal_flow_direction_Q(self, u, v):
        Q_i = np.array([u[0], v[0], u[0], v[0]]) * self.theta
        Q_i_dir = np.zeros(4, dtype=np.int64)
        Q_i_dir[Q_i >= 0] = 1

        return Q_i_dir
    
    def compute_normal_flow_mass_flux_i(self, n0, H, d, z, i):
        # calculate mass flux with Mannings equation
        d_bar_i = (d[0] + d[i+1]) / 2
        Hloc = H[0] - H[i+1]
        if Hloc < 0:
            Hloc = 0
        Q_mannings = (self.cell_length / n0) * d_bar_i**(c5_3) * \
            np.sqrt(Hloc / self.cell_length)
              
        # calculate mass flux with Weir equation
        z_bar_i = max(z[0], z[i+1])

        # calculate submergance coefficient based on head
        h0_i = H[0] - z_bar_i
        h_i =  H[i+1] - z_bar_i
        psi_i_h = 0
        if h0_i>0:
            ratio_h = h_i / h0_i
            if ratio_h>=0 and ratio_h<1:
                psi_i_h = (1 - ratio_h**1.5)**0.385
        else:
            h0_i = 0        

       # calculate submergance coefficient based on depth
        d0_i = d[0] + z[0] - z_bar_i
        d_i = d[i+1] + z[i+1] - z_bar_i
        psi_i_d = 0
        if d0_i>0:
            ratio_d = d_i/d0_i
            if ratio_d>=0 and ratio_d<1:
                psi_i_d = (1 - ratio_d**1.5)**0.385
                    
        Q_weirs_d = c2_3 * self.cell_length * \
            np.sqrt(2*self.g) * psi_i_d * h0_i**1.5
        Q_weirs_h = c2_3 * self.cell_length * \
            np.sqrt(2*self.g) * psi_i_h * h0_i**1.5
        Q_weirs = max(Q_weirs_d, Q_weirs_h)
        
        Q = 0    
        if z[0] >= z_bar_i:
            Q = non_zero_min(Q_mannings, Q_weirs)
        elif z_bar_i<1e5:
            Q = max(Q_mannings, Q_weirs_h)

        return Q * self.theta[i]
    
    # Calculate mass flux for special flow condition
    def compute_special_flow_mass_flux_i(self, d, u, v, H, deltat, i):
        dQ_1 = self.cell_length_2 / (2 * deltat) * (H[i+1] - H[0] + self.min_Head)
        dQ_2 = self.cell_length_2 / deltat * (d[i+1] - self.min_WD)
        dQ = min(dQ_1, dQ_2)
        
        if i == 0 or i == 2:
            Q = self.cell_length * d[0] * u[0]
                
        if i == 1 or i == 3:
            Q = self.cell_length * d[0] * v[0]
                
        if Q * self.theta[i] < 0:
            Q = 0
                       
        
        if Q != 0:
            tmp = abs(Q)-dQ
            if tmp > 0:
                Q = self.theta[i] * tmp
            else:
                Q = 0
            
        return Q 
    
    # Compute the velocity in both directions
    def compute_velocity(self, FD, n, d, z, u, v, H0, d0):
        vel = np.zeros(4)
        x = 0.5/self.g
        z_diff = z-z[0]
        y = 0.5 * np.sqrt(self.cell_length_2 + z_diff**2)
        for i in range(1, 5, 2):
            if FD[i-1] == 1:
                a = x 
                b = y[i] * n[i]**2 * abs(u[i]) / (d[i]**c4_3)
                c = v[i]**2 * x + d[i] + z[i] + y[i] * \
                    n[0]**2 * u[0]**2 / (d0**c4_3)  - H0
                # a = x + y[i] * n[i]**2 / (d[i]**c4_3)
                # b = 0
                # c = v[i]**2 * x + d[i] + z[i] + y[i] * \
                #     n[0]**2 * u[0]**2 / (d0**c4_3)  - H0
                vel[i-1] = quad_Eq(a, b, c, self.theta[i-1])

        for i in range(2, 5, 2):
            if FD[i-1] == 1:
                a = x
                b = y[i] * n[i]**2 * abs(v[i]) / (d[i]**c4_3)
                c = u[i]**2 * x + d[i] + z[i] + y[i] * \
                    n[0]**2 * v[0]**2 / (d0**c4_3)  - H0
                # a = x + y[i] * n[i]**2 / (d[i]**c4_3)
                # b = 0
                # c = u[i]**2 * x + d[i] + z[i] + y[i] * \
                #     n[0]**2 * v[0]**2 / (d0**c4_3)  - H0
                vel[i-1] = quad_Eq(a, b, c, self.theta[i-1])

        return vel
    
    def run_simulation(self):
        # to ensure if user changed it 
        self.cell_length_2 = self.cell_length**2
        
        current_time = 0
        iteration = 1
        self.begining = time.time()
        if self.t * self.delta_t == 0:
            sys.exit("Simulation time or timestep have not been defined")

        delta_t = min(self.delta_t, self.CFL_delta_t(self.d, self.u, self.v))
        Time = True
        t_value = 0.1                                               # assigned to use later for depth extraction of every multiples of 0.1sec

        self.initial_volume = np.sum(self.d[:,:-1]) * self.cell_length_2
        
        while current_time < self.t:
            self.set_BCs()

            H = self.compute_Bernoulli_head(self.z, self.d, self.u, self.v)         # calculate Bernoulli head

            FD = np.zeros_like(self.z, dtype=np.int64)
            delta_d= np.zeros_like(self.z, dtype=np.double)
            d_new = np.zeros_like(self.z, dtype=np.double)                  # new water depth
            u_new = np.zeros_like(self.u, dtype=np.double)
            v_new = np.zeros_like(self.v, dtype=np.double)


            # this loop caculates updated d, mass fluxes and directions
            for i in range(1, self.dem_shape[0] - 1):
                for j in range(1, self.dem_shape[1] - 1):
                    # check if the cell is dry or wet (d0>=delta)
                    if self.d[i,j] > self.min_WD:
                        # For simplicity, the central cell is indexed as 0, and
                        # its neighbor cells at the east, north, west, and
                        # south sides are indexed as 1, 2, 3, and 4.
                        z = np.array([self.z[i, j], self.z[i+1, j],
                                      self.z[i, j+1], self.z[i-1, j],
                                      self.z[i, j-1]])
                        d = np.array([self.d[i, j], self.d[i+1, j],
                                      self.d[i, j+1], self.d[i-1, j],
                                      self.d[i, j-1]])
                        u = np.array([self.u[i, j], self.u[i+1, j],
                                      self.u[i, j+1], self.u[i-1, j],
                                      self.u[i, j-1]])
                        v = np.array([self.v[i, j], self.v[i+1, j],
                                      self.v[i, j+1], self.v[i-1, j],
                                      self.v[i, j-1]])
                        H_loc = np.array([H[i, j], H[i+1, j],
                                         H[i, j+1], H[i-1, j],
                                         H[i, j-1]])

                        # calculate normal flow
                        normal_flow = self.normal_flow_direction_H(H_loc) * self.normal_flow_direction_Q(u, v)
                        # Q = self.compute_normal_flow_mass_flux(self.n0[i, j], H_loc, d, z)
                        Qn = np.zeros(4)
                        for n in range(4):
                            if normal_flow[n] > 0:
                                Qn[n] = self.compute_normal_flow_mass_flux_i(self.n0[i, j], H_loc, d, z, n)
                     
                        # compute special flow
                        special_flow = copy.deepcopy(d[1:])
                        special_flow[special_flow > self.min_WD] = 1
                        special_flow *= self.special_flow_direction_H(H_loc) * self.special_flow_direction_Q(u, v)
                        # Q_special = self.compute_special_flow_mass_flux(d, u, v, H_loc, special_flow, delta_t)
                        Qs = np.zeros(4)
                        for n in range(4):
                            if special_flow[n] > 0:
                                Qs[n] = self.compute_special_flow_mass_flux_i(d, u, v, H_loc, delta_t, n)
                  
                        # final flow flux values considering both conditions
                        Qn[special_flow == 1] = 0
                        Q = Qs + Qn
                        
                        #  save flow directions as a binary value to reterive later
                        FD_loc = np.copy(Q)
                        FD_loc[FD_loc != 0] = 1
                        FD[i, j] = int(
                            ''.join(map(str, FD_loc.astype(np.int64))), 2)
                        
                        # update the water depth according to the mass flux
                        delta_d[i,j] += 1 / self.cell_length_2 * \
                                    np.sum((-self.theta) * Q)             
                        if Q[0] != 0:
                            delta_d[i+1, j] += 1 / self.cell_length_2 * \
                                (self.theta[0] * Q[0])
                        if Q[1] != 0:
                            delta_d[i, j+1] += 1 / self.cell_length_2 * \
                                (self.theta[1] * Q[1])
                        if Q[2] != 0:
                            delta_d[i-1, j] += 1 / self.cell_length_2 * \
                                (self.theta[2] * Q[2])
                        if Q[3] != 0:
                            delta_d[i, j-1] += 1 / self.cell_length_2 * \
                                (self.theta[3] * Q[3]) 
                        
            d_new = self.d + delta_d * delta_t
            
            # adaptive time stepping => find the min required timestep
            # this section avoids over drying cells (negative d)
            if (d_new < 0).any():
                neg_indices = np.where(d_new < 0)
                indices_array = np.column_stack(neg_indices)
                tmp_values = np.array([])
                for row in indices_array:
                    r,c = row
                    tmp = (self.min_WD-self.d[r, c])/ delta_d[r, c]
                    tmp_values = np.append(tmp_values,tmp)
                    
                # for better stability, devided by 2 
                delta_t_new = np.min(tmp_values) / 2
                d_new = self.d + delta_d * delta_t_new
            else:
                delta_t_new = delta_t
            current_time += delta_t_new
            
            # this loop calculate velocities in both directions
            for i in range(1, self.dem_shape[0]-1):
                for j in range(1, self.dem_shape[1]-1):
                    # check if the cell is dry or wet (d0>=delta)
                    if self.d[i,j] > self.min_WD:
                        # For simplicity, the central cell is indexed as 0, and
                        # its neighbor cells at the east, north, west, and
                        # south sides are indexed as 1, 2, 3, and 4.
                        z = np.array([self.z[i, j], self.z[i+1, j],
                                      self.z[i, j+1], self.z[i-1, j],
                                      self.z[i, j-1]])
                        d_n = np.array([d_new[i, j], d_new[i+1, j],
                                      d_new[i, j+1], d_new[i-1, j],
                                      d_new[i, j-1]])
                        u = np.array([self.u[i, j], self.u[i+1, j],
                                      self.u[i, j+1], self.u[i-1, j],
                                      self.u[i, j-1]])
                        v = np.array([self.v[i, j], self.v[i+1, j],
                                      self.v[i, j+1], self.v[i-1, j],
                                      self.v[i, j-1]])
                        n = np.array([self.n0[i, j], self.n0[i+1, j],
                                      self.n0[i, j+1], self.n0[i-1, j],
                                      self.n0[i, j-1]])

                        # retrieve the flow direction from binary value
                        FD_loc = np.binary_repr(FD[i, j], 4)
                        FD_loc = [int(digit) for digit in FD_loc]            

                        if sum(FD_loc)>0:
                            vel = self.compute_velocity(
                                FD_loc, n, d_n, z, u, v, H[i, j], self.d[i, j])
                            
                            dmin = d_n[1:5]
                            dmin[dmin>=self.min_WD] = 1
                            dmin[dmin<1] = 0
                            
                            if FD_loc[0]*FD_loc[2]==1:
                                u_new[i, j] = delta(self.theta[0]*vel[0])*vel[0] + delta(self.theta[2]*vel[2])*vel[2]
                            else:
                                u_new[i-1, j] += vel[2] * dmin[2]
                                u_new[i+1, j] += vel[0] * dmin[0]
                                
                            if FD_loc[1]*FD_loc[3]==1:
                                v_new[i, j] = delta(self.theta[1]*vel[1])*vel[1] + delta(self.theta[3]*vel[3])*vel[3]
                            else:
                                v_new[i, j-1] += vel[3] * dmin[3]
                                v_new[i, j+1] += vel[1] * dmin[1]
                            
                        
            for i in range(1, self.dem_shape[0]-1):
                for j in range(1, self.dem_shape[1]-1):
                    if u_new[i, j] !=0:
                        self.u[i, j] = u_new[i, j]
                    if v_new[i, j] !=0:    
                       self.v[i, j] = v_new[i, j]
                       
                            
            self.d = d_new
                       
            # to print the outputs for time steps/iterations
            if iteration % 100 == 0:
                self.report_screen(iteration, delta_t_new, current_time)

            # time step adjustment
            delta_t = self.CFL_delta_t(self.d, self.u, self.v)
            if delta_t_new < delta_t:
                delta_t = (delta_t_new + (self.delta_t_bias + 1)
                           * delta_t) / (self.delta_t_bias + 2)

            # self.time_step.append([current_time, delta_t_new])
            if current_time >= t_value:
                Time = True
                t_value += 0.1
            
            self.reset_BCs()

            # to export the water level and velocity at specified times
            if Time: 
                self.export_time_series(current_time, self.z, self.d, self.d + self.z , H, self.v)
                Time = False                 
            iteration += 1
        
        self.close_simulation(self.outputs_name, iteration, current_time, delta_t_new)
        print("\nSimulation finished in", (time.time() - self.begining),
        "seconds")
    
    def export_time_series(self, current_time, z, d, wl, H, v):
        z_array = z[z.shape[0]//2, :]
        z_array = z_array.reshape(-1, 1)
        z_df = pd.DataFrame(z_array)
        
        d_array = d[d.shape[0]//2, :]
        d_array = d_array.reshape(-1, 1)
        d_df = pd.DataFrame(d_array)
        
        wl_array = wl[wl.shape[0]//2, :]   
        wl_array = wl_array.reshape(-1, 1)
        wl_df = pd.DataFrame(wl_array)
        
        H_array = H[H.shape[0]//2, :]
        H_array = H_array.reshape(-1, 1)
        H_df = pd.DataFrame(H_array)
        
        v_array = v[v.shape[0]//2, :]
        v_array = v_array.reshape(-1, 1)
        v_df = pd.DataFrame(v_array)  

        WL_name = self.WL_output +"WL" + str(np.around(current_time, decimals = 1)) + '.csv'
        # WL_name = self.WL_output +"WL" + str(current_time) + '.csv'
        concat_df = pd.concat([z_df, d_df, wl_df, H_df, v_df], axis=1)
        concat_df.to_csv(WL_name)
        concat_df.to_csv(WL_name, header = ['z', 'd', 'wl', 'H', 'v'])
            
    def close_simulation(self, name, iteration, current_time, delta_t_new):
        print("\n .....closing and reporting the CADFFM simulation.....")
        self.report_screen(iteration, delta_t_new, current_time)
        print("\n", time.ctime(), "\n")
    
    def report_screen(self, iteration, delta_t_new, current_time):

        self.remaining_volume = np.sum(self.d[:,:-1]) * self.cell_length_2
        
        print("iteration: ", iteration)
        print("simulation time: ", "{:.3f}".format(current_time))
        print('delta_t: ', "{:.3e}".format(delta_t_new))
        print('min_Head: ', "{:.3e}".format(self.min_Head))
        print('max velocity (u, v): ', "{:.3f}".format(np.max(np.absolute(self.u))),
              ", ", "{:.3f}".format(np.max(np.absolute(self.v))))
        print('water depth (min, max): ', "{:.3f}".format(np.min(self.d)),
              ", ", "{:.3f}".format(np.max(self.d)))
        print('volume of water (initial, current): ',  "{:.3f}".format(self.initial_volume), ", ", "{:.3f}".format(self.remaining_volume))
        print("\n")
        
    # Calculate mass flux for special flow condition (old)
    def compute_special_flow_mass_flux(self, d, u, v, H, SFD, deltat):
        dQ = np.zeros(4)
        dQ_1 = self.cell_length_2 / (2 * deltat) * (H[1:5] - H[0] + self.min_Head)
        dQ_2 = self.cell_length_2 / deltat * (d[1:5] - self.min_WD)
        dQ = np.minimum(dQ_1, dQ_2)
        
        Q = np.zeros(4)
        for i in range(4):
            if SFD[i] > 0:
                if i == 0 or i == 2:
                    if u[0]!=0:
                        Q[i] = self.cell_length * d[0] * u[0]
                        if Q[i]*self.theta[i]<0:
                            Q[i] = 0
                if i == 1 or i == 3:
                    if v[0]!=0:
                        Q[i] = self.cell_length * d[0] * v[0]
                        if Q[i]*self.theta[i]<0:
                            Q[i] = 0
                       
        tmp = np.abs(Q)-dQ
        for i in range(4):
            if Q[i] != 0 and tmp[i]>0:
                Q[i] = self.theta[i] * tmp[i]
            else:
                Q[i] = 0
        return Q
    
    # Calculate if (Q*theta)>0 for Special Flow Condition
    def special_flow_direction_Q(self, u, v):
        Q_i = np.array([u[0], v[0], u[0], v[0]]) * self.theta
        Q_i_dir = np.zeros(4, dtype=np.int64)
        Q_i_dir[Q_i > 0] = 1

        return Q_i_dir

    # Calculate mass flux for normal flow condition (old)
    def compute_normal_flow_mass_flux(self, n0, H, d, z):
        H_p = d + z     
        Q = np.zeros(4)

        # calculate mass flux with Mannings equation
        d_bar_i = (d[0] + d[1:5]) / 2
        Hloc = H[0] - H[1:5]
        Hloc[Hloc < 0] = 0
        Q_mannings = (self.cell_length / n0) * d_bar_i**(c5_3) * \
            np.sqrt(Hloc / self.cell_length)
              
        # calculate mass flux with Weir equation
        z_bar_i = np.maximum(z[0], z[1:5])

        # calculate submergance coefficient based on head
        h0_i = H[0] - z_bar_i
        h_i =  H[1:5] - z_bar_i
        psi_i_h = np.zeros(4)
        for i in range(4):
            if h0_i[i]>0:
                ratio_h = h_i[i] / h0_i[i]
                if ratio_h>=0 and ratio_h<1:
                    psi_i_h[i] = (1 - ratio_h**1.5)**0.385

       # calculate submergance coefficient based on depth
        d0_i = H_p[0] - z_bar_i
        d_i = H_p[1:5] - z_bar_i
        psi_i_d = np.zeros(4)
        for i in range(4):
            if d0_i[i]>0:
                ratio_d = d_i[i]/d0_i[i]
                if ratio_d>=0 and ratio_d<1:
                    psi_i_d[i] = (1 - ratio_d**1.5)**0.385
                    
        h0_i[h0_i<0] = 0
        Q_weirs_d = c2_3 * self.cell_length * \
            np.sqrt(2*self.g) * psi_i_d * h0_i**1.5
        Q_weirs_h = c2_3 * self.cell_length * \
            np.sqrt(2*self.g) * psi_i_h * h0_i**1.5
        Q_weirs = np.maximum(Q_weirs_d, Q_weirs_h)
            
        for i in range(4):
            if z[0] >= z_bar_i[i]:
                Q[i] = non_zero_min(Q_mannings[i], Q_weirs[i])
            elif z_bar_i[i]<1e5:
                Q[i] = np.maximum(Q_mannings[i], Q_weirs_h[i])

        return Q * self.theta

        
