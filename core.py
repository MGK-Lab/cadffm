import numpy as np
import util as ut
import os
import time
import math

# A Cellular Automata Dynamic Fast Flood Model


def QuadEQ(a, b, c, sign):
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        # two real solutions
        x1 = (-b + math.sqrt(discriminant)) / (2*a)
        x2 = (-b - math.sqrt(discriminant)) / (2*a)
        solution1 = sign*x1 if sign*x1 > 0 else 0
        solution2 = sign*x2 if sign*x2 > 0 else 0
        if (solution1 == 0 or solution2 == 0):
            if (solution1 == 0 and solution2 == 0):
                solution = 0
            else:
                solution = solution1 if solution1 > 0 else solution2
        else:
            solution = min(sign*solution1, sign*solution2)

        solution *= sign
    elif discriminant == 0:
        # one real solution
        x = -b / (2*a)
        solution = x if sign*x > 0 else 0
    else:
        solution = 0

    return solution


def Delta(a):
    return 1 if a > 0 else 0


class CADFFM:
    def __init__(self, dem_file):
        print("\n .....loading DEM file using CA-ffé.....")
        print("\n", time.ctime(), "\n")

        self.dem_file = dem_file
        self.z, self.ClosedBC, self.bounds = ut.DEMRead(dem_file)
        self.dem_shape = self.z.shape
        self.cell_length = 1

        self.d = np.zeros_like(self.z, dtype=np.double)
        self.d_new = np.zeros_like(self.z, dtype=np.double)
        self.u = np.zeros_like(self.z, dtype=np.double)
        self.v = np.zeros_like(self.z, dtype=np.double)
        self.n0 = np.zeros_like(self.z, dtype=np.double)

        self.theta = np.array([1, 1, -1, -1])

        self.t = 0
        self.delta_t = 0
        self.g = 9.81
        self.CFL = 0.2

        # it is the delta in the paper
        self.min_WD = 0.01
        # it is epsilon in the paper
        self.min_Head = 0.01

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
            if i >= self.min_Head:
                i = 1
            else:
                i = 0
        return tmp.astype(np.int)

    def ComputeFlowDirectionQ(self, d, u, v):
        # Check if (Q*theta)>=0 for Normal Flow Condition
        Q_i = np.array([(u[0]*d[0]-u[1]*d[1]), (v[0]*d[0]-v[2]*d[2]),
                        (u[0]*d[0]-u[3]*d[3]), (v[0]*d[0]-v[4]*d[4])])
        Q_i = Q_i * self.theta
        for i in Q_i:
            if i >= 0:
                i = 1
            else:
                i = 0
        return Q_i.astype(np.int)

    def ComputeMassFlux(self, n0, H, d, z):
        d_bar_i = (d[0] + d[1:5]) / 2

        Q_mannings = (self.cell_length / n0) * d_bar_i**(5/3) * \
            np.sqrt((H[0]-H) / self.cell_length)

        z_bar_i = np.maximum(z[0], z[1:5])
        h0_i = H[0] - z_bar_i
        h_i = np.maximum(0, H[1:5] - z_bar_i)
        psi_i = (1 - (h_i / h0_i)**1.5)**0.385

        Q_weirs = (2/3) * self.cell_length * \
            np.sqrt(2*self.g) * psi_i * h0_i**1.5

        return np.minimum(Q_mannings, Q_weirs) * self.theta

    def ComputeMassFluxSpecial(self, con, H, d, Q, deltat):
        Qs = np.zeros(4)
        for i in range(1, 5):
            if con[i] > 0:
                dQ_1 = self.cell_length**2 / \
                    (2 * deltat) * (H[i] - H[0] + self.min_Head)
                dQ_2 = self.cell_length**2 / deltat * \
                    (d[i] - self.min_WD)
                dQ = np.minimum(dQ_1, dQ_2)

                tmp = np.abs(Q[i-1])-dQ
                if tmp > 0:
                    Qs[i-1] = np.sgn(Q[i-1])*tmp
                else:
                    Qs[i-1] = 0

            else:
                Qs[i-1] = 0

        return Qs

    def ComputeVelocity(self, FD, n, d, z, u, v, H0, d0):
        vel = np.zeros(4)

        a = 0.5/self.g

        for i in range(1, 5, 2):
            if FD[i-1] == 1:
                b = -self.cell_length * n[i]**2 * abs(u[i]) / (2*d[i]**(4/3))
                c = v[i]**2 * a + d[i] + z[i] + self.cell_length * \
                    n[0]**2 * u[0]**2 / (2*d0**(4/3)) - H0
                vel[i-1] = QuadEQ(a, b, c, self.theta[i-1])

        for i in range(2, 5, 2):
            if FD[i-1] == 1:
                b = -self.cell_length * n[i]**2 * abs(v[i]) / (2*d[i]**(4/3))
                c = u[i]**2 * a + d[i] + z[i] + self.cell_length * \
                    n[0]**2 * v[0]**2 / (2*d0**(4/3)) - H0
                vel[i-1] = QuadEQ(a, b, c, self.theta[i-1])

        return vel

    def RunSimulation(self):
        current_time = 0

        while current_time < self.t:
            delta_t = min(self.delta_t, self.CFL_deltat(
                self.d, self.u, self.v))

            H = self.ComputeBernoulliHead(self.z, self.d, self.u, self.v)
            FD = np.zeros_like(self.z, dtype=np.int)

            # this loop caculates updated d, mass fluxes and directions
            for i in range(1, self.dem_shape[0] - 1):
                for j in range(1, self.dem_shape[1] - 1):
                    # check if the cell is dry or wet (d0>=delta)
                    if self.d[i, j] >= self.min_WD:
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
                        Hloc = np.array([self.H[i, j], self.H[i+1, j],
                                         self.H[i, j+1], self.H[i-1, j],
                                         self.H[i, j-1]])

                        # calculate fluxes and directions
                        FDH = self.ComputeFlowDirectionH(Hloc)
                        FDQ = self.ComputeFlowDirectionQ(d, u, v)
                        Q = self.ComputeMassFlux(self.n0[i, j], Hloc, d, z)

                        # the normal flow condition is valid
                        NFD = FDQ * FDH
                        Qn = NFD * Q

                        # the special flow condition is valid
                        SFD = d[1:]
                        SFD[SFD > 0] = 1
                        SFD *= FDH ^ 1 * FDQ
                        Qs = self.ComputeMassFluxSpecial(
                            self, SFD, Hloc, d, Q, delta_t)

                        # final flow flux values considering both conditions
                        Qn[SFD == 1] = 0
                        Q = Qs + Qn

                        # save flow directions as a binary value to reterive later
                        FDloc = Q
                        FDloc[FDloc != 0] = 1
                        FD[i, j] = int(
                            ''.join(map(str, FDloc.astype(np.int))), 2)

                        delta_d = 1 / self.cell_length**2 * \
                            np.sum(-self.theta * Q)

            # adaptive time stepping => find the min required timestep
            # this section avoids over drying cells (negative d)
            d_new = self.d + delta_d * delta_t
            min_index = np.argmin(d_new)
            if d_new[min_index] < 0:
                delta_t = (self.min_WD -
                           self.d[min_index]) / delta_d[min_index]
                d_new = self.d + delta_d * delta_t

            # this loop calculate velocities in both directions
            for i in range(1, self.dem_shape[0] - 1):
                for j in range(1, self.dem_shape[1] - 1):
                    # check if the cell is dry or wet (d0>=delta)
                    if self.d[i, j] >= self.min_WD:
                        # For simplicity, the central cell is indexed as 0, and
                        # its neighbor cells at the east, north, west, and
                        # south sides are indexed as 1, 2, 3, and 4.
                        z = np.array([self.z[i, j], self.z[i+1, j],
                                      self.z[i, j+1], self.z[i-1, j],
                                      self.z[i, j-1]])
                        d = np.array([d_new[i, j], d_new[i+1, j],
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

                        FDloc = np.binary_repr(FD[i, j], 4)
                        vel = self.ComputeVelocity(
                            FDloc, n, d, z, u, v, H[i, j], self.d[i, j])
                        self.u[i, j] = Delta(-vel[0]) * \
                            vel[0] + Delta(vel[2]) * vel[2]
                        self.v[i, j] = Delta(-vel[1]) * \
                            vel[1] + Delta(vel[2]) * vel[3]

            self.d = d_new

            current_time += delta_t
