import core
import numpy as np
import time


model = core.CADFFM('caffe_test.tif')

z = np.array([1, 1, 1, 1, 1])
d = np.array([2, 2, 3, 1, 1])
u = np.array([0.9, 1, 0, 0, 0])
v = np.zeros(5)
Hloc = model.ComputeBernoulliHead(z, d, u, v)
print('H = ', Hloc)
FDH = model.ComputeFlowDirectionH(Hloc)
print('FDH = ', FDH)
FDQ, Q_vel = model.ComputeFlowDirectionQ(d, u, v)
print('FDQ, Q_vel = ', FDQ, Q_vel)
Q = model.ComputeMassFlux(0.05, Hloc, d, z)
print('Q = ', Q)
NFD = FDQ * FDH
print('NFD = ', NFD)
Qn = NFD * Q
print('Qn = ', Qn)

SFD = d[1:]
SFD[SFD > 0] = 1
SFD *= (FDH ^ 1) * FDQ
print('SFD = ', SFD)
Qs = model.ComputeMassFluxSpecial(SFD, Hloc, d, Q_vel, 0.001)
print('Qs = ', Qs)
Q = Qs+Qn
print('Q = ', Q)
FDloc = Q
FDloc[FDloc != 0] = 1
FD = int(''.join(map(str, FDloc.astype(int))), 2)
print('FD = ', FD)
FDloc_n = np.binary_repr(FD, 4)
FDloc_n = [int(digit) for digit in FDloc_n]
print('FD = ', FDloc_n[3]*1.2)

# d = np.random.rand(10000, 10000)

# start_time = time.time()
# print(model.CFL_deltat(d, d, d))
# end_time = time.time()
# duration = end_time - start_time
# print("Duration: {:.2f} seconds".format(duration))
