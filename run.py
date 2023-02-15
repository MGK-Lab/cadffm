import core
import numpy as np
import time


# model = core.ShallowWaterFlowSolver('caffe_test.tif')

# d = np.random.rand(10000, 10000)

# start_time = time.time()
# print(model.CFL_deltat(d, d, d))
# end_time = time.time()
# duration = end_time - start_time
# print("Duration: {:.2f} seconds".format(duration))

d=np.array([1,2,3,4,5])
z_bar_i = np.maximum(d[0], d[1:5])
print(z_bar_i)
