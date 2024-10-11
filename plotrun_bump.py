import plots
import sys

validation_path = './dam_break_bump/validation/'
results_path = './dam_break_bump/results_cfl_1e2_mh_1e4_vel/WL/'
graphspath = results_path + 'GraphsProfile'
path = './dam_break_bump/'

WL_measured = validation_path + 'WaterLevel9p7 Measured Data.csv'
WL_SWFCA = validation_path + 'WaterLevel9p7 SWFCA.csv'

G4_m = validation_path + 'G4 measured data.csv'
G10_m = validation_path + 'G10 measured data.csv'
G13_m = validation_path + 'G13 measured data.csv'
G20_m = validation_path + 'G20 measured data.csv'
G4_SWFCA = validation_path + 'G4 SWFCA.csv'
G10_SWFCA = validation_path + 'G10 SWFCA.csv'
G13_SWFCA = validation_path + 'G13 SWFCA.csv'
G20_SWFCA = validation_path + 'G20 SWFCA.csv'
tnd = './results/TND/TND.csv'
tnd_1e3 = './results_1e-3/TND/TND.csv'
elev = path +  'elevation.csv'
num = [9.7]

# for i in range(len(num)):
#     n = str(num[i])
#     C = './validation data/outpath/C_'+ n +'s.csv'
#     w = './results/WL/WL'+ n +'.csv'
#     w_1e3 = './results_1e-3/WL/WL'+ n +'.csv'
#     plots.plot_WL_compare(WL_measured, WL_SWFCA, w, w_1e3, elev, num[i])
    
w_2 = results_path + 'WL2.0.csv'
w_4 = results_path + 'WL4.0.csv'
w_6 = results_path + 'WL6.0.csv'
w_8 = results_path + 'WL8.0.csv'
w_9p7 = results_path + 'WL9.2.csv'

plots.plot_WL_compare_one(WL_measured, WL_SWFCA, w_2, w_4, w_6, w_8, w_9p7, elev)

# plots.plot_TND_bump(G4_m, G4_SWFCA, tnd, tnd_1e3, 4)
# plots.plot_TND_bump(G10_m, G10_SWFCA, tnd, tnd_1e3, 10)
# plots.plot_TND_bump(G13_m, G13_SWFCA, tnd, tnd_1e3, 13)
# plots.plot_TND_bump(G20_m, G20_SWFCA, tnd, tnd_1e3, 20)

