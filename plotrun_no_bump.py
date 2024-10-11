import plots
import os

validation_path = './dam_break_no_bump/validation/'
results_path = './dam_break_no_bump/results_normal_mh_e4/WL/'
graphspath = results_path + 'GraphsProfile'
path = './dam_break_no_bump/'

if not os.path.exists(graphspath):
            os.makedirs(graphspath, exist_ok = True)

C0p5 = validation_path + 'C_0.5s.csv'
C1p0 = validation_path + 'C_1.0s.csv'
C2p0 = validation_path + 'C_2.0s.csv'
C3p0 = validation_path + 'C_3.0s.csv'

WL0p5 = results_path + 'WL0.5.csv'
WL1p0 = results_path + 'WL1.0.csv'
WL2p0 = results_path + 'WL2.0.csv'
WL3p0 = results_path + 'WL3.0.csv'
 
elev = path + 'elevation.csv'

plots.plot_WL_compare(C0p5, WL0p5, elev, graphspath)
plots.plot_WL_compare(C1p0, WL1p0, elev, graphspath)
plots.plot_WL_compare(C2p0, WL2p0, elev, graphspath)
plots.plot_WL_compare(C3p0, WL3p0, elev, graphspath)

# plots.multiple_csv(WL_m,WL_n, elev, graphspath)