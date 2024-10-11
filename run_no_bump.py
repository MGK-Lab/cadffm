from core import CADFFM
import shutil
import os

if __name__ == "__main__":
    # Path to the input raster file
    tif_file = "./dam_break_no_bump/DamBreak_NB_0.1.tif"
    # Set the simulation parameters
    tot_time = 10                                        # Total simulation time
    time_step = 1e-6                                    # Time step
    n = 0.01                                            # Manning's n      
    CFL = 0.02                                          # CFL number
    depth = 0.3                                         # Depth of reservoir
    reserv_length = 3.3                                 # Length of reservoir
    min_WD = 1e-3                                       # Minimum water depth
    min_Head = 1e-5                                     # Minimum water head difference
    g = 9.81                                            # Acceleration due to gravity
    output_folder = './dam_break_no_bump/results_normal_mh_e4/'
    
    source_file = './core.py'
    plot_file_1 = './plot_bump_new_subplot.py'
    os.makedirs(output_folder, exist_ok=True)
    shutil.copy(plot_file_1, output_folder)
    shutil.copy(source_file, output_folder)
 
    # Model run
    model=CADFFM(tif_file, n, CFL, min_WD, min_Head, g)
    model.set_simulation_time(tot_time, time_step, 0)
    model.set_output_path(output_folder)
    model.set_reservoir(depth, reserv_length)
    model.run_simulation()
    
    