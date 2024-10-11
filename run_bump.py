from core import CADFFM
import shutil
import os

if __name__ == "__main__":
    # Path to the input raster file
    tif_file = "./dam_break_bump/DamBreak_Bump_0.1.tif"      
    # Set the simulation parameters
    tot_time = 40                                       # Total simulation time
    time_step = 1e-5                                    # Time step
    n = 0.0125                                          # Manning's n      
    CFL = 0.01                                         # CFL number
    depth = 0.75                                        # Depth of reservoir
    reserv_length = 15.5                                # Length of reservoir
    min_WD = 1e-3                                       # Minimum water depth
    min_Head = 5e-6                                     # Minimum water head difference
    g = 9.81                                            # Acceleration due to gravity                                    # parameter for slope to assign mannings
    output_folder = './dam_break_bump/results_cfl_' + "{:1.0e}".format(CFL) + '_mh_' + "{:1.0e}".format(min_Head) + '_back25/'
   
    source_file = './core.py'
    plot_file_1 = './plot_bump_new_subplot.py'
    plot_file_2 = './plot_time_bump.py'
    os.makedirs(output_folder, exist_ok=True)
    shutil.copy(plot_file_1, output_folder)
    shutil.copy(plot_file_2, output_folder)
    shutil.copy(source_file, output_folder)
    
    
    # Model run
    model=CADFFM(tif_file, n, CFL, min_WD, min_Head, g)
    model.set_simulation_time(tot_time, time_step, 0)
    model.set_output_path(output_folder)
    model.set_reservoir(depth, reserv_length)
    # model.delta_t_bias = 0.5
    model.run_simulation()
    
