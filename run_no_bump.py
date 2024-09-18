from core import CADFFM

if __name__ == "__main__":
    # Path to the input raster file
    tif_file = "./dam_break_no_bump/DamBreak_NB_0.1.tif"
    # Set the simulation parameters
    tot_time = 3                                        # Total simulation time
    time_step = 1e6                                    # Time step
    n = 0.01                                            # Manning's n      
    CFL = 0.01                                          # CFL number
    depth = 0.3                                         # Depth of reservoir
    reserv_length = 3.3                                 # Length of reservoir
    min_WD = 1e-3                                       # Minimum water depth
    min_Head = 1e-4                                     # Minimum water head difference
    g = 9.81                                            # Acceleration due to gravity
    output_folder = './dam_break_no_bump/results/'
     
    # Model run
    model=CADFFM(tif_file, n, CFL, min_WD, min_Head, g)
    model.set_simulation_time(tot_time, time_step, 0)
    model.set_output_path(output_folder)
    model.set_reservoir(depth, reserv_length)
    model.run_simulation()
    
    