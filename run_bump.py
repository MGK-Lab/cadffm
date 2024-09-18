from core import CADFFM

if __name__ == "__main__":
    # Path to the input raster file
    tif_file = "./dam_break_bump/DamBreak_Bump_0.1.tif"      
    # Set the simulation parameters
    tot_time = 30                                       # Total simulation time
    time_step = 1e-5                                    # Time step
    n = 0.0125                                          # Manning's n      
    CFL = 0.005                                          # CFL number
    depth = 0.75                                        # Depth of reservoir
    reserv_length = 15.5                                # Length of reservoir
    min_WD = 1e-3                                       # Minimum water depth
    min_Head = 1e-4                                     # Minimum water head difference
    g = 9.81                                            # Acceleration due to gravity                                    # parameter for slope to assign mannings
    output_folder = './dam_break_bump/results_cfl_5e3_mh_1e4/'
    
    # Model run
    model=CADFFM(tif_file, n, CFL, min_WD, min_Head, g)
    model.set_simulation_time(tot_time, time_step, 0)
    model.set_output_path(output_folder)
    model.set_reservoir(depth, reserv_length)
    # model.delta_t_bias = 0.5
    model.run_simulation()
    
    