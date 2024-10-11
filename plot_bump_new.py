import pandas as pd
import matplotlib.pyplot as plt
import os


# Define the folder where your CSV files are located
folder_path = './dam_break_bump/results_cfl_5e3_mh_1e5_Qv_pure/WL'
# folder_path = './dam_break_no_bump/results/WL'
skip_no = 10 
low_limit = 2
up_limit = 10

# List all files in the directory
all_files = os.listdir(folder_path)
csv_files = [f for f in all_files if f.startswith('WL') and f.endswith('.csv')]
time = []

for i, file in enumerate(csv_files):
    file_label = os.path.splitext(file)[0]
    time.append(float(file_label[2:]))
    
time = [f"{num:05.2f}" for num in time]
csv_files = [time, csv_files]
csv_files = sorted(zip(csv_files[0], csv_files[1]), key=lambda x: float(x[0]))
time, csv_files = zip(*csv_files)
csv_files = csv_files[::skip_no]

# Loop through each file and plot the relevant columns
for file in csv_files:
    file_path = os.path.join(folder_path, file)  # Create full path to the file
    
    df = pd.read_csv(file_path, skiprows=[1])
    file_label = os.path.splitext(file)[0]
    print(float(file_label[2:]))
    if float(file_label[2:])>=low_limit and float(file_label[2:])<=up_limit:
        # Create primary axis for WL_new and H
        plt.figure(figsize=(10,6))
        ax1 = plt.gca()
        
        # Plot WL_new and H on the primary y-axis (left)
        ax1.plot(df['z'], linestyle=':', color='k')
        ax1.plot(df['wl'], label=f'WL ({file_label})', linestyle='-', color='b')
        ax1.plot(df['H'], label=f'H ({file_label})', linestyle='--', color='g')
        
        # Create secondary y-axis for v_new
        ax2 = ax1.twinx()
        ax2.plot(df['v'], label=f'v ({file_label})', linestyle='-.', color='r')
        
        # ax1.set_ylim(0, 0.8)
        # ax2.set_ylim(0, 4)
        # Add labels and title
        ax1.set_xlabel('Index')
        ax1.set_ylabel('WL / H', color='b')
        ax2.set_ylabel('v', color='r')
        
        # Add legends for both y-axes
        # ax1.legend(loc='upper left')
        # ax2.legend(loc='upper right')
        plt.legend()
        ax1.grid(True)
        ax1.grid(True)

        plt.show()

# plt.show()

# Show the plot

