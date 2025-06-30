import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set font to Times New Roman and enable matplotlib's built-in LaTeX rendering
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = False  # Keep this False to avoid LaTeX installation requirement
plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern for math symbols

# Set matplotlib to use non-interactive backend for remote environments
plt.switch_backend('Agg')

def read_data_file(file_path):
    """Read data file with header line"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Skip the header line and read data
    data_lines = lines[1:]
    data = []
    for line in data_lines:
        values = [float(v) for v in line.split()]
        data.append(values)
    
    return np.array(data)

# Read the data files
wave_data = read_data_file('./orgin_dataset/Wave_135_Tm_7.5_HS_3.5.out')
motion_data = read_data_file('./orgin_dataset/135_Tm_7.5_HS_3.5.out')

# Extract time and data columns
time_wave = wave_data[:, 0]
eta = wave_data[:, 1]  # Wave elevation

time_motion = motion_data[:, 0]
surge = motion_data[:, 1]  # ζ₁ [m]
sway = motion_data[:, 2]   # ζ₂ [m]
heave = motion_data[:, 3]  # ζ₃ [m]
roll = motion_data[:, 4]   # ζ₄ [degree]
pitch = motion_data[:, 5]  # ζ₅ [degree]
yaw = motion_data[:, 6]    # ζ₆ [degree]

# Create subplots
fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)

# Plot wave elevation with proper subscript
axes[0].plot(time_wave, eta, 'b-', linewidth=0.8)
axes[0].set_ylabel(r'$\eta$ [m]')  # Use LaTeX formatting for eta
axes[0].grid(True, alpha=0.3)

# Plot motions with proper LaTeX subscripts
motions = [surge, sway, heave, roll, pitch, yaw]
labels = [r'$\zeta_1$ [m]', r'$\zeta_2$ [m]', r'$\zeta_3$ [m]', 
          r'$\zeta_4$ [degree]', r'$\zeta_5$ [degree]', r'$\zeta_6$ [degree]']

for i, (motion, label) in enumerate(zip(motions, labels)):
    axes[i+1].plot(time_motion, motion, 'b-', linewidth=0.8)
    axes[i+1].set_ylabel(label)
    axes[i+1].grid(True, alpha=0.3)

# Set x-axis label only for the last subplot
axes[-1].set_xlabel('Time (sec)')

# Adjust layout
plt.tight_layout()

# Save the plot instead of showing it
plt.savefig('results/wave_motion_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'results/wave_motion_plot.png'")

# Close the figure to free memory
plt.close() 