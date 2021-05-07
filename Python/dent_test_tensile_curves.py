# AUTHOR : Boborodea Alexandru Tudor
# COURSE : LEPL1505
# =======================================================================================================================================
# Import commands
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

# =======================================================================================================================================
# Program functions
# ---------------------------------------------------------------------------------------------------------------------------------------
# Get x and y coordinates
def get_data(filename, skip_rows=[0, 1]):
    # Read file
    df = pd.read_csv(filename, skiprows=skip_rows)
    # Get x and y data
    data_x = df['mm'].to_list()
    data_y = df['N'].to_list()
    # Convert to numpy array
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    # Return statement
    return data_x, data_y

def change_values(data_x, data_y, initial_length, section):
    # Convert x coordinates to meters
    data_x = data_x / 1000
    # Change x data from elongation to deformation
    data_x = data_x / initial_length
    # Change y data from force to constraint
    data_y = data_y / section
    # Return statement
    return data_x, data_y

# =======================================================================================================================================
# Main program
# ---------------------------------------------------------------------------------------------------------------------------------------
# Input test parameters
section_1 = 1.202e-6       # [m²]
initial_length_1 = 0.025     # [m]
section_2_3 = 1.6e-6       # [m²]
initial_length_2_3 = 0.015     # [m]

# Get data from files
data_x_1, data_y_1 = get_data('Essai-001.csv')
data_x_1, data_y_1 = change_values(data_x_1, data_y_1, initial_length_1, section_1)

data_x_2, data_y_2 = get_data('Essai-002.csv')
data_x_2, data_y_2 = change_values(data_x_2, data_y_2, initial_length_2_3, section_2_3)

data_x_3, data_y_3 = get_data('Essai-003.csv')
data_x_3, data_y_3 = change_values(data_x_3, data_y_3, initial_length_2_3, section_2_3)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Set plot parameters
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (16, 8),
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'xx-large',
          'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Plot tensile curves
plt.grid(True)
plt.title("Tensile curves for dent test")
#plt.scatter(data_x_1, data_y_1, label="Sample 001", s=0.15)
plt.scatter(data_x_2, data_y_2, label="Sample 002", s=0.15)
plt.scatter(data_x_3, data_y_3, label="Sample 003", s=0.15)
plt.xlabel("Deformation [-]")
plt.ylabel("Constraint [N/m²]")
plt.legend()
plt.show()

