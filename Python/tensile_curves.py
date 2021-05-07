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

# =======================================================================================================================================
# Main program
# ---------------------------------------------------------------------------------------------------------------------------------------
# Input test parameters
section = 1.202e-6       # [m²]
initial_length = 0.025   # [m]

# Get data from files
data_x, data_y = get_data('Essai-001.csv')

# Convert x coordinates to meters
data_x = data_x / 1000

# Change x data from elongation to deformation
data_x = data_x / initial_length

# Change y data from force to constraint
data_y = data_y / section

# Linear approximation
i = 200
approx_parameters = np.polyfit(data_x[0:i], data_y[0:i], 1)
x = np.linspace(0, 0.002, 100)
y = np.polyval(approx_parameters, x)

# Compute Young's modulus
young_modulus = y[-1] / x[-1]
print("Young's modulus = ", np.round(young_modulus/1000000000, 0), "GPa")

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
plt.title("Tensile curve")
#plt.xlim(-0.001, 0.02)
plt.scatter(data_x, data_y, color="tab:blue", label="Test 001", s=0.15)
plt.xlabel("Deformation [-]")
plt.ylabel("Constraint [N/m²]")
plt.plot(x, y, color="tab:red", linestyle="dashed", label="Linear approximation")
plt.legend()
plt.show()
