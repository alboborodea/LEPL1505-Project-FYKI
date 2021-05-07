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

# Print elasticty limit
print("Elasticity limit = ", np.round(data_y[i]/1000000, 0), "MPa")

# Compute maximal resistance
r_max = 0
for i in range(0, len(data_y)):
    if (data_y[i] > r_max): r_max = data_y[i]

# Print maximal resistance
print("R max = ", np.round(r_max/1000000, 0), "MPa")

# Compute the log values
data_x_log = np.log(data_x[20:])
data_y_log = np.log(data_y[20:])

# Compute linear approximation of the log curve in the plasticity domain
i = 200
j = 500
approx_parameters_log = np.polyfit(data_x_log[i:j], data_y_log[i:j], 1)
x_log = np.linspace(-7, -2, 100)
y_log = np.polyval(approx_parameters_log, x_log)

# Print slope of linear approximation
slope = (y_log[-1] - y_log[0]) / (x_log[-1] - x_log[0])
print("Slope = ", np.round(slope, 3))

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
plt.xlabel("Engineering Strain [-]")
plt.ylabel("Engineering Stress [Pa]")
#plt.plot(x, y, color="tab:red", linestyle="dashed", label="Linear approximation")
plt.legend()
plt.show()

plt.grid(True)
plt.title("Tensile curve log")
plt.scatter(data_x_log, data_y_log, color="tab:blue", label="Test 001", s=0.15)
plt.plot(x_log, y_log, color="tab:red", linestyle="dashed", label="Linear approximation")
plt.xlabel("Engineering Strain log [-]")
plt.ylabel("Engineering Stress log [Pa]")
plt.legend()
plt.show()