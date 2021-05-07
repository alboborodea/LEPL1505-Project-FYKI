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
S_0 = 1.202e-6   # [m*m]
l_0 = 0.025      # [m]
stop = 300       # Inner plot limit index

# Get data from files
data_x, data_y = get_data('Essai-001.csv')

# Convert x coordinates to meters
data_x = data_x / 1000

# Get length
n = len(data_x)

# Variables
epsilon_eng  = np.zeros(n)
sigma_eng    = np.zeros(n)
epsilon_true = np.zeros(n)
sigma_true   = np.zeros(n)

s_max   = 0
ind_max = 0

# Computation
for i in range(0, n):

    # Compute epsilon engineering and sigma engineering
    e_eng = data_x[i] / l_0   # e_eng = (l - l_0) / l_0
    s_eng = data_y[i] / S_0   # s_eng = F / S_0

    epsilon_eng[i] = e_eng
    sigma_eng[i] = s_eng

    # Compute epsilon true and sigma true
    e_true = np.log(1 + e_eng)
    s_true = s_eng * (1 + e_eng)

    epsilon_true[i] = e_eng
    sigma_true[i] = s_eng

# Find maximum sigma
for i in range(0, n):
    if (sigma_true[i] > s_max):
        s_max = sigma_true[i]
        ind_max = i

# Linear approximation
i = 10
j = 200
approx_parameters = np.polyfit(epsilon_true[i:j], sigma_true[i:j], 1)
x = np.linspace(0, 0.002, 100)
y = np.polyval(approx_parameters, x)

# Compute Young's modulus
young_modulus = y[-1] / x[-1]

# Compute the log values
epsilon_true_log = np.log(epsilon_true[20:])
sigma_true_log = np.log(sigma_true[20:])

# Compute linear approximation of the log curve in the plasticity domain
i = 200
j = 500
approx_parameters_log = np.polyfit(epsilon_true_log[i:j], sigma_true_log[i:j], 1)
x_log = np.linspace(-7, -2, 100)
y_log = np.polyval(approx_parameters_log, x_log)

# Print slope of linear approximation
print(y_log[-1])
print(y_log[0])
print(x_log[-1])
print(x_log[0])
slope = (y_log[-1] - y_log[0]) / (x_log[-1] - x_log[0])

# Print values
print("Elasticity limit = ", np.round(sigma_true[200]/1000000, 0), "MPa")
print("R max = ", np.round(s_max/1000000, 0), "MPa")
print("Young's modulus = ", np.round(young_modulus/1000000000, 0), "GPa")
print("Work hardening point = ", np.round(epsilon_true[ind_max], 2))
print("Slope = ", np.round(slope, 3))

# ---------------------------------------------------------------------------------------------------------------------------------------
# Set plot parameters
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (14, 7),
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'xx-large',
          'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Plot tensile curves

# Create main container
fig = plt.figure()
plt.grid(True)

# Outside plot
plt.title("Stress-strain curve for sample 001")
ax = plt.scatter(epsilon_true[0:ind_max], sigma_true[0:ind_max]/1000000, color="tab:blue", label="Test 001", s=0.3)
ax = plt.plot(x, y/1000000, color="tab:red", linestyle="dashed", label="Linear approximation")
plt.xlim(0, 0.25)
plt.ylim(0, 400)
plt.xlabel("True strain [-]")
plt.ylabel("True stress [MPa]")
plt.legend()

# Inside plot
ax_new = fig.add_axes(rect=[0.5, 0.2, 0.35, 0.55])
plt.grid(True)
plt.scatter(epsilon_true[0:stop], sigma_true[0:stop]/1000000, s=0.3, color="tab:blue")
plt.plot(x, y/1000000, color="tab:red", linestyle="dashed")
plt.show()

fig_2 = plt.figure()
plt.grid(True)

# Outside plot
plt.title("Log values for the tensile test of sample 001")
ax_2 = plt.scatter(epsilon_true_log[0:ind_max], sigma_true_log[0:ind_max], color="tab:blue", label="Test 001", s=0.3)
ax_2 = plt.plot(x_log, y_log, color="tab:red", linestyle="dashed", label="Linear approximation")
plt.xlabel("True strain log [-]")
plt.ylabel("True stress log [-]")
plt.legend()
plt.show()

#plt.xlim(0, 0.25)
#plt.ylim(0, 400)
#plt.scatter(epsilon_true[0:ind_max], sigma_true[0:ind_max]/1000000, color="tab:blue", label="Test 001", s=0.3)
#plt.xlabel("True strain [-]")
#plt.ylabel("True stress [MPa]")
#plt.legend()
#plt.show()

