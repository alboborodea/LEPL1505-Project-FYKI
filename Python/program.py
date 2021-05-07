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

# ---------------------------------------------------------------------------------------------------------------------------------------
# Compute integral
def compute_integral(x, y):
    # Check length
    assert (len(x) == len(y))
    # Get length
    N = len(x)
    # Compute numerical integral I
    I = 0
    for i in range(0, N-1):
        #I = I + ( x[i+1] - x[i] ) * ( y[i+1] + y[i] ) / 2
        I = I + ( x[i+1] - x[i] ) * y[i]
    # Return statement
    return I

# =======================================================================================================================================
# Main program
# ---------------------------------------------------------------------------------------------------------------------------------------
# Input test parameters
l_1 = 6 / 1000
l_2 = 8 / 1000
l_3 = 6 / 1000
a = 0.2 / 1000

# Get data from files
data_x_1, data_y_1 = get_data('Essai-001.csv')
data_x_2, data_y_2 = get_data('Essai-002.csv')
data_x_3, data_y_3 = get_data('Essai-003.csv')

# Convert x coordinates to meters
data_x_1 = data_x_1 / 1000
data_x_2 = data_x_2 / 1000
data_x_3 = data_x_3 / 1000

# Compute integrals
I_1 = compute_integral(data_x_1, data_y_1)
I_2 = compute_integral(data_x_2, data_y_2) 
I_3 = compute_integral(data_x_3, data_y_3) 

# Normalize
#v_1 = I_1 / l_1 / a
v_2 = I_2 / l_2 / a
v_3 = I_3 / l_3 / a

# Linear approximation
approx_parameters = np.polyfit([l_2, l_3], [v_2, v_3], 1)
x = np.linspace(0, 10/1000, 100)
y = np.polyval(approx_parameters, x)

# Compute W_e
W_e = y[0]

# Print results
print("I_1 = ", I_1, "[J]")
print("I_2 = ", I_2, "[J]")
print("I_3 = ", I_3, "[J]")
#print("v_1 = ", v_1, "[J/m²]")
print("v_2 = ", v_2, "[J/m²]")
print("v_3 = ", v_3, "[J/m²]")
print("W_e = ", W_e, "[J/m²]")

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
#plt.grid(True)
#plt.title("Tensile curves")
#plt.scatter(data_x_1, data_y_1, label="Test 001", s=0.1)
#plt.scatter(data_x_2, data_y_2, label="Test 002", s=0.1)
#plt.scatter(data_x_3, data_y_3, label="Test 003", s=0.1)
#plt.xlabel("Strain [m]")
#plt.ylabel("Force [N]")
#plt.legend()
#plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------
# Plot work and approximation
plt.grid(True)
plt.title("Essential work of fracture")
plt.scatter([l_2, l_3], [v_2/1000, v_3/1000], marker='o', color="tab:blue", label="Experimental")
plt.plot(x, y/1000, color="tab:red", linestyle="dashed", label="Linear approximation")
plt.xlabel('$l_0$ [m]')
plt.ylabel(r'$\dfrac{W_{tot}}{l_0 \times a}$ [kJ/m²]')
plt.legend()
plt.show()


