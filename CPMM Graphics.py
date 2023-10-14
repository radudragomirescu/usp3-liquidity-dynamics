# Required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#plot style
sns.set_style("darkgrid")

# Creates a toy example displaying the price impact of a trade in a CPMM mechanism
def plot_CPMM_example():
    # Liquidity constant
    L = 10 

    # Define the first point (x, y) on the steeper portion of the curve
    x1 = 5
    y1 = L**2 / x1

    # Define the post-trade point (x', y') on the shallower portion of the curve
    x2 = 30
    y2 = L**2 / x2

    # Generate x values
    x_values = np.linspace(1, 100, 400)

    # Calculate corresponding y values based on the constant product rule xy = L^2
    y_values = L**2 / x_values

    # Calculate the tangent lines at (x, y) and (x', y')
    slope1 = -L**2 / (x1**2)
    intercept1 = y1 - slope1 * x1
    tangent1 = slope1 * x_values + intercept1

    slope2 = -L**2 / (x2**2)
    intercept2 = y2 - slope2 * x2
    tangent2 = slope2 * x_values + intercept2

    # Create the left plot
    plt.figure(figsize=(12, 5), dpi=300)

    # First graph with point (x, y)
    plt.subplot(1, 2, 1)
    plt.plot(x_values, y_values, label='xy = L^2', color='blue')
    plt.scatter(x1, y1, color='red', marker='o', label='Reserves (x, y)')
    plt.plot(x_values, tangent1, linestyle='--', color='green', label='(Negative) Price')
    plt.xlabel('Asset X', fontsize=16)
    plt.ylabel('Asset Y', fontsize=16)
    plt.title('CPMM Price Before Trade', fontsize=22, y =1.01)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.ylim(0, 50) 
    plt.xlim(0, 50)
    plt.tick_params(axis='both', which='both', labelsize=14) 

    # Create the right plot
    plt.subplot(1, 2, 2)

    # Second graph with point (x', y')
    plt.plot(x_values, y_values, label='xy = L^2', color='blue')
    plt.scatter(x2, y2, color='red', marker='o', label="Reserves (x', y')")
    plt.plot(x_values, tangent2, linestyle='--', color='green', label='(Negative) Price')
    plt.xlabel('Asset X', fontsize=16)
    plt.ylabel('Asset Y', fontsize=16)
    plt.title('CPMM Price After Trade for Asset Y', fontsize=22, y=1.01)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.ylim(0, 50)
    plt.xlim(0, 50)
    plt.tick_params(axis='both', which='both', labelsize=14) 

    plt.suptitle('Constant Product Market Maker (CPMM) Price Mechanism Toy Example', fontsize=26, y=1.03)

    plt.tight_layout()
    plt.show()

# Figure 2.1
plot_CPMM_example()




